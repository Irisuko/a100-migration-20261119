"""CPU-only tests of literal adopted finalist bundles and fail-closed receipts."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest
import yaml


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
POLICY = REPO / "run/a100-gpu0123-20261114/proposals/development-execution-v2.json"


def test_renderer_entrypoint_exists():
    assert (HERE / "render_finalists.py").is_file(), "Standalone CPU renderer is missing"


@pytest.fixture
def renderer():
    path = HERE / "render_finalists.py"
    assert path.exists(), "Standalone finalist renderer has not been implemented"
    spec = importlib.util.spec_from_file_location("finalist_renderer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def templates():
    return json.loads(POLICY.read_text())["finalist_trials"]["templates"]


def seed(tmp_path, family="muxpp", model="8b", topology="pp4"):
    tp, pp, replicas = {"pp4": (1, 4, 1), "pp2tp2": (2, 2, 1),
                        "pp2dp2": (1, 2, 2), "tp4": (4, 1, 1),
                        "tp2dp2": (2, 1, 2), "2p2d": (1, 1, 4),
                        "1p1d_tp2": (2, 1, 2)}[topology]
    conditional = model == "70b" and tp * pp == 2
    candidate = dict(id=f"{model}-{family}-{topology}", model=model, family=family,
                     topology=topology, tp=tp, pp=pp, replicas=replicas,
                     prefill_replicas=1 if family == "pd" and tp == 2 else 2 if family == "pd" else None,
                     conditional_two_gpu=conditional)
    config = dict(sm_group_num=8, manual_divisions=[[100, 8, 1], [92, 16, 4]],
                  decode_bs_divisor=32, split_forward_token_budget=32768,
                  pp_async_decode=True, pp_prefetch=True, pp_prefetch_slots=2,
                  untouched_nested={"retain": [1, 2]})
    if family == "muxwise":
        config.update(sm_group_num=5, manual_divisions=[[76, 32, 1], [60, 48, 8], [52, 56, 16]],
                      decode_bs_divisor=24)
    config_path = tmp_path / "seed-spatial.yml"
    config_path.write_text(yaml.safe_dump(config))
    trace = tmp_path / "toolagent.jsonl"
    trace.write_text('{"idx":1,"arrival":1,"input_len":2,"output_len":2}\n')
    argv = ["--model", model, "--family", family, "--tp", str(tp), "--pp", str(pp),
            "--replicas", str(replicas), "--mem", ".93" if conditional else ".85",
            "--running", "128", "--graph", "16", "--chunk", "8192",
            "--inflight", "3", "--baseline-chunk", "2048", "--budget", "32768",
            "--overlap", "auto" if pp == 1 and family in ("chunked", "pd") else "off",
            "--suite", "trace", "--trace", str(trace), "--time-scale", "1.0",
            "--arrival-window-sec", "150.0", "--out", str(tmp_path / "old-seed")]
    if family == "pd":
        argv += ["--prefill-replicas", str(candidate["prefill_replicas"]), "--pd-backend", "mooncake"]
    argv += ["--extra", "--enable-p2p-check"]
    if family in ("muxwise", "muxpp"):
        argv += ["--pdmux-config-path", str(config_path), "--pdmux-prefill-mlp-chunk-tokens", "4096"]
    argv += ["--log-level", "info"]
    return dict(candidate=candidate, task=dict(id="architecture-seed", phase="development",
        selection_scope="architecture_seed", workload="toolagent", candidate_id=candidate["id"],
        argv=argv, timeout=7000, dependencies=[], trace_sha256="a" * 64),
        configuration=config, configuration_path=str(config_path),
        transport="nvlink" if family == "pd" else "default", source_sha256={"python/sglang/a.py": "b" * 64},
        evidence_sha256={str(config_path): "c" * 64},
        support_full_gate=conditional or family == "pd", conditional_capacity_verified=conditional,
        actual_sm_evidence={"note": "inherited proof; changed grid must be measured"})


def own(task, name):
    argv = task["argv"][:task["argv"].index("--extra")]
    return argv[argv.index(name) + 1]


def extra(task, name):
    argv = task["argv"][task["argv"].index("--extra") + 1:]
    return argv[argv.index(name) + 1]


@pytest.mark.parametrize("family,topology,expected", [
    ("chunked", "tp4", [(".85", "128", "16", "2048", "auto"),
                           ("0.9", "256", "32", "8192", "auto"),
                           ("0.85", "64", "16", "1024", "auto"),
                           (".85", "128", "16", "2048", "off")]),
    ("chunked", "pp4", [(".85", "128", "16", "2048", "off"),
                           ("0.9", "256", "32", "8192", "off"),
                           ("0.85", "64", "16", "1024", "off"),
                           (".85", "128", "16", "4096", "off")]),
])
def test_chunked_four_fresh_profiles_preserve_exact_load(renderer, templates, tmp_path, family, topology, expected):
    original = seed(tmp_path, family, topology=topology)
    before = copy.deepcopy(original)
    result = renderer.render_profiles(original, templates, tmp_path / "profiles", "v1")
    tasks = result["tasks"]
    assert original == before
    assert len(tasks) == len({t["id"] for t in tasks}) == 4
    assert all(t["id"] != "architecture-seed" for t in tasks)
    assert [(own(t, "--mem"), own(t, "--running"), own(t, "--graph"),
             own(t, "--baseline-chunk"), own(t, "--overlap")) for t in tasks] == expected
    assert {own(t, "--trace") for t in tasks} == {str(tmp_path / "toolagent.jsonl")}
    assert {own(t, "--time-scale") for t in tasks} == {"1.0"}
    assert {own(t, "--arrival-window-sec") for t in tasks} == {"150.0"}
    assert all(t["selection_scope"] == "finalist_parameter_trial" and t["timeout"] == 7000 for t in tasks)
    assert all(t["argv"][-2:] == ["--log-level", "info"] for t in tasks)


@pytest.mark.parametrize("topology,inflight,credit,refill", [("pp4", "6", "65536", "16384"),
    ("pp2tp2", "3", "0", "0"), ("pp2dp2", "3", "65536", "16384")])
def test_muxpp_historical_slot_and_new_spatial_yaml(renderer, templates, tmp_path, topology, inflight, credit, refill):
    original = seed(tmp_path, topology=topology)
    result = renderer.render_profiles(original, templates, tmp_path / "profiles", "v1")
    tasks = result["tasks"]
    assert [own(t, "--chunk") for t in tasks] == ["8192", "16384", "4096", "8192"]
    assert [extra(t, "--pdmux-prefill-mlp-chunk-tokens") for t in tasks] == ["4096", "8192", "2048", "4096"]
    assert own(tasks[3], "--inflight") == inflight
    assert extra(tasks[3], "--pdmux-pp-tpot-coeff") == "0.75"
    assert extra(tasks[3], "--pdmux-pp-max-inflight-tokens") == credit
    assert extra(tasks[3], "--pdmux-pp-token-refill-size") == refill
    assert {extra(t, "--pdmux-pp-max-budget") for t in tasks} == {"16384"}
    config = yaml.safe_load(result["configurations"][extra(tasks[2], "--pdmux-config-path")])
    assert config["manual_divisions"] == [[92, 16, 1], [84, 24, 4], [76, 32, 8], [68, 40, 12], [52, 56, 16], [44, 64, 24]]
    assert config["untouched_nested"] == {"retain": [1, 2]}
    assert config["split_forward_token_budget"] == 32768 and config["pp_prefetch"] is True
    assert all(profile["gpu_validation"] == "pending_exact_profile_prewarm_and_resource_gate" for profile in result["profiles"])


def test_muxwise_spatial_grid_is_profile_specific_without_seed_mutation(renderer, templates, tmp_path):
    original = seed(tmp_path, "muxwise", topology="tp4")
    result = renderer.render_profiles(original, templates, tmp_path / "profiles", "v1")
    grids = [yaml.safe_load(result["configurations"][extra(t, "--pdmux-config-path")])["manual_divisions"] for t in result["tasks"]]
    assert grids[0] == [[76, 32, 1], [60, 48, 8], [52, 56, 16]]
    assert grids[1] == grids[3] == [[84, 24, 1], [76, 32, 8], [60, 48, 16]]
    assert grids[2] == [[68, 40, 1], [52, 56, 8], [44, 64, 16]]
    assert original["configuration"]["manual_divisions"] == grids[0]


def test_conditional_70b_memory_remains_proven_value_in_all_slots(renderer, templates, tmp_path):
    original = seed(tmp_path, model="70b", topology="pp2dp2")
    result = renderer.render_profiles(original, templates, tmp_path / "profiles", "v1")
    assert {own(t, "--mem") for t in result["tasks"]} == {".93"}
    original["conditional_capacity_verified"] = False
    with pytest.raises(ValueError, match="conditional|capacity"):
        renderer.render_profiles(original, templates, tmp_path / "profiles", "v1")


def test_pd_inherits_transport_and_never_applies_global_chunk(renderer, templates, tmp_path):
    original = seed(tmp_path, "pd", topology="2p2d")
    result = renderer.render_profiles(original, templates, tmp_path / "profiles", "v1")
    assert {p["transport"] for p in result["profiles"]} == {"nvlink"}
    assert [own(t, "--overlap") for t in result["tasks"]] == ["auto", "auto", "auto", "off"]
    assert all("--chunked-prefill-size" not in t["argv"] for t in result["tasks"])
    original["task"]["argv"] += ["--chunked-prefill-size", "8192"]
    with pytest.raises(ValueError, match="PD|chunk"):
        renderer.render_profiles(original, templates, tmp_path / "profiles", "v1")


@pytest.mark.parametrize("change", ["fifth", "native", "seed_resources", "limit", "duplicate_slot"])
def test_invalid_profile_or_seed_contract_fails_closed(renderer, templates, tmp_path, change):
    original = seed(tmp_path)
    if change == "fifth":
        templates["muxpp"].append({"slot": 5})
    elif change == "native":
        templates["muxpp"][1]["overlap"] = "on"
    elif change == "seed_resources":
        original["task"]["argv"][original["task"]["argv"].index("--running") + 1] = "64"
    elif change == "limit":
        original["task"]["argv"][0:0] = ["--limit", "1"]
    else:
        templates["muxpp"][3]["slot"] = 3
    with pytest.raises(ValueError):
        renderer.render_profiles(original, templates, tmp_path / "profiles", "v1")


def test_source_pins_reject_changed_and_missing_serving_files(renderer, tmp_path):
    path = tmp_path / "runtime.py"
    path.write_text("old\n")
    pins = {str(path): renderer.sha256(path)}
    renderer.verify_pins(pins, tmp_path)
    path.write_text("new\n")
    with pytest.raises(ValueError, match="hash|changed"):
        renderer.verify_pins(pins, tmp_path)
    path.unlink()
    with pytest.raises(ValueError, match="missing|hash"):
        renderer.verify_pins(pins, tmp_path)


def test_slot_ledger_preserves_failed_slot_and_rejects_fifth(renderer, templates, tmp_path):
    result = renderer.render_profiles(seed(tmp_path), templates, tmp_path / "profiles", "v1")
    ledger = renderer.make_slot_ledger(result["tasks"])
    assert len(ledger["slots"]) == 4 and ledger["failed_slot_consumes_budget"]
    first = result["tasks"][0]["id"]
    run = tmp_path / "runs" / first
    run.mkdir(parents=True)
    (run / "outcome.json").write_text(json.dumps({"status": "failed", "returncode": 1}))
    observed = renderer.audit_slot_ledger(ledger, tmp_path)
    assert observed["slots"][0]["status"] == "failed"
    assert observed["consumed_slots"] == 1 and len(observed["slots"]) == 4
    with pytest.raises(ValueError):
        renderer.make_slot_ledger(result["tasks"] + [dict(result["tasks"][0], id="fifth", profile_slot=5)])


@pytest.fixture
def receipts(renderer, tmp_path, monkeypatch):
    """Real files/pins; reuse tested reader boundaries instead of emulating CUDA logs."""
    repo = tmp_path / "repo"
    root = repo / "run/campaign"
    root.mkdir(parents=True)
    for name in ("python/sglang/runtime.py", "scripts/a100_gpu0123/server_run.py",
                 "scripts/a100_gpu0123/bench/bench_client.py"):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# CPU fixture serving source\n")
    current = renderer.builder.current_sources(repo)
    value = seed(tmp_path)
    task = value["task"]
    support = root / "preflight/support"
    support.mkdir(parents=True)
    support_config = support / "spatial.yml"
    support_config.write_text(yaml.safe_dump(value["configuration"]))
    config_paths = {"spatial": support_config, "worker0.pdmux": support_config}
    support_args = dict(model="8b", family="muxpp", tp=1, pp=4, replicas=1,
        mem=.85, running=128, graph=16, chunk=8192, baseline_chunk=2048,
        inflight=3, budget=32768, suite="correctness", stress200=True, overlap="off",
        extra=["--enable-p2p-check", "--pdmux-prefill-mlp-chunk-tokens", "4096"])
    passed = dict(metadata=dict(args=support_args, source_sha256=current),
                  config_paths=config_paths, overlap_requested="off")
    out = root / "development/architecture-seed"
    out.mkdir(parents=True)
    task["argv"] = renderer.builder.development_argv(value["candidate"], passed, out,
        {"path": str(tmp_path / "toolagent.jsonl")}, 1.0, 150.0)
    task["support_label"] = "support"
    command = root / "runs/support/command.json"
    command.parent.mkdir(parents=True)
    command.write_text(json.dumps({"argv": ["taskset", "-c", "0-31,64-95", "python", "server_run.py",
        "--out", str(support), "--suite", "correctness", "--stress200"]}))
    audit_path = support / "resource-audit.json"
    audit_path.write_text('{"status":"passed"}')
    task["eligibility_evidence"] = {str(command): renderer.sha256(command), str(audit_path): renderer.sha256(audit_path)}
    audit = dict(runtime_supported=True, exclusion_reasons=[], evidence_sha256=task["eligibility_evidence"],
        fresh_resource_audit=dict(workers=[dict(index=0, ranks=[dict(pp=i, tp=0, capacity=40000) for i in range(4)])]))
    metadata = dict(args=support_args, source_sha256=current,
        configuration_paths={"worker0.pdmux": str(support_config)},
        configuration_sha256={"spatial": renderer.sha256(support_config), "worker0.pdmux": renderer.sha256(support_config)})
    (out / "spatial.yml").write_bytes(support_config.read_bytes())
    (out / "run.json").write_text(json.dumps(metadata))
    for name in ("trace_summary.json", "trace_requests.jsonl", "trace_planned.jsonl"):
        (out / name).write_text("{}\n")
    run = root / "runs/architecture-seed"
    run.mkdir(parents=True)
    for name in ("outcome.json", "queue-task.json", "command.json"):
        (run / name).write_text("{}\n")
    (run / "queue-task.json").write_text(json.dumps({"task": task}))
    measurement = dict(selection_eligible=True, argv=copy.deepcopy(task["argv"]), label=task["id"],
        source_sha256=current.copy(), trace_sha256=task["trace_sha256"], workload="toolagent",
        trace_audit=dict(purpose="dev_tuning", base_arrival_window_sec=150,
                         source_start_sec=1200, source_end_exclusive_sec=1260),
        platform_provenance=dict(status="verified", configuration={"mooncake_transport": "default"}),
        configuration_fingerprint="config-id", source_fingerprint="source-id")
    environment = root / "environment.json"
    environment.write_text("{}\n")
    platform_manifest = root / "platform-launch.json"
    platform_manifest.write_text(json.dumps({"platform": {"environment_path": str(environment)}}))
    measurement["platform_provenance"].update(manifest_path=str(platform_manifest),
                                              manifest_sha256=renderer.sha256(platform_manifest))
    # Existing reader and support-audit suites test their internals. These doubles
    # provide verified CPU receipts; all renderer source/config/argv checks run.
    monkeypatch.setattr(renderer.selector, "read_measurement", lambda *a: measurement)
    monkeypatch.setattr(renderer.builder, "eligibility", lambda *a: (audit, passed))
    policy = root / "adopted.json"
    policy.write_bytes(POLICY.read_bytes())
    (root / "campaign.json").write_text(json.dumps(dict(candidates=[value["candidate"]],
        parameter_selection_policy=dict(execution_proposal=str(policy), execution_proposal_sha256=renderer.sha256(policy)))))
    plan = dict(source_sha256=current.copy(), tasks=[task])
    plan_path = root / "seeds.json"
    plan_path.write_text(json.dumps(plan))
    return dict(repo=repo, root=root, plan=plan, path=plan_path, measurement=measurement,
                audit=audit, passed=passed, out=root / "proposals/finalists", policy=policy)


def build_receipts(renderer, receipts, **kwargs):
    return renderer.build(receipts["path"], ["architecture-seed"], root=receipts["root"],
        repo=receipts["repo"], out_dir=receipts["out"], tag="v1", **kwargs)


def test_build_binds_receipts_writes_immutable_pinned_queue(renderer, receipts):
    result = build_receipts(renderer, receipts)
    assert len(result["queue"]["tasks"]) == 4
    assert not receipts["out"].exists()
    renderer.write_artifacts(receipts["out"], result)
    renderer.verify_pins(result["queue"]["source_sha256"], receipts["repo"])
    assert {p.name for p in receipts["out"].iterdir()} == {"configs", "tasks.json", "profiles.json", "slot-ledger.json", "launch.json"}
    command = json.loads((receipts["out"] / "launch.json").read_text())["argv"]
    assert command[command.index("--mooncake-transport") + 1] == "default"
    before = (receipts["out"] / "tasks.json").read_bytes()
    with pytest.raises(FileExistsError):
        renderer.write_artifacts(receipts["out"], result)
    assert (receipts["out"] / "tasks.json").read_bytes() == before


@pytest.mark.parametrize("change", ["source", "measured_source", "measurement_argv", "ineligible", "wrong_window", "support", "config"])
def test_build_rejects_unproven_or_mismatched_seed_before_writing(renderer, receipts, change):
    if change == "source":
        (receipts["repo"] / "python/sglang/runtime.py").write_text("# changed\n")
    elif change == "measured_source":
        receipts["measurement"]["source_sha256"] = {"old": "a" * 64}
    elif change == "measurement_argv":
        receipts["measurement"]["argv"] = ["--out", "other"]
    elif change == "ineligible":
        receipts["measurement"]["selection_eligible"] = False
    elif change == "wrong_window":
        receipts["measurement"]["trace_audit"]["base_arrival_window_sec"] = 600
    elif change == "support":
        receipts["audit"]["runtime_supported"] = False
        receipts["audit"]["exclusion_reasons"] = ["failed support"]
    else:
        (receipts["root"] / "preflight/support/spatial.yml").write_text("changed: true\n")
    with pytest.raises(ValueError):
        build_receipts(renderer, receipts)
    assert not receipts["out"].exists()


def test_optional_selector_must_bind_exact_explicit_selected_seed(renderer, receipts):
    selected = receipts["root"] / "selector.json"
    value = dict(status="draft_for_review", freeze_created=False, draft_selections=[
        dict(labels=["architecture-seed"], configuration_fingerprint="wrong", source_fingerprint="source-id")])
    selected.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="selector"):
        build_receipts(renderer, receipts, selector_path=selected)
    value["draft_selections"][0]["configuration_fingerprint"] = "config-id"
    selected.write_text(json.dumps(value))
    result = build_receipts(renderer, receipts, selector_path=selected)
    assert result["queue"]["source_sha256"][str(selected)] == renderer.sha256(selected)


def test_shortened_pd_chunk_override_is_not_silently_inherited(renderer, templates, tmp_path):
    value = seed(tmp_path, "pd", topology="2p2d")
    value["task"]["argv"] += ["--chunked-prefill-s=8192"]
    with pytest.raises(ValueError, match="PD|chunk"):
        renderer.render_profiles(value, templates, tmp_path / "profiles", "v1")


def test_queue_rejects_mixed_transport_before_artifact_creation(renderer, receipts, tmp_path, monkeypatch):
    tasks = receipts["plan"]["tasks"]
    tasks.append(dict(copy.deepcopy(tasks[0]), id="second-seed", workload="conversation"))
    receipts["path"].write_text(json.dumps(receipts["plan"]))
    def proven(task, *args):
        row = seed(tmp_path)
        row["task"] = task
        row["transport"] = "nvlink" if task["id"] == "second-seed" else "default"
        row["measurement"] = receipts["measurement"]
        row["platform_environment"] = str(receipts["root"] / "environment.json")
        return row
    monkeypatch.setattr(renderer, "load_seed", proven)
    with pytest.raises(ValueError, match="transport"):
        renderer.build(receipts["path"], ["architecture-seed", "second-seed"], root=receipts["root"],
                       repo=receipts["repo"], out_dir=receipts["out"], tag="v1")
    assert not receipts["out"].exists()


def test_seed_timeout_cannot_change_from_executed_queue_receipt(renderer, receipts):
    receipts["plan"]["tasks"][0]["timeout"] = 9000
    receipts["path"].write_text(json.dumps(receipts["plan"]))
    with pytest.raises(ValueError, match="receipt|queue"):
        build_receipts(renderer, receipts)


def test_chunked_abbreviation_cannot_override_new_bundle_chunk(renderer, templates, tmp_path):
    value = seed(tmp_path, "chunked", topology="pp4")
    value["task"]["argv"] += ["--chunked-prefill-s=2048"]
    with pytest.raises(ValueError, match="abbreviat|controlled"):
        renderer.render_profiles(value, templates, tmp_path / "profiles", "v1")


def test_a_second_declaration_cannot_add_four_more_slots(renderer, receipts):
    result = build_receipts(renderer, receipts)
    renderer.write_artifacts(receipts["out"], result)
    with pytest.raises(ValueError, match="declared|budget"):
        renderer.build(receipts["path"], ["architecture-seed"], root=receipts["root"],
                       repo=receipts["repo"], out_dir=receipts["root"] / "proposals/another", tag="v2")


@pytest.fixture
def control_receipts(renderer, receipts, tmp_path, templates, monkeypatch):
    declarations = {}
    for label, family, topology in [("muxpp-main", "muxpp", "pp4"),
                                     ("chunked-main", "chunked", "tp4"),
                                     ("chunked-control", "chunked", "pp4")]:
        value = seed(tmp_path, family, topology=topology)
        value["task"]["id"] = label
        value["task"]["argv"][value["task"]["argv"].index("--out") + 1] = str(receipts["root"] / "development" / label)
        value["measurement"] = receipts["measurement"]
        value["platform_environment"] = str(receipts["root"] / "environment.json")
        value["source_sha256"] = receipts["measurement"]["source_sha256"]
        value["evidence_sha256"] = {}
        declarations[label] = value
    receipts["plan"]["tasks"] = [value["task"] for value in declarations.values()]
    receipts["path"].write_text(json.dumps(receipts["plan"]))
    monkeypatch.setattr(renderer, "load_seed", lambda task, *a: declarations[task["id"]])
    for label in ("muxpp-main", "chunked-main"):
        directory = receipts["root"] / "proposals" / label
        directory.mkdir(parents=True)
        tasks = renderer.render_profiles(declarations[label], templates, directory, label)["tasks"]
        (directory / "slot-ledger.json").write_text(json.dumps(renderer.make_slot_ledger(tasks)))
    (receipts["root"] / "development-budget.json").write_bytes((POLICY.parent.parent / "development-budget.json").read_bytes())
    return receipts, declarations


def build_control(renderer, fixture):
    receipts, _ = fixture
    return renderer.build(receipts["path"], ["chunked-control"], root=receipts["root"],
        repo=receipts["repo"], out_dir=receipts["out"], tag="control-v1",
        role="matched_topology_control", matched_muxpp_seeds={"chunked-control": "muxpp-main"})


def test_matched_control_gets_separate_four_slots_bound_to_selected_muxpp(renderer, control_receipts):
    result = build_control(renderer, control_receipts)
    assert len(result["queue"]["tasks"]) == 4
    assert result["ledger"]["role"] == "matched_topology_control"
    assert all(p["matched_muxpp_seed_id"] == "muxpp-main" and p["family"] == "chunked"
               and p["topology"] == "pp4" for p in result["profiles"]["profiles"])
    assert all("finalist-control-" in t["id"] for t in result["queue"]["tasks"])


@pytest.mark.parametrize("change", ["same_as_main", "wrong_anchor", "not_chunked", "missing_main", "unselected_anchor", "different_load"])
def test_matched_role_cannot_expand_main_search_or_use_wrong_topology(renderer, control_receipts, change):
    receipts, declarations = control_receipts
    if change == "same_as_main":
        declarations["chunked-control"]["candidate"] = declarations["chunked-main"]["candidate"]
    elif change == "wrong_anchor":
        declarations["muxpp-main"]["candidate"]["pp"] = 2
    elif change == "not_chunked":
        declarations["chunked-control"]["candidate"]["family"] = "muxpp"
    elif change == "missing_main":
        (receipts["root"] / "proposals/chunked-main/slot-ledger.json").unlink()
    elif change == "different_load":
        argv = declarations["muxpp-main"]["task"]["argv"]
        argv[argv.index("--time-scale") + 1] = "2.0"
    else:
        (receipts["root"] / "proposals/muxpp-main/slot-ledger.json").unlink()
    with pytest.raises(ValueError):
        build_control(renderer, control_receipts)


def test_matched_control_budget_cannot_be_redeclared(renderer, control_receipts):
    receipts, _ = control_receipts
    result = build_control(renderer, control_receipts)
    renderer.write_artifacts(receipts["out"], result)
    receipts["out"] = receipts["root"] / "proposals/another-control"
    with pytest.raises(ValueError, match="declared|budget"):
        build_control(renderer, control_receipts)


def test_matched_role_has_four_group_cap_separate_from_main_budget(renderer, templates, tmp_path):
    value = seed(tmp_path, "chunked", topology="pp4")
    tasks = renderer.render_profiles(value, templates, tmp_path / "profiles", "v1", "matched_topology_control")["tasks"]
    five_groups = [dict(task, id=f"{task['id']}-{group}", workload=f"workload{group}")
                   for group in range(5) for task in tasks]
    with pytest.raises(ValueError, match="at most4"):
        renderer.make_slot_ledger(five_groups, "matched_topology_control")
