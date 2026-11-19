#!/usr/bin/env python3
"""Render four fresh, CPU-only finalist trials from explicitly selected seeds.

This does not select architectures, claim GPU feasibility, or launch processes.
Inputs must be completed, eligible architecture seeds from build_development.
The adopted four-bundle policy, current serving source, support receipts and
exact trace/load are checked before any output directory is created.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROOT = REPO / "run/a100-gpu0123-20261114"
sys.path.insert(0, str(HERE))
try:
    import build_development as builder
    import select_development as selector
finally:
    sys.path.pop(0)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def resolve(path, repo=REPO):
    path = Path(path)
    return (path if path.is_absolute() else repo / path).resolve()


def verify_pins(pins, repo=REPO):
    require(isinstance(pins, dict) and pins, "Nonempty source/evidence pins required")
    for filename, expected in pins.items():
        path = resolve(filename, repo)
        require(path.is_file(), f"Pinned file missing: {path}")
        require(isinstance(expected, str) and re.fullmatch("[0-9a-f]{64}", expected)
                and sha256(path) == expected, f"Pinned hash changed: {path}")


def split_argv(argv):
    own, extra, index = {}, [], 0
    while index < len(argv):
        token = argv[index]
        require(isinstance(token, str) and token.startswith("--"), "Invalid seed argv")
        if token == "--extra":
            extra = list(argv[index + 1:])
            break
        key, equals, value = token.partition("=")
        require(key not in own, f"Duplicate seed option: {key}")
        if not equals:
            value = True
            if index + 1 < len(argv) and not argv[index + 1].startswith("--"):
                index += 1
                value = argv[index]
        own[key] = value
        index += 1
    return own, extra


def replace_extra(extra, values):
    """Replace only declared single-value flags; preserve every other token."""
    result, index = [], 0
    while index < len(extra):
        token = extra[index]
        key, equals, _ = token.partition("=")
        require(not key.startswith("--") or key in values or not any(name.startswith(key) for name in values),
                f"Abbreviated controlled extra option is unsupported: {key}")
        if key in values:
            if not equals:
                require(index + 1 < len(extra) and not extra[index + 1].startswith("--"),
                        f"Missing extra option value: {key}")
                index += 1
        else:
            result.append(token)
        index += 1
    for key, value in values.items():
        if value is not None:
            result += [key, str(value)]
    return result


def render_profiles(seed, templates, out_dir, tag, role="main"):
    candidate, original = seed["candidate"], seed["task"]
    family, topology = candidate["family"], candidate["topology"]
    require(role in {"main", "matched_topology_control"}, "Unknown finalist role")
    require(role == "main" or family == "chunked", "Matched topology controls must be Chunked")
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,29}", tag), "Invalid fresh tag")
    require(family in {"chunked", "muxwise", "muxpp", "pd"}, "Unsupported finalist family")
    own, inherited_extra = split_argv(original["argv"])
    for key in ("model", "family", "tp", "pp", "replicas"):
        require(str(candidate[key]) == own.get("--" + key), f"Seed topology mismatch: {key}")
    require(original.get("phase") == "development" and original.get("selection_scope") == "architecture_seed",
            "Only fresh architecture seed measurements can seed finalists")
    require("--limit" not in own and own.get("--suite") == "trace", "Full seed trace required; no limit")
    require(own.get("--running") == "128" and own.get("--graph") == "16",
            "Seed resources differ from adopted128/16; a new exact policy is required")
    require(0 < float(own["--mem"]) < 1, "Invalid seed memory fraction")
    require(float(own["--budget"]) == 32768, "Preserve adopted spatial token budget32768")
    pp = candidate["pp"]
    native = pp == 1 and family in {"chunked", "pd"}
    require(own.get("--overlap") in ({"auto", "on"} if native else {"off"}),
            "Seed native overlap does not match its supported family/PP mode")
    flags = {token.split("=", 1)[0] for token in inherited_extra if token.startswith("--")}
    forbidden = {"--disable-overlap-schedule", "--mem-fraction-static", "--max-running-requests",
                 "--cuda-graph-max-bs", "--cuda-graph-bs", "--max-total-tokens",
                 "--tp-size", "--pp-size", "--dtype", "--context-length"}
    require(not any(canonical.startswith(flag) for flag in flags for canonical in forbidden),
            "Seed extras override controlled resources/topology/overlap")
    require(not (family == "pd" and any("--chunked-prefill-size".startswith(flag) for flag in flags)),
            "PD role-specific chunk sizes cannot use a global chunk override")
    require(not (family == "muxpp" and "--no-cpp" in own), "MuxPP bundles require CPP enabled")
    conditional = candidate["model"] == "70b" and candidate["tp"] * pp == 2
    if conditional:
        require(seed.get("support_full_gate") and seed.get("conditional_capacity_verified"),
                "Conditional70B requires full proven memory and per-instance capacity>=32768")
    require(seed["transport"] in {"default", "nvlink"}, "Unverified transport")
    if family == "pd":
        require(pp == 1 and seed.get("support_full_gate"), "PD requires fully gated PP1 transport")
    key = "chunked_pp1" if family == "chunked" and pp == 1 else "chunked_pp_gt1" if family == "chunked" else family
    bundles = templates[key]
    require(len(bundles) == 4 and [row.get("slot") for row in bundles] == [1, 2, 3, 4],
            "Exactly four ordered distinct profile slots are required; no fifth slot")
    tasks, profiles, configurations = [], [], {}
    for bundle in bundles:
        slot = bundle["slot"]
        require(bundle.get("overlap") == ("off" if not native or slot == 4 else "auto"),
                "Unsupported native overlap in adopted profile")
        prefix = "finalist" if role == "main" else "finalist-control"
        label = f"{prefix}-{candidate['id']}-{original['workload']}-s{slot}-{tag}"
        require(len(label) <= 96 and label != original["id"], "Invalid or reused finalist label")
        args = dict(own)
        args["--out"] = str(Path(own["--out"]).parent / label)
        for field in ("running", "graph", "baseline_chunk", "chunk", "inflight", "overlap"):
            if field in bundle:
                args["--" + field.replace("_", "-")] = str(bundle[field])
        if "baseline_chunk_by_model" in bundle:
            require(family == "chunked" and slot == 3,
                    "Model chunk mapping is limited to Chunked latency slot3")
            require("baseline_chunk" not in bundle,
                    "Ambiguous scalar and model-specific baseline chunk")
            by_model = bundle["baseline_chunk_by_model"]
            require(isinstance(by_model, dict) and set(by_model) == {"8b", "70b"}
                    and all(type(value) is int and value > 0 for value in by_model.values()),
                    "Invalid or incomplete model baseline chunk map")
            require(candidate["model"] in by_model, "Unknown model for baseline chunk map")
            args["--baseline-chunk"] = str(by_model[candidate["model"]])
        if "mem" in bundle and not conditional:
            args["--mem"] = str(bundle["mem"]["normal_8b_or_70b_four_gpu_instance"])
        if "inflight_by_topology" in bundle:
            require(topology in bundle["inflight_by_topology"], "Unsupported MuxPP historical topology")
            args["--inflight"] = str(bundle["inflight_by_topology"][topology])
        config = copy.deepcopy(seed["configuration"])
        for field in ("manual_divisions", "sm_group_num", "decode_bs_divisor"):
            if field in bundle:
                config[field] = copy.deepcopy(bundle[field])
        config_path = str(Path(out_dir).resolve() / "configs" / f"{label}.yml")
        text = yaml.safe_dump(config, sort_keys=True)
        configurations[config_path] = text
        config_hash = hashlib.sha256(text.encode()).hexdigest()
        changes = {}
        if family == "chunked":
            changes["--chunked-prefill-size"] = None  # Runner owns the per-profile value.
        if family in {"muxwise", "muxpp"}:
            changes.update({"--pdmux-config-path": config_path, "--sm-group-num": config["sm_group_num"]})
        for field, value in bundle.items():
            if field.startswith("pdmux_"):
                if field.endswith("_by_topology"):
                    require(topology in value, f"Missing topology policy: {field}")
                    field, value = field.removesuffix("_by_topology"), value[topology]
                changes["--" + field.replace("_", "-")] = value
        changes["--log-level"] = "info"
        extra = replace_extra(inherited_extra, changes)
        argv = [value for name, value in args.items() for value in ([name] if value is True else [name, str(value)])]
        argv += ["--extra", *extra]
        task = dict(id=label, phase="development", selection_scope="finalist_parameter_trial",
                    finalist_role=role,
                    seed_id=original["id"], candidate_id=candidate["id"], workload=original["workload"],
                    profile_slot=slot, profile_name=bundle["name"], argv=argv,
                    timeout=original["timeout"], dependencies=[], trace_sha256=original["trace_sha256"],
                    mooncake_transport=seed["transport"])
        tasks.append(task)
        profiles.append(dict(task_id=label, seed_id=original["id"], candidate_id=candidate["id"],
            finalist_role=role,
            model=candidate["model"], family=family, topology=topology, workload=original["workload"],
            profile_slot=slot, profile_name=bundle["name"], argv=argv,
            spatial_path=config_path, spatial_sha256=config_hash,
            spatial_consumed_by_backend=family in {"muxwise", "muxpp"},
            requested_manual_divisions=config.get("manual_divisions"), actual_sm_map=None,
            inherited_actual_sm_evidence=seed.get("actual_sm_evidence"),
            transport=seed["transport"], source_sha256=seed["source_sha256"],
            evidence_sha256=seed["evidence_sha256"],
            architecture_measurement_source=seed.get("measurement_source_compatibility"),
            memory_feasibility_fixed=conditional, seed_memory_fraction=float(own["--mem"]),
            gpu_validation="pending_exact_profile_prewarm_and_resource_gate",
            required_gates=["All workers/PD pairs: exact-profile long31663-token prewarm, graph/green-context smoke, cache/concurrent and resource checks before replay",
                            "Selected trial: full correctness/cancellation/recovery/stress200 on all replicas, resource and communication audit; numerical verdict remains separate"],
            fresh_trial=True, prior_architecture_measurement_reused=False))
    return dict(tasks=tasks, profiles=profiles, configurations=configurations)


def make_slot_ledger(tasks, role="main"):
    require(role in {"main", "matched_topology_control"}, "Unknown slot-ledger role")
    groups = {}
    labels = set()
    for task in tasks:
        require(task.get("finalist_role", "main") == role, "Mixed slot-ledger roles")
        require(role == "main" or task["candidate_id"].split("-", 2)[1] == "chunked",
                "Matched topology controls must be Chunked")
        require(task["id"] not in labels, "Duplicate trial label")
        labels.add(task["id"])
        group = (task["candidate_id"], task["workload"])
        groups.setdefault(group, []).append(task)
    limit = 16 if role == "main" else 4
    require(0 < len(groups) <= limit, f"Finalist role budget is at most{limit} groups")
    for entries in groups.values():
        require(len(entries) == 4 and sorted(t["profile_slot"] for t in entries) == [1, 2, 3, 4]
                and len({t["seed_id"] for t in entries}) == 1, "Exactly four fresh slots per selected seed; no fifth slot")
    return dict(schema_version=1, status="declared_not_executed", role=role, failed_slot_consumes_budget=True,
        retry_policy="No replacement or fifth profile. Any identical-config pure harness retry must retain the original attempt and be reviewed separately.",
        slots=[dict(task_id=t["id"], seed_id=t["seed_id"], candidate_id=t["candidate_id"],
                    workload=t["workload"], profile_slot=t["profile_slot"], status="declared_not_executed") for t in tasks])


def audit_slot_ledger(ledger, root):
    result = copy.deepcopy(ledger)
    consumed = 0
    for slot in result["slots"]:
        run = Path(root) / "runs" / slot["task_id"]
        outcome = run / "outcome.json"
        if outcome.is_file():
            slot.update(status=read(outcome)["status"], outcome_path=str(outcome), outcome_sha256=sha256(outcome))
            consumed += 1
        elif run.exists():
            slot["status"] = "started_without_terminal_outcome"
            consumed += 1
    result.update(consumed_slots=consumed, remaining_unstarted_slots=len(result["slots"]) - consumed)
    return result


def load_seed(task, plan, root, repo, current, postflight_compatibility=None):
    """Reuse established CPU evidence checks; require fresh measured seed source."""
    candidates = [c for c in read(root / "campaign.json")["candidates"] if c["id"] == task.get("candidate_id")]
    require(len(candidates) == 1, "Seed candidate missing or ambiguous")
    candidate = candidates[0]
    own, _ = split_argv(task["argv"])
    require(read(root / "runs" / task["id"] / "queue-task.json").get("task") == task,
            "Seed task differs from its executed queue receipt")
    measurement = selector.read_measurement(root / "runs" / task["id"], {})
    require(measurement.get("selection_eligible") is True, "Selected seed is not a complete eligible measured trial")
    require(measurement["argv"] == task["argv"] and measurement["label"] == task["id"], "Selected seed queue/measurement mismatch")
    measurement_source = None
    if postflight_compatibility:
        measurement_source = builder.serving_compatibility.postflight_measurement_identity(
            measurement, current, postflight_compatibility)
        measured_resource, measured_resource_pins = measured_resource_evidence(
            resolve(own["--out"], repo), candidate)
        measurement_source["resource_audit"] = measured_resource
    else:
        require(measurement["source_sha256"] == current, "Selected seed serving source is old or incomplete")
    trace = measurement["trace_audit"]
    require(trace["purpose"] == "dev_tuning" and trace["base_arrival_window_sec"] == 150
            and trace["source_start_sec"] == 1200 and trace["source_end_exclusive_sec"] == 1260,
            "Finalist trials require the exact adopted150s development trace")
    require(task["trace_sha256"] == measurement["trace_sha256"] and task["workload"] == measurement["workload"],
            "Selected seed trace/workload mismatch")
    platform = measurement["platform_provenance"]
    require(platform.get("status") == "verified", "Selected seed transport/platform provenance is not verified")
    transport = platform["configuration"]["mooncake_transport"]
    evidence = dict(task.get("eligibility_evidence", {}))
    if postflight_compatibility:
        evidence = builder.serving_compatibility.resolve_postflight_pins(evidence, repo, postflight_compatibility)
        evidence.update(measured_resource_pins)
    else:
        verify_pins(evidence, repo)
    platform_manifest = resolve(platform["manifest_path"], repo)
    verify_pins({str(platform_manifest): platform["manifest_sha256"]}, repo)
    environment = resolve(read(platform_manifest)["platform"]["environment_path"], repo)
    evidence.update({str(platform_manifest): sha256(platform_manifest), str(environment): sha256(environment)})
    label = task.get("support_label")
    require(isinstance(label, str) and label, "Seed lacks support label")
    command_path = root / "runs" / label / "command.json"
    command = read(command_path)["argv"]
    support_out = resolve(builder.option(command[5:], "--out"), repo)
    compatibility = postflight_compatibility or plan.get("serving_compatibility")
    if compatibility and not postflight_compatibility:
        compatibility = builder.serving_compatibility.load(compatibility["path"], current, repo)
    audit, passed = builder.eligibility(candidate, {candidate["id"]: (label, support_out)}, root, repo, current, compatibility)
    require(audit["runtime_supported"] and passed, "Seed support gate failed: " + "; ".join(audit["exclusion_reasons"]))
    support_args = passed["metadata"]["args"]
    support_own, _ = split_argv(command[5:])
    support_full = (support_args.get("suite") == "correctness" and support_args.get("stress200") is True
                    and support_own.get("--suite") == "correctness" and support_own.get("--stress200") is True)
    require(candidate["family"] != "pd" or support_full,
            "Selected PD architecture needs full paired correctness/cancellation/recovery/stress200 and resource gate before four-profile rendering; architecture smoke alone is insufficient")
    expected = builder.development_argv(candidate, passed, resolve(own["--out"], repo),
        {"path": own["--trace"]}, float(own["--time-scale"]), float(own["--arrival-window-sec"]))
    require(expected == task["argv"], "Seed argv differs from its proven support configuration")
    metadata_path = resolve(own["--out"], repo) / "run.json"
    metadata = read(metadata_path)
    config_paths = {"spatial": metadata_path.parent / "spatial.yml",
                    **{k: resolve(v, repo) for k, v in metadata.get("configuration_paths", {}).items()}}
    require(set(config_paths) == set(metadata.get("configuration_sha256", {})), "Seed config receipt incomplete")
    verify_pins({str(p): metadata["configuration_sha256"][key] for key, p in config_paths.items()}, repo)
    effective = [p for key, p in config_paths.items() if key.endswith(".pdmux")]
    require(not effective or len({sha256(p) for p in effective}) == 1, "Nonuniform seed spatial configs")
    config_path = effective[0] if effective else config_paths["spatial"]
    evidence.update(audit["evidence_sha256"])
    for directory, names in [(metadata_path.parent, ["run.json", "trace_summary.json", "trace_planned.jsonl", "trace_requests.jsonl"]),
                             (root / "runs" / task["id"], ["outcome.json", "queue-task.json", "command.json"])]:
        for name in names:
            path = directory / name
            evidence[str(path)] = sha256(path)
    evidence.update({str(path): sha256(path) for path in config_paths.values()})
    conditional = candidate["model"] == "70b" and candidate["tp"] * candidate["pp"] == 2
    workers = audit.get("fresh_resource_audit", {}).get("workers", [])
    capacity = bool(workers) and all(worker.get("ranks") and all(
        type(rank.get("capacity")) is int and rank["capacity"] >= 32768 for rank in worker["ranks"]) for worker in workers)
    if conditional:
        require(support_full and capacity, "Conditional70B per-instance capacity/full gate missing")
    return dict(candidate=candidate, task=task, measurement=measurement, measurement_source_compatibility=measurement_source,
                configuration=yaml.safe_load(config_path.read_text()), configuration_path=str(config_path),
                transport=transport, source_sha256=current, evidence_sha256=evidence,
                platform_environment=str(environment),
                support_full_gate=support_full, conditional_capacity_verified=conditional and capacity,
                actual_sm_evidence={"support_run": str(support_out), "seed_run": str(metadata_path.parent),
                                    "configuration_sha256": metadata["configuration_sha256"]})


def measured_resource_evidence(run, candidate):
    audit = builder.resource_audit.audit_run(run)
    expected = {(pp, tp) for pp in range(candidate["pp"]) for tp in range(candidate["tp"])}
    workers = audit.get("workers", [])
    require(audit.get("status") == "passed" and not audit.get("errors") and not audit.get("missing_evidence")
            and len(workers) == candidate["replicas"]
            and {w.get("index") for w in workers} == set(range(candidate["replicas"]))
            and all(w.get("status") == "passed" and len(w.get("ranks", [])) == len(expected)
                    and {(r.get("pp"), r.get("tp")) for r in w.get("ranks", [])} == expected for w in workers),
            "Measured architecture requires a fresh successful all-worker/all-rank resource audit")
    paths = {run / "run.json", *run.glob("server-*.log"), *run.glob("worker*/resource-phase-markers.json"),
             *run.glob("pair*/pair.json"), *run.glob("pair*/resource-phase-markers.json"), *run.rglob("*server-*.json")}
    if (run / "gpu-monitor.csv").is_file(): paths.add(run / "gpu-monitor.csv")
    return dict(status="passed", audit_sha256=builder.serving_compatibility.digest(audit),
                workers=len(workers), ranks_per_worker=len(expected)), {str(path): sha256(path) for path in paths}


def comparison_source(proven):
    identity = proven.get("measurement_source_compatibility")
    return identity["comparison_fingerprint"] if identity else proven["measurement"]["source_fingerprint"]


def build(seed_queue, seed_ids, *, root=ROOT, out_dir, tag, selector_path=None, repo=REPO,
          role="main", matched_muxpp_seeds=None, postflight_compatibility_path=None):
    root, repo, out_dir = resolve(root, repo), Path(repo).resolve(), resolve(out_dir, repo)
    require(root in out_dir.parents, "Artifacts must be under campaign root")
    require(not out_dir.exists(), "Artifact directory already exists")
    require(role in {"main", "matched_topology_control"}, "Unknown finalist role")
    matched_muxpp_seeds = dict(matched_muxpp_seeds or {})
    require(not matched_muxpp_seeds if role == "main" else set(matched_muxpp_seeds) == set(seed_ids),
            "Matched role needs one explicit selected MuxPP seed binding per control; main role accepts none")
    seed_queue = resolve(seed_queue, repo)
    plan = read(seed_queue)
    current = builder.current_sources(repo)
    compatibility = builder.serving_compatibility.load(postflight_compatibility_path, current, repo) if postflight_compatibility_path else None
    if compatibility:
        require(compatibility.get("schema_version") == 5
                and compatibility["archive"]["scripts/a100_gpu0123/render_finalists.py"]["candidate_sha256"] == sha256(__file__),
                "Exact reviewed postflight renderer required")
        execution_pins = builder.serving_compatibility.resolve_postflight_pins(plan.get("source_sha256"), repo, compatibility)
    else:
        verify_pins(plan.get("source_sha256"), repo)
        execution_pins = dict(plan["source_sha256"])
    normalized = {str(resolve(k, repo)): v for k, v in plan["source_sha256"].items()}
    require(all(normalized.get(str(resolve(k, repo))) == v or
                (compatibility and k == builder.serving_compatibility.TARGET
                 and normalized.get(str(resolve(k, repo))) == builder.serving_compatibility.CANCELLATION_CANDIDATE)
                for k, v in current.items()), "Seed queue serving source is old or incomplete")
    campaign = read(root / "campaign.json")
    adoption = campaign["parameter_selection_policy"]
    policy_path = resolve(adoption["execution_proposal"], repo)
    require(sha256(policy_path) == adoption["execution_proposal_sha256"], "Adopted profile policy hash changed")
    policy = read(policy_path)
    require(policy["finalist_trials"]["per_system_model_workload"] == 4, "Exactly four adopted profiles required")
    require(seed_ids and len(seed_ids) == len(set(seed_ids)), "Explicit unique selected seed IDs required")
    by_id = {task["id"]: task for task in plan["tasks"]}
    require(len(by_id) == len(plan["tasks"]) and all(label in by_id for label in seed_ids), "Unknown or duplicate seed task")
    selected = read(resolve(selector_path, repo)) if selector_path else None
    if selected:
        require(selected.get("status") == "draft_for_review" and selected.get("freeze_created") is False,
                "Expected an existing selector review draft")
    pins = execution_pins
    for path in (seed_queue, policy_path, Path(__file__).resolve()):
        pins[str(path)] = sha256(path)
    if role == "matched_topology_control":
        budget_path = root / "development-budget.json"
        require(read(budget_path).get("selected_best_and_controls", {}).get("additional_control_budget"),
                "Adopted additional matched-control budget is missing")
        pins[str(budget_path)] = sha256(budget_path)
    if selector_path:
        path = resolve(selector_path, repo)
        pins[str(path)] = sha256(path)
    all_tasks, all_profiles, configs = [], [], {}
    groups, transports, environments, source_cohorts = set(), set(), set(), set()
    declared_groups, control_groups, main_seeds = set(), set(), {}
    for ledger_path in root.rglob("slot-ledger.json"):
        existing = read(ledger_path)
        old_role = existing.get("role", "main")
        for slot in existing.get("slots", []):
            identity = slot["candidate_id"].split("-", 2)
            key = (identity[0], identity[1], slot["workload"])
            declared_groups.add((old_role, *key))
            if old_role == "matched_topology_control":
                control_groups.add((identity[0], slot["workload"]))
            elif old_role == "main":
                main_seeds.setdefault(key, set()).add(slot["seed_id"])
        if role == "matched_topology_control" and old_role == "main":
            pins[str(ledger_path)] = sha256(ledger_path)
    for label in seed_ids:
        task = by_id[label]
        group = (task["candidate_id"].split("-", 2)[0], task["candidate_id"].split("-", 2)[1], task["workload"])
        require(group not in groups, "More than one selected topology for a system/model/workload")
        require((role, *group) not in declared_groups,
                "Four-profile budget already declared for this system/model/workload; preserve the original ledger")
        groups.add(group)
        proven = load_seed(task, plan, root, repo, current, compatibility)
        selected_proof = proven
        anchor = None
        if role == "matched_topology_control":
            candidate = proven["candidate"]
            require(candidate["family"] == "chunked", "Matched role is only for Chunked")
            main_key = (candidate["model"], "chunked", task["workload"])
            mux_key = (candidate["model"], "muxpp", task["workload"])
            require(len(main_seeds.get(main_key, set())) == 1 and len(main_seeds.get(mux_key, set())) == 1,
                    "Matched control requires already-declared main Chunked and selected MuxPP finalists")
            anchor_id = matched_muxpp_seeds[label]
            require(anchor_id in main_seeds[mux_key] and anchor_id in by_id,
                    "Matched MuxPP anchor is not the explicitly selected main seed")
            main_id = next(iter(main_seeds[main_key]))
            require(main_id in by_id, "Main Chunked seed receipt is missing from seed queue")
            main_seed = load_seed(by_id[main_id], plan, root, repo, current, compatibility)
            anchor = load_seed(by_id[anchor_id], plan, root, repo, current, compatibility)
            require(anchor["candidate"]["family"] == "muxpp" and anchor["task"]["workload"] == task["workload"]
                    and all(candidate[key] == anchor["candidate"][key] for key in ("model", "topology", "tp", "pp", "replicas")),
                    "Matched control does not match selected MuxPP model/workload/topology")
            require(candidate["topology"] != main_seed["candidate"]["topology"],
                    "Matched topology is already the main Chunked finalist; no additional search budget")
            require(all(comparison_source(row) == comparison_source(proven)
                        for row in (main_seed, anchor)), "Matched control source differs from selected main finalists")
            control_args, _ = split_argv(task["argv"])
            for row in (main_seed, anchor):
                paired_args, _ = split_argv(row["task"]["argv"])
                require(row["task"]["trace_sha256"] == task["trace_sha256"]
                        and all(float(paired_args[key]) == float(control_args[key])
                                for key in ("--time-scale", "--arrival-window-sec")),
                        "Matched control trace/rate differs from selected main finalists")
            control_groups.add((candidate["model"], task["workload"]))
            require(len(control_groups) <= 4, "Matched topology control budget is at most4 model/workload groups")
            for row in (main_seed, anchor):
                pins.update(row["evidence_sha256"])
            selected_proof = anchor
        transports.add(proven["transport"])
        environments.add(proven["platform_environment"])
        source_cohorts.add(comparison_source(proven))
        require(len(transports) == 1 and len(environments) == 1,
                "A run_campaign queue has one transport/environment; render separate queues for each verified transport")
        require(len(source_cohorts) == 1, "Selected seeds have different runtime/platform software identities")
        if selected:
            choices = [row for row in selected["draft_selections"] if selected_proof["task"]["id"] in row["labels"]]
            require(len(choices) == 1 and choices[0]["configuration_fingerprint"] == selected_proof["measurement"]["configuration_fingerprint"]
                    and choices[0]["source_fingerprint"] == selected_proof["measurement"]["source_fingerprint"],
                    "Seed does not match the explicit selector choice")
        rendered = render_profiles(proven, policy["finalist_trials"]["templates"], out_dir, tag, role)
        if anchor:
            for profile in rendered["profiles"]:
                profile.update(matched_muxpp_seed_id=anchor["task"]["id"],
                               matched_muxpp_candidate_id=anchor["candidate"]["id"],
                               matched_muxpp_evidence_sha256=anchor["evidence_sha256"])
        for new_task in rendered["tasks"]:
            require(not (root / "runs" / new_task["id"]).exists()
                    and not resolve(builder.option(new_task["argv"], "--out"), repo).exists(), "Finalist label/output already used")
        all_tasks.extend(rendered["tasks"])
        all_profiles.extend(rendered["profiles"])
        configs.update(rendered["configurations"])
        pins.update(proven["evidence_sha256"])
    ledger = make_slot_ledger(all_tasks, role)
    pins.update({path: hashlib.sha256(text.encode()).hexdigest() for path, text in configs.items()})
    transport, environment = next(iter(transports)), next(iter(environments))
    launch = dict(status="declaration_only_not_launched", required_mooncake_transport=transport,
        argv=[str(repo / ".venv/bin/python"), str(repo / "scripts/a100_gpu0123/run_campaign.py"),
              "--root", str(root), "--tasks", str(out_dir / "tasks.json"),
              "--queue-id", f"finalists-{'control-' if role != 'main' else ''}{tag}-{transport}", "--environment", environment,
              "--mooncake-transport", transport, "--continue-independent"],
        validation="Append --dry-run for CPU validation before use. Existing run_campaign chooses transport from its CLI; use this exact declared transport flag.")
    return dict(queue=dict(schema_version=1, status="draft_not_launched", source_sha256=pins, tasks=all_tasks,
                          required_mooncake_transport=transport, postflight_compatibility=compatibility),
        profiles=dict(schema_version=1, status="four_profiles_declared_gpu_gates_pending", policy_path=str(policy_path),
                      role=role, postflight_compatibility=compatibility,
                      policy_sha256=sha256(policy_path), seed_queue=str(seed_queue), seed_queue_sha256=sha256(seed_queue),
                      selection_mode="explicit_selector_choices" if selector_path else "explicit_seed_ids_no_automatic_selection",
                      profiles=all_profiles),
        ledger=ledger, configurations=configs, launch=launch)


def write_artifacts(directory, result):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    for path, text in result["configurations"].items():
        path = Path(path)
        require(path.parent == directory / "configs", "Generated config path escaped artifact directory")
        path.parent.mkdir(exist_ok=True)
        with path.open("x") as stream:
            stream.write(text)
    for name, value in [("tasks.json", result["queue"]), ("profiles.json", result["profiles"]),
                        ("slot-ledger.json", result["ledger"]), ("launch.json", result["launch"])]:
        with (directory / name).open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-queue", type=Path, required=True)
    parser.add_argument("--seed-ids", nargs="+", required=True)
    parser.add_argument("--selector", type=Path)
    parser.add_argument("--postflight-compatibility", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--role", choices=("main", "matched_topology_control"), default="main")
    parser.add_argument("--matched-muxpp-seeds", nargs="*", default=[], metavar="CONTROL_SEED=MUXPP_SEED")
    args = parser.parse_args(argv)
    pairs = [value.split("=", 1) for value in args.matched_muxpp_seeds]
    require(all(len(pair) == 2 and all(pair) for pair in pairs)
            and len({pair[0] for pair in pairs}) == len(pairs), "Invalid or repeated matched MuxPP seed binding")
    result = build(args.seed_queue, args.seed_ids, root=args.root, out_dir=args.out_dir,
                   tag=args.tag, selector_path=args.selector, role=args.role, matched_muxpp_seeds=dict(pairs),
                   postflight_compatibility_path=args.postflight_compatibility)
    write_artifacts(args.out_dir, result)
    print(json.dumps(dict(status=result["queue"]["status"], trials=len(result["queue"]["tasks"]),
                          out_dir=str(args.out_dir), gpu_processes_started=0)))


if __name__ == "__main__":
    main()
