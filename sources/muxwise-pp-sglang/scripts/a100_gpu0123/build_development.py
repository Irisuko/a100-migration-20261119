#!/usr/bin/env python3
"""Build an immutable development draft from completed runtime support evidence.

CPU only: this file neither imports Torch nor starts a controller or server.
Runtime support is not a numerical correctness or formal readiness claim.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import serving_compatibility

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_ROOT = REPO / "run/a100-gpu0123-20261114"
MODELS = {"8b": "Llama-3.1-8B-Instruct", "70b": "Llama-3.3-70B-Instruct"}
_spec = importlib.util.spec_from_file_location("_development_resource_audit", HERE / "resource_audit.py")
resource_audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(resource_audit)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def resolve(path, repo):
    value = Path(path)
    return (value if value.is_absolute() else repo / value).resolve()


def read(path):
    return json.loads(Path(path).read_text())


def option(argv, name):
    found = []
    for index, value in enumerate(argv):
        if value == "--extra":
            break
        if value == name:
            require(index + 1 < len(argv), f"Missing value for {name}")
            found.append(argv[index + 1])
        elif value.startswith(name + "="):
            found.append(value.split("=", 1)[1])
    require(len(found) <= 1, f"Duplicate option {name}")
    return found[0] if found else None


def current_sources(repo):
    """Same serving subset as server_run.source_provenance, excluding audit tools."""
    repo = Path(repo)
    tool_root = repo / "scripts/a100_gpu0123"
    files = [*sorted((repo / "python/sglang").rglob("*.py")), tool_root / "server_run.py",
             *[path for path in sorted((tool_root / "bench").glob("*.py")) if not path.name.startswith("test_")]]
    return {str(path.relative_to(repo)): sha(path) for path in files}


def trace_records(manifest_path, parent_path, repo):
    manifest, parent = read(manifest_path), read(parent_path)
    require(manifest.get("status") == "PASS" and parent.get("status") == "PASS", "Trace manifests require PASS audits")
    result = {}
    for row in manifest["records"]:
        if row.get("purpose") not in {"dev", "dev_tuning"}:
            continue
        key = row["model"], row["workload"]
        require(key not in result, f"Ambiguous development trace for {key}")
        allowed = [p for p in parent["records"] if p.get("purpose") == "dev"
                   and (p["model"], p["workload"]) == key]
        require(any(p["source_start_sec"] <= row["source_start_sec"]
                    < row["source_end_exclusive_sec"] <= p["source_end_exclusive_sec"] for p in allowed),
                "Selected development window is outside the audited parent development window")
        require(row.get("all_rows_and_hash_blocks_validated") is True
                and row.get("length_clipping") is False and row.get("padding_tokens") == 0
                and re.fullmatch(r"[0-9a-f]{64}", row.get("source_indices_sha256", "")),
                "Development trace lacks complete raw-index/hash-block audit")
        trace, stats_path = resolve(row["path"], repo), resolve(row["stats_path"], repo)
        require(sha(trace) == row["sha256"], f"Development trace hash mismatch: {trace}")
        require(sha(stats_path) == row["stats_sha256"], f"Development stats hash mismatch: {stats_path}")
        stats = read(stats_path)
        duration = stats["duration_sec"]
        require(type(row["requests"]) is int and row["requests"] > 0
                and stats["requests"] == row["requests"] and math.isfinite(duration) and duration > 0
                and duration == row["base_arrival_window_sec"]
                and stats["start_sec"] == row["source_start_sec"]
                and stats["start_sec"] + stats["source_duration_sec"] == row["source_end_exclusive_sec"]
                and math.isclose(stats["source_duration_sec"] * stats["stretch"], duration),
                "Development trace count/window/stretch provenance differs")
        result[key] = {**row, "path": str(trace), "stats_path": str(stats_path)}
    require(result, "No audited development traces supplied")
    return result


def support_catalog(plan, repo):
    result = {}
    for task in plan.get("tasks", []):
        if task.get("candidate_id"):
            result[task["candidate_id"]] = (task["id"], resolve(option(task["argv"], "--out"), repo))
    for row in plan.get("excluded_equivalent_candidates", []):
        require(row["candidate_id"] not in result, "Candidate has multiple support sources")
        result[row["candidate_id"]] = (row["covered_by_label"], resolve(row["measurement_directory"], repo))
    return result


def custom_markers(run, args):
    if args["tp"] == 1:
        return []
    errors = []
    if not args.get("custom_allreduce"):
        errors.append("TP_custom_allreduce_not_requested_in_passed_support")
    for worker in range(args["replicas"]):
        log = run / f"server-{worker}.log"
        text = log.read_text(errors="replace") if log.exists() else ""
        if "Setup Custom allreduce failed" in text or "Custom allreduce is disabled" in text:
            errors.append(f"worker{worker}_custom_allreduce_fallback")
        ranks = set()
        for line in text.splitlines():
            match = re.search(r"Custom allreduce enabled: rank=(\d+) world_size=(\d+) physical_device_ids=(\[[\d, ]+\])", line)
            if not match or int(match[2]) != args["tp"]:
                continue
            prefix = line.split("]", 1)[0]
            pp_match, tp_match = re.search(r"\bPP(\d+)\b", prefix), re.search(r"\bTP(\d+)\b", prefix)
            pp_rank, tp_rank = int(pp_match[1]) if pp_match else 0, int(tp_match[1]) if tp_match else 0
            offset = worker * args["tp"] * args["pp"] + pp_rank * args["tp"]
            if int(match[1]) == tp_rank and json.loads(match[3]) == list(range(offset, offset + args["tp"])):
                ranks.add((pp_rank, tp_rank))
        expected = {(pp, tp) for pp in range(args["pp"]) for tp in range(args["tp"])}
        if ranks != expected:
            errors.append(f"worker{worker}_missing_custom_allreduce_rank_markers")
    return errors


def eligibility(candidate, catalog, root, repo, source, compatibility=None):
    result = dict(candidate_id=candidate["id"], runtime_supported=False, exclusion_reasons=[])
    reasons = result["exclusion_reasons"]
    if candidate["id"] not in catalog:
        reasons.append("missing_support_catalog_entry")
        return result, None
    label, run = catalog[candidate["id"]]
    result.update(support_label=label, support_run=str(run))
    paths = {"run": run / "run.json", "outcome": root / "runs" / label / "outcome.json",
             "command": root / "runs" / label / "command.json"}
    for name, path in paths.items():
        if not path.is_file():
            reasons.append(f"missing_{name}")
    if reasons:
        return result, None
    metadata, outcome = read(paths["run"]), read(paths["outcome"])
    args = metadata.get("args", {})
    result["evidence_sha256"] = {str(path): sha(path) for path in paths.values()}
    command = read(paths["command"]).get("argv", [])
    command_ok = (len(command) >= 5 and Path(command[0]).name == "taskset"
                  and command[1:3] == ["-c", "0-31,64-95"]
                  and resolve(command[3], repo) == (repo / ".venv/bin/python").resolve()
                  and resolve(command[4], repo) == (repo / "scripts/a100_gpu0123/server_run.py").resolve())
    if command_ok:
        try:
            recorded = command[5:]
            command_ok = resolve(option(recorded, "--out"), repo) == run
            for key, default in [("model", None), ("family", None), ("tp", 1), ("pp", 4), ("replicas", 1)]:
                value = option(recorded, "--" + key)
                value = default if value is None else int(value) if type(default) is int else value
                command_ok = command_ok and value == args.get(key)
        except (TypeError, ValueError):
            command_ok = False
    if not command_ok:
        reasons.append("controller_command_binding_or_support_identity_mismatch")
    conditional_70b = candidate["model"] == "70b" and candidate["tp"] * candidate["pp"] == 2
    paired_pd = candidate["family"] == "pd"
    if paired_pd:
        try:
            recorded_prefill = int(option(command[5:], "--prefill-replicas") or 0)
        except (TypeError, ValueError):
            recorded_prefill = None
        expected_prefill = candidate.get("prefill_replicas")
        if (type(expected_prefill) is not int or not 0 < expected_prefill < candidate["replicas"]
                or args.get("prefill_replicas") != expected_prefill or recorded_prefill != expected_prefill):
            reasons.append("support_pd_role_layout_mismatch")
    if conditional_70b or paired_pd:
        own_args = command[5:]
        if "--extra" in own_args:
            own_args = own_args[:own_args.index("--extra")]
        if not (args.get("suite") == "correctness" and args.get("stress200") is True
                and option(own_args, "--suite") == "correctness" and "--stress200" in own_args):
            if conditional_70b:
                reasons.append("conditional_70b_requires_full_correctness_and_stress200")
            if paired_pd:
                reasons.append("pd_requires_full_paired_correctness_and_stress200")
    if metadata.get("status") != "execution_complete" or outcome.get("status") != "completed" or outcome.get("returncode") != 0:
        reasons.append("support_execution_not_completed")
    if any(args.get(key) != candidate[key] for key in ["model", "tp", "pp", "replicas"]):
        reasons.append("support_candidate_topology_mismatch")
    if args.get("family") != candidate["family"]:
        equivalent = (candidate["family"] == "chunked" and args.get("family") == "plain" and args.get("pp", 0) > 1
                      and option(args.get("extra", []), "--chunked-prefill-size") == str(args.get("baseline_chunk", 2048)))
        if not equivalent:
            reasons.append("support_candidate_family_mismatch")
        else:
            result["family_equivalence"] = "Ordinary PP with explicit Chunked size has the same serving command as Chunked PP."
    requested_overlap=serving_compatibility.seed_request(candidate["family"]) if compatibility else args.get("overlap")
    if compatibility:
        try:
            result['support_source_compatibility']=serving_compatibility.check_support(metadata,source,compatibility,requested_overlap,run=run)
            result['evidence_sha256'].update(result['support_source_compatibility'].get('cancellation_evidence_sha256',{}))
            result['evidence_sha256'].update(compatibility['artifacts_sha256'])
            result['evidence_sha256'].update(serving_compatibility.check_actual_overlap(
                run,metadata,result['support_source_compatibility']['effective_overlap']))
            if result['support_source_compatibility'].get('scheduler_transition'):
                result['evidence_sha256'].update(serving_compatibility.check_actual_scheduler_scope(run,metadata))
            if result['support_source_compatibility'].get('pd_transport_transition'):
                result['evidence_sha256'].update(serving_compatibility.check_actual_non_pd_scope(run,metadata))
        except ValueError as error:
            reasons.append('support_overlap_or_source_compatibility_failed: '+str(error))
    elif metadata.get("source_sha256") != source:
        reasons.append("support_serving_source_differs_from_current")
    model = repo.parent / "models" / MODELS[candidate["model"]]
    model_hashes = {path.name: sha(path) for path in model.glob("*.json")}
    if not metadata.get("model_revision") or not model_hashes or metadata.get("model_provenance") != model_hashes:
        reasons.append("model_provenance_missing_or_changed")
    config_paths = {"spatial": run / "spatial.yml", **{key: resolve(path, repo) for key, path in metadata.get("configuration_paths", {}).items()}}
    config_hashes = metadata.get("configuration_sha256", {})
    if not config_hashes or set(config_paths) != set(config_hashes) or any(not path.is_file() or sha(path) != config_hashes.get(key) for key, path in config_paths.items()):
        reasons.append("support_configuration_missing_or_changed")
    effective = [value for key, value in config_hashes.items() if key.endswith(".pdmux")]
    if candidate["family"] in {"muxwise", "muxpp"} and (not effective or len(set(effective)) != 1):
        reasons.append("missing_or_nonuniform_effective_pdmux_configuration")
    audits = sorted(run.glob("resource-audit*.json"))
    if not audits:
        reasons.append("missing_stored_resource_audit")
    for path in audits:
        audit = read(path)
        result["evidence_sha256"][str(path)] = sha(path)
        if (audit.get("schema_version", 0) < 2 or audit.get("status") != "passed" or audit.get("errors")
                or audit.get("missing_evidence") or resolve(audit.get("run_dir", "."), repo) != run):
            reasons.append("stored_resource_audit_not_passed")
    if not reasons:
        fresh = resource_audit.audit_run(run)
        result["fresh_resource_audit"] = fresh
        if fresh.get("status") != "passed" or fresh.get("errors") or fresh.get("missing_evidence"):
            reasons.append("fresh_resource_audit_not_passed")
        if conditional_70b or paired_pd:
            workers = fresh.get("workers", [])
            expected_ranks = {(pp, tp) for pp in range(candidate["pp"]) for tp in range(candidate["tp"])}
            if (len(workers) != candidate["replicas"]
                    or {worker.get("index") for worker in workers} != set(range(candidate["replicas"]))
                    or any(worker.get("status") != "passed"
                           or len(worker.get("ranks", [])) != len(expected_ranks)
                           or {(rank.get("pp"), rank.get("tp")) for rank in worker.get("ranks", [])} != expected_ranks
                           for worker in workers)):
                if conditional_70b:
                    reasons.append("conditional_70b_full_audit_rank_coverage_incomplete")
                if paired_pd:
                    reasons.append("pd_full_audit_rank_coverage_incomplete")
        reasons.extend(custom_markers(run, args))
    result.update(runtime_supported=not reasons, model_revision=metadata.get("model_revision"),
                  conditional_70b_full_preflight_required=conditional_70b,
                  pd_full_paired_preflight_required=paired_pd,
                  configuration_sha256=config_hashes, numerical_correctness="separate_gate_not_established_by_runtime_support")
    return result, dict(metadata=metadata, config_paths=config_paths,overlap_requested=requested_overlap) if not reasons else None


def development_argv(candidate, passed, out, trace, scale, window):
    old = passed["metadata"]["args"]
    argv = ["--model", candidate["model"], "--family", candidate["family"]]
    for key in ["tp", "pp", "replicas", "prefill_replicas", "pd_backend", "ib_device", "mem", "kv", "running", "graph", "chunk", "baseline_chunk", "inflight", "budget", "startup_timeout", "port"]:
        if key in old and old[key] is not None:
            argv += ["--" + key.replace("_", "-"), str(old[key])]
    for key in ["no_async", "no_cpp", "custom_allreduce"]:
        if old.get(key):
            argv.append("--" + key.replace("_", "-"))
    if passed.get('overlap_requested') is not None:
        argv += ['--overlap',passed['overlap_requested']]
    argv += ["--suite", "trace", "--trace", trace["path"], "--time-scale", str(scale),
             "--arrival-window-sec", str(window), "--out", str(out)]
    extra = list(old.get("extra", []))
    if candidate["family"] in {"muxwise", "muxpp"}:
        require(option(extra, "--pdmux-config-path") is None, "Explicit pre-existing config override needs an audited seed declaration")
        effective = next(path for key, path in passed["config_paths"].items() if key.endswith(".pdmux"))
        extra += ["--pdmux-config-path", str(effective)]
    extra += ["--log-level", "info"]
    if extra:
        argv += ["--extra", *extra]
    return argv


def build(root, support_plan, manifest, tag, *, repo=REPO, models=("8b", "70b"), workloads=("toolagent", "conversation"), rate_70b=.5, serving_compatibility_path=None):
    root, repo = Path(root).resolve(), Path(repo).resolve()
    support_plan, manifest = Path(support_plan).resolve(), Path(manifest).resolve()
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,30}", tag), "Invalid unique draft tag")
    require(math.isfinite(rate_70b) and rate_70b > 0, "70B absolute rate must be positive")
    source = current_sources(repo)
    compatibility=serving_compatibility.load(serving_compatibility_path,source,repo) if serving_compatibility_path else None
    plan = read(support_plan)
    for filename, expected in plan.get("source_sha256", {}).items():
        file=resolve(filename,repo);actual=sha(file)
        key=str(file.relative_to(repo)) if file.is_relative_to(repo) else None
        reviewed_transition=serving_compatibility.source_transition_allowed(key,expected,actual,compatibility)
        require(actual==expected or reviewed_transition, f"Support plan source changed: {filename}")
    traces = trace_records(manifest, root / "traces/manifest.json", repo)
    catalog = support_catalog(plan, repo)
    candidates = [candidate for candidate in read(root / "campaign.json")["candidates"] if candidate["model"] in models]
    audit, tasks, pins = [], [], dict(source)
    for name in ("control.py", "queue.py", "build_development.py", "resource_audit.py", "serving_compatibility.py"):
        path = repo / "scripts/a100_gpu0123" / name
        if path.is_file():
            pins[str(path.relative_to(repo))] = sha(path)
    if compatibility:pins.update(compatibility['artifacts_sha256'])
    accepted = []
    for candidate in candidates:
        record, passed = eligibility(candidate, catalog, root, repo, source,compatibility)
        audit.append(record)
        if passed:
            accepted.append((candidate, passed, record))
    for model in [name for name in ("8b", "70b") if name in models]:
        for workload in [name for name in ("toolagent", "conversation") if name in workloads]:
            require((MODELS[model], workload) in traces or not any(c[0]["model"] == model for c in accepted), "Missing selected model/workload development trace")
            for candidate, passed, record in accepted:
                if candidate["model"] != model:
                    continue
                trace = traces[MODELS[model], workload]
                scale = 1.0 if model == "8b" else trace["requests"] / trace["base_arrival_window_sec"] / rate_70b
                window = trace["base_arrival_window_sec"] * scale
                label = f"development-seed-{candidate['id']}-{workload}-{tag}"
                require(len(label) <= 96, "Draft label too long")
                out = root / "development" / label
                require(not out.exists() and not (root / "runs" / label).exists(), f"Draft label already used: {label}")
                argv = development_argv(candidate, passed, out, trace, scale, window)
                tasks.append(dict(id=label, phase="development", selection_scope="architecture_seed", workload=workload,
                                  candidate_id=candidate["id"], argv=argv, timeout=7000, dependencies=[], trace_sha256=trace["sha256"],
                                  eligibility_evidence=record["evidence_sha256"], support_label=record["support_label"],
                                  absolute_requests_per_sec=trace["requests"] / window,
                                  full_planned_requests=trace["requests"], arrival_window_sec=window))
                for path in passed["config_paths"].values():
                    pins[str(path)] = sha(path)
                pins[trace["stats_path"]] = trace["stats_sha256"]
                pins.update(record["evidence_sha256"])
    queue = dict(schema_version=1, status="draft_not_launched", source_sha256=pins, tasks=tasks,
                 serving_compatibility=compatibility)
    protocol = dict(schema_version=1, status="draft_runtime_supported_seeds_only", launch_performed=False,
                    tag=tag, physical_gpus=[0, 1, 2, 3], support_plan=str(support_plan), support_plan_sha256=sha(support_plan),
                    trace_manifest=str(manifest), trace_manifest_sha256=sha(manifest), traces=list(traces.values()),
                    proposed_rates=dict(model_8b="time_scale=1", model_70b_absolute_requests_per_sec=rate_70b, frozen=False),
                    model_70b_calibration_required_before_adopting_rate=True,
                    full_replay=True, limit_used=False, timeout_sec=7000, eligibility=audit, emitted_tasks=len(tasks),
                    performance_log_level="info",
                    performance_log_policy="Every emitted trace command ends its server extras with --log-level info, overriding support debug verbosity. Resource/custom-allreduce INFO markers and separate NCCL INFO logs remain enabled.",
                    serving_compatibility=compatibility,
                    overlap_seed_policy='Ordinary/PD auto, Mux off; effective-mode change requires new preflight' if compatibility else 'Preserve recorded overlap request; exact source required',
                    budget="One seed per runtime-supported architecture/workload. The equal four-parameter finalist budget follows topology selection and is not spent by these tasks.",
                    cross_queue_dependencies="Eligibility is pinned evidence, not a fabricated cross-queue dependency. No task is emitted for missing/failed evidence.",
                    gates="Numerical equivalence, full correctness and formal readiness remain separate required gates; successful smoke/resources alone do not establish them.")
    return queue, protocol


def write_artifacts(directory, queue, protocol):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    for name, value in [("tasks.json", queue), ("protocol.json", protocol)]:
        with (directory / name).open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--support-plan", type=Path)
    parser.add_argument("--trace-manifest", type=Path)
    parser.add_argument("--models", choices=list(MODELS), nargs="+", default=list(MODELS))
    parser.add_argument("--workloads", choices=["toolagent", "conversation"], nargs="+", default=["toolagent", "conversation"])
    parser.add_argument("--rate-70b", type=float, default=.5)
    parser.add_argument('--serving-compatibility',type=Path,
                        help='Explicit known overlap-runner compatibility artifact; seeds ordinary/PD auto and Mux off')
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    queue, protocol = build(args.root, args.support_plan or args.root / "queues/support-v2.json",
                            args.trace_manifest or args.root / "traces/manifest.json", args.tag,
                            models=args.models, workloads=args.workloads, rate_70b=args.rate_70b,
                            serving_compatibility_path=args.serving_compatibility)
    write_artifacts(args.out_dir, queue, protocol)
    print(json.dumps(dict(status=queue["status"], tasks=len(queue["tasks"]), excluded=sum(not row["runtime_supported"] for row in protocol["eligibility"]),
                          out_dir=str(args.out_dir), gpu_processes_started=0)))


if __name__ == "__main__":
    main()
