#!/usr/bin/env python3
"""Draft development selections from explicit queue-task provenance, CPU only.

Only development and rate_calibration phases are accepted. Formal artifacts are
rejected. Comparisons require the same source, trace, planned arrivals and rate.
This command writes a draft review artifact, never a parameter/rate freeze.
"""
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TRACE_MANIFEST = REPO / "run/a100-gpu0123-20261114/traces/manifest.json"
TUNING_TRACE_MANIFEST = TRACE_MANIFEST.with_name("dev-tuning-manifest.json")
TRACE_BUILDER = REPO.parent / "muxwise-pp-experiments/bench/build_trace.py"
MODEL_NAMES = {"8b": "Llama-3.1-8B-Instruct", "70b": "Llama-3.3-70B-Instruct"}
sys.path.insert(0, str(HERE / "bench"))
_spec = importlib.util.spec_from_file_location("_development_cutoff_metrics", HERE / "bench/metrics.py")
metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(metrics)
_platform_spec = importlib.util.spec_from_file_location("_development_platform", HERE / "run_campaign.py")
_platform = importlib.util.module_from_spec(_platform_spec)
_platform_spec.loader.exec_module(_platform)
read_platform_provenance = _platform.read_platform_provenance
PRIMARY = "ttft_le_5s_and_tpot_le_100ms"
ALLOWED_PHASES = {"development", "rate_calibration"}
EXPECTED_SYSTEMS = {"chunked", "muxwise", "muxpp", "pd"}
NON_CONFIGURATION_OPTIONS = {"out", "trace", "time_scale", "limit", "port", "startup_timeout", "suite", "arrival_window_sec"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def resolve(value):
    path = Path(value)
    return (path if path.is_absolute() else REPO / path).resolve()


def option(argv, name):
    found = []
    for i, value in enumerate(argv):
        if value == "--extra":
            break
        if value == name:
            require(i + 1 < len(argv), f"Missing value: {name}")
            found.append(argv[i + 1])
        elif value.startswith(name + "="):
            found.append(value[len(name) + 1:])
    require(len(found) <= 1, f"Duplicate option: {name}")
    return found[0] if found else None


def queued_identity(task, evidence, platform_configuration):
    """Retain candidate/load identity even when the process never wrote results."""
    argv, parameters, i = task["argv"], {}, 0
    while i < len(argv):
        token = argv[i]
        require(token.startswith("--"), f"Unexpected queued argument: {token}")
        if token == "--extra":
            parameters["extra"] = argv[i + 1:]
            break
        key, separator, value = token[2:].partition("=")
        key = key.replace("-", "_")
        if not separator:
            value = True
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                i += 1
                value = argv[i]
        require(key not in parameters, f"Duplicate queued argument: {key}")
        parameters[key] = value
        i += 1
    load = dict(trace_sha256=evidence.get("trace_sha256"),
                trace=str(resolve(option(argv, "--trace"))), workload=task.get("workload"),
                time_scale=float(option(argv, "--time-scale") or 1),
                limit=int(option(argv, "--limit")) if option(argv, "--limit") is not None else None,
                arrival_window_sec=(float(option(argv, "--arrival-window-sec"))
                                    if option(argv, "--arrival-window-sec") is not None else None))
    config = {key: value for key, value in parameters.items() if key not in NON_CONFIGURATION_OPTIONS}
    return dict(queued_configuration=config, queued_load=load,
                queued_repeat_fingerprint=digest(dict(configuration=config, load=load,
                                                      platform_configuration=platform_configuration)))


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def trace_metadata(path, cache):
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in cache:
        rows = []
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    rows.append({k: row[k] for k in ["idx", "arrival", "input_len", "output_len"]})
        cache[key] = dict(sha256=sha256(path), rows=rows)
    return cache[key]


def audited_json(path, cache, description, expected_hash=None):
    path = resolve(path)
    require(path.is_file(), f"Missing {description}: {path}")
    stat = path.stat()
    key = ("audited_json", str(path), stat.st_size, stat.st_mtime_ns)
    if key not in cache:
        cache[key] = dict(value=json.loads(path.read_text()), sha256=sha256(path))
    evidence = cache[key]
    require(expected_hash is None or evidence["sha256"] == expected_hash,
            f"{description} hash mismatch: {path}")
    return evidence


def audit_development_trace(trace, traced, model, workload, cache):
    """Only adopted manifests authorize bytes; raw indices also prove the split."""
    require(model in MODEL_NAMES, "Unknown model in development trace allowlist")
    require(workload is None or workload in {"toolagent", "conversation"},
            "Unknown workload in development trace allowlist")
    parent = audited_json(TRACE_MANIFEST, cache, "development trace manifest")
    inventories = [(resolve(TRACE_MANIFEST), parent, "dev")]
    if TUNING_TRACE_MANIFEST.is_file():
        tuning = audited_json(TUNING_TRACE_MANIFEST, cache, "development tuning manifest")
        require(resolve(tuning["value"].get("parent_manifest", ".")) == resolve(TRACE_MANIFEST)
                and tuning["value"].get("parent_manifest_sha256") == parent["sha256"],
                "Tuning manifest parent provenance differs from adopted development manifest")
        inventories.append((resolve(TUNING_TRACE_MANIFEST), tuning, "dev_tuning"))
    require(TRACE_BUILDER.is_file(), "Missing canonical trace builder")
    builder_hash = sha256(TRACE_BUILDER)
    matches, prohibited = [], set()
    for path, evidence, purpose in inventories:
        inventory = evidence["value"]
        require(inventory.get("schema_version") == 1 and inventory.get("status") == "PASS",
                f"Development trace manifest requires schema1 PASS audit: {path}")
        builder = inventory.get("builder", {})
        require(resolve(builder.get("path", ".")) == resolve(TRACE_BUILDER)
                and builder.get("sha256") == builder_hash,
                f"Development trace manifest builder provenance changed: {path}")
        for row in inventory.get("records", []):
            if row.get("purpose") in {"formal", "heldout"}:
                prohibited.add(row.get("sha256"))
            if (row.get("purpose") == purpose and row.get("sha256") == traced["sha256"]
                    and row.get("model") == MODEL_NAMES[model]
                    and (workload is None or row.get("workload") == workload)):
                matches.append((path, evidence, row))
    require(traced["sha256"] not in prohibited,
            "A formal or heldout trace cannot be used for development selection")
    require(len(matches) == 1,
            "Trace hash/model/workload is not uniquely allowlisted as audited development data")
    manifest_path, manifest, row = matches[0]
    require(row.get("workload") in {"toolagent", "conversation"}
            and row.get("all_rows_and_hash_blocks_validated") is True
            and row.get("length_clipping") is False and row.get("padding_tokens") == 0,
            "Allowlisted development trace lacks complete row/hash-block audit")
    canonical = resolve(row["path"])
    require(canonical.is_file() and trace_metadata(canonical, cache)["sha256"] == row["sha256"],
            "Canonical development trace hash differs from allowlist")
    stats = audited_json(row["stats_path"], cache, "Development stats", row["stats_sha256"])["value"]
    start, end, stretch = row["source_start_sec"], row["source_end_exclusive_sec"], row["stretch"]
    require(all(isinstance(v, (int, float)) and math.isfinite(v) for v in [start, end, stretch])
            and 0 <= start < end and stretch > 0,
            "Invalid audited development raw window")
    require(stats.get("requests") == row["requests"] == len(traced["rows"])
            and stats.get("start_sec") == start
            and stats.get("source_duration_sec") == end - start
            and stats.get("stretch") == stretch
            and stats.get("duration_sec") == row["base_arrival_window_sec"] == (end - start) * stretch,
            "Audited development stats/window/count provenance differs")
    allowed_parents = [p for p in parent["value"]["records"] if p.get("purpose") == "dev"
                       and p.get("model") == row["model"] and p.get("workload") == row["workload"]
                       and p["source_start_sec"] <= start < end <= p["source_end_exclusive_sec"]]
    require(len(allowed_parents) == 1, "Trace raw window is not contained in adopted parent development window")
    if row["purpose"] == "dev_tuning":
        p = allowed_parents[0]
        require(row.get("parent_trace_sha256") == p["sha256"]
                and resolve(row.get("parent_trace_path", ".")) == resolve(p["path"])
                and row.get("all_child_rows_byte_identical_to_parent") is True
                and row.get("arrival_unchanged_from_parent") is True,
                "Tuning trace lacks exact parent-row provenance")
    sources = [s for s in parent["value"].get("sources", []) if s.get("workload") == row["workload"]]
    require(len(sources) == 1, "Missing unique audited raw workload source")
    source = sources[0]
    raw_path = resolve(source["path"])
    require(raw_path.is_file(), "Missing audited raw trace source")
    raw_stat = raw_path.stat()
    raw_key = ("raw_source", str(raw_path), raw_stat.st_size, raw_stat.st_mtime_ns)
    if raw_key not in cache:
        cache[raw_key] = dict(sha256=sha256(raw_path), rows=read_jsonl(raw_path))
    raw = cache[raw_key]
    require(raw["sha256"] == source["sha256"], "Audited raw source hash mismatch")
    selected = sorted([(i, r) for i, r in enumerate(raw["rows"])
                       if start * 1000 <= r["timestamp"] < end * 1000
                       and r["input_length"] + r["output_length"] <= 32768],
                      key=lambda item: (item[1]["timestamp"], item[0]))
    indices = [r["idx"] for r in traced["rows"]]
    require(indices == [i for i, _ in selected] and len(indices) == len(set(indices))
            and digest(indices) == row.get("source_indices_sha256"),
            "Development trace source rows do not match the complete audited raw window")
    forbidden_windows = [(p["source_start_sec"] * 1000, p["source_end_exclusive_sec"] * 1000)
                         for p in parent["value"]["records"]
                         if p.get("purpose") in {"formal", "heldout"} and p.get("workload") == row["workload"]]
    for actual, (_, original) in zip(traced["rows"], selected):
        require(actual["input_len"] == original["input_length"]
                and actual["output_len"] == original["output_length"]
                and actual["arrival"] == (original["timestamp"] - start * 1000) / 1000 * stretch,
                "Development trace row length/arrival differs from audited raw source")
        require(not any(a <= original["timestamp"] < b for a, b in forbidden_windows),
                "Development trace contains formal or heldout source rows")
    return dict(manifest_path=str(manifest_path), manifest_sha256=manifest["sha256"],
                record_name=row["name"], purpose=row["purpose"], model=row["model"], workload=row["workload"],
                trace_sha256=row["sha256"], stats_path=str(resolve(row["stats_path"])), stats_sha256=row["stats_sha256"],
                builder_sha256=builder_hash, raw_source_sha256=source["sha256"],
                source_indices_sha256=row["source_indices_sha256"], source_start_sec=start,
                source_end_exclusive_sec=end, base_arrival_window_sec=row["base_arrival_window_sec"],
                checked_source_rows=len(indices), formal_or_heldout_source_rows=0)


def read_measurement(task_dir, cache):
    task_dir = resolve(task_dir)
    provenance_path = task_dir / "queue-task.json"
    provenance = json.loads(provenance_path.read_text())
    task = provenance["task"]
    require(task.get("phase") in ALLOWED_PHASES,
            f"Selection input phase must be development or rate_calibration: {task_dir}: {task.get('phase')}")
    platform = read_platform_provenance(task_dir.parent.parent, provenance)
    platform_configuration = platform["configuration"]
    platform_software = platform["software"]
    out_value = option(task["argv"], "--out")
    require(out_value is not None, f"Task lacks --out: {task_dir}")
    out = resolve(out_value)
    required = [out / name for name in ["run.json", "trace_summary.json", "trace_requests.jsonl", "trace_planned.jsonl"]]
    required.append(task_dir / "outcome.json")
    missing = [str(path) for path in required if not path.is_file()]
    metadata = json.loads((out / "run.json").read_text()) if (out / "run.json").is_file() else {}
    evidence = provenance.get("evidence", {})
    trace_value = option(task["argv"], "--trace")
    require(trace_value is not None, "Development task lacks explicit trace provenance")
    trace = resolve(trace_value)
    traced = trace_metadata(trace, cache)
    require(evidence.get("trace_sha256") == traced["sha256"], f"Execution trace hash mismatch: {trace}")
    queued_model = option(task["argv"], "--model")
    if metadata.get("args"):
        require(metadata["args"].get("model") == queued_model, "Queue and runtime model provenance differ")
    trace_audit = audit_development_trace(trace, traced, queued_model, task.get("workload"), cache)
    base = dict(label=task["id"], phase=task["phase"], task_run=str(task_dir), measurement_directory=str(out),
                queue_task_sha256=sha256(provenance_path), argv=task["argv"],
                trace_audit=trace_audit,
                platform_provenance=platform, platform_configuration=platform_configuration,
                platform_software=platform_software,
                configuration_sha256=metadata.get("configuration_sha256", {}),
                model_revision=metadata.get("model_revision"), model_provenance=metadata.get("model_provenance", {}),
                **queued_identity(task, evidence, platform_configuration))
    if missing:
        return dict(base, measurement_available=False, selection_eligible=False,
                    exclusion_reasons=["missing_trace_artifacts"], missing_artifacts=missing,
                    source_sha256=metadata.get("source_sha256") or evidence.get("source_sha256", {}))
    require(metadata.get("phase", task["phase"]) == task["phase"], "Conflicting measurement phase provenance")
    summary = json.loads((out / "trace_summary.json").read_text())
    outcome = json.loads((task_dir / "outcome.json").read_text())
    args = metadata["args"]
    source = metadata.get("source_sha256") or provenance.get("evidence", {}).get("source_sha256")
    require(isinstance(source, dict) and source, f"Missing source fingerprint: {task_dir}")
    queued_source = provenance.get("evidence", {}).get("source_sha256", {})
    def serving_source(filename):
        path = resolve(filename)
        return (path.is_relative_to(REPO / "python/sglang") or path == HERE / "server_run.py"
                or path.parent == HERE / "bench" and path.suffix == ".py" and not path.name.startswith("test_"))
    def normalized_sources(mapping):
        normalized = {}
        for filename, value in mapping.items():
            path = resolve(filename)
            require(path not in normalized or normalized[path] == value,
                    "Conflicting source hashes for aliases of the same path")
            normalized[path] = value
        return normalized
    runtime_paths, queued_paths = normalized_sources(source), normalized_sources(queued_source)
    require(all(path in runtime_paths for path in queued_paths if serving_source(path)),
            "Queued serving source hash missing from runtime evidence")
    require(all(runtime_paths[path] == value for path, value in queued_paths.items() if path in runtime_paths),
            "Queue and runtime source hashes conflict")
    require(resolve(summary["trace"]) == trace, "Queue and summary trace paths differ")
    planned = read_jsonl(out / "trace_planned.jsonl")
    records = read_jsonl(out / "trace_requests.jsonl")
    require(len(planned) == summary["planned_requests"] and planned, "Planned request denominator mismatch")
    scale = float(summary["time_scale"])
    require(math.isfinite(scale) and scale > 0 and float(option(task["argv"], "--time-scale") or 1) == scale,
            "Replay time-scale differs from task provenance")
    expected_rows = traced["rows"][:len(planned)]
    require(len(planned) <= len(traced["rows"]), "Plan exceeds source trace")
    for actual, expected in zip(planned, expected_rows):
        require(all(actual[k] == expected[k] for k in ["idx", "input_len", "output_len"])
                and math.isclose(actual["arrival"], expected["arrival"] * scale, abs_tol=1e-9),
                "Planned request differs from exact trace prefix/time scale")
    planned_ids = {row["idx"]: row for row in planned}
    require(len(planned_ids) == len(planned), "Duplicate planned request IDs")
    window = float(summary["arrival_window_sec"])
    require(math.isfinite(window) and window > 0 and window >= max(row["arrival"] for row in planned),
            "Invalid planned arrival-window denominator")
    explicit_window = option(task["argv"], "--arrival-window-sec")
    if explicit_window is not None:
        expected_window = float(explicit_window)
    elif option(task["argv"], "--limit") is None:
        expected_window = trace_audit["base_arrival_window_sec"] * scale
    else:
        expected_window = max(row["arrival"] for row in planned)
    require(math.isfinite(expected_window) and math.isclose(window, expected_window, rel_tol=1e-12, abs_tol=1e-9),
            "Summary arrival-window differs from queued window or trace-duration/time-scale provenance")
    for row in records:
        require(row["idx"] in planned_ids, "Unexpected request in measurement")
        intent = planned_ids[row["idx"]]
        require(row["input_len"] == intent["input_len"] and row["output_len"] == intent["output_len"],
                "Measured request length differs from plan")
        first, finished, sent = row.get("first_token_at"), row.get("finished_at"), row["sent_at"]
        require(first is None or first >= sent, "First token precedes send")
        require(finished is None or finished >= sent, "Completion precedes send")
        require(first is None or finished is None or finished >= first, "Completion precedes first token")
        row["send_lag"] = sent - intent["arrival"]
        row["ttft"] = first - sent if first is not None else None
        row["e2e"] = finished - sent if row["success"] and finished is not None else None
        count = max((chunk["completion_tokens"] for chunk in row.get("chunks", [])), default=0)
        row["tpot"] = ((finished - first) / (count - 1)
                       if row["success"] and first is not None and finished is not None and count > 1 else None)
    historical = metrics.summarize_records(records, len(planned), summary.get("cutoff_time"))
    arrival = metrics.summarize_arrivals(records, planned, summary.get("cutoff_time"), window)
    excluded = []
    execution_eligible = bool(summary.get("measurement_eligible") and summary.get("run_status") == "completed"
                              and not summary.get("interrupted") and outcome.get("status") == "completed"
                              and outcome.get("returncode", 0) == 0)
    if not execution_eligible:
        excluded.append("execution_not_eligible")
    if historical["completed"] != len(planned):
        excluded.append("not_all_planned_completed")
    if summary.get("limit") is not None or len(planned) != len(traced["rows"]):
        excluded.append("partial_trace")
    if summary.get("output_id_validation", {}).get("successful_requests_with_verified_ids") != historical["completed"]:
        excluded.append("missing_output_id_verification")
    if historical["latencies_sec"]["ttft"]["finite_count"] != historical["completed"]:
        excluded.append("missing_completed_first_token")
    send_lag = historical["latencies_sec"]["send_lag"]
    historical_slo, arrival_slo = historical["slo"][PRIMARY]["rate"], arrival["slo"][PRIMARY]["rate"]
    flags = []
    if send_lag["p99"] is not None and send_lag["p99"] > .1:
        flags.append("send_lag_p99_gt_100ms")
    if historical_slo - arrival_slo > .01:
        flags.append("arrival_slo_more_than_1pp_below_send_slo")
    parameters = {key: value for key, value in args.items() if key not in NON_CONFIGURATION_OPTIONS}
    workload = trace_audit["workload"]
    intent_fingerprint = digest([{k: row[k] for k in ["idx", "arrival", "input_len", "output_len"]} for row in planned])
    return dict(base, measurement_available=True, model=args["model"], workload=workload, system=args["family"],
                parameters=parameters,
                configuration_fingerprint=digest(dict(parameters=parameters, configuration_sha256=base["configuration_sha256"],
                                                       platform_configuration=platform_configuration)),
                model_fingerprint=digest(dict(revision=base["model_revision"], provenance=base["model_provenance"])),
                source_sha256=source,
                source_fingerprint=digest(dict(runtime=source, platform_software=platform_software)),
                trace=str(trace), trace_sha256=traced["sha256"],
                planned_intent_fingerprint=intent_fingerprint, request_details_sha256=sha256(out / "trace_requests.jsonl"),
                planned_manifest_sha256=sha256(out / "trace_planned.jsonl"),
                planned=len(planned), completed=historical["completed"], unfinished=historical["unfinished"],
                failed=historical["failed"], unsent=historical["unsent_requests"],
                planned_arrival_window_sec=window, absolute_requests_per_sec=len(planned) / window,
                time_scale=scale, historical_main_slo=historical_slo, arrival_main_slo=arrival_slo,
                ttft_p95_sec=historical["latencies_sec"]["ttft"]["p95"],
                arrival_ttft_p95_sec=arrival["latencies_sec"]["ttft"]["p95"],
                tpot_p95_sec=historical["latencies_sec"]["tpot"]["p95"],
                output_tokens_per_sec=historical["throughput"]["output_tokens_per_sec"],
                measured_duration_sec=historical["duration_sec"], send_lag_sec=send_lag,
                arrival_lag_flags=flags, selection_eligible=not excluded, exclusion_reasons=excluded,
                execution_eligible=execution_eligible, sustainable_window_ge_180sec=window >= 180,
                historical_metrics=historical, arrival_metrics=arrival)


def finite_or_infinity(value):
    return value if isinstance(value, (float, int)) and math.isfinite(value) else math.inf


def aggregate(rows):
    first = rows[0]
    result = {key: first[key] for key in ["model", "model_fingerprint", "model_revision", "model_provenance", "workload", "system", "parameters", "configuration_sha256", "source_sha256", "source_fingerprint",
              "trace_sha256", "trace_audit", "platform_configuration", "platform_software",
              "planned_intent_fingerprint", "absolute_requests_per_sec", "configuration_fingerprint"]}
    p95s = [finite_or_infinity(row["ttft_p95_sec"]) for row in rows]
    mean_p95 = statistics.mean(p95s)
    throughputs = [row["output_tokens_per_sec"] for row in rows]
    mean_throughput = (statistics.mean(throughputs)
                       if all(isinstance(value, (int, float)) and math.isfinite(value) for value in throughputs) else None)
    result.update(labels=[row["label"] for row in rows], task_runs=[row["task_run"] for row in rows],
                  phases=sorted({row["phase"] for row in rows}), measurement_count=len(rows),
                  platform_provenance=[row["platform_provenance"] for row in rows],
                  selection_eligible=all(row["selection_eligible"] for row in rows),
                  historical_main_slo=statistics.mean(row["historical_main_slo"] for row in rows),
                  ttft_p95_sec=mean_p95 if math.isfinite(mean_p95) else ">cutoff",
                  output_tokens_per_sec=mean_throughput,
                  arrival_lag_flags=sorted({flag for row in rows for flag in row["arrival_lag_flags"]}),
                  missing_repeat_labels=sorted({label for row in rows for label in row.get("missing_repeat_labels", [])}),
                  exclusion_reasons=sorted({reason for row in rows for reason in row["exclusion_reasons"]}))
    return result


def rank_key(row):
    throughput = row["output_tokens_per_sec"]
    throughput_key = -throughput if isinstance(throughput, (int, float)) and math.isfinite(throughput) else math.inf
    return (-row["historical_main_slo"], finite_or_infinity(row["ttft_p95_sec"]),
            throughput_key, row["configuration_fingerprint"])


def suggest_rates(rows, light_slo, overload_slo):
    groups = defaultdict(list)
    for row in rows:
        if (row["measurement_available"] and row["execution_eligible"]
                and not {"partial_trace", "missing_repeat_artifacts"}.intersection(row["exclusion_reasons"])):
            groups[(row["model"], row["model_fingerprint"], row["workload"], row["source_fingerprint"], row["trace_sha256"])].append(row)
    suggestions = []
    for (model, model_fingerprint, workload, source, trace), items in groups.items():
        systems = sorted({row["system"] for row in items})
        candidate_rates = sorted(set.intersection(*[{row["absolute_requests_per_sec"] for row in items if row["system"] == system} for system in systems]))
        common, mismatched_intent_rates = [], []
        evidence = []
        for rate in candidate_rates:
            intents = set.intersection(*[
                {row["planned_intent_fingerprint"] for row in items
                 if row["system"] == system and row["absolute_requests_per_sec"] == rate}
                for system in systems])
            if not intents:
                mismatched_intent_rates.append(rate)
                continue
            common.append(rate)
            for intent in sorted(intents):
                system_rows = {}
                for system in systems:
                    candidates = [row for row in items if row["system"] == system
                                  and row["absolute_requests_per_sec"] == rate and row["planned_intent_fingerprint"] == intent]
                    configs = defaultdict(list)
                    for row in candidates:
                        configs[row["configuration_fingerprint"]].append(row)
                    # Compare repeat means; never select the best single repeat.
                    best = sorted(configs.values(), key=lambda group: (-statistics.mean(r["historical_main_slo"] for r in group),
                                  statistics.mean(finite_or_infinity(r["ttft_p95_sec"]) for r in group)))[0]
                    system_rows[system] = dict(main_slo=statistics.mean(r["historical_main_slo"] for r in best),
                                              unfinished=sum(r["unfinished"] for r in best),
                                              labels=[r["label"] for r in best],
                                              sustained_window_ge_180sec=all(r["sustainable_window_ge_180sec"] for r in best))
                evidence.append(dict(absolute_requests_per_sec=rate, planned_intent_fingerprint=intent, systems=system_rows,
                                     worst_system_main_slo=min(r["main_slo"] for r in system_rows.values()),
                                     any_unfinished=any(r["unfinished"] for r in system_rows.values())))
        light = [rate for rate in common if all(r["worst_system_main_slo"] >= light_slo and not r["any_unfinished"]
                                               for r in evidence if r["absolute_requests_per_sec"] == rate)]
        critical = [r["absolute_requests_per_sec"] for r in evidence if r["worst_system_main_slo"] < light_slo]
        overload = [r["absolute_requests_per_sec"] for r in evidence if r["any_unfinished"] or r["worst_system_main_slo"] <= overload_slo]
        suggestions.append(dict(model=model, model_fingerprint=model_fingerprint, workload=workload, source_fingerprint=source, trace_sha256=trace,
                                systems=systems, missing_expected_systems=sorted(EXPECTED_SYSTEMS - set(systems)),
                                rates_without_common_arrival_intent=mismatched_intent_rates,
                                common_measured_rates=common, light_requests_per_sec=min(light) if light else None,
                                critical_requests_per_sec=min(critical) if critical else None,
                                overload_requests_per_sec=max(overload) if overload else None,
                                evidence=evidence, status="draft_observed_rate_bracket_not_frozen",
                                caveat="Rates form a configuration envelope within observed systems; SLO-stress is not proof of throughput saturation. Missing systems/rates and short windows need development coverage before freezing."))
    return suggestions


def analyze(task_dirs, separate_sources=False, light_slo=.95, overload_slo=.5):
    require(0 <= overload_slo < light_slo <= 1, "Require 0 <= overload SLO < light SLO <= 1")
    task_dirs = [resolve(path) for path in task_dirs]
    require(len(set(task_dirs)) == len(task_dirs), "Duplicate task-run input")
    cache = {}
    rows = [read_measurement(path, cache) for path in task_dirs]
    measured = [row for row in rows if row["measurement_available"]]
    require(len({row["measurement_directory"] for row in rows}) == len(rows), "Repeated measurement directory")
    unavailable = [row for row in rows if not row["measurement_available"]]
    for row in measured:
        missing_repeats = []
        for missing in unavailable:
            if missing["queued_repeat_fingerprint"] != row["queued_repeat_fingerprint"]:
                continue
            if missing["platform_software"] != row["platform_software"]:
                continue
            # Known different executable/config/model evidence identifies another
            # candidate; absent evidence cannot excuse a failed queued repeat.
            if any(value != row["source_sha256"][key] for key, value in missing["source_sha256"].items()
                   if key in row["source_sha256"]):
                continue
            if missing["configuration_sha256"] and row["configuration_sha256"] and missing["configuration_sha256"] != row["configuration_sha256"]:
                continue
            if missing["model_revision"] and row["model_revision"] and missing["model_revision"] != row["model_revision"]:
                continue
            missing_repeats.append(missing["label"])
        if missing_repeats:
            row["selection_eligible"] = False
            row["exclusion_reasons"].append("missing_repeat_artifacts")
            row["missing_repeat_labels"] = sorted(missing_repeats)
    sources = defaultdict(set)
    models = defaultdict(set)
    for row in measured:
        sources[(row["model"], row["workload"])].add(row["source_fingerprint"])
        models[row["model"]].add(row["model_fingerprint"])
    require(all(len(values) <= 1 for values in models.values()), "Compared runs have changed model provenance")
    require(separate_sources or all(len(values) <= 1 for values in sources.values()),
            "Compared configurations have changed source fingerprints; explicitly use --separate-sources")
    groups = defaultdict(list)
    for row in measured:
        key = (row["model"], row["model_fingerprint"], row["workload"], row["source_fingerprint"], row["trace_sha256"],
               row["planned_intent_fingerprint"], row["absolute_requests_per_sec"], row["system"], row["configuration_fingerprint"])
        groups[key].append(row)
    aggregates = [aggregate(items) for items in groups.values()]
    comparable = defaultdict(list)
    for group in aggregates:
        if group["selection_eligible"]:
            key = tuple(group[field] for field in ["model", "model_fingerprint", "workload", "source_fingerprint", "trace_sha256",
                        "planned_intent_fingerprint", "absolute_requests_per_sec", "system"])
            comparable[key].append(group)
    selected = []
    for candidates in comparable.values():
        best = dict(sorted(candidates, key=rank_key)[0])
        best.update(status="draft_best_within_equal_load_group", eligible_configurations_compared=len(candidates))
        selected.append(best)
    return dict(schema_version=2, status="draft_for_review", freeze_created=False, separate_sources=separate_sources,
                measurements=rows, configuration_aggregates=aggregates, draft_selections=selected,
                rate_suggestions=suggest_rates(rows, light_slo, overload_slo),
                definitions=dict(accepted_phases=sorted(ALLOWED_PHASES),
                    trace="Exact model/workload/hash allowlist from adopted audited dev/dev_tuning manifests; full raw-window source indices checked, with no formal/heldout overlap. Unknown traces fail closed, including missing-result runs.",
                    primary="Historical send-relative TTFT<=5s and TPOT<=100ms, denominator ALL planned requests.",
                    ranking="Repeat-mean primary SLO descending; repeat-mean TTFT p95 ascending; repeat-mean full-drain output tokens/s descending. Every included repeat must complete the entire trace successfully.",
                    rate="Exact planned request count divided by explicit planned arrival-window seconds; never last-send or drain duration.",
                    light=f"Lowest tested common rate with every observed system SLO >= {light_slo} and no unfinished requests.",
                    critical=f"Lowest tested common rate at which any observed system SLO < {light_slo}.",
                    overload=f"Highest tested common rate with unfinished requests or any observed system SLO <= {overload_slo}; SLO stress only, not a saturation proof.",
                    arrival_lag="Separate flags: send lag p99 >100ms or arrival SLO >1 percentage point below historical SLO.",
                    source="Stored runtime fingerprints plus validated loader/library/package software identity; current workspace runtime source is not substituted for historical execution source.",
                    platform="Verified outer-launch Mooncake transport belongs to candidate configuration. Loader environment, libraries and packages belong to source identity. Queue IDs, timestamps and task/manifest paths are evidence only. Legacy missing platform evidence is explicit unrecorded configuration/software and cannot be mixed with verified platform cohorts."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-runs", nargs="*", type=Path, default=[])
    parser.add_argument("--separate-sources", action="store_true")
    parser.add_argument("--light-slo", type=float, default=.95)
    parser.add_argument("--overload-slo", type=float, default=.5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.task_runs, args.separate_sources, args.light_slo, args.overload_slo)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status=result["status"], measurements=len(result["measurements"]),
                          draft_selections=len(result["draft_selections"]), freeze_created=False, out=str(args.out))))


if __name__ == "__main__":
    main()
