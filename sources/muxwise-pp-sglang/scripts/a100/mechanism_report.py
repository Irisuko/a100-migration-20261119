#!/usr/bin/env python3
"""Summarize A100 mechanism evidence without loading the runtime or using a GPU.

Lifetime counters include startup, warmup, measurements, and correctness suites.
Interior snapshot deltas describe observations reported inside a trial, not exact
trial work: CUDA samples retire asynchronously. Missing runs remain explicit.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from analyze_logs import analyze_lines


# Defaults from the frozen server_args.py / pdmux_context.py, not run-name guesses.
OPTIONS = {
    "policy": ("pdmux-pp-policy", "alp"),
    "ttft_target_s": ("pdmux-pp-slo-ttft", 5.0),
    "tpot_target_s": ("pdmux-pp-slo-tpot", .1),
    "tpot_coefficient": ("pdmux-pp-tpot-coeff", 1.0),
    "prefill_chunk_tokens": ("pdmux-pp-prefill-chunk-size", 0),
    "max_inflight_tokens": ("pdmux-pp-max-inflight-tokens", 0),
    "token_refill_size": ("pdmux-pp-token-refill-size", 0),
    "rebalance": ("pdmux-pp-rebalance", True),
    "rebalance_unit": ("pdmux-pp-rebalance-unit", 8.0),
    "rebalance_interval": ("pdmux-pp-rebalance-interval", 32),
    "alp_admission": ("pdmux-pp-alp-admission", True),
    "cpu_prefill_metadata": ("pdmux-pp-cpu-prefill-metadata", False),
    "summary_interval_s": ("pdmux-pp-summary-interval", 10.0),
}
MEASUREMENT_SUITES = {"perf", "pilot", "perf-openloop", "open-loop", "open_loop"}
NOTES = [
    "Lifetime counters include server startup/health checks, warmup, performance, and any correctness suites; cache flush does not reset them.",
    "Partition durations are host scheduler residency including idle/wait time, not CUDA busy time or SM utilization.",
    "RLS readiness is >=16 observations per model; static policy trains models too. Calibration can enable adaptive partition selection before RLS readiness.",
    "Prefill RLS observations count completed local batches/chunks; by_partition prefill observations count layer slices. MAPE is rolling online prediction error, not held-out accuracy.",
    "overlap_launches counts scheduling turns that launch both lanes. It does not prove simultaneous GPU kernels; separate labeled CUDA traces provide that evidence.",
    "Prefetch configuration/idle free slots prove configuration/pool allocation, not receive/compute overlap. Async transport selection is inferred from configuration and source, not overlap counters.",
    "SLO targets constrain predicted partition selection; the policy falls back if no mixed partition is safe. Target configuration does not imply the measured SLO was met.",
    "Admission stops include partial-admission stops; blocked counts entirely rejected probes. admitted counts parent admission calls, not requests.",
    "Token credit bounds unfinished prefill credits, not KV allocations; an oversized head may exceed the cap when the pipeline is empty. Credits return on downstream chunk_done acknowledgements.",
    "Migrations count moved request IDs and may move one request repeatedly. Zero is not observed activation, even when rebalancing is enabled.",
    "Interior deltas exclude coarse timestamp boundaries but CUDA timing samples update on deferred retirement. They are observations inside a trial, never exact per-trial work attribution.",
]


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def activation(value):
    return "observed" if finite(value) and value > 0 else (
        "not_observed" if finite(value) else "unknown")


def timestamp_epoch(value):
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()


def command_option(command, name, default=None):
    result = default
    for index, token in enumerate(command):
        if token == "--no-" + name:
            result = False
        elif token == "--" + name:
            if isinstance(default, bool):
                result = True
            elif index + 1 < len(command):
                result = command[index + 1]
        elif token.startswith("--" + name + "="):
            result = token.split("=", 1)[1]
    if result is None or isinstance(default, bool):
        return result
    return type(default)(result) if default is not None else result


def configuration(launch):
    command, config = launch.get("command", []), launch.get("config") or {}
    result = {key: command_option(command, name, default)
              for key, (name, default) in OPTIONS.items()}
    result.update({
        "pp_profile_decode": config.get("pp_profile_decode", False),
        "prefetch": config.get("pp_prefetch", False),
        "prefetch_slots": config.get("pp_prefetch_slots", 2),
        "async_decode": config.get("pp_async_decode", False),
        "split_forward_token_budget": config.get("split_forward_token_budget"),
        "configuration_source": "launch.command plus launch.config; absent options use frozen runtime defaults",
    })
    return result


def summarize_stage(rank, stage, summaries, policy):
    latest = stage.get("latest_telemetry") or {}
    timing, samples = latest.get("timing") or {}, latest.get("samples") or []
    samples = [samples[i] if len(samples) > i else None for i in range(2)]
    sms = stage.get("sm_counts") or []
    residency = timing.get("partitions") or {}
    partitions = []
    for key, seconds in sorted(residency.items(), key=lambda item: int(item[0])):
        idx = int(key)
        sm = sms[idx] if idx < len(sms) else None
        observations = latest.get("by_partition") or []
        partitions.append({"partition": idx, "sm_counts_prefill_decode": sm,
                           "wall_seconds": seconds,
                           "mixed": bool(sm and min(sm) > 0),
                           "decode_and_prefill_slice_observations": observations[idx] if idx < len(observations) else None})
    total = sum(p["wall_seconds"] for p in partitions if finite(p["wall_seconds"]))
    mixed = sum(p["wall_seconds"] for p in partitions if p["mixed"] and finite(p["wall_seconds"]))
    first_ready = {}
    for idx, label in enumerate(("decode", "prefill")):
        for snapshot in summaries:
            values = snapshot.get("stages", {}).get(str(rank), {}).get("samples") or []
            if len(values) > idx and finite(values[idx]) and values[idx] >= 16:
                first_ready[label] = {"line_number": snapshot.get("line_number"),
                                      "timestamp": snapshot.get("timestamp"), "count": values[idx]}
                break
    slices = timing.get("slices") or {}
    choices = latest.get("slice_choices") or []
    admission = latest.get("admission")
    gates = {}
    for reason in ("inflight_tokens", "inflight_refill"):
        blocked = (admission.get("blocked") or {}).get(reason, 0) if admission else None
        stops = (admission.get("stops") or {}).get(reason, 0) if admission else None
        gates[reason] = {"blocked": blocked, "stops": stops, "activation": activation(stops)}
    counters = stage.get("idle_counter_max") or {}
    overlap = counters.get("overlap_launches")
    return {
        "scope": "server lifetime through latest captured telemetry",
        "latest_summary_line": stage.get("latest_summary_line"),
        "rls": {"decode_observations": samples[0], "prefill_batch_observations": samples[1],
                "warmup_threshold": 16,
                "decode_ready": samples[0] >= 16 if finite(samples[0]) else None,
                "prefill_ready": samples[1] >= 16 if finite(samples[1]) else None,
                "adaptive_policy_selected": policy == "alp",
                "individual_adaptive_decisions_counted": False,
                "first_observed_ready": first_ready, "online_mape": latest.get("mape"),
                "prefill_slice_mape": latest.get("prefill_slice_mape")},
        "partitions": {"residency": partitions, "mixed_wall_seconds": mixed if partitions and sms else None,
                       "mixed_wall_fraction": mixed / total if total and sms else None,
                       "mixed_activation": activation(mixed if partitions and sms else None),
                       "total_wall_seconds": total if partitions else None,
                       "snapshot_transitions_lower_bound": stage.get("partition_transitions_observed")},
        "layer_slicing": {"idle_prefill_slices_max": counters.get("prefill_slices"),
                          "timing_counts": {name: value[0] for name, value in slices.items() if name.startswith("prefill_")},
                          "slice_choices_planned_shortened_layers_saved": choices or None,
                          "decode_aware_shortening": activation(choices[1] if len(choices) > 1 else None)},
        "coissued_launches": {"count": overlap, "activation": activation(overlap),
                              "proves_kernel_overlap": False},
        "admission": {**gates, "blocked": admission.get("blocked") if admission else None,
                      "stops": admission.get("stops") if admission else None,
                      "decisions": admission.get("decisions") if admission else None,
                      "budget_evaluation": latest.get("admission_budget_eval")},
        "cuda_memory_bytes": latest.get("cuda_memory"),
        "cuda_timing_seconds": timing.get("cuda"),
    }


def chunk_evidence(stage, complete, drained):
    telemetry = stage.get("latest_telemetry") or {}
    slices = (telemetry.get("timing") or {}).get("slices") or {}
    finals = slices.get("prefill_final", [None])[0]
    parents = ((telemetry.get("admission") or {}).get("decisions") or {}).get("admitted")
    eligible = complete and drained and stage.get("startup_count") == 1
    excess = finals - parents if finite(finals) and finite(parents) else None
    return {"activation": activation(excess) if eligible else "unknown",
            "prefill_final_slices": finals, "parent_admission_calls": parents,
            "excess_final_slices": excess,
            "eligible_completed_drained_single_startup": bool(eligible),
            "interpretation": "Excess final slices over parent admissions imply token-split child batches only in a successful, drained run with one startup; layer slices alone do not prove token chunking."}


def snapshot_metrics(snapshot):
    result = {}
    samples = snapshot.get("samples") or []
    for idx, name in enumerate(("decode", "prefill")):
        if len(samples) > idx:
            result["rls." + name] = samples[idx]
    for idx, value in enumerate(snapshot.get("slice_choices") or []):
        result[f"slice_choices.{idx}"] = value
    timing = snapshot.get("timing") or {}
    for key, value in (timing.get("partitions") or {}).items():
        result[f"partition_wall_seconds.{key}"] = value
    for key, value in (timing.get("slices") or {}).items():
        result[f"slice_count.{key}"] = value[0]
    admission = snapshot.get("admission")
    if admission:
        for kind in ("blocked", "stops"):
            for reason in ("inflight_tokens", "inflight_refill"):
                result[f"admission.{kind}.{reason}"] = (admission.get(kind) or {}).get(reason, 0)
    return result


def interior_observations(summaries, rank, start, end, interval):
    # Log seconds have up to 1s quantization; downstream status age is <=interval/2.
    margin = max(2.0, interval / 2 + 1)
    selected = []
    for snapshot in summaries:
        if str(rank) not in snapshot.get("stages", {}) or not snapshot.get("timestamp"):
            continue
        timestamp = timestamp_epoch(snapshot["timestamp"])
        if start + margin <= timestamp and timestamp + 1 <= end - margin:
            selected.append(snapshot)
    result = {"boundary_exclusion_seconds": margin, "available": len(selected) >= 2,
              "snapshot_count": len(selected), "exact_trial_attribution": False,
              "qualification": "Reported interior observations only; second-resolution UTC logs, downstream status age, and deferred CUDA retirement prevent exact trial attribution."}
    if len(selected) < 2:
        return result
    metrics = [snapshot_metrics(s["stages"][str(rank)]) for s in selected]
    common = set.intersection(*(set(m) for m in metrics))
    resets, deltas = [], {}
    for key in sorted(common):
        values = [m[key] for m in metrics]
        if not all(finite(v) for v in values):
            continue
        if any(b < a for a, b in zip(values, values[1:])):
            resets.append(key)
            deltas[key] = None
        else:
            deltas[key] = values[-1] - values[0]
    for key, snapshot in (("first_snapshot", selected[0]), ("last_snapshot", selected[-1])):
        result[key] = {"line_number": snapshot["line_number"], "timestamp": snapshot["timestamp"]}
    result.update(observed_deltas=deltas, counter_resets=resets,
                  omitted_noncommon_metrics=sorted(set.union(*(set(m) for m in metrics)) - common))
    return result


def read_json(path, issues, sources):
    if not path.exists():
        return None
    try:
        raw = path.read_bytes()
        sources.append({"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
        return json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"nonfinite JSON: {value}")))
    except (OSError, ValueError) as error:
        issues.append({"path": str(path), "error": str(error)})
        return None


def summarize_run(directory):
    issues, sources = [], []
    launch = read_json(directory / "launch.json", issues, sources)
    outcome = read_json(directory / "outcome.json", issues, sources)
    status = ("completed" if outcome.get("passed") is True else "failed") if outcome else (
        "running" if launch else "missing")
    result = {"status": status, "sources": sources, "issues": issues,
              "outcome": {"passed": outcome.get("passed"), "error": outcome.get("error"),
                          "suite_results": [{k: run.get(k) for k in ("suite", "repeat", "exit_code", "output")}
                                            for run in outcome.get("runs", [])]} if outcome else None}
    if not launch:
        return result
    config = configuration(launch)
    mode = launch.get("mode")
    result.update(mode=mode, configuration=config, lifetime_suites=[r.get("suite") for r in (outcome or {}).get("runs", [])])
    # Prefer final parsed artifacts; for a running process take a labeled log snapshot.
    data = read_json(directory / "scheduler.json", issues, sources) if outcome else None
    if data is None and (directory / "server.log").exists():
        raw = (directory / "server.log").read_bytes()
        sources.append({"path": str((directory / "server.log").resolve()),
                        "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                        "snapshot_of_running_log": outcome is None})
        data = analyze_lines(raw.decode("utf-8", errors="replace").splitlines(), source=str(directory / "server.log"))
    data = data or {}
    stages, summaries = data.get("stages") or {}, data.get("summaries") or []
    result["telemetry_status"] = "not_applicable" if mode == "baseline" else ("available" if stages else "missing")
    result["parser_metadata"] = data.get("metadata")
    result["stages"] = {rank: summarize_stage(rank, stage, summaries, config["policy"])
                        for rank, stage in sorted(stages.items())}
    starts = [run["started_at"] for run in (outcome or {}).get("runs", [])
              if run.get("suite") in MEASUREMENT_SUITES and finite(run.get("started_at"))]
    for stage in result["stages"].values():
        stage["rls"]["ready_observed_before_first_measurement"] = {
            label: True if starts and first.get("timestamp") and timestamp_epoch(first["timestamp"]) + 1 <= min(starts) else None
            for label, first in stage["rls"]["first_observed_ready"].items()
        }
    global_values = summaries[-1].get("global_values", {}) if summaries else {}
    drained = all(global_values.get(k) == 0 for k in ("waiting_requests", "inflight", "pending")) and not any(
        row[0] for row in global_values.get("mbs", []))
    result["token_chunking"] = {"configured_chunk_tokens": config["prefill_chunk_tokens"],
                                **chunk_evidence(stages.get("0", {}), status == "completed", drained)}
    rows = data.get("calibration_rows") or []
    result["calibration"] = {"configured": config["pp_profile_decode"], "observed_rows": len(rows),
                             "expected_matrix_rows": 336 if config["pp_profile_decode"] and mode == "mux" else 0,
                             "per_stage_rows": dict(Counter(str(row["stage"]) for row in rows)),
                             "coverage": data.get("calibration_coverage"),
                             "all_observed_values_positive_finite": all(row.get("valid") for row in rows) if rows else None}
    result["calibration"]["matches_expected_count"] = len(rows) == result["calibration"]["expected_matrix_rows"]
    migrations = data.get("migrations") or {}
    result["migrations"] = {**migrations, "activation": activation(migrations.get("max")),
                            "active": migrations["max"] > 0 if finite(migrations.get("max")) else None,
                            "meaning": "moved request IDs, not unique requests or migration rounds"}
    free_slots = {}
    for check in data.get("idle_checks") or []:
        value = (check.get("counters") or {}).get("free_receive_slots")
        if finite(value):
            rank = str(check["stage"])
            free_slots[rank] = max(free_slots.get(rank, 0), value)
    result["transport"] = {"prefetch_configured": config["prefetch"], "prefetch_slots_configured": config["prefetch_slots"],
                           "idle_free_receive_slots_max": free_slots, "receive_compute_overlap_proven": False,
                           "async_decode_configured": config["async_decode"],
                           "async_transport_selection": "source_inferred_from_configuration" if config["async_decode"] and mode == "mux" else "not_inferred",
                           "async_performance_effect_proven": False}
    measurements = []
    for run in (outcome or {}).get("runs", []):
        suite = run.get("suite")
        if suite not in MEASUREMENT_SUITES:
            continue
        item = {k: run.get(k) for k in ("suite", "repeat", "started_at", "ended_at", "exit_code")}
        path = Path(run["output"]) if run.get("output") else directory / f"{suite}-{run.get('repeat', 0)}.json"
        if not path.exists():
            path = directory / path.name
        measurement = read_json(path, issues, sources)
        item["result_status"] = "available" if measurement else "missing_or_unreadable"
        if measurement:
            item["experiment_slo"] = measurement.get("experiment_slo")
            item["experiment_slo_scope"] = "HTTP request start; client queue delay excluded, including open-loop trials"
            item["aggregate"] = {key: measurement.get("aggregate", {}).get(key) for key in (
                "requested", "completed", "failed", "output_tokens_per_s", "ttft_s", "tpot_s",
                "arrival_ttft_s", "arrival_e2e_s", "queue_delay_s")}
            fixed_slo = measurement.get("aggregate", {}).get("slo") or {}
            item["fixed_target_arrival_slo"] = {key: fixed_slo.get(key) for key in (
                "ttft_target_s", "tpot_target_s", "arrival_ttft_met_fraction",
                "arrival_both_met_fraction", "arrival_scope", "denominator")}
            if not item["experiment_slo"]:
                issues.append({"path": str(path), "error": "Missing experiment_slo; fixed 50ms aggregate.slo is not substituted."})
        else:
            issues.append({"path": str(path), "error": "Measurement result missing or unreadable; coverage is incomplete."})
        if finite(run.get("started_at")) and finite(run.get("ended_at")):
            item["interior_telemetry"] = {rank: interior_observations(
                summaries, rank, run["started_at"], run["ended_at"], config["summary_interval_s"]
            ) for rank in stages}
        measurements.append(item)
    result["measurements"] = measurements
    return result


def build_report(root):
    issues, sources = [], []
    protocol = read_json(root / "main-protocol.json", issues, sources)
    jobs = read_json(root / "mechanism-jobs.json", issues, sources)
    if protocol is None or jobs is None:
        raise ValueError("main-protocol.json and mechanism-jobs.json must be readable JSON")
    names = list(protocol["main_runs"])
    for job in jobs:
        name = command_option(job["args"], "name")
        if not name or name in names or Path(name).name != name:
            raise ValueError(f"invalid/duplicate mechanism job name: {name!r}")
        names.append(name)
    runs = {name: summarize_run(root / name) for name in names}
    counts = dict(Counter(run["status"] for run in runs.values()))
    finished = all(run["status"] in ("completed", "failed") for run in runs.values())
    return {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "complete" if finished else "partial", "all_runs_finished": finished,
            "run_counts": counts, "expected_run_count": len(names), "sources": sources,
            "issues": issues, "timestamp_timezone": "UTC", "notes": NOTES,
            "main_protocol": protocol, "mechanism_jobs": jobs,
            "controlled_comparisons": protocol.get("controlled_comparisons"),
            "runs": runs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("run/a100"))
    parser.add_argument("--output", type=Path, default=Path("run/a100/mechanism-analysis.json"))
    args = parser.parse_args()
    try:
        report = build_report(args.run_root)
        inputs = [s["path"] for s in report["sources"]]
        inputs += [s["path"] for run in report["runs"].values() for s in run["sources"]]
        if str(args.output.resolve()) in inputs:
            raise ValueError("Output cannot overwrite an input artifact")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(f"Wrote {args.output}: {report['status']}, {report['run_counts']}")


if __name__ == "__main__":
    main()
