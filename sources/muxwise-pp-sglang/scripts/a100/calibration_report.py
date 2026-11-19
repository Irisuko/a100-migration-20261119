#!/usr/bin/env python3
"""Aggregate measured A100 decode tables and reproduce the initial ALP safety gate.

CPU only. Isolated graph replay measurements are not concurrent-request deadlines.
The startup scale is 1.2; actual runtime scales can grow from observed contention.
"""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


KEYS = ("stage", "partition", "prefill_sms", "decode_sms", "batch_size", "context_tokens")


def summary(values):
    values = list(values)
    mean = statistics.mean(values)
    return {"count": len(values), "median": statistics.median(values),
            "min": min(values), "max": max(values), "mean": mean,
            "cv_population": statistics.pstdev(values) / mean if mean else None}


def envelope(records, bs, context):
    """Same upper-corner/envelope semantics as pp_policy.calibrated_decode_time."""
    eligible = [(b, c, t) for b, c, t in records if b >= bs and c >= context]
    if not eligible:
        return None
    upper_b = min(row[0] for row in eligible)
    upper_c = min(row[1] for row in eligible)
    return max(t for b, c, t in records if b <= upper_b and c <= upper_c)


def command_value(command, flag, default=None):
    try:
        return command[command.index(flag) + 1]
    except (ValueError, IndexError):
        return default


def signature(launch):
    """Keep runtime/model/SM changes separate while allowing chunk/policy ablations."""
    command = launch.get("command", [])
    config = launch.get("config", {})
    flags = ("--tp-size", "--pp-size", "--dtype", "--context-length",
             "--max-total-tokens", "--max-micro-batch-size", "--cuda-graph-max-bs",
             "--attention-backend", "--sampling-backend")
    result = {"model": launch["model"], "commit": launch.get("repository_commit"),
              "command_settings": {flag: command_value(command, flag) for flag in flags},
              "sm_pairs": [row[:2] for row in config.get("manual_divisions", [])],
              "sm_group_num": config.get("sm_group_num"),
              "extra_contexts": config.get("pp_decode_extra_contexts", []),
              "gpu_ids": launch.get("environment", {}).get("CUDA_VISIBLE_DEVICES")}
    encoded = json.dumps(result, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()[:12], result


def read_runs(paths, model_name):
    runs, excluded, issues = [], [], []
    for path in sorted(paths):
        launch_path = path.parent / "launch.json"
        if not launch_path.exists():
            excluded.append({"source": str(path), "reason": "missing launch.json"})
            continue
        launch = json.loads(launch_path.read_text())
        model = Path(launch.get("model", "")).name
        if model != model_name or launch.get("mode") != "mux":
            excluded.append({"source": str(path), "reason": "different model or non-mux mode"})
            continue
        data = json.loads(path.read_text())
        rows = data.get("calibration_rows", [])
        if not rows:
            excluded.append({"source": str(path), "reason": "no calibration rows"})
            continue
        valid, seen = [], set()
        for row in rows:
            key = tuple(row.get(k) for k in KEYS)
            seconds = row.get("seconds")
            if not row.get("valid") or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
                issues.append({"source": str(path), "reason": "invalid calibration row", "key": key})
                continue
            if key in seen:
                raise ValueError(f"Repeated coordinate within one startup: {path} {key}")
            seen.add(key)
            valid.append(row)
        group, settings = signature(launch)
        runs.append({"name": path.parent.name, "source": str(path), "group": group,
                     "settings": settings, "rows": valid,
                     "calibration_elapsed": data.get("calibration_elapsed", []),
                     "parser_errors": data.get("metadata", {}).get("parse_errors", [])})
    return runs, excluded, issues


def aggregate(runs, contexts=(128, 512, 1024, 2048, 4096, 8192), batches=(1, 4, 8, 16), targets_ms=(10, 20, 50)):
    cells = defaultdict(list)
    startup_counts = defaultdict(int)
    for run in runs:
        startup_counts[run["group"]] += 1
        for row in run["rows"]:
            cells[(run["group"], *(row[k] for k in KEYS))].append((run["name"], row["seconds"]))
    measured = []
    for key, values in sorted(cells.items()):
        measured.append({"group": key[0], **dict(zip(KEYS, key[1:])),
                         **{f"seconds_{k}": v for k, v in summary(v for _, v in values).items()},
                         "runs": [name for name, _ in values]})
    per_run_predictions = []
    for run in runs:
        tables = defaultdict(list)
        sm_pairs = {}
        for row in run["rows"]:
            tables[(row["stage"], row["partition"])].append((row["batch_size"], row["context_tokens"], row["seconds"]))
            sm_pairs[row["partition"]] = (row["prefill_sms"], row["decode_sms"])
        pp_size = int(run["settings"]["command_settings"]["--pp-size"] or 2)
        for partition, (p_sm, d_sm) in sorted(sm_pairs.items()):
            for batch in batches:
                for context in contexts:
                    raw = [envelope(tables[(stage, partition)], batch, context) for stage in range(pp_size)]
                    covered = all(x is not None for x in raw)
                    scaled = [x * 1.2 + .001 for x in raw] if covered else []
                    per_run_predictions.append({
                        "group": run["group"], "run": run["name"], "partition": partition,
                        "prefill_sms": p_sm, "decode_sms": d_sm, "batch_size": batch,
                        "context_tokens": context, "covered": covered,
                        "stage_envelope_seconds": raw,
                        "stage_initial_scaled_seconds": scaled,
                        "sum_stage_initial_seconds": sum(scaled) if covered else None,
                        "both_stages_initial_gate_seconds": pp_size * max(scaled) if covered else None,
                        "mixed_candidate": p_sm > 0 and d_sm > 0,
                    })
    prediction_cells = defaultdict(list)
    for row in per_run_predictions:
        prediction_cells[(row["group"], row["partition"], row["prefill_sms"], row["decode_sms"], row["batch_size"], row["context_tokens"])].append(row)
    predicted = []
    for key, values in sorted(prediction_cells.items()):
        covered = [row for row in values if row["covered"]]
        result = {"group": key[0], "partition": key[1], "prefill_sms": key[2], "decode_sms": key[3],
                  "batch_size": key[4], "context_tokens": key[5], "startup_count": len(values),
                  "expected_startups": startup_counts[key[0]],
                  "covered_startups": len(covered), "mixed_candidate": key[2] > 0 and key[3] > 0}
        for metric in ("sum_stage_initial_seconds", "both_stages_initial_gate_seconds"):
            result[metric] = summary(row[metric] for row in covered) if covered else None
        predicted.append(result)
    grids = defaultdict(list)
    for row in predicted:
        grids[(row["group"], row["batch_size"], row["context_tokens"])].append(row)
    safety = []
    for (group, batch, context), rows in sorted(grids.items()):
        for target in targets_ms:
            mixed = [r for r in rows if r["mixed_candidate"] and r["covered_startups"] == r["expected_startups"] and r["covered_startups"]]
            robust = [r for r in mixed if r["both_stages_initial_gate_seconds"]["max"] <= target / 1000]
            median_safe = [r for r in mixed if r["both_stages_initial_gate_seconds"]["median"] <= target / 1000]
            chosen = max(robust, key=lambda r: (r["prefill_sms"], -r["partition"]), default=None)
            safety.append({"group": group, "batch_size": batch, "context_tokens": context, "target_ms": target,
                           "both_stages_safe_all_startups": [r["partition"] for r in robust],
                           "both_stages_safe_median": [r["partition"] for r in median_safe],
                           "max_prefill_safe_partition_all_startups": chosen["partition"] if chosen else None,
                           "note": "Initial scale only; runtime safety sets can shrink as contention correction grows."})
    return measured, per_run_predictions, predicted, safety


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, separators=(",", ":")) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def fallback_candidates(runs, safety):
    """Describe a batch-only fallback for the explicit full-context/50ms envelope."""
    groups = {run["group"]: run["settings"] for run in runs}
    candidates = []
    for group, settings in groups.items():
        limit = int(settings["command_settings"].get("--max-micro-batch-size") or 16)
        context = int(settings["command_settings"].get("--context-length") or 8192)
        rows = [row for row in safety if row["group"] == group and row["context_tokens"] == context and row["target_ms"] == 50]
        choices = []
        for batch in range(1, limit + 1):
            eligible = [row for row in rows if row["batch_size"] >= batch]
            corner = min(eligible, key=lambda row: row["batch_size"], default=None)
            choices.append({"batch_size": batch, "upper_grid_batch": corner["batch_size"] if corner else None,
                            "partition": corner["max_prefill_safe_partition_all_startups"] if corner else None})
        chosen = [row["partition"] for row in choices]
        representable = all(p is not None for p in chosen) and chosen == sorted(chosen)
        divisions = []
        if representable:
            for index, pair in enumerate(settings["sm_pairs"], start=1):
                threshold = min((row["batch_size"] for row in choices if row["partition"] >= index), default=limit + 1)
                divisions.append([*pair, threshold])
        candidates.append({"group": group, "target_ms": 50, "context_envelope": context,
                           "microbatch_limit": limit, "selection_by_batch": choices,
                           "representable_by_manual_thresholds": representable,
                           "manual_divisions_candidate": divisions,
                           "qualified_as": "Decode-only initial-gate fallback candidate; requires concurrent-workload validation, not a throughput optimum."})
    return candidates


def proposal(report):
    lines = ["# A100 decode calibration interpretation", "",
             f"Included startups: {len(report['runs'])}; measured coordinates: {len(report['measurements'])}.", "",
             "The fixed candidate SM partitions were verified against CUDA driver allocations on both 108-SM A100 GPUs. "
             "Manual thresholds use the maximum decode microbatch size, not total active requests. "
             "At microbatch limit 16, the original candidate threshold 24 is unreachable by the static policy.", "",
             "Each table cell measures the slower of uniform and maximally ragged sequence shapes; each shape averages three graph replays. "
             "ALP rounds batch/context upward and uses a monotonic upper envelope. Its initial per-stage estimate is envelope × 1.2 + 1 ms, "
             "and its gate is PP size × that estimate. The report's conservative both-stage gate is 2 × max(stage estimates); "
             "the stage sum is a separate diagnostic, not the implemented decision rule.", "",
             "The online scale starts at 1.2 and can increase to the 90th percentile observed/measured ratio, recalculated every 16 observations. "
             "Isolated startup profiles therefore do not guarantee concurrent request TPOT or establish an end-to-end speedup.", "",
             "## Context-aware candidate safety", "",
             "Rows below choose the most prefill SMs among partitions satisfying the initial gate on both stages in every included startup. "
             "They describe the default `latency_partition=True` preference. Fitted prefill throughput and required arrival rate affect partition selection only when that option is disabled; prefill RLS still controls admission budgets.", "",
             "| Group | Max microbatch | Mean context | TPOT target (ms) | Initially safe mixed partitions | Most prefill-safe |",
             "|---|---:|---:|---:|---|---|" ]
    for row in report["safety_sets"]:
        if row["context_tokens"] not in (512, 2048, 8192):
            continue
        lines.append(f"| {row['group']} | {row['batch_size']} | {row['context_tokens']} | {row['target_ms']} | {row['both_stages_safe_all_startups']} | {row['max_prefill_safe_partition_all_startups']} |")
    lines += ["", "## Static-threshold recommendation", ""]
    for candidate in report["fallback_candidates"]:
        lines += [f"Group {candidate['group']}: for mean context ≤ {candidate['context_envelope']}, "
                  f"microbatch ≤ {candidate['microbatch_limit']}, TPOT target 50 ms, "
                  f"the initial decode-only envelope yields `{candidate['manual_divisions_candidate']}`.", ""]
    lines += [
              "Do not encode a universal optimum from these decode-only measurements. A static batch-only threshold cannot account for context length, "
              "and these measurements contain no concurrent prefill throughput. Keep all six verified mixed partitions available to calibrated ALP. "
              "Choose static thresholds only after stating the context envelope and TPOT target, and then validate the full request workload. "
              "The safety CSV exposes empty safety sets rather than claiming infeasible targets are met.", "",
              "Across-startup CV is descriptive population variation, not a confidence interval. Different runtime/model/SM signatures remain in separate groups. "
              "Repeated trials under one server startup do not add independent calibration observations.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("run/a100"))
    parser.add_argument("--model-name", default="Llama-3.1-8B-Instruct")
    parser.add_argument("--runs", nargs="+", help="Optional fixed startup names; prevents tuning on later evaluation runs")
    parser.add_argument("--output", type=Path, default=Path("run/a100/calibration-analysis.json"))
    args = parser.parse_args()
    paths = [args.root / name / "scheduler.json" for name in args.runs] if args.runs else args.root.glob("*/scheduler.json")
    runs, excluded, issues = read_runs(paths, args.model_name)
    measured, individual, predicted, safety = aggregate(runs)
    report = {"schema_version": 1, "model_name": args.model_name,
              "runs": [{k: v for k, v in run.items() if k != "rows"} for run in runs],
              "excluded": excluded, "issues": issues, "measurements": measured,
              "per_startup_predictions": individual, "predictions": predicted, "safety_sets": safety,
              "fallback_candidates": fallback_candidates(runs, safety),
              "interpretation": {"initial_scale": 1.2, "sampling_allowance_seconds": .001,
                                 "raw_graph_cell": "max of uniform and maximally ragged shapes; 3 replays per shape",
                                 "lookup": "upper batch/context corner then monotonic max over lower coordinates",
                                 "runtime_gate": "pp_size * (table_envelope * online_scale + 0.001) <= TPOT * coefficient",
                                 "both_stages_gate": "pp_size * max(stage table envelopes * 1.2 + 0.001)",
                                 "limitations": ["No concurrent prefill during calibration", "Runtime online scale may exceed1.2",
                                                 "Measurements do not establish request SLO compliance", "CV across startups is descriptive, not a confidence interval"]}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    write_csv(args.output.with_suffix(".csv"), measured)
    write_csv(args.output.with_name(args.output.stem + "-safety.csv"), safety)
    args.output.with_suffix(".md").write_text(proposal(report))
    print(f"Included {len(runs)} startups, {len(measured)} coordinates; {len(issues)} issues; wrote {args.output}")
    if not runs:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
