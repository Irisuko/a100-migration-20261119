#!/usr/bin/env python3
"""Export A100 experiment tables and optional plots without running GPU work.

Exit zero means the export succeeded, not that its experiments passed. Inspect
issues.json. Mechanism counters cover the server lifetime, including warmup.
"""

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import statistics


METRICS = ("ttft_s", "tpot_s", "e2e_s", "arrival_ttft_s", "arrival_e2e_s", "queue_delay_s")
COUNTER_SCOPE = "server lifetime including startup/warmup/all trials/idle; never a per-trial delta"
STREAM_GAP_SCOPE = (
    "Client-observed adjacent token SSE receipt gaps within each request, pooled across the trial; "
    "includes transport/buffering effects, not server token execution latency or per-request mean TPOT. "
    "Quantiles require every request complete with every generated event containing exactly one token; "
    "zero-token metadata events are ignored; no token timestamps are interpolated."
)
BASE_COLUMNS = ["run", "classification", "experiment_purpose", "model", "mode", "policy", "chunk_tokens",
                "slice_token_layer_budget", "prefetch", "async_decode", "profile", "run_passed",
                "main_protocol_whitelisted", "superseded"]
TRIAL_COLUMNS = BASE_COLUMNS + ["suite", "repetition", "client_repeat_index", "trial_id",
                               "requested", "completed", "failed", "actual_output_tokens",
                               "output_tokens_per_s", "requests_per_s", "request_rate", "concurrency",
                               "client_stream_token_gap_s_p50", "client_stream_token_gap_s_p95",
                               "client_stream_token_gap_s_p99", "client_stream_token_gap_status",
                               "client_stream_token_gap_count", "client_stream_token_gap_expected_count",
                               "client_stream_token_gap_requests", "client_stream_token_gap_covered_requests",
                               "client_stream_generated_event_count", "client_stream_batched_event_count",
                               "client_stream_event_output_tokens", "client_stream_token_gap_unavailable_reasons",
                               "client_stream_token_gap_scope"]
CALIBRATION_COLUMNS = BASE_COLUMNS + ["stage", "partition", "prefill_sms", "decode_sms",
                                     "batch_size", "context_tokens", "seconds", "valid", "source"]


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def number(value, default=None):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def percentile(values, percent):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * percent / 100
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def flags(command):
    tokens = shlex.split(command) if isinstance(command, str) else list(command or [])
    result, index = {}, 0
    while index < len(tokens):
        token = str(tokens[index])
        if token.startswith("--"):
            if "=" in token:
                key, value = token.split("=", 1)
                result[key] = value
            else:
                values = []
                while index + 1 < len(tokens) and not str(tokens[index + 1]).startswith("--"):
                    index += 1
                    values.append(tokens[index])
                result[token] = True if not values else values[0] if len(values) == 1 else values
        index += 1
    return result


def issue(issues, code, source, run=None, severity="warning", **detail):
    issues.append({"code": code, "severity": severity, "run": run,
                   "source": str(source), **detail})


def load_json(path, issues, run=None, required=False):
    if not path.exists():
        if required:
            issue(issues, "missing_file", path, run)
        return None
    try:
        value = json.loads(path.read_text(), parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"nonfinite JSON constant {token}")))
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value
    except (OSError, ValueError) as error:
        issue(issues, "unreadable_json", path, run, error=str(error))
        return None


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, rows, columns):
    fields = list(dict.fromkeys(columns + sorted({key for row in rows for key in row})))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                             if isinstance(value, (dict, list)) else value for key, value in row.items()})


def classification(run, model):
    lower = str(model).lower()
    if "preflight" in run.lower() or "tinyllama" in lower:
        return "preflight"
    return "main8b" if "llama" in lower and "8b" in lower else "other_model"


def metadata(run_dir, launch, outcome, scheduler, issues, current_main_runs=None):
    arguments = flags(launch.get("command"))
    config = launch.get("config") or {}
    model = launch.get("model") or arguments.get("--model-path", "unknown")
    profile_artifacts = [str(path) for path in run_dir.glob("profile-*")]
    explicit_profile = launch.get("profile", outcome.get("profile"))
    profile = bool(explicit_profile) or bool(profile_artifacts)
    if explicit_profile is None and not profile_artifacts:
        issue(issues, "profile_flag_not_recorded", run_dir / "launch.json", run_dir.name,
              severity="info", detail="No profile artifacts found; timing exclusion cannot be certified.")
    mode = launch.get("mode") or ("mux" if arguments.get("--enable-pdmux") else "baseline")
    base = {
        "run": run_dir.name, "classification": classification(run_dir.name, model),
        "experiment_purpose": launch.get("experiment_purpose"),
        "main_protocol_whitelisted": (run_dir.name in current_main_runs
                                      if current_main_runs is not None else None),
        "superseded": (current_main_runs is not None and launch.get("experiment_purpose") == "main"
                       and run_dir.name not in current_main_runs),
        "model": model, "mode": mode,
        "policy": arguments.get("--pdmux-pp-policy", "alp") if mode == "mux" else "ordinary",
        "chunk_tokens": number(arguments.get("--pdmux-pp-prefill-chunk-size", 0)) if mode == "mux"
                        else number(arguments.get("--chunked-prefill-size")),
        "slice_token_layer_budget": config.get("split_forward_token_budget"),
        "prefetch": config.get("pp_prefetch"), "async_decode": config.get("pp_async_decode"),
        "profile": profile, "profile_metadata_recorded": explicit_profile is not None,
        "profile_artifacts": profile_artifacts, "run_passed": outcome.get("passed") is True,
        "context_limit": number(arguments.get("--context-length")),
        "kv_capacity": number(arguments.get("--max-total-tokens")),
        "max_running_requests": number(arguments.get("--max-running-requests")),
        "decode_microbatch": number(arguments.get("--max-micro-batch-size")),
        "graph_max_batch": number(arguments.get("--cuda-graph-max-bs")),
        "pp_size": number(arguments.get("--pp-size", 1)), "tp_size": number(arguments.get("--tp-size", 1)),
        "dtype": arguments.get("--dtype"), "attention_backend": arguments.get("--attention-backend"),
        "configured_ttft_s": number(arguments.get("--pdmux-pp-slo-ttft"), 2.0),
        "configured_tpot_s": number(arguments.get("--pdmux-pp-slo-tpot"), 0.05),
        "main_ttft_s": 2.0, "main_tpot_s": 0.05,
        "repository_commit": launch.get("repository_commit"),
        "launch_source": str(run_dir / "launch.json"), "outcome_source": str(run_dir / "outcome.json"),
        "scheduler_source": str(run_dir / "scheduler.json"),
        "server_log_source": scheduler.get("source_log", str(run_dir / "server.log")),
        "mechanism_scope": COUNTER_SCOPE,
    }
    return base, arguments, config


def coverage(base, arguments, config, scheduler, issues):
    rows = scheduler.get("calibration_rows", [])
    observed = {(r.get("stage"), r.get("partition"), r.get("batch_size"), r.get("context_tokens"))
                for r in rows if r.get("valid") is True}
    expected_enabled = base["mode"] == "mux" and config.get("pp_profile_decode") is True
    expected = set()
    contexts, batches, partitions = [], [], []
    if expected_enabled and base["context_limit"] and base["graph_max_batch"]:
        maximum = int(base["context_limit"])
        contexts = sorted({min(int(c), maximum) for c in
                           (2048, 4096, 8192, 16384, 33792, *config.get("pp_decode_extra_contexts", []))})
        graph_max = int(base["graph_max_batch"])
        explicit = arguments.get("--cuda-graph-bs")
        captures = ([int(b) for b in (explicit if isinstance(explicit, list) else [explicit])]
                    if explicit is not None else [1, *[b for b in (4, 8, 16) if b <= graph_max], graph_max])
        batches = sorted({min(captures), max(captures), *[b for b in captures if b in (4, 8, 16)]})
        partitions = list(range(1, int(config.get("sm_group_num", 0))))
        expected = {(stage, partition, batch, context)
                    for stage in range(int(base["pp_size"])) for partition in partitions
                    for batch in batches for context in contexts}
    missing = sorted(expected - observed)
    invalid = sum(row.get("valid") is not True for row in rows)
    result = {
        "calibration_enabled": expected_enabled, "calibration_rows": len(rows),
        "calibration_valid_unique_cells": len(observed), "calibration_expected_cells": len(expected),
        "calibration_missing_cells": len(missing), "calibration_invalid_rows": invalid,
        "calibration_launch_grid_complete": (not missing and not invalid) if expected else None,
        "calibration_expected_batches": batches, "calibration_expected_contexts": contexts,
        "calibration_expected_partitions": partitions,
        "calibration_grid_basis": "launch context/PP/SM counts and default or explicit CUDA graph batch grid",
        "calibration_observed_grid_coverage": scheduler.get("calibration_coverage", {}),
    }
    if expected_enabled and (not expected or missing or invalid):
        issue(issues, "incomplete_calibration_launch_grid", base["scheduler_source"], base["run"],
              expected_cells=len(expected), missing_cells=len(missing), first_missing=missing[:8], invalid_rows=invalid)
    if expected_enabled and batches and contexts and base["kv_capacity"] is not None:
        if base["kv_capacity"] <= max(batches) * max(contexts):
            issue(issues, "calibration_kv_mapping_may_wrap", base["launch_source"], base["run"],
                  capacity=base["kv_capacity"], required_strictly_above=max(batches) * max(contexts))
    return result


def mechanisms(base, scheduler):
    rows, trial_fields = [], {}
    migrations = scheduler.get("migrations", {})
    trial_fields["cumulative_global_migrations_max"] = migrations.get("max")
    for stage_id, stage in sorted(scheduler.get("stages", {}).items()):
        latest = stage.get("latest_telemetry") or {}
        row = {**base, "stage": stage_id, "cumulative_global_migrations_max": migrations.get("max"),
               "global_migration_resets": migrations.get("resets", []),
               "summary_count": stage.get("summary_count"),
               "partition_transitions_observed_lower_bound": stage.get("partition_transitions_observed"),
               "partition_seconds": stage.get("partition_seconds", {}),
               "idle_counter_resets": stage.get("idle_counter_resets", []),
               "slice_choices": latest.get("slice_choices"), "rls_observations": latest.get("samples"),
               "admission_budget_eval": latest.get("admission_budget_eval"),
               "source_line": stage.get("latest_summary_line")}
        admission = latest.get("admission") or {}
        for name in ("decisions", "blocked", "stops", "wall"):
            row["admission_" + name] = admission.get(name)
        for key, value in stage.get("idle_counter_max", {}).items():
            row["counter_max_" + key] = value
            trial_fields[f"cumulative_stage{stage_id}_{key}_max"] = value
        samples = latest.get("samples") or []
        if len(samples) >= 2:
            trial_fields[f"cumulative_stage{stage_id}_prefill_rls_observations"] = samples[1]
        rows.append(row)
    return rows, trial_fields


def token_count(record):
    return len(record["output_ids"]) if isinstance(record.get("output_ids"), list) else record.get("actual_output_tokens", 0)


def client_stream_gaps(records):
    """Only report trial-wide token gaps when complete event coverage permits it."""
    reasons, gaps = set(), []
    result = {
        "client_stream_token_gap_scope": STREAM_GAP_SCOPE,
        "client_stream_token_gap_requests": len(records),
        "client_stream_token_gap_covered_requests": 0,
        "client_stream_generated_event_count": 0,
        "client_stream_batched_event_count": 0,
        "client_stream_event_output_tokens": 0,
        "client_stream_token_gap_expected_count": 0,
    }
    if not records:
        reasons.add("missing_request_records")
    for record in records:
        request_reasons = set()
        if record.get("error") is not None:
            request_reasons.add("failed_requests")
        expected = token_count(record)
        if type(expected) is not int or expected <= 0:
            request_reasons.add("invalid_output_count")
        else:
            result["client_stream_token_gap_expected_count"] += max(0, expected - 1)
            if (record.get("actual_output_tokens") != expected
                    or record.get("requested_output_tokens") != expected):
                request_reasons.add("incomplete_output")
        events = record.get("events")
        times, observed_tokens = [], 0
        if not isinstance(events, list) or not events:
            request_reasons.add("missing_events")
            events = []
        for event in events:
            count = event.get("new_tokens") if isinstance(event, dict) else None
            if type(count) is not int or count < 0:
                request_reasons.add("invalid_event_token_count")
                continue
            if count == 0:
                continue
            result["client_stream_generated_event_count"] += 1
            result["client_stream_event_output_tokens"] += count
            observed_tokens += count
            if count != 1:
                result["client_stream_batched_event_count"] += 1
                request_reasons.add("batched_generated_events")
            elapsed = event.get("elapsed_s")
            if not finite(elapsed) or elapsed < 0 or (times and elapsed < times[-1]):
                request_reasons.add("invalid_event_timing")
            else:
                times.append(elapsed)
        if observed_tokens != expected:
            request_reasons.add("incomplete_token_events")
        if not request_reasons:
            result["client_stream_token_gap_covered_requests"] += 1
            gaps.extend(current - previous for previous, current in zip(times, times[1:]))
        reasons.update(request_reasons)
    if not reasons and not gaps:
        reasons.add("no_adjacent_token_gaps")
    available = not reasons
    result["client_stream_token_gap_status"] = "available" if available else "unavailable"
    result["client_stream_token_gap_unavailable_reasons"] = sorted(reasons)
    result["client_stream_token_gap_count"] = len(gaps) if available else None
    for p in (50, 95, 99):
        result[f"client_stream_token_gap_s_p{p}"] = percentile(gaps, p) if available else None
    return result


def trial_row(base, client, source, outer_repeat, trial, records, issues):
    aggregate = trial.get("aggregate", client.get("aggregate", {}))
    inner_repeat = trial.get("repeat_index", 0)
    row = {**base, "suite": client.get("suite"), "repetition": outer_repeat,
           "client_repeat_index": inner_repeat, "trial_id": f"{base['run']}/{source.stem}/{inner_repeat}",
           "fixture_sha256": client.get("fixture_sha256"), "fixtures_source": client.get("fixtures"),
           "order_mode": client.get("order_mode", "fixture"), "seed": client.get("seed"),
           "arrival_seed": trial.get("arrival_seed", client.get("seed")),
           "request_rate": client.get("request_rate", 0), "concurrency": client.get("concurrency"),
           "client_source": str(source), "wall_s": aggregate.get("wall_s"),
           "client_passed": client.get("passed") is True,
           "arrival_mode": client.get("experiment_slo", {}).get("arrival_mode"),
           "cache_reset_in_client": client.get("cache_reset"),
           "cache_reset_note": "runner may flush before invocation; inspect outcome command/orchestration",
           "counts_basis": "raw per-request records" if records else "stored aggregate; raw records unavailable"}
    row.update(client_stream_gaps(records))
    experiment_slo = client.get("experiment_slo", {})
    for metric in ("ttft", "tpot"):
        row[f"configured_{metric}_s"] = experiment_slo.get(f"{metric}_target_s", base[f"configured_{metric}_s"])
    if records:
        successful = [record for record in records if record.get("error") is None]
        row.update(requested=len(records), completed=len(successful), failed=len(records) - len(successful),
                   actual_output_tokens=sum(token_count(record) for record in successful),
                   all_observed_output_tokens_including_errors=sum(token_count(record) for record in records),
                   cached_tokens=sum(record.get("cached_tokens", 0) or 0 for record in successful),
                   cache_hit_requests=sum((record.get("cached_tokens", 0) or 0) > 0 for record in successful))
        for record in successful:
            if token_count(record) != record.get("requested_output_tokens"):
                issue(issues, "unexpected_output_length", source, base["run"], request_id=record.get("request_id"),
                      actual=token_count(record), requested=record.get("requested_output_tokens"))
        for metric in METRICS:
            values = [record[metric] for record in successful if finite(record.get(metric))]
            for p in (50, 95, 99):
                row[f"{metric}_p{p}"] = percentile(values, p)
        for label in ("main", "configured"):
            ttft, tpot = row[f"{label}_ttft_s"], row[f"{label}_tpot_s"]
            for arrival, prefix in ((False, ""), (True, "arrival_")):
                tkey = "arrival_ttft_s" if arrival else "ttft_s"
                qualifies = lambda r, key, target: finite(r.get(key)) and r[key] <= target
                row[f"{label}_{prefix}ttft_met_fraction"] = sum(qualifies(r, tkey, ttft) for r in successful) / len(records)
                row[f"{label}_{prefix}both_met_fraction"] = sum(
                    qualifies(r, tkey, ttft) and qualifies(r, "tpot_s", tpot) for r in successful) / len(records)
            row[f"{label}_tpot_met_fraction"] = sum(
                finite(r.get("tpot_s")) and r["tpot_s"] <= tpot for r in successful) / len(records)
    else:
        issue(issues, "missing_trial_request_records", source, base["run"], client_repeat_index=inner_repeat)
        for key in ("requested", "completed", "failed", "actual_output_tokens", "cached_tokens", "cache_hit_requests"):
            row[key] = aggregate.get(key)
        for metric in METRICS:
            for p in (50, 95, 99):
                row[f"{metric}_p{p}"] = aggregate.get(metric, {}).get(f"p{p}")
    wall = number(row["wall_s"])
    for key, count in (("output_tokens_per_s", "actual_output_tokens"), ("requests_per_s", "completed")):
        row[key] = row[count] / wall if wall and finite(row.get(count)) else None
    row["performance_plot_eligible"] = (
        row["suite"] == "perf" and row["classification"] == "main8b" and row["run_complete"]
        and row.get("experiment_purpose") in (None, "main")
        and row.get("main_protocol_whitelisted") is not False
        and row["run_passed"] and row["client_passed"] and row.get("failed") == 0
        and not row["profile"] and row["profile_metadata_recorded"])
    if row.get("failed") or not row["client_passed"]:
        issue(issues, "client_trial_failed", source, base["run"], suite=row["suite"],
              repetition=outer_repeat, client_repeat_index=inner_repeat, failed=row.get("failed"))
    return row


def collect(root, output_dir):
    issues, runs, trials, mechanism_rows, calibration = [], [], [], [], []
    protocol_path = root / "main-protocol.json"
    current_main_runs, protocol_digest = None, None
    if protocol_path.exists():
        protocol = load_json(protocol_path, issues)
        selected = protocol.get("main_runs") if protocol else None
        if isinstance(selected, list) and selected and all(isinstance(name, str) for name in selected):
            current_main_runs = set(selected)
        else:
            # An unusable existing protocol must not silently allow superseded runs.
            current_main_runs = set()
            issue(issues, "invalid_current_main_protocol", protocol_path,
                  detail="No usable main_runs whitelist; all performance table/plot rows excluded.")
        protocol_digest = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    directories = {path.parent for marker in ("launch.json", "outcome.json", "scheduler.json")
                   for path in root.glob("*/" + marker) if output_dir not in path.parents}
    for run_dir in sorted(directories):
        launch = load_json(run_dir / "launch.json", issues, run_dir.name, True)
        outcome = load_json(run_dir / "outcome.json", issues, run_dir.name, True)
        scheduler = load_json(run_dir / "scheduler.json", issues, run_dir.name, True)
        complete = all(item is not None for item in (launch, outcome, scheduler))
        if launch is None:
            issue(issues, "run_has_no_usable_launch", run_dir, run_dir.name)
            continue
        base, arguments, config = metadata(run_dir, launch, outcome or {}, scheduler or {}, issues,
                                           current_main_runs=current_main_runs)
        base.update(main_protocol_source=str(protocol_path) if protocol_digest else None,
                    main_protocol_sha256=protocol_digest)
        base["run_complete"] = complete
        base.update(coverage(base, arguments, config, scheduler or {}, issues))
        mechanism, counter_fields = mechanisms(base, scheduler or {})
        mechanism_rows.extend(mechanism)
        base.update(counter_fields)
        runs.append({**base, "config": config, "command": launch.get("command")})
        if not complete or not base["run_passed"]:
            issue(issues, "incomplete_or_failed_run", run_dir / "outcome.json", run_dir.name,
                  complete=complete, error=(outcome or {}).get("error"), cleanup_error=(outcome or {}).get("cleanup_error"))
        scheduler_metadata = (scheduler or {}).get("metadata", {})
        for name in ("parse_errors", "issues", "inactivity"):
            entries = scheduler_metadata.get(name, [])
            if entries:
                issue(issues, "scheduler_" + name, run_dir / "scheduler.json", run_dir.name,
                      severity="info" if name == "inactivity" or base["mode"] == "baseline" else "warning",
                      count=len(entries), entries=entries[:10])
        for row in (scheduler or {}).get("calibration_rows", []):
            calibration_base = {key: base.get(key) for key in BASE_COLUMNS + [
                "dtype", "attention_backend", "pp_size", "tp_size", "context_limit",
                "graph_max_batch", "repository_commit", "scheduler_source", "run_complete",
                "profile_metadata_recorded"]}
            calibration.append({**calibration_base, **row, "source": base["server_log_source"],
                                "calibration_statistic": "maximum uniform/skewed context timing; each timing averages 3 graph replays"})
        recorded = {Path(item.get("output", "")).name: item for item in (outcome or {}).get("runs", [])}
        for filename, item in recorded.items():
            if filename and not (run_dir / filename).exists():
                issue(issues, "missing_recorded_client_output", run_dir / filename, run_dir.name,
                      suite=item.get("suite"), exit_code=item.get("exit_code"))
        for source in sorted(run_dir.glob("*.json")):
            if source.name in {"launch.json", "outcome.json", "scheduler.json", "shutdown.json"}:
                continue
            client = load_json(source, issues, run_dir.name)
            if client is None or not isinstance(client.get("aggregate"), dict) or not client.get("suite"):
                continue
            match = re.search(r"-(\d+)$", source.stem)
            outer_repeat = recorded.get(source.name, {}).get("repeat", int(match.group(1)) if match else 0)
            records = client.get("requests", [])
            client_trials = client.get("trials") or [{"repeat_index": 0, "aggregate": client["aggregate"]}]
            for trial in client_trials:
                selected = [record for record in records if record.get("repeat_index", 0) == trial.get("repeat_index", 0)]
                trials.append(trial_row(base, client, source, outer_repeat, trial, selected, issues))
    return runs, trials, mechanism_rows, calibration, issues


def slug(text):
    name = re.sub(r"[^a-zA-Z0-9._-]+", "-", text).strip("-")[:90]
    return name + "-" + hashlib.sha256(text.encode()).hexdigest()[:8]


def plots(output_dir, trials, calibration, issues):
    artifacts = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        issue(issues, "optional_matplotlib_unavailable", output_dir, severity="info", error=str(error))
        return artifacts
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    groups = defaultdict(list)
    for row in calibration:
        if row.get("valid") and finite(row.get("seconds")):
            groups[(row["classification"], row["model"], row["run"], row["stage"])].append(row)
    for (kind, model, run, stage), rows in groups.items():
        batches = sorted({row["batch_size"] for row in rows})
        figure, axes = plt.subplots(math.ceil(len(batches) / 2), 2,
                                    figsize=(11, 3.5 * math.ceil(len(batches) / 2)), squeeze=False)
        for ax, batch in zip(axes.flat, batches):
            partitions = sorted({row["partition"] for row in rows if row["batch_size"] == batch})
            for partition in partitions:
                selected = [row for row in rows if row["batch_size"] == batch and row["partition"] == partition]
                by_context = defaultdict(list)
                for row in selected:
                    by_context[row["context_tokens"]].append(row["seconds"] * 1000)
                xs = sorted(by_context)
                line, = ax.plot(xs, [statistics.median(by_context[x]) for x in xs],
                                label=f"decode SMs {selected[0]['decode_sms']}", linewidth=1)
                ax.scatter([row["context_tokens"] for row in selected], [row["seconds"] * 1000 for row in selected],
                           color=line.get_color(), s=12, alpha=0.65)
                ax.fill_between(xs, [min(by_context[x]) for x in xs], [max(by_context[x]) for x in xs],
                                color=line.get_color(), alpha=0.12)
            ax.set(title=f"Decode batch {batch}", xlabel="Context tokens", ylabel="Decode graph time (ms)")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7)
        for ax in list(axes.flat)[len(batches):]:
            ax.set_visible(False)
        figure.suptitle(f"{Path(model).name} | {run} | PP stage {stage}\nRetained samples; lines: medians; bands: observed ranges", fontsize=11)
        figure.tight_layout(rect=(0, 0, 1, 0.94))
        path = plot_dir / (f"{kind}-calibration-" + slug(f"{model}-{run}-stage{stage}") + ".png")
        figure.savefig(path, dpi=150)
        plt.close(figure)
        artifacts.append(str(path))
    groups.clear()
    for row in trials:
        if row.get("performance_plot_eligible"):
            key = (row["model"], row["request_rate"], row["concurrency"], row["fixture_sha256"], row["order_mode"])
            groups[key].append(row)
    for key, rows in groups.items():
        model, rate, concurrency, fixture_hash, order = key
        labels = sorted({row["run"] for row in rows})
        figure, axes = plt.subplots(2, 2, figsize=(13, max(7, 0.65 * len(labels) + 3)))
        metrics = [("output_tokens_per_s", "Output tokens/s", None), ("ttft_s_p95", "TTFT p95 (s)", 2),
                   ("tpot_s_p95", "TPOT p95 (s)", 0.05), ("e2e_s_p95", "End-to-end p95 (s)", None)]
        for ax, (metric, title, target) in zip(axes.flat, metrics):
            for index, label in enumerate(labels):
                values = [row[metric] for row in rows if row["run"] == label and finite(row.get(metric))]
                if values:
                    ax.hlines(index, min(values), max(values), color="tab:blue", alpha=0.5)
                    offsets = [0.12 * (i - (len(values) - 1) / 2) / max(1, len(values) / 3) for i in range(len(values))]
                    ax.scatter(values, [index + offset for offset in offsets], s=25, color="tab:blue")
            if target is not None:
                ax.axvline(target, color="tab:red", linestyle="--", alpha=0.6, label="Main SLO target")
            ax.set_yticks(range(len(labels)), labels)
            ax.set_title(title)
            ax.grid(axis="x", alpha=0.2)
            ax.invert_yaxis()
        figure.suptitle(f"{Path(model).name} | arrival rate {rate} req/s (0 = saturated) | concurrency {concurrency}\n"
                       f"Order {order}; each dot is one trial; bars are observed min–max ranges", fontsize=11)
        figure.tight_layout(rect=(0, 0, 1, 0.94))
        path = plot_dir / ("performance-" + slug(json.dumps(key)) + ".png")
        figure.savefig(path, dpi=150)
        plt.close(figure)
        artifacts.append(str(path))
    return artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("run/a100"))
    parser.add_argument("--output-dir", type=Path, default=Path("run/a100/report-data"))
    parser.add_argument("--no-plots", action="store_true", help="Write tables only")
    args = parser.parse_args()
    root, output_dir = args.root.resolve(), args.output_dir.resolve()
    if not root.is_dir():
        parser.error(f"root directory does not exist: {root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    runs, trials, mechanisms_data, calibration, issues = collect(root, output_dir)
    artifacts = []
    for kind, prefix in (("main8b", ""), ("preflight", "preflight-"), ("other_model", "other-model-")):
        for name, rows, columns in (("runs", runs, BASE_COLUMNS), ("trials", trials, TRIAL_COLUMNS),
                                    ("mechanisms", mechanisms_data, BASE_COLUMNS + ["stage"]),
                                    ("calibration", calibration, CALIBRATION_COLUMNS)):
            path = output_dir / (prefix + name + ".csv")
            write_csv(path, [row for row in rows if row["classification"] == kind], columns)
            artifacts.append(str(path))
    performance_path = output_dir / "performance.csv"
    write_csv(performance_path, [row for row in trials if row.get("performance_plot_eligible")], TRIAL_COLUMNS)
    artifacts.append(str(performance_path))
    if not args.no_plots:
        artifacts.extend(plots(output_dir, trials, calibration, issues))
    write_json(output_dir / "issues.json", {"issues": issues, "count": len(issues)})
    summary = {
        "schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(), "root": str(root),
        "run_counts": {kind: sum(row["classification"] == kind for row in runs)
                       for kind in ("main8b", "preflight", "other_model")},
        "trial_count": len(trials), "calibration_row_count": len(calibration), "issue_count": len(issues),
        "performance_csv_filter": "main8b, suite=perf, experiment_purpose=main or legacy missing/null, current main-protocol.main_runs whitelist when present, completed/passed run and client, zero errors, explicit profile=false; actual arrival rate and concurrency come from each client file",
        "artifacts": artifacts, "issues_source": str(output_dir / "issues.json"),
        "preflight_metadata": [row for row in runs if row["classification"] == "preflight"],
        "notes": [COUNTER_SCOPE,
                  STREAM_GAP_SCOPE,
                  "All SLO fractions use all requests as denominator; errors count as misses.",
                  "Latency quantiles use successful requests; arrival metrics include client queue delay.",
                  "Performance tables/plots exclude explicit non-main purposes (mechanism/diagnostic/correctness/pilot), diagnostic profiles, unknown profile metadata, failed/incomplete runs, and non-perf suites; full trial/mechanism tables retain them.",
                  "When main-protocol.json exists, only its main_runs enter performance tables/plots; earlier explicit main runs are marked superseded and retained in full tables. An invalid existing whitelist excludes all performance rows.",
                  "Calibration rows are retained separately by run/model/stage, including repeated measurements.",
                  "Calibration seconds represent the maximum uniform/skewed-context timing, each averaged over 3 graph replays.",
                  "Plots show individual trial values and observed ranges; no statistical significance is claimed."],
    }
    write_json(output_dir / "metadata.json", summary)
    print(json.dumps({"output_dir": str(output_dir), "run_counts": summary["run_counts"],
                      "trials": len(trials), "calibration_rows": len(calibration), "issues": len(issues)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
