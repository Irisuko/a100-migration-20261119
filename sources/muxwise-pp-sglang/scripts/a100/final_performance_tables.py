#!/usr/bin/env python3
"""Strict v2 performance tables, reusing summarize_results statistics; CPU only."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

from summarize_results import finite, flags, metadata, trial_row

REPO = Path(__file__).resolve().parents[2]
OPEN_LOOP = {"v2-main-cpp", "v2-main-cpu-metadata", "v2-main-baseline-chunk2048"}
LABELS = {
    "v2-main-baseline-whole": "Baseline whole",
    "v2-main-baseline-chunk2048": "Baseline chunk 2048",
    "v2-main-static": "Static Mux",
    "v2-main-alp-whole": "ALP whole",
    "v2-main-cpp": "CPP 2048 / 32768",
    "v2-main-no-prefetch": "CPP without prefetch",
    "v2-main-no-async": "CPP without async decode",
    "v2-main-no-calibration": "CPP without startup calibration",
    "v2-main-cpu-metadata": "CPP CPU prefill metadata",
}
INTERPRETATIONS = {
    "partition_policy": "Static versus ALP policy: both partition selection and model-based admission change; this does not isolate partition selection alone.",
    "chunk_only": "Only token chunk cap changes (0 to 2048); the shared 32768 token-layer budget can imply different layer-slice lengths as a consequence.",
    "prefetch": "Prefetch path ablation includes receive-slot allocation, receive stream and READY send dispatch; attribute the result to that path as a whole.",
    "async_decode": "Asynchronous decode transport path versus legacy PP ring; it is not a pure kernel-overlap-only comparison.",
    "startup_calibration": "After common warmup, calibrated decode prediction versus online-model fallback. Measured throughput excludes startup calibration time.",
    "cpu_prefill_metadata": "Only CPU-known prefill metadata flag changes; it can remove metadata-triggered synchronization.",
    "baseline_chunk_only": "Ordinary baseline whole versus 2048-token chunking; compare the first three saturation trials.",
}
METRICS = ("output_tokens_per_s", "ttft_s_p50", "ttft_s_p95", "ttft_s_p99",
           "tpot_s_p50", "tpot_s_p95", "tpot_s_p99", "e2e_s_p95",
           "arrival_ttft_s_p95", "arrival_ttft_s_p99", "arrival_e2e_s_p95",
           "queue_delay_s_p95", "main_both_met_fraction", "main_tpot_met_fraction",
           "main_arrival_both_met_fraction", "client_stream_token_gap_s_p50",
           "client_stream_token_gap_s_p95", "client_stream_token_gap_s_p99")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (
        _ for _ in ()).throw(ValueError(f"nonfinite JSON constant {value}")))


def validate_whitelist(protocol):
    names = protocol.get("main_runs", [])
    require(protocol.get("protocol_revision") == 2, "requires v2 main protocol")
    require(len(names) == len(set(names)) == 9 and all(
        isinstance(n, str) and n.startswith("v2-main-") and Path(n).name == n for n in names),
        "requires nine unique v2 main whitelist names")
    return names


def validate_client(client, fixture, fixture_hash, repetition, suite, client_sha):
    client_suite = "warmup" if suite == "warmup" else "perf"
    count = 24 if suite == "warmup" else 64
    require(client.get("passed") is True and client.get("suite") == client_suite, f"{suite} client failed")
    require(client.get("fixture_sha256") == fixture_hash, "fixture hash differs")
    require(client.get("client_source_sha256") == client_sha, "client source differs")
    require(client.get("repeat") == 1 and client.get("seed") == 20261114 + repetition,
            "unpaired repetition/seed")
    require((client.get("request_rate"), client.get("concurrency")) == (
        (8, 64) if suite == "perf-openloop" else (0, 1 if suite == "warmup" else 32)),
        "arrival rate/concurrency differs")
    expected = fixture["suites"][client_suite]
    require(client.get("order_mode") == "fixture" and client.get("request_order") == expected,
            "request order differs")
    require(client.get("cache_reset") is True and client.get("external_cache_reset", {}).get("http_status") == 200,
            "cache reset acknowledgement missing")
    require(client.get("experiment_slo", {}).get("ttft_target_s") == 2 and
            client.get("experiment_slo", {}).get("tpot_target_s") == .05, "SLO target differs")
    records = client.get("requests", [])
    require(len(records) == len(expected) == count, f"incomplete {count}-request trial")
    require([r["fixture_id"] for r in records] == expected, "request fixture order differs")
    require(len({r["request_id"] for r in records}) == count, "duplicate request IDs")
    for r in records:
        case = fixture["cases"][r["fixture_id"]]
        require(r.get("error") is None and r.get("sse_done") is True, "failed request or missing SSE completion")
        require(r.get("input_tokens") == case["input_tokens"], "input token count differs")
        require(r.get("requested_output_tokens") == r.get("actual_output_tokens") ==
                len(r.get("output_ids", [])) == case["max_new_tokens"], "truncated or changed output")
        for key in ("ttft_s", "tpot_s", "e2e_s", "arrival_ttft_s", "arrival_e2e_s", "queue_delay_s"):
            require(finite(r.get(key)) and r[key] >= 0, f"invalid request {key}")
        previous = 0
        require(bool(r.get("events")), "missing SSE events")
        for event in r["events"]:
            completion = event.get("server_completion_tokens")
            require(type(completion) is int and completion >= previous and
                    event.get("new_tokens") == completion - previous and
                    event.get("total_tokens") == completion and
                    event.get("raw_tokens", -1) - event.get("prompt_lookback_dropped", 0) == completion - previous,
                    "SSE completion delta lost or inconsistent")
            previous = completion
        require(previous == r["actual_output_tokens"], "SSE final completion count differs")
    aggregate = client["aggregate"]
    tokens = sum(r["actual_output_tokens"] for r in records)
    require(aggregate.get("completed") == aggregate.get("requested") == count and aggregate.get("failed") == 0,
            "incomplete aggregate")
    require(aggregate.get("actual_output_tokens") == tokens and finite(aggregate.get("wall_s")) and aggregate["wall_s"] > 0,
            "invalid aggregate count/wall time")
    require(math.isclose(aggregate["output_tokens_per_s"], tokens / aggregate["wall_s"], rel_tol=1e-12),
            "throughput aggregate differs")
    require(len(client.get("trials", [])) == 1 and client["trials"][0].get("repeat_index") == 0,
            "expected one client trial")
    require(client["trials"][0].get("aggregate") == aggregate, "inner/top aggregate mismatch")


def validate_launch_snapshot(launch, expected):
    actual = flags(launch.get("command", []))
    require(actual == expected["server_flags"], "actual server flags differ from frozen plan")
    require(launch.get("mode") == expected["mode"], "launch mode differs from plan")
    require(all(actual.get(k) == v for k, v in expected["shared_server_flags"].items()),
            "shared server flags differ from frozen plan")
    if expected["mode"] == "mux":
        require((launch.get("config") or {}) == expected["config"], "actual config snapshot differs from frozen plan")
        require(actual.get("--pdmux-pp-policy", "alp") == expected["policy"] and
                int(actual.get("--pdmux-pp-prefill-chunk-size", 0)) == expected["chunk_tokens"],
                "policy/chunk differs from frozen plan")
        require(bool(actual.get("--pdmux-pp-cpu-prefill-metadata", False)) == expected["cpu_prefill_metadata"],
                "CPU prefill metadata flag differs from frozen plan")
    else:
        require(int(actual.get("--chunked-prefill-size", -1)) == expected["baseline_chunk_tokens"],
                "baseline chunk differs from frozen plan")
    return {"matches_frozen_plan": True, "planned_config_sha256": expected.get("config_sha256"),
            "actual_config_value_sha256": hashlib.sha256(json.dumps(
                launch.get("config") or {}, sort_keys=True).encode()).hexdigest()}


def aggregate_trials(name, suite, rows):
    require(len(rows) == 3, "requires exactly three complete trials")
    result = {"run": name, "label": LABELS.get(name, name), "suite": suite, "repetitions": 3}
    for metric in METRICS:
        values = [row.get(metric) for row in rows]
        if not all(finite(v) for v in values):
            result[metric] = None
            continue
        result[metric] = {"median": statistics.median(values), "min": min(values), "max": max(values)}
        if metric.endswith("fraction"):
            result[metric]["mean"] = statistics.mean(values)
    return result


def build_report(root):
    protocol = load(root / "main-protocol.json")
    names = validate_whitelist(protocol)
    require(OPEN_LOOP <= set(names), "open-loop cohort missing from whitelist")
    review = load(root / "main-review.json")
    require(review.get("protocol_revision") == 2 and review.get("passed") is True and
            review.get("protocol_sha256") == digest(root / "main-protocol.json"), "protocol review absent or stale")
    expected_hashes = {p: row["expected"] for p, row in review["runtime_sources"].items()}
    require(len(expected_hashes) == 5, "requires frozen five-source review")
    require(set(review.get("plan_snapshots", {})) == set(names), "review must contain all nine frozen plan snapshots")
    fixture_path = root / "fixtures.json"
    fixture, fixture_hash = load(fixture_path), digest(fixture_path)
    client_sha = digest(REPO / "scripts/a100/experiment_client.py")
    rows, trials, statuses, sources = {"perf": [], "perf-openloop": []}, [], {}, []
    for path in (root / "main-protocol.json", root / "main-review.json", fixture_path):
        sources.append({"path": str(path.resolve()), "sha256": digest(path)})
    for name in names:
        directory = root / name
        required = [directory / p for p in ("launch.json", "outcome.json", "scheduler.json", "shutdown.json")]
        if required[1].exists():
            try:
                outcome = load(required[1])
                require(outcome.get("passed") is True, f"run failed: {outcome.get('error')}")
            except (OSError, ValueError) as error:
                statuses[name] = {"status": "excluded", "reason": str(error)}
                continue
        if (directory / "server.pid").exists() or not all(p.exists() for p in required):
            statuses[name] = {"status": "pending"}
            continue
        try:
            launch, outcome, scheduler, shutdown = map(load, required)
            require(outcome.get("passed") is True, "run did not pass")
            require(shutdown.get("remaining") == [] and shutdown.get("forced") is False, "server cleanup was incomplete or forced")
            require(launch.get("experiment_purpose") == "main" and all(
                launch.get(k) is False for k in ("profile", "trace", "plan_reuse_check")), "diagnostic instrumentation or wrong purpose")
            require(launch.get("runtime_source_sha256") == expected_hashes, "frozen runtime hashes differ")
            config_check = validate_launch_snapshot(launch, review["plan_snapshots"][name])
            require(Path(launch["model"]).resolve() == Path(fixture["model_path"]).resolve(), "model differs from fixture")
            history = [("warmup", 0), *(('perf', i) for i in range(3))]
            if name in OPEN_LOOP:
                history += [("perf-openloop", i) for i in range(3)]
            history += [("serial", 0), ("cache", 0), ("concurrent", 0)]
            require([(r["suite"], r["repeat"]) for r in outcome.get("runs", [])] == history, "actual suite/training history differs")
            require(all(r.get("exit_code") == 0 for r in outcome["runs"]), "client process failed")
            issues = []
            base, _, _ = metadata(directory, launch, outcome, scheduler, issues, set(names))
            base["run_complete"] = True
            staged, staged_trials, staged_sources = [], [], []
            warmup_path = directory / "warmup-0.json"
            warmup = load(warmup_path)
            validate_client(warmup, fixture, fixture_hash, 0, "warmup", client_sha)
            staged_sources.append({"path": str(warmup_path.resolve()), "sha256": digest(warmup_path)})
            request_ids = {r["request_id"] for r in warmup["requests"]}
            for suite in ("perf", "perf-openloop") if name in OPEN_LOOP else ("perf",):
                group = []
                for repetition in range(3):
                    path = directory / f"{suite}-{repetition}.json"
                    client = load(path)
                    validate_client(client, fixture, fixture_hash, repetition, suite, client_sha)
                    trial_ids = {r["request_id"] for r in client["requests"]}
                    require(request_ids.isdisjoint(trial_ids), "request IDs reused between server trials")
                    request_ids.update(trial_ids)
                    row = trial_row(base, client, path, repetition, client["trials"][0], client["requests"], issues)
                    require(row["performance_plot_eligible"], "existing exporter excludes performance trial")
                    row["measurement_suite"] = suite
                    group.append(row)
                    staged_sources.append({"path": str(path.resolve()), "sha256": digest(path)})
                staged.append((suite, aggregate_trials(name, suite, group)))
                staged_trials.extend(group)
            require(not issues, f"statistics exporter reported issues: {issues}")
            for suite, row in staged:
                rows[suite].append(row)
            trials.extend(staged_trials)
            sources.extend(staged_sources + [{"path": str(p.resolve()), "sha256": digest(p)} for p in required])
            statuses[name] = {"status": "included", "runtime_hashes_verified": 5,
                              "configuration": config_check, "observed_suite_history": history,
                              "validated_warmup_requests": len(warmup["requests"]),
                              "cache_reset_acknowledgements_verified": len(staged_trials) + 1}
        except (OSError, ValueError, KeyError, TypeError) as error:
            statuses[name] = {"status": "excluded", "reason": str(error)}
    complete = all(s["status"] == "included" for s in statuses.values())
    return {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "complete" if complete else "partial", "whitelist": names,
            "expected_rows": {"perf": 9, "perf-openloop": 3}, "runs": statuses,
            "tables": rows, "trials": trials, "sources": sources,
            "source_sha256": {"script": digest(__file__), "statistics": digest(REPO / "scripts/a100/summarize_results.py"),
                              "client": client_sha},
            "controlled_comparison_interpretations": INTERPRETATIONS,
            "notes": ["Only complete successful v2-whitelisted runs with all three paired trials enter tables.",
                      "Eligibility verifies client/output integrity. Numerical model correctness is evaluated separately using oracle and teacher-forcing checks.",
                      "Latency columns are median/min/max of three trial quantiles, not pooled request quantiles.",
                      "SLO fractions report the arithmetic mean of three equal 64-request trials; errors would count as misses, but failed trials are excluded entirely.",
                      "Saturation uses HTTP-start TTFT; open-loop emphasizes scheduled-arrival TTFT including client delay. Both scopes remain in the tables.",
                      "Each TPOT value is a per-request mean. SSE token-gap quantiles remain unavailable if any generated event batches tokens; timestamps are never interpolated.",
                      "Three repeats share one server startup and cumulative online-model training; ranges are observations, not confidence intervals.",
                      "Partial tables are provisional. Missing/failed runs are not imputed, and these artifacts do not establish global optimality or statistical significance."]}


def write_tables(report, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "final-performance.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for suite, rows in report["tables"].items():
        flat = []
        for row in rows:
            item = {key: row[key] for key in ("run", "label", "suite", "repetitions")}
            for metric in METRICS:
                for stat in ("median", "min", "max", "mean"):
                    item[f"{metric}_{stat}"] = (row.get(metric) or {}).get(stat)
            flat.append(item)
        fields = list(flat[0]) if flat else ["run", "label", "suite", "repetitions"]
        with (output / f"{suite}-summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(flat)


def plot_tables(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none"})
    artifacts = []
    for suite, rows in report["tables"].items():
        if not rows:
            continue
        rows = sorted(rows, key=lambda row: list(LABELS).index(row["run"]))
        opened = suite == "perf-openloop"
        metrics = [("output_tokens_per_s", "Output throughput (tokens/s)", 1, None),
                   ("arrival_ttft_s_p95" if opened else "ttft_s_p95", "Arrival TTFT p95 (s)" if opened else "HTTP TTFT p95 (s)", 1, 2),
                   ("tpot_s_p95", "Request-mean TPOT p95 (ms)", 1000, 50),
                   ("main_arrival_both_met_fraction" if opened else "main_both_met_fraction", "Joint SLO attainment (%)", 100, None)]
        fig, axes = plt.subplots(2, 2, figsize=(14, max(6.7, len(rows) * .45 + 3.4)))
        for ax, (metric, title, scale, target) in zip(axes.flat, metrics):
            for index, row in enumerate(rows):
                values = [t[metric] * scale for t in report["trials"] if t["run"] == row["run"] and t["measurement_suite"] == suite]
                color = "#21618C" if "baseline" not in row["run"] else "#666666"
                ax.hlines(index, min(values), max(values), color=color, linewidth=2, alpha=.55)
                ax.scatter(values, [index - .12, index, index + .12], color=color, s=24)
            if target is not None:
                ax.axvline(target, color="#BA4A00", linestyle="--", linewidth=1)
            if metric.endswith("fraction"):
                ax.set_xlim(-2, 102)
            ax.set_yticks(range(len(rows)), [r["label"] for r in rows], fontsize=8)
            ax.set_ylim(len(rows) - .5, -.5); ax.set_title(title, fontsize=11); ax.grid(axis="x", alpha=.2)
            ax.spines[["top", "right"]].set_visible(False)
        expected = report["expected_rows"][suite]
        mode = "8 requests/s · 64 client slots" if opened else "Saturation · 32 client slots"
        fig.suptitle(f"A100 PP2 v2 · {mode} · {len(rows)}/{expected} configurations complete", fontsize=13)
        fig.text(.5, .01, "Each dot: one 64-request trial; bars: observed min–max, not confidence intervals. Joint SLO: TTFT ≤ 2 s and request-mean TPOT ≤ 50 ms.", ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .035, 1, .95))
        for ext in ("png", "pdf", "svg"):
            path = output / f"{suite}-summary.{ext}"; fig.savefig(path, dpi=180, bbox_inches="tight"); artifacts.append(str(path))
        plt.close(fig)
    return artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO / "run/a100")
    parser.add_argument("--output-dir", type=Path, default=REPO / "run/a100/report-data/final-performance")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    report = build_report(args.root)
    write_tables(report, args.output_dir)
    plots = [] if args.no_plots else plot_tables(report, args.output_dir)
    print(json.dumps({"status": report["status"], "rows": {k: len(v) for k, v in report["tables"].items()},
                      "runs": report["runs"], "plots": plots}))


if __name__ == "__main__":
    main()
