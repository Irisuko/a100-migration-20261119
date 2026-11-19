#!/usr/bin/env python3
"""Select the best observed pilot only after the complete frozen matrix succeeds.

Criterion, fixed before measurement: highest mean per-request joint SLO fraction
(HTTP TTFT <= 2 s and mean TPOT <= 50 ms), then highest median output tokens/s.
This is a small pilot ranking, not a global optimum or a significance test.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[2]
EXPECTED_PAIRS = {(256, 2048), (256, 8192), (1024, 2048), (1024, 8192),
                  (1024, 32768), (2048, 2048), (2048, 8192), (2048, 32768)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def value(arguments, flag):
    try:
        return arguments[arguments.index(flag) + 1]
    except (ValueError, IndexError):
        raise ValueError(f"Missing argument {flag}")


def distribution(values):
    values = list(values)
    return {"min": min(values), "median": statistics.median(values),
            "max": max(values), "mean": statistics.mean(values)}


def cohort_signature(launch):
    command = launch["command"]
    allowed = {"--pdmux-config-path", "--pdmux-pp-prefill-chunk-size"}
    filtered, i = [], 0
    while i < len(command):
        if command[i] in allowed:
            i += 2
        else:
            filtered.append(command[i])
            i += 1
    config = dict(launch["config"])
    config.pop("split_forward_token_budget")
    return {"command_without_chunk_config": filtered, "config_without_slice_budget": config,
            "model": launch["model"], "environment": launch["environment"],
            "runtime_source_sha256": launch["runtime_source_sha256"],
            "repository_commit": launch["repository_commit"]}


def check_trial(client, fixture, fixture_hash, repetition):
    require(client.get("passed") is True, "client did not pass")
    require(client.get("suite") == "pilot", "not a pilot client result")
    require(client.get("fixture_sha256") == fixture_hash, "fixture hash mismatch")
    require(client.get("repeat") == 1, "expected one client trial per result")
    require(client.get("seed") == 20261114 + repetition, "unpaired trial seed")
    require(client.get("request_rate") == 0 and client.get("concurrency") == 32,
            "pilot must use saturation with concurrency 32")
    require(client.get("order_mode") == "fixture", "pilot request order changed")
    require(client.get("request_order") == fixture["suites"]["pilot"], "pilot cases/order changed")
    require(client.get("cache_reset") is True, "cold cache reset not recorded")
    require(client.get("external_cache_reset", {}).get("http_status") == 200,
            "missing successful external reset acknowledgement")
    records, aggregate = client["requests"], client["aggregate"]
    expected = fixture["suites"]["pilot"]
    require(len(records) == len(expected) == 24, "pilot coverage incomplete")
    require([row["fixture_id"] for row in records] == expected, "pilot record order changed")
    require(len({row["request_id"] for row in records}) == len(records), "duplicate request IDs")
    output_tokens = 0
    for row in records:
        case = fixture["cases"][row["fixture_id"]]
        require(row.get("error") is None, "request failed")
        require(row.get("input_tokens") == case["input_tokens"], "input length changed")
        require(row.get("requested_output_tokens") == case["max_new_tokens"], "output budget changed")
        require(len(row.get("output_ids", [])) == case["max_new_tokens"] == row.get("actual_output_tokens"),
                "truncated output")
        require(row.get("sse_done") is True, "missing SSE completion")
        for key in ("ttft_s", "tpot_s"):
            require(isinstance(row.get(key), (float, int)) and math.isfinite(row[key]) and row[key] >= 0,
                    f"invalid {key}")
        output_tokens += row["actual_output_tokens"]
    require(aggregate.get("failed") == 0 and aggregate.get("completed") == 24, "failed/incomplete aggregate")
    slo = aggregate["slo"]
    require(slo.get("ttft_target_s") == 2 and slo.get("tpot_target_s") == .05, "SLO target changed")
    actual_fraction = sum(r["ttft_s"] <= 2 and r["tpot_s"] <= .05 for r in records) / 24
    require(slo.get("both_met_fraction") == actual_fraction, "joint SLO aggregate mismatch")
    require(output_tokens == aggregate.get("actual_output_tokens"), "output count aggregate mismatch")
    throughput = aggregate["output_tokens_per_s"]
    require(math.isclose(throughput, output_tokens / aggregate["wall_s"], rel_tol=1e-12),
            "throughput aggregate mismatch")
    return {"joint_slo_fraction": actual_fraction, "output_tokens_per_s": throughput,
            "ttft_p95_s": aggregate["ttft_s"]["p95"], "tpot_p95_s": aggregate["tpot_s"]["p95"],
            "wall_s": aggregate["wall_s"], "completed": 24, "output_tokens": output_tokens}


def ranking_key(row):
    # Last fields only make exact metric ties deterministic; they are not extra objectives.
    return (-row["joint_slo_fraction"]["mean"], -row["output_tokens_per_s"]["median"],
            row["chunk_tokens"], row["slice_token_layer_budget"], row["run"])


def select(jobs_path, root, fixture_path):
    jobs = load(jobs_path)
    require(len(jobs) == 8, "frozen pilot matrix must contain eight jobs")
    names = [value(job["args"], "--name") for job in jobs]
    require(len(set(names)) == 8, "duplicate pilot job name")
    pending = [name for name in names
               if (root / name / "server.pid").exists()
               or any(not (root / name / artifact).exists()
                      for artifact in ("outcome.json", "shutdown.json", "scheduler.json"))]
    if pending:
        return None, pending
    fixture = load(fixture_path)
    fixture_hash = digest(fixture_path)
    rows, reference, client_hash, actual_pairs = [], None, None, set()
    for job, name in zip(jobs, names):
        directory = root / name
        outcome, launch = load(directory / "outcome.json"), load(directory / "launch.json")
        require(outcome.get("passed") is True, f"{name}: run failed")
        require(load(directory / "shutdown.json").get("remaining") == [], f"{name}: cleanup incomplete")
        require(launch.get("mode") == "mux" and value(launch["command"], "--pdmux-pp-policy") == "alp",
                f"{name}: non-ALP pilot")
        require(Path(launch["model"]).resolve() == Path(fixture["model_path"]).resolve(),
                f"{name}: server model differs from frozen tokenizer fixture")
        require(all(launch.get(key) is False for key in ("profile", "trace", "plan_reuse_check")),
                f"{name}: diagnostic instrumentation enabled or unknown")
        require(value(job["args"], "--suites") == "warmup,pilot" and value(job["args"], "--repetitions") == "2",
                f"{name}: training history/repetition plan changed")
        require([(r["suite"], r["repeat"]) for r in outcome["runs"]] == [("warmup", 0), ("pilot", 0), ("pilot", 1)],
                f"{name}: actual suite history differs")
        require(all(r["exit_code"] == 0 for r in outcome["runs"]), f"{name}: client command failed")
        sig = cohort_signature(launch)
        if reference is None:
            reference = sig
        require(sig == reference, f"{name}: runtime/model/hardware/config differs outside chunk/budget")
        for relative, sha in launch["runtime_source_sha256"].items():
            require(digest(REPO / relative) == sha, f"{name}: runtime source differs from currently validated fix: {relative}")
        chunk = int(value(launch["command"], "--pdmux-pp-prefill-chunk-size"))
        budget = int(launch["config"]["split_forward_token_budget"])
        require(chunk == int(value(job["args"], "--chunk")), f"{name}: chunk differs from planned job")
        actual_pairs.add((chunk, budget))
        trials = []
        for repetition in range(2):
            path = directory / f"pilot-{repetition}.json"
            client = load(path)
            trial = check_trial(client, fixture, fixture_hash, repetition)
            if client_hash is None:
                client_hash = client.get("client_source_sha256")
            require(client.get("client_source_sha256") == client_hash and client_hash,
                    f"{name}: measurement client source differs")
            trials.append({"source": str(path), "sha256": digest(path), "repetition": repetition, **trial})
        row = {"run": name, "chunk_tokens": chunk, "slice_token_layer_budget": budget,
               "launch_source": str(directory / "launch.json"), "trials": trials}
        for metric in ("joint_slo_fraction", "output_tokens_per_s", "ttft_p95_s", "tpot_p95_s", "wall_s"):
            row[metric] = distribution(t[metric] for t in trials)
        rows.append(row)
    require(actual_pairs == EXPECTED_PAIRS, f"actual pilot matrix changed: {actual_pairs}")
    rows.sort(key=ranking_key)
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return {"schema_version": 1, "completed": True, "jobs_source": str(jobs_path),
            "jobs_sha256": digest(jobs_path), "fixture_sha256": fixture_hash,
            "client_source_sha256": client_hash, "cohort_signature": reference,
            "criterion": {"primary": "maximize mean joint-SLO fraction over two trials; HTTP TTFT <= 2 s and per-request mean TPOT <= 50 ms",
                          "secondary": "maximize median actual output tokens/second over the same two trials",
                          "exact_tie": "smaller chunk, then smaller slice budget, then run name"},
            "winner": {key: rows[0][key] for key in ("run", "chunk_tokens", "slice_token_layer_budget")},
            "rows": rows,
            "skipped_equivalent": {"chunk_tokens": 256, "slice_token_layer_budget": 32768,
                                   "equivalent_to_budget": 8192, "reason": "Both budgets allow all 16 local layers for every chunk <=256 tokens."},
            "limitations": ["Best observed among these eight pilot configurations, not a global optimum.",
                            "Two trials per configuration share one server startup and sequential online-model training.",
                            "Pilot has 24 requests, so concurrency32 does not create more than24 active clients.",
                            "The same synthetic passage family is used in pilot and evaluation with different fixed request IDs/prompts.",
                            "TTFT excludes client queue delay; measured TPOT is a request mean, not individual token-gap latency."]}, []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, default=REPO / "run/a100/pilot-jobs.json")
    parser.add_argument("--root", type=Path, default=REPO / "run/a100")
    parser.add_argument("--fixtures", type=Path, default=REPO / "run/a100/fixtures.json")
    parser.add_argument("--output", type=Path, default=REPO / "run/a100/pilot-selection.json")
    args = parser.parse_args()
    try:
        report, pending = select(args.jobs, args.root, args.fixtures)
    except (OSError, ValueError, KeyError) as error:
        print(f"Pilot selection refused: {error}", file=sys.stderr)
        return 1
    if pending:
        print(json.dumps({"selection_written": False, "pending": pending}))
        return 2
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    flat = []
    for row in report["rows"]:
        item = {key: row[key] for key in ("rank", "run", "chunk_tokens", "slice_token_layer_budget")}
        for metric in ("joint_slo_fraction", "output_tokens_per_s", "ttft_p95_s", "tpot_p95_s", "wall_s"):
            item.update({f"{metric}_{key}": number for key, number in row[metric].items()})
        flat.append(item)
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps({"winner": report["winner"], "completed_configurations": len(report["rows"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
