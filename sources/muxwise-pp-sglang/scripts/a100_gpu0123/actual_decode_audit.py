#!/usr/bin/env python3
"""Bounded actual-decode diagnosis under control.py; no performance eligibility.

Example: actual_decode_audit.py --out ROOT/diagnostics/LABEL -- \
  --model 70b --family plain --tp 1 --pp 4 --mem .85

Only server_run.preflight is replaced in this controlled child process. The
ordinary family is fixed to chunk2048. The serving source is never edited.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import threading
import time
from urllib.error import HTTPError

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
LENGTHS = [2047, 2048, 2049, 4095, 4096, 4097, 8192, 8193]
FOCUS = 8193
TOKENS = 32


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_float": repr(value)}
    if isinstance(value, dict):
        return {key: safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(item) for item in value]
    return value


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(safe_json(value), indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def condition_plan():
    return [{"id": f"repeat{repeat}-{cache}-{mode}", "repeat": repeat, "cache": cache, "mode": mode,
             "input_lengths": [FOCUS] if mode == "serial" else list(LENGTHS), "output_tokens": TOKENS}
            for repeat in [1, 2] for cache in ["cold", "warm"] for mode in ["serial", "concurrent8"]]


def parse_decode_response(body, input_ids, count):
    require(isinstance(body, dict) and not body.get("error"), "Invalid generate response")
    ids, meta = body.get("output_ids"), body.get("meta_info", {})
    require(isinstance(ids, list) and len(ids) == count
            and all(type(token) is int and 0 <= token < 128256 for token in ids), "Output token count/IDs invalid")
    require(meta.get("prompt_tokens") == len(input_ids) and meta.get("completion_tokens") == count,
            "Prompt/completion token coverage differs")
    require(isinstance(meta.get("finish_reason"), dict) and meta["finish_reason"].get("type") == "length",
            "Generation did not finish at the requested length")
    cached = meta.get("cached_tokens")
    require(type(cached) is int and 0 <= cached <= len(input_ids), "Missing or invalid actual cache count")
    chosen, tops = meta.get("output_token_logprobs"), meta.get("output_top_logprobs")
    require(isinstance(chosen, list) and isinstance(tops, list) and len(chosen) == len(tops) == count,
            "Actual decode logprob coverage incomplete")
    def entry(item):
        require(isinstance(item, (list, tuple)) and len(item) >= 2, "Malformed logprob entry")
        score, token = item[:2]
        require(type(score) in (int, float) and math.isfinite(score), "Nonfinite decode logprob")
        require(type(token) is int and 0 <= token < 128256, "Invalid logprob token ID")
        return score, token
    tokens = []
    for position, (token_id, actual, alternatives) in enumerate(zip(ids, chosen, tops)):
        score, token = entry(actual)
        require(token == token_id, "Actual decode logprob token alignment mismatch")
        require(isinstance(alternatives, list) and len(alternatives) == 5, "Expected five alternatives at every decode token")
        ranked = [entry(item) for item in alternatives]
        scores, top_ids = [item[0] for item in ranked], [item[1] for item in ranked]
        require(scores == sorted(scores, reverse=True) and len(set(top_ids)) == 5,
                "Top5 ordering or unique-token coverage invalid")
        ties = sorted(t for lp, t in ranked if lp == scores[0])
        tokens.append(dict(position=position, chosen=token_id, chosen_logprob=score,
            reported_top1_token=top_ids[0], reported_top1_logprob=scores[0],
            top5_ids=top_ids, top5_logprobs=scores, regret=scores[0] - score,
            top_two_margin=scores[0] - scores[1], reported_top1_match=token_id == top_ids[0],
            reported_max_tie_ids=ties, canonical_top5_token=min(ties),
            canonical_top5_match=token_id == min(ties), chosen_equals_reported_max=score == scores[0],
            all_max_ties_in_top5=scores[-1] < scores[0], negative_regret_observed=score > scores[0]))
    return dict(audited_tokens=count, full_token_coverage=True, cached_tokens=cached, tokens=tokens,
                interpretation="Coverage only. Reported ties, low regret and token matches do not create a numerical correctness PASS.")


def capture_request(runner, port, ids, count, path, label, scope, barrier=None):
    payload = dict(input_ids=ids, rid=f"actual-decode-{label}", return_logprob=True,
                   top_logprobs_num=5, logprob_start_len=len(ids) - 1, stream=False,
                   sampling_params=dict(temperature=0, max_new_tokens=count, ignore_eos=True))
    record = dict(label=label, scope=scope, request=payload, diagnostic_only=True,
                  measurement_eligible=False, status="pending", full_token_coverage=False)
    atomic_json(path, record)
    try:
        if barrier is not None:
            barrier.wait(timeout=30)
        record["sent_unix"] = time.time()
        record["sent_monotonic"] = time.monotonic()
        body = runner.request(port, "/generate", payload, timeout=600)
        record.update(raw_response=body, finished_unix=time.time(), finished_monotonic=time.monotonic(), status="response_saved")
        atomic_json(path, record)  # Preserve complete API response before validation.
        parsed = parse_decode_response(body, ids, count)
        record.update(parsed=parsed, output_ids=body["output_ids"], status="diagnostic_complete", full_token_coverage=True)
        atomic_json(path, record)
        return record
    except BaseException as error:
        if isinstance(error, HTTPError):
            record["raw_http_error"] = dict(status=error.code, body=error.read().decode("utf-8", errors="replace"))
        record.update(status="diagnostic_incomplete", error=repr(error))
        atomic_json(path, record)
        raise


def run_preflight(runner, port, model, out, family):
    out = Path(out)
    directory = out / "actual-decode"
    directory.mkdir(exist_ok=False)
    summary_path = out / "actual-decode-summary.json"
    summary = dict(schema_version=1, status="running", diagnostic_only=True, measurement_eligible=False,
        full_token_coverage=False, expected_measured_requests=36, expected_measured_output_tokens=1152,
        measured_requests=0, measured_output_tokens=0, measured_completed_requests=0,
        expected_warmup_requests=19, expected_warmup_output_tokens=26,
        warmup_requests=0, warmup_output_tokens=0,
        planned_cells=condition_plan(), cells=[], errors=[], family=family,
        interpretation="Actual autoregressive output-token logprobs. No teacher-forced continuation and no correctness tolerance.")
    atomic_json(summary_path, summary)
    try:
        server = runner.request(port, "/get_server_info", timeout=30)
        require(Path(server.get("model_path", "")).resolve() == Path(model).resolve(), "Diagnostic endpoint model mismatch")
        require(server.get("tp_size") == 1 and server.get("pp_size") == 4
                and server.get("dtype") in {"bfloat16", "bf16", "torch.bfloat16"}, "Expected BF16 TP1/PP4 endpoint")
        require(server.get("enable_pdmux") is (family == "muxpp"), "Diagnostic endpoint family mismatch")
        if family == "plain":
            require(server.get("chunked_prefill_size") == 2048, "Ordinary diagnostic must use chunk2048")
        summary["server_before"] = server
        fixtures = runner.fixture_cases(model)
        selected = {length: fixtures[runner.CORRECTNESS_LENGTHS.index(length)] for length in LENGTHS}
        require(all(len(ids) == length for length, ids in selected.items()), "Fixture length mismatch")
        atomic_json(directory / "prompts.json", {str(k): v for k, v in selected.items()})
        summary["fixture_sha256"] = sha(directory / "prompts.json")
        runner.flush(port)
        rng = random.Random(730031)
        long_ids = [128000] + [rng.randrange(128000) for _ in range(31662)]
        prewarm = capture_request(runner, port, long_ids, 8, directory / "request-long-prewarm.json", "long-prewarm", "long_prewarm")
        summary.update(warmup_requests=1, warmup_output_tokens=prewarm["parsed"]["audited_tokens"])
        runner.flush(port)
        runner.resource_phase(out, "initial", [port])
        for condition in condition_plan():
            folder = directory / condition["id"]
            folder.mkdir()
            cell = {**condition, "status": "running", "cache_flush_acknowledged": False,
                    "warmup_files": [], "measured_files": [], "completed_requests": 0}
            summary["cells"].append(cell)
            atomic_json(summary_path, summary)
            runner.flush(port)
            cell.update(cache_flush_acknowledged=True, cache_flush_unix=time.time())
            if condition["cache"] == "warm":
                for index, length in enumerate(condition["input_lengths"]):
                    path = folder / f"request-warmup-{index:02d}.json"
                    warmup = capture_request(runner, port, selected[length], 1, path,
                                             f"{condition['id']}-warmup-{index}", "cache_warmup")
                    summary["warmup_requests"] += 1
                    summary["warmup_output_tokens"] += warmup["parsed"]["audited_tokens"]
                    cell["warmup_files"].append(str(path))
            ids_list = [selected[length] for length in condition["input_lengths"]]
            barrier = threading.Barrier(8) if condition["mode"] == "concurrent8" else None
            results = {}
            failures = []
            with ThreadPoolExecutor(max_workers=len(ids_list)) as pool:
                jobs = {}
                for index, ids in enumerate(ids_list):
                    path = folder / f"request-measured-{index:02d}.json"
                    cell["measured_files"].append(str(path))
                    jobs[pool.submit(capture_request, runner, port, ids, TOKENS, path,
                                     f"{condition['id']}-measured-{index}", "measured", barrier)] = index
                for future in as_completed(jobs):
                    try:
                        result = future.result()
                        results[jobs[future]] = result
                        cell["completed_requests"] += 1
                        summary["measured_completed_requests"] += 1
                        summary["measured_output_tokens"] += result["parsed"]["audited_tokens"]
                    except BaseException as error:
                        failures.append(repr(error))
                    summary["measured_requests"] += 1
                    atomic_json(summary_path, summary)
            if failures:
                cell.update(status="diagnostic_incomplete", errors=failures)
                raise ValueError(f"Actual decode coverage failed in {condition['id']}: {failures}")
            focus = results[condition["input_lengths"].index(FOCUS)]
            cell.update(status="diagnostic_complete", focus_output_ids=focus["output_ids"],
                        actual_cached_tokens={str(i): result["parsed"]["cached_tokens"] for i, result in sorted(results.items())},
                        send_spread_sec=max(result["sent_monotonic"] for result in results.values())
                                        - min(result["sent_monotonic"] for result in results.values()))
            atomic_json(summary_path, summary)
        require(len(summary["cells"]) == 8 and summary["measured_requests"] == summary["measured_completed_requests"] == 36
                and summary["measured_output_tokens"] == 1152
                and summary["warmup_requests"] == 19 and summary["warmup_output_tokens"] == 26,
                "Fixed diagnostic request/token budget coverage incomplete")
        runner.resource_phase(out, "final", [port])
        atomic_json(out / "server-after.json", {str(port): runner.request(port, "/get_server_info", timeout=30)})
        summary.update(status="diagnostic_complete", full_token_coverage=True)
        return summary
    except BaseException as error:
        summary.update(status="diagnostic_incomplete", errors=[*summary["errors"], repr(error)])
        raise
    finally:
        summary["ended_unix"] = time.time()
        atomic_json(summary_path, summary)


def make_plan(argv, out):
    argv = list(argv)
    split = argv.index("--extra") if "--extra" in argv else len(argv)
    before, extra = argv[:split], argv[split + 1:] if split < len(argv) else []
    allowed = {"model", "family", "tp", "pp", "replicas", "mem", "kv", "running", "graph", "chunk",
               "baseline-chunk", "inflight", "budget", "no-async", "no-cpp", "custom-allreduce", "startup-timeout", "port"}
    seen = set()
    for arg in before:
        if arg.startswith("--"):
            key = arg[2:].split("=", 1)[0]
            require(key in allowed and key not in seen, f"Unsupported or duplicate diagnostic runner option: {arg}")
            seen.add(key)
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--model", choices=["8b", "70b"], required=True)
    parser.add_argument("--family", choices=["plain", "muxpp"], required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=4)
    parser.add_argument("--replicas", type=int, default=1)
    known, _ = parser.parse_known_args(before)
    require((known.tp, known.pp, known.replicas) == (1, 4, 1), "Diagnostic is bounded to TP1/PP4/one replica")
    if known.family == "plain":
        for index, arg in enumerate(extra):
            if arg == "--chunked-prefill-size":
                require(index + 1 < len(extra) and extra[index + 1] == "2048", "Ordinary chunk override must equal2048")
            if arg.startswith("--chunked-prefill-size="):
                require(arg.split("=", 1)[1] == "2048", "Ordinary chunk override must equal2048")
        extra += ["--chunked-prefill-size", "2048"]
    out = Path(out).resolve()
    return dict(schema_version=1, out=str(out), model=known.model, family=known.family,
        created_at_utc=datetime.now(timezone.utc).isoformat(), diagnostic_only=True, measurement_eligible=False,
        cells=condition_plan(), measured_requests=36, measured_output_tokens=1152,
        cache_warmup_requests=18, long_prewarm_requests=1, long_prewarm_input_tokens=31663, long_prewarm_output_tokens=8,
        concurrency_note="Eight barrier-released HTTP calls; actual server batching/order is observed, not guaranteed identical.",
        replacement="Only server_run.preflight in this child; standard lifecycle and run-final resource audit retained.",
        server_args=before + ["--suite", "smoke", "--out", str(out / "serving")] + (["--extra", *extra] if extra else []))


def mark_diagnostic(out, plan):
    path = Path(out) / "serving/run.json"
    if not path.is_file():
        return
    original = path.with_name("run.before-diagnostic-marking.json")
    if not original.exists():
        with original.open("xb") as f:
            f.write(path.read_bytes())
    metadata = json.loads(path.read_text())
    metadata.update(diagnostic_only=True, measurement_eligible=False, diagnostic_kind="actual_decode_logprobs",
                    diagnostic_wrapper_sha256=plan["wrapper_sha256"], diagnostic_plan_sha256=sha(Path(out) / "plan.json"),
                    diagnostic_original_metadata_sha256=sha(original))
    atomic_json(path, metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("server_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    argv = args.server_args[1:] if args.server_args[:1] == ["--"] else args.server_args
    plan = make_plan(argv, args.out)
    import server_run
    server_run.require_active_controller()  # Before output creation or any endpoint call.
    out = Path(plan["out"])
    out.mkdir(parents=True, exist_ok=False)
    plan.update(wrapper_path=str(HERE), wrapper_sha256=sha(HERE),
                source_provenance=server_run.source_provenance(REPO),
                original_preflight_source=str(HERE.with_name("server_run.py")),
                original_preflight_source_sha256=sha(HERE.with_name("server_run.py")))
    atomic_json(out / "plan.json", plan)
    original, original_argv = server_run.preflight, sys.argv
    result = dict(status="running", diagnostic_only=True, measurement_eligible=False, plan_sha256=sha(out / "plan.json"))
    atomic_json(out / "diagnostic.json", result)
    def replacement(port, model, worker_out, suite, transport=None, worker_ports=None, stress200=False):
        require(suite == "smoke" and transport is None and not stress200
                and (worker_ports is None or worker_ports == [port]), "Unexpected diagnostic preflight scope")
        return run_preflight(server_run, port, model, worker_out, plan["family"])
    server_run.preflight = replacement
    sys.argv = [str(HERE.with_name("server_run.py")), *plan["server_args"]]
    try:
        server_run.main()
        summary = json.loads((out / "serving/worker0/actual-decode-summary.json").read_text())
        require(summary["full_token_coverage"] and summary["measured_requests"] == 36, "Missing completed bounded diagnosis")
        result.update(status="diagnostic_complete", full_token_coverage=True,
                      measured_requests=36, measured_output_tokens=1152)
    except BaseException as error:
        result.update(status="diagnostic_incomplete", full_token_coverage=False, error=repr(error))
        raise
    finally:
        server_run.preflight, sys.argv = original, original_argv
        mark_diagnostic(out, plan)
        result["ended_unix"] = time.time()
        atomic_json(out / "diagnostic.json", result)


if __name__ == "__main__":
    main()
