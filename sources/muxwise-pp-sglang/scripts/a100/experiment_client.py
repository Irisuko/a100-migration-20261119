#!/usr/bin/env python3
"""Frozen-token workloads and native SGLang SSE measurements for A100 experiments.

Preparation requires transformers; execution uses only the standard library.
The caller owns cache flushing, server lifecycle, and GPU experiment scheduling.
Native /generate output_ids are incremental, with validated prompt lookback IDs
in the first frame on this repository's streaming API.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
import uuid
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener


SCHEMA_VERSION = 1
SUITES = ("serial", "concurrent", "cache", "warmup", "pilot", "perf")
PASSAGES = [
    "A coastal town restored its public library after a winter storm. Residents "
    "catalogued damaged books, moved computers upstairs, and installed drainage "
    "around the entrance. The project team kept a weekly spending ledger and "
    "published progress reports. Volunteers offered reading sessions at the "
    "community hall while the building was closed. The final inspection checked "
    "electrical safety, wheelchair access, and emergency exits before reopening.",
    "A field station monitors a river with temperature sensors and manual water "
    "samples. Researchers record the date, location, weather, and instrument "
    "calibration for every observation. Measurements taken after heavy rainfall "
    "are compared with dry weeks. Duplicate samples help detect measurement "
    "errors. The team shares both cleaned tables and original notes so another "
    "research group can reproduce the analysis without guessing missing details.",
    "A small manufacturer reorganized its workshop to reduce delivery delays. "
    "Workers separated inspection from packing and placed frequently used tools "
    "near assembly stations. A daily planning board lists orders, available "
    "materials, and maintenance tasks. Managers compare completed orders with "
    "the original schedule and investigate late shipments. The trial tracks "
    "quality defects as well as speed, because faster production is useful only "
    "when finished products continue to meet the agreed specifications.",
    "A school garden grows beans, carrots, and herbs in raised beds. Students "
    "measure rainfall, record planting dates, and rotate watering duties. Fallen "
    "leaves become compost, while labeled paths protect young seedlings. During "
    "summer holidays a neighborhood group maintains the beds. At harvest time "
    "the class compares crop yields and discusses why sunlight, soil moisture, "
    "and spacing affected growth. The next planting plan uses those observations.",
    "An archive digitizes handwritten travel journals. Each page is photographed "
    "with a color reference and assigned a stable identifier. Two volunteers "
    "transcribe difficult passages independently, then compare their work. "
    "Uncertain words remain marked instead of being silently replaced. The "
    "catalog links images, transcriptions, place names, and dates. Researchers "
    "can therefore inspect the source whenever a historical interpretation "
    "depends on a disputed word or an unusual spelling.",
    "A city evaluates a new bus timetable using passenger counts and travel "
    "times. Planners distinguish crowded morning trips from quiet afternoon "
    "services. Drivers report delays caused by roadworks and difficult turns. "
    "The evaluation includes missed connections and accessibility at stops. "
    "Results are compared across several weeks so one rainy day does not drive "
    "the conclusion. Residents receive a summary explaining the proposed changes.",
    "A software team prepares a release by reviewing changes, testing important "
    "workflows, and recording known limitations. The deployment plan identifies "
    "who monitors errors and how to restore the previous version if necessary. "
    "Small releases make unexpected behavior easier to investigate. After the "
    "release, the team compares observed results with its original expectations "
    "and updates documentation when users encounter confusing instructions.",
    "A mountain rescue group practices navigation and radio communication on "
    "marked trails. Teams carry maps, spare batteries, water, and first-aid "
    "supplies. Each exercise has a written route and a scheduled check-in time. "
    "Observers record communication gaps without interrupting safe practice. "
    "The group reviews the exercise afterward and assigns specific improvements "
    "for the next session. Weather conditions determine whether an activity "
    "continues, changes route, or returns to the meeting point.",
]
NATURAL_PROMPTS = [
    "Explain why keeping a spending ledger helps a community renovation project.",
    "Give three practical ways to organize a school garden watering schedule.",
    "Write a brief scene in which a librarian discovers an old travel journal.",
    "Compare measuring a river once with measuring it every week for a year.",
    "Explain how two independent transcriptions can improve an archive.",
    "Outline a simple plan for evaluating whether a new bus timetable works.",
    "Describe the difference between an observation and an explanation.",
    "Suggest a clear checklist for handing a workshop order to the packing team.",
    "Write a friendly invitation to a neighborhood library reopening.",
    "Explain why a software release should have a documented recovery plan.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, label, minimum=0):
    require(isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
            f"invalid {label}: {value!r}")
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def percentile(values, percent):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values):
    if not values:
        return {"count": 0}
    return {
        "count": len(values), "min": min(values), "mean": statistics.fmean(values),
        "p50": percentile(values, 50), "p90": percentile(values, 90),
        "p95": percentile(values, 95), "p99": percentile(values, 99), "max": max(values),
    }


def iter_sse(lines):
    """Yield SSE data fields; do not assume network reads equal SSE messages."""
    fields = []
    for raw in lines:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if fields:
                yield "\n".join(fields)
                fields = []
        elif line.startswith("data:"):
            data = line[5:]
            fields.append(data[1:] if data.startswith(" ") else data)
    # An event without its terminating blank line is incomplete and is ignored.


class StreamAccumulator:
    def __init__(self, prompt_ids=None):
        self.prompt_ids = list(prompt_ids) if prompt_ids is not None else []
        self.output_ids = []
        self.raw_output_ids = []
        self.prompt_lookback_dropped_ids = []
        self.last_completion_tokens = 0
        self.saw_token_ids = False
        self.events = []
        self.meta = {}
        self.first_token_s = None
        self.last_token_s = None

    def feed(self, event, elapsed):
        require(math.isfinite(elapsed) and elapsed >= 0, "invalid event time")
        require(isinstance(event, dict), "SSE data must be an object")
        require("error" not in event, f"server error: {event.get('error')}")
        token_ids = event.get("output_ids", [])
        require(isinstance(token_ids, list), "output_ids must be a list")
        for token in token_ids:
            integer(token, "output token id")
        self.raw_output_ids.extend(token_ids)
        metadata = event.get("meta_info", {})
        require(isinstance(metadata, dict), "meta_info must be an object")
        completion = integer(metadata.get("completion_tokens"), "completion_tokens")
        delta = completion - self.last_completion_tokens
        require(delta >= 0, "completion_tokens decreased during stream")
        raw_count = len(token_ids)
        dropped = 0
        if len(token_ids) > delta and not self.saw_token_ids:
            dropped = len(token_ids) - delta
            prefix = token_ids[:dropped]
            require(dropped <= len(self.prompt_ids) and prefix == self.prompt_ids[-dropped:],
                    "initial excess output_ids do not exactly match the prompt suffix")
            self.prompt_lookback_dropped_ids = prefix
            token_ids = token_ids[dropped:]
        require(len(token_ids) == delta,
                f"output_ids length {len(token_ids)} != completion delta {delta}")
        self.saw_token_ids = self.saw_token_ids or raw_count > 0
        self.last_completion_tokens = completion
        if token_ids:
            if self.first_token_s is None:
                self.first_token_s = elapsed
            self.last_token_s = elapsed
            self.output_ids.extend(token_ids)
        self.meta.update(metadata)
        self.events.append({
            "elapsed_s": elapsed, "new_tokens": len(token_ids),
            "total_tokens": len(self.output_ids), "raw_tokens": raw_count,
            "prompt_lookback_dropped": dropped, "server_completion_tokens": completion,
        })

    def finish(self, done, elapsed, input_tokens, requested_tokens):
        require(done, "SSE stream ended without [DONE]")
        require(len(self.output_ids) == requested_tokens,
                f"output length {len(self.output_ids)} != requested {requested_tokens}")
        require(integer(self.meta.get("completion_tokens"), "completion_tokens")
                == len(self.output_ids), "completion_tokens differs from output_ids length")
        require(integer(self.meta.get("prompt_tokens"), "prompt_tokens", 1) == input_tokens,
                "server prompt_tokens differs from fixed input length")
        cached = integer(self.meta.get("cached_tokens"), "cached_tokens")
        require(cached <= input_tokens, "cached_tokens exceeds prompt length")
        reason = self.meta.get("finish_reason")
        require(isinstance(reason, dict) and reason.get("type") == "length",
                f"unexpected finish_reason: {reason!r}")
        require(self.first_token_s is not None, "stream returned no output tokens")
        tpot = ((self.last_token_s - self.first_token_s) / (len(self.output_ids) - 1)
                if len(self.output_ids) > 1 else 0.0)
        require(all(math.isfinite(value) and value >= 0
                    for value in (elapsed, self.first_token_s, self.last_token_s, tpot)),
                "invalid latency metric")
        return {
            "output_ids": self.output_ids, "actual_output_tokens": len(self.output_ids),
            "raw_output_ids": self.raw_output_ids,
            "prompt_lookback_dropped_ids": self.prompt_lookback_dropped_ids,
            "prompt_lookback_dropped_count": len(self.prompt_lookback_dropped_ids),
            "finish_reason": reason, "cached_tokens": cached, "ttft_s": self.first_token_s,
            "last_token_s": self.last_token_s, "e2e_s": elapsed, "tpot_s": tpot,
            "events": self.events, "sse_done": done,
        }


def prepare(args):
    from transformers import AutoTokenizer
    import transformers

    model_path = Path(args.model_path).resolve()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    rng = random.Random(args.seed)
    cases = {}

    def encode(content):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True
        )

    def add(case_id, target, output, category, content=None):
        if content is not None:
            ids = encode(content)
        else:
            passages = list(PASSAGES)
            rng.shuffle(passages)
            intro = f"Reference case {case_id}. Read these project notes carefully.\n\n"
            ending = "\n\nSummarize the notes, identify two practical lessons, and explain your reasoning."
            ids = encode(intro + "\n\n".join(passages * 12) + ending)
            require(len(ids) >= target, f"source text too short for {case_id}")
            tail = min(48, target // 4)
            ids = ids[:target - tail] + ids[-tail:]
        ids = [int(token) for token in ids]
        require(len(ids) + output <= 8192, f"context budget exceeded: {case_id}")
        cases[case_id] = {
            "input_ids": ids, "input_tokens": len(ids), "max_new_tokens": output,
            "category": category, "input_sha256": digest_bytes(json.dumps(ids).encode()),
        }
        return case_id

    serial = [add(f"natural-{i:02d}", None, 64, "natural", prompt)
              for i, prompt in enumerate(NATURAL_PROMPTS)]
    for length in (255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049):
        serial.append(add(f"boundary-{length}", length, 32, "boundary"))
    serial += [add("long-4096", 4096, 64, "long"), add("long-6144", 6144, 128, "long")]

    concurrent = []
    for i in range(32):
        length = (64, 128, 256, 2048, 4096, 6144, 512, 128)[i % 8]
        output = (256, 128, 128, 64, 64, 32, 256, 128)[i % 8]
        concurrent.append(add(f"concurrent-{i:02d}", length, output, "concurrent"))

    base = add("cache-base", 2048, 64, "cache")
    branch = add("cache-branch", 2048, 64, "cache")
    branch_ids = cases[base]["input_ids"][:1792] + cases[branch]["input_ids"][-256:]
    cases[branch]["input_ids"] = branch_ids
    cases[branch]["input_sha256"] = digest_bytes(json.dumps(branch_ids).encode())

    warmup = [add(f"warmup-{i:02d}", (256, 512, 1024)[i % 3], 32, "warmup")
              for i in range(24)]
    pilot = [add(f"pilot-{i:02d}", (128, 512, 2048, 4096, 64, 6144)[i % 6],
                 (256, 128, 64, 64, 256, 32)[i % 6], "pilot") for i in range(24)]
    perf = [add(f"perf-{i:02d}", (64, 128, 4096, 256, 512, 128, 6144, 256)[i % 8],
                (256, 128, 64, 256, 128, 256, 32, 128)[i % 8], "perf") for i in range(64)]
    fingerprint = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json"):
        path = model_path / name
        if path.exists():
            fingerprint[name] = digest_bytes(path.read_bytes())
    fixture = {
        "schema_version": SCHEMA_VERSION, "model_path": str(model_path),
        "tokenizer_fingerprint": fingerprint, "transformers_version": transformers.__version__,
        "seed": args.seed, "context_limit": 8192, "cases": cases,
        "suites": {"serial": serial, "concurrent": concurrent,
                   "cache": [base, base, branch, branch, base],
                   "warmup": warmup, "pilot": pilot, "perf": perf},
    }
    write_json(args.output, fixture)
    print(json.dumps({"output": str(args.output), "cases": len(cases),
                      "suite_counts": {key: len(value) for key, value in fixture["suites"].items()}}))
    return 0


def send_request(base_url, case, record, trial_start, timeout):
    scheduled = trial_start + record["scheduled_s"]
    remaining = scheduled - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)
    began = time.monotonic()
    record.update({"started_s": began - trial_start,
                   "queue_delay_s": max(0.0, began - scheduled), "error": None})
    stream = StreamAccumulator(case["input_ids"])
    done = False
    payload = {
        "rid": record["request_id"], "input_ids": case["input_ids"],
        "sampling_params": {"temperature": 0, "max_new_tokens": case["max_new_tokens"],
                            "ignore_eos": True},
        "stream": True,
    }
    request = Request(base_url.rstrip("/") + "/generate", data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
    try:
        # Ignore HTTP_PROXY/HTTPS_PROXY even if NO_PROXY is missing on this host.
        with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
            require(response.status == 200, f"unexpected HTTP status: {response.status}")
            require("text/event-stream" in response.headers.get("Content-Type", ""),
                    "response is not text/event-stream")
            for data in iter_sse(response):
                if data == "[DONE]":
                    done = True
                    break
                stream.feed(json.loads(data), time.monotonic() - began)
        record.update(stream.finish(done, time.monotonic() - began,
                                    case["input_tokens"], case["max_new_tokens"]))
        record["arrival_ttft_s"] = record["queue_delay_s"] + record["ttft_s"]
        record["arrival_e2e_s"] = record["queue_delay_s"] + record["e2e_s"]
    except Exception as error:
        message = str(error)
        if isinstance(error, HTTPError):
            message += ": " + error.read(1500).decode(errors="replace")
        record.update({
            "error": message, "output_ids": stream.output_ids,
            "raw_output_ids": stream.raw_output_ids,
            "prompt_lookback_dropped_ids": stream.prompt_lookback_dropped_ids,
            "prompt_lookback_dropped_count": len(stream.prompt_lookback_dropped_ids),
            "actual_output_tokens": len(stream.output_ids), "events": stream.events,
            "finish_reason": stream.meta.get("finish_reason"),
            "cached_tokens": stream.meta.get("cached_tokens"), "sse_done": done,
            "e2e_s": time.monotonic() - began,
        })
    record["completed_s"] = time.monotonic() - trial_start
    return record


def aggregate(requests, elapsed):
    successful = [record for record in requests if record["error"] is None]
    tokens = sum(record["actual_output_tokens"] for record in successful)
    result = {
        "requested": len(requests), "completed": len(successful),
        "failed": len(requests) - len(successful), "wall_s": elapsed,
        "actual_output_tokens": tokens,
        "input_tokens": sum(record["input_tokens"] for record in successful),
        "output_tokens_per_s": tokens / elapsed if elapsed > 0 else 0.0,
        "requests_per_s": len(successful) / elapsed if elapsed > 0 else 0.0,
        "cache_hit_requests": sum(record["cached_tokens"] > 0 for record in successful),
        "cached_tokens": sum(record["cached_tokens"] for record in successful),
    }
    for key in ("ttft_s", "tpot_s", "e2e_s", "queue_delay_s", "arrival_ttft_s", "arrival_e2e_s"):
        result[key] = distribution([record[key] for record in successful])
    if requests:
        denominator = len(requests)
        result["slo"] = {
            "ttft_target_s": 2.0, "tpot_target_s": 0.05,
            "scope": "HTTP request start; client queue delay excluded",
            "denominator": "all requests; failures count as SLO misses",
            "ttft_met_fraction": sum(r["ttft_s"] <= 2 for r in successful) / denominator,
            "tpot_met_fraction": sum(r["tpot_s"] <= 0.05 for r in successful) / denominator,
            "both_met_fraction": sum(r["ttft_s"] <= 2 and r["tpot_s"] <= 0.05
                                     for r in successful) / denominator,
            "arrival_ttft_met_fraction": sum(r["arrival_ttft_s"] <= 2
                                             for r in successful) / denominator,
            "arrival_both_met_fraction": sum(r["arrival_ttft_s"] <= 2 and r["tpot_s"] <= 0.05
                                             for r in successful) / denominator,
            "arrival_scope": "seeded scheduled arrival; client queue delay included",
        }
    return result


def validate_cache_requests(records, shared_prefix_tokens=1792):
    """Check actual reuse at page size 1, alongside repeated-token consistency."""
    seen = {}
    for record in records:
        if record["error"] is not None:
            continue
        case_id = record["fixture_id"]
        expected = shared_prefix_tokens if case_id == "cache-branch" else 0
        errors = []
        if case_id in seen:
            expected = max(expected, record["input_tokens"] - 1)
            if record["output_ids"] != seen[case_id]:
                errors.append("repeated cached prompt changed output_ids")
        else:
            seen[case_id] = record["output_ids"]
        record["required_cached_tokens"] = expected
        if record["cached_tokens"] < expected:
            errors.append(f"cached_tokens {record['cached_tokens']} < required {expected}")
        if errors:
            record["error"] = "; ".join(errors)


def run(args):
    raw_fixture = Path(args.fixtures).read_bytes()
    fixture = json.loads(raw_fixture)
    require(fixture.get("schema_version") == SCHEMA_VERSION, "unsupported fixture schema")
    cases, order = fixture["cases"], list(fixture["suites"][args.suite])
    require(args.suite != "cache" or args.order == "fixture", "cache suite requires fixture order")
    if args.order == "reverse":
        order.reverse()
    elif args.order == "shuffle":
        random.Random(args.seed).shuffle(order)
    for case_id in order:
        case = cases[case_id]
        require(len(case["input_ids"]) == case["input_tokens"], f"invalid input length: {case_id}")
        require(case["input_tokens"] + case["max_new_tokens"] <= 8192,
                f"context budget exceeded: {case_id}")
        require(digest_bytes(json.dumps(case["input_ids"]).encode()) == case["input_sha256"],
                f"input hash mismatch: {case_id}")
    concurrency = 1 if args.suite in ("serial", "cache", "warmup") else args.concurrency
    # The detokenizer retains finished request states, so separate invocations
    # need distinct IDs. A shared prefix preserves all within-trial tie breaks.
    request_namespace = uuid.uuid4().hex
    report = {
        "schema_version": SCHEMA_VERSION, "suite": args.suite,
        "base_url": args.base_url.rstrip("/"), "fixture_sha256": digest_bytes(raw_fixture),
        "fixtures": str(Path(args.fixtures).resolve()), "seed": args.seed,
        "repeat": args.repeat, "concurrency": concurrency, "request_rate": args.request_rate,
        "cache_reset": False, "order_mode": args.order, "request_order": order,
        "request_namespace": request_namespace,
        "request_id_scheme": "namespace:a100-suite-rREPEAT-nINDEX-fixture_id",
        "client_source_sha256": digest_bytes(Path(__file__).read_bytes()),
        "started_at": datetime.now(timezone.utc).isoformat(), "requests": [], "trials": [],
        "timing_definition": {
            "ttft_s": "HTTP request start to first SSE event containing output token IDs",
            "e2e_s": "HTTP request start through [DONE] and response close",
            "tpot_s": "(last token event time - first token event time)/(actual output tokens - 1)",
            "events": "SSE event receipt times and incremental token counts; not per-token latencies",
            "queue_delay_s": "actual HTTP start minus seeded scheduled arrival",
            "arrival_ttft_s": "seeded scheduled arrival to first output token event, including client queue",
            "arrival_e2e_s": "seeded scheduled arrival through request completion, including client queue",
            "note": "TPOT is client-observed; multiple-token events limit timestamp resolution.",
        },
    }
    for repeat_index in range(args.repeat):
        rng = random.Random(args.seed + repeat_index)
        offsets = [0.0]
        for _ in order[1:]:
            offsets.append(offsets[-1] + rng.expovariate(args.request_rate)
                           if args.request_rate > 0 else 0.0)
        trial_start = time.monotonic()
        records = []
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = []
            for index, case_id in enumerate(order):
                case = cases[case_id]
                record = {
                    "request_id": f"{request_namespace}:a100-{args.suite}-r{repeat_index:02d}-n{index:03d}-{case_id}",
                    "fixture_id": case_id, "repeat_index": repeat_index, "request_index": index,
                    "input_tokens": case["input_tokens"],
                    "requested_output_tokens": case["max_new_tokens"], "scheduled_s": offsets[index],
                }
                futures.append(pool.submit(send_request, args.base_url, case, record,
                                           trial_start, args.timeout))
            for future in as_completed(futures):
                records.append(future.result())
        elapsed = time.monotonic() - trial_start
        records.sort(key=lambda record: record["request_index"])
        if args.suite == "cache":
            validate_cache_requests(records)
        report["requests"].extend(records)
        trial = {"repeat_index": repeat_index, "arrival_seed": args.seed + repeat_index,
                 "aggregate": aggregate(records, elapsed)}
        report["trials"].append(trial)
        report["aggregate"] = aggregate(report["requests"], sum(
            item["aggregate"]["wall_s"] for item in report["trials"]))
        report["passed"] = report["aggregate"]["failed"] == 0
        write_json(args.output, report)
        print(json.dumps({"suite": args.suite, "repeat_index": repeat_index,
                          **trial["aggregate"]}), flush=True)
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Tokenize and save fixed workloads (CPU only)")
    prep.add_argument("--model-path", required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--seed", type=int, default=42)
    execute = commands.add_parser("run", help="Run a frozen suite against an existing server")
    execute.add_argument("--fixtures", type=Path, required=True)
    execute.add_argument("--suite", choices=SUITES, required=True)
    execute.add_argument("--base-url", default="http://127.0.0.1:30000")
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--concurrency", type=int, default=16)
    execute.add_argument("--request-rate", type=float, default=0)
    execute.add_argument("--seed", type=int, default=42)
    execute.add_argument("--repeat", type=int, default=1)
    execute.add_argument("--order", choices=("fixture", "reverse", "shuffle"), default="fixture")
    execute.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    if args.command == "run":
        require(args.concurrency > 0 and args.repeat > 0, "concurrency/repeat must be positive")
        require(math.isfinite(args.request_rate) and args.request_rate >= 0,
                "request-rate must be finite and nonnegative")
        require(math.isfinite(args.timeout) and args.timeout > 0, "timeout must be positive")
    return prepare(args) if args.command == "prepare" else run(args)


if __name__ == "__main__":
    sys.exit(main())
