#!/usr/bin/env python3
"""Validate a local A100 SGLang server using only the Python standard library.

Run from the repository root:
    python scripts/smoke_local.py --base-url http://127.0.0.1:30000

Concurrent requests exercise the overlap workload; successful responses alone do
not prove that the scheduler actually overlapped prefill and decode on the GPUs.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener


def require(condition, message):
    if not condition:
        raise ValueError(message)


def count(value, label, minimum=0):
    require(
        isinstance(value, int)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= minimum,
        f"Invalid {label}: {value!r}",
    )
    return value


class Client:
    def __init__(self, base_url, timeout):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.opener = build_opener(ProxyHandler({}))

    def open(self, path, payload=None):
        body = json.dumps(payload).encode() if payload is not None else None
        request = Request(
            self.base_url + path,
            data=body,
            headers={"Content-Type": "application/json", "Accept": "*/*"},
        )
        try:
            response = self.opener.open(request, timeout=self.timeout)
        except HTTPError as error:
            detail = error.read(1200).decode(errors="replace")
            raise RuntimeError(f"HTTP {error.code} for {path}: {detail}") from error
        require(response.status == 200, f"Unexpected HTTP {response.status}: {path}")
        return response

    def json(self, path, payload=None):
        with self.open(path, payload) as response:
            return json.load(response)

    def generate(self, prompt, max_tokens=64):
        result = self.json(
            "/generate",
            {
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_tokens,
                    "ignore_eos": True,
                },
                "stream": False,
            },
        )
        require(isinstance(result.get("text"), str) and result["text"].strip(),
                "Generation returned empty text")
        metadata = result["meta_info"]
        prompt_tokens = count(metadata.get("prompt_tokens"), "prompt_tokens", 1)
        completion_tokens = count(metadata.get("completion_tokens"), "completion_tokens", 1)
        cached_tokens = count(metadata.get("cached_tokens"), "cached_tokens")
        require(prompt_tokens < 1000, f"Prompt exceeded smoke budget: {prompt_tokens}")
        require(completion_tokens == max_tokens,
                f"Generation stopped early: {completion_tokens}/{max_tokens} tokens")
        finish_reason = metadata.get("finish_reason")
        if finish_reason is not None:
            require(finish_reason.get("type") == "length",
                    f"Unexpected generation finish reason: {finish_reason}")
        require(cached_tokens <= prompt_tokens, "Cache count exceeds prompt length")
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cached_tokens": cached_tokens,
            "text_sample": result["text"][:120],
        }


def validate_usage(usage, max_tokens=64):
    prompt = count(usage.get("prompt_tokens"), "prompt_tokens", 1)
    completion = count(usage.get("completion_tokens"), "completion_tokens", 1)
    total = count(usage.get("total_tokens"), "total_tokens", 1)
    require(total == prompt + completion, "Usage total does not match token counts")
    require(prompt < 1000 and completion <= max_tokens, "Chat exceeded smoke token budget")
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}


def health(client):
    with client.open("/health") as response:
        response.read()
    return {"status": 200}


def models(client, model):
    ids = [item["id"] for item in client.json("/v1/models")["data"]]
    require(model in ids, f"Expected model {model!r}; received {ids!r}")
    return {"models": ids}


def chat(client, model, stream=False):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Write one short sentence about a garden."}],
        "temperature": 0,
        "max_tokens": 64,
        "stream": stream,
    }
    if not stream:
        result = client.json("/v1/chat/completions", payload)
        content = result["choices"][0]["message"]["content"]
        require(isinstance(content, str) and content.strip(), "Chat returned empty text")
        return {"usage": validate_usage(result["usage"]), "text_sample": content[:120]}

    payload["stream_options"] = {"include_usage": True}
    chunks, usage, done, events = [], None, False, 0
    with client.open("/v1/chat/completions", payload) as response:
        require("text/event-stream" in response.headers.get("Content-Type", ""),
                "Streaming response is not an SSE stream")
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                break
            event = json.loads(data)
            require("error" not in event, f"Stream error: {event}")
            events += 1
            if event.get("usage") is not None:
                usage = validate_usage(event["usage"])
            for choice in event.get("choices", []):
                content = choice.get("delta", {}).get("content")
                if content:
                    chunks.append(content)
    content = "".join(chunks)
    require(done, "SSE stream ended without [DONE]")
    require(content.strip(), "SSE stream returned empty text")
    require(usage is not None, "SSE stream omitted requested usage")
    return {"events": events, "usage": usage, "text_sample": content[:120]}


def cache(client):
    prompt = (
        "Cache verification story. "
        + "A small fox walks through a quiet garden beside the river. " * 24
        + "Continue the story in a few sentences:"
    )
    first = client.generate(prompt)
    repeated = client.generate(prompt)
    require(first["prompt_tokens"] == repeated["prompt_tokens"], "Repeated prompt length changed")
    require(repeated["cached_tokens"] >= repeated["prompt_tokens"] - 1,
            "Repeated prompt did not reuse the full available prefix")
    return {"first": first, "repeated": repeated}


def concurrent(client):
    start = time.monotonic()

    def request(index):
        # Alternate initial decode work and later large prefills over four waves.
        delay = (index // 4) * 0.20 + (index % 4) * 0.03
        time.sleep(delay)
        is_long = index % 3 == 0
        prompt = f"Task {index}: "
        if is_long:
            prompt += "The quick brown fox walks through the quiet garden. " * 32
            prompt += "Continue this story with more details:"
        else:
            prompt += "Tell a story about a fox exploring a garden:"
        began = time.monotonic()
        result = client.generate(prompt, max_tokens=64 if is_long else 128)
        return {
            "index": index,
            "kind": "long" if is_long else "short",
            "started_s": round(began - start, 3),
            "elapsed_s": round(time.monotonic() - began, 3),
            **result,
        }

    results, errors = [], []
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(request, index): index for index in range(16)}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as error:
                errors.append({"index": futures[future], "error": str(error)})
    require(not errors, f"Concurrent requests failed: {json.dumps(errors)}")
    return {
        "count": len(results),
        "elapsed_s": round(time.monotonic() - start, 3),
        "requests": sorted(results, key=lambda result: result["index"]),
        "note": "Overlapping client workload; GPU scheduler overlap requires server logs/metrics.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="llama-3.1-8b")
    parser.add_argument("--output", type=Path, default=Path("run/smoke-results.json"))
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    client = Client(args.base_url, args.timeout)
    report = {
        "base_url": client.base_url,
        "model": args.model,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "checks": [],
    }
    checks = [
        ("health", lambda: health(client)),
        ("models", lambda: models(client, args.model)),
        ("chat", lambda: chat(client, args.model)),
        ("chat_stream", lambda: chat(client, args.model, stream=True)),
        ("prompt_cache", lambda: cache(client)),
        ("concurrent_mixed", lambda: concurrent(client)),
    ]
    for name, check in checks:
        began = time.monotonic()
        try:
            entry = {"name": name, "passed": True, "result": check()}
        except Exception as error:
            entry = {"name": name, "passed": False, "error": str(error)}
        entry["elapsed_s"] = round(time.monotonic() - began, 3)
        report["checks"].append(entry)
        print(f"{'PASS' if entry['passed'] else 'FAIL'} {name}: "
              f"{entry.get('error', str(entry['elapsed_s']) + 's')}", flush=True)
        if name == "health" and not entry["passed"]:
            break
    report["passed"] = len(report["checks"]) == len(checks) and all(
        check["passed"] for check in report["checks"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Results: {args.output}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
