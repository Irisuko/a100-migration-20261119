#!/usr/bin/env python3
"""Diagnose token differences against an ordinary SGLang baseline over HTTP.

This teacher-forces each candidate's actual continuation. It explains differences;
it never changes the exact-match result or applies a tolerance-based pass rule.
Exit zero means the selected diagnostic checks completed, not correctness passed.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request

from compare_outputs import compare


def require(condition, message):
    if not condition:
        raise ValueError(message)


def token_ids(value):
    return isinstance(value, list) and bool(value) and all(
        type(token) is int and token >= 0 for token in value
    )


def load_inputs(fixtures_path, reference_path, candidate_path):
    raw = fixtures_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    fixtures = json.loads(raw)
    require(isinstance(fixtures.get("cases"), dict), "Missing fixture cases")
    results = []
    for path in (reference_path, candidate_path):
        data = json.loads(path.read_text())
        require(data.get("fixture_sha256") == digest, f"Fixture SHA256 mismatch: {path}")
        require(isinstance(data.get("requests"), list), f"Missing requests: {path}")
        for row in data["requests"]:
            require(isinstance(row, dict) and isinstance(row.get("fixture_id"), str)
                    and isinstance(row.get("request_id"), str), f"Invalid request identity: {path}")
            require(row["fixture_id"] in fixtures["cases"],
                    f"Unknown fixture {row['fixture_id']}: {path}")
            require(isinstance(row.get("output_ids"), list) and all(
                type(token) is int and token >= 0 for token in row["output_ids"]
            ), f"Invalid output_ids: {path}")
        results.append(data)
    for key, case in fixtures["cases"].items():
        ids = case.get("input_ids")
        require(token_ids(ids), f"Invalid input_ids: {key}")
        require(case.get("input_tokens") == len(ids), f"Invalid input length: {key}")
        actual_hash = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        require(case.get("input_sha256") == actual_hash, f"Input SHA256 mismatch: {key}")
    return fixtures, results[0], results[1]


def api_json(base_url, path, payload=None):
    """Use no environment proxies; retain HTTP error bodies for unsupported APIs."""
    parsed = urllib.parse.urlsplit(base_url)
    require(parsed.scheme in ("http", "https") and bool(parsed.netloc), "Invalid base URL")
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload, allow_nan=False).encode() if payload is not None else None,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=300) as response:
            data = json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read(8192).decode(errors="replace")
        raise RuntimeError(f"Baseline API {path}: HTTP {error.code}: {body}") from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"Baseline API {path}: {error}") from error
    require(isinstance(data, dict), f"Baseline API {path}: expected a JSON object")
    if data.get("error"):
        raise RuntimeError(f"Baseline API {path}: {data['error']}")
    return data


def parse_logprobs(response, prompt, continuation):
    require(token_ids(prompt) and token_ids(continuation), "Nonempty token IDs required")
    require(isinstance(response, dict) and not response.get("error"),
            f"Baseline API error: {response.get('error') if isinstance(response, dict) else response}")
    meta = response.get("meta_info", {})
    require(isinstance(meta, dict), "Baseline API missing meta_info")
    require(meta.get("prompt_tokens") == len(prompt) + len(continuation),
            "Baseline API prompt token count does not match the submitted continuation")
    finish_reason = meta.get("finish_reason") or {}
    require(isinstance(finish_reason, dict) and finish_reason.get("type") != "abort",
            f"Baseline API request aborted: {finish_reason}")
    chosen = meta.get("input_token_logprobs")
    alternatives = meta.get("input_top_logprobs")
    require(isinstance(chosen, list) and isinstance(alternatives, list),
            "Baseline API missing input logprobs; use an ordinary baseline, not PDMux")
    expected = [prompt[-1]] + continuation
    require(len(chosen) == len(alternatives) == len(expected),
            "Input logprob coverage mismatch: expected one anchor plus every continuation token")
    for position, (entry, expected_id) in enumerate(zip(chosen, expected)):
        require(isinstance(entry, (list, tuple)) and len(entry) == 3,
                f"Invalid input logprob tuple at position {position}")
        require(type(entry[1]) is int and entry[1] == expected_id,
                f"Input logprob token alignment mismatch at position {position}")
    require(chosen[0][0] is None and alternatives[0] is None,
            "Input logprobs must begin with a None anchor for prompt[-1]")

    def finite(value):
        return type(value) in (int, float) and math.isfinite(value)

    tokens = []
    for position, (entry, top) in enumerate(zip(chosen[1:], alternatives[1:])):
        require(finite(entry[0]), f"Nonfinite chosen logprob at continuation position {position}")
        require(isinstance(top, list) and len(top) == 5,
                f"Expected five oracle alternatives at continuation position {position}")
        require(all(isinstance(item, (list, tuple)) and len(item) == 3
                    and finite(item[0]) and type(item[1]) is int and item[1] >= 0 for item in top),
                f"Invalid oracle alternatives at continuation position {position}")
        require(len({item[1] for item in top}) == 5
                and all(a[0] >= b[0] for a, b in zip(top, top[1:])),
                f"Oracle alternatives are not unique and ranked at continuation position {position}")
        tokens.append({
            "position": position, "chosen": entry[1], "chosen_logprob": entry[0],
            "oracle_best_token": top[0][1], "oracle_best_logprob": top[0][0],
            "oracle_top5_ids": [item[1] for item in top],
            "oracle_top5_logprobs": [item[0] for item in top],
            "regret": top[0][0] - entry[0], "top_two_margin": top[0][0] - top[1][0],
            "oracle_greedy_match": entry[1] == top[0][1],
        })
    matches = sum(row["oracle_greedy_match"] for row in tokens)
    return {
        "tokens": tokens, "oracle_greedy_matches": matches,
        "all_tokens_oracle_greedy": matches == len(tokens),
        "first_oracle_divergence": next((row["position"] for row in tokens
                                         if not row["oracle_greedy_match"]), None),
        "max_regret": max(row["regret"] for row in tokens),
        "mean_regret": sum(row["regret"] for row in tokens) / len(tokens),
    }


def diagnose(fixtures, reference, candidate, api, max_cases=12):
    require(max_cases > 0, "max_cases must be positive")
    comparison = compare(reference, candidate)
    report = {
        "schema_version": 1, "oracle": "ordinary SGLang baseline /generate",
        "interpretation": "Teacher-forced diagnostic at each candidate's actual prefix; exact token mismatches remain failures, with no tolerance-based pass conversion. This is not an independent HF check.",
        "comparison": comparison, "max_cases": max_cases, "checks": [],
        "invalid_cases": [], "errors": [], "completed": False,
        "full_mismatch_coverage": False,
    }
    expected = defaultdict(list)
    for row in reference["requests"]:
        expected[row["fixture_id"]].append(row)
    occurrences = Counter()
    pending = []
    for index, (actual, compared) in enumerate(zip(candidate["requests"], comparison["comparisons"])):
        key = actual["fixture_id"]
        occurrence = occurrences[key]
        occurrences[key] += 1
        if compared["exact_match"]:
            continue
        base = expected[key][occurrence] if occurrence < len(expected[key]) else None
        identity = {"fixture_id": key, "occurrence": occurrence, "candidate_index": index,
                    "request_id": actual["request_id"],
                    "reference_request_id": base.get("request_id") if base else None,
                    "first_divergence": compared["first_divergence"],
                    "reference_token": compared["reference_token"],
                    "candidate_token": compared["candidate_token"]}
        reasons = []
        if base is None:
            reasons.append("Reference occurrence missing")
        for label, row in (("Reference", base), ("Candidate", actual)):
            if row is None:
                continue
            if row.get("error"):
                reasons.append(f"{label} request error: {row['error']}")
            ids = row.get("output_ids", [])
            if not token_ids(ids) or len(ids) != row.get("requested_output_tokens") or len(ids) != row.get("actual_output_tokens"):
                reasons.append(f"{label} output coverage invalid")
        fixture = fixtures["cases"].get(key)
        if fixture is None or not token_ids(fixture.get("input_ids")):
            reasons.append("Fixture prompt missing or invalid")
        if reasons:
            report["invalid_cases"].append({**identity, "errors": reasons})
        else:
            pending.append((identity, fixture["input_ids"], actual["output_ids"]))
    selected = pending[:max_cases]
    report["eligible_mismatch_count"] = len(pending)
    report["selected_mismatch_count"] = len(selected)
    report["unselected_mismatch_count"] = len(pending) - len(selected)
    report["unselected_mismatches"] = [row[0] for row in pending[max_cases:]]
    if selected:
        try:
            server = api("/get_server_info")
            require(server.get("enable_pdmux") is False,
                    "Baseline API must confirm enable_pdmux=false; PDMux logprob diagnostics are unsupported")
            report["server"] = {key: server[key] for key in (
                "model_path", "dtype", "tp_size", "pp_size", "attention_backend", "enable_pdmux"
            ) if key in server}
        except Exception as error:
            report["errors"].append(f"Baseline API preflight failed: {error}")
            return report
        for identity, prompt, output in selected:
            check = {**identity, "input_tokens": len(prompt), "output_tokens": len(output)}
            start = time.monotonic()
            try:
                response = api("/generate", {
                    "input_ids": prompt + output, "return_logprob": True,
                    "logprob_start_len": len(prompt) - 1, "top_logprobs_num": 5,
                    "sampling_params": {"max_new_tokens": 1, "ignore_eos": True, "temperature": 0},
                    "stream": False,
                })
                check.update(parse_logprobs(response, prompt, output))
                check["error"] = None
            except Exception as error:
                check["error"] = str(error)
                report["errors"].append(f"{identity['fixture_id']} occurrence {identity['occurrence']}: {error}")
            check["elapsed_s"] = time.monotonic() - start
            report["checks"].append(check)
    report["completed"] = bool(comparison["request_count"]) and not (
        report["errors"] or report["invalid_cases"] or comparison["missing_fixtures"]
        or comparison["extra_fixtures"]
    ) and len(report["checks"]) == len(selected)
    report["full_mismatch_coverage"] = report["completed"] and not report["unselected_mismatch_count"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=12)
    args = parser.parse_args()
    metadata = {"started_at": datetime.now(timezone.utc).isoformat(),
                "fixtures": str(args.fixtures.resolve()), "reference": str(args.reference.resolve()),
                "candidate": str(args.candidate.resolve()), "base_url": args.base_url.rstrip("/")}
    try:
        fixtures, reference, candidate = load_inputs(args.fixtures, args.reference, args.candidate)
        report = diagnose(fixtures, reference, candidate,
                          lambda path, payload=None: api_json(args.base_url, path, payload), args.max_cases)
        report["fixture_sha256"] = reference["fixture_sha256"]
    except Exception as error:
        report = {"schema_version": 1, "completed": False, "full_mismatch_coverage": False,
                  "checks": [], "errors": [f"Input validation or diagnostic error: {error}"]}
    report.update(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Diagnostic checks: {len(report['checks'])}; completed={report['completed']}; "
          f"full_mismatch_coverage={report['full_mismatch_coverage']}")
    return 0 if report["completed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
