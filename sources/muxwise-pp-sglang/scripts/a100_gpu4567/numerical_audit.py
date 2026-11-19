#!/usr/bin/env python3
"""Teacher-force provided continuations through an ordinary local endpoint.

audit(port, cases_path, output_path, expected_metadata=None) returns a report.
completed/full_token_coverage certify diagnostic coverage only, never model
correctness. Expected model_path, tp_size and pp_size are mandatory; if not
passed, they are read from cases_path.with_suffix('.metadata.json').
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request


LEGACY = Path(__file__).resolve().parents[1] / "a100"


def load_legacy_parser():
    """Load two stdlib-only legacy modules without changing import search paths."""
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    comparison = load("_a100_numerical_compare", LEGACY / "compare_outputs.py")
    previous = sys.modules.get("compare_outputs")
    sys.modules["compare_outputs"] = comparison
    try:
        return load("_a100_numerical_legacy", LEGACY / "diagnose_differences.py")
    finally:
        if previous is None:
            sys.modules.pop("compare_outputs", None)
        else:
            sys.modules["compare_outputs"] = previous


_legacy = load_legacy_parser()
parse_logprobs = _legacy.parse_logprobs


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
    # Invalid API floats are still preserved losslessly as explicit tagged
    # values while the containing audit remains standards-compliant JSON.
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_float": repr(value)}
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def persist(path, report):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(json_safe(report), stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(stream.name, path)


def request_json(port, path, payload=None):
    """No environment proxy; preserve complete HTTP errors and non-JSON bodies."""
    require(type(port) is int and 1 <= port <= 65535, "Invalid local server port")
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload, allow_nan=False).encode() if payload is not None else None,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST" if payload is not None else "GET")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=300) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        return {"_http_error_status": error.code, "_http_error_body": error.read().decode("utf-8", errors="replace")}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"_invalid_json_body": raw}


def validate_api_response(value):
    require(isinstance(value, dict), "API response must be a JSON object")
    require("_http_error_status" not in value, f"HTTP error: {value.get('_http_error_status')}")
    require("_invalid_json_body" not in value, "API returned a non-JSON body")
    require(not value.get("error"), f"API error: {value.get('error')}")


def normalize_dtype(value):
    return {"bf16": "bfloat16", "torch.bfloat16": "bfloat16"}.get(value, value)


def validate_server(server, expected):
    validate_api_response(server)
    require(server.get("enable_pdmux") is False, "Ordinary oracle must confirm enable_pdmux=false")
    require(Path(server.get("model_path", "")).resolve() == Path(expected["model_path"]).resolve(),
            "Oracle model_path differs from expected model")
    for key in ["tp_size", "pp_size"]:
        require(type(server.get(key)) is int and server[key] == expected[key], f"Oracle {key} differs from expected topology")
    require(normalize_dtype(server.get("dtype")) == normalize_dtype(expected.get("dtype", "bfloat16")),
            "Oracle dtype differs from expected BF16 precision")
    if "dp_size" in expected:
        require(server.get("dp_size") == expected["dp_size"], "Oracle dp_size differs from expected topology")


def validate_cases(cases, cases_path):
    require(isinstance(cases, list) and cases, "Cases must be a nonempty list")
    labels, hashes = set(), {}
    for case in cases:
        require(isinstance(case, dict), "Each case must be an object")
        label = case.get("label")
        require(isinstance(label, str) and label and label not in labels, "Case labels must be nonempty and unique")
        labels.add(label)
        for key in ["input_ids", "output_ids"]:
            ids = case.get(key)
            require(isinstance(ids, list) and ids and all(type(token) is int and 0 <= token < 128256 for token in ids),
                    f"{label}: invalid {key}")
        for field, actual in [("input_tokens", len(case["input_ids"])), ("output_tokens", len(case["output_ids"])),
                              ("requested_output_tokens", len(case["output_ids"]))]:
            if field in case:
                require(case[field] == actual, f"{label}: {field} coverage mismatch")
        mismatch = case.get("first_mismatch")
        if mismatch is not None and type(mismatch) is int:
            require(0 <= mismatch < len(case["output_ids"]), f"{label}: first_mismatch outside continuation")
        source_value = case.get("source")
        require(isinstance(source_value, str) and source_value, f"{label}: source file required")
        source = Path(source_value)
        if not source.is_absolute():
            source = cases_path.parent / source
        source = source.resolve()
        expected_hash = case.get("source_sha256", "")
        require(isinstance(expected_hash, str) and re.fullmatch(r"[0-9a-f]{64}", expected_hash), f"{label}: source SHA256 required")
        if source not in hashes:
            hashes[source] = sha256(source)
        require(hashes[source] == expected_hash, f"{label}: source hash mismatch: {source}")


def tie_details(parsed):
    for token in parsed["tokens"]:
        best = token["oracle_best_logprob"]
        ties = sorted(token_id for token_id, logprob in zip(token["oracle_top5_ids"], token["oracle_top5_logprobs"]) if logprob == best)
        canonical = min(ties)
        token.update(reported_max_tie_ids=ties, canonical_top5_token=canonical,
                     canonical_top5_match=token["chosen"] == canonical,
                     chosen_in_reported_max_tie=token["chosen_logprob"] == best,
                     all_max_ties_in_top5=token["oracle_top5_logprobs"][-1] < best,
                     negative_regret_observed=token["regret"] < 0)
    parsed["canonical_top5_matches"] = sum(token["canonical_top5_match"] for token in parsed["tokens"])
    parsed["reported_max_tie_tokens"] = sum(len(token["reported_max_tie_ids"]) > 1 for token in parsed["tokens"])
    return parsed


def audit(port, cases_path, output_path, expected_metadata=None):
    cases_path, output_path = Path(cases_path).resolve(), Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Reserve output before any network call and never overwrite another audit.
    with output_path.open("x"):
        pass
    report = dict(schema_version=1, status="running", completed=False, full_token_coverage=False,
                  oracle="ordinary SGLang same-model/topology teacher forcing", port=port,
                  cases_path=str(cases_path), started_at_utc=datetime.now(timezone.utc).isoformat(),
                  selected_cases=0, requested_output_tokens=0, audited_output_tokens=0,
                  checks=[], errors=[],
                  interpretation="Diagnostic coverage is not model correctness. Original exact mismatches remain mismatches. No epsilon, tolerance, tie membership, or low regret converts a mismatch into PASS. The ordinary endpoint is not an independent HF oracle.",
                  tie_definition="Canonical top5 token is the smallest ID among alternatives with exactly equal returned maximum logprob. If the fifth alternative ties the maximum, the complete tie set is unknown. Equal returned logprobs can reflect output precision; no tolerance is used.",
                  parser=dict(path=str(LEGACY / "diagnose_differences.py"), sha256=sha256(LEGACY / "diagnose_differences.py")))
    persist(output_path, report)
    try:
        require(type(port) is int and 1 <= port <= 65535, "Invalid local server port")
        report["cases_sha256"] = sha256(cases_path)
        cases = json.loads(cases_path.read_text())
        validate_cases(cases, cases_path)
        report["selected_cases"] = len(cases)
        report["requested_output_tokens"] = sum(len(case["output_ids"]) for case in cases)
        report["original_mismatch_cases"] = sum(case.get("first_mismatch") is not None for case in cases)
        if expected_metadata is None:
            metadata_path = cases_path.with_suffix(".metadata.json")
            expected_metadata = json.loads(metadata_path.read_text())
            report["expected_metadata_file"] = str(metadata_path)
            report["expected_metadata_sha256"] = sha256(metadata_path)
        require(isinstance(expected_metadata, dict), "Expected metadata object required")
        require(isinstance(expected_metadata.get("model_path"), str) and expected_metadata["model_path"], "Expected model_path required")
        require(all(type(expected_metadata.get(key)) is int and expected_metadata[key] > 0 for key in ["tp_size", "pp_size"]),
                "Expected tp_size and pp_size required")
        require(expected_metadata.get("enable_pdmux", False) is False, "Expected oracle must be ordinary (enable_pdmux=false)")
        report["expected_metadata"] = expected_metadata
        server = request_json(port, "/get_server_info")
        report["server_response"] = server
        persist(output_path, report)
        validate_server(server, expected_metadata)
        context = server.get("context_length")
        if context is not None:
            require(type(context) is int and context > 0, "Invalid oracle context length")
            require(all(len(case["input_ids"]) + len(case["output_ids"]) + 1 <= context for case in cases),
                    "Teacher-forced continuation plus one token exceeds oracle context")
        for case in cases:
            prompt, continuation = case["input_ids"], case["output_ids"]
            check = dict(case=case, label=case["label"], input_tokens=len(prompt), output_tokens=len(continuation),
                         original_first_mismatch=case.get("first_mismatch"), error=None, audited_tokens=0)
            report["checks"].append(check)
            started = time.monotonic()
            try:
                payload = dict(input_ids=prompt + continuation, return_logprob=True,
                               logprob_start_len=len(prompt) - 1, top_logprobs_num=5,
                               sampling_params=dict(max_new_tokens=1, ignore_eos=True, temperature=0), stream=False)
                check["request"] = payload
                response = request_json(port, "/generate", payload)
                check["raw_response"] = response
                persist(output_path, report)
                validate_api_response(response)
                require(response.get("meta_info", {}).get("completion_tokens") == 1,
                        "Oracle must generate exactly one requested probe token")
                parsed = tie_details(parse_logprobs(response, prompt, continuation))
                check.update(parsed, audited_tokens=len(parsed["tokens"]))
                mismatch = case.get("first_mismatch")
                if type(mismatch) is int:
                    check["original_first_mismatch_diagnostic"] = parsed["tokens"][mismatch]
                report["audited_output_tokens"] += len(parsed["tokens"])
            except Exception as error:
                check["error"] = f"{type(error).__name__}: {error}"
                report["errors"].append(f"{case['label']}: {check['error']}")
            check["elapsed_sec"] = time.monotonic() - started
            persist(output_path, report)
        report["completed"] = len(report["checks"]) == len(cases) and not report["errors"]
        report["full_token_coverage"] = report["completed"] and report["audited_output_tokens"] == report["requested_output_tokens"]
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    report["status"] = "diagnostic_complete" if report["completed"] else "diagnostic_incomplete"
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    persist(output_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-metadata", type=Path)
    args = parser.parse_args()
    expected = json.loads(args.expected_metadata.read_text()) if args.expected_metadata else None
    result = audit(args.port, args.cases, args.output, expected)
    print(json.dumps({key: result[key] for key in ["status", "completed", "full_token_coverage", "audited_output_tokens", "requested_output_tokens"]}))
    return 0 if result["completed"] and result["full_token_coverage"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
