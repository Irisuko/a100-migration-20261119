#!/usr/bin/env python3
"""Summarize exact comparisons and oracle diagnostics without tolerance verdicts.

Run phases must be explicit: --phase-map maps run names to phase labels, or use
--pre-fix-runs / --postfix-runs. Unassigned runs remain unclassified. Diagnostic
completion and coverage never imply output correctness.
"""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    location = (len(ordered) - 1) * p / 100
    low, high = math.floor(location), math.ceil(location)
    return ordered[low] + (ordered[high] - ordered[low]) * (location - low)


def resolve_source(raw, root, report_path):
    path = Path(raw or "")
    if not raw:
        return None
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, Path(__file__).resolve().parents[2] / path,
                  report_path.parent / path, root / path, root / path.parent.name / path.name]
    return next((candidate.resolve() for candidate in candidates if candidate.exists()), candidates[0].resolve())


def load(path, issues, required=False):
    if path is None or not path.exists():
        if required:
            issues.append({"code": "missing_source", "source": str(path)})
        return None
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
        if not isinstance(value, dict):
            if required:
                raise ValueError("expected object")
            return None
        return value, hashlib.sha256(raw).hexdigest()
    except (ValueError, OSError) as error:
        issues.append({"code": "unreadable_source", "source": str(path), "error": str(error)})
        return None


def write_csv(path, rows, first):
    columns = list(dict.fromkeys(first + sorted({key for row in rows for key in row})))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                             if isinstance(value, (list, dict)) else value for key, value in row.items()})


def client_info(source, cache, issues):
    if source not in cache:
        loaded = load(source, issues, required=True)
        cache[source] = loaded[0] if loaded else {}
    return cache[source]


def exact_row(document, report_path, root, phases, cache, issues, digest):
    candidate = resolve_source(document.get("candidate"), root, report_path)
    reference = resolve_source(document.get("reference"), root, report_path)
    run = candidate.parent.name if candidate else report_path.parent.name
    client = client_info(candidate, cache, issues) if candidate else {}
    comparisons = document.get("comparisons", [])
    request_count = document.get("request_count")
    reference_count = document.get("reference_request_count")
    missing, extra = document.get("missing_fixtures", {}), document.get("extra_fixtures", {})
    verified_exact = sum(row.get("exact_match") is True for row in comparisons)
    consistent = len(comparisons) == request_count and verified_exact == document.get("exact_matches")
    coverage = (consistent and bool(request_count) and request_count == reference_count and not missing and not extra)
    if not consistent:
        issues.append({"code": "inconsistent_exact_comparison_counts", "source": str(report_path)})
    return {
        "phase": phases.get(run, "unclassified"), "run": run, "suite": client.get("suite", "unknown"),
        "comparison_source": str(report_path), "comparison_sha256": digest,
        "candidate_source": str(candidate) if candidate else None, "reference_source": str(reference) if reference else None,
        "candidate_requests": request_count, "reference_requests": reference_count,
        "exact_matches": verified_exact, "mismatches": len(comparisons) - verified_exact,
        "exact_match_fraction": verified_exact / len(comparisons) if comparisons else None,
        "full_request_comparison_coverage": coverage,
        "strict_exact_passed": coverage and verified_exact == len(comparisons),
        "saved_passed": document.get("passed"), "missing_fixtures": missing, "extra_fixtures": extra,
        "first_divergences": [{"fixture_id": row.get("fixture_id"), "first_divergence": row.get("first_divergence"),
                               "reference_token": row.get("reference_token"), "candidate_token": row.get("candidate_token"),
                               "error": row.get("error")} for row in comparisons if row.get("exact_match") is not True],
    }


def numerical_row(document, checks, candidate, report_path, kind, phases, cache, issues, digest):
    client = client_info(candidate, cache, issues)
    records = client.get("requests", [])
    expected_ids = {record.get("fixture_id") for record in records}
    if kind == "hf":
        version = document.get("scorer_version", "legacy-topk-v1")
        semantics = document.get("greedy_equality_semantics", "Legacy equality to torch.topk(...)[0]; exact top ties are not canonically ordered.")
        selected_ids = set(document.get("selected_fixtures", []))
        selected_expected = len(selected_ids) if selected_ids else None
        scope = "selected unique fixtures, full candidate continuation at each actual prefix"
    else:
        version = document.get("scorer_version", "oracle-top5-order-v1")
        semantics = "Equality to first returned top-5 token; exact top ties may not match canonical argmax."
        selected_ids = {check.get("fixture_id") for check in checks}
        selected_expected = document.get("selected_mismatch_count")
        scope = "selected mismatching request occurrences only; matching requests are not numerically scored"
    valid = []
    check_rows = []
    all_regrets = []
    source_occurrences = defaultdict(list)
    for record in records:
        source_occurrences[record.get("fixture_id")].append(record)
    for check in checks:
        count = check.get("output_tokens")
        errors = []
        if check.get("error"):
            errors.append(str(check["error"]))
        valid_count = type(count) is int and count > 0
        if not valid_count:
            errors.append("invalid output token count")
        occurrence = check.get("occurrence", 0)
        available = source_occurrences.get(check.get("fixture_id"), [])
        original = available[occurrence] if type(occurrence) is int and 0 <= occurrence < len(available) else None
        if original is None:
            errors.append("source request occurrence missing")
        elif (original.get("error") is not None or count != len(original.get("output_ids", []))
              or count != original.get("actual_output_tokens") or count != original.get("requested_output_tokens")
              or check.get("input_tokens") != original.get("input_tokens")):
            errors.append("check does not cover the complete valid source request")
        if not finite(check.get("max_regret")) or not finite(check.get("mean_regret")):
            errors.append("missing/nonfinite regret")
        details = check.get("divergences", []) if kind == "hf" else check.get("tokens", [])
        positions = [token.get("position") for token in details]
        if len(set(positions)) != len(positions) or any(not valid_count or type(pos) is not int or not 0 <= pos < count for pos in positions):
            errors.append("invalid/duplicate token positions")
        if kind == "oracle" and (not valid_count or positions != list(range(count))):
            errors.append("oracle token coverage is not complete")
        if any(not finite(token.get("regret")) for token in details):
            errors.append("nonfinite per-token regret")
        if errors:
            issues.append({"code": "invalid_numerical_check", "source": str(report_path),
                           "candidate_source": str(candidate), "fixture_id": check.get("fixture_id"), "errors": errors})
        else:
            valid.append(check)
            # HF details omit canonical/top-k equal tokens, whose regret is zero.
            regrets = [0.0] * count
            for token in details:
                regrets[token["position"]] = token["regret"]
            all_regrets.extend(regrets)
        check_rows.append({
            "phase": phases.get(candidate.parent.name, "unclassified"), "kind": kind,
            "run": candidate.parent.name, "suite": client.get("suite", "unknown"),
            "scorer_version": version, "report_source": str(report_path), "candidate_source": str(candidate),
            "fixture_id": check.get("fixture_id"), "occurrence": check.get("occurrence", 0),
            "input_tokens": check.get("input_tokens"), "output_tokens": count,
            "max_regret": check.get("max_regret"), "mean_regret": check.get("mean_regret"),
            "reported_greedy_matches": check.get("hf_greedy_matches", check.get("oracle_greedy_matches")),
            "hf_max_probability_matches": check.get("hf_max_probability_matches"),
            "first_exact_comparison_divergence": check.get("first_divergence"),
            "check_data_complete": not errors, "errors": errors,
        })
    scored_ids = {check.get("fixture_id") for check in valid}
    scored_occurrences = {(check.get("fixture_id"), check.get("occurrence", 0)) for check in valid}
    expected_occurrences = set()
    seen = Counter()
    for record in records:
        key = record.get("fixture_id")
        expected_occurrences.add((key, seen[key]))
        seen[key] += 1
    selected_complete = selected_expected is not None and len(valid) == selected_expected
    if kind == "hf":
        selected_complete = selected_complete and selected_ids <= scored_ids
    mismatch_occurrences, occurrence_counts = set(), Counter()
    for compared in document.get("comparison", {}).get("comparisons", []):
        key = compared.get("fixture_id")
        if not compared.get("exact_match"):
            mismatch_occurrences.add((key, occurrence_counts[key]))
        occurrence_counts[key] += 1
    full_mismatch_coverage = (document.get("full_mismatch_coverage") is True and selected_complete
                              and mismatch_occurrences <= scored_occurrences
                              and len(scored_occurrences) == len(valid)
                              and not document.get("invalid_cases") and not document.get("errors"))
    token_count = sum(check["output_tokens"] for check in valid)
    model = document.get("model", document.get("server", {}).get("model_path"))
    row = {
        "phase": phases.get(candidate.parent.name, "unclassified"), "run": candidate.parent.name,
        "suite": client.get("suite", "unknown"), "kind": kind, "model": model,
        "backend": document.get("backend", document.get("server", {}).get("attention_backend")),
        "scorer_version": version, "scorer_source_sha256": document.get("scorer_source_sha256"),
        "greedy_match_semantics": semantics, "coverage_scope": scope,
        "report_source": str(report_path), "report_sha256": digest, "candidate_source": str(candidate),
        "fixture_sha256": document.get("fixture_sha256", client.get("fixture_sha256")),
        "source_requests": len(records) if records else None, "source_unique_fixtures": len(expected_ids) if records else None,
        "expected_selected_checks": selected_expected, "observed_checks": len(checks), "valid_checks": len(valid),
        "scored_unique_fixtures": len(scored_ids), "scored_request_occurrences": len(scored_occurrences),
        "selected_coverage_complete": selected_complete,
        "full_source_unique_fixture_coverage": bool(expected_ids) and expected_ids <= scored_ids,
        "full_source_request_occurrence_coverage": bool(expected_occurrences) and expected_occurrences <= scored_occurrences,
        "unscored_source_fixtures": sorted(expected_ids - scored_ids),
        "source_report_completed": document.get("completed"),
        "source_report_expected_checks_all_files": document.get("expected_checks"),
        "source_report_observed_checks_all_files": len(document.get("checks", [])),
        "source_report_missing_or_failed_cases": document.get("missing_or_failed_cases", []),
        "full_mismatch_coverage": full_mismatch_coverage if kind == "oracle" else None,
        "declared_full_mismatch_coverage": document.get("full_mismatch_coverage") if kind == "oracle" else None,
        "eligible_mismatch_count": document.get("eligible_mismatch_count"),
        "unselected_mismatch_count": document.get("unselected_mismatch_count"),
        "invalid_cases": document.get("invalid_cases", []), "report_errors": document.get("errors", []),
        "scored_tokens": token_count,
        "max_regret": max((check["max_regret"] for check in valid), default=None),
        "token_weighted_mean_regret": sum(check["mean_regret"] * check["output_tokens"] for check in valid) / token_count if token_count else None,
        "token_regret_p50": percentile(all_regrets, 50), "token_regret_p95": percentile(all_regrets, 95),
        "token_regret_p99": percentile(all_regrets, 99),
        "reported_greedy_matches": sum(check.get("hf_greedy_matches", check.get("oracle_greedy_matches", 0)) for check in valid),
        "hf_max_probability_matches": sum(check["hf_max_probability_matches"] for check in valid)
                                      if valid and all("hf_max_probability_matches" in check for check in valid) else None,
    }
    if not selected_complete or document.get("completed") is not True:
        issues.append({"code": "incomplete_numerical_coverage", "source": str(report_path),
                       "candidate_source": str(candidate), "selected_coverage_complete": selected_complete,
                       "source_report_completed": document.get("completed")})
    return row, check_rows


def summarize(root, output_dir, phases):
    exact, numerical, details, issues, cache = [], [], [], [], {}
    report_coverage = []
    candidates = sorted(set(root.glob("*.json")) | set(root.glob("*/*.json")))
    for path in candidates:
        if output_dir in path.parents:
            continue
        loaded = load(path, issues)
        if loaded is None:
            continue
        document, digest = loaded
        if isinstance(document.get("comparisons"), list) and "exact_matches" in document:
            exact.append(exact_row(document, path, root, phases, cache, issues, digest))
        elif isinstance(document.get("checks"), list) and document.get("oracle"):
            report_coverage.append({
                "kind": "oracle", "report_source": str(path), "report_sha256": digest,
                "expected_selected_checks": document.get("selected_mismatch_count"),
                "observed_checks": len(document["checks"]), "declared_completed": document.get("completed"),
                "declared_full_mismatch_coverage": document.get("full_mismatch_coverage"),
                "unselected_mismatch_count": document.get("unselected_mismatch_count"),
            })
            candidate = resolve_source(document.get("candidate"), root, path)
            if candidate is None:
                issues.append({"code": "oracle_candidate_missing", "source": str(path)})
                continue
            row, check_rows = numerical_row(document, document["checks"], candidate, path, "oracle", phases, cache, issues, digest)
            numerical.append(row)
            details.extend(check_rows)
        elif isinstance(document.get("checks"), list) and (
                document.get("backend", "").startswith("HF") or document.get("scorer_version", "").startswith("hf-")):
            expected, observed = document.get("expected_checks"), len(document["checks"])
            declared_complete = document.get("completed") is True
            full_selected = declared_complete and expected == observed and not document.get("missing_or_failed_cases")
            report_coverage.append({
                "kind": "hf", "report_source": str(path), "report_sha256": digest,
                "scorer_version": document.get("scorer_version", "legacy-topk-v1"),
                "expected_selected_checks": expected, "observed_checks": observed,
                "declared_completed": document.get("completed"),
                "full_selected_report_coverage": full_selected,
                "selected_fixtures": document.get("selected_fixtures", []),
                "missing_or_failed_cases": document.get("missing_or_failed_cases", []),
            })
            if not full_selected:
                issues.append({"code": "incomplete_hf_report", "source": str(path),
                               "expected_selected_checks": expected, "observed_checks": observed})
            groups = defaultdict(list)
            for check in document["checks"]:
                source = resolve_source(check.get("source"), root, path)
                if source is not None:
                    groups[source].append(check)
                else:
                    issues.append({"code": "hf_candidate_missing", "source": str(path), "fixture_id": check.get("fixture_id")})
            for missing in document.get("missing_or_failed_cases", []):
                source = resolve_source(missing.get("source"), root, path)
                if source is not None:
                    groups.setdefault(source, [])
            for source, checks in sorted(groups.items()):
                row, check_rows = numerical_row(document, checks, source, path, "hf", phases, cache, issues, digest)
                numerical.append(row)
                details.extend(check_rows)
    stage_rows = []
    groups = defaultdict(list)
    for row in exact:
        groups[(row["phase"], row["suite"])].append(row)
    for (phase, suite), rows in groups.items():
        stage_rows.append({
            "phase": phase, "kind": "exact", "suite": suite, "comparison_reports": len(rows),
            "candidate_files": len({row["candidate_source"] for row in rows}),
            "request_comparison_observations": sum(row["candidate_requests"] or 0 for row in rows),
            "exact_match_observations": sum(row["exact_matches"] for row in rows),
            "mismatch_observations": sum(row["mismatches"] for row in rows),
            "all_full_request_comparison_coverage": all(row["full_request_comparison_coverage"] for row in rows),
            "failed_exact_comparison_reports": sum(not row["strict_exact_passed"] for row in rows),
            "coverage_scope": "Identified comparisons only; repeated references to one candidate remain comparison observations.",
            "sources": [row["comparison_source"] for row in rows],
        })
    groups.clear()
    for row in numerical:
        groups[(row["phase"], row["kind"], row["suite"], row["scorer_version"], row["model"], row["backend"], row["fixture_sha256"])].append(row)
    for (phase, kind, suite, version, model, backend, fixture_hash), rows in groups.items():
        tokens = sum(row["scored_tokens"] for row in rows)
        stage_rows.append({
            "phase": phase, "kind": kind, "suite": suite, "scorer_version": version,
            "model": model, "backend": backend, "fixture_sha256": fixture_hash,
            "reports": len({row["report_source"] for row in rows}),
            "candidate_files": len({row["candidate_source"] for row in rows}),
            "check_observations": sum(row["valid_checks"] for row in rows), "token_observations": tokens,
            "max_regret": max((row["max_regret"] for row in rows if finite(row["max_regret"])), default=None),
            "token_weighted_mean_regret": sum(row["token_weighted_mean_regret"] * row["scored_tokens"]
                                               for row in rows if row["scored_tokens"]) / tokens if tokens else None,
            "all_selected_coverage_complete": all(row["selected_coverage_complete"] for row in rows),
            "all_full_source_request_coverage": all(row["full_source_request_occurrence_coverage"] for row in rows),
            "coverage_scope": "Per-row selected diagnostic coverage; repeated scoring of one candidate remains repeated observations, not new requests.",
            "sources": sorted({row["report_source"] for row in rows}),
        })
    return {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
            "root": str(root), "phase_map": phases, "exact_comparisons": exact,
            "numerical_reports": numerical, "numerical_checks": details, "stage_summary": stage_rows,
            "diagnostic_report_coverage": report_coverage,
            "issues": issues,
            "interpretation": [
                "Only strict_exact_passed is a correctness verdict, and only for the identified exact comparison.",
                "No numerical regret threshold converts mismatches into passes.",
                "Diagnostic completion, selected coverage, full unique-fixture coverage, and full request-occurrence coverage are separate fields.",
                "Legacy HF greedy counts use topk tie order; v2 canonical greedy and exact-max-probability counts remain separately labeled.",
                "Ordinary SGLang oracle checks cover selected mismatches, not an independent full-suite HF validation.",
                "Stage aggregates retain repeated scoring as repeated observations and separate scorer versions, models, backends, suites, and fixture hashes.",
            ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("run/a100"))
    parser.add_argument("--output-dir", type=Path, default=Path("run/a100/correctness-data"))
    parser.add_argument("--phase-map", type=Path, help="JSON object mapping exact run names to phase labels")
    parser.add_argument("--pre-fix-runs", nargs="*", default=[])
    parser.add_argument("--post-fix-runs", "--postfix-runs", nargs="*", default=[], dest="postfix_runs")
    args = parser.parse_args()
    root, output_dir = args.root.resolve(), args.output_dir.resolve()
    if not root.is_dir():
        parser.error(f"root does not exist: {root}")
    phases = json.loads(args.phase_map.read_text()) if args.phase_map else {}
    if not isinstance(phases, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in phases.items()):
        parser.error("phase-map must map run-name strings to phase-label strings")
    for name in args.pre_fix_runs:
        phases[name] = "pre_fix"
    for name in args.postfix_runs:
        phases[name] = "post_fix"
    output_dir.mkdir(parents=True, exist_ok=True)
    report = summarize(root, output_dir, phases)
    (output_dir / "correctness-summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    for key in ("exact_comparisons", "numerical_reports", "numerical_checks", "stage_summary", "diagnostic_report_coverage"):
        write_csv(output_dir / (key.replace("_", "-") + ".csv"), report[key], ["phase", "run", "suite"])
    (output_dir / "issues.json").write_text(json.dumps(report["issues"], indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output_dir": str(output_dir), "exact_comparisons": len(report["exact_comparisons"]),
                      "numerical_reports": len(report["numerical_reports"]), "checks": len(report["numerical_checks"]),
                      "issues": len(report["issues"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
