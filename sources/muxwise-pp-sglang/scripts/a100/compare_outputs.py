#!/usr/bin/env python3
"""Compare actual generated IDs by fixture; never hide token mismatches."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def compare(reference, candidate):
    expected = defaultdict(list)
    for row in reference["requests"]:
        expected[row["fixture_id"]].append(row)
    expected_counts = Counter({key: len(value) for key, value in expected.items()})
    actual_counts = Counter(row["fixture_id"] for row in candidate["requests"])
    missing = dict(expected_counts - actual_counts)
    extra = dict(actual_counts - expected_counts)
    occurrences = Counter()
    rows = []
    for actual in candidate["requests"]:
        key = actual["fixture_id"]
        occurrence = occurrences[key]
        occurrences[key] += 1
        base = expected[key][occurrence] if occurrence < len(expected[key]) else None
        a, b = (base or {}).get("output_ids", []), actual.get("output_ids", [])
        first = next((i for i, pair in enumerate(zip(a, b)) if pair[0] != pair[1]), None)
        if first is None and len(a) != len(b):
            first = min(len(a), len(b))
        valid_lengths = base is not None and all(
            len(row.get("output_ids", [])) == row.get("requested_output_tokens") == row.get("actual_output_tokens")
            for row in (base, actual)
        )
        equal = (base is not None and not base.get("error") and not actual.get("error")
                 and valid_lengths and bool(a) and a == b)
        rows.append({"fixture_id": key, "request_id": actual["request_id"],
                     "exact_match": equal, "reference_available": base is not None,
                     "reference_tokens": len(a), "candidate_tokens": len(b),
                     "first_divergence": first,
                     "equal_prefix_tokens": len(a) if equal else (first or 0),
                     "reference_token": a[first] if first is not None and first < len(a) else None,
                     "candidate_token": b[first] if first is not None and first < len(b) else None,
                     "error": actual.get("error")})
    exact = sum(row["exact_match"] for row in rows)
    return {"request_count": len(rows), "reference_request_count": len(reference["requests"]),
            "missing_fixtures": missing, "extra_fixtures": extra, "exact_matches": exact,
            "exact_match_fraction": exact / len(rows) if rows else None,
            "passed": bool(rows) and not missing and not extra and exact == len(rows), "comparisons": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    reference, candidate = json.loads(args.reference.read_text()), json.loads(args.candidate.read_text())
    if reference["fixture_sha256"] != candidate["fixture_sha256"]:
        raise ValueError("Fixture hashes differ")
    result = compare(reference, candidate)
    result.update(reference=str(args.reference.resolve()), candidate=str(args.candidate.resolve()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Exact token matches {result['exact_matches']}/{result['request_count']}")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
