#!/usr/bin/env python3
"""Independent HF teacher-forced check at each candidate's actual token prefix."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import time


def greedy_match_masks(logprobs, chosen, selected_scores, best_scores):
    """Separate canonical argmax equality from membership in an exact top tie."""
    import torch

    argmax_ids = torch.argmax(logprobs, dim=-1)
    return argmax_ids == chosen, selected_scores == best_scores, argmax_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=12)
    parser.add_argument("--fixture-ids", default="")
    parser.add_argument("--gpu", type=int, choices=(6, 7), default=6)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from transformers import AutoModelForCausalLM

    fixture_bytes = args.fixtures.read_bytes()
    fixture_sha256 = hashlib.sha256(fixture_bytes).hexdigest()
    fixtures = json.loads(fixture_bytes)
    inputs = fixtures["cases"]
    selected = set(filter(None, args.fixture_ids.split(",")))
    # Spread the default subset across the reference's serial order.
    first = json.loads(args.results[0].read_text())["requests"]
    if not selected:
        unique = list(dict.fromkeys(r["fixture_id"] for r in first if not r.get("error")))
        step = max(1, len(unique) // args.max_cases)
        selected = set(unique[::step][:args.max_cases])
    if not selected or not selected <= inputs.keys():
        raise ValueError("No valid selected fixture cases")
    result_files = {}
    missing = []
    for path in args.results:
        data = json.loads(path.read_text())
        if data.get("fixture_sha256") != fixture_sha256:
            raise ValueError(f"Fixture SHA256 mismatch: {path}")
        result_files[path] = data
        available = {r["fixture_id"] for r in data["requests"] if not r.get("error")}
        missing.extend({"source": str(path), "fixture_id": key} for key in sorted(selected - available))
    torch.set_num_threads(4)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                attn_implementation="sdpa", local_files_only=True)
    model = model.to("cuda").eval()
    report = {"model": args.model, "torch": torch.__version__, "backend": "HF SDPA BF16",
              "scorer_version": "hf-actual-prefix-v2",
              "scorer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "greedy_equality_semantics": "Canonical torch.argmax over HF logprobs; the first token ID wins exact maximum ties.",
              "max_probability_equality_semantics": "Chosen logprob equals maximum logprob exactly; every exactly tied maximum token counts.",
              "top5_tie_semantics": "torch.topk tie order is not canonical; top5[0] may differ from torch.argmax.",
              "physical_gpu": args.gpu, "fixture_sha256": fixture_sha256,
              "selected_fixtures": sorted(selected), "missing_or_failed_cases": missing,
              "expected_checks": len(selected) * len(args.results),
              "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "checks": [],
              "interpretation": "Diagnostic teacher forcing; token mismatches remain visible, no tolerance-based pass conversion."}
    with torch.inference_mode():
        for path in args.results:
            records = result_files[path]["requests"]
            seen = set()
            for row in records:
                key = row["fixture_id"]
                if key not in selected or key in seen or row.get("error"):
                    continue
                seen.add(key)
                prefix, tokens = inputs[key]["input_ids"], row["output_ids"]
                ids = torch.tensor([prefix + tokens], device="cuda", dtype=torch.long)
                start = time.monotonic()
                logits = model(input_ids=ids, use_cache=False, logits_to_keep=len(tokens) + 1).logits[0, :-1].float()
                assert logits.shape[0] == len(tokens)
                logprobs = logits.log_softmax(-1)
                values, indices = logprobs.topk(5, dim=-1)
                chosen = torch.tensor(tokens, device="cuda")
                selected_scores = logprobs.gather(1, chosen[:, None]).squeeze(1)
                regrets = values[:, 0] - selected_scores
                matches, maximum_matches, argmax_ids = greedy_match_masks(
                    logprobs, chosen, selected_scores, values[:, 0]
                )
                details = []
                for i in (~matches).nonzero().flatten().tolist():
                    details.append({"position": i, "chosen": tokens[i],
                                    "hf_argmax_token": argmax_ids[i].item(),
                                    "chosen_is_max_probability": bool(maximum_matches[i]),
                                    "hf_top5_ids": indices[i].tolist(), "hf_top5_logprobs": values[i].tolist(),
                                    "chosen_logprob": selected_scores[i].item(), "regret": regrets[i].item(),
                                    "top_two_margin": (values[i, 0] - values[i, 1]).item()})
                result = {"source": str(path), "fixture_id": key, "input_tokens": len(prefix),
                          "output_tokens": len(tokens), "hf_greedy_matches": int(matches.sum()),
                          "hf_max_probability_matches": int(maximum_matches.sum()),
                          "all_tokens_hf_max_probability": bool(maximum_matches.all()),
                          "all_tokens_hf_greedy": bool(matches.all()), "max_regret": regrets.max().item(),
                          "mean_regret": regrets.mean().item(), "divergences": details,
                          "elapsed_s": time.monotonic() - start}
                report["checks"].append(result)
                print(f"HF {path.parent.name} {key}: {result['hf_greedy_matches']}/{len(tokens)}, max_regret={result['max_regret']:.6f}", flush=True)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                del ids, logits, logprobs, values, indices, chosen, regrets, matches, maximum_matches, argmax_ids
    report["completed"] = not missing and len(report["checks"]) == report["expected_checks"]
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not report["completed"]:
        raise SystemExit("HF diagnostic coverage incomplete; inspect missing_or_failed_cases")


if __name__ == "__main__":
    main()
