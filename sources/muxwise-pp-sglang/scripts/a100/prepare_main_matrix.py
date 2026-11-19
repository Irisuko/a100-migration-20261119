#!/usr/bin/env python3
"""Freeze final comparisons after the complete pilot, without starting GPUs."""
import hashlib
import json
from pathlib import Path
import random

import yaml


def main():
    repo = Path(__file__).resolve().parents[2]
    root = repo / "run/a100"
    selection = json.loads((root / "pilot-selection.json").read_text())
    if not selection.get("completed") or len(selection["rows"]) != 8:
        raise ValueError("A complete eight-configuration pilot selection is required")
    chunk = selection["winner"]["chunk_tokens"]
    budget = selection["winner"]["slice_token_layer_budget"]
    base_path = Path("configs/pdmux_pp_a100.yml")
    base = yaml.safe_load((repo / base_path).read_text())
    if base["split_forward_token_budget"] != 32768:
        raise ValueError("The whole-prefill reference slice budget must remain 32768")
    cpp = dict(base, split_forward_token_budget=budget)
    cpp_path = Path("configs/pdmux_pp_a100_cpp.yml")
    (repo / cpp_path).write_text(
        f"# Pilot-selected CPP profile; pass --pdmux-pp-prefill-chunk-size {chunk}.\n"
        "# Same A100 calibration as pdmux_pp_a100.yml; see run/a100/pilot-selection.json.\n"
        + yaml.safe_dump(cpp, sort_keys=False))
    ablations = {}
    for name, key in (("no-prefetch", "pp_prefetch"), ("no-async", "pp_async_decode"),
                      ("no-calibration", "pp_profile_decode")):
        path = Path(f"run/a100/configs/main-{name}.yml")
        (repo / path).write_text(yaml.safe_dump(dict(cpp, **{key: False}), sort_keys=False))
        ablations[name] = path
    common = ["--purpose", "main", "--suites", "warmup,perf,serial,cache,concurrent", "--repetitions", "3",
              "--correctness-repetitions", "1", "--concurrency", "32"]
    jobs = []

    def add(name, args, open_loop=False, extra=()):
        arguments = ["--name", name, *args, *common]
        if open_loop:
            arguments += ["--also-open-loop-rate", "8"]
        if extra:
            arguments += ["--extra-args", *extra]
        jobs.append({"args": arguments})

    add("main-baseline-whole", ["--mode", "baseline", "--baseline-chunk", "-1"])
    add("main-baseline-chunk2048", ["--mode", "baseline", "--baseline-chunk", "2048"], True)
    add("main-static", ["--mode", "mux", "--policy", "static", "--config", str(base_path)])
    add("main-alp-whole", ["--mode", "mux", "--policy", "alp", "--config", str(base_path)])
    cpp_args = ["--mode", "mux", "--policy", "alp", "--chunk", str(chunk), "--config", str(cpp_path)]
    add("main-cpp", cpp_args, True)
    if budget != 32768:
        add("main-cpp-budget32768", ["--mode", "mux", "--policy", "alp", "--chunk", str(chunk),
                                    "--config", str(base_path)])
    for name, path in ablations.items():
        add("main-" + name, ["--mode", "mux", "--policy", "alp", "--chunk", str(chunk),
                            "--config", str(path)])
    add("main-cpu-metadata", cpp_args, True, ["--pdmux-pp-cpu-prefill-metadata"])
    random.Random(20261114).shuffle(jobs)
    (root / "main-jobs.json").write_text(json.dumps(jobs, indent=2) + "\n")
    trace_jobs = []
    for name, path in (("trace-cpp", cpp_path), ("trace-no-prefetch", ablations["no-prefetch"])):
        trace_jobs.append({"args": ["--name", name, "--purpose", "diagnostic", "--mode", "mux", "--policy", "alp", "--chunk", str(chunk),
                                    "--config", str(path), "--fixtures", "run/a100/fixtures-trace.json",
                                    "--suites", "warmup,pilot", "--concurrency", "8", "--request-rate", "8",
                                    "--profile", "--trace"]})
    (root / "trace-jobs.json").write_text(json.dumps(trace_jobs, indent=2) + "\n")
    protocol = {
        "pilot_selection_sha256": hashlib.sha256((root / "pilot-selection.json").read_bytes()).hexdigest(),
        "selected_chunk": chunk, "selected_slice_budget": budget,
        "job_order_seed": 20261114, "main_runs": [j["args"][1] for j in jobs],
        "controlled_comparisons": {
            "partition_policy": ["main-static", "main-alp-whole"],
            "chunk_only": ["main-alp-whole", "main-cpp" if budget == 32768 else "main-cpp-budget32768"],
            "slice_budget": None if budget == 32768 else ["main-cpp-budget32768", "main-cpp"],
            "prefetch": ["main-cpp", "main-no-prefetch"],
            "async_decode": ["main-cpp", "main-no-async"],
            "startup_calibration": ["main-cpp", "main-no-calibration"],
            "cpu_prefill_metadata": ["main-cpp", "main-cpu-metadata"],
        },
        "training_history": "Identical warmup, then 3 saturation trials; optional 3 open-loop trials immediately follow; correctness suites run after performance.",
        "scope": "Three within-server repeats, fixed 64-request burst/drain workloads; no statistical significance or global optimality claim.",
    }
    (root / "main-protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps({"main_runs": len(jobs), "trace_runs": len(trace_jobs), "chunk": chunk, "budget": budget}))


if __name__ == "__main__":
    main()
