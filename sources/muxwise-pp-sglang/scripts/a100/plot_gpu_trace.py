#!/usr/bin/env python3
"""Export a per-stage trace table and labeled-kernel overlap example windows."""

import argparse
import csv
import gzip
import json
from pathlib import Path

from analyze_gpu_trace import analyze_trace, intersection, union


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows, windows = [], []
    for run in args.run:
        proof = json.loads((run / "gpu-trace-validation.json").read_text())
        for stage in proof["stages"]:
            mapped = stage["mapping"]["mapped_kernels"]
            for device, stats in stage["devices"].items():
                rows.append({
                    "run": run.name, "pp_rank": stage["pp_rank"], "cuda_device": device,
                    "total_kernels": stage["kernel_count"], "mapped_kernels": mapped,
                    "mapped_fraction": mapped / stage["kernel_count"],
                    "unmapped_kernels": stage["mapping"]["unmapped_kernels"],
                    "ambiguous_kernels": stage["mapping"]["ambiguous_kernels"],
                    "prefill_kernels": stats["lanes"]["prefill"]["kernel_count"],
                    "decode_kernels": stats["lanes"]["decode"]["kernel_count"],
                    "prefill_union_ms": stats["lanes"]["prefill"]["kernel_union_us"] / 1000,
                    "decode_union_ms": stats["lanes"]["decode"]["kernel_union_us"] / 1000,
                    "prefill_decode_overlap_ms": stats["prefill_decode_overlap_us"] / 1000,
                    "overlap_fraction_of_prefill": stats["overlap_fraction_of_prefill_union"],
                    "overlap_fraction_of_decode": stats["overlap_fraction_of_decode_union"],
                    "generic_stream_overlap_ms": stats["generic_different_stream_overlap_us"] / 1000,
                    "trace": stage["source"], "trace_sha256": stage["source_sha256"],
                    "interpretation": "diagnostic execution overlap; not throughput or SM occupancy",
                })
            pair = stage.get("verified_pair")
            if pair is None:
                continue
            source_path = Path(stage["source"])
            with gzip.open(source_path, "rt") as source:
                data = json.load(source)
            # Reuse the validated attribution rules; retain all evidence only in
            # memory so a later window is not limited to the first 100 kernels.
            analyzed = analyze_trace(data, str(source_path), evidence_limit=stage["kernel_count"])
            selected = [pair[lane]["kernel"] for lane in ("prefill", "decode")]
            start = min(event["ts"] for event in selected) - 250
            end = max(event["ts"] + event["dur"] for event in selected) + 250
            device = str(selected[0]["args"]["device"])
            by_lane = {lane: [] for lane in ("prefill", "decode")}
            methods, graph_correlations = set(), set()
            for item in analyzed["kernel_evidence"]:
                classification = item["classification"]
                if not classification or not item["device_stream_known"] or item["device"] != device:
                    continue
                left = max(start, item["start_us"])
                right = min(end, item["start_us"] + item["duration_us"])
                if right > left:
                    by_lane[classification["lane"]].append([left, right])
                    methods.update(item["methods"])
                    if classification["lane"] == "decode":
                        graph_correlations.add(data["traceEvents"][item["event_index"]].get("args", {}).get("correlation"))
            by_lane = {lane: union(intervals) for lane, intervals in by_lane.items()}
            windows.append({
                "run": run.name, "pp_rank": stage["pp_rank"], "cuda_device": device,
                "source": str(source_path), "source_sha256": stage["source_sha256"],
                "selection": "window around the maximum independently verified overlapping GEMM pair",
                "start_us": start, "end_us": end, "kernel_unions_us": by_lane,
                "overlap_us": intersection(by_lane["prefill"], by_lane["decode"]),
                "attribution_methods": sorted(methods),
                "decode_kernel_correlations": sorted(value for value in graph_correlations if value is not None),
            })
            del data, analyzed
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "gpu-trace-summary.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "gpu-trace-window-data.json").write_text(
        json.dumps({"schema_version": 1, "windows": windows}, indent=2, allow_nan=False) + "\n"
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    figure, axes = plt.subplots(len(windows), 1, figsize=(11, 2.0 * len(windows) + 1.5), squeeze=False)
    colors = {"prefill": "#4477AA", "decode": "#EE7733"}
    for axis, window in zip(axes[:, 0], windows):
        start = window["start_us"]
        for lane, y in (("prefill", 0), ("decode", 1)):
            spans = [((left - start) / 1000, (right - left) / 1000)
                     for left, right in window["kernel_unions_us"][lane]]
            axis.broken_barh(spans, (y + 0.2, 0.6), facecolors=colors[lane])
        for left, right in window["overlap_us"]:
            axis.axvspan((left - start) / 1000, (right - start) / 1000, color="#BBBBBB", alpha=0.35, zorder=0)
        axis.set(yticks=[0.5, 1.5], yticklabels=["Prefill", "Decode"],
                 ylim=(2, 0), xlim=(0, (window["end_us"] - start) / 1000),
                 xlabel="Time within selected window (ms)")
        name = "CPP" if window["run"] == "trace-cpp" else "No prefetch"
        axis.set_title(f"{name} · PP{window['pp_rank']} · CUDA device {window['cuda_device']}", loc="left", fontsize=11)
        axis.grid(axis="x", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Diagnostic GPU kernel overlap: explicitly labeled lanes", fontsize=14, y=0.99)
    figure.legend(handles=[Patch(color=colors["prefill"], label="Prefill kernels"),
                           Patch(color=colors["decode"], label="Decode kernels"),
                           Patch(color="#BBBBBB", alpha=0.35, label="Both lanes executing")],
                  loc="upper center", bbox_to_anchor=(0.5, 0.963), ncol=3, frameon=False)
    figure.text(0.5, 0.012, "Example windows around verified overlapping GEMM pairs; independent clocks per panel.\n"
                "Kernel intervals establish execution overlap, not throughput or SM occupancy.",
                ha="center", fontsize=9, color="#444444")
    figure.tight_layout(rect=(0, 0.055, 1, 0.93))
    for suffix in ("png", "svg"):
        figure.savefig(args.output_dir / f"gpu-trace-timeline.{suffix}", dpi=180)
    plt.close(figure)
    print(f"Wrote {len(rows)} stage rows and {len(windows)} diagnostic windows to {args.output_dir}")


if __name__ == "__main__":
    main()
