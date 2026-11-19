#!/usr/bin/env python3
"""Plot frozen calibration and pilot observations; never run GPU work."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

REPO = Path(__file__).resolve().parents[2]
FORMATS = ("png", "pdf", "svg")


def save(fig, stem):
    paths = []
    for extension in FORMATS:
        path = stem.with_suffix("." + extension)
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        paths.append(str(path))
    plt.close(fig)
    return paths


def calibration_figure(data, directory):
    expected = {"static-correctness", "alp-correctness", "cpp-correctness"}
    assert {row["name"] for row in data["runs"]} == expected
    rows = data["measurements"]
    assert len(rows) == 336 and all(row["seconds_count"] == 3 for row in rows)
    assert len({row["group"] for row in rows}) == 1
    contexts = (512, 2048, 8192)
    colors = ("#21618C", "#C05C10", "#6C3483")
    raw_limit_ms = 1000 * (.05 / 2 - .001) / 1.2
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), sharex=True, sharey=True)
    for stage, ax in enumerate(axes):
        for context, color in zip(contexts, colors):
            selected = sorted((row for row in rows if row["stage"] == stage
                               and row["batch_size"] == 16 and row["context_tokens"] == context),
                              key=lambda row: row["decode_sms"])
            assert len(selected) == 7
            x = np.array([row["decode_sms"] for row in selected])
            median = 1000 * np.array([row["seconds_median"] for row in selected])
            lower = 1000 * np.array([row["seconds_min"] for row in selected])
            upper = 1000 * np.array([row["seconds_max"] for row in selected])
            ax.plot(x, median, marker="o", markersize=4, color=color,
                    linewidth=1.6, label=f"Context {context:,}")
            ax.errorbar(x, median, yerr=np.array([median - lower, upper - median]),
                        fmt="none", color=color, capsize=3, linewidth=1)
        ax.axhline(raw_limit_ms, color="#555555", linestyle="--", linewidth=1.2,
                   label="Initial 50 ms PP2 gate: raw 20 ms")
        ax.set_title(f"PP{stage} · decode batch 16", fontsize=12)
        ax.set_xlabel("Decode SMs")
        ax.set_xticks([8, 16, 24, 32, 48, 56, 108])
        ax.tick_params(axis="x", labelrotation=35)
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    max_observed_ms = 1000 * max(row["seconds_max"] for row in rows
                                if row["batch_size"] == 16
                                and row["context_tokens"] in contexts)
    axes[0].set_ylim(0, max_observed_ms * 1.06)
    axes[0].set_ylabel("Raw CUDA graph replay time (ms)")
    axes[1].legend(fontsize=8.5, frameon=False, loc="upper right")
    fig.suptitle("A100-SXM4-80GB · Llama-3.1-8B · BF16 · TP1/PP2", fontsize=13, y=.99)
    fig.text(.5, -.025,
             "Median and observed min–max across 3 startup calibrations; ranges are not confidence intervals.\n"
             "Dashed limit = (50 ms / 2 − 1 ms) / 1.2. Runtime contention correction can tighten this gate.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, .94))
    return save(fig, directory / "a100-calibration-frozen")


def pilot_figure(data, directory):
    assert data["completed"] and len(data["rows"]) == 8
    chunks, budgets = (256, 1024, 2048), (2048, 8192, 32768)
    throughput, slo = np.full((3, 3), np.nan), np.full((3, 3), np.nan)
    rows_by_cell = {}
    for row in data["rows"]:
        assert len(row["trials"]) == 2
        i, j = chunks.index(row["chunk_tokens"]), budgets.index(row["slice_token_layer_budget"])
        rows_by_cell[(i, j)] = row
        throughput[i, j] = row["output_tokens_per_s"]["median"]
        slo[i, j] = 100 * row["joint_slo_fraction"]["mean"]
    selected = (chunks.index(data["winner"]["chunk_tokens"]),
                budgets.index(data["winner"]["slice_token_layer_budget"]))
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8))
    for ax, values, title, colorbar_label, cmap_name in zip(
            axes, (throughput, slo), ("Output throughput", "Joint SLO attainment"),
            ("tokens / second", "% of requests"), ("Blues", "YlGnBu")):
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("#E7E7E7")
        image = ax.imshow(values, cmap=cmap, vmin=0, vmax=np.nanmax(values), aspect="auto")
        fig.colorbar(image, ax=ax, fraction=.046, pad=.035, label=colorbar_label)
        for i in range(3):
            for j in range(3):
                if np.isnan(values[i, j]):
                    label = "Equivalent\nnot run"
                    color = "#444444"
                elif values is throughput:
                    row = rows_by_cell[(i, j)]["output_tokens_per_s"]
                    label = f"{values[i,j]:.1f}\n({row['min']:.1f}–{row['max']:.1f})"
                    color = "white" if values[i, j] > .58 * np.nanmax(values) else "#17202A"
                else:
                    label = f"{values[i,j]:.2f}%"
                    color = "white" if values[i, j] > .58 * np.nanmax(values) else "#17202A"
                ax.text(j, i, label, ha="center", va="center", fontsize=10, color=color)
        i, j = selected
        ax.add_patch(Rectangle((j - .48, i - .48), .96, .96, fill=False,
                               edgecolor="#F39C12", linewidth=3))
        ax.set_title(title, fontsize=12)
        ax.set_xticks(range(3), [f"{budget:,}" for budget in budgets])
        ax.set_yticks(range(3), [f"{chunk:,}" for chunk in chunks])
        ax.set_xlabel("Slice token-layer budget")
        ax.set_ylabel("Prefill chunk tokens")
        ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)
    fig.suptitle("Frozen pilot · 24 requests × 2 trials per configuration", fontsize=13, y=.99)
    fig.text(.5, -.045,
             "Throughput: median (observed min–max); SLO: mean fraction with HTTP TTFT ≤ 2 s and request-mean TPOT ≤ 50 ms.\n"
             "Outline: best observed by SLO first, throughput second. Two repeats are not confidence intervals.\n"
             "256 / 32,768 is equivalent to 256 / 8,192: both permit all 16 local layers.",
             ha="center", fontsize=8.7)
    fig.tight_layout(rect=(0, .06, 1, .94))
    return save(fig, directory / "a100-pilot-selection")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=REPO / "run/a100/calibration-selection.json")
    parser.add_argument("--pilot", type=Path, default=REPO / "run/a100/pilot-selection.json")
    parser.add_argument("--output-dir", type=Path, default=REPO / "run/a100/report-data/plots")
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sources = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in (args.calibration, args.pilot)}
    outputs = calibration_figure(json.loads(args.calibration.read_text()), args.output_dir)
    outputs += pilot_figure(json.loads(args.pilot.read_text()), args.output_dir)
    manifest = {"sources_sha256": sources, "outputs": outputs,
                "raw_initial_gate_limit_ms": 1000 * (.05 / 2 - .001) / 1.2,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output_dir / "calibration-pilot-plots.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"outputs": outputs}))


if __name__ == "__main__":
    main()
