#!/usr/bin/env python3
"""Plot preserved paired communication measurements; no GPU access."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report = json.loads(args.report.read_text())
    assert report['status'] == 'passed' and report['independent_launches'] == 1
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), sharey=True)
    colors = {'nccl': '#315e9c', 'custom_dispatch': '#c76a20'}
    for ax, topology, title in zip(axes, ['tp4', 'tp2_pairs'], ['TP4', 'Two concurrent TP2 groups']):
        for mode, label in [('nccl', 'PyNCCL'), ('custom_dispatch', 'Custom dispatch / NCCL fallback')]:
            rows = sorted((r for r in report['rows'] if r['topology'] == topology and r['mode'] == mode), key=lambda r:r['bytes'])
            assert len(rows) == 5 and all(r['validated'] and r['rank_count'] == 4 for r in rows)
            ax.errorbar([r['bytes'] for r in rows], [r['rank_mean_us'] for r in rows],
                        yerr=[[r['rank_mean_us']-r['rank_min_us'] for r in rows],
                              [r['rank_max_us']-r['rank_mean_us'] for r in rows]],
                        color=colors[mode], marker='o', capsize=3, label=label)
        ax.set(xscale='log', yscale='log', xlabel='BF16 message payload', title=title)
        ax.set_xticks([8, 16384, 262144, 4194304, 33554432], ['8 B', '16 KiB', '256 KiB', '4 MiB', '32 MiB'])
        ax.tick_params(axis='x', labelsize=8)
        ax.grid(alpha=.2, which='both')
    axes[0].set_ylabel('Allreduce GPU time (microseconds)')
    axes[0].legend(fontsize=8, loc='upper left')
    fig.suptitle('A100 80GB NVLink, fixed GPU0–3', fontsize=13)
    fig.text(.5, .015, 'Single launch; bars show rank range, not independent-repeat uncertainty. Custom kernels at 16 KiB–4 MiB.', ha='center', fontsize=8)
    fig.tight_layout(rect=[0, .055, 1, .94])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.png', '.pdf', '.svg'):
        path = args.out.with_suffix(suffix)
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
