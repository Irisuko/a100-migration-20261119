#!/usr/bin/env python3
"""CPU-only tables and static plots for one saved communication probe."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    source = args.input.resolve()
    data = json.loads(source.read_text())
    if data['status'] != 'passed' or {r['rank'] for r in data['ranks']} != set(range(4)):
        raise ValueError('A complete, passing four-rank probe is required')
    grouped = defaultdict(list)
    run_ids = {r['run_id'] for r in data['ranks']}
    if len(run_ids) != 1 or any(r['status'] != 'passed' for r in data['ranks']):
        raise ValueError('Mixed or incomplete rank reports')
    for rank in data['ranks']:
        for row in rank['rows']:
            if row['role'] == 'sender':
                continue
            if not row['full_payload_validated'] or row['graph_capture'] != 'passed':
                raise ValueError('Unvalidated numerical or graph result')
            if row['samples'] != len(row['gpu_ms']) or row['median_gpu_ms'] != statistics.median(row['gpu_ms']):
                raise ValueError('Saved sample counts/medians are inconsistent')
            grouped[row['topology'], row['bytes']].append({'rank': rank['rank'], **row})
    table = []
    for (topology, nbytes), records in sorted(grouped.items()):
        medians = [r['median_gpu_ms'] for r in records]
        speeds = [r['algorithmic_GB_per_s'] for r in records]
        entry = dict(topology=topology, bytes=nbytes, kind=records[0]['kind'],
                     rank_selection='all members' if records[0]['kind'] == 'allreduce' else 'receivers only',
                     ranks=','.join(str(r['rank']) for r in records), rank_count=len(records),
                     samples_per_rank=records[0]['samples'],
                     rank_median_mean_ms=statistics.mean(medians),
                     rank_median_min_ms=min(medians), rank_median_max_ms=max(medians),
                     rank_mean_mean_ms=statistics.mean(statistics.mean(r['gpu_ms']) for r in records),
                     sample_min_ms=min(min(r['gpu_ms']) for r in records),
                     sample_max_ms=max(max(r['gpu_ms']) for r in records),
                     algorithmic_GB_per_s_mean=statistics.mean(speeds),
                     algorithmic_GB_per_s_min=min(speeds), algorithmic_GB_per_s_max=max(speeds))
        if records[0]['kind'] == 'allreduce':
            entry['ring_bus_GB_per_s_estimate_mean'] = statistics.mean(r['ring_bus_GB_per_s_estimate'] for r in records)
        else:
            entry['ring_bus_GB_per_s_estimate_mean'] = None
        table.append(entry)
    transports, pairs, log_sources, path_evidence, topology_evidence = Counter(), defaultdict(Counter), {}, [], []
    for path in sorted(source.parent.glob(source.name + '.nccl-*.log')):
        log_sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            match = re.search(r'Channel\s+\S+\s+:\s+\d+\[(\d+)\]\s+->\s+\d+\[(\d+)\]\s+via\s+(\S+)', line)
            if match:
                a, b, transport = match.groups()
                transports[transport] += 1
                pairs[f'{a}->{b}'][transport] += 1
                path_evidence.append({'file': str(path), 'line': line_number, 'source_device': int(a),
                                      'destination_device': int(b), 'transport': transport})
            if 'GPU/' in line and re.search(r'\(\d+/[\d.]+/NVL\)', line):
                topology_evidence.append({'file': str(path), 'line': line_number, 'text': line.strip()})
    report = dict(schema_version=1, status='passed', source=str(source),
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  run_id=next(iter(run_ids)), independent_launches=1,
                  warmup_per_case=data['ranks'][0]['warmup'],
                  measured_samples_per_rank_case=data['ranks'][0]['samples_per_size_topology'],
                  aggregation='Arithmetic mean/min/max of per-rank 30-sample medians; P2P uses receiver ranks only.',
                  sample_scope='Thirty sequential event pairs from the final replay of one captured graph, after five eager warmups and three changed-input validation replays. These are not independent repeated deployments.',
                  rows=table, nccl=dict(version=data['ranks'][0]['nccl_version'],
                     log_sha256=log_sources, observed_channel_transports=dict(transports),
                     observed_directed_device_pairs={k: dict(v) for k, v in sorted(pairs.items())},
                     path_evidence=path_evidence, nvl_topology_evidence=topology_evidence,
                     interpretation='Channel connection records show the actual selected P2P/CUMEM paths; NCCL topology also reports NVL connectivity among these devices. Socket bootstrap messages are not evidence that measured payloads used sockets. Logs do not measure physical link traffic or attribute each channel line to a specific timing sample.'),
                  limitations=['One independent four-rank launch; rank ranges are not confidence intervals.',
                               'TP2 and paired PP cases execute two groups concurrently; isolated PP4 cases execute one boundary.',
                               'CUDA events include NCCL stream dependencies and peer skew, exclude payload copy/reset and host polling.',
                               'Algorithmic throughput is payload bytes per GPU time. Ring bus bandwidth is a normalization estimate.',
                               'No model throughput, end-to-end SLO, green-context communication, custom all-reduce, or loaded-model graph compatibility claim.'])
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    targets = [out / ('communication-report.' + suffix) for suffix in ('json', 'csv', 'png', 'svg')]
    if any(path.exists() for path in targets):
        raise FileExistsError('Choose a new report directory; existing artifacts are preserved')
    targets[0].write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    with targets[1].open('w') as destination:
        writer = csv.DictWriter(destination, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FixedFormatter
    labels = {'tp4_allreduce': 'TP4 all-reduce', 'tp2_allreduce': 'TP2 × 2 all-reduce',
              'pp2dp2_p2p': 'PP pairs 0→1, 2→3', 'pp2tp2_p2p': 'PP pairs 0→2, 1→3',
              'pp4_boundary_01': 'Isolated 0→1', 'pp4_boundary_12': 'Isolated 1→2',
              'pp4_boundary_23': 'Isolated 2→3'}
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.6), sharex=True)
    for column, kind in enumerate(('allreduce', 'p2p')):
        selected = sorted({row['topology'] for row in table if row['kind'] == kind})
        for topology in selected:
            records = sorted((r for r in table if r['topology'] == topology), key=lambda r: r['bytes'])
            sizes = [r['bytes'] for r in records]
            mean = [r['rank_median_mean_ms'] * 1000 for r in records]
            low = [r['rank_median_min_ms'] * 1000 for r in records]
            high = [r['rank_median_max_ms'] * 1000 for r in records]
            line, = axes[0, column].plot(sizes, mean, marker='o', markersize=4, label=labels[topology])
            axes[0, column].fill_between(sizes, low, high, color=line.get_color(), alpha=.16)
            axes[1, column].plot(sizes, [r['algorithmic_GB_per_s_mean'] for r in records],
                                 marker='o', markersize=4, color=line.get_color())
            axes[1, column].fill_between(sizes, [r['algorithmic_GB_per_s_min'] for r in records],
                                        [r['algorithmic_GB_per_s_max'] for r in records],
                                        color=line.get_color(), alpha=.16)
        axes[0, column].set_title('All-reduce · all member ranks' if kind == 'allreduce' else 'PP transfers · receiver ranks only')
        axes[0, column].legend(frameon=False, fontsize=9, loc='upper left')
        axes[0, column].set_ylabel('GPU latency (µs), rank median mean')
        axes[1, column].set_ylabel('Algorithmic payload throughput (GB/s)')
        axes[1, column].set_xlabel('BF16 payload size')
        for row in range(2):
            ax = axes[row, column]
            ax.set_xscale('log', base=2)
            ax.set_yscale('log')
            ax.grid(True, which='major', color='#dedede', linewidth=.6)
            ax.xaxis.set_major_locator(FixedLocator([8, 16384, 262144, 4194304, 33554432]))
            ax.xaxis.set_major_formatter(FixedFormatter(['8 B', '16 KiB', '256 KiB', '4 MiB', '32 MiB']))
    fig.suptitle('A100 GPU0–3 · captured NCCL communication · one independent launch', fontsize=14)
    fig.text(.5, .024, '30 event samples from the last graph replay. Shading: min–max across rank medians, not confidence intervals.\n'
             'Payload reset and host polling excluded; no model performance inference.', ha='center', fontsize=9, color='#454545')
    fig.tight_layout(rect=[0, .075, 1, .955], h_pad=2)
    fig.savefig(targets[2], dpi=180)
    fig.savefig(targets[3])
    plt.close(fig)
    print(json.dumps({'rows': len(table), 'artifacts': [str(p) for p in targets],
                      'channel_transports': dict(transports)}, indent=2))


if __name__ == '__main__':
    main()
