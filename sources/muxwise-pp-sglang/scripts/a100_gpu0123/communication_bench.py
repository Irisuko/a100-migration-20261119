#!/usr/bin/env python3
"""Four-rank NCCL measurements with GPU event samples inside one CUDA graph.

Run only through the GPU0-3 controller, with torchrun --nproc-per-node=4.
Each topology/size uses five eager warmups, then captures thirty operations.
Payload reset precedes each start event; all thirty samples execute in one
graph replay, removing Python submission/polling gaps between timed kernels.
Three changed-input graph replays check full BF16 payloads. The last replay's
thirty samples are reported. Capture failure is preserved and stops this probe.
No custom all-reduce or model inference compatibility is established here.
"""
import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import statistics
import time
import traceback

from platform_probe import DEFAULT_SIZES, launch_config, parse_sizes, write_json

TIMEOUT = timedelta(seconds=60)


def plans():
    return {
        'tp4_allreduce': {'kind': 'allreduce', 'groups': [[0, 1, 2, 3]]},
        'tp2_allreduce': {'kind': 'allreduce', 'groups': [[0, 1], [2, 3]]},
        'pp2dp2_p2p': {'kind': 'p2p', 'groups': [[0, 1], [2, 3]]},
        'pp2tp2_p2p': {'kind': 'p2p', 'groups': [[0, 2], [1, 3]]},
        **{f'pp4_boundary_{a}{b}': {'kind': 'p2p', 'groups': [[a, b]]}
           for a, b in ((0, 1), (1, 2), (2, 3))},
    }


def enqueue_dependency(work):
    # Torch2.8 ProcessGroupNCCL.hpp documents synchronize() as a CUDA-stream
    # dependency. Unlike wait(timeout=...), this does not poll/block the host.
    work.synchronize()


def summarize(samples, nbytes, group_size, kind):
    if not samples or any(not math.isfinite(v) or v <= 0 for v in samples):
        raise ValueError('GPU timings must be finite and strictly positive')
    ordered = sorted(samples)
    median = statistics.median(ordered)
    result = {'bytes': nbytes, 'bf16_elements': nbytes // 2, 'samples': len(samples),
              'gpu_ms': samples, 'median_gpu_ms': median, 'min_gpu_ms': min(samples),
              'max_gpu_ms': max(samples), 'p95_gpu_ms': ordered[math.ceil(.95 * len(ordered)) - 1],
              'algorithmic_GB_per_s': nbytes / (median * 1e6)}
    if kind == 'allreduce':
        result['ring_bus_GB_per_s_estimate'] = result['algorithmic_GB_per_s'] * 2 * (group_size - 1) / group_size
    return result


def measure(torch, dist, rank, members, group, kind, nbytes, warmup, count):
    """All calls run on a dedicated current stream; no timed host waits."""
    source = torch.empty(nbytes // 2, dtype=torch.bfloat16, device=rank)
    payload = torch.empty_like(source)
    stream = torch.cuda.Stream(device=rank)
    stream.wait_stream(torch.cuda.current_stream())
    role = 'allreduce' if kind == 'allreduce' else 'sender' if rank == members[0] else 'receiver'

    def prepare():
        if role != 'receiver':
            payload.copy_(source)
        else:
            payload.fill_(-1)

    def issue():
        if kind == 'allreduce':
            enqueue_dependency(dist.all_reduce(payload, group=group, async_op=True))
        else:
            fn, peer = (dist.isend, members[1]) if role == 'sender' else (dist.irecv, members[0])
            for work in dist.batch_isend_irecv([dist.P2POp(fn, payload, peer, group=group)]):
                enqueue_dependency(work)

    def validate(offset):
        expected = sum(r + 1 + offset for r in members) if kind == 'allreduce' else members[0] + 1 + offset
        if role != 'sender' and not bool(torch.all(payload == expected).item()):
            raise RuntimeError(f'{kind} payload mismatch rank={rank} members={members} bytes={nbytes}')
        return expected

    with torch.cuda.stream(stream):
        source.fill_(rank + 1)
        for _ in range(warmup):
            prepare()
            issue()
    stream.synchronize()
    validate(0)
    starts = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(count)]
    ends = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(count)]
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            for start, end in zip(starts, ends):
                prepare()
                start.record()
                issue()
                end.record()
        expected = None
        for offset in (0, 3, 7):
            with torch.cuda.stream(stream):
                source.fill_(rank + 1 + offset)
                graph.replay()
            stream.synchronize()
            expected = validate(offset)
        samples = [start.elapsed_time(end) for start, end in zip(starts, ends)]
        return {**summarize(samples, nbytes, len(members), kind), 'kind': kind,
                'members': members, 'role': role, 'graph_capture': 'passed',
                'changed_input_graph_replays': 3, 'full_payload_validated': role != 'sender',
                'expected_last_value': expected,
                'timing_scope': 'GPU events around each captured NCCL operation plus stream dependency; payload copy/reset excluded'}
    finally:
        stream.synchronize()
        graph.reset()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sizes', type=parse_sizes, default=DEFAULT_SIZES)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--samples', type=int, default=30)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.warmup <= 20 or not 1 <= args.samples <= 100:
        parser.error('Use 1..20 warmups and 1..100 GPU samples')
    rank, local_rank, world_size, mask = launch_config(os.environ)
    if args.dry_run:
        print(json.dumps({'rank': rank, 'uuids': mask, 'sizes': args.sizes,
                          'warmup': args.warmup, 'samples': args.samples, 'plans': plans()}))
        return
    # Import the runner only for ownership checks; Torch remains unimported until
    # the controller token, assignment and unique output paths are validated.
    from server_run import require_active_controller
    require_active_controller()
    output = args.output.resolve()
    rank_path = output.with_name(output.name + f'.rank{rank}.json')
    if rank_path.exists() or (rank == 0 and output.exists()):
        raise FileExistsError('Use a new output label for every probe')
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.update(NCCL_DEBUG='INFO', NCCL_DEBUG_SUBSYS='INIT,GRAPH,P2P,NET',
                      NCCL_DEBUG_FILE=str(output.with_name(output.name + '.nccl-%p.log')),
                      TORCH_NCCL_BLOCKING_WAIT='0', TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
    import torch
    import torch.distributed as dist

    report = {'schema_version': 1, 'rank': rank, 'cuda_visible_devices': mask,
              'run_id': os.environ.get('TORCHELASTIC_RUN_ID'), 'started_unix': time.time(),
              'status': 'running', 'stage': 'initialization', 'rows': [],
              'torch': torch.__version__, 'cuda': torch.version.cuda, 'warmup': args.warmup,
              'samples_per_size_topology': args.samples,
              'nccl_environment': {k: v for k, v in os.environ.items() if k.startswith(('NCCL_', 'TORCH_NCCL_'))},
              'limitations': ['Isolated NCCL communication, not loaded-model overlap or custom all-reduce.',
                              'CUDA graph capture failure is retained; no eager substitute enters these timings.',
                              'Per-operation GPU timings include stream dependency/event overhead and peer skew.',
                              'TP2 and paired PP cases run two groups concurrently; PP4 boundaries are isolated.',
                              'Bandwidth uses payload bytes; ring bus normalization is an estimate, not physical link traffic.']}
    initialized = False

    def save():
        report['elapsed_seconds'] = time.time() - report['started_unix']
        write_json(rank_path, report)

    try:
        save()
        torch.cuda.set_device(local_rank)
        props = torch.cuda.get_device_properties(local_rank)
        if torch.cuda.device_count() != 4 or 'A100' not in props.name or (props.major, props.minor) != (8, 0):
            raise RuntimeError('Exactly four assigned A100 sm80 devices are required')
        report['device_name'] = props.name
        report['nccl_version'] = list(torch.cuda.nccl.version())
        dist.init_process_group('nccl', timeout=TIMEOUT, device_id=torch.device('cuda', local_rank))
        initialized = True
        cpu = dist.new_group(list(range(world_size)), backend='gloo', timeout=TIMEOUT)
        groups = {(0, 1, 2, 3): dist.group.WORLD}
        for members in sorted({tuple(g) for p in plans().values() for g in p['groups']}):
            if members not in groups:
                groups[members] = dist.new_group(list(members), backend='nccl', timeout=TIMEOUT)
        for name, plan in plans().items():
            members = next((g for g in plan['groups'] if rank in g), None)
            for nbytes in args.sizes:
                report['stage'] = f'{name}/{nbytes}'
                save()
                dist.monitored_barrier(group=cpu, timeout=TIMEOUT, wait_all_ranks=True)
                if members is not None:
                    row = measure(torch, dist, rank, members, groups[tuple(members)], plan['kind'],
                                  nbytes, args.warmup, args.samples)
                    report['rows'].append({'topology': name, **row})
                    save()
                dist.monitored_barrier(group=cpu, timeout=TIMEOUT, wait_all_ranks=True)
        report.update(status='passed', stage='complete')
        save()
        dist.monitored_barrier(group=cpu, timeout=TIMEOUT, wait_all_ranks=True)
        if rank == 0:
            ranks = [json.loads(output.with_name(output.name + f'.rank{r}.json').read_text()) for r in range(4)]
            if not all(r['status'] == 'passed' and r['run_id'] == report['run_id'] for r in ranks):
                raise RuntimeError('Rank status or launch identity mismatch')
            write_json(output, {'schema_version': 1, 'status': 'passed', 'ranks': ranks})
            print(f'NCCL graph communication probe passed: {output}', flush=True)
        dist.monitored_barrier(group=cpu, timeout=TIMEOUT, wait_all_ranks=True)
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        save()
        if rank == 0:
            write_json(output, {'schema_version': 1, 'status': 'failed', 'rank0': report})
        raise
    finally:
        if initialized:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
