#!/usr/bin/env python3
"""CustomAllreduce dispatch versus PyNccl under identical CUDA graph sampling.

Controller-owned torchrun4 only. No models or core changes. Five eager warmups,
three changed-input eager checks, thirty captured operations and three changed-
input graph replays per size/mode/topology. Optional kernel profiling is an extra
untimed replay after timings are read. TP2 pairs execute concurrently.
"""
import argparse
from collections import Counter
from contextlib import nullcontext
from datetime import timedelta
import gzip
import hashlib
import json
import logging
import os
from pathlib import Path
import time
import traceback

from communication_bench import summarize
from platform_probe import DEFAULT_SIZES, launch_config, parse_sizes, write_json

REPO = Path(__file__).resolve().parents[2]
TIMEOUT = timedelta(seconds=60)


def plans():
    return {'tp4': [[0, 1, 2, 3]], 'tp2_pairs': [[0, 1], [2, 3]]}


class Dispatcher:
    """Count Python dispatches, never invent counts for CUDA graph replays."""
    def __init__(self, custom, nccl, mode, capturing):
        if mode not in ('nccl', 'custom_dispatch'):
            raise ValueError('unknown communication mode')
        self.custom, self.nccl, self.mode, self.capturing = custom, nccl, mode, capturing
        self.counts = {f'{phase}_{kind}': 0 for phase in ('eager', 'capture') for kind in ('custom', 'nccl')}

    def __call__(self, payload):
        phase = 'capture' if self.capturing() else 'eager'
        if self.mode == 'custom_dispatch':
            output = self.custom.custom_all_reduce(payload)
            if output is not None:
                self.counts[phase + '_custom'] += 1
                return output
        self.nccl.all_reduce(payload)
        self.counts[phase + '_nccl'] += 1
        return payload


def require_evidence(counts, mode, eligible, count, warmup):
    kind = 'custom' if mode == 'custom_dispatch' and eligible else 'nccl'
    other = 'nccl' if kind == 'custom' else 'custom'
    if counts['eager_' + kind] != warmup + 3:
        raise RuntimeError('missing positive actual eager dispatch and changed-input evidence')
    if counts['capture_' + kind] != count or counts['eager_' + other] or counts['capture_' + other]:
        raise RuntimeError('capture dispatch differs from expected eligibility or mixes paths')
    return kind


def check_actual_peers(rank, members, check):
    results = []
    for peer in members:
        if peer != rank:
            passed = bool(check(rank, peer))
            results.append(dict(source=rank, target=peer, passed=passed))
            if not passed:
                raise RuntimeError(f'actual-device P2P failed: {rank}->{peer}')
    return results


def validate_outputs(torch, outputs, expected):
    unique = {}
    for output in outputs:
        unique[output.data_ptr()] = output
    for output in unique.values():
        if not bool(torch.all(output == expected).item()):
            raise RuntimeError('full BF16 allreduce payload mismatch')
    return len(unique)


def kernel_evidence(path):
    with gzip.open(path, 'rt') as stream:
        events = json.load(stream)['traceEvents']
    names = Counter(e['name'] for e in events if e.get('cat') == 'kernel' and e.get('dur', 0) > 0)
    if not names:
        raise RuntimeError('requested kernel profile contains no GPU kernels')
    return dict(path=str(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                gpu_kernel_events=sum(names.values()), kernel_names=dict(names),
                interpretation='Actual CUDA symbols, not a heuristic custom/NCCL classification.')


def capture_operations(torch, custom, graph, stream, record, use_custom):
    # Registration performs CPU gloo exchanges and must be outside CUDA capture.
    with custom.capture() if use_custom else nullcontext():
        with torch.cuda.graph(graph, stream=stream):
            record()


def close_custom(custom, dispose):
    """Release this benchmark's subgroup buffers using their subgroup rank."""
    if custom.disabled or not custom._ptr:
        return
    # Runtime close() omits group= for free_shared_buffer; global ranks2/3
    # cannot index a two-entry TP2 pointer list. Keep that serving source fixed
    # while this isolated benchmark explicitly supplies the owning group.
    dispose(custom._ptr)
    custom._ptr = 0
    custom.free_shared_buffer(custom.meta_ptrs, group=custom.group)
    custom.free_shared_buffer(custom.buffer_ptrs, group=custom.group)


def measure(torch, rank, members, custom, nccl, mode, nbytes, warmup, count, profile_path=None):
    source = torch.empty(nbytes // 2, dtype=torch.bfloat16, device=rank)
    payload = torch.empty_like(source)
    stream = torch.cuda.Stream(device=rank)
    stream.wait_stream(torch.cuda.current_stream())
    dispatch = Dispatcher(custom, nccl, mode, torch.cuda.is_current_stream_capturing)
    eligible = bool(custom.should_custom_ar(payload))
    expected_eligibility = nbytes % 16 == 0 and nbytes < custom.max_size
    if eligible != expected_eligibility:
        raise RuntimeError('enabled NVLink communicator eligibility differs from supported contiguous size')
    graph = torch.cuda.CUDAGraph()
    starts = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(count)]
    ends = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(count)]
    outputs = []
    def expected(offset):
        return sum(member + 1 + offset for member in members)
    try:
        with nccl.change_state(enable=True, stream=stream):
            with torch.cuda.stream(stream):
                source.fill_(rank + 1)
                for _ in range(warmup):
                    payload.copy_(source)
                    output = dispatch(payload)
            stream.synchronize()
            validate_outputs(torch, [output], expected(0))
            for offset in (0, 3, 7):
                with torch.cuda.stream(stream):
                    source.fill_(rank + 1 + offset)
                    payload.copy_(source)
                    output = dispatch(payload)
                stream.synchronize()
                validate_outputs(torch, [output], expected(offset))
            def record():
                for begin, end in zip(starts, ends):
                    payload.copy_(source)
                    begin.record()
                    outputs.append(dispatch(payload))
                    end.record()
            capture_operations(torch, custom, graph, stream, record, mode == 'custom_dispatch')
            kind = require_evidence(dispatch.counts, mode, eligible, count, warmup)
            distinct = None
            for offset in (0, 3, 7):
                with torch.cuda.stream(stream):
                    source.fill_(rank + 1 + offset)
                    graph.replay()
                stream.synchronize()
                distinct = validate_outputs(torch, outputs, expected(offset))
            samples = [begin.elapsed_time(end) for begin, end in zip(starts, ends)]
            row = summarize(samples, nbytes, len(members), 'allreduce')
            # Custom kernels need not implement a ring; do not imply physical bus traffic.
            row.pop('ring_bus_GB_per_s_estimate', None)
            row.update(mode=mode, actual_dispatch=kind, custom_eligible=eligible,
                       dispatch_counts=dispatch.counts.copy(), graph_capture='passed',
                       changed_input_eager_validations=3, changed_input_graph_replays=3,
                       full_payload_validated=True, distinct_output_buffers_validated=distinct,
                       expected_last_value=expected(7), members=members,
                       graph_replay_python_dispatch_count=0,
                       timing_scope='Per-operation GPU events inside one graph replay; reset before start. Out-of-place custom output is not copied back. Allocation/registration/host submission excluded.',
                       kernel_profile=None)
            if profile_path is not None:
                profile_path.parent.mkdir(parents=True, exist_ok=True)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA],
                                            with_stack=False, record_shapes=False) as profiler:
                    with torch.cuda.stream(stream):
                        graph.replay()
                    stream.synchronize()
                validate_outputs(torch, outputs, expected(7))
                profiler.export_chrome_trace(str(profile_path))
                row['kernel_profile'] = kernel_evidence(profile_path)
                row['kernel_profile']['extra_untimed_graph_replays'] = 1
            return row
    finally:
        stream.synchronize()
        graph.reset()


def source_hashes():
    names = ['scripts/a100_gpu0123/custom_dispatch_bench.py',
             'scripts/a100_gpu0123/communication_bench.py', 'scripts/a100_gpu0123/platform_probe.py',
             'python/sglang/srt/distributed/parallel_state.py', 'python/sglang/srt/_custom_ops.py']
    names += ['python/sglang/srt/distributed/device_communicators/' + name for name in
              ('custom_all_reduce.py', 'custom_all_reduce_utils.py', 'pynccl.py', 'pynccl_wrapper.py')]
    return {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in names}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sizes', type=parse_sizes, default=DEFAULT_SIZES)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--samples', type=int, default=30)
    parser.add_argument('--kernel-profile', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.warmup <= 20 or not 1 <= args.samples <= 100:
        parser.error('Use1..20 warmups and1..100 samples')
    rank, local_rank, world, mask = launch_config(os.environ)
    if args.dry_run:
        print(json.dumps(dict(rank=rank, uuids=mask, plans=plans(), sizes=args.sizes,
                              modes=['nccl', 'custom_dispatch'], kernel_profile=args.kernel_profile)))
        return
    from server_run import require_active_controller
    require_active_controller()
    output = args.output.resolve()
    rank_path = output.with_name(output.name + f'.rank{rank}.json')
    if rank_path.exists() or (rank == 0 and output.exists()):
        raise FileExistsError('Every probe requires a new output name')
    output.parent.mkdir(parents=True, exist_ok=True)
    profile_dir = output.with_name(output.name + '.profiles') / f'rank{rank}'
    if args.kernel_profile and profile_dir.exists():
        raise FileExistsError(profile_dir)
    os.environ.update(SGLANG_SKIP_P2P_CHECK='0', NCCL_DEBUG='INFO',
                      NCCL_DEBUG_SUBSYS='INIT,GRAPH,P2P,NET',
                      NCCL_DEBUG_FILE=str(output.with_name(output.name + '.nccl-%p.log')),
                      TORCH_NCCL_BLOCKING_WAIT='0', TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
    logging.basicConfig(level=logging.INFO, format=f'[rank{rank}] %(message)s', force=True)
    import torch
    import torch.distributed as dist
    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment, destroy_distributed_environment, get_world_group)
    from sglang.srt.distributed.device_communicators.custom_all_reduce import CustomAllreduce
    from sglang.srt.distributed.device_communicators.custom_all_reduce_utils import (
        gpu_p2p_access_check, resolve_physical_device_ids)
    from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator
    from sglang.srt import _custom_ops
    report = dict(schema_version=1, status='running', rank=rank, run_id=os.environ.get('TORCHELASTIC_RUN_ID'),
                  stage='initialization', rows=[], started_unix=time.time(), cuda_visible_devices=mask,
                  torch=torch.__version__, cuda=torch.version.cuda, source_sha256=source_hashes(),
                  warmup=args.warmup, samples=args.samples,
                  modes=['nccl', 'custom_dispatch'], nccl_backend='PyNcclCommunicator on current stream',
                  custom_backend='vllm' if _custom_ops.use_vllm_custom_allreduce else 'sgl_kernel',
                  environment={key: value for key, value in os.environ.items()
                               if key.startswith(('NCCL_', 'TORCH_NCCL_')) or key in ('SGLANG_SKIP_P2P_CHECK', 'USE_VLLM_CUSTOM_ALLREDUCE')},
                  limitations=['Isolated communication; no model-loaded overlap or serving performance claim.',
                               'Fresh paired PyNccl baseline; previous communication-v1 used ProcessGroupNCCL with stream dependency.',
                               'Eager dispatch counters prove actual API dispatch; captured counts occur at capture, not on graph replay.',
                               'Kernel profiling is optional, extra and untimed; its symbols are retained without heuristic classification.',
                               'TP2 pairs execute concurrently. GPU event timing includes peer skew and event overhead.',
                               'Existing runtime P2P cache may be reused; actual directed device pairs and cache provenance are recorded.'])
    initialized = False
    active_custom = None
    def save():
        report['elapsed_seconds'] = time.time() - report['started_unix']
        write_json(rank_path, report)
    try:
        save()
        torch.cuda.set_device(local_rank)
        props = torch.cuda.get_device_properties(local_rank)
        if torch.cuda.device_count() != 4 or 'A100' not in props.name or (props.major, props.minor) != (8, 0):
            raise RuntimeError('Exactly the assigned four A100 sm80 devices are required')
        report['physical_device_ids'] = resolve_physical_device_ids(','.join(mask), 4)
        if report['physical_device_ids'] != [0, 1, 2, 3]:
            raise RuntimeError('UUID mapping differs from physical GPU0-3')
        init_distributed_environment(world_size=world, rank=rank, local_rank=local_rank,
                                     distributed_init_method='env://', backend='nccl', timeout=60)
        initialized = True
        cpu = get_world_group().cpu_group
        report['actual_device_p2p_checks'] = check_actual_peers(rank, list(range(4)), gpu_p2p_access_check)
        cache = Path.home() / '.cache/sglang' / f'gpu_p2p_access_cache_for_{",".join(mask)}.json'
        report['p2p_cache'] = dict(path=str(cache), sha256=hashlib.sha256(cache.read_bytes()).hexdigest())
        for topology, groups in plans().items():
            group = members = None
            for ranks in groups:
                candidate = dist.new_group(ranks, backend='gloo', timeout=TIMEOUT)
                if rank in ranks:
                    group, members = candidate, ranks
            report['stage'] = f'{topology}/communicator_initialization'
            save()
            active_custom = CustomAllreduce(group, torch.device('cuda', local_rank))
            if active_custom.disabled:
                raise RuntimeError('custom communicator disabled; fallback cannot establish custom support')
            nccl = PyNcclCommunicator(group, torch.device('cuda', local_rank), use_current_stream=True)
            if not nccl.available:
                raise RuntimeError('PyNccl communicator unavailable')
            report.setdefault('communicators', []).append(dict(topology=topology, members=members,
                custom_enabled=True, full_nvlink=active_custom.full_nvlink,
                max_custom_bytes_exclusive=active_custom.max_size))
            save()
            for nbytes in args.sizes:
                for mode in ('nccl', 'custom_dispatch'):
                    report['stage'] = f'{topology}/{nbytes}/{mode}'
                    save()
                    dist.monitored_barrier(group=cpu, timeout=TIMEOUT, wait_all_ranks=True)
                    profile = profile_dir / f'{topology}-{nbytes}-{mode}.trace.json.gz' if args.kernel_profile else None
                    row = measure(torch, rank, members, active_custom, nccl, mode,
                                  nbytes, args.warmup, args.samples, profile)
                    report['rows'].append(dict(topology=topology, **row))
                    save()
                    dist.monitored_barrier(group=cpu, timeout=TIMEOUT, wait_all_ranks=True)
            close_custom(active_custom, _custom_ops.dispose)
            active_custom = None
        report.update(status='passed', stage='complete')
        save()
        dist.monitored_barrier(group=cpu, timeout=TIMEOUT, wait_all_ranks=True)
        if rank == 0:
            ranks = [json.loads(output.with_name(output.name + f'.rank{r}.json').read_text()) for r in range(4)]
            if not all(r['status'] == 'passed' and r['run_id'] == report['run_id'] for r in ranks):
                raise RuntimeError('rank completion or launch identity mismatch')
            write_json(output, dict(schema_version=1, status='passed', ranks=ranks))
            print(f'Custom/NCCL graph dispatch probe passed: {output}', flush=True)
        dist.monitored_barrier(group=cpu, timeout=TIMEOUT, wait_all_ranks=True)
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        save()
        if rank == 0:
            write_json(output, dict(schema_version=1, status='failed', rank0=report))
        raise
    finally:
        if active_custom is not None:
            close_custom(active_custom, _custom_ops.dispose)
        if initialized:
            destroy_distributed_environment()


if __name__ == '__main__':
    main()
