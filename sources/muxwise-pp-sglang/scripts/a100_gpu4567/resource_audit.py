#!/usr/bin/env python3
"""Audit saved logs/snapshots without importing Torch or contacting a GPU/server.

Positive recovery requires every expected PP/TP rank and each recorded quiet
phase. API capacity/graph snapshots and sampled GPU memory remain descriptive;
neither can substitute for allocator conservation evidence.
"""
import argparse
import csv
import json
from pathlib import Path
import re

MARKER = 'A100 resource audit '
ERROR = re.compile(
    r'OutOfMemoryError|CUDA out of memory|illegal memory access|device-side assert|'
    r'memory leak detected|Watchdog timeout|Scheduler hit an exception|'
    r'Traceback \(most recent call last\)|NCCL[^\n]*\b(?:error|timed out)\b|'
    r'nccl(?:UnhandledCuda|Remote|System|Internal)Error|'
    r'collective[^\n]*timed out|startup timeout|deadlock', re.I)
COUNTS = ('capacity', 'free_kv', 'evictable', 'protected', 'free_reqs',
          'request_capacity', 'free_receive_slots', 'receive_slot_capacity')


def _rank(line):
    prefix = line.split(']', 1)[0]
    pp = re.search(r'\bPP(\d+)\b', prefix)
    tp = re.search(r'\bTP(\d+)\b', prefix)
    return (int(pp[1]) if pp else 0, int(tp[1]) if tp else 0)


def _status(errors, missing):
    return 'failed' if errors else 'incomplete' if missing else 'passed'


def _validate(record):
    problems = []
    for key in COUNTS:
        if type(record.get(key)) is not int or record[key] < 0:
            problems.append(f'invalid or missing {key}')
    if problems:
        return problems
    if record['capacity'] <= 0 or record['request_capacity'] <= 0:
        problems.append('capacity must be positive')
    if record['free_kv'] + record['evictable'] + record['protected'] != record['capacity']:
        problems.append('KV accounting does not equal capacity')
    if record['protected'] or record['free_kv'] + record['evictable'] != record['capacity']:
        problems.append('idle dense KV has unrecovered protected slots')
    if record['free_reqs'] != record['request_capacity']:
        problems.append('request slots not recovered')
    if record['free_receive_slots'] != record['receive_slot_capacity']:
        problems.append('PP receive slots not recovered')
    return problems


def _load_json(path, errors):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        errors.append(f'{path}: {type(exc).__name__}: {exc}')
        return None


def _memory_entries(value):
    if isinstance(value, dict):
        if isinstance(value.get('memory_usage'), dict):
            yield value['memory_usage']
        for child in value.values():
            yield from _memory_entries(child)
    elif isinstance(value, list):
        for child in value:
            yield from _memory_entries(child)


def _tokenizer_entries(value):
    if isinstance(value, dict):
        if 'tokenizer_pending_requests' in value:
            yield {k: value[k] for k in ('tokenizer_pending_requests', 'tokenizer_finished_states') if k in value}
        for child in value.values():
            yield from _tokenizer_entries(child)
    elif isinstance(value, list):
        for child in value:
            yield from _tokenizer_entries(child)


def _check_tokenizer(entries, location, errors):
    for entry in entries:
        for key, value in entry.items():
            if type(value) is not int or value < 0:
                errors.append(f'{location}: invalid {key}={value!r}')
            elif value:
                errors.append(f'{location}: tokenizer state not recovered: {key}={value}')


def audit_run(run_dir):
    root = Path(run_dir)
    errors, missing, workers, snapshots = [], [], [], []
    metadata = _load_json(root / 'run.json', errors) or {}
    args = metadata.get('args', {})
    if metadata.get('status') == 'failed':
        errors.append(f'run.json: failed: {metadata.get("error", "unspecified")}')
    elif metadata.get('status') != 'execution_complete':
        missing.append('run.json: execution has not completed')
    for key in ('tp', 'pp', 'replicas'):
        if type(args.get(key)) is not int or args[key] < 1:
            missing.append(f'run.json: positive {key} is required for rank coverage')
    tp, pp, replicas = (args.get(k, 0) for k in ('tp', 'pp', 'replicas'))
    if not all(type(n) is int and n > 0 for n in (tp, pp, replicas)):
        tp = pp = replicas = 0
    for index in range(replicas):
        logpath = root / f'server-{index}.log'
        worker_errors, worker_missing, records, graph, shutdown_pending = [], [], {}, {}, []
        if not logpath.exists():
            worker_missing.append(f'{logpath.name}: missing server log')
        else:
            with logpath.open(errors='replace') as source:
                for number, line in enumerate(source, 1):
                    location = f'{logpath.name}:{number}'
                    if ERROR.search(line):
                        worker_errors.append(f'{location}: {line.strip()[:800]}')
                    pending = re.search(r'(?:Gracefully exiting|Force exiting|Exiting).*remaining number of requests:?\s+(\d+)', line)
                    if pending:
                        shutdown_pending.append({'source': location, 'pending': int(pending[1])})
                    if MARKER in line:
                        try:
                            record = json.loads(line.split(MARKER, 1)[1])
                            rank = record['pp_rank'], record['tp_rank']
                            if any(type(r) is not int for r in rank):
                                raise ValueError('rank must be an integer')
                            if not (0 <= rank[0] < pp and 0 <= rank[1] < tp):
                                raise ValueError(f'unexpected rank {rank}')
                            stamp = record['time_unix']
                            if type(stamp) not in (int, float) or not (0 < stamp < float('inf')):
                                raise ValueError('invalid time_unix')
                            record = {**record, 'source': location}
                            records.setdefault(rank, []).append(record)
                            worker_errors.extend(f'{location}: {p}' for p in _validate(record))
                        except (ValueError, KeyError, TypeError) as exc:
                            worker_errors.append(f'{location}: malformed audit evidence: {exc}')
                    if 'Capture cuda graph end.' in line:
                        match = re.search(r'mem usage=([\d.]+) GB', line)
                        if match:
                            graph[_rank(line)] = float(match[1])
        if shutdown_pending and shutdown_pending[-1]['pending'] > 0:
            last = shutdown_pending[-1]
            worker_errors.append(f'{last["source"]}: tokenizer shutdown still has {last["pending"]} pending requests; no subsequent zero-count shutdown record')
        phasepath = root / f'worker{index}' / 'resource-phase-markers.json'
        phases = _load_json(phasepath, worker_errors) if phasepath.exists() else None
        pair_phases = []
        for pairpath in sorted(root.glob('pair*/pair.json')):
            pair = _load_json(pairpath, worker_errors) or {}
            if index not in [pair.get(role, {}).get('index') for role in ('prefill', 'decode')]:
                continue
            markerpath = pairpath.with_name('resource-phase-markers.json')
            if markerpath.exists():
                value = _load_json(markerpath, worker_errors)
                if isinstance(value, list):
                    pair_phases.extend({**p, 'phase': f'{pairpath.parent.name}/{p["phase"]}'}
                                       for p in value if isinstance(p, dict) and 'phase' in p)
        if pair_phases:
            phases = (phases if isinstance(phases, list) else []) + pair_phases
        if phases is None:
            worker_missing.append(f'worker{index}: no quiet phase markers; idle logs may be startup-only')
            phases = []
        if not isinstance(phases, list):
            worker_errors.append(f'{phasepath}: phase markers must be a list')
            phases = []
        valid_phases = []
        for phase in phases:
            try:
                if not isinstance(phase['phase'], str) or not phase['phase']:
                    raise ValueError('phase name is required')
                start, end = phase['start_unix'], phase['end_unix']
                if (type(start) not in (int, float) or type(end) not in (int, float)
                        or not (0 < start <= end < float('inf'))):
                    raise ValueError('invalid quiet interval')
                valid_phases.append(phase)
            except (KeyError, TypeError, ValueError) as exc:
                worker_errors.append(f'{phasepath}: invalid phase: {exc}')
        if not valid_phases:
            worker_missing.append(f'worker{index}: no valid quiet phase evidence requested')
        final_names = {phase['phase'].split('/')[-1] for phase in valid_phases}
        if not final_names.intersection({'final', 'run-final'}):
            worker_missing.append(f'worker{index}: no final recovery phase')
        if (args.get('suite') == 'trace' or replicas > 1) and 'run-final' not in final_names:
            worker_missing.append(f'worker{index}: no run-final phase after replay/router workload')
        required_pre_flush = {'run-final-pre-flush'}
        if args.get('suite') == 'correctness':
            required_pre_flush.add('after-cancel-pre-flush')
        if args.get('stress200'):
            required_pre_flush.add('after-stress-pre-flush')
        for phase in sorted(required_pre_flush - final_names):
            worker_missing.append(f'worker{index}: missing required pre-reset phase {phase}')
        final_snapshot = root / f'worker{index}' / 'server-final.json'
        final_tokenizer = []
        if final_snapshot.exists():
            value = _load_json(final_snapshot, worker_errors)
            final_tokenizer = list(_tokenizer_entries(value))
            _check_tokenizer(final_tokenizer, str(final_snapshot.relative_to(root)), worker_errors)
        if not final_tokenizer:
            worker_missing.append(f'worker{index}: no explicit final tokenizer pending-state count')
        ranks = []
        for pr in range(pp):
            for tr in range(tp):
                name = f'worker{index} pp={pr} tp={tr}'
                entries = sorted(records.get((pr, tr), []), key=lambda r: r['time_unix'])
                phases_result = []
                if not entries:
                    worker_missing.append(f'{name}: no positive allocator evidence')
                for phase in valid_phases:
                    matched = [r for r in entries if phase['start_unix'] <= r['time_unix'] <= phase['end_unix']]
                    if not matched:
                        worker_missing.append(f'{name}: no idle evidence in phase {phase["phase"]}')
                    phases_result.append({**phase, 'status': 'passed' if matched else 'incomplete',
                                          'evidence': [r['source'] for r in matched]})
                peaks = {}
                for record in entries:
                    for key, value in record.get('cuda_memory', {}).items():
                        if type(value) is int and value >= 0:
                            peaks[key] = max(peaks.get(key, 0), value)
                ranks.append({'pp': pr, 'tp': tr, 'capacity': entries[-1].get('capacity') if entries else None,
                              'request_capacity': entries[-1].get('request_capacity') if entries else None,
                              'idle_record_count': len(entries), 'last_idle': entries[-1] if entries else None,
                              'cuda_memory_peak_bytes': peaks, 'graph_capture_gb': graph.get((pr, tr)),
                              'phases': phases_result,
                              'conservation': {'status': 'failed' if any(_validate(r) for r in entries)
                                               else 'passed' if entries else 'incomplete'}})
        workers.append({'index': index, 'status': _status(worker_errors, worker_missing),
                        'ranks': ranks, 'errors': worker_errors, 'missing_evidence': worker_missing,
                        'tokenizer_final': final_tokenizer, 'tokenizer_shutdown': shutdown_pending})
        errors.extend(worker_errors)
        missing.extend(worker_missing)
    for path in sorted(root.rglob('*server-*.json')):
        value = _load_json(path, errors)
        tokenizer = list(_tokenizer_entries(value))
        # These artifacts are collected after quiet intervals. The initial
        # pre-workload server-before snapshot is descriptive only.
        if path.name != 'server-before.json':
            _check_tokenizer(tokenizer, str(path.relative_to(root)), errors)
        snapshots.append({'path': str(path.relative_to(root)), 'memory_usage': list(_memory_entries(value)),
                          'tokenizer_state': tokenizer,
                          'scope': 'API per-DP representative; not per-rank allocator conservation'})
    gpu_peaks = {}
    monitor = root / 'gpu-monitor.csv'
    if monitor.exists():
        with monitor.open() as source:
            for row in csv.DictReader(source, skipinitialspace=True):
                try:
                    uuid = row['uuid'].strip()
                    memory = next(v for k, v in row.items() if k.strip().startswith('memory.used'))
                    value = float(memory.split()[0])
                    gpu_peaks[uuid] = max(gpu_peaks.get(uuid, 0), value)
                except (ValueError, KeyError, TypeError, StopIteration, IndexError):
                    continue
    return {'schema_version': 2, 'run_dir': str(root.resolve()), 'status': _status(errors, missing),
            'errors': errors, 'missing_evidence': missing, 'workers': workers, 'snapshots': snapshots,
            'gpu_memory_peak_mib': gpu_peaks,
            'limitations': ['Conservation is checked only during saved quiet intervals.',
                            'PyTorch peaks exclude allocations outside its allocator; nvidia-smi peaks are sampled.',
                            'No detected timeout is not proof that a completed run was deadlock-free.',
                            'This audit does not establish numerical output correctness.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit_run(args.run)
    with args.output.open('x') as destination:
        json.dump(result, destination, indent=2, allow_nan=False)
        destination.write('\n')
    print(result['status'])
    raise SystemExit(0 if result['status'] == 'passed' else 1)


if __name__ == '__main__':
    main()
