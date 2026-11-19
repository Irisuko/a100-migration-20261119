#!/usr/bin/env python3
"""Build audited heldout streams only after explicit development-rate freeze.

Window choice uses source counts/timestamps, never formal performance. The CLI
trusts the adopted immutable parent manifest. No rates are inferred by this tool.
"""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PARENT = REPO / 'run/a100-gpu0123-20261114/traces/manifest.json'
PARENT_SHA256 = '30efd62c489c3fabe4a37f9b7befc9ee1f46b93b73c6663d7618e5b2d6b89eb9'
BUILDER = REPO.parent / 'muxwise-pp-experiments/bench/build_trace.py'
BUILDER_SHA256 = '784323ac240f84bcee854a51b94e51c7c6d2b8270b815e4283a1b83289b07607'
MODEL_NAMES = {'8b': 'Llama-3.1-8B-Instruct', '70b': 'Llama-3.3-70B-Instruct'}
START, END, GRID, STRETCH = 1440, 3000, 3, 2.5
MAX_TOTAL, BLOCK = 32768, 512
POLICY = dict(source_start_sec=START, source_limit_end_exclusive_sec=END,
              end_grid_sec=GRID, stretch=STRETCH, minimum_arrival_window_sec=210,
              minimum_active_arrival_span_sec=180, minimum_distinct_timestamps=3,
              choice='earliest qualifying grid end using only raw counts and timestamps',
              length_filter='exclude whole requests with input+output >32768',
              no_clipping=True, no_padding=True, no_request_duplication=True)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def rate_fraction(rate):
    require(type(rate) in (int, float) and math.isfinite(rate) and rate > 0,
            'Rate must be a finite positive JSON number')
    return Fraction(str(rate))


def validate_rates(document):
    require(document.get('schema_version') == 1 and document.get('status') == 'frozen',
            'Explicit schema1 frozen rates are required; no draft or calibration inference')
    require(valid_sha(document.get('selection_manifest_sha256')), 'Missing selected-configuration digest')
    require(document.get('evidence_phases') == ['rate_calibration'],
            'Frozen rates must cite development rate_calibration evidence only')
    evidence = document.get('evidence_sha256')
    require(isinstance(evidence, dict) and evidence and
            all(isinstance(p, str) and p and valid_sha(h) for p, h in evidence.items()),
            'Missing reviewed rate-calibration evidence hashes')
    records = document.get('records')
    require(isinstance(records, list) and records, 'Frozen rates contain no records')
    seen = set()
    for record in records:
        require(isinstance(record, dict) and record.get('model') in MODEL_NAMES and
                record.get('workload') in ('toolagent', 'conversation') and
                record.get('role') in ('main', 'critical'), 'Invalid frozen rate record')
        rate_fraction(record.get('absolute_requests_per_sec'))
        key = digest(record)
        require(key not in seen, 'Duplicate frozen rate record')
        seen.add(key)
    return records


def validate_rows(rows):
    last_idx, last_ts = -1, -1
    for idx, row in rows:
        require(type(idx) is int and idx > last_idx, 'Duplicate or unsorted source indices')
        require(isinstance(row, dict), 'Invalid raw source row')
        ts, inp, out = (row.get(k) for k in ('timestamp', 'input_length', 'output_length'))
        require(type(ts) is int and ts >= last_ts and type(inp) is int and inp >= 1 and
                type(out) is int and out >= 0, 'Invalid raw timestamp/order/length')
        last_idx, last_ts = idx, ts


def choose_window(indexed_rows, rate):
    """Select by exact rational arithmetic; output arrivals retain source timing."""
    value = rate_fraction(rate)
    rows = list(indexed_rows)
    validate_rows(rows)
    available = [(i, r) for i, r in rows if START * 1000 <= r['timestamp'] < END * 1000]
    eligible = [(i, r) for i, r in available if r['input_length'] + r['output_length'] <= MAX_TOTAL]
    search, selected, cursor = [], [], 0
    for end in range(START + GRID, END + 1, GRID):
        while cursor < len(eligible) and eligible[cursor][1]['timestamp'] < end * 1000:
            selected.append(eligible[cursor]); cursor += 1
        n = len(selected)
        nominal = n / value
        raw_span_ms = selected[-1][1]['timestamp'] - selected[0][1]['timestamp'] if n else 0
        active = nominal * Fraction(raw_span_ms, (end - START) * 1000)
        distinct = len({r['timestamp'] for _, r in selected})
        qualifies = nominal >= 210 and active >= 180 and distinct >= 3
        search.append(dict(source_end_exclusive_sec=end, requests=n,
                           arrival_window_sec_exact=str(nominal), active_arrival_span_sec_exact=str(active),
                           distinct_timestamps=distinct, qualifies=qualifies))
        if qualifies:
            base = Fraction((end - START) * 5, 2)
            return dict(source_start_sec=START, source_end_exclusive_sec=end, requests=n,
                        source_indices=[i for i, _ in selected],
                        filtered_requests_in_window=sum(r['timestamp'] < end * 1000 and
                                                        r['input_length'] + r['output_length'] > MAX_TOTAL
                                                        for _, r in available),
                        raw_active_span_ms=raw_span_ms, distinct_timestamps=distinct,
                        base_arrival_window_sec=float(base), arrival_window_sec=float(nominal),
                        active_arrival_span_sec=float(active), time_scale=float(nominal / base),
                        arrival_window_sec_exact=str(nominal), active_arrival_span_sec_exact=str(active),
                        time_scale_exact=str(nominal / base), search=search)
    raise ValueError('No qualifying heldout window at the frozen rate; no padding or repetition allowed')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def distribution(values):
    values = list(values)
    require(values, 'Cannot summarize an empty source cohort')
    return dict(count=len(values), min=min(values), max=max(values), mean=float(np.mean(values)),
                **{'p' + str(p): float(np.percentile(values, p)) for p in (50, 90, 95, 99)})


def length_summary(rows):
    return dict(input_length=distribution(r['input_length'] for _, r in rows),
                output_length=distribution(r['output_length'] for _, r in rows),
                total_length=distribution(r['input_length'] + r['output_length'] for _, r in rows),
                input_ge_16384=sum(r['input_length'] >= 16384 for _, r in rows),
                input_ge_30000=sum(r['input_length'] >= 30000 for _, r in rows),
                input_ge_32000=sum(r['input_length'] >= 32000 for _, r in rows))


def build_windows(rates_path, parent_manifest_path, out_dir, *, expected_parent_sha256=PARENT_SHA256):
    """CPU-only library entry. An alternate parent trust hash is for isolated fixtures."""
    rates_path, parent_path, out = (Path(p).resolve() for p in (rates_path, parent_manifest_path, out_dir))
    require(not out.exists(), f'Refusing to replace existing output: {out}')
    pinned = {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        actual = sha(path)
        require(expected is None or actual == expected, f'Input hash mismatch: {path}')
        require(str(path) not in pinned or pinned[str(path)] == actual, f'Changed input: {path}')
        pinned[str(path)] = actual
        return actual

    pin(parent_path, expected_parent_sha256)
    parent = json.loads(parent_path.read_text())
    require(parent.get('status') == 'PASS', 'Parent trace audit did not pass')
    pin(rates_path)
    document = json.loads(rates_path.read_text())
    rates = validate_rates(document)
    for name, expected in document['evidence_sha256'].items():
        path = Path(name)
        pin(path if path.is_absolute() else rates_path.parent / path, expected)
    require(Path(parent['builder']['path']).resolve() == BUILDER.resolve() and
            parent['builder']['sha256'] == BUILDER_SHA256, 'Unrecognized canonical builder')
    pin(BUILDER, BUILDER_SHA256)
    pin(BUILDER.with_name('common.py'))
    pin(Path(__file__))
    inventory_path = HERE / 'trace_inventory.py'
    pin(inventory_path)
    inventory = load_module('_formal_window_inventory', inventory_path)
    sys.path.insert(0, str(BUILDER.parent))
    try:
        canonical = load_module('_formal_window_canonical', BUILDER)
    finally:
        sys.path.pop(0)

    workloads = sorted({r['workload'] for r in rates})
    model_keys = sorted({r['model'] for r in rates})
    sources, models, raw, tokenizers = {}, {}, {}, {}
    for workload in workloads:
        matches = [s for s in parent['sources'] if s['workload'] == workload]
        require(len(matches) == 1, f'Ambiguous source: {workload}')
        source = matches[0]
        pin(source['path'], source['sha256'])
        with Path(source['path']).open() as stream:
            rows = [(i, json.loads(line)) for i, line in enumerate(stream) if line.strip()]
        validate_rows(rows)
        sources[workload], raw[workload] = source, rows
    for key in model_keys:
        matches = [m for m in parent['models'] if Path(m['path']).name == MODEL_NAMES[key]]
        require(len(matches) == 1, f'Ambiguous target tokenizer/model: {key}')
        model = matches[0]
        tokenizer = Path(model['path'])
        hashes = {}
        for name in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json'):
            hashes[name] = pin(tokenizer / name, model['nonweight_file_sha256'][name])
        ordinary, bos, excluded = canonical.load_vocabulary(tokenizer)
        require(np.array_equal(ordinary, np.arange(128000, dtype=np.int32)) and bos == 128000 and excluded == 256,
                f'Target tokenizer vocabulary changed: {tokenizer}')
        require(hashlib.sha256(ordinary.tobytes()).hexdigest() == model['ordinary_vocab_int32_sha256'],
                'Ordinary vocabulary hash changed')
        models[key], tokenizers[key] = model, dict(path=str(tokenizer), file_sha256=hashes,
                ordinary_vocab_size=len(ordinary), bos_token_id=bos, excluded_tokens=excluded,
                ordinary_vocab_int32_sha256=hashlib.sha256(ordinary.tobytes()).hexdigest())

    plans = []
    for rate in rates:
        key, workload = rate['model'], rate['workload']
        records = [r for r in parent['records'] if r['model'] == MODEL_NAMES[key] and r['workload'] == workload]
        held = [r for r in records if r['purpose'] == 'heldout']
        dev = [r for r in records if r['purpose'] == 'dev']
        require(len(held) == len(dev) == 1 and held[0]['source_start_sec'] == START and
                held[0]['source_end_exclusive_sec'] == END and dev[0]['source_start_sec'] == 1200 and
                dev[0]['source_end_exclusive_sec'] == START, 'Parent split differs from adopted policy')
        choice = choose_window(raw[workload], rate['absolute_requests_per_sec'])
        wanted = set(choice['source_indices'])
        selected = [(i, r) for i, r in raw[workload] if i in wanted]
        dev_ids = {i for i, r in raw[workload] if 1200000 <= r['timestamp'] < START * 1000}
        require(not wanted & dev_ids, 'Development/raw-heldout overlap')
        for i, row in selected:
            hashes = row.get('hash_ids')
            require(isinstance(hashes, list) and all(type(h) is int and h >= 0 for h in hashes),
                    f'Invalid source prefix hashes at row {i}')
            require(len(hashes) * BLOCK >= row['input_length'], f'Padding would be needed at source row {i}')
        plans.append((rate, choice, selected))

    out.mkdir(parents=True, exist_ok=False)
    built, output_records = {}, []
    for rate, choice, selected in plans:
        key, workload, end = rate['model'], rate['workload'], choice['source_end_exclusive_sec']
        cache_key = (key, workload, end)
        if cache_key not in built:
            name = f'{key}_formal_{workload}_{START}_{end - START}_x2.5'
            path = out / (name + '.jsonl')
            command = [sys.executable, str(BUILDER), '--source', str(Path(sources[workload]['path']).resolve()),
                       '--start-sec', str(START), '--duration-sec', str(end - START), '--stretch', str(STRETCH),
                       '--max-total-len', str(MAX_TOTAL), '--block-size', str(BLOCK),
                       '--tokenizer', models[key]['path'], '--out', str(path)]
            with (out / (name + '.build.log')).open('x') as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                           env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1'},
                                           timeout=3600, check=False)
            require(completed.returncode == 0, f'Canonical builder failed; see {name}.build.log')
            stats_path = path.with_suffix('.stats.json')
            stats = json.loads(stats_path.read_text())
            checked = inventory.validate_generated(path, selected,
                       dict(start_sec=START, duration_sec=end - START, stretch=STRETCH))
            require(stats['requests'] == len(selected) and stats['padded_tokens'] == stats['padded_requests'] == 0 and
                    stats['duration_sec'] == choice['base_arrival_window_sec'] and
                    stats['filtered_requests'] == choice['filtered_requests_in_window'] and
                    stats['ordinary_vocab_size'] == 128000 and stats['bos_token_id'] == 128000,
                    f'Canonical stats disagree with source audit: {path}')
            raw_audit_path = out / (name + '.raw-index-audit.json')
            require(checked['source_indices_sha256'] == digest(choice['source_indices']),
                    'Raw source indices changed during canonical build')
            write_json(raw_audit_path, dict(source=sources[workload], source_indices=choice['source_indices'],
                       selected_raw_rows_sha256=digest([[i, r] for i, r in selected]),
                       dev_overlap_source_rows=0, **checked))
            full = [(i, r) for i, r in raw[workload] if START * 1000 <= r['timestamp'] < END * 1000 and
                    r['input_length'] + r['output_length'] <= MAX_TOTAL]
            built[cache_key] = dict(name=name, path=str(path), sha256=sha(path), bytes=path.stat().st_size,
                     stats_path=str(stats_path), stats_sha256=sha(stats_path), command=command,
                     raw_index_audit_path=str(raw_audit_path), raw_index_audit_sha256=sha(raw_audit_path),
                     source_sha256=sources[workload]['sha256'], dev_overlap_source_rows=0,
                     tokenizer=tokenizers[key], length_distributions=length_summary(selected),
                     full_heldout_length_distributions=length_summary(full), **checked)
        output_records.append(dict(**built[cache_key], model=MODEL_NAMES[key], model_key=key, workload=workload,
                    purpose='formal_frozen_rate', stretch=STRETCH, rate_record=rate, rate_record_sha256=digest(rate),
                    absolute_requests_per_sec=rate['absolute_requests_per_sec'], role=rate['role'],
                    window_choice={k: v for k, v in choice.items() if k != 'source_indices'},
                    **{k: v for k, v in choice.items() if k not in ('search', 'source_indices', 'requests', 'base_arrival_window_sec')}))
    for path, expected in pinned.items():
        require(sha(path) == expected, f'Input changed during build: {path}')
    manifest = dict(schema_version=1, status='PASS', purpose='formal_frozen_rate',
                    audited_at_utc=datetime.now(timezone.utc).isoformat(), formal_ready=False,
                    readiness='Trace audit only; selected configurations, support gates and formal queue freeze remain separate',
                    parent_manifest_path=str(parent_path), parent_manifest_sha256=pinned[str(parent_path)],
                    rates_manifest_path=str(rates_path), rates_manifest_sha256=pinned[str(rates_path)],
                    selection_manifest_sha256=document['selection_manifest_sha256'],
                    policy=POLICY, builder=parent['builder'], sources=list(sources.values()),
                    models=list(models.values()), input_sha256=pinned, records=output_records,
                    unique_generated_files=len(built), request_duplication=False,
                    limitations=['Anonymized hash-prefix reconstruction preserves lengths and block structure, not private original text.',
                                 'Natural window length/prefix distributions are reported, not balanced or optimized.',
                                 'Formal outcomes are never inputs to window choice. Original 600s reference streams remain separate.'])
    write_json(out / 'manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rates', required=True, type=Path, help='Reviewed status=frozen rate document')
    parser.add_argument('--parent-manifest', type=Path, default=PARENT)
    parser.add_argument('--out', required=True, type=Path, help='New exclusive-create directory')
    args = parser.parse_args()
    result = build_windows(args.rates, args.parent_manifest, args.out)
    print(json.dumps(dict(status=result['status'], manifest=str(args.out.resolve() / 'manifest.json'),
                          records=len(result['records']), unique_generated_files=result['unique_generated_files'])))


if __name__ == '__main__':
    main()
