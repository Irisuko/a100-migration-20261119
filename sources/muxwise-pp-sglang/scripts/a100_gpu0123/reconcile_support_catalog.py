#!/usr/bin/env python3
"""Reconcile explicit support labels using existing CPU eligibility gates.

This never launches a process, contacts a model endpoint, rewrites a run, or
changes campaign.json. Re-run with a new output directory after more attempts
complete. Catalog output is a builder input with no executable queue tasks.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

import build_development as development
import serving_compatibility

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROOT = REPO / 'run/a100-gpu0123-20261114'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_inventory(rows, candidates):
    require(isinstance(rows, list) and rows, 'Explicit nonempty attempt inventory required')
    known = {candidate['id'] for candidate in candidates}
    labels, directories = set(), set()
    for row in rows:
        require(row.get('candidate_id') in known, 'Unknown explicitly mapped candidate')
        label = row.get('label', '')
        require(isinstance(label, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', label),
                'Invalid attempt label')
        directory = row.get('measurement_directory')
        require(isinstance(directory, str) and directory, 'Missing measurement directory')
        require(label not in labels and directory not in directories,
                'Duplicate label or measurement directory binding')
        labels.add(label)
        directories.add(directory)
    return rows


def select_latest(rows):
    eligible = [row for row in rows if row.get('eligible') is True
                and row.get('controller_status') == 'completed'
                and row.get('controller_returncode') == 0
                and row.get('runtime_status') == 'execution_complete'
                and type(row.get('completed_unix')) in (float, int)
                and math.isfinite(row['completed_unix'])]
    return max(eligible, key=lambda row: (row['completed_unix'], row['label'])) if eligible else None


def inspect_attempt(entry, candidate, root, repo, source, compatibility):
    label = entry['label']
    run = development.resolve(entry['measurement_directory'], repo)
    require(root in run.parents, 'Measurement directory must stay inside campaign')
    record = dict(entry, eligible=False, completed_unix=None)
    outcome_path = root / 'runs' / label / 'outcome.json'
    metadata_path = run / 'run.json'
    outcome = development.read(outcome_path) if outcome_path.is_file() else {}
    metadata = development.read(metadata_path) if metadata_path.is_file() else {}
    record.update(controller_status=outcome.get('status', 'not_recorded'),
                  controller_returncode=outcome.get('returncode'),
                  runtime_status=metadata.get('status', 'not_recorded'),
                  completed_unix=outcome.get('ended'))
    # Existing exact-source/compatibility, mode, full70B, model, controller
    # binding, custom-allreduce, stored and fresh resource gates stay intact.
    evidence, passed = development.eligibility(
        candidate, {candidate['id']: (label, run)}, root, repo, source, compatibility)
    fresh = evidence.pop('fresh_resource_audit', None)
    if fresh is not None:
        evidence['fresh_resource_audit_summary'] = {
            key: fresh.get(key) for key in ('status', 'errors', 'missing_evidence')}
    record.update(eligible=passed is not None, eligibility=evidence)
    return record


def reconcile(root, inventory_path, compatibility_path, repo=REPO):
    root, repo = Path(root).resolve(), Path(repo).resolve()
    inventory_path = Path(inventory_path).resolve()
    inventory = development.read(inventory_path)
    require(inventory.get('schema_version') == 1, 'Unknown inventory schema')
    for filename, expected in inventory.get('origin_sha256', {}).items():
        require(development.sha(development.resolve(filename, repo)) == expected,
                f'Explicit inventory origin changed: {filename}')
    campaign = development.read(root / 'campaign.json')
    candidates = campaign['candidates']
    entries = validate_inventory(inventory.get('attempts'), candidates)
    source = development.current_sources(repo)
    compatibility = serving_compatibility.load(compatibility_path, source, repo)
    rows, mappings, missing = [], [], []
    for candidate in candidates:
        attempts = [inspect_attempt(entry, candidate, root, repo, source, compatibility)
                    for entry in entries if entry['candidate_id'] == candidate['id']]
        selected = select_latest(attempts)
        rows.append(dict(candidate_id=candidate['id'], attempts=attempts,
                         selected_label=selected['label'] if selected else None,
                         numerical_correctness='Separate required gate; never inferred from execution/resource support'))
        if selected:
            mappings.append(dict(candidate_id=candidate['id'], covered_by_label=selected['label'],
                                 measurement_directory=selected['measurement_directory'],
                                 observed_status='execution_complete',
                                 completed_unix=selected['completed_unix'],
                                 reason='Latest explicitly mapped completed attempt that passes existing builder support gates.',
                                 not_an_automatic_correctness_pass=True))
        else:
            missing.append(candidate['id'])
    pins = dict(source)
    for name in ('control.py', 'queue.py', 'build_development.py', 'resource_audit.py',
                 'serving_compatibility.py', 'reconcile_support_catalog.py'):
        path = repo / 'scripts/a100_gpu0123' / name
        pins[str(path.relative_to(repo))] = development.sha(path)
    pins[str(inventory_path)] = development.sha(inventory_path)
    pins.update(compatibility['artifacts_sha256'])
    common = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
                  inventory_path=str(inventory_path), inventory_sha256=development.sha(inventory_path),
                  serving_compatibility=compatibility)
    catalog = dict(**common, status='support_catalog_only_not_executable', tasks=[],
                   source_sha256=pins, excluded_equivalent_candidates=mappings,
                   candidates_without_qualified_completed_attempt=missing,
                   adapter_note='excluded_equivalent_candidates is the existing builder mapping field; no new family equivalence is introduced.',
                   selection_rule='Existing builder eligibility gates, then latest finite controller completion time; all attempts retained in audit.json.',
                   numerical_correctness='Separate gate', formal_ready=False)
    audit = dict(**common, status='CPU_RECONCILED_SUPPORT_ONLY', candidates=rows,
                 selected_count=len(mappings), total_candidates=len(candidates),
                 campaign_sha256=development.sha(root / 'campaign.json'),
                 gpu_processes_started=0, campaign_modified=False)
    return catalog, audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--serving-compatibility', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    root, output = args.root.resolve(), args.out_dir.resolve()
    require(root in output.parents, 'Output must stay inside campaign')
    require(not output.exists(), 'Use a new output directory; prior catalogs are immutable')
    catalog, audit = reconcile(root, args.inventory, args.serving_compatibility)
    output.mkdir(parents=True)
    for name, value in [('catalog.json', catalog), ('audit.json', audit)]:
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=audit['status'], selected=audit['selected_count'],
                          total=audit['total_candidates'], catalog=str(output / 'catalog.json'),
                          gpu_processes_started=0)))


if __name__ == '__main__':
    main()
