#!/usr/bin/env python3
"""Both models must be prepared before any formal order.py launch, including dryrun.

Preparation/split and native run_campaign.py --dry-run remain separate CPU paths.
Use `check --model 8b` to inspect readiness before a separately reviewed resource
handoff; `launch` always checks both models. This program never transfers GPUs.

Fixed automation/readiness/{8b,70b}.json receipt contract:
  schema_version: 1; status: ROOT_REVIEWED_MODEL_FORMAL_READINESS; reviewer: root;
  model: 8b|70b; workloads: [toolagent, conversation]; gpu_budget: 4;
  precision: bfloat16; max_total_tokens: 32768; policy_reference: {path,sha256};
  references: {configuration_freeze,rate_freeze,formal_trace_freeze,
               final_reverification,prepared_formal_manifest,dry_run_evidence};
  additional_formal_manifests: optional list of exact reviewed auxiliary refs.
All references are absolute path/SHA256 objects, never readiness booleans.

configuration_freeze is the original reviewed_selected_configurations document;
rate_freeze is the frozen DEV-only rate manifest; formal_trace_freeze is the
PASS/formal_frozen_rate trace manifest. Their selection/rate hashes must agree.
prepared_formal_manifest is an existing ROOT_REVIEWED_FROZEN_PERFORMANCE_BUNDLE
root review, binding original protocol, complete freeze, producer and actual
producer-validation receipt. Main model readiness cannot use an auxiliary.

final_reverification is ROOT_REVIEWED_FINAL_REVERIFICATION, reviewer=root, with
model/workloads/gpu_budget, configuration_freeze_reference and nonempty actual
evidence_sha256. This wrapper does not reinterpret numerical disposition or
require bitwise equivalence; the reviewed producer/scientific gates remain final.

dry_run_evidence is ROOT_REVIEWED_NATIVE_FORMAL_DRY_RUN, reviewer=root, model,
prepared_formal_manifest_reference and checks[]. Each check binds
command_reference ({argv,returncode:0}), result_reference (saved native stdout
JSON: status PASS, gpu_processes_started:0, command), task_plan_reference.
Commands must use the fixed pool's native launcher --dry-run with the exact full
freeze. The union of checked task plans equals the complete frozen task matrix.
This avoids depending on this wrapper's own dual-readiness dryrun.

Formal `launch` accepts exactly the original v1 launch arguments. An exclusive
<receipt>.readiness.json sidecar binds both current readiness documents and every
checked hash before exec of the unchanged pinned v1 delegate. Existing scientific,
global repeat, exact plan, fresh label, four-GPU and ownership checks still run.
No existing code, freeze, plan, or historical authority is modified by this gate.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / 'run/a100-gpu0123-20261114'
PYTHON = REPO / '.venv/bin/python'
POLICY_PATH = ROOT / 'automation/policies/both-models-ready-before-formal-v1.json'
POLICY_SHA256 = 'b07c578fe6b021cb640cbd47c61cda7e18a8bf52bc96064c89f6c5bb1c32041e'
DELEGATE_PATH = ROOT / 'proposals/global-formal-round-order-v1/order.py'
DELEGATE_SHA256 = 'e8c184a81f0144aaedabe2fe9fe1d2dc5f79db69565722a18baab18f1a4c9a71'
MODELS = ('8b', '70b')
WORKLOADS = {'toolagent', 'conversation'}
REQUIRED = {'configuration_freeze', 'rate_freeze', 'formal_trace_freeze',
            'final_reverification', 'prepared_formal_manifest', 'dry_run_evidence'}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def read(path):
    value = json.loads(Path(path).read_text())
    require(isinstance(value, dict), f'JSON object required: {path}')
    return value


def pin(ref, pins, *, document=True):
    require(isinstance(ref, dict) and set(ref) == {'path', 'sha256'}, 'Exact path/hash reference required')
    require(isinstance(ref['path'], str) and Path(ref['path']).is_absolute(), 'Absolute evidence path required')
    require(isinstance(ref['sha256'], str) and re.fullmatch('[a-f0-9]{64}', ref['sha256']), 'Invalid evidence SHA256')
    path = Path(ref['path']).resolve()
    require(sha(path) == ref['sha256'], f'Evidence hash changed: {path}')
    require(str(path) not in pins or pins[str(path)] == ref['sha256'], 'Conflicting evidence pins')
    pins[str(path)] = ref['sha256']
    return read(path) if document else path


def pin_mapping(mapping, pins):
    require(isinstance(mapping, dict) and mapping, 'Nonempty concrete evidence hashes required')
    for path, digest in mapping.items():
        p = Path(path)
        pin({'path': str(p if p.is_absolute() else REPO / p), 'sha256': digest}, pins, document=False)


def workloads(value):
    return isinstance(value, list) and len(value) == 2 and all(isinstance(x, str) for x in value) and set(value) == WORKLOADS


def option(argv, flag):
    require(isinstance(argv, list) and all(isinstance(x, str) for x in argv), 'Explicit string argv required')
    positions = [i for i, value in enumerate(argv) if value == flag or value.startswith(flag + '=')]
    require(len(positions) == 1 and argv[positions[0]] == flag and positions[0] + 1 < len(argv), f'One explicit {flag} required')
    return argv[positions[0] + 1]


def load_policy(pins):
    policy = pin({'path': str(POLICY_PATH), 'sha256': POLICY_SHA256}, pins)
    require(policy.get('status') == 'ACTIVE_USER_INSTRUCTION' and policy.get('id') == 'both-models-ready-before-formal-v1', 'Wrong policy type')
    require(policy.get('required_models') == list(MODELS) and workloads(policy.get('required_workloads'))
            and type(policy.get('gpu_budget_per_experiment')) is int and policy['gpu_budget_per_experiment'] == 4
            and policy.get('precision') == 'bfloat16' and policy.get('max_total_tokens') == 32768, 'Policy scope differs')
    expected = {m: str(ROOT / 'automation/readiness' / (m + '.json')) for m in MODELS}
    require(policy.get('readiness_paths') == expected and set(policy.get('required_readiness_references', [])) == REQUIRED, 'Fixed readiness registry/schema differs')
    gate = policy['formal_gate']
    require(gate.get('entrypoint') == str(Path(__file__).resolve()) and gate.get('delegate') == str(DELEGATE_PATH)
            and gate.get('delegate_sha256') == DELEGATE_SHA256
            and gate.get('both_models_ready_required_for_every_formal_launch') is True
            and gate.get('legacy_direct_formal_launch_not_authorized') is True, 'Fixed formal delegate/policy differs')
    pin({'path': str(DELEGATE_PATH), 'sha256': DELEGATE_SHA256}, pins, document=False)
    return policy


def prepared_bundle(ref, model, pins, *, auxiliary=False):
    bundle = pin(ref, pins)
    pool = 'gpu4567' if model == '8b' else 'gpu0123'
    producer = ('pool_' if model == '8b' else 'main_') + ('auxiliary' if auxiliary else 'formal')
    require(bundle.get('status') == 'ROOT_REVIEWED_FROZEN_PERFORMANCE_BUNDLE' and bundle.get('reviewer') == 'root'
            and bundle.get('pool') == pool and bundle.get('producer_kind') == producer
            and bundle.get('producer_validation_passed') is True, 'Prepared bundle type/model/pool differs')
    protocol = pin(bundle['protocol_reference'], pins)
    frozen = pin(bundle['freeze_reference'], pins)
    pin(bundle['producer_reference'], pins, document=False)
    validation = pin(bundle['producer_validation_reference'], pins)
    require(validation.get('status') == 'PASS' and all(validation.get(key) == bundle[key]
            for key in ('protocol_reference', 'freeze_reference', 'producer_reference')), 'Original producer validation binding differs')
    pin_mapping(validation.get('evidence_sha256'), pins)
    performance_stage = (protocol.get('stage') in ('formal', 'auxiliary') or protocol.get('auxiliary_only') is True) if auxiliary else protocol.get('stage') == 'formal'
    require(performance_stage
            and protocol.get('root') == str(ROOT / 'pool-gpu4567' if model == '8b' else ROOT)
            and frozen.get('status') == 'frozen' and protocol.get('source_sha256') == frozen.get('source_sha256'), 'Protocol/full freeze type or root differs')
    pin_mapping(frozen.get('source_sha256'), pins)
    tasks = frozen.get('tasks')
    require(isinstance(tasks, dict) and tasks and set(protocol.get('task_labels', [])) == set(tasks), 'Complete frozen task inventory required')
    for label, task in tasks.items():
        require(isinstance(task, dict) and task.get('id') == label and option(task.get('argv'), '--model') == model, 'Frozen task model/identity differs')
        budget = 1
        for flag in ('--tp', '--pp', '--replicas'):
            value = option(task['argv'], flag)
            require(re.fullmatch('[1-9][0-9]*', value), 'Positive explicit topology required')
            budget *= int(value)
        require(budget == 4, 'Frozen experiment is not exactly four GPUs')
    return bundle, protocol, frozen


def native_dryruns(document, bundle_ref, bundle, frozen, model, pins):
    require(document.get('status') == 'ROOT_REVIEWED_NATIVE_FORMAL_DRY_RUN' and document.get('reviewer') == 'root'
            and document.get('model') == model and document.get('prepared_formal_manifest_reference') == bundle_ref, 'Native dryrun review binding differs')
    checks = document.get('checks')
    require(isinstance(checks, list) and checks, 'Actual native dryrun checks required')
    pool = 'gpu4567' if model == '8b' else 'gpu0123'
    launcher = str(REPO / 'scripts' / ('a100_' + pool) / 'run_campaign.py')
    covered = []
    for check in checks:
        command = pin(check['command_reference'], pins)
        result = pin(check['result_reference'], pins)
        plan = pin(check['task_plan_reference'], pins)
        argv = command.get('argv', [])
        require(type(command.get('returncode')) is int and command['returncode'] == 0
                and argv[:2] == [str(PYTHON), launcher] and argv.count('--dry-run') == 1
                and '--extra' not in argv, 'Successful native CPU-only dryrun command required')
        require(result.get('status') == 'PASS' and type(result.get('gpu_processes_started')) is int
                and result['gpu_processes_started'] == 0, 'Native dryrun result did not pass without GPU launch')
        for actual in (argv, result.get('command')):
            require(option(actual, '--tasks') == check['task_plan_reference']['path']
                    and option(actual, '--freeze') == bundle['freeze_reference']['path'], 'Native dryrun checked another plan/freeze')
        rows = plan.get('tasks')
        require(isinstance(rows, list) and rows and plan.get('source_sha256') == frozen['source_sha256'], 'Native dryrun source/plan differs')
        for task in rows:
            require(isinstance(task, dict) and frozen['tasks'].get(task.get('id')) == task, 'Native dryrun task not in exact freeze')
            covered.append(task['id'])
    require(len(covered) == len(set(covered)) and set(covered) == set(frozen['tasks']), 'Native dryruns must cover every frozen task exactly once')


def model_readiness(model, policy, pins):
    own_ref = reference(policy['readiness_paths'][model])
    receipt = pin(own_ref, pins)
    require(type(receipt.get('schema_version')) is int and receipt['schema_version'] == 1
            and receipt.get('status') == 'ROOT_REVIEWED_MODEL_FORMAL_READINESS' and receipt.get('reviewer') == 'root'
            and receipt.get('model') == model and workloads(receipt.get('workloads'))
            and type(receipt.get('gpu_budget')) is int and receipt['gpu_budget'] == 4
            and receipt.get('precision') == 'bfloat16' and receipt.get('max_total_tokens') == 32768,
            'Model readiness receipt identity/scope differs')
    require(receipt.get('policy_reference') == {'path': str(POLICY_PATH), 'sha256': POLICY_SHA256}, 'Readiness binds another policy')
    refs = receipt.get('references')
    require(isinstance(refs, dict) and set(refs) == REQUIRED, 'All six exact readiness references required')
    docs = {key: pin(value, pins) for key, value in refs.items()}
    configs, rates, traces = [docs[key] for key in ('configuration_freeze', 'rate_freeze', 'formal_trace_freeze')]
    require(configs.get('status') == 'reviewed_selected_configurations', 'Configuration selection is not frozen/reviewed')
    require(rates.get('status') == 'frozen' and rates.get('evidence_phases') == ['rate_calibration'], 'Rates must be frozen using DEV calibration only')
    require(traces.get('status') == 'PASS' and traces.get('purpose') == 'formal_frozen_rate', 'Formal traces lack audited heldout freeze')
    for document, field in ((configs, 'configuration'), (rates, 'rate'), (traces, 'trace')):
        records = document.get('records')
        require(isinstance(records, list) and records, f'Empty {field} freeze')
        require(all(isinstance(row, dict) and (row.get('model_key') if field == 'trace' else
                    row.get('model', (row.get('candidate') or {}).get('model'))) == model for row in records), f'{field} freeze model differs')
        require({row.get('workload') for row in records} == WORKLOADS, f'{field} freeze lacks both workloads')
    config_hash, rate_hash = refs['configuration_freeze']['sha256'], refs['rate_freeze']['sha256']
    require(rates.get('selection_manifest_sha256') == config_hash and traces.get('selection_manifest_sha256') == config_hash
            and traces.get('rates_manifest_sha256') == rate_hash, 'Config/rate/formal trace freezes are unrelated')
    bundle, protocol, frozen = prepared_bundle(refs['prepared_formal_manifest'], model, pins)
    require(protocol.get('selection_manifest_sha256') == config_hash and protocol.get('selected_rate_manifest_sha256') == rate_hash
            and protocol.get('trace_manifest_sha256') == refs['formal_trace_freeze']['sha256'], 'Prepared bundle is not the frozen selection/rate/trace matrix')
    final = docs['final_reverification']
    require(final.get('status') == 'ROOT_REVIEWED_FINAL_REVERIFICATION' and final.get('reviewer') == 'root'
            and final.get('model') == model and workloads(final.get('workloads'))
            and type(final.get('gpu_budget')) is int and final['gpu_budget'] == 4
            and final.get('configuration_freeze_reference') == refs['configuration_freeze'], 'Final full reverification does not bind selected model/configuration')
    pin_mapping(final.get('evidence_sha256'), pins)
    native_dryruns(docs['dry_run_evidence'], refs['prepared_formal_manifest'], bundle, frozen, model, pins)
    allowed = [(refs['prepared_formal_manifest'], bundle)]
    extra = receipt.get('additional_formal_manifests', [])
    require(isinstance(extra, list), 'Additional auxiliary allowlist must be explicit')
    for ref in extra:
        additional, _, _ = prepared_bundle(ref, model, pins, auxiliary=True)
        require(ref not in [item[0] for item in allowed], 'Duplicate prepared formal bundle')
        allowed.append((ref, additional))
    return {'receipt_reference': own_ref, 'allowed_bundles': allowed}


def recheck(pins):
    for path, digest in pins.items():
        require(sha(path) == digest, f'Evidence changed before delegation: {path}')


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = parser.add_subparsers(dest='action', required=True)
    check = sub.add_parser('check', allow_abbrev=False)
    check.add_argument('--model', choices=MODELS)
    launch = sub.add_parser('launch', allow_abbrev=False)
    launch.add_argument('--schedule-reference', type=Path, required=True)
    launch.add_argument('--global-inventory-reference', type=Path)
    launch.add_argument('--repeat', type=int, required=True)
    launch.add_argument('--transport', choices=('default', 'nvlink'), required=True)
    launch.add_argument('--queue-id', required=True)
    launch.add_argument('--receipt', type=Path, required=True)
    launch.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    pins = {}
    policy = load_policy(pins)
    models = [args.model] if args.action == 'check' and args.model else list(MODELS)
    readiness = {model: model_readiness(model, policy, pins) for model in models}
    proof = {'status': 'READINESS_VALIDATED_NO_SCIENTIFIC_RECLASSIFICATION', 'models_checked': models,
             'policy_reference': {'path': str(POLICY_PATH), 'sha256': POLICY_SHA256},
             'model_readiness': {m: r['receipt_reference'] for m, r in readiness.items()},
             'evidence_sha256': pins, 'checked_unix': time.time()}
    if args.action == 'check':
        recheck(pins)
        print(json.dumps(proof, indent=2))
        return 0
    require(args.repeat >= 1, 'Positive independent repeat required')
    schedule_ref = pin(reference(args.schedule_reference), pins)
    schedule = pin(schedule_ref, pins)
    model = {'gpu4567': '8b', 'gpu0123': '70b'}.get(schedule.get('pool'))
    require(model in readiness, 'Schedule pool/model is outside current formal mapping')
    matches = [bundle for ref, bundle in readiness[model]['allowed_bundles'] if schedule.get('review_reference') == ref]
    require(len(matches) == 1 and schedule.get('freeze_reference') == matches[0]['freeze_reference'], 'Schedule is not an explicitly ready root-reviewed freeze/bundle')
    command = [str(PYTHON), str(DELEGATE_PATH), *argv]
    proof.update(schedule_reference=schedule_ref, delegate_reference={'path': str(DELEGATE_PATH), 'sha256': DELEGATE_SHA256},
                 command=command, execution_requested=args.execute)
    recheck(pins)
    sidecar = Path(str(args.receipt) + '.readiness.json')
    with sidecar.open('x') as handle:
        json.dump(proof, handle, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    directory_fd = os.open(sidecar.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    recheck(pins)
    return os.execv(command[0], command)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        print(json.dumps({'status': 'FORMAL_READINESS_REFUSED', 'error': f'{type(error).__name__}: {error}'}), file=sys.stderr)
        raise SystemExit(2)
