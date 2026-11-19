#!/usr/bin/env python3
"""Record a verified campaign platform, then delegate to unchanged queue.py.

Dry runs perform CPU-only file/hash validation and never create queue state.
Environment JSON may set only LD_LIBRARY_PATH prefixes and RDMAV_DRIVERS;
GPU assignment and communication settings remain owned by the controller.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROOT = REPO / 'run/a100-gpu0123-20261114'
PYTHON = REPO / '.venv/bin/python'
ALLOWED = {'LD_LIBRARY_PATH', 'RDMAV_DRIVERS'}


def require(value, message):
    if not value:
        raise ValueError(message)


def resolve(value):
    path = Path(value)
    return (path if path.is_absolute() else REPO / path).resolve()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def checked_hash(path, expected):
    require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected), f'Invalid expected hash: {path}')
    actual = sha256(path)
    require(actual == expected, f'Platform hash mismatch: {path}')
    return actual


def load_platform(path, base_env):
    path = resolve(path)
    data = json.loads(path.read_text())
    require(data.get('schema_version') == 1, 'Unsupported environment schema')
    supplied = data.get('environment', {})
    require(isinstance(supplied, dict) and set(supplied) == ALLOWED,
            'Unsupported environment keys; only LD_LIBRARY_PATH and RDMAV_DRIVERS are validated')
    require(all(isinstance(value, str) and value and '\x00' not in value for value in supplied.values()),
            'Environment values must be nonempty strings without NUL')
    prefixes = supplied['LD_LIBRARY_PATH'].split(':')
    require(all(value and Path(value).is_absolute() and Path(value).is_dir() for value in prefixes),
            'Validated LD_LIBRARY_PATH prefixes must be existing absolute directories')
    require(Path(supplied['RDMAV_DRIVERS']).is_absolute(), 'RDMAV_DRIVERS must be an absolute provider base path')
    evidence = {'environment_path': str(path), 'environment_sha256': sha256(path)}
    metadata = {}
    for name in ('package-sources.json', 'packages-manifest.json', 'verification.json'):
        file = path.with_name(name)
        metadata[name] = json.loads(file.read_text())
        evidence[name] = dict(path=str(file), sha256=sha256(file))
    verification = metadata['verification.json']
    require(verification.get('schema_version') == 1 and verification.get('returncode') == 0
            and verification.get('status') == 'CPU_IMPORT_PROVIDER_PASS_GPU_TRANSFER_UNTESTED'
            and verification.get('environment') == supplied,
            'Platform requires matching successful CPU import/provider verification')
    packages = metadata['packages-manifest.json']
    sources = metadata['package-sources.json']
    require(isinstance(packages, list) and packages and isinstance(sources, list) and sources,
            'Nonempty package provenance is required')
    package_hashes = {str(resolve(row['file'])): checked_hash(resolve(row['file']), row['sha256']) for row in packages}
    source_hashes = {str(resolve(row['deb'])): row['sha256'] for row in sources}
    require(source_hashes == package_hashes and all(row.get('apt_index_sha256_match') is True for row in sources),
            'Package archive provenance differs from verified package bytes')
    libraries = verification.get('library_hashes', {})
    require(isinstance(libraries, dict) and libraries, 'Installed library hashes are required')
    library_hashes = {}
    for name, expected in libraries.items():
        library = path.parent / name
        require(not Path(name).is_absolute() and path.parent in library.resolve().parents,
                'Installed library must remain within the campaign platform directory')
        library_hashes[str(library)] = dict(sha256=checked_hash(library, expected), resolved_path=str(library.resolve()))
    provider = supplied['RDMAV_DRIVERS'] + '-rdmav34.so'
    require(provider in library_hashes, 'Explicit rdma-core39 provider is absent from verified libraries')
    env = dict(base_env)
    values = prefixes + env.get('LD_LIBRARY_PATH', '').split(':')
    env['LD_LIBRARY_PATH'] = ':'.join(dict.fromkeys(value for value in values if value))
    env['RDMAV_DRIVERS'] = supplied['RDMAV_DRIVERS']
    evidence.update(packages_sha256=package_hashes, installed_libraries=library_hashes,
                    verification_status=verification['status'], gpu_transfer_verified=False,
                    public_environment={key: env[key] for key in sorted(ALLOWED)})
    return env, evidence


def prepare(tasks, queue_id, *, root=ROOT, environment=None, freeze=None,
            continue_independent=False, base_env=None, mooncake_transport='default'):
    root, tasks = resolve(root), resolve(tasks)
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', queue_id), 'Invalid queue ID')
    environment = resolve(environment) if environment else root / 'environment/rdma-local/env.json'
    freeze = resolve(freeze) if freeze else None
    env, platform = load_platform(environment, os.environ if base_env is None else base_env)
    require(mooncake_transport in ('default', 'nvlink'), 'Unsupported Mooncake transport selection')
    # Mooncake checks presence, so an inherited value of "0" also enables it.
    env.pop('MC_INTRANODE_NVLINK', None)
    if mooncake_transport == 'nvlink':
        env['MC_INTRANODE_NVLINK'] = '1'
    public_environment = {**platform['public_environment'], 'MC_INTRANODE_NVLINK': env.get('MC_INTRANODE_NVLINK')}
    # Import file helpers only. No controller method or GPU library is invoked.
    spec = importlib.util.spec_from_file_location('_campaign_queue_validation', HERE / 'queue.py')
    queue = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(queue)
    plan = json.loads(tasks.read_text())
    queue.validate_plan(plan, root)
    pins, traces = {}, {}
    for task in plan['tasks']:
        checked = queue.check_task(task, plan, freeze)
        for name, digest in checked['source_sha256'].items():
            name = str(resolve(name))
            require(name not in pins or pins[name] == digest, 'Conflicting queue source pins')
            pins[name] = digest
        traces[task['id']] = checked['trace_sha256']
    command = [str(PYTHON), str(HERE / 'queue.py'), 'run', '--root', str(root),
               '--tasks', str(tasks), '--queue-id', queue_id]
    if freeze:
        command += ['--freeze', str(freeze)]
    if continue_independent:
        command += ['--continue-independent']
    launcher = {str(HERE / name): sha256(HERE / name)
                for name in ('run_campaign.py', 'queue.py', 'control.py', 'server_run.py')}
    manifest = dict(schema_version=1, queue_id=queue_id, campaign_root=str(root), command=command,
                    cwd=str(REPO), tasks_path=str(tasks), tasks_sha256=sha256(tasks),
                    freeze_path=str(freeze) if freeze else None, freeze_sha256=sha256(freeze) if freeze else None,
                    source_sha256=pins, task_trace_sha256=traces, launcher_sha256=launcher,
                    platform=platform, public_environment=public_environment, mooncake_transport=mooncake_transport,
                    transport_environment_policy='Always clear inherited MC_INTRANODE_NVLINK, including0; set1 only for explicit nvlink. Default selects the runtime default path. This flag is separate from verified loader variables and does not prove GPU transport support.',
                    scope='Loader variables passed unchanged through queue to controller. All families use this platform. CPU verification does not establish GPU registration or KV transfer. Controller owns GPU masks and NCCL settings.')
    return manifest, env


def persist_manifest(path, manifest, dry_run=False):
    path = Path(path)
    if path.exists():
        saved = json.loads(path.read_text())
        saved.pop('created_utc', None)
        require(saved == manifest, 'Platform environment, command, or pins changed; use a new queue ID')
        return
    require(not any((path.parent / name).exists() for name in ('status.json', 'plan.json')),
            'Existing queue has no platform provenance; use a new queue ID')
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Link an fsynced temporary file without overwriting an existing record.
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(dict(manifest, created_utc=datetime.now(timezone.utc).isoformat()), stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def read_platform_provenance(root, provenance):
    """Validate outer launch evidence and return stable software/config identities.

    Task paths, queue IDs and launch timestamps belong to evidence, not to the
    identity shared by independent repetitions. Historical runtime source files
    need not still match: recorded queue source pins are compared to each other.
    """
    root = resolve(root)
    queue_id = provenance.get('queue_id')
    missing = dict(status='missing', software=None, configuration={'mooncake_transport': 'unrecorded'})
    if not queue_id:
        return missing
    require(isinstance(queue_id, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', queue_id), 'Invalid platform queue ID')
    path = root / 'queues' / queue_id / 'platform-launch.json'
    if not path.is_file():
        return missing
    saved = json.loads(path.read_text())
    require(saved.get('schema_version') == 1 and saved.get('queue_id') == queue_id
            and saved.get('campaign_root') == str(root), 'Platform queue identity mismatch')
    tasks = resolve(saved['tasks_path'])
    checked_hash(tasks, saved['tasks_sha256'])
    plan = json.loads(tasks.read_text())
    canonical = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    require(provenance.get('plan_sha256') == canonical, 'Platform and executed queue plan differ')
    task = provenance['task']
    matches = [row for row in plan.get('tasks', []) if row.get('id') == task.get('id')]
    require(len(matches) == 1 and matches[0] == task, 'Platform and executed task differ')
    command = [str(PYTHON), str(HERE / 'queue.py'), 'run', '--root', str(root),
               '--tasks', str(tasks), '--queue-id', queue_id]
    if saved.get('freeze_path'):
        command += ['--freeze', saved['freeze_path']]
    recorded = saved.get('command', [])
    require(recorded in (command, command + ['--continue-independent']), 'Platform queue command mismatch')
    evidence = provenance.get('evidence', {})
    require(saved.get('freeze_sha256') == evidence.get('freeze_sha256'), 'Platform freeze provenance mismatch')
    pins = saved.get('source_sha256', {})
    require(all(pins.get(str(resolve(key))) == value for key, value in evidence.get('source_sha256', {}).items()),
            'Platform and executed source pins differ')
    require(saved.get('task_trace_sha256', {}).get(task['id']) == evidence.get('trace_sha256'),
            'Platform and executed trace pins differ')
    transport = saved.get('mooncake_transport')
    require(transport in ('default', 'nvlink'), 'Missing explicit platform transport choice')
    public = saved.get('public_environment', {})
    require(set(public) == ALLOWED | {'MC_INTRANODE_NVLINK'}
            and public['MC_INTRANODE_NVLINK'] == ('1' if transport == 'nvlink' else None),
            'Platform transport environment contradicts its selection')
    _, current_platform = load_platform(saved['platform']['environment_path'], public)
    require(current_platform == saved['platform'], 'Recorded platform libraries or loader provenance changed')
    require(current_platform['public_environment'] == {key: public[key] for key in ALLOWED},
            'Resolved loader environment differs from platform evidence')
    software = dict(loader_environment=current_platform['public_environment'],
                    installed_libraries=current_platform['installed_libraries'],
                    packages_sha256=current_platform['packages_sha256'])
    return dict(status='verified', software=software, configuration={'mooncake_transport': transport},
                public_environment=public, manifest_path=str(path), manifest_sha256=sha256(path))


def delegate(command, env):
    """Only live boundary: queue/controller retains ownership and cleanup."""
    child = subprocess.Popen(command, cwd=REPO, env=env, start_new_session=True)
    previous = {}
    def forward(signum, _frame):
        if child.poll() is None:
            try:
                child.send_signal(signum)
            except ProcessLookupError:
                pass
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, forward)
        return child.wait()
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--queue-id', required=True)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--environment', type=Path)
    parser.add_argument('--freeze', type=Path)
    parser.add_argument('--continue-independent', action='store_true')
    parser.add_argument('--mooncake-transport', choices=('default', 'nvlink'), default='default')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    def preparation():
        return prepare(args.tasks, args.queue_id, root=args.root, environment=args.environment,
                       freeze=args.freeze, continue_independent=args.continue_independent,
                       mooncake_transport=args.mooncake_transport)
    manifest, env = preparation()
    path = resolve(args.root) / 'queues' / args.queue_id / 'platform-launch.json'
    if args.dry_run:
        persist_manifest(path, manifest, dry_run=True)
        print(json.dumps(dict(status='PASS', gpu_processes_started=0, platform_manifest=str(path),
                              command=manifest['command'], public_environment=manifest['public_environment'],
                              source_pin_count=len(manifest['source_sha256']), platform_verified='CPU only')))
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with (path.parent / 'platform-launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Recheck after acquiring the lock; resumptions cannot change platform bytes.
        manifest, env = preparation()
        persist_manifest(path, manifest)
        return delegate(manifest['command'], env)


if __name__ == '__main__':
    raise SystemExit(main())
