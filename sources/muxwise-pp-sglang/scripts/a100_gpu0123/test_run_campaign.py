"""Campaign platform preparation and dry runs must never start GPU work."""
import importlib.util
import json
from pathlib import Path
from unittest import mock

import pytest

SCRIPT = Path(__file__).with_name('run_campaign.py')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def module():
    assert SCRIPT.is_file(), 'run_campaign.py is missing'
    spec = importlib.util.spec_from_file_location('campaign_launch_test', SCRIPT)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def platform(tmp_path, module):
    root = tmp_path / 'campaign'
    folder = root / 'environment/rdma-local'
    lib = folder / 'root/lib'; lib.mkdir(parents=True)
    provider = lib / 'libmlx5-rdmav34.so'; provider.write_bytes(b'verified-provider')
    package = folder / 'packages/provider.deb'; package.parent.mkdir(); package.write_bytes(b'package')
    env = {'LD_LIBRARY_PATH': str(lib), 'RDMAV_DRIVERS': str(lib / 'libmlx5')}
    write(folder / 'env.json', {'schema_version': 1, 'environment': env})
    write(folder / 'packages-manifest.json', [{'file': str(package), 'sha256': module.sha256(package)}])
    write(folder / 'package-sources.json', [{'deb': str(package), 'sha256': module.sha256(package), 'apt_index_sha256_match': True}])
    write(folder / 'verification.json', dict(schema_version=1, status='CPU_IMPORT_PROVIDER_PASS_GPU_TRANSFER_UNTESTED',
          returncode=0, environment=env, library_hashes={'root/lib/libmlx5-rdmav34.so': module.sha256(provider)}))
    source = tmp_path / 'source.py'; source.write_text('# pinned\n')
    tasks = root / 'tasks.json'
    write(tasks, {'source_sha256': {str(source): module.sha256(source)}, 'tasks': [
        {'id': 'new-pd-label', 'phase': 'support', 'argv': ['--model', '8b', '--family', 'pd', '--out', str(root / 'preflight/new-pd-label')],
         'timeout': 2400, 'dependencies': []}]})
    return root, folder, tasks


def test_path_prefixes_merge_without_empty_or_duplicate_entries_and_exact_provider(module, platform):
    _, folder, _ = platform
    env, evidence = module.load_platform(folder / 'env.json', {'LD_LIBRARY_PATH': f'/old::{folder}/root/lib:/tail',
                                                             'RDMAV_DRIVERS': '/old/provider', 'PRIVATE_TOKEN': 'not-recorded'})
    assert env['LD_LIBRARY_PATH'] == f'{folder}/root/lib:/old:/tail'
    assert env['RDMAV_DRIVERS'] == f'{folder}/root/lib/libmlx5'
    assert 'PRIVATE_TOKEN' not in json.dumps(evidence)


@pytest.mark.parametrize('key', ['CUDA_VISIBLE_DEVICES', 'NCCL_IB_DISABLE', 'LD_PRELOAD', 'SGLANG_SKIP_P2P_CHECK'])
def test_environment_json_rejects_unvalidated_gpu_nccl_and_loader_overrides(module, platform, key):
    _, folder, _ = platform
    path = folder / 'env.json'; value = json.loads(path.read_text()); value['environment'][key] = '1'; write(path, value)
    with pytest.raises(ValueError, match='Unsupported environment'):
        module.load_platform(path, {})


def test_verification_and_installed_library_hashes_are_required(module, platform):
    _, folder, _ = platform
    (folder / 'root/lib/libmlx5-rdmav34.so').write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        module.load_platform(folder / 'env.json', {})


def test_dry_run_checks_pins_without_spawning_or_persisting(module, platform, capsys):
    root, folder, tasks = platform
    with mock.patch.dict(module.os.environ, {'CUDA_VISIBLE_DEVICES': ''}, clear=True), mock.patch.object(module, 'delegate', side_effect=AssertionError('launch forbidden')):
        assert module.main(['--tasks', str(tasks), '--queue-id', 'pd-first', '--root', str(root), '--dry-run']) == 0
    output = json.loads(capsys.readouterr().out)
    assert output['gpu_processes_started'] == 0 and output['status'] == 'PASS'
    assert not (root / 'queues/pd-first').exists()


def test_immutable_resume_accepts_identical_evidence_and_rejects_environment_command_or_pins(module, platform):
    root, folder, tasks = platform
    manifest, _ = module.prepare(tasks, 'pd-first', root=root, base_env={})
    path = root / 'queues/pd-first/platform-launch.json'
    module.persist_manifest(path, manifest)
    original = path.read_bytes()
    module.persist_manifest(path, manifest)
    assert path.read_bytes() == original
    for field, changed in [('public_environment', {'LD_LIBRARY_PATH': '/other'}), ('command', ['other']), ('source_sha256', {})]:
        with pytest.raises(ValueError, match='new queue ID'):
            module.persist_manifest(path, {**manifest, field: changed})
    assert path.read_bytes() == original


def test_existing_unrecorded_queue_cannot_gain_platform_provenance(module, platform):
    root, folder, tasks = platform
    manifest, _ = module.prepare(tasks, 'old-queue', root=root, base_env={})
    write(root / 'queues/old-queue/status.json', {'status': 'completed'})
    with pytest.raises(ValueError, match='new queue ID'):
        module.persist_manifest(root / 'queues/old-queue/platform-launch.json', manifest)


def test_launch_records_exact_command_before_delegating_with_merged_environment(module, platform):
    root, folder, tasks = platform
    def launch(command, env):
        saved = json.loads((root / 'queues/pd-first/platform-launch.json').read_text())
        assert saved['command'] == command
        assert saved['public_environment']['LD_LIBRARY_PATH'] == env['LD_LIBRARY_PATH']
        assert command[-1] == '--continue-independent'
        return 7
    with mock.patch.dict(module.os.environ, {'LD_LIBRARY_PATH': '/base'}, clear=True), mock.patch.object(module, 'delegate', side_effect=launch):
        assert module.main(['--tasks', str(tasks), '--queue-id', 'pd-first', '--root', str(root), '--continue-independent']) == 7


def test_plan_or_package_changes_cannot_be_laundered_through_resume(module, platform):
    root, folder, tasks = platform
    first, _ = module.prepare(tasks, 'pd-first', root=root, base_env={})
    path = root / 'queues/pd-first/platform-launch.json'; module.persist_manifest(path, first)
    plan = json.loads(tasks.read_text()); plan['tasks'][0]['timeout'] = 1200; write(tasks, plan)
    changed, _ = module.prepare(tasks, 'pd-first', root=root, base_env={})
    with pytest.raises(ValueError, match='new queue ID'):
        module.persist_manifest(path, changed)
    (folder / 'packages/provider.deb').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='hash mismatch'):
        module.prepare(tasks, 'new-id', root=root, base_env={})


def test_mooncake_transport_is_explicit_and_inherited_zero_does_not_enable_nvlink(module, platform):
    root, _, tasks = platform
    default, env = module.prepare(tasks, 'pd-first', root=root, base_env={'MC_INTRANODE_NVLINK': '0'})
    assert 'MC_INTRANODE_NVLINK' not in env
    assert default['mooncake_transport'] == 'default'
    assert default['public_environment']['MC_INTRANODE_NVLINK'] is None
    selected, env = module.prepare(tasks, 'pd-first', root=root, base_env={}, mooncake_transport='nvlink')
    assert env['MC_INTRANODE_NVLINK'] == '1'
    assert selected['public_environment']['MC_INTRANODE_NVLINK'] == '1'
    assert selected['mooncake_transport'] == 'nvlink'
    path = root / 'queues/pd-first/platform-launch.json'
    module.persist_manifest(path, default)
    with pytest.raises(ValueError, match='new queue ID'):
        module.persist_manifest(path, selected)


def provenance(module, root, tasks, queue_id, transport='default'):
    manifest, _ = module.prepare(tasks, queue_id, root=root, base_env={}, mooncake_transport=transport)
    module.persist_manifest(root / 'queues' / queue_id / 'platform-launch.json', manifest)
    plan = json.loads(tasks.read_text()); task = plan['tasks'][0]
    digest = module.hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(queue_id=queue_id, plan_sha256=digest, task=task,
                evidence=dict(source_sha256=plan['source_sha256'], trace_sha256=None))


def test_platform_identity_is_stable_across_queues_but_transport_is_a_configuration(module, platform):
    root, _, tasks = platform
    default = module.read_platform_provenance(root, provenance(module, root, tasks, 'default-one'))
    repeat = module.read_platform_provenance(root, provenance(module, root, tasks, 'default-two'))
    nvlink = module.read_platform_provenance(root, provenance(module, root, tasks, 'nvlink-one', 'nvlink'))
    assert default['software'] == repeat['software'] == nvlink['software']
    assert default['configuration'] == repeat['configuration'] != nvlink['configuration']
    assert default['manifest_sha256'] != repeat['manifest_sha256']


def test_platform_reader_rejects_changed_task_or_library_and_labels_legacy_missing(module, platform):
    root, folder, tasks = platform
    p = provenance(module, root, tasks, 'recorded')
    assert module.read_platform_provenance(root, {})['status'] == 'missing'
    changed = {**p, 'task': {**p['task'], 'timeout': 3}}
    with pytest.raises(ValueError, match='executed task'):
        module.read_platform_provenance(root, changed)
    (folder / 'root/lib/libmlx5-rdmav34.so').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='hash mismatch'):
        module.read_platform_provenance(root, p)
