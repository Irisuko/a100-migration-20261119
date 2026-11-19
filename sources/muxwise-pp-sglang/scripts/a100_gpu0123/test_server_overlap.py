"""CPU-only checks of staged overlap policy; never applies the runner patch."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parents[2]
SERVER = REPO / 'scripts/a100_gpu0123/server_run.py'
PROPOSAL = REPO / 'run/a100-gpu0123-20261114/proposals/native-overlap-v1'


@pytest.fixture
def proposed():
    source = SERVER.read_text()
    if 'def overlap_policy(' not in source and (PROPOSAL / 'server_run.py.candidate').exists():
        metadata = json.loads((PROPOSAL / 'proposal.json').read_text())
        assert hashlib.sha256(SERVER.read_bytes()).hexdigest() == metadata['base_sha256'], 'Runner changed; rebase the unapplied patch'
        source = (PROPOSAL / 'server_run.py.candidate').read_text()
        assert hashlib.sha256(source.encode()).hexdigest() == metadata['candidate_sha256']
    module = ModuleType('_overlap_candidate')
    module.__file__ = str(SERVER)
    with mock.patch('subprocess.Popen', side_effect=AssertionError('CPU test cannot launch a process')):
        exec(compile(source, str(SERVER), 'exec'), module.__dict__)
        yield module, source


@pytest.mark.parametrize('family,pp,supported', [
    ('plain',1,True), ('chunked',1,True), ('pd',1,True),
    ('plain',4,False), ('chunked',2,False), ('muxwise',1,False), ('muxpp',4,False)])
def test_auto_only_enables_supported_native_schedulers(proposed, family, pp, supported):
    module, _ = proposed
    policy = module.overlap_policy(family, pp, 'auto')
    assert policy == dict(requested='auto', effective='on' if supported else 'off',
                          enabled=supported, supported=supported)
    assert module.overlap_policy(family, pp, 'off')['enabled'] is False
    if supported:
        assert module.overlap_policy(family, pp, 'on')['enabled'] is True
    else:
        with pytest.raises(ValueError, match='overlap'):
            module.overlap_policy(family, pp, 'on')


def arguments(tmp_path, family='chunked', tp=4, pp=1, replicas=1):
    return ['server_run.py','--model','8b','--family',family,'--tp',str(tp),'--pp',str(pp),
            '--replicas',str(replicas),'--out',str(tmp_path/'never-created')]


def test_cli_default_remains_off_and_auto_metadata_is_resolved_before_controller(proposed, tmp_path):
    module, _ = proposed
    class Boundary(Exception):
        pass
    observed=[]
    def boundary():
        observed.append(sys._getframe(1).f_locals['overlap'].copy())
        raise Boundary()
    for options, expected in [([], 'off'), (['--overlap','auto'], 'on'), (['--overlap','on'], 'on')]:
        with mock.patch.object(sys,'argv',arguments(tmp_path)+options), mock.patch.object(module,'require_active_controller',new=boundary):
            with pytest.raises(Boundary):module.main()
        assert observed[-1]['effective'] == expected
    assert not (tmp_path/'never-created').exists()


@pytest.mark.parametrize('family,tp,pp,replicas', [('chunked',1,4,1),('muxwise',4,1,1),('muxpp',1,4,1)])
def test_explicit_on_rejects_pp_and_mux_before_controller(proposed, tmp_path, family, tp, pp, replicas):
    module, _ = proposed
    with mock.patch.object(sys,'argv',arguments(tmp_path,family,tp,pp,replicas)+['--overlap','on']), mock.patch.object(module,'require_active_controller',side_effect=AssertionError('must reject before controller')):
        with pytest.raises(ValueError, match='overlap'):module.main()
    assert not (tmp_path/'never-created').exists()


@pytest.mark.parametrize('override',['--disable-overlap-schedule','--disable-overlap','--disable-overlap-schedule=true'])
def test_extra_cannot_contradict_recorded_overlap_policy(proposed,tmp_path,override):
    module, _ = proposed
    with mock.patch.object(sys,'argv',arguments(tmp_path)+['--overlap','auto','--extra',override]), mock.patch.object(module,'require_active_controller',side_effect=AssertionError('must reject before controller')):
        with pytest.raises(ValueError,match='overlap'):module.main()


@pytest.mark.parametrize('family,pp,requested,disabled',[('plain',1,'off',True),('chunked',1,'auto',False),('pd',1,'on',False),('chunked',4,'auto',True),('muxpp',4,'auto',True)])
def test_actual_worker_command_and_saved_policy_match(proposed,family,pp,requested,disabled):
    module, source = proposed
    tree=ast.parse(source)
    main=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='main')
    assignment=next(node for node in ast.walk(main) if isinstance(node,ast.Assign) and isinstance(node.value,ast.List)
                    and any(isinstance(item,ast.Constant) and item.value=='sglang.launch_server' for item in node.value.elts))
    policy=module.overlap_policy(family,pp,requested)
    namespace={**module.__dict__,'model':Path('/model'),'port':32100,'overlap':policy,
               'a':SimpleNamespace(tp=4//pp,pp=pp,mem=.85,running=128,graph=16)}
    command=eval(compile(ast.Expression(assignment.value),'<worker-command>','eval'),namespace)
    assert command.count('--disable-overlap-schedule') == int(disabled)
    saved=next(node for node in ast.walk(main) if isinstance(node,ast.Assign)
               and any(isinstance(target,ast.Subscript) and isinstance(target.value,ast.Name) and target.value.id=='metadata'
                       and isinstance(target.slice,ast.Constant) and target.slice.value=='overlap_policy' for target in node.targets))
    namespace['metadata']={}
    exec(compile(ast.Module(body=[saved],type_ignores=[]),'<metadata>','exec'),namespace)
    assert namespace['metadata']['overlap_policy'] == policy


def test_staged_patch_exactly_matches_saved_base_and_candidate():
    metadata=json.loads((PROPOSAL/'proposal.json').read_text())
    before=(PROPOSAL/'server_run.py.base').read_text();after=(PROPOSAL/'server_run.py.candidate').read_text()
    assert hashlib.sha256(before.encode()).hexdigest()==metadata['base_sha256']
    assert hashlib.sha256(after.encode()).hexdigest()==metadata['candidate_sha256']
    patch=''.join(difflib.unified_diff(before.splitlines(keepends=True),after.splitlines(keepends=True),
                    fromfile='a/scripts/a100_gpu0123/server_run.py',tofile='b/scripts/a100_gpu0123/server_run.py'))
    assert (PROPOSAL/'server-overlap.patch').read_text()==patch
