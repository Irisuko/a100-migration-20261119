"""Only the reviewed runner SHA pair may reuse an unchanged off-mode preflight."""
import importlib.util
import json
from pathlib import Path

import pytest
import build_development

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
PROPOSAL=REPO/'run/a100-gpu0123-20261114/proposals/native-overlap-v1'


@pytest.fixture
def module():
    path=HERE/'serving_compatibility.py'
    assert path.is_file(), 'Compatibility verifier is missing'
    spec=importlib.util.spec_from_file_location('compatibility_test',path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


@pytest.fixture
def checked(module):
    current={key:value for key,value in build_development.current_sources(REPO).items() if not key.startswith('python/sglang/')}
    current.update(json.loads((PROPOSAL/'compatibility.json').read_text())['runtime_source_sha256'])
    current[module.TARGET]=module.CANDIDATE
    return module.load(PROPOSAL/'compatibility.json',current,REPO),current


def metadata(module,current,family='chunked',pp=4):
    old={**current,module.TARGET:module.BASE}
    return dict(source_sha256=old,args=dict(family=family,tp=4//pp,pp=pp,replicas=1,extra=[]),
                commands=[{'command':['python','-m','sglang.launch_server','--disable-overlap-schedule']}])


@pytest.mark.parametrize('family,pp,requested',[('chunked',4,'auto'),('muxpp',4,'off'),('muxwise',1,'off')])
def test_only_off_to_off_known_transition_reuses_evidence(module,checked,family,pp,requested):
    compatibility,current=checked
    result=module.check_support(metadata(module,current,family,pp),current,compatibility,requested)
    assert result['kind']=='reviewed_default_off_compatibility'
    assert result['source_differences']==[module.TARGET]
    assert result['effective_overlap']=='off'


@pytest.mark.parametrize('family',['plain','chunked','pd'])
def test_enabling_native_overlap_always_requires_new_preflight(module,checked,family):
    compatibility,current=checked
    with pytest.raises(ValueError,match='new preflight'):
        module.check_support(metadata(module,current,family,1),current,compatibility,'auto')


@pytest.mark.parametrize('kind',['unknown_path','unknown_hash','runtime_change','missing_disable_flag'])
def test_no_other_source_or_mode_relaxation(module,checked,kind):
    compatibility,current=checked
    saved=metadata(module,current)
    if kind=='unknown_path':saved['source_sha256']['unknown.py']='0'*64
    elif kind=='unknown_hash':saved['source_sha256'][module.TARGET]='0'*64
    elif kind=='runtime_change':saved['source_sha256'][next(k for k in current if k.startswith('python/sglang/'))]='0'*64
    else:saved['commands'][0]['command'].remove('--disable-overlap-schedule')
    with pytest.raises(ValueError):module.check_support(saved,current,compatibility,'auto')


def test_exact_current_source_requires_matching_explicit_native_overlap_evidence(module,checked):
    compatibility,current=checked
    saved=metadata(module,current,'chunked',1);saved['source_sha256']=current
    saved['args']['overlap']='auto'
    saved['overlap_policy']=dict(requested='auto',effective='on',enabled=True,supported=True)
    saved['commands'][0]['command'].remove('--disable-overlap-schedule')
    assert module.check_support(saved,current,compatibility,'auto')['kind']=='exact_current_source'
    saved['args']['overlap']='off'
    with pytest.raises(ValueError):module.check_support(saved,current,compatibility,'auto')


def test_artifact_cannot_expand_allowed_path_hash_or_runtime_snapshot(module,checked,tmp_path):
    _,current=checked
    original=json.loads((PROPOSAL/'compatibility.json').read_text())
    for key,value in [('target','another.py'),('base_sha256','0'*64),('candidate_sha256','0'*64),('runtime_fingerprint','0'*64)]:
        path=tmp_path/(key+'.json');path.write_text(json.dumps({**original,key:value}))
        with pytest.raises(ValueError):module.load(path,current,REPO)
    altered={**current,next(k for k in current if k.startswith('python/sglang/')):'0'*64}
    with pytest.raises(ValueError,match='runtime'):module.load(PROPOSAL/'compatibility.json',altered,REPO)


def test_target_runner_must_be_applied_before_opt_in_generation(module):
    current=build_development.current_sources(REPO);current[module.TARGET]=module.BASE
    with pytest.raises(ValueError,match='applied'):
        module.load(PROPOSAL/'compatibility.json',current,REPO)


def test_actual_server_flags_must_match_effective_mode_for_every_instance(module,tmp_path):
    saved={'args':{'replicas':2}}
    for index in range(2):
        file=tmp_path/f'worker{index}/server-final.json';file.parent.mkdir()
        file.write_text(json.dumps({'disable_overlap_schedule':False}))
    assert len(module.check_actual_overlap(tmp_path,saved,'on'))==2
    (tmp_path/'worker1/server-final.json').write_text(json.dumps({'disable_overlap_schedule':True}))
    with pytest.raises(ValueError,match='actual'):
        module.check_actual_overlap(tmp_path,saved,'on')
    (tmp_path/'worker1/server-final.json').unlink()
    with pytest.raises(ValueError,match='actual'):
        module.check_actual_overlap(tmp_path,saved,'on')
