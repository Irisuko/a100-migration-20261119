"""Exact staged scheduler transition, with no live runtime mutation or GPU use."""
import json
from pathlib import Path

import pytest

import build_development as builder
import serving_compatibility as compat
import test_build_development as build_tests
from test_build_development import write
from _compatibility_test_fixtures import populate_runtime

REPO=Path(__file__).resolve().parents[2]
PROPOSALS=REPO/'run/a100-gpu0123-20261114/proposals'
V1=PROPOSALS/'native-overlap-v1/compatibility.json'
V2=PROPOSALS/'serving-compatibility-v2/compatibility.json'
SCHEDULER='python/sglang/srt/managers/scheduler.py'
OLD='df664cd14626ee84e70b5b871ffed794f142b3d65e560af9f4cfef11d0681f27'
NEW='a1d9decd0fd22343a96d3b4865fb60312a3128a0d6d06d15be53dc635f21caab'


def prepare_repo(repo):
    # Symlink immutable runtime files individually; candidate is a private copy.
    populate_runtime(repo,json.loads(V1.read_text())['runtime_source_sha256'],
                     {SCHEDULER:PROPOSALS/'muxwise-abort-v1/scheduler.py.candidate'})
    path=repo/compat.TARGET;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes((PROPOSALS/'native-overlap-v1/server_run.py.candidate').read_bytes())


@pytest.fixture
def checked(tmp_path):
    repo=tmp_path/'repo';prepare_repo(repo)
    current=builder.current_sources(repo)
    return repo,current,compat.load(V2,current,repo)


def metadata(current,family='chunked',pp=1,runner_old=False,scheduler_old=True,overlap='auto'):
    source=dict(current)
    if runner_old:source[compat.TARGET]=compat.BASE
    if scheduler_old:source[SCHEDULER]=OLD
    mode=compat.effective(family,pp,overlap)
    args=dict(family=family,pp=pp,tp=4//pp,replicas=1,extra=[])
    result=dict(source_sha256=source,args=args,commands=[dict(command=['python','-m','sglang.launch_server']+
                       (['--disable-overlap-schedule'] if mode=='off' else [])+
                       (['--enable-pdmux'] if family in ('muxwise','muxpp') else []))])
    if not runner_old:
        args['overlap']=overlap
        result['overlap_policy']=dict(requested=overlap,effective=mode,enabled=mode=='on',supported=family in ('plain','chunked','pd') and pp==1)
    return result


@pytest.mark.parametrize('family,pp,requested',[('plain',4,'auto'),('chunked',4,'auto'),('muxpp',4,'off')])
def test_composed_old_runner_and_scheduler_transition_for_unaffected_off_modes(checked,family,pp,requested):
    repo,current,proof=checked
    result=compat.check_support(metadata(current,family,pp,runner_old=True,overlap='off'),current,proof,requested)
    assert set(result['source_differences'])=={compat.TARGET,SCHEDULER}
    assert result['scheduler_transition'] and result['effective_overlap']=='off'


@pytest.mark.parametrize('family',['plain','chunked','pd'])
def test_current_runner_native_on_crosses_only_unaffected_scheduler_transition(checked,family):
    _,current,proof=checked
    result=compat.check_support(metadata(current,family),current,proof,'auto')
    assert result['source_differences']==[SCHEDULER] and result['effective_overlap']=='on'


@pytest.mark.parametrize('runner_old',[False,True])
def test_old_muxwise_never_crosses_scheduler_fix(checked,runner_old):
    _,current,proof=checked
    with pytest.raises(ValueError,match='MuxWise|Muxwise|muxwise'):
        compat.check_support(metadata(current,'muxwise',runner_old=runner_old,overlap='off'),current,proof,'off')


def test_old_off_still_cannot_qualify_native_auto_on(checked):
    _,current,proof=checked
    with pytest.raises(ValueError,match='new preflight'):
        compat.check_support(metadata(current,runner_old=True,overlap='off'),current,proof,'auto')


def test_fresh_muxwise_current_source_is_allowed(checked):
    _,current,proof=checked
    result=compat.check_support(metadata(current,'muxwise',scheduler_old=False,overlap='off'),current,proof,'off')
    assert result['kind']=='exact_current_source' and not result['scheduler_transition']


@pytest.mark.parametrize('change',['extra_key','unknown_scheduler','another_runtime','unknown_runner'])
def test_v2_never_relaxes_other_source_keys_or_hashes(checked,change):
    _,current,proof=checked;saved=metadata(current)
    key={'extra_key':'unknown.py','unknown_scheduler':SCHEDULER,'another_runtime':'python/sglang/srt/utils.py','unknown_runner':compat.TARGET}[change]
    saved['source_sha256'][key]='0'*64
    with pytest.raises(ValueError):compat.check_support(saved,current,proof,'auto')


def test_v2_requires_live_candidate_not_a_spoofed_future_source_map(checked):
    repo,current,_=checked
    (repo/SCHEDULER).write_bytes((PROPOSALS/'muxwise-abort-v1/scheduler.py.base').read_bytes())
    with pytest.raises(ValueError,match='applied|live'):
        compat.load(V2,current,repo)


def test_v1_is_preserved_and_cannot_claim_new_runtime(checked):
    repo,current,_=checked
    assert compat.sha(V1)=='5e52cc885ad3b623d837fe7cbdde479e55f0a92ce4ae24720c0d9880a8e8571c'
    with pytest.raises(ValueError,match='runtime'):compat.load(V1,current,repo)


@pytest.mark.parametrize('mutation',['scheduler_hash','target','runtime_hash','proof'])
def test_artifact_cannot_expand_the_exact_two_proofs(checked,tmp_path,mutation):
    repo,current,_=checked;data=json.loads(V2.read_text())
    if mutation=='scheduler_hash':data['scheduler_transition']['base_sha256']='0'*64
    elif mutation=='target':data['scheduler_transition']['target']='another.py'
    elif mutation=='runtime_hash':data['candidate_runtime_fingerprint']='0'*64
    else:data['scheduler_transition']['ast_proof']={}
    path=tmp_path/'bad.json';path.write_text(json.dumps(data))
    with pytest.raises(ValueError):compat.load(path,current,repo)


def test_scheduler_proof_rejects_any_other_edit():
    base=(PROPOSALS/'muxwise-abort-v1/scheduler.py.base').read_text()
    candidate=(PROPOSALS/'muxwise-abort-v1/scheduler.py.candidate').read_text()
    assert compat.prove_scheduler_scope(base,candidate)['removed_nodes']=={'flush_guard':1,'abort_enumeration':1}
    with pytest.raises(ValueError):compat.prove_scheduler_scope(base,candidate+'\nchanged = True\n')


def test_actual_scheduler_scope_checks_every_replica_without_weakening_overlap(tmp_path):
    saved=dict(args=dict(family='chunked',pp=1,replicas=2))
    for index in range(2):
        write(tmp_path/f'worker{index}/server-final.json',dict(disable_overlap_schedule=False,enable_pdmux=False,pp_size=1))
    assert len(compat.check_actual_scheduler_scope(tmp_path,saved))==2
    assert len(compat.check_actual_overlap(tmp_path,saved,'on'))==2
    write(tmp_path/'worker1/server-final.json',dict(disable_overlap_schedule=False,enable_pdmux=True,pp_size=1))
    with pytest.raises(ValueError):compat.check_actual_scheduler_scope(tmp_path,saved)


def test_builder_pins_composed_proof_and_preserves_api_checks():
    fixture=build_tests.DevelopmentBuilderTests();fixture.setUp()
    try:
        runtime=fixture.repo/'python/sglang';(runtime/'engine.py').unlink();runtime.rmdir()
        prepare_repo(fixture.repo)
        current=builder.current_sources(fixture.repo)
        saved=metadata(current,'muxpp',4,runner_old=True,overlap='off')
        fixture.metadata['source_sha256']=saved['source_sha256'];fixture.metadata['commands']=saved['commands']
        write(fixture.run/'run.json',fixture.metadata)
        write(fixture.run/'worker0/server-final.json',dict(disable_overlap_schedule=True,enable_pdmux=True,pp_size=4))
        plan=json.loads(fixture.plan.read_text());plan['source_sha256']={SCHEDULER:OLD,compat.TARGET:compat.BASE};write(fixture.plan,plan)
        queue,protocol=fixture.build(serving_compatibility_path=V2)
        assert len(queue['tasks'])==2 and queue['source_sha256'][SCHEDULER]==NEW
        assert str(V2) in queue['source_sha256'] and str(V1) in queue['source_sha256']
        assert protocol['eligibility'][0]['support_source_compatibility']['scheduler_transition']
        write(fixture.run/'worker0/server-final.json',dict(disable_overlap_schedule=False,enable_pdmux=True,pp_size=4))
        assert fixture.build(serving_compatibility_path=V2)[0]['tasks']==[]
    finally:fixture.doCleanups()
