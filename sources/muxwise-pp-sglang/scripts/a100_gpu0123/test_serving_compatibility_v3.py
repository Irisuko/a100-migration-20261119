"""Final combined runtime transitions require exact proofs and inactive modes."""
import json
from pathlib import Path

import pytest

import build_development as builder
import serving_compatibility as compat
import test_build_development as build_tests
import test_serving_compatibility_v2 as prior
from _compatibility_test_fixtures import populate_runtime
from test_build_development import write

V3=prior.PROPOSALS/'serving-compatibility-v3/compatibility.json'
CLIENT='python/sglang/srt/managers/tp_worker_overlap_thread.py'
CLIENT_OLD='bfc0e817bbfd46e3ea47c36b036c52cf8ae599ae08778f78653bb237fbfcf551'
CLIENT_NEW='59fa602db9960a4cb3a4f6f17b83b7654f901b80c379ccfdcae133cd4f8577f3'
PD='python/sglang/srt/disaggregation/mooncake/conn.py'
PD_OLD='02ce2e35a0c37179e09197c7d48554699cbea6cdd80b891dc068024696310ff7'
PD_NEW='bf12fa235bb22860c507bdc62742211a8cdc481143a83735d141e85d307a690c'


def prepare_repo(repo):
    populate_runtime(repo,json.loads(prior.V1.read_text())['runtime_source_sha256'],{
        prior.SCHEDULER:prior.PROPOSALS/'muxwise-abort-v1/scheduler.py.candidate',
        CLIENT:prior.PROPOSALS/'native-overlap-dispatch-v1/tp_worker_overlap_thread.py.candidate',
        PD:prior.PROPOSALS/'pd-nvlink-aux-tcp-v1/conn.py.candidate'})
    path=repo/compat.TARGET;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes((prior.PROPOSALS/'native-overlap-v1/server_run.py.candidate').read_bytes())


@pytest.fixture
def checked(tmp_path):
    repo=tmp_path/'repo';prepare_repo(repo);current=builder.current_sources(repo)
    return repo,current,compat.load(V3,current,repo)


def metadata(current,family='chunked',pp=4,overlap='off',runner_old=False):
    result=prior.metadata(current,family,pp,runner_old=runner_old,overlap=overlap)
    result['source_sha256'].update({CLIENT:CLIENT_OLD,PD:PD_OLD})
    return result


@pytest.mark.parametrize('family,pp,requested',[('plain',4,'auto'),('chunked',4,'auto'),('muxpp',2,'off'),('muxpp',4,'off'),('chunked',1,'off')])
@pytest.mark.parametrize('runner_old',[False,True])
def test_only_unaffected_off_non_pd_families_cross_combined_runtime(checked,family,pp,requested,runner_old):
    _,current,proof=checked
    result=compat.check_support(metadata(current,family,pp,runner_old=runner_old),current,proof,requested)
    assert result['scheduler_transition'] and result['native_overlap_client_transition'] and result['pd_transport_transition']
    assert result['effective_overlap']=='off'


@pytest.mark.parametrize('family,pp,overlap,requested',[('muxwise',1,'off','off'),('pd',1,'off','off'),('chunked',1,'auto','auto'),('pd',1,'auto','auto')])
def test_all_changed_modes_require_fresh_preflight(checked,family,pp,overlap,requested):
    _,current,proof=checked
    with pytest.raises(ValueError,match='preflight|PD|MuxWise|native'):
        compat.check_support(metadata(current,family,pp,overlap),current,proof,requested)


@pytest.mark.parametrize('family,overlap',[('muxwise','off'),('pd','auto'),('chunked','auto')])
def test_exact_final_source_accepts_fresh_evidence_in_affected_modes(checked,family,overlap):
    _,current,proof=checked;saved=prior.metadata(current,family,1,scheduler_old=False,overlap=overlap)
    result=compat.check_support(saved,current,proof,overlap)
    assert result['kind']=='exact_current_source' and not any(result[k] for k in ('scheduler_transition','native_overlap_client_transition','pd_transport_transition'))


@pytest.mark.parametrize('key',[CLIENT,PD,prior.SCHEDULER,'python/sglang/srt/utils.py','unknown.py'])
def test_unknown_source_keys_and_hashes_are_never_compatible(checked,key):
    _,current,proof=checked;saved=metadata(current);saved['source_sha256'][key]='0'*64
    with pytest.raises(ValueError):compat.check_support(saved,current,proof,'auto')


@pytest.mark.parametrize('key,base_file',[(CLIENT,'native-overlap-dispatch-v1/tp_worker_overlap_thread.py.base'),(PD,'pd-nvlink-aux-tcp-v1/conn.py.base')])
def test_every_candidate_must_be_applied_before_v3_generation(checked,key,base_file):
    repo,current,_=checked;(repo/key).write_bytes((prior.PROPOSALS/base_file).read_bytes())
    with pytest.raises(ValueError,match='live|applied'):compat.load(V3,current,repo)


def test_abort_only_v2_cannot_load_combined_runtime(checked):
    repo,current,_=checked
    with pytest.raises(ValueError,match='Runtime|runtime'):compat.load(prior.V2,current,repo)


def test_combined_proofs_reject_unreviewed_method_edits():
    client=prior.PROPOSALS/'native-overlap-dispatch-v1';pd=prior.PROPOSALS/'pd-nvlink-aux-tcp-v1'
    assert compat.prove_overlap_client_scope((client/'tp_worker_overlap_thread.py.base').read_text(),(client/'tp_worker_overlap_thread.py.candidate').read_text())['unchanged_existing_function_body']
    assert compat.prove_pd_scope((pd/'conn.py.base').read_text(),(pd/'conn.py.candidate').read_text())['whole_module_ast_equal']
    with pytest.raises(ValueError):compat.prove_pd_scope((pd/'conn.py.base').read_text(),(pd/'conn.py.candidate').read_text()+'\nunreviewed=True\n')


def test_actual_non_pd_scope_must_be_present_on_every_worker(tmp_path):
    saved=dict(args=dict(family='chunked',replicas=2))
    for i in range(2):write(tmp_path/f'worker{i}/server-final.json',dict(disaggregation_mode='null'))
    assert len(compat.check_actual_non_pd_scope(tmp_path,saved))==2
    write(tmp_path/'worker1/server-final.json',dict(disaggregation_mode='prefill'))
    with pytest.raises(ValueError):compat.check_actual_non_pd_scope(tmp_path,saved)
    write(tmp_path/'worker1/server-final.json',{})
    with pytest.raises(ValueError):compat.check_actual_non_pd_scope(tmp_path,saved)


def test_builder_pins_all_combined_evidence_and_checks_actual_non_pd_scope():
    fixture=build_tests.DevelopmentBuilderTests();fixture.setUp()
    try:
        runtime=fixture.repo/'python/sglang';(runtime/'engine.py').unlink();runtime.rmdir();prepare_repo(fixture.repo)
        current=builder.current_sources(fixture.repo);saved=metadata(current,'muxpp',4,runner_old=True)
        fixture.metadata['source_sha256']=saved['source_sha256'];fixture.metadata['commands']=saved['commands']
        write(fixture.run/'run.json',fixture.metadata)
        write(fixture.run/'worker0/server-final.json',dict(disable_overlap_schedule=True,enable_pdmux=True,pp_size=4,disaggregation_mode='null'))
        plan=json.loads(fixture.plan.read_text());plan['source_sha256']={prior.SCHEDULER:prior.OLD,compat.TARGET:compat.BASE,CLIENT:CLIENT_OLD,PD:PD_OLD};write(fixture.plan,plan)
        queue,protocol=fixture.build(serving_compatibility_path=V3)
        assert len(queue['tasks'])==2 and queue['source_sha256'][CLIENT]==CLIENT_NEW and queue['source_sha256'][PD]==PD_NEW
        assert all(str(path) in queue['source_sha256'] for path in (V3,prior.V2,prior.V1))
        assert protocol['eligibility'][0]['support_source_compatibility']['pd_transport_transition']
        write(fixture.run/'worker0/server-final.json',dict(disable_overlap_schedule=True,enable_pdmux=True,pp_size=4,disaggregation_mode='decode'))
        assert fixture.build(serving_compatibility_path=V3)[0]['tasks']==[]
    finally:fixture.doCleanups()
