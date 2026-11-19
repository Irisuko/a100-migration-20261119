"""Lifecycle and cache-state regressions without model/CUDA subprocesses."""
from pathlib import Path
from unittest import mock
import json
import ast
import asyncio
import dataclasses
import os
import sys
from types import SimpleNamespace
import pytest
import server_run


class Process:
    def __init__(self, code=None):
        self.returncode=code
    def poll(self):
        return self.returncode


def test_trace_preparation_flushes_each_replica_after_prewarm():
    workers=[{'port':32101},{'port':32102}]
    calls=[]
    with mock.patch.object(server_run,'request',side_effect=lambda port,path,timeout: calls.append((port,path)) or 'Cache flushed. Please check server logs for more details.'):
        result=server_run.prepare_trace(workers)
    assert calls == [(32101,'/flush_cache'),(32102,'/flush_cache')]
    assert result['worker_ports'] == [32101,32102]
    assert result['client_warmup_requests'] == 2
    assert result['status'] == 'acknowledged'


def test_trace_cache_flush_failure_cannot_be_recorded_as_ready():
    with mock.patch.object(server_run,'request',return_value='Failed to flush cache. There are pending requests.'):
        with pytest.raises(RuntimeError,match='not acknowledged'):
            server_run.prepare_trace([{'port':32101}])


@pytest.mark.parametrize('code',[0,1,-9])
def test_trace_replay_rejects_any_exited_server_even_if_client_succeeded(code):
    with pytest.raises(RuntimeError,match='worker0'):
        server_run.wait_client(Process(0),[(Process(code),'worker0')])


def test_trace_replay_preserves_client_failure_with_live_servers():
    with pytest.raises(RuntimeError,match='client exited'):
        server_run.wait_client(Process(2),[(Process(),'worker0')])


def test_trace_replay_requires_successful_client_with_all_services_live():
    assert server_run.wait_client(Process(0),[(Process(),'worker0'),(Process(),'monitor')]) == 0

def test_resource_markers_measure_quiet_recovery_before_and_after_flush(tmp_path):
    calls=[]
    def request(port,path,timeout):
        calls.append((port,path))
        return 'Cache flushed.' if path=='/flush_cache' else {'tokenizer_pending_requests':0,'tokenizer_finished_states':0}
    def pause(seconds):
        assert seconds >= 1.2
        flushes=[x for x in calls if x[1]=='/flush_cache']
        assert flushes in ([],[(32101,'/flush_cache'),(32102,'/flush_cache')])
    with mock.patch.object(server_run,'request',side_effect=request), \
         mock.patch.object(server_run.time,'sleep',side_effect=pause), \
         mock.patch.object(server_run.time,'time',side_effect=[10.,11.5,12.,13.5]):
        server_run.resource_phase(tmp_path,'after-cancel',[32101,32102])
    assert json.loads((tmp_path/'resource-phase-markers.json').read_text()) == [
        {'phase':'after-cancel-pre-flush','start_unix':10.,'end_unix':11.5},
        {'phase':'after-cancel','start_unix':12.,'end_unix':13.5}]
    assert calls[:2]==[(32101,'/get_server_info'),(32102,'/get_server_info')]

def test_resource_phase_cannot_hide_tokenizer_leak_with_flush(tmp_path):
    calls=[]
    def request(port,path,timeout):
        calls.append(path)
        return {'tokenizer_pending_requests':3,'tokenizer_finished_states':3}
    with mock.patch.object(server_run,'request',side_effect=request), \
         mock.patch.object(server_run.time,'sleep',return_value=None):
        with pytest.raises(RuntimeError,match='tokenizer'):
            server_run.resource_phase(tmp_path,'after-cancel',[32101])
    assert '/flush_cache' not in calls
    assert json.loads((tmp_path/'after-cancel-server-before-flush.json').read_text())['32101']['tokenizer_pending_requests']==3

@pytest.mark.parametrize('enabled',[True,False])
def test_server_info_exposes_opt_in_tokenizer_state_counts(enabled):
    path=Path(__file__).resolve().parents[2]/'python/sglang/srt/entrypoints/http_server.py'
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.AsyncFunctionDef) and n.name=='get_server_info')
    node.decorator_list=[]
    @dataclasses.dataclass
    class Args:
        tp_size:int=1
    async def internal():return []
    manager=SimpleNamespace(server_args=Args(),get_internal_state=internal,
                            rid_to_state={'active':SimpleNamespace(finished=False),'leaked':SimpleNamespace(finished=True)})
    scope={'dataclasses':dataclasses,'os':os,'__version__':'test',
           '_global_state':SimpleNamespace(tokenizer_manager=manager,scheduler_info={})}
    module=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
    exec(compile(module,str(path),'exec'),scope)
    with mock.patch.dict(os.environ,{'SGLANG_A100_RESOURCE_AUDIT':'1' if enabled else '0'}):
        result=asyncio.run(scope['get_server_info']())
    if enabled:
        assert result['tokenizer_pending_requests']==2
        assert result['tokenizer_finished_states']==1
    else:
        assert 'tokenizer_pending_requests' not in result

def test_source_equivalence_excludes_reporting_tests_but_tracks_serving_code(tmp_path):
    paths=['python/sglang/runtime.py','scripts/a100_gpu0123/server_run.py',
           'scripts/a100_gpu0123/report.py','scripts/a100_gpu0123/test_example.py',
           'scripts/a100_gpu0123/bench/bench_client.py','scripts/a100_gpu0123/bench/test_example.py']
    for name in paths:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('original')
    original=server_run.source_provenance(tmp_path)
    (tmp_path/'scripts/a100_gpu0123/report.py').write_text('updated report')
    (tmp_path/'scripts/a100_gpu0123/test_example.py').write_text('updated test')
    changed_tools=server_run.source_provenance(tmp_path)
    assert changed_tools['source_sha256']==original['source_sha256']
    assert changed_tools['tool_sha256']!=original['tool_sha256']
    (tmp_path/'scripts/a100_gpu0123/bench/bench_client.py').write_text('changed replay')
    assert server_run.source_provenance(tmp_path)['source_sha256']!=original['source_sha256']
    assert list(original['runtime_source_sha256'])==['python/sglang/runtime.py']

@pytest.mark.parametrize('data,expected',[
    ({'official_revision':'official','revision':'fallback'},'official'),
    ({'revision':'fallback'},'fallback'),({},None)])
def test_model_revision_uses_official_download_provenance(tmp_path,data,expected):
    (tmp_path/'download-provenance.json').write_text(json.dumps(data))
    assert server_run.model_revision(tmp_path)==expected

def test_effective_command_config_hash_uses_last_override_including_equals_form(tmp_path):
    first=tmp_path/'generated.yml';first.write_text('first')
    last=tmp_path/'override.yml';last.write_text('last')
    before=server_run.configuration_provenance(['--pdmux-config-path',first])
    after=server_run.configuration_provenance(['--pdmux-config-path',first,'--pdmux-config-path='+str(last)])
    assert before['path']==str(first.resolve())
    assert after['path']==str(last.resolve())
    assert before['sha256']!=after['sha256']
    assert server_run.configuration_provenance(['--chunked-prefill-size','2048']) is None

def test_numerical_audit_routes_each_plain_replica_and_requires_full_coverage(tmp_path):
    calls=[]
    def audit(port,cases,output,expected_metadata):
        calls.append((port,cases,output,expected_metadata))
        return {'completed':True,'full_token_coverage':True,'exact':False}
    args=SimpleNamespace(tp=2,pp=1)
    with mock.patch.dict(sys.modules,{'numerical_audit':SimpleNamespace(audit=audit)}):
        server_run.run_numerical_audits([{'port':32101,'index':0},{'port':32102,'index':1}],
                                        Path('/model'),Path('/cases.json'),tmp_path,args)
    assert [c[0] for c in calls]==[32101,32102]
    assert len({c[2] for c in calls})==2
    assert calls[0][3]=={'model_path':'/model','tp_size':2,'pp_size':1,'dtype':'bfloat16','enable_pdmux':False}
    with mock.patch.dict(sys.modules,{'numerical_audit':SimpleNamespace(audit=lambda *args,**kwargs:{'completed':False,'full_token_coverage':False})}):
        with pytest.raises(RuntimeError,match='coverage'):
            server_run.run_numerical_audits([{'port':32101,'index':0}],Path('/model'),Path('/cases.json'),tmp_path,args)

def test_numerical_audit_is_rejected_for_nonplain_family_before_controller_access():
    args=['server_run.py','--model','8b','--family','muxpp','--tp','1','--pp','4',
          '--oracle-cases','/cases.json','--out','/never-created']
    with mock.patch.object(sys,'argv',args),mock.patch.object(server_run,'require_active_controller',
         side_effect=AssertionError('invalid oracle must be rejected before control')):
        with pytest.raises(ValueError,match='plain'):
            server_run.main()

def test_stress_inputs_are_distinct_bounded_deterministic_and_do_not_repeat_long_cases():
    class Tokenizer:
        def encode(self,text,add_special_tokens=False):
            return list(text.encode())
    with mock.patch('transformers.AutoTokenizer.from_pretrained',return_value=Tokenizer()):
        first=server_run.stress_cases(Path('/model'))
        second=server_run.stress_cases(Path('/model'))
    assert len(first)==200
    assert first==second
    assert len({tuple(ids) for ids in first})==200
    assert min(map(len,first))==128
    assert max(map(len,first))==4096
    assert all(ids[0]==128000 for ids in first)
