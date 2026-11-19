"""CPU checks for real dispatch decisions and benchmark evidence gates."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import custom_dispatch_bench as bench


def test_fixed_four_rank_plans_have_tp4_and_two_concurrent_tp2_pairs():
    assert bench.plans()=={'tp4':[[0,1,2,3]],'tp2_pairs':[[0,1],[2,3]]}


def real_eligibility():
    path=Path(__file__).resolve().parents[2]/'python/sglang/srt/distributed/device_communicators/custom_all_reduce.py'
    cls=next(x for x in ast.parse(path.read_text()).body if isinstance(x,ast.ClassDef) and x.name=='CustomAllreduce')
    node=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='should_custom_ar')
    node.returns=None
    for arg in node.args.args:arg.annotation=None
    scope={'_is_hip':False,'is_weak_contiguous':lambda tensor:tensor.contiguous}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'actual-runtime-eligibility','exec'),scope)
    return scope['should_custom_ar']


class Tensor:
    contiguous=True
    def __init__(self,nbytes):self.nbytes=nbytes
    def numel(self):return self.nbytes//2
    def element_size(self):return 2


class Custom:
    disabled=False;world_size=4;full_nvlink=True;max_size=8388608
    def should_custom_ar(self,value):return real_eligibility()(self,value)
    def custom_all_reduce(self,value):
        if not self.should_custom_ar(value):return None
        return ('custom-result',value.nbytes)


class Nccl:
    def all_reduce(self,value):value.reduced=True


@pytest.mark.parametrize('nbytes,kind',[(8,'nccl'),(16384,'custom'),(262144,'custom'),(4194304,'custom'),(33554432,'nccl')])
def test_actual_runtime_size_eligibility_drives_custom_or_nccl(nbytes,kind):
    dispatcher=bench.Dispatcher(Custom(),Nccl(),'custom_dispatch',lambda:False)
    tensor=Tensor(nbytes);result=dispatcher(tensor)
    if kind=='custom':assert result==('custom-result',nbytes)
    else:assert result is tensor and tensor.reduced
    assert dispatcher.counts[f'eager_{kind}']==1
    assert sum(dispatcher.counts.values())==1


def test_forced_nccl_has_no_custom_dispatch_even_for_supported_payload():
    dispatcher=bench.Dispatcher(Custom(),Nccl(),'nccl',lambda:False)
    tensor=Tensor(16384)
    assert dispatcher(tensor) is tensor and tensor.reduced
    assert dispatcher.counts['eager_custom']==0


def test_capture_counts_are_not_reported_as_eager_execution_evidence():
    dispatcher=bench.Dispatcher(Custom(),Nccl(),'custom_dispatch',lambda:True)
    dispatcher(Tensor(16384))
    assert dispatcher.counts['capture_custom']==1
    with pytest.raises(RuntimeError,match='eager'):
        bench.require_evidence(dispatcher.counts,'custom_dispatch',True,1,0)


def test_expected_custom_cannot_silently_fallback_or_mix_paths():
    good={'eager_custom':8,'capture_custom':30,'eager_nccl':0,'capture_nccl':0}
    assert bench.require_evidence(good,'custom_dispatch',True,30,5)=='custom'
    for bad in [{**good,'eager_custom':0},{**good,'capture_nccl':1},{**good,'capture_custom':29}]:
        with pytest.raises(RuntimeError):bench.require_evidence(bad,'custom_dispatch',True,30,5)


def test_pair23_checks_actual_visible_indices_not_subgroup_rank01():
    calls=[]
    result=bench.check_actual_peers(2,[2,3],lambda a,b:calls.append((a,b)) or True)
    assert calls==[(2,3)] and result==[{'source':2,'target':3,'passed':True}]
    with pytest.raises(RuntimeError,match='P2P'):
        bench.check_actual_peers(2,[2,3],lambda a,b:False)


def test_failed_full_payload_validation_cannot_pass_from_first_element_only():
    import torch
    values=[torch.full((8,),10,dtype=torch.bfloat16)]
    assert bench.validate_outputs(torch,values,10)==1
    values[0][-1]=11
    with pytest.raises(RuntimeError,match='payload'):
        bench.validate_outputs(torch,values,10)


def test_profile_evidence_uses_actual_gpu_kernel_events(tmp_path):
    import gzip,json
    path=tmp_path/'probe.json.gz'
    with gzip.open(path,'wt') as f:json.dump({'traceEvents':[{'cat':'kernel','name':'custom_symbol','dur':3},{'cat':'cpu_op','name':'ncclAllReduce','dur':5}]},f)
    result=bench.kernel_evidence(path)
    assert result['kernel_names']=={'custom_symbol':1}
    assert result['gpu_kernel_events']==1


def test_dry_plan_rejects_foreign_gpu_assignment_before_torch():
    with pytest.raises(ValueError):bench.launch_config({'CUDA_VISIBLE_DEVICES':'GPU-4,GPU-5,GPU-6,GPU-7'})


def test_graph_buffers_register_only_after_cuda_capture_has_ended():
    from contextlib import contextmanager
    events=[]
    @contextmanager
    def ca_capture():
        events.append('ca-enter');yield;events.append('register-buffers')
    @contextmanager
    def cuda_graph(*args,**kwargs):
        events.append('graph-enter');yield;events.append('graph-exit')
    bench.capture_operations(SimpleNamespace(cuda=SimpleNamespace(graph=cuda_graph)),
                             SimpleNamespace(capture=ca_capture),object(),object(),
                             lambda:events.append('operations'),True)
    assert events==['ca-enter','graph-enter','operations','graph-exit','register-buffers']


def test_dry_run_validates_plan_without_importing_torch_or_creating_output(tmp_path):
    import json,os,subprocess,sys
    from control import GPUS
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(GPUS.values()),RANK='0',LOCAL_RANK='0',WORLD_SIZE='4',LOCAL_WORLD_SIZE='4')
    path=tmp_path/'must-not-exist.json'
    code='import sys;sys.path.insert(0,sys.argv[1]);import custom_dispatch_bench as b;b.main(["--output",sys.argv[2],"--dry-run"]);assert "torch" not in sys.modules'
    result=subprocess.run([sys.executable,'-c',code,str(Path(bench.__file__).parent),str(path)],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['sizes']==[8,16384,262144,4194304,33554432]
    assert not path.exists()


def test_close_pair23_uses_subgroup_rank_and_does_not_double_dispose():
    group=object(); freed=[];disposed=[]
    def free(pointers,group=None):
        # Global rank2 has subgroup rank0 in the second TP2 pair.
        index=2 if group is None else 0
        freed.append((pointers[index],group))
    custom=SimpleNamespace(disabled=False,_ptr=71,group=group,
                           meta_ptrs=[101,102],buffer_ptrs=[201,202],
                           free_shared_buffer=free)
    bench.close_custom(custom,disposed.append)
    assert disposed==[71]
    assert freed==[(101,group),(201,group)]
    assert custom._ptr==0
    bench.close_custom(custom,disposed.append)
    assert disposed==[71] and len(freed)==2


def test_close_disabled_custom_never_calls_device_cleanup():
    custom=SimpleNamespace(disabled=True)
    bench.close_custom(custom,lambda pointer:(_ for _ in ()).throw(AssertionError('disabled cleanup')))


def test_close_marks_pointer_disposed_before_free_failure():
    disposed=[]
    def fail(*args,**kwargs):raise RuntimeError('CUDA free failed')
    custom=SimpleNamespace(disabled=False,_ptr=73,group=object(),
                           meta_ptrs=[1,2],buffer_ptrs=[3,4],free_shared_buffer=fail)
    with pytest.raises(RuntimeError,match='CUDA free'):
        bench.close_custom(custom,disposed.append)
    assert custom._ptr==0 and disposed==[73]
