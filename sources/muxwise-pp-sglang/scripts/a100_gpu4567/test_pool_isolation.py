"""CPU-only pool isolation tests: all process/socket/GPU operations are doubles."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
from types import SimpleNamespace
import pytest

HERE=Path(__file__).resolve().parent
EXPECTED={4:'GPU-725d9550-4cdc-f401-0cb0-3d31bccec1af',5:'GPU-69f3abce-1e88-4391-95dc-4ab4c2c77593',6:'GPU-fd1b979e-abe3-dc82-8a17-890c8cd3cf14',7:'GPU-0c16ece9-91b7-1520-f2ea-69811b52901a'}
CPUS=set(range(32,64))|set(range(96,128))

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

@pytest.fixture
def modules(monkeypatch,tmp_path):
    c=load('_pool4567_control',HERE/'control.py');monkeypatch.setitem(sys.modules,'control',c)
    s=load('_pool4567_server',HERE/'server_run.py')
    monkeypatch.setattr(c,'DEFAULT_ROOT',tmp_path/'pool');monkeypatch.setattr(c,'GLOBAL_SUSPENSION',tmp_path/'pool/RESERVATION_DISABLED.json')
    monkeypatch.setattr(c.os,'sched_getaffinity',lambda pid:CPUS)
    def forbidden(*args,**kwargs):raise AssertionError('CPU test cannot launch/signal/query/connect')
    monkeypatch.setattr(c.subprocess,'Popen',forbidden);monkeypatch.setattr(c.subprocess,'check_output',forbidden)
    monkeypatch.setattr(c.signal,'pidfd_send_signal',forbidden);monkeypatch.setattr(c.os,'pidfd_open',forbidden)
    monkeypatch.setattr(s.socket,'socket',forbidden)
    return c,s

def test_assignment_environment_root_and_cache_are_isolated(modules):
    c,s=modules
    assert c.GPUS==EXPECTED and s.UUIDS==list(EXPECTED.values())
    assert c.TOKEN=='A100_GPU4567_CONTROL_TOKEN'
    env=c.environment(c.DEFAULT_ROOT,{'CUDA_VISIBLE_DEVICES':'0,1,2,3','A100_GPU0123_CONTROL_TOKEN':'foreign','A100_GPU0123_CAMPAIGN_ROOT':'/main','NCCL_P2P_DISABLE':'1'})
    assert env['CUDA_VISIBLE_DEVICES']==','.join(EXPECTED.values())
    assert env['A100_GPU4567_CAMPAIGN_ROOT']==str(c.DEFAULT_ROOT.resolve())
    assert 'A100_GPU0123_CONTROL_TOKEN' not in env and 'A100_GPU0123_CAMPAIGN_ROOT' not in env
    assert 'NCCL_P2P_DISABLE' not in env
    assert all(str(c.DEFAULT_ROOT) in env[k] for k in ['FLASHINFER_WORKSPACE_BASE','TRITON_CACHE_DIR','TORCH_EXTENSIONS_DIR'])

def test_other_campaign_roots_cannot_be_read_or_created(modules,tmp_path):
    c,s=modules
    with pytest.raises(ValueError,match='pool root'):c.Controller(tmp_path/'main')
    with pytest.raises(ValueError,match='pool root'):c.environment(tmp_path/'main',{})
    assert not (tmp_path/'main').exists()

def test_only_assigned_gpu_indices_are_queried(modules,monkeypatch):
    c,s=modules;calls=[]
    def query(argv,**kwargs):
        calls.append(argv);assert argv[argv.index('-i')+1]=='4,5,6,7'
        return ''.join(f'{i}, {u}\n' for i,u in EXPECTED.items()) if '--query-gpu=index,uuid' in argv else ''
    monkeypatch.setattr(c.subprocess,'check_output',query)
    assert c.gpu_processes()==[] and len(calls)==2

def test_foreign_process_is_not_owned_or_signalled(modules,monkeypatch):
    c,s=modules
    monkeypatch.setattr(c,'gpu_processes',lambda:[{'gpu_uuid':EXPECTED[4],'pid':123}])
    monkeypatch.setattr(c,'identity',lambda pid:{'pid':pid,'uid':-1,'starttime':1})
    monkeypatch.setattr(c,'has_token',lambda *args:False);monkeypatch.setattr(c,'owned_processes',lambda t:[])
    with pytest.raises(RuntimeError,match='Foreign GPU process'):c.assert_clear('own')
    assert c.signal_process({'pid':123,'uid':-1,'starttime':1},signal.SIGTERM) is False

@pytest.mark.parametrize('family,tp,pp,replicas,prefills',[
 ('plain',4,1,1,0),('chunked',1,1,4,0),('chunked',2,1,2,0),('chunked',1,4,1,0),('chunked',2,2,1,0),
 ('muxwise',4,1,1,0),('muxwise',2,1,2,0),('muxpp',1,4,1,0),('muxpp',2,2,1,0),('muxpp',1,2,2,0),
 ('pd',1,1,4,1),('pd',1,1,4,2),('pd',1,1,4,3),('pd',2,1,2,1)])
def test_full_four_gpu_family_layout(modules,family,tp,pp,replicas,prefills):
    c,s=modules;workers=s.worker_layout(family,tp,pp,replicas,prefills,36100)
    assert [g for w in workers for g in w['gpus']]==list(EXPECTED.values())
    assert all(36100<=w['port']<=36104 and 37100<=w['bootstrap']<=37103 for w in workers)
    assert [w['nccl_port'] for w in workers]==[38200+100*i for i in range(replicas)]
    if family=='pd':assert {w['index'] for pair in s.pd_pairs(workers) for w in pair}==set(range(replicas))

def test_runner_owned_controller_requires_new_token_and_cpu_mask(modules,monkeypatch):
    c,s=modules;c.DEFAULT_ROOT.mkdir();parent=c.identity(os.getpid())
    state={'active':{'token':'own','controller':parent},'holder':None}
    (c.DEFAULT_ROOT/'controller.json').write_text(json.dumps(state))
    monkeypatch.setenv('A100_GPU4567_CAMPAIGN_ROOT',str(c.DEFAULT_ROOT));monkeypatch.setenv('A100_GPU4567_CONTROL_TOKEN','own')
    assert s.require_active_controller().root==c.DEFAULT_ROOT
    monkeypatch.setenv('A100_GPU4567_CONTROL_TOKEN','wrong')
    with pytest.raises(RuntimeError,match='active controller'):s.require_active_controller()
    monkeypatch.setenv('A100_GPU4567_CONTROL_TOKEN','own');monkeypatch.setattr(c.os,'sched_getaffinity',lambda p:{0,32})
    with pytest.raises(RuntimeError,match='NUMA1'):s.require_active_controller()

@pytest.mark.parametrize('flag',['--nccl-port','--port','--host','--disaggregation-bootstrap-port','--base-gpu-id','--gpu-id-step','--tp-size','--pp-size'])
def test_extra_cannot_override_pool_ports_or_devices(modules,monkeypatch,flag):
    c,s=modules
    monkeypatch.setattr(sys,'argv',['server_run.py','--model','8b','--family','plain','--tp','4','--pp','1','--out',str(c.DEFAULT_ROOT/'never'),'--extra',flag,'32100'])
    with pytest.raises(ValueError,match='override'):s.main()
    assert not (c.DEFAULT_ROOT/'never').exists()

@pytest.mark.parametrize('port',[32100,36099,36200,65535])
def test_worker_port_range_rejects_main_or_other_ranges(modules,port):
    c,s=modules
    with pytest.raises(ValueError,match='port'):s.worker_layout('plain',4,1,1,0,port)

def test_cpu_bound_command_is_numa1_and_preserves_arguments(modules):
    c,s=modules;command=['python','runner','--extra','literal']
    assert c.bound_command(command)==['taskset','-c','32-63,96-127',*command]
    assert command==['python','runner','--extra','literal']

def test_identical_bench_and_numerical_payload_helpers():
    for name in ['bench/bench_client.py','bench/common.py','bench/metrics.py','numerical_audit.py']:
        assert (HERE/name).read_bytes()==(HERE.parent/'a100_gpu0123'/name).read_bytes()


def test_pool_acquires_legacy_overlap_locks_and_never_main_lock(modules,monkeypatch,tmp_path):
    import fcntl
    c,s=modules;monkeypatch.setattr(c,'REPO',tmp_path)
    legacy=tmp_path/'run/a100-gpu67-controller.lock';legacy.parent.mkdir()
    with legacy.open('a') as owned:
        fcntl.flock(owned,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            with c.pool_locks():raise AssertionError('overlapping controller must block')
    with c.pool_locks():
        for name in ('4567','45','67'):
            with (tmp_path/f'run/a100-gpu{name}-controller.lock').open('a') as other:
                with pytest.raises(BlockingIOError):fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (tmp_path/'run/a100-gpu0123-controller.lock').exists()


def test_run_records_actual_cpu_bound_owned_command_only(modules,monkeypatch):
    c,s=modules;manager=c.Controller(c.DEFAULT_ROOT);events=[];captured=[]
    monkeypatch.setattr(manager,'release',lambda:events.append('release'));monkeypatch.setattr(manager,'reserve',lambda:events.append('reserve'))
    monkeypatch.setattr(c,'assert_clear',lambda *a:None);monkeypatch.setattr(c,'stop_owned',lambda token:events.append(('stop_own',token)))
    child=SimpleNamespace(pid=9999999,returncode=0,poll=lambda:0,wait=lambda timeout:0)
    def spawn(command,**kwargs):captured.append((command,kwargs));return child
    monkeypatch.setattr(c.subprocess,'Popen',spawn)
    result=manager.run('cpu-fixture',10,['python','server'])
    assert result['status']=='completed' and len(captured)==1
    argv,kwargs=captured[0];assert argv==['taskset','-c','32-63,96-127','python','server']
    assert kwargs['env']['CUDA_VISIBLE_DEVICES']==','.join(EXPECTED.values())
    token=kwargs['env'][c.TOKEN];assert ('stop_own',token) in events
    record=json.loads((c.DEFAULT_ROOT/'runs/cpu-fixture/command.json').read_text())
    assert record['argv']==argv and record['gpu_mapping']=={str(k):v for k,v in EXPECTED.items()}
    with pytest.raises(FileExistsError):manager.run('cpu-fixture',10,['python','server'])


@pytest.mark.parametrize('bad',['mask','ready_path','token','affinity'])
def test_private_holder_guards_before_gpu_import(modules,monkeypatch,bad,tmp_path):
    c,s=modules;ready=c.DEFAULT_ROOT/'holders/fixture.ready.json'
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',','.join(EXPECTED.values()));monkeypatch.setenv(c.TOKEN,'own')
    monkeypatch.setattr(c,'assert_clear',lambda *a:(_ for _ in ()).throw(AssertionError('must reject before GPU query')))
    if bad=='mask':monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0,1,2,3')
    if bad=='ready_path':ready=tmp_path/'main/ready.json'
    if bad=='token':monkeypatch.delenv(c.TOKEN)
    if bad=='affinity':monkeypatch.setattr(c.os,'sched_getaffinity',lambda p:{0})
    with pytest.raises((ValueError,RuntimeError)):c.holder(c.DEFAULT_ROOT,ready)


@pytest.mark.parametrize('family,tp,pp,replicas,prefills,suite',[
 ('plain',4,1,1,0,'smoke'),('chunked',2,1,2,0,'correctness'),('muxwise',4,1,1,0,'trace'),
 ('muxpp',2,2,1,0,'trace'),('pd',1,1,4,2,'correctness'),('pd',2,1,2,1,'trace')])
def test_actual_runner_commands_full_lifecycle_remain_pool_local(modules,monkeypatch,family,tp,pp,replicas,prefills,suite):
    c,s=modules;c.DEFAULT_ROOT.mkdir();out=c.DEFAULT_ROOT/'test-run';trace=c.DEFAULT_ROOT/'trace.jsonl';trace.write_text('{}\n')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',','.join(EXPECTED.values()))
    monkeypatch.setattr(s,'require_active_controller',lambda:c.Controller(c.DEFAULT_ROOT))
    monkeypatch.setattr(s.subprocess,'check_output',lambda argv,**kwargs:b'' if kwargs.get('text') is not True else 'head\n')
    class Sock:
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def connect_ex(self,target):assert 36100<=target[1]<=36104;return 1
    monkeypatch.setattr(s.socket,'socket',Sock)
    calls=[];preflights=[]
    class Child:
        returncode=None
        def __init__(self,argv,**kwargs):self.argv=argv;self.pid=9000+len(calls);calls.append((argv,kwargs))
        def poll(self):return self.returncode
        def terminate(self):self.returncode=0
        def kill(self):self.returncode=-9
        def wait(self,timeout=None):self.returncode=0;return 0
    monkeypatch.setattr(s.subprocess,'Popen',Child)
    monkeypatch.setattr(s,'request',lambda *a,**k:{})
    monkeypatch.setattr(s,'preflight',lambda *a,**k:preflights.append((a,k)))
    monkeypatch.setattr(s,'fixture_cases',lambda model:[[1],[2],[3],[4]])
    monkeypatch.setattr(s,'generate',lambda *a,**k:{'output_ids':[1]})
    monkeypatch.setattr(s,'prepare_trace',lambda workers:{'workers':len(workers)})
    monkeypatch.setattr(s,'wait_client',lambda client,services:client.wait())
    monkeypatch.setattr(s,'resource_phase',lambda *a,**k:None)
    monkeypatch.setattr(s.time,'sleep',lambda *a:None)
    argv=['server_run.py','--model','8b','--family',family,'--tp',str(tp),'--pp',str(pp),'--replicas',str(replicas),
          '--prefill-replicas',str(prefills),'--suite',suite,'--out',str(out)]
    if suite=='trace':argv+=['--trace',str(trace),'--arrival-window-sec','150']
    if suite=='correctness':argv+=['--stress200']
    monkeypatch.setattr(sys,'argv',argv);s.main()
    metadata=json.loads((out/'run.json').read_text());assert metadata['status']=='execution_complete'
    assert metadata['physical_pool']['indices']==[4,5,6,7] and metadata['physical_pool']['numa_node']==1
    workers=metadata['workers'];servers=[(a,k) for a,k in calls if 'sglang.launch_server' in a]
    assert len(servers)==replicas and len(preflights)==(max(prefills,replicas-prefills) if family=='pd' else replicas)
    for i,(cmd,kwargs) in enumerate(servers):
        assert cmd[cmd.index('--nccl-port')+1]==str(38200+100*i)
        assert kwargs['env']['CUDA_VISIBLE_DEVICES']==','.join(workers[i]['gpus'])
        assert cmd[cmd.index('--dtype')+1]=='bfloat16' and cmd[cmd.index('--port')+1]==str(workers[i]['port'])
    assert metadata['source_sha256']['scripts/a100_gpu4567/bench/bench_client.py']=='e54c76a2b07143f31e756b64480707cc771baa6eebd256cce877ae9b9bdc564f'
    if suite=='trace':assert str(HERE/'bench/bench_client.py') in metadata['client_command']


def test_output_cannot_escape_pool_before_creation(modules,monkeypatch,tmp_path):
    c,s=modules;other=tmp_path/'main-output'
    monkeypatch.setattr(s,'require_active_controller',lambda:c.Controller(c.DEFAULT_ROOT))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',','.join(EXPECTED.values()))
    monkeypatch.setattr(sys,'argv',['server_run.py','--model','8b','--family','plain','--tp','4','--pp','1','--out',str(other)])
    with pytest.raises(ValueError,match='Output must stay'):s.main()
    assert not other.exists()


def test_probe_requires_owner_before_torch_and_dry_run_never_imports_gpu(modules,monkeypatch,capsys):
    import builtins
    c,s=modules;monkeypatch.setitem(sys.modules,'server_run',s)
    probe=load('_pool4567_probe',HERE/'platform_probe.py')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',','.join(EXPECTED.values()))
    for k,v in {'RANK':'0','LOCAL_RANK':'0','WORLD_SIZE':'4','LOCAL_WORLD_SIZE':'4'}.items():monkeypatch.setenv(k,v)
    original=builtins.__import__
    def importing(name,*a,**k):
        if name=='torch' or name.startswith('torch.'):raise AssertionError('CUDA import reached without ownership')
        return original(name,*a,**k)
    monkeypatch.setattr(builtins,'__import__',importing)
    assert probe.main(['--output',str(c.DEFAULT_ROOT/'platform.json'),'--dry-run']) is None
    assert json.loads(capsys.readouterr().out)['visible_gpu_uuids']==list(EXPECTED.values())
    with pytest.raises(RuntimeError,match='active controller'):probe.main(['--output',str(c.DEFAULT_ROOT/'platform.json')])
    assert not c.DEFAULT_ROOT.exists()
