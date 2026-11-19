"""CPU boundary tests: real HTTP, file handshakes and disposable subprocesses."""
import gzip
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import profile_run as p


@pytest.fixture(autouse=True)
def cpu_ownership_token(monkeypatch):
    monkeypatch.setenv('A100_GPU0123_CONTROL_TOKEN','profile-cpu-tests')


def test_plan_injects_trace_output_before_extra_and_never_overwrites(tmp_path):
    args=['--model','8b','--family','muxpp','--trace','/trace.jsonl','--extra','--no-pdmux-pp-rebalance']
    plan=p.make_plan(args,tmp_path/'new',30)
    position=plan['server_args'].index('--extra')
    assert plan['server_args'][position-4:position]==['--suite','trace','--out',str(tmp_path/'new'/'serving')]
    assert plan['server_args'][position+1:]==['--no-pdmux-pp-rebalance']
    assert len(plan['workers'])==1 and plan['tp']==1 and plan['pp']==4
    with pytest.raises(ValueError,match='owned'):
        p.make_plan(args[:6]+['--out','/other'],tmp_path/'new',30)


@pytest.mark.parametrize('args',[
    ['--suite','smoke'],['--out=/other'],['--extra','--pdmux-pp-request-trace=/collision'],
])
def test_wrapper_rejects_conflicting_lifecycle_or_semantic_output(args,tmp_path):
    with pytest.raises(ValueError):
        p.make_plan(['--model','8b','--family','muxpp','--trace','/trace',*args],tmp_path/'new',30)


def test_worker_shim_assigns_unique_paths_and_checks_actual_mask(tmp_path):
    plan=p.make_plan(['--model','8b','--family','muxpp','--tp','1','--pp','2','--replicas','2','--trace','/trace'],tmp_path,30)
    results=[]
    for worker in plan['workers']:
        cmd=['-m','sglang.launch_server','--port',str(worker['port']),'--tp-size','1','--pp-size','2']
        actual,env=p.worker_command(cmd,{'CUDA_VISIBLE_DEVICES':','.join(worker['gpus'])},plan)
        results.append(env['SGLANG_PDMUX_PP_TIMING_TRACE'])
        assert actual[-2]=='--pdmux-pp-request-trace'
        assert f'worker{worker["index"]}' in actual[-1]
    assert len(set(results))==2
    with pytest.raises(ValueError,match='mask'):
        p.worker_command(cmd,{'CUDA_VISIBLE_DEVICES':'unassigned'},plan)
    with pytest.raises(ValueError,match='duplicate'):
        p.worker_command(cmd+['--port',str(worker['port'])],{'CUDA_VISIBLE_DEVICES':','.join(worker['gpus'])},plan)


def trace(path,rank=0,semantic=True):
    path.parent.mkdir(parents=True,exist_ok=True)
    events=[{'cat':'kernel','name':'gemm','ts':1,'dur':2}]
    if semantic:events.append({'cat':'cpu_op','name':f'pdmux_pp|pp={rank}|tp=0|lane=prefill|partition=1|phase=slice','ts':0,'dur':3})
    with gzip.open(path,'wt') as f:json.dump({'traceEvents':events},f)


def test_exports_require_all_four_unique_ranks_and_real_gpu_work(tmp_path):
    plan=p.make_plan(['--model','8b','--family','muxpp','--trace','/trace'],tmp_path,30)
    folder=tmp_path/'worker0'/'profile'
    for rank in range(3):trace(folder/f'123-PP-{rank}-TP-0.trace.json.gz',rank)
    with pytest.raises(ValueError,match='rank'):
        p.inspect_exports(plan)
    trace(folder/'123-PP-3-TP-0.trace.json.gz',3)
    assert len(p.inspect_exports(plan))==4
    trace(folder/'456-PP-3-TP-0.trace.json.gz',3)
    with pytest.raises(ValueError,match='rank'):
        p.inspect_exports(plan)


def test_rank_start_log_requires_coordinates_and_own_output_path(tmp_path):
    plan=p.make_plan(['--model','8b','--family','muxpp','--tp','2','--pp','2','--trace','/trace'],tmp_path,30)
    (tmp_path/'serving').mkdir()
    log=tmp_path/'serving/server-0.log'
    own=tmp_path/'worker0/profile'
    log.write_text('\n'.join(f'[time TP{tp} PP{pp}] Profiling starts for True. Traces will be saved to: {own}' for pp,tp in [(0,0),(0,1),(1,0)]))
    assert not p.all_ranks_started(plan)
    with log.open('a') as f:f.write(f'\n[time TP1 PP1] Profiling starts for True. Traces will be saved to: /wrong')
    assert not p.all_ranks_started(plan)
    with log.open('a') as f:f.write(f'\n[time TP1 PP1] Profiling starts for True. Traces will be saved to: {own}')
    assert p.all_ranks_started(plan)


class Endpoint:
    def __init__(self,plan,index=0,fail_start=False,exports=True):
        owner=self;self.calls=[];self.plan=plan;self.index=index
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                owner.calls.append((self.path,body))
                if self.path=='/start_profile':
                    if fail_start:self.send_error(500);return
                    assert body['profile_by_stage'] is False and body['with_stack'] is False
                    log=Path(plan['out'])/'serving'/f'server-{index}.log';log.parent.mkdir(exist_ok=True)
                    log.write_text('\n'.join(f'[time PP{pp}] Profiling starts for True. Traces will be saved to: {body["output_dir"]}' for pp in range(plan['pp'])))
                if self.path=='/stop_profile' and exports:
                    for pp in range(plan['pp']):trace(Path(plan['out'])/f'worker{index}'/'profile'/f'123-PP-{pp}-TP-0.trace.json.gz',pp)
                self.send_response(200);self.end_headers();self.wfile.write(b'OK')
            def log_message(self,*args):pass
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        plan['workers'][index]['port']=self.server.server_port
    def close(self):self.server.shutdown();self.server.server_close();self.thread.join()


def sleeper():
    from control import identity
    child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
    child.profile_identity=identity(child.pid)
    return child


def test_capture_real_http_boundaries_and_releases_finished_client(tmp_path):
    plan=p.make_plan(['--model','8b','--family','muxpp','--trace','/trace'],tmp_path,30)
    endpoint=Endpoint(plan);child=sleeper()
    p.atomic_json(tmp_path/'client-ready.json',{'pid':1})
    def done():
        while not (tmp_path/'client-go.json').exists():time.sleep(.005)
        p.atomic_json(tmp_path/'client-done.json',{'returncode':0})
    thread=threading.Thread(target=done,daemon=True);thread.start()
    try:
        report=p.capture(plan,child,seconds=.2,startup_timeout=1,export_timeout=1,poll=.01)
        assert report['status']=='completed' and len(report['exports'])==4
        assert report['measurement_eligible'] is False
        assert (tmp_path/'client-release.json').exists()
        assert [c[0] for c in endpoint.calls]==['/start_profile','/stop_profile']
    finally:p.stop_process_group(child,grace=.1);endpoint.close()


def test_partial_endpoint_failure_stops_successful_peer_and_never_starts_client(tmp_path):
    plan=p.make_plan(['--model','8b','--family','muxpp','--pp','2','--replicas','2','--trace','/trace'],tmp_path,30)
    endpoints=[Endpoint(plan,0),Endpoint(plan,1,fail_start=True)];child=sleeper()
    p.atomic_json(tmp_path/'client-ready.json',{'pid':1})
    try:
        report=p.capture(plan,child,seconds=.2,startup_timeout=1,export_timeout=.2,poll=.01)
        assert report['status']=='failed'
        assert '/stop_profile' in [c[0] for c in endpoints[0].calls]
        assert not (tmp_path/'client-go.json').exists()
        assert (tmp_path/'ABORT.json').exists()
    finally:
        p.stop_process_group(child,grace=.1)
        for ep in endpoints:ep.close()


def test_missing_exports_fails_and_unblocks_client(tmp_path):
    plan=p.make_plan(['--model','8b','--family','muxpp','--trace','/trace'],tmp_path,30)
    endpoint=Endpoint(plan,exports=False);child=sleeper();p.atomic_json(tmp_path/'client-ready.json',{})
    try:
        report=p.capture(plan,child,seconds=.01,startup_timeout=1,export_timeout=.1,poll=.01)
        assert report['status']=='failed' and report['exports']==[]
        assert (tmp_path/'ABORT.json').exists()
    finally:p.stop_process_group(child,grace=.1);endpoint.close()


def test_dead_runner_fails_startup_without_contacting_endpoints(tmp_path):
    plan=p.make_plan(['--model','8b','--family','plain','--trace','/trace'],tmp_path,30)
    child=subprocess.Popen([sys.executable,'-c','pass'],start_new_session=True);child.wait()
    report=p.capture(plan,child,seconds=.01,startup_timeout=.1,export_timeout=.1,poll=.01)
    assert report['status']=='failed' and 'exited' in ' '.join(report['errors'])


def test_group_cleanup_terminates_cpu_child():
    child=sleeper();p.stop_process_group(child,grace=.1)
    assert child.poll() is not None


def test_diagnostic_marking_preserves_raw_failures_and_disables_metrics(tmp_path):
    serving=tmp_path/'serving';serving.mkdir()
    original={'measurement_eligible':True,'failed_requests':2,'unsent_requests':7}
    (serving/'trace_summary.json').write_text(json.dumps(original))
    p.mark_diagnostic(tmp_path)
    assert json.loads((serving/'trace_summary.raw-diagnostic.json').read_text())==original
    marked=json.loads((serving/'trace_summary.json').read_text())
    assert marked['failed_requests']==2 and marked['unsent_requests']==7
    assert marked['measurement_eligible'] is False and marked['diagnostic_only'] is True


def test_client_gate_cannot_run_before_start_or_exit_before_exports(tmp_path):
    plan={'out':str(tmp_path),'parent_pid':os.getpid(),'gate_timeout':2}
    marker=tmp_path/'executed'
    result=[]
    command=[sys.executable,'-c',f'from pathlib import Path;Path({str(marker)!r}).write_text("ran");raise SystemExit(7)']
    thread=threading.Thread(target=lambda:result.append(p.client_gate(command,plan)))
    thread.start()
    try:
        end=time.monotonic()+1
        while not (tmp_path/'client-ready.json').exists() and time.monotonic()<end:time.sleep(.01)
        assert (tmp_path/'client-ready.json').exists() and not marker.exists()
        p.atomic_json(tmp_path/'client-go.json',{})
        while not (tmp_path/'client-done.json').exists() and time.monotonic()<end:time.sleep(.01)
        assert marker.exists() and thread.is_alive()
        assert json.loads((tmp_path/'client-done.json').read_text())['returncode']==7
        p.atomic_json(tmp_path/'client-release.json',{})
        thread.join(1)
        assert result==[7]
    finally:
        p.atomic_json(tmp_path/'ABORT.json',{})
        thread.join(2)


def test_client_gate_aborts_without_launching_when_parent_dies(tmp_path):
    plan={'out':str(tmp_path),'parent_pid':99999999,'gate_timeout':.1}
    with pytest.raises(RuntimeError,match='parent'):
        p.client_gate([sys.executable,'-c','raise AssertionError("must not run")'],plan)


def test_summary_deltas_exclude_calibration_and_keep_lanes_separate(tmp_path):
    plan=p.make_plan(['--model','8b','--family','muxpp','--trace','/trace'],tmp_path,30)
    (tmp_path/'serving').mkdir();log=tmp_path/'serving/server-0.log'
    first={0:{'at':200.,'timing':{'elapsed':121.,'wall':{'decode':{'control':106.},'prefill':{'empty':100.}},'cuda':{'decode':4.},'slices':{},'partitions':{0:100.}},'admission':{'decisions':{'admit':5},'blocked':{},'stops':{},'wall':{}}}}
    log.write_text('MuxWise-PP summary stages='+repr(first)+' mbs=[] waiting=0\n')
    before=p.summary_snapshot(plan)
    second={0:{'at':230.,'timing':{'elapsed':151.,'wall':{'decode':{'control':108.},'prefill':{'empty':109.}},'cuda':{'decode':20.},'slices':{},'partitions':{0:103.}},'admission':{'decisions':{'admit':9},'blocked':{},'stops':{},'wall':{}}}}
    with log.open('a') as f:f.write('MuxWise-PP summary stages='+repr(second)+' mbs=[] waiting=0\n')
    delta=p.summary_deltas(before,p.summary_snapshot(plan))
    stage=delta['workers']['0']['0']
    assert stage['sample_interval_seconds']==30
    assert stage['timing']['wall']['decode']['control']==2
    assert stage['timing']['wall']['prefill']['empty']==9
    assert stage['timing']['cuda']['decode']==16
    assert stage['admission']['decisions']['admit']==4
    assert delta['gpu_bubble_seconds'] is None


def test_group_cleanup_reaps_live_descendant_after_leader_has_exited(tmp_path):
    pidfile=tmp_path/'grandchild'
    code='import subprocess,sys;from pathlib import Path;p=subprocess.Popen([sys.executable,"-c","import signal,time;signal.signal(signal.SIGINT,signal.SIG_IGN);time.sleep(30)"]);Path(sys.argv[1]).write_text(str(p.pid))'
    leader=subprocess.Popen([sys.executable,'-c',code,str(pidfile)],start_new_session=True)
    from control import identity
    leader.profile_identity=identity(leader.pid)
    leader.wait();pid=int(pidfile.read_text())
    try:
        p.stop_process_group(leader,grace=.05)
        from control import identity
        assert identity(pid) is None
    finally:
        try:os.killpg(leader.pid,signal.SIGKILL)
        except ProcessLookupError:pass


def test_client_gate_interrupts_running_cpu_client_on_parent_abort(tmp_path):
    plan={'out':str(tmp_path),'parent_pid':os.getpid(),'gate_timeout':2}
    errors=[]
    def work():
        try:p.client_gate([sys.executable,'-c','import time;time.sleep(30)'],plan)
        except RuntimeError as exc:errors.append(str(exc))
    p.atomic_json(tmp_path/'client-go.json',{})
    thread=threading.Thread(target=work,daemon=True);thread.start()
    time.sleep(.1);p.atomic_json(tmp_path/'ABORT.json',{})
    thread.join(1)
    assert not thread.is_alive()
    assert errors and json.loads((tmp_path/'client-done.json').read_text())['returncode']!=0


def test_public_entry_rejects_missing_controller_before_creating_outputs(tmp_path,monkeypatch):
    import server_run
    def reject():raise RuntimeError('active controller required')
    monkeypatch.setattr(server_run,'require_active_controller',reject)
    out=tmp_path/'must-not-exist'
    with pytest.raises(RuntimeError,match='controller'):
        p.main(['--out',str(out),'--','--model','8b','--family','plain','--trace','/unused'])
    assert not out.exists()


def test_cleanup_refuses_tokenless_member_without_signalling_it():
    from control import identity
    env=dict(os.environ);env.pop('A100_GPU0123_CONTROL_TOKEN')
    child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],env=env,start_new_session=True)
    child.profile_identity=identity(child.pid)
    try:
        with pytest.raises(RuntimeError,match='ownership'):
            p.stop_process_group(child,grace=.01)
        assert child.poll() is None
    finally:child.kill();child.wait()


def test_cleanup_refuses_missing_initial_identity():
    child=sleeper();del child.profile_identity
    try:
        with pytest.raises(RuntimeError,match='identity'):
            p.stop_process_group(child,grace=.01)
        assert child.poll() is None
    finally:child.kill();child.wait()


def test_client_outputs_publish_only_ineligible_summary_and_preserve_raw(tmp_path):
    plan={'out':str(tmp_path)};serving=tmp_path/'serving';serving.mkdir()
    command=[str(p.BENCH),'--out-dir',str(serving),'--label','trace']
    actual=p.client_command(command,plan)
    assert actual==[str(p.PYTHON),str(p.BENCH),'--out-dir',str(tmp_path/'client-raw'),'--label','trace']
    raw=tmp_path/'client-raw';raw.mkdir()
    data={'measurement_eligible':True,'failed_requests':2,'unsent_requests':7}
    (raw/'trace_summary.json').write_text(json.dumps(data))
    (raw/'trace_requests.jsonl').write_text('{"idx":3}\n')
    assert not (serving/'trace_summary.json').exists()
    p.publish_client_artifacts(plan)
    assert json.loads((serving/'trace_summary.json').read_text())['measurement_eligible'] is False
    assert json.loads((raw/'trace_summary.json').read_text())==data
    assert (serving/'trace_requests.jsonl').read_text()=='{"idx":3}\n'
    p.mark_diagnostic(tmp_path)
    assert json.loads((serving/'trace_summary.raw-diagnostic.json').read_text())==data


def test_main_preserves_failure_report_even_when_cleanup_raises(tmp_path,monkeypatch):
    import server_run
    monkeypatch.setattr(server_run,'require_active_controller',lambda:None)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',','.join(server_run.UUIDS))
    child=sleeper();previous=signal.getsignal(signal.SIGTERM)
    monkeypatch.setattr(p.subprocess,'Popen',lambda *a,**kw:child)
    monkeypatch.setattr(p,'capture',lambda *a,**kw:{'status':'failed','errors':['endpoint failed'],'exports':[],'diagnostic_only':True,'measurement_eligible':False})
    def fail_cleanup(*a,**kw):raise RuntimeError('ownership refused')
    monkeypatch.setattr(p,'stop_process_group',fail_cleanup)
    source=tmp_path/'trace.jsonl';source.write_text('{}\n');out=tmp_path/'output'
    try:
        assert p.main(['--out',str(out),'--','--model','8b','--family','plain','--trace',str(source)])==1
        report=json.loads((out/'profile.json').read_text())
        assert any('ownership refused' in e for e in report['errors'])
        assert signal.getsignal(signal.SIGTERM)==previous
    finally:child.kill();child.wait();signal.signal(signal.SIGTERM,previous)


def test_delayed_profile_releases_client_and_waits_for_post_warmup_origin(tmp_path):
    plan=p.make_plan(['--model','8b','--family','muxpp','--trace','/trace'],tmp_path,30)
    endpoint=Endpoint(plan);child=sleeper();p.atomic_json(tmp_path/'client-ready.json',{})
    observed=[]
    def client():
        while not (tmp_path/'client-go.json').exists():time.sleep(.005)
        observed.append(len(endpoint.calls))
        time.sleep(.03)
        folder=tmp_path/'serving';folder.mkdir(exist_ok=True)
        observed.append(time.time());(folder/'client.log').write_text('Loaded 30 requests; warmup=2; target=http://local\n')
    thread=threading.Thread(target=client,daemon=True);thread.start()
    try:
        report=p.capture(plan,child,seconds=.02,startup_timeout=1,export_timeout=1,poll=.005,delay=.04)
        assert report['status']=='completed' and report['includes_client_warmups'] is False
        assert observed[0]==0
        assert report['endpoints'][0]['started_unix']>=observed[1]+.04
        assert report['requested_delay_seconds']==.04
    finally:p.stop_process_group(child,grace=.1);endpoint.close();thread.join(1)


def test_client_finishing_before_delayed_window_is_failed_diagnostic(tmp_path):
    plan=p.make_plan(['--model','8b','--family','muxpp','--trace','/trace'],tmp_path,30)
    endpoint=Endpoint(plan);child=sleeper();p.atomic_json(tmp_path/'client-ready.json',{})
    p.atomic_json(tmp_path/'client-done.json',{'returncode':0})
    try:
        report=p.capture(plan,child,seconds=.02,startup_timeout=1,export_timeout=1,poll=.005,delay=.1)
        assert report['status']=='failed' and not endpoint.calls
        assert any('before delayed' in error for error in report['errors'])
    finally:p.stop_process_group(child,grace=.1);endpoint.close()


def test_client_finishing_during_last_delay_sleep_never_starts_profile(tmp_path):
    plan=p.make_plan(['--model','8b','--family','muxpp','--trace','/trace'],tmp_path,30)
    endpoint=Endpoint(plan);child=sleeper();p.atomic_json(tmp_path/'client-ready.json',{})
    (tmp_path/'serving').mkdir();(tmp_path/'serving/client.log').write_text('Loaded 1 requests; warmup=2\n')
    def finish():
        while not (tmp_path/'client-go.json').exists():time.sleep(.001)
        time.sleep(.02);p.atomic_json(tmp_path/'client-done.json',{'returncode':0})
    thread=threading.Thread(target=finish,daemon=True);thread.start()
    try:
        report=p.capture(plan,child,seconds=.02,startup_timeout=1,export_timeout=1,poll=.1,delay=.06)
        assert report['status']=='failed' and not endpoint.calls
    finally:p.stop_process_group(child,grace=.1);endpoint.close();thread.join(1)
