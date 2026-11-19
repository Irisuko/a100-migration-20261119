#!/usr/bin/env python3
"""One four-GPU topology: lifecycle, preflight, or faithful trace replay.

Run under control.py; this program never manages idle reservations itself.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import random
import secrets
import signal
import socket
import subprocess
import sys
import time
from urllib.request import build_opener, ProxyHandler, Request

REPO = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO.parent / 'muxwise-pp-experiments'
PYTHON = REPO / '.venv/bin/python'
UUIDS = ['GPU-725d9550-4cdc-f401-0cb0-3d31bccec1af','GPU-69f3abce-1e88-4391-95dc-4ab4c2c77593','GPU-fd1b979e-abe3-dc82-8a17-890c8cd3cf14','GPU-0c16ece9-91b7-1520-f2ea-69811b52901a']
HTTP = build_opener(ProxyHandler({}))
CONTEXT_LENGTH = 33792
CORRECTNESS_LENGTHS = [100,512,2047,2048,2049,4095,4096,4097,8191,8192,8193,12288,31663,32704]

def require_active_controller():
    from control import Controller, DEFAULT_ROOT, matches
    controller=Controller(Path(os.environ.get('A100_CAMPAIGN_ROOT',str(DEFAULT_ROOT))))
    controller.assert_enabled()
    active=controller.state.get('active') or {}
    token=os.environ.get('A100_CONTROL_TOKEN')
    if (not token or active.get('token')!=token or controller.state.get('holder')
            or not matches(active.get('controller'))):
        raise RuntimeError('An active controller with matching ownership is required; run through control.py')
    return controller

def validate_topology(family, tp, pp, replicas, prefill_replicas=0):
    if min(tp,pp,replicas)<1 or tp*pp*replicas != 4:
        raise ValueError('Every topology must use exactly four GPUs')
    if family=='muxwise' and pp != 1:
        raise ValueError('Original MuxWise does not support PP')
    if family=='muxpp' and pp < 2:
        raise ValueError('MuxWise-PP requires PP>=2')
    if family=='pd':
        if pp!=1 or tp not in (1,2) or not 0<prefill_replicas<replicas:
            raise ValueError('PD requires PP1 and at least one prefill and one decode replica within four GPUs')
    elif prefill_replicas:
        raise ValueError('--prefill-replicas is only valid for PD')

def graph_sizes(limit):
    if limit < 1:
        raise ValueError('CUDA graph batch limit must be positive')
    return sorted({1, limit, *(size for size in (4, 8, 16, 32) if size <= limit)})

def worker_layout(family,tp,pp,replicas,prefill_replicas,port):
    validate_topology(family,tp,pp,replicas,prefill_replicas)
    return [{'index':i,'port':port+1+i if replicas>1 else port,
             'bootstrap':port+1000+i,'gpus':UUIDS[i*tp*pp:(i+1)*tp*pp],
             'role':('prefill' if i<prefill_replicas else 'decode') if family=='pd' else 'model'}
            for i in range(replicas)]

def pd_pairs(workers):
    prefills=[w for w in workers if w['role']=='prefill']
    decodes=[w for w in workers if w['role']=='decode']
    return [(prefills[i%len(prefills)],decodes[i%len(decodes)]) for i in range(max(len(prefills),len(decodes)))]

def router_command(workers,port,family):
    cmd=[str(PYTHON),'-m','sglang_router.launch_router','--host','127.0.0.1','--port',str(port),
         '--policy','random' if family=='pd' else 'cache_aware','--request-timeout-secs','3600',
         '--max-concurrent-requests','4096','--log-level','info']
    env={}
    if family=='pd':
        # Installed router0.1.9 lacks --mini-lb; use this runtime's Python PD router.
        env['PYTHONPATH']=str(REPO/'sgl-router/py_src')+':'+str(REPO/'python')
        cmd+=['--mini-lb','--pd-disaggregation']
        for worker in workers:
            url=f"http://127.0.0.1:{worker['port']}"
            cmd+=['--prefill',url,str(worker['bootstrap'])] if worker['role']=='prefill' else ['--decode',url]
    else:
        cmd+=['--worker-urls',*[f"http://127.0.0.1:{worker['port']}" for worker in workers]]
    return cmd,env

def paired_request(prefill,decode,payload):
    payload={**payload,'bootstrap_host':'127.0.0.1','bootstrap_port':prefill['bootstrap'],
             'bootstrap_room':secrets.randbits(63)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        p=pool.submit(request,prefill['port'],'/generate',payload)
        d=pool.submit(request,decode['port'],'/generate',payload)
        first,last=p.result(),d.result()
    return {**last,'_prefill_meta_info':first.get('meta_info',{})}

def save(path, data):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def request(port, path, payload=None, timeout=600):
    req=Request(f'http://127.0.0.1:{port}{path}',data=None if payload is None else json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with HTTP.open(req,timeout=timeout) as response:
        raw=response.read().decode()
        try:return json.loads(raw)
        except ValueError:return raw

def flush(port):
    answer=request(port,'/flush_cache',timeout=30)
    if not isinstance(answer,str) or not answer.startswith('Cache flushed.'):
        raise RuntimeError(f'cache flush not acknowledged: {answer!r}')

def generate(port, ids, n=32, rid=None, transport=None):
    payload={'input_ids':ids,'sampling_params':{'temperature':0,'max_new_tokens':n,'ignore_eos':True}}
    if rid:payload['rid']=rid
    start=time.monotonic()
    body=transport(payload) if transport else request(port,'/generate',payload)
    if len(body.get('output_ids',[]))!=n or body['meta_info']['completion_tokens']!=n:
        raise RuntimeError(f'incomplete output: {body}')
    result={'input_ids':ids,'output_ids':body['output_ids'],'meta_info':body['meta_info'],'wall_s':time.monotonic()-start}
    if '_prefill_meta_info' in body:result['prefill_meta_info']=body['_prefill_meta_info']
    return result

def fixture_cases(model):
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(model),local_files_only=True)
    cases=[]
    for i,n in enumerate(CORRECTNESS_LENGTHS):
        text=(f'Report family {i%4}: the robot records temperature and rainfall. Water the apple trees daily. ')*6000
        ids=tok.encode(text,add_special_tokens=False)
        suffix=tok.encode(f'\nSummarize report number {i} in two sentences.\nAnswer:',add_special_tokens=False)
        cases.append([128000]+ids[:n-len(suffix)-1]+suffix)
    return cases

def preflight(port, model, out, suite, transport=None, worker_ports=None):
    ports=worker_ports or [port]
    def flush_group():
        for worker_port in ports:flush(worker_port)
    def generate_at(ids,n=32):return generate(port,ids,n,transport=transport)
    cases=fixture_cases(model)
    save(out/'prompts.json',cases)
    flush_group()
    # Same long random prewarm as historical protocol; never use it for timing claims.
    rng=random.Random(730031)
    ids=[128000]+[rng.randrange(128000) for _ in range(31662)]
    save(out/'long-warmup.json',generate_at(ids,8))
    flush_group()
    selected=cases if suite=='correctness' else [cases[0],cases[8],cases[-1]]
    outputs=[]
    for ids in selected:
        outputs.append(generate_at(ids,64 if suite=='correctness' else 16))
        save(out/'serial.json',outputs)
    flush_group()
    cached=[generate_at(cases[9],32),generate_at(cases[9],32)]
    cache_meta=cached[1].get('prefill_meta_info',cached[1]['meta_info'])
    if cached[0]['output_ids']!=cached[1]['output_ids'] or cache_meta.get('cached_tokens',0)<8000:
        raise RuntimeError('prefix reuse consistency failed')
    save(out/'cache.json',cached)
    flush_group()
    count=32 if suite=='correctness' else 8
    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
        results=list(pool.map(lambda i:generate_at(cases[i%len(cases)],32),range(count)))
    save(out/'concurrent.json',results)
    if suite=='correctness':
        flush_group()
        prefix=f'a100-cancel-{time.time_ns()}-'
        cancellation_prompt=cases[CORRECTNESS_LENGTHS.index(31663)]
        def cancel_req(i):
            payload={'rid':prefix+str(i),'input_ids':cancellation_prompt if i>3 else cases[i], 'sampling_params':{'temperature':0,'max_new_tokens':1000,'ignore_eos':True}}
            try:return transport(payload) if transport else request(port,'/generate',payload)
            except Exception as e:return {'error':repr(e)}
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            pending=[pool.submit(cancel_req,i) for i in range(16)]
            time.sleep(.5)
            for worker_port in ports:request(worker_port,'/abort_request',{'rid':prefix,'abort_all':True},timeout=30)
            aborted=[f.result() for f in pending]
        save(out/'cancellation.json',aborted)
        if not all('abort' in json.dumps(x).lower() for x in aborted):
            raise RuntimeError('not all cancellation requests confirmed abort')
        save(out/'after-cancel.json',generate_at(cases[0],16))
    flush_group()
    save(out/'server-after.json',{str(worker_port):request(worker_port,'/get_server_info',timeout=30) for worker_port in ports})

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['8b','70b'],required=True)
    p.add_argument('--family',choices=['plain','chunked','muxwise','muxpp','pd'],required=True)
    p.add_argument('--tp',type=int,default=1);p.add_argument('--pp',type=int,default=4);p.add_argument('--replicas',type=int,default=1)
    p.add_argument('--prefill-replicas',type=int,default=0)
    p.add_argument('--pd-backend',choices=['mooncake','nixl'],default='mooncake')
    p.add_argument('--ib-device')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mem',type=float,default=.85);p.add_argument('--kv',type=int)
    p.add_argument('--running',type=int,default=128);p.add_argument('--graph',type=int,default=16)
    p.add_argument('--chunk',type=int,default=8192);p.add_argument('--baseline-chunk',type=int,default=2048)
    p.add_argument('--inflight',type=int,default=3);p.add_argument('--budget',type=int,default=32768)
    p.add_argument('--no-async',action='store_true');p.add_argument('--no-cpp',action='store_true');p.add_argument('--custom-allreduce',action='store_true')
    p.add_argument('--suite',choices=['smoke','correctness','trace'],default='smoke')
    p.add_argument('--trace',type=Path);p.add_argument('--time-scale',type=float,default=1);p.add_argument('--limit',type=int)
    p.add_argument('--arrival-window-sec',type=float)
    p.add_argument('--startup-timeout',type=float,default=900);p.add_argument('--port',type=int,default=32100)
    p.add_argument('--extra',nargs=argparse.REMAINDER,default=[])
    a=p.parse_args();validate_topology(a.family,a.tp,a.pp,a.replicas,a.prefill_replicas)
    require_active_controller()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != ','.join(UUIDS):raise RuntimeError('fixed four-GPU UUID mask required')
    if a.model=='70b' and a.tp*a.pp<2:raise ValueError('70B BF16 weights require at least two80GB GPUs per replica')
    if a.suite=='trace' and not a.trace:raise ValueError('--trace required')
    a.out=a.out.resolve()
    a.out.mkdir(parents=True,exist_ok=False)
    model=REPO.parent/'models'/('Llama-3.1-8B-Instruct' if a.model=='8b' else 'Llama-3.3-70B-Instruct')
    import yaml
    cfg={'sm_group_num':8,'manual_divisions':[[100,8,1],[92,16,4],[84,24,8],[76,32,12],[60,48,16],[52,56,24]],'decode_bs_divisor':32,'split_forward_token_budget':a.budget,'pp_profile_decode':True,'pp_decode_extra_contexts':[128,512,1024],'pp_prefetch':True,'pp_prefetch_slots':2,'pp_async_decode':not a.no_async}
    if a.model=='70b':cfg['pp_decode_profile_max_seconds']=600
    if a.family=='muxwise':cfg={'sm_group_num':5,'manual_divisions':[[76,32,1],[60,48,8],[52,56,16]],'decode_bs_divisor':24,'split_forward_token_budget':a.budget}
    config=a.out/'spatial.yml';config.write_text(yaml.safe_dump(cfg))
    processes=[];logs=[];metadata={'args':vars(a).copy(),'commands':[],'status':'running','start_unix':time.time()}
    metadata['args']={k:str(v) if isinstance(v,Path) else v for k,v in metadata['args'].items()}
    metadata['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    (a.out/'source.patch').write_bytes(subprocess.check_output(['git','diff','--binary','HEAD'],cwd=REPO))
    source_files=[REPO/'python/sglang/srt/models/llama.py',REPO/'python/sglang/srt/layers/attention/flashinfer_backend.py',REPO/'python/sglang/srt/managers/tokenizer_manager.py',*sorted((REPO/'python/sglang/srt/multiplex').glob('*.py')),*sorted((REPO/'scripts/a100_4gpu').rglob('*.py'))]
    metadata['source_sha256']={str(f.relative_to(REPO)):hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files}
    metadata['model_provenance']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in model.glob('*.json')}
    metadata['model_revision']=json.loads((model/'download-provenance.json').read_text()).get('revision') if (model/'download-provenance.json').exists() else None
    if a.trace:metadata['trace_sha256']=hashlib.sha256(a.trace.read_bytes()).hexdigest()
    metadata['runtime_environment']={k:v for k,v in os.environ.items() if k.startswith(('NCCL_','TORCH_CUDA_','FLASHINFER_','TRITON_CACHE','CUDA_VISIBLE','CUDA_HOME')) or k in ('OMP_NUM_THREADS','MAX_JOBS')}
    def launch(cmd,env,logname):
        require_active_controller()
        log=(a.out/logname).open('x');logs.append(log)
        proc=subprocess.Popen([str(x) for x in cmd],env=env,stdout=log,stderr=subprocess.STDOUT)
        processes.append(proc);return proc
    def wait_ready(proc,port):
        end=time.monotonic()+a.startup_timeout
        while time.monotonic()<end:
            if proc.poll() is not None:raise RuntimeError(f'server port{port} exited {proc.returncode}')
            try:
                request(port,'/health',timeout=2);return
            except Exception:time.sleep(1)
        raise TimeoutError(f'server port{port} startup timeout')
    try:
        require_active_controller()
        monitor_log=(a.out/'gpu-monitor.csv').open('x');logs.append(monitor_log)
        monitor=subprocess.Popen(['nvidia-smi','-i','4,5,6,7','--query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,power.draw,clocks.sm','--format=csv','--loop=1'],stdout=monitor_log,stderr=subprocess.STDOUT)
        processes.append(monitor)
        servers=[]
        workers=worker_layout(a.family,a.tp,a.pp,a.replicas,a.prefill_replicas,a.port)
        metadata['workers']=workers
        for worker in workers:
            replica=worker['index'];port=worker['port'];group=worker['gpus']
            with socket.socket() as sock:
                if sock.connect_ex(('127.0.0.1',port))==0:raise RuntimeError(f'port occupied {port}')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(group),NCCL_DEBUG='INFO',NCCL_DEBUG_SUBSYS='INIT,GRAPH,P2P,SHM',NCCL_DEBUG_FILE=str((a.out/f'nccl-worker{replica}-%p.log').resolve()))
            cmd=[PYTHON,'-m','sglang.launch_server','--model-path',model,'--tp-size',str(a.tp),'--pp-size',str(a.pp),'--dtype','bfloat16','--host','127.0.0.1','--port',str(port),'--context-length',str(CONTEXT_LENGTH),'--mem-fraction-static',str(a.mem),'--max-running-requests',str(a.running),'--cuda-graph-max-bs',str(a.graph),'--cuda-graph-bs',*[str(size) for size in graph_sizes(a.graph)],'--max-prefill-tokens','16384','--disable-overlap-schedule','--attention-backend','flashinfer','--sampling-backend','pytorch','--schedule-policy','fcfs','--watchdog-timeout','180','--log-level','debug','--enable-cache-report','--skip-tokenizer-init','--stream-output','--random-seed','20261114']
            # Use verified NCCL first; custom path is an explicit separate feasibility candidate.
            if not a.custom_allreduce:cmd+=['--disable-custom-all-reduce']
            if a.kv:cmd+=['--max-total-tokens',str(a.kv)]
            if a.family in ('muxwise','muxpp'):
                cmd+=['--enable-pdmux','--chunked-prefill-size','-1','--pdmux-config-path',config,'--sm-group-num',str(cfg['sm_group_num'])]
                if a.family=='muxpp':cmd+=['--pdmux-pp-prefill-chunk-size','0' if a.no_cpp else str(a.chunk),'--pdmux-pp-max-inflight-prefill',str(a.inflight),'--pdmux-prefill-mlp-chunk-tokens','4096','--pdmux-pp-slo-ttft','5','--pdmux-pp-slo-tpot','0.1','--pdmux-pp-summary-interval','5','--pdmux-pp-compact-attention','--pdmux-pp-stage-local-lm-head']
            elif a.family=='pd':
                cmd+=['--disaggregation-mode',worker['role'],'--disaggregation-transfer-backend',a.pd_backend,
                      '--disaggregation-bootstrap-port',str(worker['bootstrap']),
                      '--chunked-prefill-size','4096' if worker['role']=='prefill' else '-1']
                if a.ib_device:cmd+=['--disaggregation-ib-device',a.ib_device]
            else:
                cmd+=['--chunked-prefill-size',str(a.baseline_chunk) if a.family=='chunked' else '-1']
                if a.family=='chunked' and a.pp==1:cmd+=['--enable-mixed-chunk']
            cmd+=a.extra
            metadata['commands'].append({'command':[str(x) for x in cmd],'CUDA_VISIBLE_DEVICES':env['CUDA_VISIBLE_DEVICES']})
            save(a.out/'run.json',metadata)
            proc=launch(cmd,env,f'server-{replica}.log');servers.append((proc,port))
        for proc,port in servers:wait_ready(proc,port)
        metadata['startup_s']=time.time()-metadata['start_unix'];save(a.out/'run.json',metadata)
        if a.family=='pd':
            metadata['preflight_kind']='paired PD; each prefill and decode instance is covered'
            for i,(prefill_worker,decode_worker) in enumerate(pd_pairs(workers)):
                sub=a.out/f'pair{i}';sub.mkdir()
                save(sub/'pair.json',{'prefill':prefill_worker,'decode':decode_worker})
                ports=[prefill_worker['port'],decode_worker['port']]
                save(sub/'server-before.json',{str(port):request(port,'/get_server_info',timeout=30) for port in ports})
                transport=lambda payload,prefill_worker=prefill_worker,decode_worker=decode_worker:paired_request(prefill_worker,decode_worker,payload)
                preflight(decode_worker['port'],model,sub,'correctness' if a.suite=='correctness' else 'smoke',transport,ports)
        else:
            for i,(proc,port) in enumerate(servers):
                sub=a.out/f'worker{i}';sub.mkdir()
                save(sub/'server-before.json',request(port,'/get_server_info',timeout=30))
                preflight(port,model,sub,'correctness' if a.suite=='correctness' else 'smoke')
        if a.replicas>1:
            with socket.socket() as sock:
                if sock.connect_ex(('127.0.0.1',a.port))==0:raise RuntimeError(f'router port occupied {a.port}')
            router_cmd,router_env=router_command(workers,a.port,a.family)
            metadata['router']={'command':router_cmd,'environment':router_env,
                                'implementation':'repository Python mini_lb' if a.family=='pd' else 'installed sglang-router0.1.9'}
            router=launch(router_cmd,dict(os.environ,**router_env),'router.log')
            wait_ready(router,a.port)
            if a.suite!='trace':
                # Per-worker preflights above establish coverage; this checks the common endpoint.
                cases=fixture_cases(model)
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                    outputs=list(pool.map(lambda ids:generate(a.port,ids,16),cases[:4]))
                save(a.out/'router-smoke.json',outputs)
        if a.suite=='trace':
            cmd=[PYTHON,REPO/'scripts/a100_4gpu/bench/bench_client.py','--trace',a.trace,'--base-url',f'http://127.0.0.1:{a.port}','--label','trace','--out-dir',a.out,'--time-scale',str(a.time_scale),'--warmup','2','--force-close','--drain-timeout','300','--progress-interval','30','--output-id-mode','incremental']
            window=a.arrival_window_sec
            stats=a.trace.with_suffix('.stats.json')
            if window is None and stats.exists() and a.limit is None:window=json.loads(stats.read_text())['duration_sec']*a.time_scale
            if window is not None:cmd+=['--arrival-window-sec',str(window)]
            if a.limit:cmd+=['--limit',str(a.limit)]
            metadata['client_command']=[str(x) for x in cmd];save(a.out/'run.json',metadata)
            client=launch(cmd,os.environ.copy(),'client.log');rc=client.wait()
            if rc:raise RuntimeError(f'client exited{rc}')
        metadata['status']='execution_complete'
        metadata['correctness_verdict']='not_compared'
    except BaseException as e:
        metadata.update(status='failed',error=repr(e));raise
    finally:
        metadata['end_unix']=time.time();save(a.out/'run.json',metadata)
        for proc in reversed(processes):
            if proc.poll() is None:proc.terminate()
        for proc in reversed(processes):
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        for log in logs:log.close()

if __name__=='__main__':main()
