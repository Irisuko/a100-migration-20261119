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
from control import GPUS
UUIDS = list(GPUS.values())
HTTP = build_opener(ProxyHandler({}))
CONTEXT_LENGTH = 33792
CORRECTNESS_LENGTHS = [100,512,2047,2048,2049,4095,4096,4097,8191,8192,8193,12288,31663,32704]

def require_active_controller():
    from control import Controller, DEFAULT_ROOT, matches
    controller=Controller(Path(os.environ.get('A100_GPU0123_CAMPAIGN_ROOT',str(DEFAULT_ROOT))))
    controller.assert_enabled()
    active=controller.state.get('active') or {}
    token=os.environ.get('A100_GPU0123_CONTROL_TOKEN')
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

def overlap_policy(family, pp, requested):
    if requested not in ('off','auto','on'):
        raise ValueError('Unknown overlap policy')
    supported=family in ('plain','chunked','pd') and pp==1
    if requested=='on' and not supported:
        raise ValueError('--overlap on requires ordinary or PD scheduling with PP1; PP and Mux disable native overlap')
    enabled=supported and requested!='off'
    return {'requested':requested,'effective':'on' if enabled else 'off',
            'enabled':enabled,'supported':supported}

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

def source_provenance(repo):
    def hashes(files):
        return {str(path.relative_to(repo)):hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    runtime=hashes(sorted((repo/'python/sglang').rglob('*.py')))
    tool_root=repo/'scripts/a100_gpu0123'
    tools=hashes(sorted(tool_root.rglob('*.py')))
    serving_files=[tool_root/'server_run.py',*[p for p in sorted((tool_root/'bench').glob('*.py')) if not p.name.startswith('test_')]]
    serving=hashes(serving_files)
    return {'runtime_source_sha256':runtime,'tool_sha256':tools,
            'serving_tool_sha256':serving,'source_sha256':{**runtime,**serving}}

def model_revision(model):
    path=model/'download-provenance.json'
    data=json.loads(path.read_text()) if path.exists() else {}
    return data.get('official_revision') or data.get('revision')

def configuration_provenance(command):
    selected=None
    for i,value in enumerate(command):
        value=str(value)
        if value=='--pdmux-config-path':selected=Path(command[i+1])
        elif value.startswith('--pdmux-config-path='):selected=Path(value.split('=',1)[1])
    return ({'path':str(selected.resolve()),'sha256':hashlib.sha256(selected.read_bytes()).hexdigest()}
            if selected is not None else None)

def run_numerical_audits(workers,model,cases,out,args):
    from numerical_audit import audit
    for worker in workers:
        name='numerical-audit.json' if len(workers)==1 else f'numerical-audit-worker{worker["index"]}.json'
        result=audit(worker['port'],cases,out/name,expected_metadata={
            'model_path':str(model),'tp_size':args.tp,'pp_size':args.pp,
            'dtype':'bfloat16','enable_pdmux':False})
        if not result.get('completed') or not result.get('full_token_coverage'):
            raise RuntimeError(f'Numerical audit coverage incomplete for worker{worker["index"]}; see {name}')

def request(port, path, payload=None, timeout=600):
    req=Request(f'http://127.0.0.1:{port}{path}',data=None if payload is None else json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with HTTP.open(req,timeout=timeout) as response:
        raw=response.read().decode()
        try:return json.loads(raw)
        except ValueError:return raw

def cancellation_http_request(port, payload):
    """Capture both successful and failed HTTP bodies only for deliberate abort tests."""
    import base64
    from urllib.error import HTTPError
    result={'requested_rid':payload['rid'],'port':port}
    try:
        req=Request(f'http://127.0.0.1:{port}/generate',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        try:response=HTTP.open(req,timeout=600)
        except HTTPError as error:response=error
        result.update(http_status=response.status,http_reason=str(response.reason))
        with response:raw=response.read()
        result.update(raw_body=raw.decode('utf-8',errors='replace'),raw_body_base64=base64.b64encode(raw).decode('ascii'))
        try:result['response']=json.loads(result['raw_body'])
        except ValueError:result['response']=None
    except Exception as error:
        result['error']=repr(error)
    return result


def paired_cancellation_request(prefill, decode, payload):
    """Keep each role's response; a failed prefill must not hide the decode result."""
    payload={**payload,'bootstrap_host':'127.0.0.1','bootstrap_port':prefill['bootstrap'],
             'bootstrap_room':secrets.randbits(63)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        first=pool.submit(cancellation_http_request,prefill['port'],payload)
        last=pool.submit(cancellation_http_request,decode['port'],payload)
        responses={'prefill':first.result(),'decode':last.result()}
    return {'requested_rid':payload['rid'],'bootstrap_room':payload['bootstrap_room'],'responses':responses}


def _cancellation_response_aborted(value, rid, role=None):
    if not isinstance(value,dict) or value.get('requested_rid')!=rid or 'error' in value:return False
    body=value.get('response')
    if not isinstance(body,dict):return False
    if value.get('http_status')==200:
        meta=body.get('meta_info',{})
        if not isinstance(meta,dict):return False
        reason=meta.get('finish_reason',{})
        return (isinstance(reason,dict) and meta.get('id')==rid and reason.get('type')=='abort'
                and reason.get('status_code') is None and reason.get('err_type') is None
                and reason.get('message') in ('Aborted','Abort before prefill','Aborted before prefill','Aborted by AbortReq.'))
    message=body.get('message')
    prefixes={
        'prefill':('Prefill bootstrap failed for request rank=','Prefill transfer failed for request rank='),
        'decode':('Decode handshake failed for request rank=','Decode transfer failed for request rank='),
        None:('Prefill transfer failed for request rank=','Decode transfer failed for request rank='),
    }.get(role,())
    return (value.get('http_status')==500 and body.get('object')=='error'
            and body.get('code')==500 and body.get('type') in ('500','HTTPStatus.INTERNAL_SERVER_ERROR')
            and isinstance(message,str) and message.startswith(prefixes)
            and f'rid={rid!r}' in message and message.endswith(': Aborted by AbortReq.'))


def cancellation_confirmed(record, rid):
    if not isinstance(record,dict) or record.get('requested_rid')!=rid:return False
    responses=record.get('responses',{})
    if not isinstance(responses,dict):return False
    if set(responses)=={'model'}:return _cancellation_response_aborted(responses['model'],rid)
    if set(responses)!={'prefill','decode'} or not _cancellation_response_aborted(responses['decode'],rid,'decode'):return False
    prefill=responses['prefill']
    if _cancellation_response_aborted(prefill,rid,'prefill'):return True
    if not isinstance(prefill,dict) or prefill.get('requested_rid')!=rid or 'error' in prefill or prefill.get('http_status')!=200:return False
    body=prefill.get('response');meta=body.get('meta_info',{}) if isinstance(body,dict) else {}
    if not isinstance(meta,dict):return False
    reason=meta.get('finish_reason',{})
    return (isinstance(reason,dict) and meta.get('id')==rid and reason.get('type')=='length' and reason.get('length')==0
            and reason.get('status_code') is None and reason.get('err_type') is None)


def flush(port):
    answer=request(port,'/flush_cache',timeout=30)
    if not isinstance(answer,str) or not answer.startswith('Cache flushed.'):
        raise RuntimeError(f'cache flush not acknowledged: {answer!r}')

def prepare_trace(workers):
    """Reset every replica; preserve the common two-request client prewarm."""
    ports=[worker['port'] for worker in workers]
    for port in ports:flush(port)
    return {'status':'acknowledged','worker_ports':ports,'time_unix':time.time(),
            'client_warmup_requests':2,
            'cache_precondition':'All workers flushed, then two common 16-token-prefix warmup requests.'}

def require_services(services):
    for proc,label in services:
        code=proc.poll()
        if code is not None:raise RuntimeError(f'{label} exited {code}')

def quiesce_router(router,services,metadata):
    """Stop only our router after measurement; keep backend/resource gates strict."""
    record={'status':'checking_services','pid':None,'terminate_requested':False,
            'kill_requested':False,'quiet_seconds':0.0,'started_unix':time.time()}
    metadata['postflight_router_shutdown']=record
    try:
        require_services(services)
        if router is None:
            if any(label=='router' for _,label in services):
                raise RuntimeError('Router service exists without an owned router object')
            record.update(status='not_started',ended_unix=time.time())
            return
        owned=[(proc,label) for proc,label in services if proc is router]
        if len(owned)!=1 or owned[0][1]!='router' or sum(label=='router' for _,label in services)!=1:
            raise RuntimeError('Planned shutdown requires the unique owned router Popen object')
        record.update(status='stopping',pid=router.pid,terminate_requested=True,
                      terminate_requested_unix=time.time())
        router.terminate()
        try:
            code=router.wait(timeout=5)
        except subprocess.TimeoutExpired:
            record.update(kill_requested=True,kill_requested_unix=time.time())
            router.kill()
            code=router.wait(timeout=5)
        record.update(returncode=code,exited_unix=time.time())
        expected={0,-signal.SIGTERM}
        if record['kill_requested']:expected.add(-signal.SIGKILL)
        if code not in expected:
            raise RuntimeError(f'Planned router shutdown returned unexpected exit returncode {code}')
        services[:]=[(proc,label) for proc,label in services if proc is not router]
        require_services(services)
        # Health handlers poll every second; terminal output may clear state sooner.
        # This quiet period does not replace the unchanged strict zero-state checks.
        record.update(status='settling_inflight_health',quiet_seconds=2.0,
                      quiet_started_unix=time.time())
        time.sleep(2.0)
        require_services(services)
        record.update(status='quiesced',ended_unix=time.time())
    except BaseException as exc:
        record.update(status='failed',error=repr(exc),ended_unix=time.time())
        raise

def wait_client(client,services):
    """Do not accept partial service loss as a completed benchmark."""
    try:
        while True:
            require_services(services)
            code=client.poll()
            if code is not None:
                if code:raise RuntimeError(f'client exited {code}')
                return code
            time.sleep(.25)
    except BaseException:
        # SIGINT lets the client save unsent/failed/unfinished outcomes first.
        if client.poll() is None:
            client.send_signal(signal.SIGINT)
            try:client.wait(timeout=30)
            except subprocess.TimeoutExpired:
                client.terminate()
                try:client.wait(timeout=10)
                except subprocess.TimeoutExpired:client.kill();client.wait()
        raise

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

def stress_cases(model):
    """Common deterministic 200-request pressure set, independent of traces."""
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(model),local_files_only=True)
    rng=random.Random(202611140123)
    cases=[]
    for index in range(200):
        n=(128,256,512,1024,2048,4096)[index%6]
        text=' '.join(f'Experiment {index} measurement {rng.randrange(1000000)}.' for _ in range(1600))
        suffix=tok.encode(f'\nSummarize experiment {index}, verification code {index:04d}.\nAnswer:',add_special_tokens=False)
        ids=tok.encode(text,add_special_tokens=False)
        cases.append([128000]+ids[:n-len(suffix)-1]+suffix)
    return cases

def resource_phase(out,phase,ports):
    path=out/'resource-phase-markers.json'
    def quiet_interval(name):
        start=time.time()
        time.sleep(1.5)
        marker={'phase':name,'start_unix':start,'end_unix':time.time()}
        previous=json.loads(path.read_text()) if path.exists() else []
        save(path,[*previous,marker])
    def snapshot(suffix):
        snapshots={str(port):request(port,'/get_server_info',timeout=30) for port in ports}
        save(out/f'{phase}-server-{suffix}.json',snapshots)
        for port,info in snapshots.items():
            if info.get('tokenizer_pending_requests')!=0 or info.get('tokenizer_finished_states')!=0:
                raise RuntimeError(f'tokenizer states not recovered on port{port}: {info.get("tokenizer_pending_requests")}, finished={info.get("tokenizer_finished_states")}')
    quiet_interval(phase+'-pre-flush')
    snapshot('before-flush')
    for worker_port in ports:flush(worker_port)
    quiet_interval(phase)
    snapshot('after-flush')

def preflight(port, model, out, suite, transport=None, worker_ports=None, stress200=False, cancel_transport=None):
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
    resource_phase(out,'initial',ports)
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
            try:
                if transport:
                    if cancel_transport is None:raise RuntimeError('Paired cancellation capture is required')
                    return cancel_transport(payload)
                return {'requested_rid':payload['rid'],'responses':{'model':cancellation_http_request(port,payload)}}
            except Exception as e:return {'requested_rid':payload['rid'],'error':repr(e)}
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            pending=[pool.submit(cancel_req,i) for i in range(16)]
            time.sleep(.5)
            for worker_port in ports:request(worker_port,'/abort_request',{'rid':prefix,'abort_all':True},timeout=30)
            aborted=[f.result() for f in pending]
        save(out/'cancellation.json',aborted)
        if not all(cancellation_confirmed(x,prefix+str(i)) for i,x in enumerate(aborted)):
            raise RuntimeError('not all cancellation requests confirmed abort')
        save(out/'after-cancel.json',generate_at(cases[0],16))
        resource_phase(out,'after-cancel',ports)
        if stress200:
            pressure=stress_cases(model)
            save(out/'stress-prompts.json',pressure)
            with concurrent.futures.ThreadPoolExecutor(max_workers=200) as pool:
                stressed=list(pool.map(lambda ids:generate_at(ids,32),pressure))
            save(out/'stress200.json',stressed)
            resource_phase(out,'after-stress',ports)
    flush_group()
    resource_phase(out,'final',ports)
    save(out/'server-after.json',{str(worker_port):request(worker_port,'/get_server_info',timeout=30) for worker_port in ports})

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['8b','70b'],required=True)
    p.add_argument('--family',choices=['plain','chunked','muxwise','muxpp','pd'],required=True)
    p.add_argument('--tp',type=int,default=1);p.add_argument('--pp',type=int,default=4);p.add_argument('--replicas',type=int,default=1)
    p.add_argument('--prefill-replicas',type=int,default=0)
    p.add_argument('--pd-backend',choices=['mooncake','nixl'],default='mooncake')
    p.add_argument('--ib-device')
    p.add_argument('--overlap',choices=['off','auto','on'],default='off',help='Native scheduler overlap: auto enables ordinary/PD PP1; off preserves historical runs')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mem',type=float,default=.85);p.add_argument('--kv',type=int)
    p.add_argument('--running',type=int,default=128);p.add_argument('--graph',type=int,default=16)
    p.add_argument('--chunk',type=int,default=8192);p.add_argument('--baseline-chunk',type=int,default=2048)
    p.add_argument('--inflight',type=int,default=3);p.add_argument('--budget',type=int,default=32768)
    p.add_argument('--no-async',action='store_true');p.add_argument('--no-cpp',action='store_true');p.add_argument('--custom-allreduce',action='store_true')
    p.add_argument('--suite',choices=['smoke','correctness','trace'],default='smoke')
    p.add_argument('--stress200',action='store_true',help='Add 200 distinct bounded requests after cancellation in full correctness preflight')
    p.add_argument('--oracle-cases',type=Path,help='Audit supplied actual-prefix cases against ordinary inference after preflight')
    p.add_argument('--trace',type=Path);p.add_argument('--time-scale',type=float,default=1);p.add_argument('--limit',type=int)
    p.add_argument('--arrival-window-sec',type=float)
    p.add_argument('--startup-timeout',type=float,default=900);p.add_argument('--port',type=int,default=32100)
    p.add_argument('--extra',nargs=argparse.REMAINDER,default=[])
    a=p.parse_args();validate_topology(a.family,a.tp,a.pp,a.replicas,a.prefill_replicas)
    overlap=overlap_policy(a.family,a.pp,a.overlap)
    if any('--disable-overlap-schedule'.startswith(flag.split('=',1)[0])
           for flag in a.extra if flag.startswith('--')):
        raise ValueError('--extra cannot override native overlap; use --overlap')
    if a.stress200 and a.suite!='correctness':raise ValueError('--stress200 requires --suite correctness')
    if a.oracle_cases and a.family!='plain':raise ValueError('--oracle-cases requires --family plain')
    reserved = {'--base-gpu-id','--gpu-id-step','--tp','--tp-size','--tensor-parallel-size',
                '--pp','--pp-size','--pipeline-parallel-size','--dp','--dp-size',
                '--data-parallel-size','--nnodes','--node-rank','--dist-init-addr',
                '--device','--enable-dp-attention','--enable-dp-lm-head'}
    if any(any(option.startswith(flag.split('=',1)[0]) for option in reserved)
           for flag in a.extra if flag.startswith('--')):
        raise ValueError('--extra cannot override the assigned GPU mapping or four-card topology')
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
    processes=[];services=[];logs=[];metadata={'args':vars(a).copy(),'commands':[],'status':'running','start_unix':time.time()}
    metadata['args']={k:str(v) if isinstance(v,Path) else v for k,v in metadata['args'].items()}
    metadata['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    (a.out/'source.patch').write_bytes(subprocess.check_output(['git','diff','--binary','HEAD'],cwd=REPO))
    metadata.update(source_provenance(REPO))
    metadata['configuration_sha256']={'spatial':hashlib.sha256(config.read_bytes()).hexdigest()}
    metadata['model_provenance']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in model.glob('*.json')}
    metadata['model_revision']=model_revision(model)
    if a.trace:metadata['trace_sha256']=hashlib.sha256(a.trace.read_bytes()).hexdigest()
    metadata['runtime_environment']={k:v for k,v in os.environ.items() if k.startswith(('NCCL_','TORCH_CUDA_','FLASHINFER_','TRITON_CACHE','CUDA_VISIBLE','CUDA_HOME')) or k in ('OMP_NUM_THREADS','MAX_JOBS')}
    metadata['worker_environment']={'SGLANG_A100_RESOURCE_AUDIT':'1'}
    metadata['overlap_policy']=overlap
    def launch(cmd,env,logname):
        require_active_controller()
        log=(a.out/logname).open('x');logs.append(log)
        proc=subprocess.Popen([str(x) for x in cmd],env=env,stdout=log,stderr=subprocess.STDOUT)
        processes.append(proc);return proc
    def wait_ready(proc,port):
        end=time.monotonic()+a.startup_timeout
        while time.monotonic()<end:
            require_services(services)
            if proc.poll() is not None:raise RuntimeError(f'server port{port} exited {proc.returncode}')
            try:
                request(port,'/health',timeout=2);return
            except Exception:time.sleep(1)
        raise TimeoutError(f'server port{port} startup timeout')
    try:
        require_active_controller()
        monitor_log=(a.out/'gpu-monitor.csv').open('x');logs.append(monitor_log)
        monitor=subprocess.Popen(['nvidia-smi','-i',','.join(map(str,GPUS)),'--query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,power.draw,clocks.sm','--format=csv','--loop=1'],stdout=monitor_log,stderr=subprocess.STDOUT)
        processes.append(monitor)
        services.append((monitor,'gpu-monitor'))
        servers=[]
        workers=worker_layout(a.family,a.tp,a.pp,a.replicas,a.prefill_replicas,a.port)
        metadata['workers']=workers
        for worker in workers:
            replica=worker['index'];port=worker['port'];group=worker['gpus']
            with socket.socket() as sock:
                if sock.connect_ex(('127.0.0.1',port))==0:raise RuntimeError(f'port occupied {port}')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(group),NCCL_DEBUG='INFO',NCCL_DEBUG_SUBSYS='INIT,GRAPH,P2P,SHM',NCCL_DEBUG_FILE=str((a.out/f'nccl-worker{replica}-%p.log').resolve()),SGLANG_A100_RESOURCE_AUDIT='1')
            cmd=[PYTHON,'-m','sglang.launch_server','--model-path',model,'--tp-size',str(a.tp),'--pp-size',str(a.pp),'--dtype','bfloat16','--host','127.0.0.1','--port',str(port),'--context-length',str(CONTEXT_LENGTH),'--mem-fraction-static',str(a.mem),'--max-running-requests',str(a.running),'--cuda-graph-max-bs',str(a.graph),'--cuda-graph-bs',*[str(size) for size in graph_sizes(a.graph)],'--max-prefill-tokens','16384',*([] if overlap['enabled'] else ['--disable-overlap-schedule']),'--attention-backend','flashinfer','--sampling-backend','pytorch','--schedule-policy','fcfs','--watchdog-timeout','180','--log-level','debug','--enable-cache-report','--skip-tokenizer-init','--stream-output','--random-seed','20261114']
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
            actual_config=configuration_provenance(cmd)
            if actual_config:
                metadata['configuration_sha256'][f'worker{replica}.pdmux']=actual_config['sha256']
                metadata.setdefault('configuration_paths',{})[f'worker{replica}.pdmux']=actual_config['path']
            metadata['commands'].append({'command':[str(x) for x in cmd],'CUDA_VISIBLE_DEVICES':env['CUDA_VISIBLE_DEVICES']})
            save(a.out/'run.json',metadata)
            proc=launch(cmd,env,f'server-{replica}.log');servers.append((proc,port))
            services.append((proc,f'worker{replica}'))
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
                cancel_transport=lambda payload,prefill_worker=prefill_worker,decode_worker=decode_worker:paired_cancellation_request(prefill_worker,decode_worker,payload)
                preflight(decode_worker['port'],model,sub,'correctness' if a.suite=='correctness' else 'smoke',transport,ports,a.stress200,cancel_transport=cancel_transport)
        else:
            for i,(proc,port) in enumerate(servers):
                sub=a.out/f'worker{i}';sub.mkdir()
                save(sub/'server-before.json',request(port,'/get_server_info',timeout=30))
                preflight(port,model,sub,'correctness' if a.suite=='correctness' else 'smoke',stress200=a.stress200)
        if a.replicas>1:
            with socket.socket() as sock:
                if sock.connect_ex(('127.0.0.1',a.port))==0:raise RuntimeError(f'router port occupied {a.port}')
            router_cmd,router_env=router_command(workers,a.port,a.family)
            metadata['router']={'command':router_cmd,'environment':router_env,
                                'implementation':'repository Python mini_lb' if a.family=='pd' else 'installed sglang-router0.1.9'}
            router=launch(router_cmd,dict(os.environ,**router_env),'router.log')
            services.append((router,'router'))
            wait_ready(router,a.port)
            if a.suite!='trace':
                # Per-worker preflights above establish coverage; this checks the common endpoint.
                cases=fixture_cases(model)
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                    outputs=list(pool.map(lambda ids:generate(a.port,ids,16),cases[:4]))
                save(a.out/'router-smoke.json',outputs)
        if a.suite=='trace':
            require_services(services)
            metadata['trace_cache_reset']=prepare_trace(workers)
            cmd=[PYTHON,REPO/'scripts/a100_gpu0123/bench/bench_client.py','--trace',a.trace,'--base-url',f'http://127.0.0.1:{a.port}','--label','trace','--out-dir',a.out,'--time-scale',str(a.time_scale),'--warmup','2','--force-close','--drain-timeout','300','--progress-interval','30','--output-id-mode','incremental']
            window=a.arrival_window_sec
            stats=a.trace.with_suffix('.stats.json')
            if window is None and stats.exists() and a.limit is None:window=json.loads(stats.read_text())['duration_sec']*a.time_scale
            if window is not None:cmd+=['--arrival-window-sec',str(window)]
            if a.limit:cmd+=['--limit',str(a.limit)]
            metadata['client_command']=[str(x) for x in cmd];save(a.out/'run.json',metadata)
            client=launch(cmd,os.environ.copy(),'client.log')
            wait_client(client,services)
        require_services(services)
        if a.oracle_cases:
            run_numerical_audits(workers,model,a.oracle_cases,a.out,a)
        quiesce_router(router if a.replicas>1 else None,services,metadata)
        save(a.out/'run.json',metadata)
        for worker in workers:
            final_dir=a.out/f'worker{worker["index"]}'
            final_dir.mkdir(exist_ok=True)
            resource_phase(final_dir,'run-final',[worker['port']])
            save(final_dir/'server-final.json',request(worker['port'],'/get_server_info',timeout=30))
        require_services(services)
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
