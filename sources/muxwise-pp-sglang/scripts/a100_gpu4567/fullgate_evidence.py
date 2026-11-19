"""Finite saved full-preflight coverage checks; no runtime/tokenizer/GPU import."""
import ast
import base64
import importlib.util
import json
from pathlib import Path
import random
import re
import sys

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
_spec=importlib.util.spec_from_file_location('_fullgate_pool_evidence',HERE/'pool_evidence.py')
selected=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(selected)

require=selected.require
LENGTHS=[100,512,2047,2048,2049,4095,4096,4097,8191,8192,8193,12288,31663,32704]
RECOVERY=REPO/'run/a100-gpu0123-20261114/proposals/pd-8b-full-v4/check_recovery.py'
GENERATION=REPO/'run/a100-gpu0123-20261114/analysis/aux-70b-pp2-oracle-assessment-v1/analyze_saved.py'


def take(path,pins,repo):
    require(Path(path).is_file(),f'Missing raw full-gate evidence: {path}')
    return selected.read(selected.pin(pins,path,repo=repo))


def pure_functions(path,names,pins,repo):
    path=selected.pin(pins,path,repo=repo)
    nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes}==set(names),'Required pure evidence validators changed')
    namespace={};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec',optimize=0),namespace)
    return namespace


def paired_folders(run,metadata,pins,repo):
    args=metadata['args'];workers=metadata.get('workers',[]);replicas=args['replicas']
    require(isinstance(workers,list) and len(workers)==replicas
            and [w.get('index') for w in workers]==list(range(replicas)), 'Missing full worker coverage')
    if args['family']!='pd':
        require(all(w.get('role')=='model' for w in workers),'Ordinary worker role differs')
        return [(run/f'worker{i}',None) for i in range(replicas)]
    p=args.get('prefill_replicas');require(type(p) is int and 0<p<replicas,'Invalid PD role split')
    require(all(w.get('role')==('prefill' if i<p else 'decode') for i,w in enumerate(workers)), 'PD role assignment differs')
    count=max(p,replicas-p);result=[]
    require({x.name for x in run.glob('pair*') if x.is_dir()}=={f'pair{i}' for i in range(count)}, 'Missing or extra PD pair folders')
    for i in range(count):
        folder=run/f'pair{i}';pair={'prefill':workers[i%p],'decode':workers[p+i%(replicas-p)]}
        require(take(folder/'pair.json',pins,repo)==pair,'Saved pair mapping differs from worker roles')
        result.append((folder,pair))
    return result


def worker_views(run,metadata,candidate,pins,repo):
    """Bind physical UUID subsets; API base_gpu_id is local within each subset."""
    controller=selected.pin(pins,Path(repo)/'scripts/a100_gpu4567/control.py',repo=repo)
    tables=[ast.literal_eval(n.value) for n in ast.parse(controller.read_text()).body if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=='GPUS' for t in n.targets)]
    require(len(tables)==1 and set(tables[0])=={4,5,6,7},'Fixed physical GPU4-7 mapping missing')
    uuids=[tables[0][i] for i in range(4,8)]
    require(len(set(uuids))==4 and all(isinstance(u,str) and u.startswith('GPU-') for u in uuids),'Invalid fixed GPU UUID mapping')
    require(metadata.get('runtime_environment',{}).get('CUDA_VISIBLE_DEVICES')==','.join(uuids),'Full-gate physical UUID mask differs')
    workers=metadata.get('workers',[]);commands=metadata.get('commands',[]);width=candidate['tp']*candidate['pp']
    require(len(workers)==candidate['replicas'] and len(commands)==len(workers),'Missing worker/command placement coverage')
    ports=[]
    for i,(worker,command) in enumerate(zip(workers,commands)):
        expected=uuids[i*width:(i+1)*width]
        require(worker.get('index')==i and worker.get('gpus')==expected and command.get('CUDA_VISIBLE_DEVICES')==','.join(expected),
                'Worker physical UUID subset or command mask differs')
        port=worker.get('port');require(type(port) is int and port>0,'Missing actual worker port');ports.append(port)
        require(selected.option(command.get('command',[]),'--port')==str(port),'Worker command/port binding differs')
    require(len(set(ports))==len(ports),'Duplicate worker ports')
    views={i:[] for i in range(len(workers))}
    for folder,pair in paired_folders(run,metadata,pins,repo):
        raw=take(folder/'server-before.json',pins,repo)
        if pair is None:
            index=int(folder.name.removeprefix('worker'));views[index].append(raw)
        else:
            require(isinstance(raw,dict) and set(raw)=={str(w['port']) for w in pair.values()},'PD API pair must contain both exact role ports')
            for worker in pair.values():views[worker['index']].append(raw[str(worker['port'])])
    require(all(views.values()),'Missing initial API for a worker/role')
    return views


def checked_response(row,prompt,length,pd,generation):
    require(isinstance(row,dict) and 'error' not in row,'Invalid generation body')
    generation(row,prompt,length)
    reason=row['meta_info']['finish_reason']
    require(reason.get('length')==length and reason.get('status_code') is None and reason.get('err_type') is None,
            'Generation finish reason differs from requested length')
    if pd:
        meta=row.get('prefill_meta_info',{});finish=meta.get('finish_reason',{})
        require(isinstance(meta,dict) and meta.get('prompt_tokens')==len(prompt)
                and type(meta.get('completion_tokens')) is int and meta['completion_tokens'] in (0,1)
                and finish.get('type')=='length' and finish.get('length')==0
                and finish.get('status_code') is None and finish.get('err_type') is None,
                'Missing/failed PD prefill completion')


def normalized_model_cancellation(rows,port,classifier):
    require(isinstance(rows,list) and len(rows)==16,'All16 model cancellation captures required')
    prefix=None
    for i,row in enumerate(rows):
        rid=row.get('requested_rid') if isinstance(row,dict) else None
        require(isinstance(rid,str) and re.fullmatch(r'a100-cancel-\d+-\d+',rid),'Missing cancellation RID')
        if prefix is None:prefix=rid.rsplit('-',1)[0]+'-'
        require(rid==prefix+str(i) and set(row.get('responses',{}))=={'model'},'Cancellation order/role differs')
        capture=row['responses']['model']
        require(capture.get('port')==port and capture.get('requested_rid')==rid and 'error' not in capture,
                'Cancellation capture port/RID/error differs')
        raw=capture.get('raw_body');encoded=capture.get('raw_body_base64')
        require(isinstance(raw,str) and isinstance(encoded,str),'Missing raw cancellation response')
        require(base64.b64decode(encoded,validate=True).decode('utf-8',errors='replace')==raw
                and json.loads(raw)==capture.get('response'),'Raw cancellation bytes/text/JSON differ')
        require(classifier(row,rid),'Cancellation is not a matching AbortReq terminal')


def audit_full_gate(run,metadata,pins,repo):
    run=Path(run).resolve();args=metadata['args'];pd=args['family']=='pd'
    require(metadata.get('status')=='execution_complete' and args.get('suite')=='correctness'
            and args.get('stress200') is True,'Raw coverage requires completed full correctness/stress200')
    generation=pure_functions(GENERATION,{'checked_generation'},pins,repo)['checked_generation']
    recovery=pure_functions(RECOVERY,{'require','compare'},pins,repo)['compare']
    classifier=pure_functions(Path(repo)/'scripts/a100_gpu4567/server_run.py',{'_cancellation_response_aborted','cancellation_confirmed'},pins,repo)['cancellation_confirmed']
    folders=paired_folders(run,metadata,pins,repo);result=[]
    for folder,pair in folders:
        prompts=take(folder/'prompts.json',pins,repo);pressure=take(folder/'stress-prompts.json',pins,repo)
        require(isinstance(prompts,list) and len(prompts)==14 and [len(p) for p in prompts]==LENGTHS,'Actual full14 prompt lengths differ')
        require(isinstance(pressure,list) and len(pressure)==200
                and [len(p) for p in pressure]==[(128,256,512,1024,2048,4096)[i%6] for i in range(200)]
                and len({selected.digest(p) for p in pressure})==200,'Actual distinct stress200 prompts differ')
        require(all(p and p[0]==128000 and all(type(v) is int and 0<=v<128256 for v in p) for p in prompts+pressure),'Invalid actual prompt token IDs')
        payloads={}
        for name,count,length in [('serial',14,64),('cache',2,32),('concurrent',32,32),('after-cancel',1,16),('stress200',200,32)]:
            value=take(folder/f'{name}.json',pins,repo);rows=[value] if name=='after-cancel' else value
            require(isinstance(rows,list) and len(rows)==count,f'Missing/truncated {name} response coverage')
            for i,row in enumerate(rows):
                prompt=pressure[i] if name=='stress200' else prompts[9] if name=='cache' else prompts[0] if name=='after-cancel' else prompts[i%14]
                checked_response(row,prompt,length,pd,generation)
            payloads[name]=rows
        cached=payloads['cache'];cache_meta=cached[1].get('prefill_meta_info',cached[1]['meta_info'])
        require(cached[0]['output_ids']==cached[1]['output_ids'] and cache_meta.get('cached_tokens',0)>=8000,'Cache reuse coverage failed')
        require(recovery(prompts,payloads['serial'],payloads['after-cancel'][0])['status']=='passed','Same-run recovery16-token prefix differs')
        warm=take(folder/'long-warmup.json',pins,repo);rng=random.Random(730031)
        checked_response(warm,[128000]+[rng.randrange(128000) for _ in range(31662)],8,pd,generation)
        cancellations=take(folder/'cancellation.json',pins,repo)
        require(isinstance(cancellations,list) and len(cancellations)==16,'All16 cancellation rows required')
        normalized=all(isinstance(row,dict) and 'responses' in row for row in cancellations)
        require(normalized, 'Pool full gates require normalized raw cancellation captures')
        if normalized:
            if pd:selected.validate_pd_capture_records(cancellations,pair,classifier)
            else:normalized_model_cancellation(cancellations,metadata['workers'][int(folder.name.removeprefix('worker'))]['port'],classifier)
        result.append(dict(folder=str(folder),serial=14,cache=2,concurrent=32,cancellation=16,recovery_tokens=16,stress200=200))
    return dict(status='passed',folders=result,scope='Raw coverage and same-run recovery only; source/platform/resources and cross-system numerical scope remain separate gates.')
