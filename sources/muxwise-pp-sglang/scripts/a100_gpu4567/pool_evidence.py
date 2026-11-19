"""CPU-only evidence boundary for the independent physical GPU4-7 pool.

No historical main-pool run is qualified here. Serving identities remain the
actual saved paths and bytes; a full functional gate is not numerical equivalence.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
ROOT=REPO/'run/a100-gpu0123-20261114/pool-gpu4567'
UUIDS=['GPU-725d9550-4cdc-f401-0cb0-3d31bccec1af','GPU-69f3abce-1e88-4391-95dc-4ab4c2c77593',
       'GPU-fd1b979e-abe3-dc82-8a17-890c8cd3cf14','GPU-0c16ece9-91b7-1520-f2ea-69811b52901a']
CPUS=list(range(32,64))+list(range(96,128))
NN_FINGERPRINT='1bab3148e9f3ceb49d77b370a4525e623df6c61af8a5b8669958627c453b79f2'

def require(value,message):
    if not value:raise ValueError(message)

def read(path):return json.loads(Path(path).read_text())
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):result.update(block)
    return result.hexdigest()
def resolve(path,repo=REPO):
    path=Path(path);return (path if path.is_absolute() else Path(repo)/path).resolve()
def pin(pins,path,expected=None,repo=REPO):
    path=resolve(path,repo);actual=sha(path)
    require(expected is None or actual==expected,f'Evidence changed: {path}')
    require(str(path) not in pins or pins[str(path)]==actual,f'Conflicting evidence: {path}')
    pins[str(path)]=actual;return path

def local_module(name):
    spec=importlib.util.spec_from_file_location('_pool4567_'+name,HERE/(name+'.py'))
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

validate_pd_capture_records=local_module('pd_capture').validate_pd_capture_records

def option(argv,name):
    found=[]
    for i,value in enumerate(argv):
        if value=='--extra':break
        if value==name:
            require(i+1<len(argv),f'Missing option value: {name}');found.append(argv[i+1])
        elif value.startswith(name+'='):found.append(value[len(name)+1:])
    require(len(found)<=1,f'Duplicate option: {name}')
    return found[0] if found else None

def pool_identity():
    return dict(pool_id='gpu4567-numa1',indices=[4,5,6,7],uuids=list(UUIDS),numa_node=1,
                cpu_set='32-63,96-127',controller_root=str(ROOT.resolve()))

def normalized_sources(mapping,repo=REPO):
    require(isinstance(mapping,dict) and mapping,'Missing source pins')
    result={}
    for key,value in mapping.items():
        path=resolve(key,repo)
        require(isinstance(value,str) and re.fullmatch('[a-f0-9]{64}',value),'Invalid source hash')
        require(path not in result or result[path]==value,'Conflicting source path aliases')
        result[path]=value
    return result

def current_sources(repo=REPO):
    repo=Path(repo).resolve();runtime={str(p.relative_to(repo)):sha(p) for p in sorted((repo/'python/sglang').rglob('*.py'))}
    require(len(runtime)==618 and digest(runtime)==NN_FINGERPRINT,'Reviewed NN source fingerprint changed')
    folder=repo/'scripts/a100_gpu4567'
    serving=[folder/'server_run.py',*[p for p in sorted((folder/'bench').glob('*.py')) if not p.name.startswith('test_')]]
    require(len(serving)==4,'Pool serving source keyset changed')
    return {**runtime,**{str(p.relative_to(repo)):sha(p) for p in serving}}

def validate_source_pins(pins,repo=REPO,exact=False):
    actual=normalized_sources(pins,repo);expected=normalized_sources(current_sources(repo),repo)
    require(all(actual.get(k)==v for k,v in expected.items()),'Missing or changed pool current serving source')
    if exact:require(actual==expected,'Runtime source keyset is not the pool current622')
    return actual

def validate_metadata(run,metadata,repo=REPO):
    run=resolve(run,repo);require(ROOT.resolve() in run.parents,'Measurement outside fixed pool root')
    args=metadata.get('args',{});require(resolve(args.get('out',''),repo)==run,'Metadata output path differs')
    validate_source_pins(metadata.get('source_sha256'),repo,exact=True)
    pool=metadata.get('physical_pool',{})
    for key,value in [('indices',[4,5,6,7]),('uuids',UUIDS),('numa_node',1),('controller_root',str(ROOT.resolve()))]:
        require(pool.get(key)==value,f'Actual physical pool differs: {key}')
    affinity=pool.get('cpu_affinity')
    require(isinstance(affinity,list) and affinity and len(set(affinity))==len(affinity)
            and all(type(c) is int and c in CPUS for c in affinity),'Actual CPU affinity escapes NUMA1')
    require(metadata.get('runtime_environment',{}).get('CUDA_VISIBLE_DEVICES')==','.join(UUIDS),'Actual runtime UUID mask differs')
    tp,pp,replicas=[args.get(k) for k in ('tp','pp','replicas')]
    require(all(type(x) is int and x>0 for x in (tp,pp,replicas)) and tp*pp*replicas==4,'Each experiment must use exactly four GPUs')
    workers,commands=metadata.get('workers',[]),metadata.get('commands',[])
    require(len(workers)==replicas==len(commands),'Missing worker command coverage')
    width=tp*pp;base=args.get('port');require(type(base) is int and 36100<=base<=36199,'Pool HTTP base outside reserved range')
    prefill=args.get('prefill_replicas',0)
    if args.get('family')=='pd':require(type(prefill) is int and 0<prefill<replicas,'Invalid PD topology')
    for i,(w,c) in enumerate(zip(workers,commands)):
        mask=UUIDS[i*width:(i+1)*width];port=base+(i+1 if replicas>1 else 0)
        role=('prefill' if i<prefill else 'decode') if args.get('family')=='pd' else 'model'
        require(w.get('index')==i and w.get('gpus')==mask and w.get('role')==role,'Worker physical mask/index/role differs')
        require(w.get('port')==port and w.get('bootstrap')==base+1000+i and w.get('nccl_port')==38200+100*i,'Worker reserved port placement differs')
        require(c.get('CUDA_VISIBLE_DEVICES')==','.join(mask),'Worker command CUDA mask differs')
        command=c.get('command',[])
        require(command[:3]==[str(Path(repo)/'.venv/bin/python'),'-m','sglang.launch_server'],'Unexpected worker executable')
        for name,value in [('tp-size',tp),('pp-size',pp),('port',port),('nccl-port',38200+100*i),('dtype','bfloat16')]:
            require(option(command,'--'+name)==str(value),f'Worker command differs: {name}')
    return pool

def validate_command(run,metadata,command,task=None,repo=REPO):
    prefix=['taskset','-c','32-63,96-127',str(Path(repo)/'.venv/bin/python'),str(Path(repo)/'scripts/a100_gpu4567/server_run.py')]
    argv=command.get('argv',[]);require(argv[:5]==prefix,'Controller command lacks exact pool executable/CPU binding')
    tail=argv[5:];args=metadata['args']
    if task is not None:require(tail==task['argv'],'Actual controller command differs from queued task')
    require(resolve(option(tail,'--out') or '',repo)==resolve(run,repo),'Controller output differs')
    require(command.get('gpu_mapping')=={str(i):u for i,u in zip(range(4,8),UUIDS)},'Controller physical GPU mapping differs')
    env=command.get('environment',{})
    require(env.get('CUDA_VISIBLE_DEVICES')==','.join(UUIDS)
            and env.get('A100_GPU4567_CAMPAIGN_ROOT')==str(ROOT.resolve()),'Controller pool environment differs')
    # Compare every explicit runner option (including flags) with saved effective args.
    i=0
    while i<len(tail):
        token=tail[i];require(token.startswith('--'),'Unexpected controller argument')
        key=token[2:].replace('-','_')
        if key=='extra':require(tail[i+1:]==args.get('extra',[]),'Controller extra args differ');break
        require(key in args,'Unknown controller option')
        value=True
        if i+1<len(tail) and not tail[i+1].startswith('--'):i+=1;value=tail[i]
        expected=args[key]
        if type(expected) is bool: equal=value is expected
        elif type(expected) in (int,float): equal=type(value) is str and type(expected)(value)==expected
        else: equal=str(expected)==str(value)
        require(equal,f'Controller/runtime argument differs: {key}')
        i+=1
    return tail

def worker_api(run,metadata,pins,repo=REPO):
    args=metadata['args'];result=[]
    enabled=args.get('overlap')=='on' or (args.get('overlap')=='auto' and args['family'] in ('plain','chunked','pd') and args['pp']==1)
    for w in metadata['workers']:
        path=pin(pins,Path(run)/f"worker{w['index']}"/'server-final.json',repo=repo);api=read(path)
        values=api.get('server_args',api)
        expected={'tp_size':args['tp'],'pp_size':args['pp'],'base_gpu_id':0,'dtype':'bfloat16',
                  'disable_overlap_schedule':not enabled,'enable_pdmux':args['family'] in ('muxwise','muxpp'),
                  'disaggregation_mode':w['role'] if args['family']=='pd' else 'null',
                  'disable_custom_all_reduce':not args.get('custom_allreduce',False),
                  'mem_fraction_static':args['mem'],'max_running_requests':args['running'],'cuda_graph_max_bs':args['graph']}
        for key,value in expected.items():require(values.get(key)==value,f'Actual worker API differs: worker{w["index"]}/{key}')
        require(api.get('tokenizer_pending_requests')==0 and api.get('tokenizer_finished_states')==0,'Tokenizer is not recovered')
        result.append(dict(index=w['index'],role=w['role'],port=w['port'],actual=expected,evidence_path=str(path)))
    return result

def resources(run,metadata,pins,repo=REPO):
    module=local_module('resource_audit');pin(pins,HERE/'resource_audit.py',repo=repo)
    report=module.audit_run(run)
    require(report['status']=='passed',f'Resource gate failed: {report["errors"]+report["missing_evidence"]}')
    count=sum(len(w['ranks']) for w in report['workers'])
    require(count==4 and len(report['workers'])==metadata['args']['replicas'],'Missing four-rank resource coverage')
    require(set(report['gpu_memory_peak_mib'])==set(UUIDS),'GPU memory monitor lacks exact four UUIDs')
    return report

def custom_allreduce(run,metadata):
    args=metadata['args'];markers=[];covered=set()
    expression=re.compile(r'Custom allreduce enabled: rank=(\d+) world_size=(\d+) physical_device_ids=(\[[^]]+\]) full_nvlink=(True|False)')
    for w in metadata['workers']:
        for number,line in enumerate((Path(run)/f"server-{w['index']}.log").read_text(errors='replace').splitlines(),1):
            match=expression.search(line)
            if not match:continue
            rank,world=int(match[1]),int(match[2]);ids=ast.literal_eval(match[3]);start=4+w['index']*args['tp']*args['pp']
            groups=[list(range(start+j*args['tp'],start+(j+1)*args['tp'])) for j in range(args['pp'])]
            require(ids in groups and world==args['tp'] and 0<=rank<world,'Custom allreduce marker escapes actual TP group')
            require(match[4]=='True','Pool custom allreduce NVLink support not positive')
            covered.add((tuple(ids),rank));markers.append(dict(worker=w['index'],line=number,rank=rank,world_size=world,physical_device_ids=ids,full_nvlink=True))
    requested=args.get('custom_allreduce',False)
    if requested and args['tp']>1:require(len(covered)==4,'Missing positive custom allreduce rank markers')
    return dict(requested=requested,positive_rank_markers=len(covered),markers=markers,
                scope='Initialization support only; does not establish per-message kernel dispatch or latency.')

def transport_evidence(run,metadata,command,pins,repo=REPO):
    if metadata['args']['family']!='pd':
        return dict(mooncake_transport='default',scope='Non-PD run: Mooncake KV transport is not exercised.')
    label=Path(run).name;taskfile=ROOT/'runs'/label/'queue-task.json'
    launcher=local_module('run_campaign')
    if taskfile.is_file():
        provenance=read(pin(pins,taskfile,repo=repo));platform=launcher.read_platform_provenance(ROOT,provenance)
        require(platform.get('status')=='verified' and platform['configuration']['mooncake_transport']=='nvlink','Missing verified pool NVLink queue platform')
        pin(pins,platform['manifest_path'],platform['manifest_sha256'],repo)
    else:
        path=ROOT/'environment'/f'{label}-root-launch.json';launch=read(pin(pins,path,repo=repo))
        argv=launch.get('argv',[])
        require('--' in argv and argv[argv.index('--')+1:]==command['argv'][3:]
                and option(argv,'--label')==label,'Root launch does not bind actual controller command')
        require(launch.get('physical_gpus')==[4,5,6,7] and launch.get('numa_node')==1
                and launch.get('mooncake_transport')=='nvlink' and launch.get('MC_INTRANODE_NVLINK')=='1','Root launch NVLink/pool evidence missing')
        recorded=launch['platform'];_,checked=launcher.load_platform(recorded['environment_path'],recorded['public_environment'])
        require(checked==recorded,'Root launch platform bytes differ')
        platform=dict(status='verified_root_launch',manifest_path=str(path),manifest_sha256=sha(path))
        pin(pins,recorded['environment_path'],recorded['environment_sha256'],repo)
        for key in ('package-sources.json','packages-manifest.json','verification.json'):
            pin(pins,recorded[key]['path'],recorded[key]['sha256'],repo)
        for path,value in recorded['packages_sha256'].items():pin(pins,path,value,repo)
        for path,value in recorded['installed_libraries'].items():pin(pins,path,value['sha256'],repo)
    observed=[]
    for w in metadata['workers']:
        lines=(Path(run)/f"server-{w['index']}.log").read_text(errors='replace').splitlines()
        matches=[i for i,line in enumerate(lines,1) if 'Using Intra-Node NVLink transport (MC_INTRANODE_NVLINK set)' in line]
        require(len(matches)>=metadata['args']['tp']*metadata['args']['pp'],'Missing per-rank actual NVLink transport initialization')
        observed.append(dict(worker=w['index'],lines=matches))
    return dict(mooncake_transport='nvlink',platform=platform,actual_transport_markers=observed,
                scope='Positive transfer-engine transport selection plus successful paired KV workloads; no latency attribution.')

def audit_full_gate(run,repo=REPO):
    run=resolve(run,repo);pins={};result=dict(schema_version=1,status='failed',label=run.name,run=str(run),
        campaign_root=str(ROOT.resolve()),physical_pool=pool_identity(),cross_system_numerical_equivalence=False,errors=[])
    try:
        for name in ('pool_evidence.py','fullgate_evidence.py','pd_capture.py','control.py','server_run.py'):pin(pins,HERE/name,repo=repo)
        metadata=read(pin(pins,run/'run.json',repo=repo));validate_metadata(run,metadata,repo)
        taskdir=ROOT/'runs'/run.name
        outcome=read(pin(pins,taskdir/'outcome.json',repo=repo));command=read(pin(pins,taskdir/'command.json',repo=repo))
        require(outcome.get('status')=='completed' and outcome.get('returncode')==0
                and outcome.get('reservation')=='held' and not outcome.get('cleanup_error'),'Controller did not complete and restore holder')
        validate_command(run,metadata,command,repo=repo)
        for path in sorted(run.rglob('*')):
            if path.is_file() and path.suffix in ('.json','.jsonl','.csv','.log'):pin(pins,path,repo=repo)
        for path,value in metadata['source_sha256'].items():pin(pins,path,value,repo)
        for key,value in metadata.get('configuration_sha256',{}).items():
            path=run/'spatial.yml' if key=='spatial' else metadata.get('configuration_paths',{}).get(key)
            require(path is not None, 'Missing effective configuration path')
            pin(pins,path,value,repo)
        model_path=Path(option(metadata['commands'][0]['command'],'--model-path'))
        for key,value in metadata.get('model_provenance',{}).items():pin(pins,model_path/key,value,repo)
        raw=local_module('fullgate_evidence')
        candidate={k:metadata['args'][k] for k in ('tp','pp','replicas')}
        raw.worker_views(run,metadata,candidate,pins,repo)
        apis=worker_api(run,metadata,pins,repo)
        coverage=raw.audit_full_gate(run,metadata,pins,repo)
        resource=resources(run,metadata,pins,repo)
        transport=transport_evidence(run,metadata,command,pins,repo)
        result.update(status='passed',source_sha256=metadata['source_sha256'],source_fingerprint=digest(metadata['source_sha256']),
            actual_physical_pool=metadata['physical_pool'],topology={k:metadata['args'][k] for k in ('model','family','tp','pp','replicas','prefill_replicas')},
            correctness_verdict=metadata.get('correctness_verdict'),controller=dict(outcome=outcome,command=command),
            worker_api=apis,raw_coverage=coverage,resource_audit=resource,rank_count=4,
            custom_allreduce=custom_allreduce(run,metadata),mooncake_transport=transport['mooncake_transport'],transport_evidence=transport,
            scope='This exact pool configuration completed full functional/cancellation/recovery/stress and quiet resource checks. Same-run recovery is exact; cross-system numerical equivalence is not established.')
    except (ValueError,AssertionError,KeyError,TypeError,OSError,IndexError) as exc:
        result['status']='failed';result['errors'].append(f'{type(exc).__name__}: {exc}')
    result['evidence_sha256']=pins
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=audit_full_gate(args.run)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    return 0 if result['status']=='passed' else 1

if __name__=='__main__':raise SystemExit(main())
