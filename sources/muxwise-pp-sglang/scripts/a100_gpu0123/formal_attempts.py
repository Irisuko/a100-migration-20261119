"""CPU-only formal intent inventory and request accounting, including missing runs."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
NONFORMAL={'support','feasibility','smoke','preflight','correctness','model_correctness',
          'topology_feasibility','development','development_selection','rate_calibration','pilot',
          'unclassified_not_formal'}
AUXILIARY_PHASES={
    'mechanism_ablation':{'mechanism_ablation','formal_mechanism_ablation'},
    'matched_kv':{'matched_kv_ablation','formal_matched_kv_ablation'},
    'matched_kv_ablation':{'matched_kv_ablation','formal_matched_kv_ablation'},
    'burst_auxiliary':{'burst_toolagent','burst_auxiliary','formal_burst_auxiliary'},
    'communication_auxiliary':{'communication','communication_auxiliary','formal_communication_auxiliary'},
}
sys.path.insert(0,str(HERE/'bench'))
try:
    spec=importlib.util.spec_from_file_location('_formal_request_metrics',HERE/'bench/metrics.py')
    metrics=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics)
finally:
    sys.path.pop(0)


def require(value,message):
    if not value:raise ValueError(message)


def resolve(value):
    path=Path(value)
    return (path if path.is_absolute() else REPO/path).resolve()


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def option(argv,name,default=None):
    found=[]
    for index,arg in enumerate(argv):
        if arg=='--extra':break
        if arg==name:
            require(index+1<len(argv),f'Missing {name} value');found.append(argv[index+1])
        elif arg.startswith(name+'='):found.append(arg.split('=',1)[1])
    require(len(found)<=1,f'Duplicate {name}')
    return found[0] if found else default


def repetition_policy(frozen_task):
    """Only exact, hashed auxiliary task declarations may lower the default."""
    role=frozen_task.get('role','main')
    count=frozen_task.get('required_independent_repetitions',3)
    require(isinstance(role,str) and role,'Invalid frozen comparison role')
    require(type(count) is int and count>=1,'Invalid frozen independent repetition requirement')
    if count<3:
        require(count==1 and role in AUXILIARY_PHASES and frozen_task['phase'] in AUXILIARY_PHASES[role]
                and 'role' in frozen_task and 'required_independent_repetitions' in frozen_task,
                'Only an explicitly frozen auxiliary role and matching auxiliary phase may require one run; main/critical/default require at least three')
    return dict(role=role,required_independent_repetitions=count)


def intent(task,frozen,cache):
    argv=task['argv'];path=resolve(option(argv,'--trace'))
    expected=task.get('trace_sha256')
    require(expected and sha(path)==expected,'Frozen trace bytes are missing or changed')
    key=(str(path),expected)
    if key not in cache:
        with path.open() as stream:
            cache[key]=[{k:row[k] for k in ('idx','arrival','input_len','output_len')}
                        for line in stream if line.strip() for row in [json.loads(line)]]
    scale=float(option(argv,'--time-scale',1));limit=option(argv,'--limit')
    require(math.isfinite(scale) and scale>0,'Invalid frozen replay scale')
    require(limit is None or int(limit)>0,'Invalid frozen request limit')
    rows=cache[key][:int(limit)] if limit is not None else cache[key]
    planned=[{**row,'arrival':float(row['arrival'])*scale} for row in rows]
    require(planned and len({r['idx'] for r in planned})==len(planned),'Frozen request plan is empty or duplicated')
    explicit=option(argv,'--arrival-window-sec')
    window=(float(explicit) if explicit is not None else max(r['arrival'] for r in planned) if limit is not None
            else float(json.loads(path.with_suffix('.stats.json').read_text())['duration_sec'])*scale)
    require(math.isfinite(window) and window>0 and all(math.isfinite(r['arrival']) and 0<=r['arrival']<=window for r in planned),
            'Frozen arrival window does not cover the complete planned schedule')
    excluded={'--out','--port','--startup-timeout','--trace','--time-scale','--arrival-window-sec','--limit'}
    configuration=[];index=0
    while index<len(argv):
        arg=argv[index]
        if arg=='--extra':configuration+=argv[index:];break
        name=arg.split('=',1)[0]
        if name in excluded:index+=1 if '=' in arg else 2
        else:configuration.append(arg);index+=1
    identity=dict(phase=task['phase'],configuration=configuration,source_sha256=frozen['source_sha256'],
                  trace_sha256=expected,planned_intent_sha256=digest(planned),time_scale=scale,arrival_window_sec=window)
    return dict(base_fingerprint=digest(identity),identity=identity,planned_requests=len(planned),
                model=option(argv,'--model','unknown'),family=option(argv,'--family','unknown'),
                workload=task.get('workload') or ('toolagent' if 'toolagent' in path.name else 'conversation'),
                absolute_req_s=len(planned)/window,_planned=planned)


def equal_numbers(actual,expected):
    if type(expected) in (int,float):
        return type(actual) in (int,float) and math.isfinite(actual) and math.isclose(actual,expected,rel_tol=1e-9,abs_tol=1e-9)
    if isinstance(expected,dict):
        return isinstance(actual,dict) and all(key in actual and equal_numbers(actual[key],value) for key,value in expected.items())
    return actual==expected


def request_accounting(output,planned,summary=None):
    total=len(planned);path=Path(output)/'trace_requests.jsonl'
    result=dict(status='unknown',planned_requests=total,observed_record_count=0,completed_requests=None,
                failed_requests=None,unfinished_requests=None,unsent_requests=None,unknown_requests=total,
                observed_completed_records=None,observed_failed_records=None,observed_unfinished_records=None,
                missing_record_requests=total,
                metrics_observed=False,errors=[],historical=None,arrival=None)
    if not path.is_file():
        result['errors'].append('Raw request records are missing; absence does not prove zero traffic or unsent requests')
        return result
    try:
        records=[]
        with path.open() as stream:
            for line in stream:
                if line.strip():records.append(json.loads(line))
        result['observed_record_count']=len(records)
        ids={r['idx']:r for r in planned}
        require(len({r['idx'] for r in records})==len(records) and all(r['idx'] in ids for r in records),'Duplicate or unplanned raw request IDs')
        for row in records:
            require(all(row[k]==ids[row['idx']][k] for k in ('input_len','output_len')),'Raw request lengths differ from frozen plan')
            sent=row['sent_at'];first=row.get('first_token_at');finished=row.get('finished_at')
            require(type(row.get('success')) is bool,'Raw success state must be explicit')
            require(row.get('status') in ('completed','failed','unfinished') and
                    row['success']==(row['status']=='completed'),'Inconsistent raw terminal state')
            require(type(sent) in (int,float) and math.isfinite(sent) and sent>=0,'Invalid send timestamp')
            require(all(t is None or type(t) in (int,float) and math.isfinite(t) and t>=sent for t in (first,finished)),
                    'Invalid first-token or terminal timestamp')
            require(first is None or finished is None or finished>=first,'Completion precedes first token')
            require('chunks' in row and isinstance(row['chunks'],list),'Timestamped output chunks required')
            previous_time=sent;previous_count=0;token_times=[]
            for chunk in row['chunks']:
                require(type(chunk['at']) in (int,float) and math.isfinite(chunk['at']) and chunk['at']>=previous_time
                        and type(chunk['completion_tokens']) is int and previous_count<=chunk['completion_tokens']<=row['output_len'],
                        'Invalid chunk timestamp or cumulative output count')
                if chunk['completion_tokens']>previous_count:token_times.append(chunk['at'])
                previous_time=chunk['at'];previous_count=chunk['completion_tokens']
            require(first==(token_times[0] if token_times else None),'First-token timestamp differs from output chunks')
            require(finished is None or previous_time<=finished,'Output chunk follows terminal timestamp')
            require(not row['success'] or (finished is not None and previous_count==row['output_len']),
                    'Successful response lacks the complete requested output')
            row.update(output_tokens=previous_count,send_lag=sent-ids[row['idx']]['arrival'],
                       itl=[right-left for left,right in zip(token_times,token_times[1:])],
                       ttft=first-sent if first is not None else None,
                       e2e=finished-sent if row['success'] and finished is not None else None,
                       tpot=((finished-sent)-(first-sent))/(previous_count-1) if row['success'] and first is not None and finished is not None and previous_count>1 else None)
        result.update(observed_completed_records=sum(row['status']=='completed' for row in records),
                      observed_failed_records=sum(row['status']=='failed' for row in records),
                      observed_unfinished_records=sum(row['status']=='unfinished' for row in records),
                      missing_record_requests=total-len(records))
        if summary is None:
            result['errors'].append('No finalized client summary; missing records cannot be classified as unsent')
            return result
        require(summary.get('planned_requests')==total,'Summary planned denominator differs from frozen intent')
        if summary.get('measurement_started') is False:
            require(not records,'Client claims replay never started but request records exist')
            require(summary.get('run_started_unix') is None and all(summary.get(key)==0 for key in
                    ('completed_requests','failed_requests','unfinished_requests')) and summary.get('unsent_requests')==total,
                    'Replay-not-started summary has inconsistent origin or request counts')
            result.update(status='not_started',completed_requests=0,failed_requests=0,unfinished_requests=0,
                          unsent_requests=total,unknown_requests=0)
            return result
        require(summary.get('measurement_started') is True
                and type(summary.get('run_started_unix')) in (int,float) and math.isfinite(summary['run_started_unix']),
                'No explicit measured replay origin')
        cutoff=summary.get('cutoff_time')
        require(cutoff is None or type(cutoff) in (int,float) and math.isfinite(cutoff) and cutoff>=0,'Invalid cutoff')
        historical=metrics.summarize_records(records,total,cutoff)
        arrival=metrics.summarize_arrivals(records,planned,cutoff,float(summary['arrival_window_sec']))
        checks={key:historical[key] for key in ('completed_requests','failed_requests','unfinished_requests','unsent_requests','slo','throughput','latencies_sec')}
        require(equal_numbers(summary,checks) and equal_numbers(summary.get('arrival_metrics'),arrival),
                'Saved summary disagrees with metrics recomputed from all frozen planned requests')
        result.update(status='complete',metrics_observed=True,unknown_requests=0,historical=historical,arrival=arrival,
                      **{key:historical[key] for key in ('completed_requests','failed_requests','unfinished_requests','unsent_requests')})
    except (OSError,ValueError,KeyError,TypeError,AttributeError) as error:
        result['status']='invalid';result['errors'].append(str(error))
    return result


def collect(root,platform_reader):
    """Union frozen intents and durable queue plans, not just successful outputs."""
    root=Path(root).resolve();references={};sources={};errors=[];cache={}
    def register(task,data,path):
        if task.get('phase') in NONFORMAL:return
        output=resolve(option(task['argv'],'--out'))
        require(root in output.parents,'Formal output lies outside campaign root')
        entry=sources.setdefault(str(output),dict(task=task,data=None,queue_plan=None,paths=[]))
        require(all(entry['task'].get(k)==task.get(k) for k in ('id','phase','argv','timeout')),'Conflicting formal attempt identities')
        if path.name=='queue-task.json':entry['data']=data
        else:entry['queue_plan']=data
        entry['paths'].append(str(path))
    for path in sorted((root/'runs').glob('*/queue-task.json')):
        data=json.loads(path.read_text());task=data['task']
        if task.get('phase') in NONFORMAL:continue
        register(task,data,path)
        e=data.get('evidence',{})
        if e.get('freeze_path') and e.get('freeze_sha256'):references[str(resolve(e['freeze_path']))]=e['freeze_sha256']
    folders={p.parent for pattern in ('*/plan.json','*/platform-launch.json') for p in (root/'queues').glob(pattern)}
    for folder in sorted(folders):
        path=folder/'plan.json';platform_path=folder/'platform-launch.json'
        platform=json.loads(platform_path.read_text()) if platform_path.is_file() else {}
        if path.is_file():plan=json.loads(path.read_text())
        else:
            tasks=resolve(platform['tasks_path'])
            require(sha(tasks)==platform['tasks_sha256'],'Scheduled platform task bytes changed')
            plan=json.loads(tasks.read_text());path=platform_path
        formal=[t for t in plan.get('tasks',[]) if t.get('phase') not in NONFORMAL]
        if not formal:continue
        state_path=folder/'status.json';state=json.loads(state_path.read_text()) if state_path.is_file() else {}
        require(not state or state.get('plan_sha256')==digest(plan),'Durable queue plan hash differs from queue state')
        freeze_path=platform.get('freeze_path');freeze_hash=platform.get('freeze_sha256')
        if freeze_path and freeze_hash:references[str(resolve(freeze_path))]=freeze_hash
        for task in formal:
            data=dict(queue_id=folder.name,plan_sha256=digest(plan),task=task,
                      evidence=state.get('tasks',{}).get(task['id'],{}))
            prior=data['evidence']
            if prior.get('freeze_path') and prior.get('freeze_sha256'):
                references[str(resolve(prior['freeze_path']))]=prior['freeze_sha256']
            if freeze_path:data['freeze_reference']=(str(resolve(freeze_path)),freeze_hash)
            register(task,data,path)
    attempts={}
    for name,expected in references.items():
        file=Path(name)
        try:
            require(sha(file)==expected,'Freeze bytes changed')
            frozen=json.loads(file.read_text())
            require(frozen.get('status')=='frozen' and frozen.get('source_sha256'),'Missing frozen source identity')
            for label,frozen_task in frozen.get('tasks',{}).items():
                if frozen_task.get('phase') in NONFORMAL:continue
                task={**frozen_task,'id':label};output=resolve(option(task['argv'],'--out'))
                require(root in output.parents,'Frozen output lies outside campaign root')
                key=str(output);source=sources.get(key,{});data=source.get('data') or source.get('queue_plan')
                entry=dict(label=label,output=key,phase=task['phase'],intent_valid=True,errors=[],_task=task,
                           freeze_path=name,freeze_sha256=expected,evidence_paths=source.get('paths',[]),
                           queue_status=(source.get('queue_plan') or {}).get('evidence',{}).get('status'),
                           platform_status='missing',platform_fingerprint=None)
                try:
                    entry.update(intent(task,frozen,cache))
                    entry['repetition_policy']=repetition_policy(frozen_task)
                    if source:
                        require(all(source['task'].get(k)==task.get(k) for k in ('id','phase','argv','timeout')),'Frozen task differs from scheduled task')
                    if data:
                        effective=dict(data,evidence=dict(source_sha256=frozen['source_sha256'],trace_sha256=task['trace_sha256'],
                                                         freeze_path=name,freeze_sha256=expected))
                        if source.get('data'):
                            require(all(data.get('evidence',{}).get(k)==v for k,v in effective['evidence'].items()),
                                    'Executed task freeze/source/trace evidence differs')
                        platform=platform_reader(root,effective)
                        entry['_platform']=platform
                        entry['platform_status']=platform['status']
                        if platform['status']=='verified':entry['platform_fingerprint']=digest({k:platform[k] for k in ('software','configuration')})
                except (OSError,ValueError,KeyError,TypeError) as error:
                    entry['intent_valid']=False;entry['errors'].append(str(error))
                    if 'base_fingerprint' not in entry:
                        entry.update(base_fingerprint='invalid:'+digest(task),planned_requests=None,
                                     model=option(task['argv'],'--model','unknown'),family=option(task['argv'],'--family','unknown'),
                                     workload=task.get('workload','unknown'))
                        errors.append(f'Unassignable frozen intent {label}: {error}')
                if key in attempts:
                    require(attempts[key].get('base_fingerprint')==entry.get('base_fingerprint'),'Conflicting frozen attempts share an output')
                attempts[key]=entry
        except (OSError,ValueError,KeyError,TypeError) as error:errors.append(f'{name}: {error}')
    for output,source in sources.items():
        if output not in attempts:
            task=source['task']
            attempts[output]=dict(label=task['id'],output=output,phase=task['phase'],intent_valid=False,
                                 errors=['No verified frozen intent for scheduled formal attempt'],evidence_paths=source['paths'],
                                 platform_status='missing',platform_fingerprint=None,planned_requests=None,
                                 base_fingerprint='invalid:'+digest(task),model=option(task['argv'],'--model','unknown'),
                                 family=option(task['argv'],'--family','unknown'),workload=task.get('workload','unknown'))
    for entry in attempts.values():
        out=Path(entry['output']);outcome_path=root/'runs'/entry['label']/'outcome.json'
        summary_path=out/'trace_summary.json'
        try:
            outcome=json.loads(outcome_path.read_text()) if outcome_path.is_file() else {}
            summary=json.loads(summary_path.read_text()) if summary_path.is_file() else None
            entry['controller_status']=outcome.get('status','missing')
            entry['controller_returncode']=outcome.get('returncode')
            entry['controller_cleanup_error']=outcome.get('cleanup_error')
            entry['summary_present']=summary is not None
            entry['request_accounting']=(request_accounting(out,entry['_planned'],summary) if entry.get('_planned') else
                                         dict(status='unknown',planned_requests=None,unknown_requests=None,metrics_observed=False,errors=['Unverified request denominator']))
            if summary is not None and entry.get('identity'):
                require(summary.get('time_scale')==entry['identity']['time_scale']
                        and summary.get('arrival_window_sec')==entry['identity']['arrival_window_sec'],
                        'Client summary scale/window differs from frozen intent')
        except (OSError,ValueError,KeyError,TypeError) as error:
            entry['errors'].append(str(error));entry.setdefault('controller_status','invalid')
            entry['request_accounting']=dict(status='invalid',planned_requests=entry.get('planned_requests'),
                                             unknown_requests=entry.get('planned_requests'),metrics_observed=False,errors=[str(error)])
    return dict(attempts=list(attempts.values()),errors=errors)


def assess(items,ledger):
    """Only a complete frozen group can contribute a repeated comparison."""
    bases={row.get('formal_intent_fingerprint') for row in items}
    platforms={row.get('formal_platform_fingerprint') for row in items}
    matching=[entry for entry in ledger['attempts'] if entry['base_fingerprint'] in bases
              and (entry.get('platform_fingerprint') is None or entry['platform_fingerprint'] in platforms)]
    valid={row['output_path']:row for row in items if row.get('formal_eligible')}
    reasons=list(ledger['errors']);invalid=bool(reasons)
    policies={(entry.get('repetition_policy',{}).get('role','main'),
               entry.get('repetition_policy',{}).get('required_independent_repetitions',3)) for entry in matching}
    if len(policies)>1:
        reasons.append('Frozen role/repetition requirements differ across attempts; a group cannot downgrade or split its repeat requirement')
        invalid=True
    role,required=next(iter(policies)) if len(policies)==1 else ('inconsistent',max((count for _,count in policies),default=3))
    for entry in matching:
        issues=list(entry['errors']);account=entry['request_accounting']
        if entry.get('queue_status') not in (None,'completed'):
            issues.append(f"Queue task did not complete: {entry['queue_status']}")
            invalid|=entry['queue_status'] not in ('pending','running','blocked')
        if not entry['intent_valid']:issues.append('Frozen intent is unverified');invalid=True
        if entry['controller_status']!='completed' or entry.get('controller_returncode')!=0 or entry.get('controller_cleanup_error'):
            issues.append('Controller did not complete cleanly')
            invalid|=entry['controller_status'] not in ('missing','pending','running','blocked')
        if account['status']!='complete':issues+=account.get('errors',[])+[f"Request accounting: {account['status']}"]
        invalid|=account['status']=='invalid'
        if entry['output'] not in valid:issues.append('No eligible measurement with this exact source/model/configuration/platform identity')
        if issues:reasons.append(dict(label=entry['label'],issues=issues))
    if not matching:reasons.append('No verified frozen attempts match this measurement group');invalid=True
    independent=len({row['deployment_start_unix'] for row in valid.values() if row.get('deployment_start_unix') is not None})
    complete=not reasons and {entry['output'] for entry in matching}==set(valid)
    status='complete' if complete else 'invalid' if invalid else 'incomplete'
    return dict(scheduled_attempt_count=len(matching),valid_measurement_count=len(valid),
                independent_run_count=independent,repetition_requirement_met=independent>=required,
                required_independent_repetitions=required,frozen_comparison_role=role,
                attempt_accounting_status=status,attempt_accounting_errors=reasons,
                attempt_labels=[entry['label'] for entry in matching],
                formal_eligible=complete,formal_repeated_comparison_ready=complete and independent>=required)


def public(value):
    if isinstance(value,dict):return {key:public(item) for key,item in value.items() if not key.startswith('_')}
    if isinstance(value,list):return [public(item) for item in value]
    return value
