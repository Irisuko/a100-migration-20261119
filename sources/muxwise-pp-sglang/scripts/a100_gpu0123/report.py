#!/usr/bin/env python3
"""Summarize preserved measurements; never invent metrics for unexecuted jobs."""
import argparse,csv,json,statistics,hashlib,math
from collections import defaultdict
from pathlib import Path
import resource_audit
import formal_attempts
from send_goodput import send_relative_goodput
from run_campaign import read_platform_provenance

PRIMARY='ttft_le_5s_and_tpot_le_100ms'
REPO=Path(__file__).resolve().parents[2]
NONFORMAL={'support','feasibility','smoke','preflight','correctness','model_correctness',
           'topology_feasibility','development','development_selection','rate_calibration','pilot',
           'unclassified_not_formal'}

# The primary protocol retains send-relative TTFT. Scheduled-arrival curves
# explicitly include client send lag; their goodput is a separate supplement.
CURVE_METRICS={
 'send_ttft_p95_s':'TTFT p95 (s; request-send origin)',
 'send_tpot_p95_s':'TPOT p95 (s; request-send metrics)',
 'send_e2e_p95_s':'E2E p95 (s; request-send origin)',
 'slo_send':'SLO fraction (TTFT<=5s, TPOT<=100ms; send origin)',
 'arrival_ttft_p95_s':'TTFT p95 (s; scheduled-arrival origin)',
 'slo_arrival':'SLO fraction (TTFT<=5s, TPOT<=100ms; scheduled-arrival origin)',
 'send_goodput_req_s':'Goodput (requests/s; send-origin SLO, arrival window)',
 'send_goodput_output_s':'Goodput (output tokens/s; send-origin SLO, arrival window)',
 'send_drain_goodput_req_s':'Goodput (requests/s; send-origin SLO, arrival plus drain)',
 'send_drain_goodput_output_s':'Goodput (output tokens/s; send-origin SLO, arrival plus drain)',
 'goodput_req_s':'Goodput (requests/s; scheduled-arrival SLO, arrival window)',
 'goodput_output_s':'Goodput (output tokens/s; scheduled-arrival SLO, arrival window)',
 'arrival_output_s':'Observed output tokens/s (arrival window)',
 'drain_output_s':'Observed output tokens/s (arrival plus drain)',
}

def curve_series_key(record):
 """Join validated rate points only by stable serving/config/model/platform IDs.

 Trace, planned intent, arrival rate and their queue evidence differ by point.
 Their exact identities remain in rows/aggregates and all repetition gates.
 This key is for drawing lines only; it never pools or validates measurements.
 """
 identity=json.loads(record['source_identity'])
 keys=('runtime','configuration','model_revision','model_provenance','platform_software')
 if isinstance(identity,dict) and all(identity.get(key) for key in keys):
  source=json.dumps({key:identity[key] for key in keys},sort_keys=True,separators=(',',':'))
 else:
  # Unknown provenance must not be silently merged into another rate point.
  source=record['source_identity']
 return (record['phase'],record['family'],record['configuration'],source)

def send_goodput_fields(output,account,arrival_window_sec):
 """Primary goodput only after finalized, all-planned raw-accounting validation."""
 fields={'send_goodput_req_s':None,'send_goodput_output_s':None,
         'send_drain_goodput_req_s':None,'send_drain_goodput_output_s':None,
         'send_goodput_scope':'Send-relative TTFT<=5s and TPOT<=100ms; successful complete requests only. Arrival-window duration stays nominal; full drain ends at the validated observation/cutoff. Unknown accounting is not zero service.'}
 if account.get('status')!='complete' or account.get('metrics_observed') is not True:return fields
 records=[json.loads(line) for line in (Path(output)/'trace_requests.jsonl').read_text().splitlines() if line.strip()]
 values=send_relative_goodput(records,account,arrival_window_sec)
 for prefix,window in [('send_goodput','arrival_window'),('send_drain_goodput','full_drain')]:
  fields[prefix+'_req_s']=values[window]['main_slo_requests_per_sec']
  fields[prefix+'_output_s']=values[window]['main_slo_output_tokens_per_sec']
 return fields

def resolve(path):
 value=Path(path)
 return (value if value.is_absolute() else REPO/value).resolve()

def source_binding_matches(runtime,queued):
 """Compare resolved paths without rewriting either recorded source identity."""
 def normalized(mapping):
  result={}
  for name,value in mapping.items():
   path=resolve(name)
   if path in result and result[path]!=value:return None
   result[path]=value
  return result
 actual,frozen=normalized(runtime),normalized(queued)
 return bool(actual) and frozen is not None and all(frozen.get(path)==value for path,value in actual.items())

def option(argv,name):
 values=[]
 for i,value in enumerate(argv):
  if value=='--extra':break
  if value==name and i+1<len(argv):values.append(argv[i+1])
  elif value.startswith(name+'='):values.append(value.split('=',1)[1])
 if len(values)>1:raise ValueError(f'Ambiguous {name} in queue task')
 return values[0] if values else None

def queue_provenance(root,output):
 matches=[]
 for path in sorted((root/'runs').glob('*/queue-task.json')):
  data=json.loads(path.read_text());task=data.get('task',{})
  destination=option(task.get('argv',[]),'--out')
  if destination and resolve(destination)==output.resolve():matches.append((path,data))
 if len(matches)>1:raise ValueError(f'Ambiguous queue provenance for output {output}')
 if not matches:return {'phase':'unclassified_not_formal','source_sha256':{},'formal_eligible':False,'reason':'No queue task matches this exact output path'}
 path,data=matches[0];task=data['task'];evidence=data.get('evidence',{})
 phase=task['phase'];formal=phase not in NONFORMAL
 result={'phase':phase,'source_sha256':evidence.get('source_sha256',{}),'formal_eligible':False,
         'queue_task_path':str(path),'queue_id':data.get('queue_id'),'task_id':task.get('id'),
         'plan_sha256':data.get('plan_sha256'),'freeze_sha256':evidence.get('freeze_sha256'),
         'trace_sha256':evidence.get('trace_sha256'),'task_argv':task.get('argv',[]),
         'platform':read_platform_provenance(root,data),
         'reason':'Nonformal task' if not formal else 'Missing or mismatched freeze evidence'}
 outcome_path=path.parent/'outcome.json'
 if outcome_path.is_file():
  try:
   outcome=json.loads(outcome_path.read_text())
   result.update(controller_status=outcome.get('status'),controller_returncode=outcome.get('returncode'),
                 controller_cleanup_error=outcome.get('cleanup_error'),controller_outcome_sha256=file_hash(outcome_path))
  except (OSError,ValueError,TypeError):result['controller_status']='invalid_outcome'
 result['controller_completed']=(result.get('controller_status')=='completed'
                                 and result.get('controller_returncode')==0
                                 and not result.get('controller_cleanup_error'))
 if formal and evidence.get('freeze_path') and evidence.get('freeze_sha256'):
  try:
   freeze_path=resolve(evidence['freeze_path']);raw=freeze_path.read_bytes();frozen=json.loads(raw)
   frozen_task=frozen.get('tasks',{}).get(task['id'],{})
   valid=(hashlib.sha256(raw).hexdigest()==evidence['freeze_sha256']
          and frozen.get('status')=='frozen' and bool(evidence.get('source_sha256'))
          and frozen.get('source_sha256')==evidence['source_sha256']
          and all(frozen_task.get(key)==task.get(key) for key in ('phase','argv','timeout'))
          and bool(evidence.get('trace_sha256'))
          and frozen_task.get('trace_sha256')==evidence['trace_sha256'])
   if valid:result.update(formal_eligible=True,reason='Exact task/source/trace freeze evidence verified')
  except (OSError,ValueError,KeyError,TypeError):pass
 return result

def phase_group(phase):
 if phase=='development_selection':return 'selection'
 if phase in ('development','pilot','rate_calibration'):return 'development'
 if phase in NONFORMAL:return phase
 return 'formal'

def write_csv(path,rows,fields):
 with path.open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)

def file_hash(path):
 value=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block)
 return value.hexdigest()

def observed_number(value):
 return value if type(value) in (int,float) and math.isfinite(value) and value>=0 else None

def planned_intent_summary(output,metadata,summary,provenance,cache=None):
 """Verify the complete arrival schedule before treating formal runs as repeats."""
 result=dict(status='failed',sha256=None,errors=[])
 cache={} if cache is None else cache
 def require(value,message):
  if not value:raise ValueError(message)
 def rows(path):
  with path.open() as stream:
   return [{key:row[key] for key in ('idx','arrival','input_len','output_len')}
           for line in stream if line.strip() for row in [json.loads(line)]]
 try:
  args=metadata['args'];argv=provenance.get('task_argv',[])
  trace=resolve(summary['trace']);queued_trace=option(argv,'--trace')
  require(queued_trace is not None and resolve(queued_trace)==trace==resolve(args['trace']),
          'Queue, runtime and summary trace paths differ')
  scale=float(summary['time_scale'])
  require(math.isfinite(scale) and scale>0 and scale==float(args.get('time_scale',1))
          ==float(option(argv,'--time-scale') or 1),'Queue, runtime and summary time scale differ')
  limit=option(argv,'--limit');limit=int(limit) if limit is not None else None
  require(limit==args.get('limit')==summary.get('limit') and (limit is None or limit>0),
          'Queue, runtime and summary request limit differ')
  stat=trace.stat();key=(str(trace),stat.st_size,stat.st_mtime_ns)
  if key not in cache:cache[key]=rows(trace)
  expected=cache[key][:limit] if limit is not None else cache[key]
  planned=rows(Path(output)/'trace_planned.jsonl')
  require(planned and len(planned)==len(expected)==summary['planned_requests'],
          'Planned count differs from complete queued trace/window')
  require(len({row['idx'] for row in planned})==len(planned),'Duplicate planned request IDs')
  for actual,source in zip(planned,expected):
   require(all(actual[field]==source[field] for field in ('idx','input_len','output_len')),
           'Planned request ID or length differs from actual trace')
   arrival=float(actual['arrival']);source_arrival=float(source['arrival'])
   require(math.isfinite(arrival) and arrival>=0 and math.isfinite(source_arrival)
           and math.isclose(arrival,source_arrival*scale,rel_tol=1e-12,abs_tol=1e-9),
           'Planned arrival differs from actual trace times queued scale')
  window=float(summary['arrival_window_sec']);explicit=option(argv,'--arrival-window-sec')
  require((explicit is None and args.get('arrival_window_sec') is None)
          or (explicit is not None and float(explicit)==args.get('arrival_window_sec')),
          'Queue and runtime arrival-window flags differ')
  expected_window=(float(explicit) if explicit is not None else
                   max(row['arrival'] for row in planned) if limit is not None else
                   float(json.loads(trace.with_suffix('.stats.json').read_text())['duration_sec'])*scale)
  require(math.isfinite(window) and window>0 and window>=max(row['arrival'] for row in planned)
          and math.isclose(window,expected_window,rel_tol=1e-12,abs_tol=1e-9),
          'Summary arrival window differs from queued window or trace duration')
  normalized=[{**row,'arrival':float(row['arrival'])} for row in planned]
  result.update(status='verified',sha256=hashlib.sha256(json.dumps(normalized,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest())
 except (OSError,ValueError,KeyError,TypeError,AttributeError) as error:
  result['errors'].append(str(error))
 return result

def resource_summary(metadata,audit):
 """TP/PP ranks replicate logical token capacity; only independent instances add."""
 args=metadata.get('args',{})
 tp,pp,replicas=(args.get(key,0) for key in ('tp','pp','replicas'))
 if any(type(n) is not int or n<1 for n in (tp,pp,replicas)):tp=pp=replicas=0
 workers={worker['index']:worker for worker in audit.get('workers',[])}
 layout={worker['index']:worker for worker in metadata.get('workers',[])}
 gpu_peaks=audit.get('gpu_memory_peak_mib',{})
 instances=[];ranks=[]
 for index in range(replicas):
  worker=workers.get(index,{})
  entries=worker.get('ranks',[])
  recorded={(rank.get('pp'),rank.get('tp')):rank for rank in entries}
  expected={(p,t) for p in range(pp) for t in range(tp)}
  complete=len(entries)==len(expected) and set(recorded)==expected
  capacities=[]
  assigned=layout.get(index,{})
  for p,t in sorted(expected):
   rank=recorded.get((p,t),{})
   capacity=rank.get('capacity')
   capacity=capacity if type(capacity) is int and capacity>0 else None
   capacities.append(capacity)
   gpu_index=p*tp+t;gpus=assigned.get('gpus',[])
   gpu=gpus[gpu_index] if gpu_index<len(gpus) else None
   memory=rank.get('cuda_memory_peak_bytes',{})
   ranks.append(dict(worker=index,role=assigned.get('role'),pp=p,tp=t,gpu_uuid=gpu,
                     kv_capacity_tokens=capacity,request_capacity=observed_number(rank.get('request_capacity')),
                     peak_torch_allocated_bytes=observed_number(memory.get('max_allocated')),
                     peak_torch_reserved_bytes=observed_number(memory.get('max_reserved')),
                     graph_capture_gib=observed_number(rank.get('graph_capture_gb')),
                     peak_device_memory_mib=observed_number(gpu_peaks.get(gpu)) if gpu else None))
  known={value for value in capacities if value is not None}
  consistent=False if len(known)>1 else True if complete and all(value is not None for value in capacities) else None
  capacity=next(iter(known)) if consistent is True else None
  instances.append(dict(index=index,role=assigned.get('role'),logical_kv_capacity_tokens=capacity,
                        rank_capacities_consistent=consistent,expected_ranks=len(expected),observed_ranks=len(recorded)))
 consistency=False if any(row['rank_capacities_consistent'] is False for row in instances) else True if instances and all(row['rank_capacities_consistent'] is True for row in instances) else None
 total=sum(row['logical_kv_capacity_tokens'] for row in instances) if consistency is True else None
 roles={}
 for role in {row['role'] for row in instances if row['role'] is not None}:
  values=[row['logical_kv_capacity_tokens'] for row in instances if row['role']==role]
  roles[role]=sum(values) if all(value is not None for value in values) else None
 return dict(status=audit.get('status','incomplete'),logical_kv_capacity_tokens=total,
             rank_capacities_consistent=consistency,logical_kv_capacity_by_role=roles,
             instances=instances,ranks=ranks,errors=audit.get('errors',[]),missing_evidence=audit.get('missing_evidence',[]),
             scope='KV is summed once per independent model instance or PD role pool, never over TP/PP ranks. PD pool sums are not end-to-end request capacity. Memory peaks cover deployment, including warmup/calibration; device memory is sampled and Torch excludes non-Torch allocations. Graph capture is the logged before/after free-memory difference in GiB (runtime prints GB but divides bytes by 2**30), rounded to two decimals; it is not a peak. Per-device peaks need not be simultaneous.')

def latency_slo_fields(summary):
 result={}
 scopes={'send':summary,'arrival':summary.get('arrival_metrics',{})}
 for scope,data in scopes.items():
  latencies=data.get('latencies_sec',{})
  result[f'latencies_{scope}_sec']=json.dumps(latencies,sort_keys=True)
  for name in ('ttft','tpot','e2e'):
   for percentile in (50,90,95,99):
    result[f'{scope}_{name}_p{percentile}_s']=latencies.get(name,{}).get(f'p{percentile}')
  result[f'slo_{scope}_all']=json.dumps(data.get('slo',{}),sort_keys=True)
  for name,value in data.get('slo',{}).items():
   result[f'slo_{scope}_{name}']=value.get('rate')
 # Preserve historical report aliases while also naming both timing origins.
 for name in ('ttft','tpot','e2e'):
  scope='send' if name=='tpot' else 'arrival'
  for percentile in (50,90,95,99):result[f'{name}_p{percentile}_s']=result[f'{scope}_{name}_p{percentile}_s']
 return result

def summarize(root):
 dest=root/'report';dest.mkdir(exist_ok=True)
 manifest=json.loads((root/'campaign.json').read_text())
 write_csv(dest/'candidate-status.csv',manifest['candidates'],['id','model','family','topology','tp','pp','replicas','conditional_two_gpu','status','blocking_reason'])
 rows=[];resources=[];trace_hashes={};trace_intents={}
 ledger=formal_attempts.collect(root,read_platform_provenance)
 attempt_by_output={entry['output']:entry for entry in ledger['attempts']}
 for path in sorted(root.rglob('trace_summary.json')):
  meta_path=path.parent/'run.json'
  if not meta_path.exists():continue
  attempt=attempt_by_output.get(str(path.parent.resolve()))
  try:
   s=json.loads(path.read_text());meta=json.loads(meta_path.read_text());args=meta['args']
   for key in ('trace','time_scale','planned_requests','arrival_window_sec','completed_requests','failed_requests','unfinished_requests','unsent_requests'):s[key]
   for key in ('model','family','tp','pp','replicas'):args[key]
   s['slo'][PRIMARY]['rate'];s['throughput']['input_tokens_per_sec'];s['latencies_sec']['tpot']['p95']
   for scope in ('arrival_window','full_drain'):
    for key in ('main_slo_requests_per_sec','main_slo_output_tokens_per_sec','observed_output_tokens_per_sec'):s['arrival_metrics']['goodput'][scope][key]
   s['arrival_metrics']['slo'][PRIMARY]['rate']
   for key in ('ttft','e2e'):s['arrival_metrics']['latencies_sec'][key]['p95']
  except (OSError,ValueError,KeyError,TypeError) as error:
   if not attempt:raise
   attempt['errors'].append(f'Unusable measurement metadata/summary: {error}')
   continue
  trace=resolve(s['trace'])
  if trace not in trace_hashes:trace_hashes[trace]=file_hash(trace) if trace.is_file() else None
  actual_trace_hash=trace_hashes[trace]
  provenance=queue_provenance(root,path.parent)
  intent=planned_intent_summary(path.parent,meta,s,provenance,trace_intents)
  platform=provenance.get('platform',{'status':'missing','software':None,'configuration':{'mooncake_transport':'unrecorded'}})
  phase=provenance['phase']
  if attempt:phase=attempt['phase']
  formal=phase not in NONFORMAL
  resource=resource_summary(meta,resource_audit.audit_run(path.parent))
  resources.append(dict(label=path.parent.name,run_dir=str(path.parent),**resource))
  source_frozen=source_binding_matches(meta.get('source_sha256',{}),provenance['source_sha256'])
  model_known=bool(meta.get('model_revision')) and bool(meta.get('model_provenance'))
  formal_eligible=(provenance['formal_eligible'] and meta.get('status')=='execution_complete'
                   and provenance.get('controller_completed',False)
                   and bool(actual_trace_hash) and actual_trace_hash==meta.get('trace_sha256')==provenance.get('trace_sha256')
                   and source_frozen and model_known and resource['status']=='passed'
                   and intent['status']=='verified' and platform['status']=='verified'
                   and attempt is not None and attempt['intent_valid']
                   and attempt.get('queue_status') in (None,'completed')
                   and attempt['request_accounting']['status']=='complete'
                   and s.get('run_status')=='completed' and s.get('failed_requests')==0)
  arrival=s['arrival_metrics'];win=arrival['goodput']['arrival_window'];drain=arrival['goodput']['full_drain']
  config={k:v for k,v in args.items() if k not in ['out','trace','time_scale','limit','port','startup_timeout','suite','arrival_window_sec']}
  config['platform_transport']=platform['configuration']
  row=dict(label=path.parent.name,deployment_start_unix=meta.get('start_unix'),measurement_eligible=s.get('measurement_eligible',False),run_status=s.get('run_status','unknown'),source_identity=json.dumps(meta.get('source_sha256',{}),sort_keys=True),phase=meta.get('phase','unclassified_not_formal'),model=args['model'],family=args['family'],topology=f'tp{args["tp"]}pp{args["pp"]}rep{args["replicas"]}',workload='toolagent' if 'toolagent' in trace.name else 'conversation',time_scale=s['time_scale'],absolute_req_s=s['planned_requests']/s['arrival_window_sec'] if s['arrival_window_sec'] else None,planned=s['planned_requests'],completed=s['completed_requests'],failed=s['failed_requests'],unfinished=s['unfinished_requests'],unsent=s['unsent_requests'],completion_rate=s['completed_requests']/s['planned_requests'],slo_send=s['slo'][PRIMARY]['rate'],slo_arrival=arrival['slo'][PRIMARY]['rate'],ttft_p95_s=arrival['latencies_sec']['ttft']['p95'],tpot_p95_s=s['latencies_sec']['tpot']['p95'],e2e_p95_s=arrival['latencies_sec']['e2e']['p95'],goodput_req_s=win['main_slo_requests_per_sec'],goodput_output_s=win['main_slo_output_tokens_per_sec'],arrival_output_s=win['observed_output_tokens_per_sec'],drain_output_s=drain['observed_output_tokens_per_sec'],drain_input_s=s['throughput']['input_tokens_per_sec'],cache_ratio=s.get('cached_tokens',{}).get('token_weighted_ratio'),configuration=json.dumps(config,sort_keys=True),source=str(path))
  row.update(phase=phase,phase_group=phase_group(phase),formal_eligible=bool(s.get('measurement_eligible',False) and formal_eligible),
             aggregation_eligible=bool(s.get('measurement_eligible',False) and (not formal or formal_eligible)),
             source_identity=json.dumps({'runtime':meta.get('source_sha256',{}),'queue':provenance['source_sha256'],
                                         'configuration':meta.get('configuration_sha256',{}),
                                         'model_revision':meta.get('model_revision'),
                                         'model_provenance':meta.get('model_provenance',{}),
                                         'actual_trace_sha256':actual_trace_hash,
                                         'runtime_trace_sha256':meta.get('trace_sha256'),
                                         'queued_trace_sha256':provenance.get('trace_sha256')},sort_keys=True),
             queue_provenance=json.dumps(provenance,sort_keys=True))
  identity=json.loads(row['source_identity'])
  identity.update(planned_intent_sha256=intent['sha256'],time_scale=s['time_scale'],
                  arrival_window_sec=s['arrival_window_sec'],platform_software=platform['software'])
  row.update(source_identity=json.dumps(identity,sort_keys=True),planned_intent_status=intent['status'],
             planned_intent_sha256=intent['sha256'],planned_intent_errors=json.dumps(intent['errors']),
             platform_provenance_status=platform['status'],platform_configuration=json.dumps(platform['configuration'],sort_keys=True))
  row.update(latency_slo_fields(s))
  pd=args['family']=='pd'
  reported_cache=s.get('cached_tokens',{}).get('token_weighted_ratio')
  row.update(cache_ratio=None if pd else reported_cache,
             cache_ratio_reported_decode=reported_cache if pd else None,
             cache_ratio_scope='global_unknown_pd_prefill_counters_unavailable' if pd else 'serving_endpoint_reported_token_weighted',
             resource_audit_status=resource['status'],logical_kv_capacity_tokens=resource['logical_kv_capacity_tokens'],
             kv_rank_capacities_consistent=resource['rank_capacities_consistent'],
             kv_capacity_by_instance=json.dumps(resource['instances'],sort_keys=True),
             kv_capacity_by_role=json.dumps(resource['logical_kv_capacity_by_role'],sort_keys=True),
             rank_memory_and_capacity=json.dumps(resource['ranks'],sort_keys=True),
             resource_scope=resource['scope'],trace_sha256=actual_trace_hash,
             formal_source_frozen=source_frozen,formal_model_provenance_present=model_known)
  if meta.get('diagnostic_only') or s.get('diagnostic_only'):
   row.update(measurement_eligible=False,aggregation_eligible=False,formal_eligible=False)
  if formal:
   account=attempt['request_accounting'] if attempt else {}
   row.update(send_goodput_fields(path.parent,account,s['arrival_window_sec']))
   row.update(output_path=str(path.parent.resolve()),formal_intent_fingerprint=attempt.get('base_fingerprint') if attempt else None,
              formal_platform_fingerprint=attempt.get('platform_fingerprint') if attempt else None,
              request_accounting_status=account.get('status','unknown'),reported_planned_requests=s['planned_requests'],
              planned=attempt.get('planned_requests') if attempt else None,
              absolute_req_s=attempt.get('absolute_req_s') if attempt else None)
   if not account.get('metrics_observed'):
    for name in list(row):
     if name in ('completion_rate','cache_ratio','cache_ratio_reported_decode','goodput_req_s','goodput_output_s','arrival_output_s','drain_output_s','drain_input_s') or name.startswith(('slo_','latencies_')) or (name.endswith('_s') and '_p' in name):row[name]=None
    for target,source in [('completed','completed_requests'),('failed','failed_requests'),('unfinished','unfinished_requests'),('unsent','unsent_requests')]:row[target]=account.get(source)
   if attempt:
    attempt['comparison_eligible']=row['formal_eligible']
    attempt['observed_source_identity']=row['source_identity']
  rows.append(row)
 fields=list(dict.fromkeys(key for row in rows for key in row)) if rows else ['label','phase','model','family','topology','workload','absolute_req_s','planned','completed','slo_arrival','goodput_req_s']
 write_csv(dest/'measurements.csv',rows,fields)
 (dest/'resource-details.json').write_text(json.dumps(resources,indent=2,allow_nan=False)+'\n')
 groups=defaultdict(list)
 for r in rows:
  if r['aggregation_eligible'] or r['phase'] not in NONFORMAL:groups[(r['phase'],r['model'],r['workload'],r['family'],r['configuration'],r['source_identity'],r['absolute_req_s'])].append(r)
 # Frozen attempts with no summary still have an explicit, unmeasured group.
 represented={(r.get('formal_intent_fingerprint'),r.get('formal_platform_fingerprint')) for r in rows}
 for entry in ledger['attempts']:
  identity=(entry['base_fingerprint'],entry.get('platform_fingerprint'))
  if identity in represented or (identity[1] is None and any(base==identity[0] for base,_ in represented)):continue
  key=(entry['phase'],entry['model'],entry['workload'],entry['family'],json.dumps({'frozen_argv':entry.get('identity',{}).get('configuration'),'platform_fingerprint':identity[1]}),entry['base_fingerprint'],entry.get('absolute_req_s'))
  groups[key].append(dict(formal_intent_fingerprint=entry['base_fingerprint'],formal_platform_fingerprint=entry.get('platform_fingerprint'),formal_eligible=False,
                          output_path=entry['output'],deployment_start_unix=None,_missing_measurement=True))
 aggregates=[]
 metrics=['slo_arrival','slo_send','ttft_p95_s','tpot_p95_s','e2e_p95_s','goodput_req_s','goodput_output_s','arrival_output_s','drain_output_s']
 metrics+=sorted(set(CURVE_METRICS)-set(metrics))
 metrics+=sorted({key for row in rows for key in row
                  if ((key.startswith(('send_','arrival_','ttft_p','tpot_p','e2e_p')) and key.endswith('_s') and '_p' in key)
                      or key.startswith(('slo_send_','slo_arrival_')) and not key.endswith('_all'))}-set(metrics))
 for key,items in groups.items():
  record=dict(zip(['phase','model','workload','family','configuration','source_identity','absolute_req_s'],key));record['measurement_count']=sum(not row.get('_missing_measurement') for row in items)
  record['independent_run_count']=len({r['deployment_start_unix'] for r in items if r['deployment_start_unix'] is not None})
  record['repetition_requirement_met']=record['independent_run_count']>=3
  record['phase_group']=phase_group(record['phase'])
  record['formal_eligible']=all(r['formal_eligible'] for r in items)
  record['formal_repeated_comparison_ready']=record['formal_eligible'] and record['repetition_requirement_met']
  if record['phase'] not in NONFORMAL:record.update(formal_attempts.assess(items,ledger))
  for metric in metrics:
   values=[r.get(metric) for r in items]
   finite=sum(type(x) in (int,float) and math.isfinite(x) for x in values)
   record[metric]=({'mean':statistics.mean(values),'min':min(values),'max':max(values),
                    'status':'observed','censored_count':0,'finite_count':finite}
                   if finite==len(values) else
                   {'mean':None,'min':None,'max':None,'status':'censored',
                    'reason':'missing or censored values; finite-only averaging is prohibited',
                    'censored_count':len(values)-finite,'finite_count':finite,'raw':values})
   if record['phase'] not in NONFORMAL and not record['formal_repeated_comparison_ready']:
    record[metric].update(mean=None,min=None,max=None,status='withheld',raw=values,
                          reason='Every frozen attempt must be eligible and the explicit frozen independent repetition requirement must be met (default three); survivor averages are prohibited')
  aggregates.append(record)
 ledger['schema_version']=1
 ledger['definitions']='Every referenced frozen formal attempt is retained. Missing records without a finalized summary are unknown, not zero service or proven unsent; observed raw record states are separate from final cutoff accounting. Explicit replay-not-started evidence can establish unsent requests but never observed throughput. Group means require all frozen attempts and the consistent hashed task role/required_independent_repetitions policy. Default/main/critical require at least three; one is allowed only for an explicit auxiliary role with a matching auxiliary phase. Missing attempts never lower the requirement.'
 (dest/'formal-attempts.json').write_text(json.dumps(formal_attempts.public(ledger),indent=2,allow_nan=False)+'\n')
 attempts_csv=[dict(formal_attempts.public(entry),**{key:value for key,value in entry['request_accounting'].items() if key!='errors'},
                    errors=entry['errors']+entry['request_accounting'].get('errors',[])) for entry in ledger['attempts']]
 attempt_fields=['label','output','phase','queue_status','controller_status','controller_returncode','controller_cleanup_error','intent_valid','platform_status','comparison_eligible','status','planned_requests','observed_record_count','observed_completed_records','observed_failed_records','observed_unfinished_records','missing_record_requests','completed_requests','failed_requests','unfinished_requests','unsent_requests','unknown_requests','metrics_observed','errors']
 write_csv(dest/'formal-attempts.csv',attempts_csv,attempt_fields)
 write_csv(dest/'formal-failures.csv',[entry for entry in attempts_csv if not entry.get('comparison_eligible')],attempt_fields)
 (dest/'aggregates.json').write_text(json.dumps(aggregates,indent=2,allow_nan=False)+'\n')
 if aggregates:
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  for metric,ylabel in CURVE_METRICS.items():
   for model in ['8b','70b']:
    for workload in ['toolagent','conversation']:
     subset=[r for r in aggregates if r['model']==model and r['workload']==workload]
     if not subset:continue
     fig,ax=plt.subplots(figsize=(7,4));series=defaultdict(list)
     for r in subset:
      if r[metric]['mean'] is not None:series[curve_series_key(r)].append(r)
     for (phase,family,configuration,source_identity),points in series.items():
      points.sort(key=lambda r:r['absolute_req_s']);x=[r['absolute_req_s'] for r in points];y=[r[metric]['mean'] for r in points]
      cfg=json.loads(configuration);label=f'{phase}/{family} TP{cfg["tp"]} PP{cfg["pp"]} replicas{cfg["replicas"]} src{hashlib.sha256(source_identity.encode()).hexdigest()[:8]}'
      ax.errorbar(x,y,yerr=[[r[metric]['mean']-r[metric]['min'] for r in points],[r[metric]['max']-r[metric]['mean'] for r in points]],marker='o',capsize=3,label=label)
     ax.set(xlabel='Planned requests / second',ylabel=ylabel,title=f'{model} {workload}');ax.grid(alpha=.25)
     if series:ax.legend(fontsize=7)
     fig.tight_layout();fig.savefig(dest/f'{model}-{workload}-{metric}.png',dpi=160);fig.savefig(dest/f'{model}-{workload}-{metric}.pdf');plt.close(fig)
 (dest/'generation.json').write_text(json.dumps({'measurement_count':len(rows),'formal_ready':manifest['formal_ready'],'plots_generated':bool(aggregates),'no_measurements_reason':'No trace measurements exist in this GPU0-3 campaign' if not rows else None},indent=2)+'\n')
 print(json.dumps({'candidates':len(manifest['candidates']),'measurements':len(rows),'report':str(dest)}))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]/'run/a100-gpu0123-20261114');summarize(p.parse_args().root)
