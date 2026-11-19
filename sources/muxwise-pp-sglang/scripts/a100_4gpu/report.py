#!/usr/bin/env python3
"""Summarize preserved measurements; never invent metrics for unexecuted jobs."""
import argparse,csv,json,statistics,hashlib
from collections import defaultdict
from pathlib import Path

PRIMARY='ttft_le_5s_and_tpot_le_100ms'

def write_csv(path,rows,fields):
 with path.open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)

def summarize(root):
 dest=root/'report';dest.mkdir(exist_ok=True)
 manifest=json.loads((root/'campaign.json').read_text())
 write_csv(dest/'candidate-status.csv',manifest['candidates'],['id','model','family','topology','tp','pp','replicas','conditional_two_gpu','status','blocking_reason'])
 rows=[]
 for path in sorted(root.rglob('trace_summary.json')):
  meta_path=path.parent/'run.json'
  if not meta_path.exists():continue
  s=json.loads(path.read_text());meta=json.loads(meta_path.read_text());args=meta['args']
  trace=Path(s['trace']);stats=json.loads(trace.with_suffix('.stats.json').read_text())
  arrival=s['arrival_metrics'];win=arrival['goodput']['arrival_window'];drain=arrival['goodput']['full_drain']
  config={k:v for k,v in args.items() if k not in ['out','trace','time_scale','limit','port','startup_timeout','suite','arrival_window_sec']}
  row=dict(label=path.parent.name,deployment_start_unix=meta.get('start_unix'),measurement_eligible=s.get('measurement_eligible',False),run_status=s.get('run_status','unknown'),source_identity=json.dumps(meta.get('source_sha256',{}),sort_keys=True),phase=meta.get('phase','unclassified_not_formal'),model=args['model'],family=args['family'],topology=f'tp{args["tp"]}pp{args["pp"]}rep{args["replicas"]}',workload='toolagent' if 'toolagent' in trace.name else 'conversation',time_scale=s['time_scale'],absolute_req_s=s['planned_requests']/s['arrival_window_sec'] if s['arrival_window_sec'] else None,planned=s['planned_requests'],completed=s['completed_requests'],failed=s['failed_requests'],unfinished=s['unfinished_requests'],unsent=s['unsent_requests'],completion_rate=s['completed_requests']/s['planned_requests'],slo_send=s['slo'][PRIMARY]['rate'],slo_arrival=arrival['slo'][PRIMARY]['rate'],ttft_p95_s=arrival['latencies_sec']['ttft']['p95'],tpot_p95_s=s['latencies_sec']['tpot']['p95'],e2e_p95_s=arrival['latencies_sec']['e2e']['p95'],goodput_req_s=win['main_slo_requests_per_sec'],goodput_output_s=win['main_slo_output_tokens_per_sec'],arrival_output_s=win['observed_output_tokens_per_sec'],drain_output_s=drain['observed_output_tokens_per_sec'],drain_input_s=s['throughput']['input_tokens_per_sec'],cache_ratio=s['cached_tokens']['token_weighted_ratio'],configuration=json.dumps(config,sort_keys=True),source=str(path))
  rows.append(row)
 fields=list(rows[0]) if rows else ['label','phase','model','family','topology','workload','absolute_req_s','planned','completed','slo_arrival','goodput_req_s']
 write_csv(dest/'measurements.csv',rows,fields)
 groups=defaultdict(list)
 for r in rows:
  if r['measurement_eligible']:groups[(r['phase'],r['model'],r['workload'],r['family'],r['configuration'],r['source_identity'],r['absolute_req_s'])].append(r)
 aggregates=[]
 metrics=['slo_arrival','slo_send','ttft_p95_s','tpot_p95_s','e2e_p95_s','goodput_req_s','goodput_output_s','arrival_output_s','drain_output_s']
 for key,items in groups.items():
  record=dict(zip(['phase','model','workload','family','configuration','source_identity','absolute_req_s'],key));record['measurement_count']=len(items)
  record['independent_run_count']=len({r['deployment_start_unix'] for r in items if r['deployment_start_unix'] is not None})
  record['repetition_requirement_met']=record['independent_run_count']>=3
  for metric in metrics:
   values=[r[metric] for r in items]
   record[metric]={'mean':statistics.mean(values),'min':min(values),'max':max(values)} if all(isinstance(x,(int,float)) for x in values) else {'mean':None,'min':None,'max':None,'reason':'missing or censored values','raw':values}
  aggregates.append(record)
 (dest/'aggregates.json').write_text(json.dumps(aggregates,indent=2)+'\n')
 if aggregates:
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  for metric in ['ttft_p95_s','tpot_p95_s','slo_arrival','goodput_req_s','arrival_output_s']:
   for model in ['8b','70b']:
    for workload in ['toolagent','conversation']:
     subset=[r for r in aggregates if r['model']==model and r['workload']==workload]
     if not subset:continue
     fig,ax=plt.subplots(figsize=(7,4));series=defaultdict(list)
     for r in subset:
      if r[metric]['mean'] is not None:series[(r['phase'],r['family'],r['configuration'],r['source_identity'])].append(r)
     for (phase,family,configuration,source_identity),points in series.items():
      points.sort(key=lambda r:r['absolute_req_s']);x=[r['absolute_req_s'] for r in points];y=[r[metric]['mean'] for r in points]
      cfg=json.loads(configuration);label=f'{phase}/{family} TP{cfg["tp"]} PP{cfg["pp"]} replicas{cfg["replicas"]} src{hashlib.sha256(source_identity.encode()).hexdigest()[:8]}'
      ax.errorbar(x,y,yerr=[[r[metric]['mean']-r[metric]['min'] for r in points],[r[metric]['max']-r[metric]['mean'] for r in points]],marker='o',capsize=3,label=label)
     ax.set(xlabel='Planned requests / second',ylabel=metric,title=f'{model} {workload}');ax.grid(alpha=.25);ax.legend(fontsize=7);fig.tight_layout();fig.savefig(dest/f'{model}-{workload}-{metric}.png',dpi=160);fig.savefig(dest/f'{model}-{workload}-{metric}.pdf');plt.close(fig)
 (dest/'generation.json').write_text(json.dumps({'measurement_count':len(rows),'formal_ready':manifest['formal_ready'],'plots_generated':bool(aggregates),'no_measurements_reason':'No new four-GPU model runs; user releasedGPU4/5' if not rows else None},indent=2)+'\n')
 print(json.dumps({'candidates':len(manifest['candidates']),'measurements':len(rows),'report':str(dest)}))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]/'run/a100-4gpu-20261114');summarize(p.parse_args().root)
