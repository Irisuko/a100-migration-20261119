#!/usr/bin/env python3
"""Resumable inventory and checked preflight entrypoint. No automatic unfreeze."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'run/a100-4gpu-20261114'

def candidates():
 rows=[]
 topologies={'dp4':(1,1,4),'tp2dp2':(2,1,2),'tp4':(4,1,1),'pp2dp2':(1,2,2),'pp4':(1,4,1),'pp2tp2':(2,2,1),'2p2d':(1,1,4),'1p3d':(1,1,4),'3p1d':(1,1,4),'1p1d_tp2':(2,1,2)}
 matrix={'8b':{'chunked':['dp4','tp2dp2','tp4','pp2dp2','pp4','pp2tp2'],'muxwise':['dp4','tp2dp2','tp4'],'muxpp':['pp2dp2','pp4','pp2tp2'],'pd':['2p2d','1p3d','3p1d','1p1d_tp2']},'70b':{'chunked':['tp4','pp4','pp2tp2','tp2dp2','pp2dp2'],'muxwise':['tp4','tp2dp2'],'muxpp':['pp4','pp2tp2','pp2dp2'],'pd':['1p1d_tp2']}}
 for model,families in matrix.items():
  for family,names in families.items():
   for name in names:
    tp,pp,replicas=topologies[name]
    rows.append(dict(id=f'{model}-{family}-{name}',model=model,family=family,topology=name,tp=tp,pp=pp,replicas=replicas,prefill_replicas=int(name[0]) if family=='pd' else None,conditional_two_gpu=model=='70b' and tp*pp==2,status='not_executed',selected_parameters=None,evidence=[]))
 return rows

def create_manifest():
 return {'schema_version':1,'campaign':'a100-4gpu-20261114','created_utc':datetime.now(timezone.utc).isoformat(),'formal_ready':False,'resource_budget':4,'physical_gpus':[4,5,6,7],'precision':'bfloat16','trace_total_length_limit':32768,'primary_slo':{'ttft_seconds':5,'tpot_seconds':.1},'original_metric':'TTFT from actual HTTP send; additional scheduled-arrival SLO accounts send_lag','candidates':candidates(),'phases':{k:'not_executed' for k in ['model_correctness','topology_feasibility','development_selection','rate_calibration','formal_matrix','formal_rates','sustained_toolagent','burst_toolagent','mechanism_ablation','matched_kv_ablation','runtime_profiling']},'freeze':None,'repetition_policy':{'main_comparisons':3,'critical_rates':3,'screening':1},'parameter_selection_policy':{'status':'draft_not_selected','budget':'For every supported system/topology: one common initial configuration and at most three one-factor development alternatives; compare full dev windows for finalists. Unsupported/OOM candidates retain failures and no replacement formal data.','common_dimensions':['mem_fraction_static','max_running_requests','cuda_graph_batch_sizes'],'mux_dimensions':['actual_SM_pairs','CPP_chunk','MLP_chunk','inflight'],'selection':'Primary joint SLO, then TTFT p95, then output throughput on development only; retain best per system and same-topology TP/PP controls.','no_formal_feedback':True},'required_artifacts':['source_patch_and_hashes','model_revision_and_tokenizer_hashes','trace_source_and_generated_hashes','exact_command_and_configuration','UUID_mapping','GPU_and_process_monitoring','per_request_events_and_all_planned_requests','KV_capacity_and_memory_peak','cache_reset_ack','completion_failure_unsent_counts','explicit_arrival_and_drain_windows']}

def write(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('action',choices=['init','status','preflight'])
 p.add_argument('--candidate');p.add_argument('--attempt',type=int,default=1)
 a=p.parse_args();path=OUT/'campaign.json'
 if a.action=='init':
  if path.exists():raise FileExistsError('Campaign exists; never overwrite recorded state')
  write(path,create_manifest());print(path);return
 manifest=json.loads(path.read_text())
 if a.action=='status':
  print(json.dumps({'gpu_execution_disabled':(OUT/'RESERVATION_DISABLED.json').exists(),'formal_ready':manifest['formal_ready'],'phases':manifest['phases'],'candidates':[{k:r[k] for k in ['id','status','conditional_two_gpu','evidence']} for r in manifest['candidates']]},indent=2));return
 if (OUT/'RESERVATION_DISABLED.json').exists():
  raise RuntimeError('GPU execution suspended by user: GPUs4/5 released. Do not remove until user explicitly restores resources.')
 rows=[r for r in manifest['candidates'] if r['id']==a.candidate]
 if len(rows)!=1:raise ValueError('Specify one candidate ID from campaign status')
 row=rows[0];label=f'preflight-{row["id"]}-v{a.attempt}'
 out=OUT/'preflight'/label
 cmd=[str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/a100_4gpu/server_run.py'),'--model',row['model'],'--family',row['family'],'--tp',str(row['tp']),'--pp',str(row['pp']),'--replicas',str(row['replicas']),'--mem','.94' if row['model']=='70b' else '.85','--suite','correctness','--out',str(out)]
 if row['family']=='pd':cmd+=['--prefill-replicas',str(row['prefill_replicas'])]
 controller=[str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/a100_4gpu/control.py'),'run','--label',label,'--timeout','2400','--',*cmd]
 row.update(status='running');row['evidence'].append(str(out.relative_to(ROOT)));write(path,manifest)
 result=subprocess.run(controller,cwd=ROOT)
 row['status']='preflight_executed_pending_comparison' if result.returncode==0 else 'preflight_failed'
 write(path,manifest)
 raise SystemExit(result.returncode)

if __name__=='__main__':main()
