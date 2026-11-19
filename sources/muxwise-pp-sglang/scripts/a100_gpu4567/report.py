#!/usr/bin/env python3
"""Pool-local finalist draft from explicit40-slot plans and an audited budget.

This CPU report does not scan the parent campaign, redeclare slots, adopt a
winner, or assert cross-pool numerical/performance equivalence.
"""
import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent

def load(name):
    spec=importlib.util.spec_from_file_location('_pool_report_'+name,HERE/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

evidence=load('pool_evidence')
selector=load('select_development')
require=evidence.require

def summarize(plans,budget_path,budget_sha256):
    pins={};budget=evidence.read(evidence.pin(pins,budget_path,budget_sha256))
    require(budget.get('total_original_slots')==40 and budget.get('additional_slots')==0,'Expected original40 budget without additional slots')
    require(budget.get('status') in ('valid_budget_snapshot','review_required','deferred_live_queue'),'Unknown budget-audit status')
    require(isinstance(budget.get('evidence_sha256'),dict) and budget['evidence_sha256'],'Budget audit must bind its authoritative plan/ledger/helper evidence')
    for path,sha in budget['evidence_sha256'].items():evidence.pin(pins,path,sha)
    tasks=[]
    for path in plans:
        plan=evidence.read(evidence.pin(pins,path));tasks.extend(plan['tasks'])
    require(len(tasks)==40 and len({t['id'] for t in tasks})==40,'Exactly40 distinct existing execution tasks required')
    canonical=[t.get('canonical_slot_id') for t in tasks]
    require(all(isinstance(v,str) and v for v in canonical) and len(set(canonical))==40,'Canonical slots must be one-to-one')
    slots=budget.get('slots',[])
    require(len(slots)==40 and len({s['execution_task_id'] for s in slots})==40,'Budget execution coverage differs')
    by_id={s['execution_task_id']:s for s in slots}
    require(set(by_id)=={t['id'] for t in tasks},'Budget task identities differ from exact plans')
    rows=[];cache={};groups=defaultdict(list)
    for task in tasks:
        slot=by_id[task['id']]
        require(slot['canonical_slot_id']==task['canonical_slot_id'],'Budget canonical slot differs')
        require(task.get('physical_pool_id')=='gpu4567-numa1','Task belongs to another pool')
        require(type(task.get('profile_slot')) is int and task['profile_slot'] in range(1,5),'Invalid profile slot')
        directory=evidence.ROOT/'runs'/task['id']
        row=dict(label=task['id'],canonical_slot_id=task['canonical_slot_id'],profile_slot=task['profile_slot'],
                 budget_state=slot.get('state'),consumed=slot.get('consumed'),terminal_consistent=slot.get('terminal_consistent'),
                 selection_eligible=False,measurement_available=False,exclusion_reasons=['no_execution_receipt'])
        if (directory/'queue-task.json').is_file():
            try:
                provenance=evidence.read(evidence.pin(pins,directory/'queue-task.json'))
                require(provenance.get('task')==task,'Executed task differs from declared plan')
                row.update(selector.read_measurement(directory,cache))
                for path,sha in row.get('pool_evidence_sha256',{}).items():evidence.pin(pins,path,sha)
            except (ValueError,KeyError,TypeError,OSError,AssertionError) as exc:
                row.update(selection_eligible=False,exclusion_reasons=['invalid_measurement_evidence'],error=f'{type(exc).__name__}: {exc}')
        if slot.get('terminal_consistent') is not True or budget['status']!='valid_budget_snapshot':
            row['selection_eligible']=False;row['exclusion_reasons'].append('budget_not_terminal_and_consistent')
        rows.append(row)
        groups[(task['seed_id'],task['candidate_id'],task['workload'],task['finalist_role'])].append(row)
    results=[]
    for (seed,candidate,workload,role),items in groups.items():
        require(len(items)==4 and {r['profile_slot'] for r in items}=={1,2,3,4},'Each declared group must contain exactly its four original slots')
        terminal=budget['status']=='valid_budget_snapshot' and all(r['terminal_consistent'] is True for r in items)
        eligible=[r for r in items if r['selection_eligible']]
        comparable=False
        if eligible:
            keys=('source_fingerprint','model_fingerprint','trace_sha256','planned_intent_fingerprint','absolute_requests_per_sec')
            comparable=len({evidence.digest({k:r.get(k) for k in keys}) for r in eligible})==1
        ranked=sorted(eligible,key=selector.rank_key) if terminal and comparable else []
        results.append(dict(seed_id=seed,candidate_id=candidate,workload=workload,finalist_role=role,
            all_four_terminal=terminal,common_stream_source_model=comparable,
            status='draft_for_root_review' if ranked else 'withheld',
            draft_best_label=ranked[0]['label'] if ranked else None,
            ranked_eligible_labels=[r['label'] for r in ranked],all_declared_labels=[r['label'] for r in items]))
    return dict(schema_version=1,status='pool_local_draft_not_adopted',physical_pool=evidence.pool_identity(),
                budget_audit=dict(path=str(Path(budget_path).resolve()),sha256=budget_sha256,status=budget['status']),
                total_declared_slots=40,additional_slots=0,rows=rows,groups=results,evidence_sha256=pins,
                scope='Fixed original selector ordering; every declared failed/missing/unfinished slot is retained. No cross-pool merge, parameter declaration, numerical approval, rate freeze, or formal conclusion.')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--tasks',type=Path,action='append',required=True)
    parser.add_argument('--budget-audit',type=Path,required=True);parser.add_argument('--budget-sha256',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=summarize(args.tasks,args.budget_audit,args.budget_sha256)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')

if __name__=='__main__':main()
