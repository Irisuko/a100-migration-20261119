"""A frozen failed or missing attempt cannot disappear behind successful repeats."""
import csv
import json
from unittest import mock

import pytest

import formal_attempts as attempts
import report
from test_report import measurement, write, audit_fixture


PLATFORM={'status':'verified','software':{'libs':'same'},'configuration':{'mooncake_transport':'default'}}


def measured(out, records=None):
    planned=[json.loads(line) for line in (out/'trace_planned.jsonl').read_text().splitlines()]
    if records is None:
        records=[dict(idx=0,input_len=2,output_len=2,sent_at=.5,first_token_at=.6,finished_at=.65,
                      success=True,status='completed',output_tokens=2,ttft=.6-.5,e2e=.65-.5,tpot=(.65-.5)-(.6-.5),
                      send_lag=0,itl=[.65-.6],chunks=[dict(at=.6,completion_tokens=1),dict(at=.65,completion_tokens=2)])]
    (out/'trace_requests.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
    summary=json.loads((out/'trace_summary.json').read_text())
    summary.update(attempts.metrics.summarize_records(records,len(planned),1))
    summary.update(arrival_metrics=attempts.metrics.summarize_arrivals(records,planned,1,1),
                   measurement_started=True,run_started_unix=10,run_status='completed')
    write(out/'trace_summary.json',summary)
    return summary


def campaign(root,count=4):
    outputs=[];provenances=[];tasks={}
    for i in range(count):
        out,p=measurement(root,f'repeat{i}',phase='formal_main')
        meta=json.loads((out/'run.json').read_text())
        meta.update(start_unix=i+1,model_revision='model1',model_provenance={'config.json':'hash'})
        write(out/'run.json',meta)
        p['task']['argv']+=['--model','8b','--family','chunked','--tp','4','--pp','1','--replicas','1']
        p['evidence']['source_sha256']=meta['source_sha256']
        tasks[out.name]={**p['task'],'trace_sha256':meta['trace_sha256']}
        write(root/'runs'/out.name/'outcome.json',dict(status='completed',returncode=0))
        measured(out);outputs.append(out);provenances.append(p)
    freeze=root/'freeze-all.json'
    write(freeze,dict(status='frozen',source_sha256={'runtime.py':'runtime-hash'},tasks=tasks))
    for out,p in zip(outputs,provenances):
        p['evidence'].update(freeze_path=str(freeze),freeze_sha256=attempts.sha(freeze))
        write(root/'runs'/out.name/'queue-task.json',p)
    return outputs


def summarize(root):
    with mock.patch.object(report.resource_audit,'audit_run',return_value=audit_fixture(((100,100,100,100),))), mock.patch.object(report,'read_platform_provenance',return_value=PLATFORM):
        report.summarize(root)
    return json.loads((root/'report/aggregates.json').read_text()),json.loads((root/'report/formal-attempts.json').read_text())


def test_three_successes_and_failed_fourth_never_average_survivors(tmp_path):
    outputs=campaign(tmp_path)
    write(tmp_path/'runs/repeat3/outcome.json',dict(status='failed',returncode=1))
    groups,ledger=summarize(tmp_path)
    assert len(groups)==1
    group=groups[0]
    assert group['scheduled_attempt_count']==4 and group['valid_measurement_count']==3
    assert not group['formal_repeated_comparison_ready']
    assert group['goodput_req_s']['mean'] is None
    assert group['attempt_accounting_status']=='invalid'
    assert len(ledger['attempts'])==4
    assert 'repeat3' in (tmp_path/'report/formal-failures.csv').read_text()


def test_missing_frozen_repeat_without_any_directory_still_blocks_group(tmp_path):
    outputs=campaign(tmp_path)
    for path in outputs[-1].iterdir():path.unlink()
    for path in (tmp_path/'runs/repeat3').iterdir():path.unlink()
    groups,ledger=summarize(tmp_path)
    assert groups[0]['scheduled_attempt_count']==4 and groups[0]['goodput_req_s']['mean'] is None
    missing=next(row for row in ledger['attempts'] if row['label']=='repeat3')
    assert missing['request_accounting']['planned_requests']==1
    assert missing['request_accounting']['unknown_requests']==1
    assert missing['request_accounting']['unsent_requests'] is None


def test_explicit_startup_failure_counts_unsent_but_never_fake_zero_throughput(tmp_path):
    out=campaign(tmp_path,1)[0]
    summary=json.loads((out/'trace_summary.json').read_text())
    summary.update(measurement_started=False,measurement_eligible=False,run_status='startup_failed',run_started_unix=None)
    summary.update(completed_requests=0,failed_requests=0,unfinished_requests=0,unsent_requests=1)
    summary['throughput']['input_tokens_per_sec']=0
    write(out/'trace_summary.json',summary)
    (out/'trace_requests.jsonl').write_text('')
    write(tmp_path/'runs/repeat0/outcome.json',dict(status='failed',returncode=1))
    groups,ledger=summarize(tmp_path)
    account=ledger['attempts'][0]['request_accounting']
    assert account['status']=='not_started' and account['unsent_requests']==1
    assert account['historical'] is None and groups[0]['goodput_req_s']['mean'] is None
    row=next(csv.DictReader((tmp_path/'report/measurements.csv').open()))
    assert row['goodput_req_s']==row['drain_input_s']==''


def test_raw_failed_and_unsent_requests_remain_in_all_planned_denominator(tmp_path):
    out,_=measurement(tmp_path)
    planned=[dict(idx=i,arrival=i*.1,input_len=2,output_len=2) for i in range(3)]
    (out/'trace_planned.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in planned))
    records=[dict(idx=0,input_len=2,output_len=2,sent_at=.1,first_token_at=None,finished_at=.2,
                  success=False,status='failed',chunks=[],itl=[],send_lag=.1,output_tokens=0,ttft=None,e2e=None,tpot=None)]
    summary=measured(out,records)
    result=attempts.request_accounting(out,planned,summary)
    assert result['status']=='complete' and result['failed_requests']==1 and result['unsent_requests']==2
    assert result['arrival']['slo'][report.PRIMARY]['total']==3
    assert result['arrival']['slo'][report.PRIMARY]['rate']==0
    summary['completed_requests']=1
    assert attempts.request_accounting(out,planned,summary)['status']=='invalid'


def test_four_valid_independent_repeats_can_be_ready(tmp_path):
    campaign(tmp_path)
    groups,ledger=summarize(tmp_path)
    assert groups[0]['formal_repeated_comparison_ready']
    assert groups[0]['scheduled_attempt_count']==4
    assert groups[0]['goodput_req_s']['mean']==1
    assert all(row['request_accounting']['status']=='complete' for row in ledger['attempts'])


def test_missing_other_transport_has_own_failure_group(tmp_path):
    outputs=campaign(tmp_path)
    for path in outputs[-1].iterdir():path.unlink()
    write(tmp_path/'runs/repeat3/outcome.json',dict(status='failed',returncode=1))
    def platform(root,provenance):
        return {**PLATFORM,'configuration':{'mooncake_transport':'nvlink' if provenance['task']['id']=='repeat3' else 'default'}}
    with mock.patch.object(report.resource_audit,'audit_run',return_value=audit_fixture(((100,100,100,100),))), mock.patch.object(report,'read_platform_provenance',side_effect=platform):
        report.summarize(tmp_path)
    groups=json.loads((tmp_path/'report/aggregates.json').read_text())
    assert len(groups)==2
    assert sorted(group['scheduled_attempt_count'] for group in groups)==[1,3]
    assert sum(group['formal_repeated_comparison_ready'] for group in groups)==1
    failed=next(group for group in groups if group['measurement_count']==0)
    assert failed['goodput_req_s']['mean'] is None and failed['attempt_labels']==['repeat3']


def test_raw_only_records_preserve_observed_states_without_final_denominator_claim(tmp_path):
    out=campaign(tmp_path,1)[0]
    planned=[json.loads(line) for line in (out/'trace_planned.jsonl').read_text().splitlines()]
    planned.append(dict(idx=1,arrival=.8,input_len=2,output_len=2))
    account=attempts.request_accounting(out,planned,None)
    assert account['status']=='unknown' and not account['metrics_observed']
    assert account['observed_completed_records']==1 and account['missing_record_requests']==1
    assert account['completed_requests'] is None and account['unsent_requests'] is None


def test_durable_platform_plan_inventories_all_frozen_tasks_before_any_start(tmp_path):
    outputs=campaign(tmp_path,2)
    tasks=[]
    for out in outputs:
        tasks.append(json.loads((tmp_path/'runs'/out.name/'queue-task.json').read_text())['task'])
        for path in (tmp_path/'runs'/out.name).iterdir():path.unlink()
        for path in out.iterdir():path.unlink()
    plan=tmp_path/'tasks.json';write(plan,dict(tasks=tasks))
    freeze=tmp_path/'freeze-all.json'
    write(tmp_path/'queues/before-start/platform-launch.json',dict(tasks_path=str(plan),tasks_sha256=attempts.sha(plan),
          freeze_path=str(freeze),freeze_sha256=attempts.sha(freeze)))
    groups,ledger=summarize(tmp_path)
    assert len(ledger['attempts'])==2 and groups[0]['scheduled_attempt_count']==2
    assert groups[0]['measurement_count']==0 and groups[0]['goodput_req_s']['mean'] is None
    assert all(entry['request_accounting']['unknown_requests']==1 for entry in ledger['attempts'])


def test_corrupt_summary_is_preserved_as_invalid_attempt(tmp_path):
    out=campaign(tmp_path,1)[0]
    (out/'trace_summary.json').write_text('{not-json')
    groups,ledger=summarize(tmp_path)
    assert groups[0]['measurement_count']==0 and not groups[0]['formal_repeated_comparison_ready']
    assert ledger['attempts'][0]['request_accounting']['planned_requests']==1
    assert ledger['attempts'][0]['request_accounting']['status']=='invalid'


def test_normal_cutoff_with_unfinished_requests_is_valid_measurement_not_failed_attempt(tmp_path):
    outputs=campaign(tmp_path,3)
    for out in outputs:
        records=[dict(idx=0,input_len=2,output_len=2,sent_at=.5,first_token_at=.6,finished_at=1.2,
                      success=False,status='unfinished',output_tokens=1,ttft=.6-.5,e2e=1.2-.5,tpot=None,
                      send_lag=0,itl=[],chunks=[dict(at=.6,completion_tokens=1)])]
        measured(out,records)
    groups,ledger=summarize(tmp_path)
    assert groups[0]['formal_repeated_comparison_ready'] and groups[0]['slo_arrival']['mean']==0
    assert all(entry['request_accounting']['unfinished_requests']==1 for entry in ledger['attempts'])


def freeze_policy(root,phase,roles_counts):
    freeze=root/'freeze-all.json';data=json.loads(freeze.read_text())
    for label,(role,count) in zip(data['tasks'],roles_counts):
        data['tasks'][label].update(phase=phase,role=role,required_independent_repetitions=count)
        path=root/'runs'/label/'queue-task.json';provenance=json.loads(path.read_text())
        provenance['task']['phase']=phase;write(path,provenance)
    write(freeze,data)
    for label in data['tasks']:
        path=root/'runs'/label/'queue-task.json';provenance=json.loads(path.read_text())
        provenance['evidence']['freeze_sha256']=attempts.sha(freeze);write(path,provenance)


def test_frozen_auxiliary_role_may_explicitly_allow_one_independent_run(tmp_path):
    campaign(tmp_path,1)
    freeze_policy(tmp_path,'mechanism_ablation',[('mechanism_ablation',1)])
    groups,_=summarize(tmp_path)
    assert groups[0]['required_independent_repetitions']==1
    assert groups[0]['frozen_comparison_role']=='mechanism_ablation'
    assert groups[0]['formal_repeated_comparison_ready'] and groups[0]['goodput_req_s']['mean']==1


@pytest.mark.parametrize('phase,role',[('formal_main','main'),('formal_main','mechanism_ablation'),('critical_rates','communication_auxiliary')])
def test_main_or_critical_cannot_claim_auxiliary_one_repeat(tmp_path,phase,role):
    campaign(tmp_path,1)
    freeze_policy(tmp_path,phase,[(role,1)])
    groups,ledger=summarize(tmp_path)
    assert not groups[0]['formal_repeated_comparison_ready'] and groups[0]['goodput_req_s']['mean'] is None
    assert not ledger['attempts'][0]['intent_valid']


def test_auxiliary_one_repeat_never_drops_failed_second_attempt(tmp_path):
    campaign(tmp_path,2)
    freeze_policy(tmp_path,'mechanism_ablation',[('mechanism_ablation',1)]*2)
    write(tmp_path/'runs/repeat1/outcome.json',dict(status='failed',returncode=1))
    groups,_=summarize(tmp_path)
    assert groups[0]['scheduled_attempt_count']==2 and groups[0]['valid_measurement_count']==1
    assert groups[0]['goodput_req_s']['mean'] is None


def test_inconsistent_frozen_repetition_policy_cannot_split_or_downgrade_group(tmp_path):
    campaign(tmp_path,3)
    freeze_policy(tmp_path,'mechanism_ablation',[('mechanism_ablation',1),('mechanism_ablation',3),('mechanism_ablation',3)])
    groups,_=summarize(tmp_path)
    assert len(groups)==1 and groups[0]['attempt_accounting_status']=='invalid'
    assert not groups[0]['formal_repeated_comparison_ready']


def test_queue_terminal_failure_cannot_be_hidden_by_completed_controller_outcome(tmp_path):
    outputs=campaign(tmp_path,3)
    plan=dict(tasks=[json.loads((tmp_path/'runs'/out.name/'queue-task.json').read_text())['task'] for out in outputs])
    write(tmp_path/'queues/development-v1/plan.json',plan)
    state=dict(plan_sha256=attempts.digest(plan),tasks={out.name:dict(status='failed' if out.name=='repeat2' else 'completed') for out in outputs})
    write(tmp_path/'queues/development-v1/status.json',state)
    groups,ledger=summarize(tmp_path)
    assert not groups[0]['formal_repeated_comparison_ready']
    assert groups[0]['valid_measurement_count']==2
    assert next(row for row in ledger['attempts'] if row['label']=='repeat2')['queue_status']=='failed'


def test_invalid_summary_cannot_replace_frozen_planned_denominator_in_csv(tmp_path):
    out=campaign(tmp_path,1)[0]
    path=out/'trace_summary.json';summary=json.loads(path.read_text());summary['planned_requests']=999;write(path,summary)
    groups,ledger=summarize(tmp_path)
    row=next(csv.DictReader((tmp_path/'report/measurements.csv').open()))
    assert row['planned']=='1' and row['reported_planned_requests']=='999'
    assert row['goodput_req_s']=='' and not groups[0]['formal_repeated_comparison_ready']
