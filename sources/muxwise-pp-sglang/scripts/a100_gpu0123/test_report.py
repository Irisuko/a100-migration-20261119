"""Reports must join exact outputs to queue evidence and keep formal gates."""
import hashlib
import csv
import json
from pathlib import Path
from unittest import mock

import pytest
import report


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def measurement(root,label='trial',phase='development',frozen=False):
    out=root/'measurements'/label
    trace=root/'toolagent.jsonl';trace.write_text('{"idx":0,"arrival":0.5,"input_len":2,"output_len":2}\n')
    trace.with_suffix('.stats.json').write_text('{"duration_sec":1}')
    trace_hash=hashlib.sha256(trace.read_bytes()).hexdigest()
    args=dict(model='8b',family='chunked',tp=4,pp=1,replicas=1,suite='trace',
              out=str(out),trace=str(trace),time_scale=1,mem=.85)
    write(out/'run.json',{'args':args,'start_unix':1,'status':'execution_complete',
                         'source_sha256':{'runtime.py':'runtime-hash'},'trace_sha256':trace_hash})
    (out/'trace_planned.jsonl').write_text(trace.read_text())
    metrics=dict(main_slo_requests_per_sec=1,main_slo_output_tokens_per_sec=2,observed_output_tokens_per_sec=3)
    latency={'ttft':{'p95':1},'tpot':{'p95':.1},'e2e':{'p95':2}}
    slo={'ttft_le_5s_and_tpot_le_100ms':{'rate':1}}
    summary=dict(trace=str(trace),measurement_eligible=True,run_status='complete',time_scale=1,
                 planned_requests=1,completed_requests=1,failed_requests=0,unfinished_requests=0,
                 unsent_requests=0,arrival_window_sec=1,slo=slo,latencies_sec=latency,
                 arrival_metrics=dict(goodput={'arrival_window':metrics,'full_drain':metrics},
                                      slo=slo,latencies_sec=latency),
                 throughput={'input_tokens_per_sec':1},cached_tokens={'token_weighted_ratio':0})
    write(out/'trace_summary.json',summary)
    task=dict(id=label,phase=phase,argv=['--out',str(out),'--trace',str(trace)],timeout=60)
    evidence=dict(source_sha256={'frozen.py':'source-hash'},trace_sha256=trace_hash)
    if frozen:
        freeze=root/'freeze.json'
        write(freeze,{'status':'frozen','source_sha256':evidence['source_sha256'],
                      'tasks':{label:{**task,'trace_sha256':trace_hash}}})
        evidence.update(freeze_path=str(freeze),freeze_sha256=hashlib.sha256(freeze.read_bytes()).hexdigest())
    provenance={'queue_id':'development-v1','task':task,'evidence':evidence,'plan_sha256':'plan-hash'}
    write(root/'runs'/label/'queue-task.json',provenance)
    write(root/'campaign.json',{'candidates':[],'formal_ready':False})
    return out,provenance


def test_queue_phase_is_joined_by_exact_output_path_and_sources_preserved(tmp_path):
    out,provenance=measurement(tmp_path,phase='development_selection')
    wrong={**provenance,'task':{**provenance['task'],'phase':'formal_main','argv':['--out',str(out)+'-other']}}
    write(tmp_path/'runs'/'other'/'queue-task.json',wrong)
    result=report.queue_provenance(tmp_path,out)
    assert result['phase']=='development_selection'
    assert result['source_sha256']=={'frozen.py':'source-hash'}
    assert result['formal_eligible'] is False


def test_ambiguous_queue_outputs_are_rejected(tmp_path):
    out,provenance=measurement(tmp_path)
    write(tmp_path/'runs'/'duplicate'/'queue-task.json',provenance)
    with pytest.raises(ValueError,match='Ambiguous'):
        report.queue_provenance(tmp_path,out)


@pytest.mark.parametrize('frozen,expected',[(False,False),(True,True)])
def test_formal_evidence_requires_matching_freeze(tmp_path,frozen,expected):
    out,_=measurement(tmp_path,phase='formal_main',frozen=frozen)
    result=report.queue_provenance(tmp_path,out)
    assert result['formal_eligible'] is expected
    if frozen:
        (tmp_path/'freeze.json').write_text('{}')
        assert report.queue_provenance(tmp_path,out)['formal_eligible'] is False


def test_unfrozen_formal_measurement_is_preserved_but_not_aggregated(tmp_path):
    measurement(tmp_path,phase='formal_main')
    report.summarize(tmp_path)
    rows=(tmp_path/'report/measurements.csv').read_text()
    assert 'formal_main' in rows
    groups=json.loads((tmp_path/'report/aggregates.json').read_text())
    assert len(groups)==1 and not groups[0]['formal_repeated_comparison_ready']
    assert groups[0]['goodput_req_s']['mean'] is None


def test_selection_and_development_are_separate_aggregate_groups(tmp_path):
    measurement(tmp_path,'dev',phase='development')
    measurement(tmp_path,'selected',phase='development_selection')
    report.summarize(tmp_path)
    groups=json.loads((tmp_path/'report/aggregates.json').read_text())
    assert {row['phase'] for row in groups}=={'development','development_selection'}
    assert {row['phase_group'] for row in groups}=={'development','selection'}
    assert all(not row['formal_eligible'] for row in groups)


def test_censored_repeats_are_counted_without_averaging_only_finite_observations(tmp_path):
    for label,value in [('one',1),('two',None),('three','>cutoff')]:
        out,_=measurement(tmp_path,label,phase='development')
        path=out/'trace_summary.json'
        summary=json.loads(path.read_text())
        summary['arrival_metrics']['latencies_sec']['ttft']['p95']=value
        write(path,summary)
    report.summarize(tmp_path)
    groups=json.loads((tmp_path/'report/aggregates.json').read_text())
    assert len(groups)==1
    result=groups[0]['ttft_p95_s']
    assert result['mean'] is None
    assert result['status']=='censored'
    assert result['censored_count']==2
    assert result['finite_count']==1
    assert set(str(v) for v in result['raw'])=={'1','None','>cutoff'}
    assert groups[0]['slo_arrival']['mean']==1

def test_effective_config_hash_changes_keep_measurements_in_separate_groups(tmp_path):
    for label,digest in [('one','config1'),('two','config2')]:
        out,_=measurement(tmp_path,label,phase='development')
        path=out/'run.json';metadata=json.loads(path.read_text())
        metadata['configuration_sha256']={'worker0.pdmux':digest}
        write(path,metadata)
    report.summarize(tmp_path)
    groups=json.loads((tmp_path/'report/aggregates.json').read_text())
    assert len(groups)==2


def audit_fixture(capacities=((100,100),(200,200))):
    workers=[]
    for index,values in enumerate(capacities):
        workers.append(dict(index=index,status='passed',ranks=[
            dict(pp=0,tp=rank,capacity=value,request_capacity=128,
                 cuda_memory_peak_bytes={'max_allocated':1000+index*100+rank,'max_reserved':2000+index*100+rank},
                 graph_capture_gb=.25+rank*.1)
            for rank,value in enumerate(values)]))
    return dict(schema_version=2,status='passed',workers=workers,gpu_memory_peak_mib={'gpu0':10,'gpu1':20,'gpu2':30,'gpu3':40})


def resource_metadata(family='chunked'):
    return dict(args=dict(tp=2,pp=1,replicas=2,family=family),workers=[
        dict(index=0,role='prefill' if family=='pd' else 'model',gpus=['gpu0','gpu1']),
        dict(index=1,role='decode' if family=='pd' else 'model',gpus=['gpu2','gpu3'])])


def test_logical_kv_sums_instances_once_and_preserves_rank_memory():
    result=report.resource_summary(resource_metadata(),audit_fixture())
    assert result['logical_kv_capacity_tokens']==300
    assert result['rank_capacities_consistent'] is True
    assert [row['logical_kv_capacity_tokens'] for row in result['instances']]==[100,200]
    assert len(result['ranks'])==4
    assert result['ranks'][3]['peak_device_memory_mib']==40
    assert result['ranks'][3]['peak_torch_reserved_bytes']==2101
    assert result['ranks'][3]['graph_capture_gib']==.35
    assert 'divides bytes by 2**30' in result['scope']


@pytest.mark.parametrize('capacities,consistent',[(((100,99),(200,200)),False)],ids=['unequal-rank-capacities'])
def test_inconsistent_rank_capacity_is_unknown_not_summed(capacities,consistent):
    result=report.resource_summary(resource_metadata(),audit_fixture(capacities))
    assert result['logical_kv_capacity_tokens'] is None
    assert result['rank_capacities_consistent'] is consistent


def test_missing_resource_counts_remain_null_not_zero():
    result=report.resource_summary(resource_metadata(),{'status':'incomplete','workers':[]})
    assert result['logical_kv_capacity_tokens'] is None
    assert result['rank_capacities_consistent'] is None
    assert len(result['ranks'])==4
    assert all(row['peak_device_memory_mib'] is None and row['kv_capacity_tokens'] is None for row in result['ranks'])


def test_pd_cache_global_unknown_and_decode_reported_value_separate(tmp_path):
    out,_=measurement(tmp_path)
    path=out/'run.json';meta=json.loads(path.read_text());meta['args']['family']='pd';write(path,meta)
    path=out/'trace_summary.json';summary=json.loads(path.read_text());summary['cached_tokens']['token_weighted_ratio']=.7;write(path,summary)
    report.summarize(tmp_path)
    row=next(csv.DictReader((tmp_path/'report/measurements.csv').open()))
    assert row['cache_ratio']==''
    assert float(row['cache_ratio_reported_decode'])==.7
    assert row['cache_ratio_scope']=='global_unknown_pd_prefill_counters_unavailable'


def test_missing_pd_cache_counters_are_not_fabricated_as_zero(tmp_path):
    out,_=measurement(tmp_path)
    path=out/'run.json';meta=json.loads(path.read_text());meta['args']['family']='pd';write(path,meta)
    path=out/'trace_summary.json';summary=json.loads(path.read_text());summary.pop('cached_tokens');write(path,summary)
    report.summarize(tmp_path)
    row=next(csv.DictReader((tmp_path/'report/measurements.csv').open()))
    assert row['cache_ratio']==row['cache_ratio_reported_decode']==''


def test_all_latency_percentiles_and_existing_slos_are_reported(tmp_path):
    out,_=measurement(tmp_path)
    path=out/'trace_summary.json';summary=json.loads(path.read_text())
    for name in ('ttft','tpot','e2e'):
        for percentile,value in [(50,1),(90,2),(95,3),(99,'>cutoff')]:
            summary['latencies_sec'][name]['p'+str(percentile)]=value
            summary['arrival_metrics']['latencies_sec'][name]['p'+str(percentile)]=value if percentile==99 else value+10
    summary['slo']['ttft_le_1s']={'rate':.2,'successful':1,'denominator':5}
    summary['arrival_metrics']['slo']['ttft_le_1s']={'rate':.1,'successful':1,'denominator':10}
    write(path,summary)
    report.summarize(tmp_path)
    row=next(csv.DictReader((tmp_path/'report/measurements.csv').open()))
    for name in ('ttft','tpot','e2e'):
        assert float(row[f'send_{name}_p50_s'])==1
        assert float(row[f'arrival_{name}_p90_s'])==12
        assert row[f'send_{name}_p99_s']=='>cutoff'
    assert float(row['slo_send_ttft_le_1s'])==.2
    assert float(row['slo_arrival_ttft_le_1s'])==.1
    group=json.loads((tmp_path/'report/aggregates.json').read_text())[0]
    assert group['send_ttft_p99_s']['mean'] is None
    assert group['send_ttft_p99_s']['censored_count']==1


def test_same_rate_different_trace_hashes_never_aggregate_together(tmp_path):
    first,_=measurement(tmp_path,'one')
    second,provenance=measurement(tmp_path,'two')
    trace=tmp_path/'other_toolagent.jsonl';trace.write_text('{"different":true}\n')
    digest=hashlib.sha256(trace.read_bytes()).hexdigest()
    path=second/'run.json';meta=json.loads(path.read_text());meta['args']['trace']=str(trace);meta['trace_sha256']=digest;write(path,meta)
    path=second/'trace_summary.json';summary=json.loads(path.read_text());summary['trace']=str(trace);write(path,summary)
    provenance['evidence']['trace_sha256']=digest
    provenance['task']['argv']=['--out',str(second),'--trace',str(trace)]
    write(tmp_path/'runs/two/queue-task.json',provenance)
    report.summarize(tmp_path)
    assert len(json.loads((tmp_path/'report/aggregates.json').read_text()))==2


def test_formal_three_repeats_require_same_model_and_completed_controller(tmp_path):
    from test_formal_attempts import measured
    records=[];frozen_tasks={}
    for index,label in enumerate(['first','second','third']):
        out,provenance=measurement(tmp_path,label,phase='formal_main')
        measured(out)
        path=out/'run.json';meta=json.loads(path.read_text())
        meta.update(start_unix=index+1,model_revision='same-revision',model_provenance={'config.json':'same-model-hash'})
        write(path,meta)
        provenance['evidence']['source_sha256']=meta['source_sha256']
        frozen_tasks[label]={**provenance['task'],'trace_sha256':meta['trace_sha256']}
        records.append((out,provenance))
        write(tmp_path/'runs'/label/'outcome.json',dict(status='completed',returncode=0))
    freeze=tmp_path/'freeze-all.json'
    write(freeze,dict(status='frozen',source_sha256={'runtime.py':'runtime-hash'},tasks=frozen_tasks))
    for out,provenance in records:
        provenance['evidence'].update(freeze_path=str(freeze),freeze_sha256=hashlib.sha256(freeze.read_bytes()).hexdigest())
        write(tmp_path/'runs'/out.name/'queue-task.json',provenance)
    platform={'status':'verified','software':{'libs':'same'},'configuration':{'mooncake_transport':'default'}}
    with mock.patch.object(report.resource_audit,'audit_run',return_value=audit_fixture(((100,100,100,100),))), mock.patch.object(report,'read_platform_provenance',return_value=platform) as platform_reader:
        report.summarize(tmp_path)
        groups=json.loads((tmp_path/'report/aggregates.json').read_text())
        assert len(groups)==1 and groups[0]['formal_repeated_comparison_ready'] is True
        write(tmp_path/'runs/third/outcome.json',dict(status='failed',returncode=0,cleanup_error='holder failed'))
        report.summarize(tmp_path)
        groups=json.loads((tmp_path/'report/aggregates.json').read_text())
        assert len(groups)==1 and groups[0]['measurement_count']==3 and groups[0]['valid_measurement_count']==2
        assert groups[0]['goodput_req_s']['mean'] is None
        assert groups[0]['formal_repeated_comparison_ready'] is False
        write(tmp_path/'runs/third/outcome.json',dict(status='completed',returncode=0))
        path=records[-1][0]/'run.json';meta=json.loads(path.read_text());meta['model_revision']='other-revision';write(path,meta)
        report.summarize(tmp_path)
        groups=json.loads((tmp_path/'report/aggregates.json').read_text())
        assert len(groups)==2 and all(not group['formal_repeated_comparison_ready'] for group in groups)
        meta['model_revision']='same-revision';write(path,meta)
        (records[-1][0]/'trace_planned.jsonl').write_text('{"idx":0,"arrival":0.75,"input_len":2,"output_len":2}\n')
        report.summarize(tmp_path)
        groups=json.loads((tmp_path/'report/aggregates.json').read_text())
        assert all(not group['formal_repeated_comparison_ready'] for group in groups)
        assert all(group['goodput_req_s']['mean'] is None for group in groups)
        platform_reader.return_value={'status':'missing','software':None,'configuration':{'mooncake_transport':'unrecorded'}}
        report.summarize(tmp_path)
        assert all(not group['formal_repeated_comparison_ready'] for group in json.loads((tmp_path/'report/aggregates.json').read_text()))


def test_equal_rate_and_trace_different_arrival_intent_never_share_aggregate(tmp_path):
    measurement(tmp_path,'first')
    out,provenance=measurement(tmp_path,'second')
    path=out/'run.json';meta=json.loads(path.read_text());meta['args'].update(time_scale=.5,arrival_window_sec=1);write(path,meta)
    path=out/'trace_summary.json';summary=json.loads(path.read_text());summary['time_scale']=.5;write(path,summary)
    (out/'trace_planned.jsonl').write_text('{"idx":0,"arrival":0.25,"input_len":2,"output_len":2}\n')
    provenance['task']['argv'] += ['--time-scale','.5','--arrival-window-sec','1']
    write(tmp_path/'runs/second/queue-task.json',provenance)
    report.summarize(tmp_path)
    groups=json.loads((tmp_path/'report/aggregates.json').read_text())
    assert len(groups)==2
    identities=[json.loads(group['source_identity']) for group in groups]
    assert len({row['planned_intent_sha256'] for row in identities})==2


@pytest.mark.parametrize('change',['arrival','idx','input_len','output_len','count','scale','window'])
def test_planned_intent_requires_exact_trace_scale_count_and_queue_window(tmp_path,change):
    out,provenance=measurement(tmp_path)
    summary=json.loads((out/'trace_summary.json').read_text())
    meta=json.loads((out/'run.json').read_text())
    planned=json.loads((out/'trace_planned.jsonl').read_text())
    if change in planned:planned[change]+=1
    elif change=='count':summary['planned_requests']=2
    elif change=='scale':summary['time_scale']=.5
    else:summary['arrival_window_sec']=2
    (out/'trace_planned.jsonl').write_text(json.dumps(planned)+'\n')
    result=report.planned_intent_summary(out,meta,summary,report.queue_provenance(tmp_path,out))
    assert result['status']=='failed' and result['errors']


def test_valid_planned_intent_and_explicit_window_are_verified(tmp_path):
    out,provenance=measurement(tmp_path)
    provenance['task']['argv'] += ['--arrival-window-sec','2']
    write(tmp_path/'runs/trial/queue-task.json',provenance)
    summary=json.loads((out/'trace_summary.json').read_text());summary['arrival_window_sec']=2
    meta=json.loads((out/'run.json').read_text());meta['args']['arrival_window_sec']=2
    result=report.planned_intent_summary(out,meta,summary,report.queue_provenance(tmp_path,out))
    assert result['status']=='verified' and len(result['sha256'])==64


def test_pd_transports_are_different_configurations_with_identical_platform_software(tmp_path):
    for label in ['default','nvlink']:
        out,_=measurement(tmp_path,label)
        path=out/'run.json';meta=json.loads(path.read_text());meta['args']['family']='pd';write(path,meta)
    def platform(root,provenance):
        return {'status':'verified','software':{'libs':'same'},'configuration':{'mooncake_transport':provenance['task']['id']}}
    with mock.patch.object(report,'read_platform_provenance',side_effect=platform):report.summarize(tmp_path)
    groups=json.loads((tmp_path/'report/aggregates.json').read_text())
    assert len(groups)==2 and len({group['source_identity'] for group in groups})==1
    assert {json.loads(group['configuration'])['platform_transport']['mooncake_transport'] for group in groups}=={'default','nvlink'}
