"""Frozen selections must keep configurations, streams and failed intents intact."""
import copy
import json
from pathlib import Path

import pytest

import build_selected as selected


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row)+'\n' for row in values))
    return path


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    repo=tmp_path/'repo'; root=repo/'run/campaign'; root.mkdir(parents=True)
    serving=repo/'python/sglang/runtime.py'; serving.parent.mkdir(parents=True); serving.write_text('unchanged\n')
    source={'python/sglang/runtime.py':selected.sha(serving)}
    monkeypatch.setattr(selected.development,'current_sources',lambda _:source)
    configs=[]
    for family in ('chunked','muxpp'):
        p=repo/f'{family}.yml'; p.write_text('fixed: true\n')
        configs.append(dict(id=family,model='8b',workload='toolagent',family=family,
                            argv=['--model','8b','--family',family,'--tp','1','--pp','4','--replicas','1',
                                  '--overlap','off','--extra','--pdmux-config-path',str(p),'--log-level','info'],
                            transport='default',evidence_sha256={str(p):selected.sha(p)},
                            model_fingerprint='same-model',platform_software={'same':'software'},
                            configuration_fingerprint=family))
    selection_path=write(root/'reviewed-selection.json',dict(schema_version=1,status='reviewed_selected_configurations',
                         source_sha256=source,records=[{'id':r['id']} for r in configs]))
    monkeypatch.setattr(selected,'selected_records',lambda *args:(copy.deepcopy(configs),{str(selection_path):selected.sha(selection_path)}))
    raw=rows(root/'raw.jsonl',[dict(timestamp=1440000+i*3000,input_length=2,output_length=2) for i in range(4)])
    trace=rows(root/'trace.jsonl',[dict(idx=i,arrival=i*7.5,input_len=2,output_len=2,input_ids=[1,2]) for i in range(4)])
    stats=write(root/'trace.stats.json',dict(requests=4,start_sec=1440,source_duration_sec=12,stretch=2.5,duration_sec=30))
    rate=dict(model='8b',workload='toolagent',absolute_requests_per_sec=4/300,role='main')
    record=dict(purpose='formal_frozen_rate',model='Llama-3.1-8B-Instruct',workload='toolagent',path=str(trace),
                stats_path=str(stats),sha256=selected.sha(trace),stats_sha256=selected.sha(stats),requests=4,
                source_start_sec=1440,source_end_exclusive_sec=1452,stretch=2.5,base_arrival_window_sec=30,
                all_rows_and_hash_blocks_validated=True,length_clipping=False,padding_tokens=0,
                source_indices_sha256=selected.digest([0,1,2,3]),rate_record_sha256=selected.digest(rate))
    manifest=write(root/'trace-manifest.json',dict(schema_version=1,status='PASS',records=[record],
                       sources=[dict(workload='toolagent',path=str(raw),sha256=selected.sha(raw))]))
    rate_evidence=write(root/'rate-review.json',dict(phase='rate_calibration',review='synthetic fixture only'))
    rates=write(root/'rates.json',dict(schema_version=1,status='frozen',selection_manifest_sha256=selected.sha(selection_path),
                      evidence_phases=['rate_calibration'],records=[rate],evidence_sha256={str(rate_evidence):selected.sha(rate_evidence)}))
    return dict(repo=repo,root=root,selection=selection_path,rates=rates,manifest=manifest,configs=configs,source=source)


def build(fixture, **kwargs):
    return selected.build(fixture['selection'],fixture['rates'],fixture['manifest'],root=fixture['root'],
                          repo=fixture['repo'],stage=kwargs.pop('stage','formal'),tag='v1',**kwargs)


def test_formal_three_fresh_repeats_share_exact_stream_and_accounting_intent(fixture):
    result=build(fixture)
    tasks=result['tasks']; assert len(tasks)==6 and len({t['id'] for t in tasks})==6
    assert {t['required_independent_repetitions'] for t in tasks}=={3}
    assert {t['phase'] for t in tasks}=={'formal_main'}
    assert len({t['planned_intent_sha256'] for t in tasks})==1
    assert {t['full_planned_requests'] for t in tasks}=={4}
    assert {t['active_arrival_span_sec'] for t in tasks}=={225}
    assert all('--limit' not in t['argv'] for t in tasks)
    frozen=selected.make_freeze(result,result['source_sha256'],{})
    assert set(frozen['tasks'])=={t['id'] for t in tasks}
    assert all(frozen['tasks'][t['id']]['argv']==t['argv'] for t in tasks)


@pytest.mark.parametrize('change,match',[
    ({'absolute_requests_per_sec':1},'span|window'),
    ({'role':'main','required_independent_repetitions':1},'three'),
    ({'role':'critical','required_independent_repetitions':2},'three'),
])
def test_short_or_under_repeated_main_stream_rejected(fixture,change,match):
    rates=json.loads(fixture['rates'].read_text()); rates['records'][0].update(change); write(fixture['rates'],rates)
    manifest=json.loads(fixture['manifest'].read_text()); manifest['records'][0]['rate_record_sha256']=selected.digest(rates['records'][0]); write(fixture['manifest'],manifest)
    with pytest.raises(ValueError,match=match):build(fixture)


def test_missing_cohort_or_formal_results_cannot_drive_rate_selection(fixture):
    fixture['configs'][1]['workload']='conversation'
    with pytest.raises(ValueError,match='cohort'):build(fixture)
    fixture['configs'][1]['workload']='toolagent'
    rates=json.loads(fixture['rates'].read_text()); rates['evidence_phases']=['formal_main']; write(fixture['rates'],rates)
    with pytest.raises(ValueError,match='formal|development'):build(fixture)


def test_tampered_trace_and_development_overlap_fail_closed(fixture):
    manifest=json.loads(fixture['manifest'].read_text()); record=manifest['records'][0]
    Path(record['path']).write_text('{}\n')
    with pytest.raises(ValueError,match='hash'):build(fixture)


def test_existing_formal_label_or_output_rejected(fixture):
    first=build(fixture)['tasks'][0]
    (fixture['root']/'runs'/first['id']).mkdir(parents=True)
    with pytest.raises(ValueError,match='fresh|used'):build(fixture)


def test_archive_preserves_external_symlink_bytes_and_rejects_overwrite(tmp_path):
    target=tmp_path/'outside.yml';target.write_text('actual config\n'); link=tmp_path/'linked.yml';link.symlink_to(target)
    out=tmp_path/'archive';out.mkdir()
    evidence=selected.archive_content({str(link):selected.sha(link)},out)
    manifest=json.loads(Path(evidence['manifest_path']).read_text())
    assert manifest['records'][0]['symlink_target']==str(target)
    assert manifest['records'][0]['sha256']==selected.sha(target)
    assert evidence['restore_verified'] is True
    with pytest.raises(FileExistsError):selected.archive_content({str(link):selected.sha(link)},out)


def test_archive_rejects_changed_pinned_bytes(tmp_path):
    p=tmp_path/'runtime.py';p.write_text('old'); pins={str(p):selected.sha(p)};p.write_text('new')
    with pytest.raises(ValueError,match='hash'):selected.archive_content(pins,tmp_path)


def test_immutable_emission_matches_existing_queue_and_all_missing_denominators(fixture,monkeypatch):
    result=build(fixture);out=fixture['root']/'frozen-v1';selected.emit(result,out)
    plan=json.loads((out/'tasks-default.json').read_text());freeze=json.loads((out/'freeze.json').read_text())
    assert len(plan['tasks'])==6 and set(plan['source_sha256'])==set(freeze['source_sha256'])
    assert freeze['source_archive_manifest_sha256']==selected.sha(out/'archive-manifest.json')
    from importlib.util import spec_from_file_location,module_from_spec
    spec=spec_from_file_location('queue_for_selected_test',Path(selected.__file__).with_name('queue.py')); queue=module_from_spec(spec);spec.loader.exec_module(queue)
    monkeypatch.setattr(queue,'REPO',fixture['repo'])
    queue.validate_plan(plan,fixture['root'])
    for task in plan['tasks']:queue.check_task(task,plan,out/'freeze.json')
    import formal_attempts
    for task in plan['tasks']:
        intent=formal_attempts.intent(task,freeze,{})
        account=formal_attempts.request_accounting(fixture['root']/'not-created',intent['_planned'])
        assert account['planned_requests']==4 and account['unknown_requests']==4 and account['unsent_requests'] is None
    with pytest.raises(FileExistsError):selected.emit(result,out)


def test_serving_pin_keys_remain_identical_to_runtime_metadata(fixture):
    result=build(fixture)
    for key,value in fixture['source'].items():assert result['source_sha256'][key]==value
    assert str(fixture['repo']/'python/sglang/runtime.py') not in result['source_sha256']


def test_rate_calibration_uses_full600_pool_once_per_selected_system(fixture):
    manifest=json.loads(fixture['manifest'].read_text());record=manifest['records'][0]
    record.update(purpose='dev',source_start_sec=1200,source_end_exclusive_sec=1440,base_arrival_window_sec=600)
    raw=Path(manifest['sources'][0]['path']);rows(raw,[dict(timestamp=1200000+i*60000,input_length=2,output_length=2) for i in range(4)])
    manifest['sources'][0]['sha256']=selected.sha(raw)
    trace=Path(record['path']);rows(trace,[dict(idx=i,arrival=i*150,input_len=2,output_len=2) for i in range(4)]);record['sha256']=selected.sha(trace)
    stats=Path(record['stats_path']);write(stats,dict(requests=4,start_sec=1200,source_duration_sec=240,stretch=2.5,duration_sec=600));record['stats_sha256']=selected.sha(stats)
    write(fixture['manifest'],manifest)
    rates=json.loads(fixture['rates'].read_text());rates['status']='reviewed_rates';rates['records'][0]['absolute_requests_per_sec']=4/600;write(fixture['rates'],rates)
    result=build(fixture,stage='rate_calibration')
    assert len(result['tasks'])==2 and {t['time_scale'] for t in result['tasks']}=={1}
    assert all(t['phase']=='rate_calibration' and t['full_planned_requests']==4 for t in result['tasks'])
    result_dir=fixture['root']/'rate-artifacts';selected.emit(result,result_dir);assert not (result_dir/'freeze.json').exists()
    record['source_end_exclusive_sec']=1260;record['base_arrival_window_sec']=150
    write(stats,dict(requests=4,start_sec=1200,source_duration_sec=60,stretch=2.5,duration_sec=150));record['stats_sha256']=selected.sha(stats);write(fixture['manifest'],manifest)
    with pytest.raises(ValueError,match='600-second'):build(fixture,stage='rate_calibration')


def test_serving_config_is_not_changed_by_replay_fields():
    base=['--model','70b','--mem','.92','--no-cpp','--extra','--pdmux-config-path','/fixed.yml']
    argv=base[:5]+['--suite','correctness','--stress200','--out','/old']+base[5:]
    assert selected.serving_argv(argv)==base
    with pytest.raises(ValueError,match='--limit'):selected.serving_argv(['--limit','3'])


@pytest.fixture
def chosen_gate(tmp_path,monkeypatch):
    repo=tmp_path/'repo';root=repo/'run/campaign';task=root/'runs/chosen';gate=root/'preflight/full';measured_dir=root/'development/chosen'
    source={'runtime.py':'same-source'}
    argv=['--model','8b','--family','chunked','--tp','1','--pp','4','--replicas','1','--mem','.85','--overlap','off']
    config=gate/'spatial.yml';config.parent.mkdir(parents=True);config.write_text('same-config')
    model=write(repo.parent/'models/Llama-3.1-8B-Instruct/config.json',{'model':'test'})
    measured=dict(phase='development',selection_eligible=True,source_sha256=source,model='8b',system='chunked',workload='toolagent',
                  argv=argv+['--suite','trace','--trace','/dev','--out',str(measured_dir)],configuration_sha256={'spatial':selected.sha(config)},
                  model_revision='model-v1',model_provenance={'config.json':selected.sha(model)},configuration_fingerprint='config-fingerprint',
                  source_fingerprint='source-fingerprint',model_fingerprint='model-fingerprint',measurement_directory=str(measured_dir))
    metadata=dict(args={'suite':'correctness','stress200':True},configuration_sha256=measured['configuration_sha256'],
                  model_revision=measured['model_revision'],model_provenance=measured['model_provenance'],overlap_policy={'effective':'off'})
    for file in [task/'queue-task.json',task/'outcome.json',*[measured_dir/name for name in ('run.json','trace_summary.json','trace_requests.jsonl','trace_planned.jsonl')]]:write(file,{})
    command=write(root/'runs/full/command.json',dict(argv=['taskset','-c','0-31,64-95','python','server_run.py',*argv,
                  '--suite','correctness','--stress200','--out',str(gate)]))
    evidence=write(root/'gate-evidence.json',{'phase':'correctness'})
    review=write(root/'gate-review.json',dict(status='REVIEWED',configuration_fingerprint='config-fingerprint',source_fingerprint='source-fingerprint',
                  model_fingerprint='model-fingerprint',full_gate_run=str(gate),checks={'communication':True,'full_preflight':True},
                  numerical=dict(disposition='strict_pass',strict_token_equality=True,topology=dict(model='8b',family='chunked',tp=1,pp=4,replicas=1),
                                 limitations=['Only these tested cases.'],evidence_sha256={str(evidence):selected.sha(evidence)}),
                  evidence_phases=['correctness','communication'],evidence_sha256={str(evidence):selected.sha(evidence)}))
    record=dict(id='selected-chunked',candidate={'id':'8b-chunked-pp4','model':'8b','family':'chunked','tp':1,'pp':4,'replicas':1},workload='toolagent',
                selected_task_dir=str(task),full_gate_run=str(gate),full_gate_label='full',gate_review={'path':str(review),'sha256':selected.sha(review)})
    selection=write(root/'choices.json',dict(schema_version=1,status='reviewed_selected_configurations',source_sha256=source,records=[record]))
    passed={'metadata':metadata,'config_paths':{'spatial':config}}
    eligibility={'evidence_sha256':{str(command):selected.sha(command)},'exclusion_reasons':[]}
    monkeypatch.setattr(selected.selector,'read_measurement',lambda *a:measured)
    platform={'status':'verified','software':{'same':'platform'},'configuration':{'mooncake_transport':'default'}}
    monkeypatch.setattr(selected,'platform_pins',lambda *a:platform)
    monkeypatch.setattr(selected.development,'eligibility',lambda *a:(eligibility,passed))
    monkeypatch.setattr(selected.serving_compatibility,'check_actual_overlap',lambda *a:{})
    return dict(repo=repo,root=root,selection=selection,source=source,measured=measured,metadata=metadata,command=command,review=review,
                eligibility=eligibility,record=record,passed=passed)


def choose(fixture):
    return selected.selected_records(fixture['selection'],fixture['root'],fixture['repo'],fixture['source'])


def test_explicit_selected_record_binds_review_model_configuration_and_full_gate(chosen_gate):
    records,pins=choose(chosen_gate)
    assert len(records)==1 and records[0]['id']=='selected-chunked'
    assert records[0]['transport']=='default' and str(chosen_gate['review']) in pins
    assert records[0]['argv']==selected.serving_argv(chosen_gate['measured']['argv'])


@pytest.mark.parametrize('change,match',[
    ('formal','development'),('ineligible','eligible'),('source','source'),('stress','stress200'),
    ('parameters','parameters'),('model','model/configuration'),('review','review'),('formal_review','Formal'),
])
def test_unqualified_choice_never_reaches_rate_or_freeze_generation(chosen_gate,change,match):
    f=chosen_gate
    if change=='formal':f['measured']['phase']='formal_main'
    elif change=='ineligible':f['measured']['selection_eligible']=False
    elif change=='source':f['measured']['source_sha256']={'runtime.py':'changed'}
    elif change=='stress':f['metadata']['args']['stress200']=False
    elif change=='parameters':
        value=json.loads(f['command'].read_text());value['argv'][value['argv'].index('--mem')+1]='.92';write(f['command'],value)
    elif change=='model':f['metadata']['model_revision']='different-model'
    elif change in ('review','formal_review'):
        value=json.loads(f['review'].read_text())
        if change=='review':value['configuration_fingerprint']='different-config'
        else:value['evidence_phases']=['formal_main']
        write(f['review'],value);selection=json.loads(f['selection'].read_text());selection['records'][0]['gate_review']['sha256']=selected.sha(f['review']);write(f['selection'],selection)
    with pytest.raises(ValueError,match=match):choose(f)


def test_failed_fresh_resource_gate_is_not_overridden_by_review(chosen_gate,monkeypatch):
    monkeypatch.setattr(selected.development,'eligibility',lambda *a:({'exclusion_reasons':['fresh_resource_audit_not_passed']},None))
    with pytest.raises(ValueError,match='ineligible'):choose(chosen_gate)


def test_scoped_nonbitwise_performance_permission_is_explicit_and_preserved(chosen_gate):
    f=chosen_gate;value=json.loads(f['review'].read_text());numerical=value['numerical']
    numerical.update(disposition='performance_only_non_bitwise',strict_token_equality=False,
                     allowed_claims=['fixed_input_fixed_length_performance'],prohibited_claims=['quality','numerical_equivalence'],
                     limitations=['Observed differences remain; no magnitude conclusion from another topology.'])
    write(f['review'],value);selection=json.loads(f['selection'].read_text());selection['records'][0]['gate_review']['sha256']=selected.sha(f['review']);write(f['selection'],selection)
    records,_=choose(f)
    assert records[0]['numerical_disposition']==numerical
    assert records[0]['numerical_disposition']['strict_token_equality'] is False


@pytest.mark.parametrize('field,value',[
    ('topology',dict(model='8b',family='chunked',tp=2,pp=2,replicas=1)),
    ('limitations',[]),('evidence_sha256',{}),('strict_token_equality',False),
])
def test_numerical_attestation_cannot_omit_evidence_scope_or_limits(chosen_gate,field,value):
    review=json.loads(chosen_gate['review'].read_text());review['numerical'][field]=value
    with pytest.raises(ValueError):selected.numerical_disposition(review,chosen_gate['record']['candidate'],{},chosen_gate['repo'])


def test_raw_physical_line_indices_survive_blank_lines(fixture):
    manifest=json.loads(fixture['manifest'].read_text());record=manifest['records'][0]
    raw=Path(manifest['sources'][0]['path']);lines=raw.read_text().splitlines();raw.write_text(lines[0]+'\n\n'+'\n'.join(lines[1:])+'\n')
    manifest['sources'][0]['sha256']=selected.sha(raw)
    path=Path(record['path']);values=[json.loads(line) for line in path.read_text().splitlines()]
    for row in values:
        if row['idx']>0:row['idx']+=1
    rows(path,values);record['sha256']=selected.sha(path);record['source_indices_sha256']=selected.digest([0,2,3,4]);write(fixture['manifest'],manifest)
    assert len(build(fixture)['tasks'])==6


def test_exact_span_claim_cannot_hide_a_subthreshold_stream(fixture):
    rates=json.loads(fixture['rates'].read_text());rates['records'][0]['absolute_requests_per_sec']=.01666666666666667;write(fixture['rates'],rates)
    manifest=json.loads(fixture['manifest'].read_text());record=manifest['records'][0]
    record['rate_record_sha256']=selected.digest(rates['records'][0]);record['active_arrival_span_sec_exact']='180';write(fixture['manifest'],manifest)
    with pytest.raises(ValueError,match='span'):build(fixture)
