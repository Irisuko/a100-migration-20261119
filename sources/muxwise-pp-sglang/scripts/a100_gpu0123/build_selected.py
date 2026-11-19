#!/usr/bin/env python3
"""Emit finite queues from reviewed selections; never select, launch or retry.

Inputs are explicit reviewed configuration/rate manifests and audited trace
records. All performance inputs to selection are development-only. Existing
selector, full-preflight and formal-attempt checks remain authoritative.
"""
import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tarfile
import tempfile

import build_development as development
import formal_attempts
import select_development as selector
import serving_compatibility

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
ROOT=REPO/'run/a100-gpu0123-20261114'
sha=formal_attempts.sha
digest=formal_attempts.digest
require=formal_attempts.require
read=development.read
option=formal_attempts.option
LOAD_OPTIONS={'--out','--port','--startup-timeout','--suite','--trace','--time-scale','--arrival-window-sec'}


def absolute(path,repo=REPO):
    """Keep the original symlink name for byte-archive provenance."""
    path=Path(path)
    return Path(os.path.abspath(path if path.is_absolute() else Path(repo)/path))


def pin(mapping,path,expected=None,repo=REPO):
    path=absolute(path,repo); actual=sha(path)
    require(expected is None or actual==expected,f'Pinned hash changed: {path}')
    require(str(path) not in mapping or mapping[str(path)]==actual,f'Conflicting pin: {path}')
    mapping[str(path)]=actual
    return path


def serving_argv(argv):
    """Remove only replay/lifecycle fields; preserve exact serving tokens."""
    result=[];i=0
    while i<len(argv):
        token=argv[i];name=token.split('=',1)[0]
        require(name!='--limit','Selected configurations must use the full trace, not --limit')
        if token=='--extra':
            require(not any(x.split('=',1)[0] in LOAD_OPTIONS|{'--limit'} for x in argv[i+1:]),
                    'Replay fields cannot be hidden in server extras')
            result+=argv[i:];break
        if name in LOAD_OPTIONS:
            option(argv,name); i+=1 if '=' in token else 2
        elif name=='--stress200':i+=1
        else:result.append(token);i+=1
    return result


def platform_pins(root,task_dir,pins,repo):
    provenance=read(task_dir/'queue-task.json')
    verified=selector.read_platform_provenance(root,provenance)
    require(verified['status']=='verified','Selected and full-gate deployments require verified platform provenance')
    path=pin(pins,root/'queues'/provenance['queue_id']/'platform-launch.json',repo=repo)
    platform=read(path)['platform']
    pin(pins,platform['environment_path'],platform['environment_sha256'],repo)
    for key in ('package-sources.json','packages-manifest.json','verification.json'):
        pin(pins,platform[key]['path'],platform[key]['sha256'],repo)
    for path,value in platform['packages_sha256'].items():pin(pins,path,value,repo)
    for path,value in platform['installed_libraries'].items():pin(pins,path,value['sha256'],repo)
    return verified


def numerical_disposition(review,candidate,pins,repo):
    """A scoped performance permission is never renamed numerical equivalence."""
    value=review.get('numerical',{})
    require(value.get('disposition') in ('strict_pass','performance_only_non_bitwise'),
            'An explicit scoped numerical disposition is required')
    topology_keys=('model','family','tp','pp','replicas')+(('prefill_replicas',) if candidate['family']=='pd' else ())
    require(value.get('topology')=={k:candidate[k] for k in topology_keys},
            'Numerical evidence scope differs from the selected model/topology')
    require(isinstance(value.get('limitations'),list) and value['limitations']
            and all(isinstance(x,str) and x.strip() for x in value['limitations'])
            and value.get('evidence_sha256'),'Numerical disposition requires evidence and limitations')
    if value['disposition']=='strict_pass':
        require(value.get('strict_token_equality') is True,'Strict numerical PASS requires explicit token equality')
    else:
        require(value.get('strict_token_equality') is False
                and value.get('allowed_claims')==['fixed_input_fixed_length_performance']
                and value.get('prohibited_claims')==['quality','numerical_equivalence'],
                'Non-bitwise disposition permits only scoped performance, never quality or equivalence')
    for name,expected in value['evidence_sha256'].items():pin(pins,name,expected,repo)
    return value


def selected_records(path,root,repo,source):
    """Validate explicit choices without reranking or consulting formal results."""
    data=read(path);pins={};pin(pins,path,repo=repo)
    require(data.get('schema_version')==1 and data.get('status')=='reviewed_selected_configurations',
            'An explicit reviewed selected-configuration manifest is required')
    require(data.get('source_sha256')==source,'Selected serving source differs from current source')
    records=[];cache={};ids=set()
    for record in data.get('records',[]):
        name=record['id'];require(re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,45}',name) and name not in ids,'Duplicate/invalid selection ID');ids.add(name)
        task_dir=absolute(record['selected_task_dir'],repo)
        require(root/'runs'==task_dir.parent,'Selected task must belong to this campaign')
        measured=selector.read_measurement(task_dir,cache)
        require(measured['phase']=='development' and measured['selection_eligible'],
                'Choice must identify a complete eligible development measurement')
        require(measured['source_sha256']==source,'Selected measurement uses different serving source')
        platform=platform_pins(root,task_dir,pins,repo)
        candidate=record['candidate']
        require(candidate['model']==measured['model'] and candidate['family']==measured['system']
                and record['workload']==measured['workload'],'Selected candidate identity differs from measured choice')
        run=absolute(record['full_gate_run'],repo);label=record['full_gate_label']
        require(root in run.parents and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,95}',label),
                'Full preflight must have a campaign-local run and label')
        evidence,passed=development.eligibility(candidate,{candidate['id']:(label,run)},root,repo,source)
        require(passed is not None,f'Exact-source full preflight is ineligible: {evidence["exclusion_reasons"]}')
        metadata=passed['metadata'];args=metadata['args']
        require(args.get('suite')=='correctness' and args.get('stress200') is True,'Every selected configuration requires full correctness plus stress200')
        command=read(root/'runs'/label/'command.json')['argv'][5:]
        require(serving_argv(command)==serving_argv(measured['argv']),'Full-gate serving parameters differ from selected measurement')
        require(metadata['configuration_sha256']==measured['configuration_sha256']
                and metadata['model_revision']==measured['model_revision']
                and metadata['model_provenance']==measured['model_provenance'],
                'Full-gate model/configuration bytes differ from selected measurement')
        gate_platform=platform_pins(root,root/'runs'/label,pins,repo)
        require(gate_platform['software']==platform['software'] and gate_platform['configuration']==platform['configuration'],
                'Full-gate platform differs from selected measurement')
        effective=metadata.get('overlap_policy',{}).get('effective')
        require(effective in ('on','off'),'Missing full-gate overlap policy')
        evidence['evidence_sha256'].update(serving_compatibility.check_actual_overlap(run,metadata,effective))
        review_path=pin(pins,record['gate_review']['path'],record['gate_review']['sha256'],repo);review=read(review_path)
        require(review.get('status')=='REVIEWED' and review.get('configuration_fingerprint')==measured['configuration_fingerprint']
                and review.get('source_fingerprint')==measured['source_fingerprint']
                and review.get('model_fingerprint')==measured['model_fingerprint']
                and review.get('full_gate_run')==str(run)
                and review.get('checks')=={'communication':True,'full_preflight':True},
                'Explicit numerical/communication/full-gate review must match this exact choice')
        numerical=numerical_disposition(review,candidate,pins,repo)
        require(review.get('evidence_phases') and set(review['evidence_phases'])<={'development','correctness','communication','numerical_audit'},
                'Formal results cannot be selection/gate inputs')
        require(review.get('evidence_sha256'),'Gate review must reference concrete evidence')
        for name,expected in {**review['evidence_sha256'],**evidence['evidence_sha256']}.items():pin(pins,name,expected,repo)
        measured_dir=Path(measured['measurement_directory'])
        for file in [task_dir/'queue-task.json',task_dir/'outcome.json',*[measured_dir/name for name in ('run.json','trace_summary.json','trace_requests.jsonl','trace_planned.jsonl')]]:pin(pins,file,repo=repo)
        for file in passed['config_paths'].values():pin(pins,file,repo=repo)
        for name,expected in metadata['model_provenance'].items():pin(pins,repo.parent/'models'/development.MODELS[candidate['model']]/name,expected,repo)
        argv=serving_argv(measured['argv'])
        if '--extra' in argv:
            config=option(argv[argv.index('--extra')+1:],'--pdmux-config-path')
            if config:pin(pins,config,repo=repo)
        records.append(dict(id=record['id'],candidate_id=candidate['id'],model=candidate['model'],family=candidate['family'],
                            workload=record['workload'],argv=argv,transport=platform['configuration']['mooncake_transport'],
                            evidence_sha256=dict(pins),configuration_fingerprint=measured['configuration_fingerprint'],
                            model_fingerprint=measured['model_fingerprint'],platform_software=platform['software'],numerical_disposition=numerical))
    require(records,'No reviewed selected configurations')
    return records,pins


def audited_trace(manifest,record,rate,stage,pins,repo,cache):
    for key,digest_key in (('path','sha256'),('stats_path','stats_sha256')):pin(pins,record[key],record[digest_key],repo)
    if 'raw_index_audit_path' in record:pin(pins,record['raw_index_audit_path'],record['raw_index_audit_sha256'],repo)
    path=absolute(record['path'],repo);rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    stats=read(absolute(record['stats_path'],repo));start=record['source_start_sec'];end=record['source_end_exclusive_sec'];stretch=record['stretch']
    require(record.get('all_rows_and_hash_blocks_validated') is True and record.get('length_clipping') is False
            and record.get('padding_tokens')==0,'Complete unmodified trace audit is required')
    require(all(type(v) in (int,float) and math.isfinite(v) for v in (start,end,stretch)) and 0<=start<end and stretch>0,
            'Invalid raw trace window')
    require(stats.get('requests')==record['requests']==len(rows)>0 and stats.get('start_sec')==start
            and stats.get('source_duration_sec')==end-start and stats.get('stretch')==stretch
            and stats.get('duration_sec')==record['base_arrival_window_sec']==(end-start)*stretch,
            'Audited trace count/window differs from stats')
    if stage=='rate_calibration':
        require(record['purpose']=='dev' and start==1200 and end==1440 and record['base_arrival_window_sec']==600,
                'Rate calibration requires the full 600-second development pool')
    else:
        require(record['purpose'] in ('formal','heldout','formal_frozen_rate') and (end<=1200 or start>=1440),
                'Formal trace overlaps the development window')
        if record['purpose']=='formal_frozen_rate':
            require(record.get('rate_record_sha256',record.get('freeze_record_sha256'))==digest(rate),
                    'Formal trace is bound to a different reviewed rate record')
    sources=[s for s in manifest['sources'] if s['workload']==record['workload']]
    require(len(sources)==1,'Unique raw workload source required')
    source=sources[0];rawpath=pin(pins,source['path'],source['sha256'],repo)
    if str(rawpath) not in cache:cache[str(rawpath)]=[(i,json.loads(line)) for i,line in enumerate(rawpath.read_text().splitlines()) if line.strip()]
    expected=sorted([(i,r) for i,r in cache[str(rawpath)] if start*1000<=r['timestamp']<end*1000
                     and r['input_length']+r['output_length']<=32768],key=lambda x:(x[1]['timestamp'],x[0]))
    indices=[r['idx'] for r in rows]
    require(indices==[i for i,_ in expected] and len(set(indices))==len(indices) and digest(indices)==record['source_indices_sha256'],
            'Trace does not preserve the complete eligible raw window')
    for row,(_,raw) in zip(rows,expected):
        require(row['input_len']==raw['input_length'] and row['output_len']==raw['output_length']
                and row['arrival']==(raw['timestamp']-start*1000)/1000*stretch,
                'Trace arrival/length differs from its raw source')
    exact_window=Fraction(len(rows))/Fraction(str(rate['absolute_requests_per_sec']))
    exact_span=(Fraction(str(expected[-1][1]['timestamp']))-Fraction(str(expected[0][1]['timestamp']))) / ((Fraction(str(end))-Fraction(str(start)))*1000) * exact_window
    window=float(exact_window);scale=float(exact_window/Fraction(str(record['base_arrival_window_sec'])))
    planned=[{**{k:r[k] for k in ('idx','input_len','output_len')},'arrival':r['arrival']*scale} for r in rows]
    span=float(exact_span)
    if stage=='formal':require(exact_window>=210 and exact_span>=180 and len({r['arrival'] for r in planned})>=3,'Formal window/span is too short; no idle padding or clipping is allowed')
    for key,actual in (('time_scale',scale),('arrival_window_sec',window),('active_arrival_span_sec',span)):
        if key in record:require(math.isclose(record[key],actual,rel_tol=1e-12,abs_tol=1e-9),f'Formal trace {key} differs from reviewed rate')
    for key,actual in (('time_scale_exact',exact_window/Fraction(str(record['base_arrival_window_sec']))),
                       ('arrival_window_sec_exact',exact_window),('active_arrival_span_sec_exact',exact_span)):
        if key in record:require(Fraction(record[key])==actual,f'Formal trace exact {key} differs from reviewed rate')
    return dict(path=str(path),trace_sha256=record['sha256'],requests=len(rows),time_scale=scale,arrival_window_sec=window,
                active_arrival_span_sec=span,planned_intent_sha256=digest(planned))


def build(selection_path,rates_path,manifest_path,*,root=ROOT,repo=REPO,stage,tag):
    root=Path(root).resolve();repo=Path(repo).resolve()
    require(stage in ('rate_calibration','formal') and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,16}',tag),'Invalid stage/tag')
    source=development.current_sources(repo)
    selected,pins=selected_records(absolute(selection_path,repo),root,repo,source)
    for name,expected in source.items():pin(pins,name,expected,repo)
    for path in (repo/'scripts/a100_gpu0123').rglob('*.py'):
        if not path.name.startswith('test_') and not path.name.startswith('_compatibility_test_'):pin(pins,path,repo=repo)
    selection_path=pin(pins,selection_path,repo=repo);rates_path=pin(pins,rates_path,repo=repo);manifest_path=pin(pins,manifest_path,repo=repo)
    rates=read(rates_path);manifest=read(manifest_path)
    require(rates.get('schema_version')==1 and rates.get('status')==('frozen' if stage=='formal' else 'reviewed_rates')
            and rates.get('selection_manifest_sha256')==sha(selection_path),'Rates must be explicitly reviewed for this selected manifest')
    require(rates.get('evidence_phases') and set(rates['evidence_phases'])<={'development','rate_calibration'},
            'Rate decisions must use development evidence, never formal results')
    if stage=='formal':
        require(rates['evidence_phases']==['rate_calibration'] and rates.get('evidence_sha256'),
                'Formal rates require nonempty reviewed rate_calibration evidence')
    for path,expected in rates.get('evidence_sha256',{}).items():pin(pins,path,expected,repo)
    require(manifest.get('schema_version')==1 and manifest.get('status')=='PASS','Audited PASS trace manifest required')
    for path,expected in manifest.get('input_sha256',{}).items():pin(pins,path,expected,repo)
    if manifest.get('purpose')=='formal_frozen_rate':
        require(manifest.get('rates_manifest_sha256')==sha(rates_path)
                and manifest.get('selection_manifest_sha256')==sha(selection_path),'Formal trace manifest belongs to different frozen rates/selection')
    if manifest.get('builder'):pin(pins,manifest['builder']['path'],manifest['builder']['sha256'],repo)
    groups=defaultdict(list)
    for record in selected:
        require(record['transport'] in ('default','nvlink'),'Unknown selected transport')
        groups[record['model'],record['workload']].append(record)
    require(set(groups)=={(r['model'],r['workload']) for r in rates.get('records',[])},'Rate records must cover every selected model/workload cohort exactly')
    tasks=[];seen=set();cache={}
    for rate in rates['records']:
        model,workload=rate['model'],rate['workload'];r=rate['absolute_requests_per_sec']
        require(type(r) in (int,float) and math.isfinite(r) and r>0,'Positive finite absolute rate required')
        key=model,workload,r;require(key not in seen,'Duplicate rate cohort');seen.add(key)
        configs=groups[model,workload]
        require(len({c['model_fingerprint'] for c in configs})==1 and len({digest(c['platform_software']) for c in configs})==1,
                'Compared systems must share model revision and platform software')
        role=rate.get('role','main');count=rate.get('required_independent_repetitions',3)
        if stage=='formal':
            require(role in ('main','critical') and type(count) is int and count>=3,'Main/critical formal comparisons require at least three independent deployments')
        else:count=1
        matches=[t for t in manifest['records'] if t['model']==development.MODELS[model] and t['workload']==workload
                 and (t['purpose']=='dev' if stage=='rate_calibration' else t['purpose'] in ('formal','heldout','formal_frozen_rate'))
                 and (rate.get('trace_sha256') is None or t['sha256']==rate['trace_sha256'])
                 and (t['purpose']!='formal_frozen_rate' or t.get('rate_record_sha256',t.get('freeze_record_sha256'))==digest(rate))]
        require(len(matches)==1,'Rate record must identify one unambiguous audited trace')
        stream=audited_trace(manifest,matches[0],rate,stage,pins,repo,cache)
        for config in configs:
            for repetition in range(1,count+1):
                label=f'{"formal" if stage=="formal" else "rate"}-{config["id"]}-{digest(rate)[:8]}-r{repetition}-{tag}'
                require(len(label)<=96 and not (root/'runs'/label).exists(),'Every emitted deployment label must be fresh')
                output=root/stage/label;require(not output.exists(),'Measurement output was already used')
                base=config['argv'];cut=base.index('--extra') if '--extra' in base else len(base)
                argv=base[:cut]+['--suite','trace','--trace',stream['path'],'--time-scale',str(stream['time_scale']),
                                '--arrival-window-sec',str(stream['arrival_window_sec']),'--out',str(output)]+base[cut:]
                tasks.append(dict(id=label,phase='formal_main' if stage=='formal' else 'rate_calibration',argv=argv,
                                  timeout=math.ceil(max(7000,stream['arrival_window_sec']+2400)),dependencies=[],
                                  workload=workload,candidate_id=config.get('candidate_id'),selected_configuration_id=config['id'],
                                  configuration_fingerprint=config['configuration_fingerprint'],mooncake_transport=config['transport'],
                                  numerical_disposition=config.get('numerical_disposition'),
                                  role=role,required_independent_repetitions=count,independent_repetition=repetition,
                                  absolute_requests_per_sec=r,full_planned_requests=stream['requests'],
                                  **{k:stream[k] for k in ('trace_sha256','time_scale','arrival_window_sec','active_arrival_span_sec','planned_intent_sha256')}))
    require(tasks and len({t['id'] for t in tasks})==len(tasks),'Empty or duplicate emitted tasks')
    # Runtime records use repository-relative serving keys. Preserve those exact
    # keys so selector.check queued/runtime provenance can compare them directly.
    serving_keys={str(absolute(name,repo)):name for name in source}
    pins={serving_keys.get(name,name):value for name,value in pins.items()}
    return dict(schema_version=1,stage=stage,status='reviewed_inputs_rendered_not_launched',root=str(root),repo=str(repo),tasks=tasks,source_sha256=pins,
                selection_manifest_sha256=sha(selection_path),selected_rate_manifest_sha256=sha(rates_path),trace_manifest_sha256=sha(manifest_path),
                cutoff_policy=dict(client_drain_timeout_sec=300,controller_timeout='ceil(max(7000,arrival_window_sec+2400))',
                                   failure_denominator='All frozen planned requests; failed/missing attempts retained, unknown outcomes never fabricated as zero throughput'),
                limitations=['No automatic configuration/rate selection; no formal results read.',
                             'Existing client drains until last actual send plus 300 seconds; controller timeout is the outer bound.',
                             'Only main/critical repeats are rendered; auxiliary protocols require a separate explicit reviewed renderer.'])


def archive_content(pins,directory,repo=REPO):
    """Store dereferenced bytes; restore every unique member into a fresh directory."""
    directory=Path(directory);archive=directory/'content.tar';manifest_path=directory/'archive-manifest.json';records=[];objects={}
    if manifest_path.exists():raise FileExistsError(manifest_path)
    for name,expected in sorted(pins.items()):
        path=absolute(name,repo);require(sha(path)==expected,f'Archive source hash changed: {path}')
        record=dict(pin_key=name,path=str(path),sha256=expected,size=path.stat().st_size,mode=path.stat().st_mode&0o777,
                    symlink_target=os.readlink(path) if path.is_symlink() else None,member=f'objects/{expected}')
        records.append(record);objects.setdefault(expected,(path,record))
    with tarfile.open(archive,'x') as stream:
        for expected,(path,record) in sorted(objects.items()):
            member=tarfile.TarInfo(record['member']);member.size=record['size'];member.mode=record['mode'];member.mtime=0
            with path.open('rb') as source:stream.addfile(member,source)
    with tempfile.TemporaryDirectory(prefix='selected-restore-') as temporary,tarfile.open(archive,'r') as stream:
        members=stream.getmembers();require(len(members)==len(objects),'Duplicate archive members')
        for member in members:
            expected=member.name.removeprefix('objects/')
            require(expected in objects and member.name==f'objects/{expected}' and member.isfile(),'Unexpected archive member')
            restored=Path(temporary)/expected
            with stream.extractfile(member) as src,restored.open('xb') as dst:shutil.copyfileobj(src,dst)
            require(sha(restored)==expected and restored.stat().st_size==objects[expected][1]['size'],'Archive restore verification failed')
    manifest=dict(schema_version=1,archive_sha256=sha(archive),records=records,restore_verified=True,
                  scope='Exact bytes of every payload pin. Archive and its manifest are added afterward as control pins to avoid self-reference; unpinned installed binaries are not claimed archived.')
    with manifest_path.open('x') as stream:json.dump(manifest,stream,indent=2,allow_nan=False);stream.write('\n')
    return dict(archive_path=str(archive),archive_sha256=sha(archive),manifest_path=str(manifest_path),manifest_sha256=sha(manifest_path),restore_verified=True)


def make_freeze(result,pins,archive):
    return dict(schema_version=1,status='frozen',source_sha256=pins,tasks={t['id']:t for t in result['tasks']},
                selected_configuration_manifest_sha256=result['selection_manifest_sha256'],selected_rate_manifest_sha256=result['selected_rate_manifest_sha256'],
                formal_trace_manifest_sha256=result['trace_manifest_sha256'],source_archive_manifest_sha256=archive.get('manifest_sha256'),
                source_archive_sha256=archive.get('archive_sha256'),cutoff_policy=result['cutoff_policy'],cutoff_policy_sha256=digest(result['cutoff_policy']))


def emit(result,directory):
    directory=Path(directory).resolve();root=Path(result['root'])
    require(root in directory.parents,'Artifacts must stay inside this campaign')
    directory.mkdir(parents=True,exist_ok=False)
    archive=archive_content(result['source_sha256'],directory,result['repo']);pins=dict(result['source_sha256'])
    pin(pins,archive['archive_path'],archive['archive_sha256']);pin(pins,archive['manifest_path'],archive['manifest_sha256'])
    def write(name,value):
        with (directory/name).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    for transport in sorted({t['mooncake_transport'] for t in result['tasks']}):
        write(f'tasks-{transport}.json',dict(schema_version=1,source_sha256=pins,tasks=[t for t in result['tasks'] if t['mooncake_transport']==transport],
              required_launcher='run_campaign.py',required_mooncake_transport=transport))
    if result['stage']=='formal':write('freeze.json',make_freeze(result,pins,archive))
    write('protocol.json',{**{k:v for k,v in result.items() if k!='tasks'},'source_sha256':pins,'content_archive':archive,
                           'task_labels':[t['id'] for t in result['tasks']],'launch_performed':False})
    return directory


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ('selection','rates','trace-manifest','out-dir'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--stage',choices=['rate_calibration','formal'],required=True)
    parser.add_argument('--tag',required=True);args=parser.parse_args()
    result=build(args.selection,args.rates,args.trace_manifest,root=args.root,stage=args.stage,tag=args.tag)
    emit(result,args.out_dir)
    print(json.dumps(dict(status=result['status'],tasks=len(result['tasks']),out_dir=str(args.out_dir),gpu_processes_started=0)))


if __name__=='__main__':main()
