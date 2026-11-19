#!/usr/bin/env python3
"""One bounded tokens-only MuxPP diagnostic under the ordinary controller.

This separate wrapper preserves the scored wrapper byte-for-byte. It records
the same eight cells and36 measured requests, but never requests logprobs.
Output-token coverage is distinct from unavailable logprob coverage. Only the
child's server_run.preflight callback is replaced; serving source is unchanged.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import threading
import time
from urllib.error import HTTPError

HERE=Path(__file__).resolve()
REPO=HERE.parents[2]
HELPER=HERE.with_name('actual_decode_audit.py')
HELPER_SHA256='b744a618e649813c53b9bfa2a63737704ff46756b58dea5cf9045c8b508526ea'
if hashlib.sha256(HELPER.read_bytes()).hexdigest()!=HELPER_SHA256:
    raise ValueError('Original scored diagnostic helper changed; new review required')
_spec=importlib.util.spec_from_file_location('_scoreless_pure_helpers',HELPER)
helpers=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(helpers)
require,sha,atomic_json=helpers.require,helpers.sha,helpers.atomic_json
condition_plan=helpers.condition_plan
LENGTHS,FOCUS,TOKENS=helpers.LENGTHS,helpers.FOCUS,helpers.TOKENS


def parse_token_response(body,input_ids,count):
    require(isinstance(body,dict) and not body.get('error'),'Invalid generate response')
    ids,meta=body.get('output_ids'),body.get('meta_info',{})
    require(isinstance(ids,list) and len(ids)==count
            and all(type(token) is int and 0<=token<128256 for token in ids),'Output token count/IDs invalid')
    require(meta.get('prompt_tokens')==len(input_ids) and meta.get('completion_tokens')==count,
            'Prompt/completion token coverage differs')
    require(isinstance(meta.get('finish_reason'),dict) and meta['finish_reason'].get('type')=='length',
            'Generation did not finish at requested length')
    cached=meta.get('cached_tokens')
    require(type(cached) is int and 0<=cached<=len(input_ids),'Missing or invalid actual cache count')
    require(not meta.get('output_token_logprobs') and not meta.get('output_top_logprobs'),
            'Unexpected scores in a tokens-only response; do not claim zero scored tokens')
    return dict(audited_output_tokens=count,full_output_token_coverage=True,full_logprob_coverage=False,
                actual_logprob_tokens=0,cached_tokens=cached,
                interpretation='Output token IDs and metadata only. No logprob/top5 coverage or numerical correctness verdict.')


def capture_request(runner,port,ids,count,path,label,scope,barrier=None):
    payload=dict(input_ids=ids,rid=f'actual-decode-tokens-{label}',stream=False,
                 sampling_params=dict(temperature=0,max_new_tokens=count,ignore_eos=True))
    record=dict(label=label,scope=scope,request=payload,diagnostic_mode='tokens_only',diagnostic_only=True,
                measurement_eligible=False,status='pending',full_output_token_coverage=False,
                full_logprob_coverage=False,actual_logprob_tokens=0,logprobs_requested=False)
    atomic_json(path,record)
    try:
        if barrier is not None:barrier.wait(timeout=30)
        record['sent_unix']=time.time();record['sent_monotonic']=time.monotonic()
        body=runner.request(port,'/generate',payload,timeout=600)
        record.update(raw_response=body,finished_unix=time.time(),finished_monotonic=time.monotonic(),status='response_saved')
        atomic_json(path,record)
        parsed=parse_token_response(body,ids,count)
        record.update(parsed=parsed,output_ids=body['output_ids'],status='tokens_only_diagnostic_complete',
                      full_output_token_coverage=True)
        atomic_json(path,record)
        return record
    except BaseException as error:
        if isinstance(error,HTTPError):
            record['raw_http_error']=dict(status=error.code,body=error.read().decode('utf-8',errors='replace'))
        record.update(status='tokens_only_diagnostic_incomplete',error=repr(error))
        atomic_json(path,record)
        raise


def run_preflight(runner,port,model,out):
    out=Path(out);directory=out/'actual-decode-tokens';directory.mkdir(exist_ok=False)
    summary_path=out/'actual-decode-tokens-summary.json'
    summary=dict(schema_version=1,status='running',diagnostic_mode='tokens_only',diagnostic_only=True,
        measurement_eligible=False,full_output_token_coverage=False,full_logprob_coverage=False,
        actual_logprob_tokens=0,logprobs_requested=False,expected_measured_requests=36,
        expected_measured_output_tokens=1152,measured_requests=0,measured_output_tokens=0,
        measured_completed_requests=0,expected_warmup_requests=19,expected_warmup_output_tokens=26,
        warmup_requests=0,warmup_output_tokens=0,planned_cells=condition_plan(),cells=[],errors=[],family='muxpp',
        interpretation='Autoregressive output token IDs with actual cache/send metadata. No logprobs, top5, teacher forcing, or tolerance-based correctness verdict.')
    atomic_json(summary_path,summary)
    try:
        server=runner.request(port,'/get_server_info',timeout=30)
        require(Path(server.get('model_path','')).resolve()==Path(model).resolve(),'Diagnostic endpoint model mismatch')
        require(server.get('tp_size')==1 and server.get('pp_size')==4
                and server.get('dtype') in {'bfloat16','bf16','torch.bfloat16'}
                and server.get('enable_pdmux') is True and server.get('disable_overlap_schedule') is True,
                'Expected overlap-off BF16 MuxPP TP1/PP4 endpoint')
        summary['server_before']=server
        fixtures=runner.fixture_cases(model)
        selected={length:fixtures[runner.CORRECTNESS_LENGTHS.index(length)] for length in LENGTHS}
        require(all(len(ids)==length for length,ids in selected.items()),'Fixture length mismatch')
        atomic_json(directory/'prompts.json',{str(k):v for k,v in selected.items()})
        summary['fixture_sha256']=sha(directory/'prompts.json')
        runner.flush(port)
        rng=random.Random(730031)
        long_ids=[128000]+[rng.randrange(128000) for _ in range(31662)]
        prewarm=capture_request(runner,port,long_ids,8,directory/'request-long-prewarm.json','long-prewarm','long_prewarm')
        summary.update(warmup_requests=1,warmup_output_tokens=prewarm['parsed']['audited_output_tokens'])
        runner.flush(port);runner.resource_phase(out,'initial',[port])
        for condition in condition_plan():
            folder=directory/condition['id'];folder.mkdir()
            cell={**condition,'status':'running','cache_flush_acknowledged':False,
                  'warmup_files':[],'measured_files':[],'completed_requests':0}
            summary['cells'].append(cell);atomic_json(summary_path,summary)
            runner.flush(port);cell.update(cache_flush_acknowledged=True,cache_flush_unix=time.time())
            if condition['cache']=='warm':
                for index,length in enumerate(condition['input_lengths']):
                    path=folder/f'request-warmup-{index:02d}.json'
                    warm=capture_request(runner,port,selected[length],1,path,
                                         f"{condition['id']}-warmup-{index}",'cache_warmup')
                    summary['warmup_requests']+=1
                    summary['warmup_output_tokens']+=warm['parsed']['audited_output_tokens']
                    cell['warmup_files'].append(str(path))
            ids_list=[selected[length] for length in condition['input_lengths']]
            barrier=threading.Barrier(8) if condition['mode']=='concurrent8' else None
            results,failures={},[]
            with ThreadPoolExecutor(max_workers=len(ids_list)) as pool:
                jobs={}
                for index,ids in enumerate(ids_list):
                    path=folder/f'request-measured-{index:02d}.json';cell['measured_files'].append(str(path))
                    jobs[pool.submit(capture_request,runner,port,ids,TOKENS,path,
                                     f"{condition['id']}-measured-{index}",'measured',barrier)]=index
                for future in as_completed(jobs):
                    try:
                        result=future.result();results[jobs[future]]=result
                        cell['completed_requests']+=1;summary['measured_completed_requests']+=1
                        summary['measured_output_tokens']+=result['parsed']['audited_output_tokens']
                    except BaseException as error:failures.append(repr(error))
                    summary['measured_requests']+=1;atomic_json(summary_path,summary)
            if failures:
                cell.update(status='tokens_only_diagnostic_incomplete',errors=failures)
                raise ValueError(f"Output coverage failed in {condition['id']}: {failures}")
            focus=results[condition['input_lengths'].index(FOCUS)]
            cell.update(status='tokens_only_diagnostic_complete',focus_output_ids=focus['output_ids'],
                        actual_cached_tokens={str(i):r['parsed']['cached_tokens'] for i,r in sorted(results.items())},
                        send_spread_sec=max(r['sent_monotonic'] for r in results.values())-min(r['sent_monotonic'] for r in results.values()))
            atomic_json(summary_path,summary)
        require(len(summary['cells'])==8 and summary['measured_requests']==summary['measured_completed_requests']==36
                and summary['measured_output_tokens']==1152 and summary['warmup_requests']==19
                and summary['warmup_output_tokens']==26,'Fixed tokens-only diagnostic budget incomplete')
        runner.resource_phase(out,'final',[port])
        atomic_json(out/'server-after.json',{str(port):runner.request(port,'/get_server_info',timeout=30)})
        summary.update(status='tokens_only_diagnostic_complete',full_output_token_coverage=True)
        return summary
    except BaseException as error:
        summary.update(status='tokens_only_diagnostic_incomplete',errors=[*summary['errors'],repr(error)])
        raise
    finally:
        summary['ended_unix']=time.time();atomic_json(summary_path,summary)


def make_plan(argv,out):
    plan=helpers.make_plan(argv,out)
    require(plan['model']=='70b' and plan['family']=='muxpp','This single follow-up is70B MuxPP only')
    plan.update(diagnostic_mode='tokens_only',logprobs_requested=False,full_logprob_coverage=False,
                actual_logprob_tokens=0,expected_full_output_token_coverage=True,
                interpretation='Planned output-ID coverage only; actual completion is recorded separately. No score/top5 coverage is claimed.',
                original_helper_path=str(HELPER),original_helper_sha256=HELPER_SHA256)
    return plan


def mark_diagnostic(out,plan):
    path=Path(out)/'serving/run.json'
    if not path.is_file():return
    original=path.with_name('run.before-token-diagnostic-marking.json')
    if not original.exists():
        with original.open('xb') as stream:stream.write(path.read_bytes())
    summary_path=Path(out)/'serving/worker0/actual-decode-tokens-summary.json'
    summary=json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    metadata=json.loads(path.read_text())
    metadata.update(diagnostic_only=True,measurement_eligible=False,diagnostic_kind='actual_decode_tokens_only',
        diagnostic_wrapper_sha256=plan['wrapper_sha256'],diagnostic_plan_sha256=sha(Path(out)/'plan.json'),
        diagnostic_original_metadata_sha256=sha(original),full_output_token_coverage=summary.get('full_output_token_coverage') is True,
        full_logprob_coverage=False,actual_logprob_tokens=0,logprobs_requested=False)
    atomic_json(path,metadata)


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('server_args',nargs=argparse.REMAINDER)
    args=parser.parse_args();argv=args.server_args[1:] if args.server_args[:1]==['--'] else args.server_args
    plan=make_plan(argv,args.out)
    import server_run
    server_run.require_active_controller()
    out=Path(plan['out']);out.mkdir(parents=True,exist_ok=False)
    plan.update(wrapper_path=str(HERE),wrapper_sha256=sha(HERE),source_provenance=server_run.source_provenance(REPO),
                original_preflight_source=str(HERE.with_name('server_run.py')),
                original_preflight_source_sha256=sha(HERE.with_name('server_run.py')))
    atomic_json(out/'plan.json',plan)
    original,original_argv=server_run.preflight,sys.argv
    result=dict(status='running',diagnostic_mode='tokens_only',diagnostic_only=True,measurement_eligible=False,
                full_output_token_coverage=False,full_logprob_coverage=False,actual_logprob_tokens=0,
                logprobs_requested=False,plan_sha256=sha(out/'plan.json'))
    atomic_json(out/'diagnostic.json',result)
    def replacement(port,model,worker_out,suite,transport=None,worker_ports=None,stress200=False):
        require(suite=='smoke' and transport is None and not stress200 and (worker_ports is None or worker_ports==[port]),
                'Unexpected tokens-only preflight scope')
        return run_preflight(server_run,port,model,worker_out)
    server_run.preflight=replacement;sys.argv=[str(HERE.with_name('server_run.py')),*plan['server_args']]
    try:
        server_run.main()
        summary=json.loads((out/'serving/worker0/actual-decode-tokens-summary.json').read_text())
        require(summary['full_output_token_coverage'] is True and summary['measured_requests']==36
                and summary['actual_logprob_tokens']==0 and summary['full_logprob_coverage'] is False,
                'Missing completed tokens-only diagnostic')
        result.update(status='tokens_only_diagnostic_complete',full_output_token_coverage=True,
                      measured_requests=36,measured_output_tokens=1152)
    except BaseException as error:
        result.update(status='tokens_only_diagnostic_incomplete',error=repr(error));raise
    finally:
        server_run.preflight,sys.argv=original,original_argv
        mark_diagnostic(out,plan);result['ended_unix']=time.time();atomic_json(out/'diagnostic.json',result)


if __name__=='__main__':main()
