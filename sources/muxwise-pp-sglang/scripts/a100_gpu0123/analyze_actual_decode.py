#!/usr/bin/env python3
"""CPU-only integrity and first-divergence analysis of two actual-decode runs.

Reads completed diagnostic artifacts only. It neither contacts endpoints nor
imports the serving runtime. Numeric descriptions never change strict matches.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('_actual_decode_reader', HERE / 'actual_decode_audit.py')
wrapper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wrapper)
require = wrapper.require
sha = wrapper.sha


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def describe_token(token, other_choice):
    scores = dict(zip(token['top5_ids'], token['top5_logprobs']))
    other = scores.get(other_choice)
    margin = token['top_two_margin']
    # These labels are descriptions, never acceptance tolerances. Raw values
    # and strict output-token comparisons are retained independently.
    description = ('exact_tie_in_reported_logprobs' if margin == 0 else
                   'near_0.125_nat_reported_margin' if abs(margin - .125) <= 1e-6 else
                   'reported_margin_above_0.5_nat' if margin > .5 else 'other_reported_margin')
    return {**token, 'other_choice': other_choice, 'other_choice_in_top5': other is not None,
            'other_choice_logprob': other,
            'chosen_minus_other': None if other is None else token['chosen_logprob'] - other,
            'other_choice_regret_lower_bound': (token['reported_top1_logprob'] - token['top5_logprobs'][-1]
                                                if other is None else token['reported_top1_logprob'] - other),
            'other_choice_bound_only': other is None, 'margin_description': description,
            'chosen_logprob_matches_same_token_top5': (None if token['chosen'] not in scores else
                                                       token['chosen_logprob'] == scores[token['chosen']])}


def compare_records(left, right):
    require(left['request']['input_ids'] == right['request']['input_ids'], 'Cannot compare unequal prompt token IDs')
    a, b = left['output_ids'], right['output_ids']
    require(len(a) == len(b), 'Cannot compare unequal output budgets')
    position = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    first = None
    if position is not None:
        first = dict(position=position, common_output_prefix=a[:position],
                     actual_prefix_sha256=digest([*left['request']['input_ids'], *a[:position]]),
                     left=describe_token(left['parsed']['tokens'][position], b[position]),
                     right=describe_token(right['parsed']['tokens'][position], a[position]))
    return dict(exact_match=position is None, output_tokens_compared=len(a),
                common_output_prefix_tokens=len(a) if position is None else position,
                first_divergence=first, prompt_tokens=len(left['request']['input_ids']),
                prompt_ids_sha256=digest(left['request']['input_ids']),
                left=dict(path=left['path'], sha256=left['sha256'], output_ids=a,
                          cached_tokens=left['parsed']['cached_tokens']),
                right=dict(path=right['path'], sha256=right['sha256'], output_ids=b,
                           cached_tokens=right['parsed']['cached_tokens']))


def load_run(directory):
    directory = Path(directory).resolve()
    worker = directory / 'serving/worker0'
    rawdir = worker / 'actual-decode'
    paths = {name: path for name, path in [
        ('plan', directory / 'plan.json'), ('diagnostic', directory / 'diagnostic.json'),
        ('run', directory / 'serving/run.json'), ('original_run', directory / 'serving/run.before-diagnostic-marking.json'),
        ('summary', worker / 'actual-decode-summary.json'), ('prompts', rawdir / 'prompts.json'),
        ('server_final', worker / 'server-final.json')]}
    require(all(path.is_file() for path in paths.values()), 'Missing completed diagnostic evidence')
    data = {name: read(path) for name, path in paths.items()}
    plan, diagnostic, run, summary, prompts = [data[name] for name in ['plan','diagnostic','run','summary','prompts']]
    require(plan.get('diagnostic_only') is True and plan.get('measurement_eligible') is False
            and diagnostic.get('diagnostic_only') is True and diagnostic.get('measurement_eligible') is False
            and run.get('diagnostic_only') is True and run.get('measurement_eligible') is False,
            'Diagnostic-only provenance is absent')
    require(plan['wrapper_sha256'] == sha(HERE / 'actual_decode_audit.py') == run.get('diagnostic_wrapper_sha256'),
            'Diagnostic wrapper source hash differs')
    require(diagnostic.get('plan_sha256') == run.get('diagnostic_plan_sha256') == sha(paths['plan'])
            and run.get('diagnostic_original_metadata_sha256') == sha(paths['original_run']),
            'Diagnostic metadata hash linkage differs')
    original = dict(data['original_run'])
    for key in ['diagnostic_only','measurement_eligible','diagnostic_kind','diagnostic_wrapper_sha256',
                'diagnostic_plan_sha256','diagnostic_original_metadata_sha256']:
        original[key] = run.get(key)
    require(original == run, 'Diagnostic marking altered original serving metadata')
    require(run.get('status') == 'execution_complete' and diagnostic.get('status') == 'diagnostic_complete'
            and summary.get('status') == 'diagnostic_complete' and not summary.get('errors'),
            'Diagnostic execution incomplete')
    for name, obj in [('diagnostic', diagnostic), ('summary', summary)]:
        require(obj.get('full_token_coverage') is True and obj.get('measured_requests') == 36
                and obj.get('measured_output_tokens') == 1152, f'{name} measured coverage differs')
    args = run.get('args', {})
    require(args.get('model') == plan.get('model') == '70b' and args.get('family') == plan.get('family') == summary.get('family')
            and (args.get('tp'), args.get('pp'), args.get('replicas')) == (1, 4, 1)
            and args.get('suite') == 'smoke', 'Diagnostic model/topology differs')
    require(args.get('overlap', 'off') == 'off' and run.get('overlap_policy', {}).get('effective', 'off') == 'off',
            'Diagnostic requires native overlap off')
    server = summary.get('server_before', {})
    final_server = data['server_final']
    require(server.get('disable_overlap_schedule') is True
            and final_server.get('disable_overlap_schedule') is True,
            'Actual before/final endpoint overlap must be disabled')
    require(all(server.get(key)==final_server.get(key) for key in
                ['model_path','tp_size','pp_size','dtype','enable_pdmux','chunked_prefill_size']),
            'Actual before/final endpoint identity differs')
    require(Path(server.get('model_path','')).name == 'Llama-3.3-70B-Instruct'
            and (server.get('tp_size'),server.get('pp_size')) == (1,4)
            and server.get('dtype') in {'bfloat16','bf16','torch.bfloat16'}
            and server.get('enable_pdmux') is (plan['family']=='muxpp')
            and (plan['family']!='plain' or server.get('chunked_prefill_size')==2048),
            'Actual endpoint model/topology/dtype/family differs')
    configs = {'spatial':directory/'serving/spatial.yml',
               **{key:Path(value).resolve() for key,value in run.get('configuration_paths',{}).items()}}
    if plan['family']=='muxpp':
        require(isinstance(run.get('configuration_paths',{}).get('worker0.pdmux'),str)
                and bool(run['configuration_paths']['worker0.pdmux'])
                and 'worker0.pdmux' in run.get('configuration_sha256',{}),
                'MuxPP requires effective worker0 PDMux configuration path and hash')
    require(set(configs)==set(run.get('configuration_sha256',{}))
            and all(path.is_file() and sha(path)==run['configuration_sha256'][key] for key,path in configs.items()),
            'Diagnostic configuration hash differs')
    for key in ['source_sha256', 'runtime_source_sha256']:
        require(bool(run.get(key)) and run[key] == plan.get('source_provenance', {}).get(key),
                f'Diagnostic source provenance differs: {key}')
    require(run.get('model_revision') and run.get('model_provenance'), 'Missing model provenance')
    expected = wrapper.condition_plan()
    require(plan.get('cells') == summary.get('planned_cells') == expected
            and len(summary.get('cells', [])) == len(expected)
            and plan.get('measured_requests')==36 and plan.get('measured_output_tokens')==1152
            and Path(plan.get('out','')).resolve()==directory, 'Missing or changed planned cells')
    require(summary.get('fixture_sha256') == sha(paths['prompts'])
            and set(prompts) == set(map(str, wrapper.LENGTHS))
            and all(isinstance(prompts[str(n)], list) and len(prompts[str(n)]) == n
                    and all(type(t) is int and 0 <= t < 128256 for t in prompts[str(n)]) for n in wrapper.LENGTHS),
            'Fixture hash or token lengths differ')
    measured, all_requests, seen_paths = {}, [], set()

    def request(path, ids, count, scope):
        path = Path(path).resolve()
        require(rawdir in path.parents and path.is_file() and path not in seen_paths, 'Invalid or duplicated raw request path')
        seen_paths.add(path)
        item = read(path)
        payload = item.get('request', {})
        require(item.get('status') == 'diagnostic_complete' and item.get('full_token_coverage') is True
                and item.get('diagnostic_only') is True and item.get('measurement_eligible') is False
                and item.get('scope') == scope, 'Request scope/completion differs')
        require(payload.get('input_ids') == ids and payload.get('return_logprob') is True
                and payload.get('top_logprobs_num') == 5 and payload.get('logprob_start_len') == len(ids)-1
                and payload.get('stream') is False and payload.get('sampling_params') ==
                dict(temperature=0, max_new_tokens=count, ignore_eos=True), 'Request prompt or sampling parameters differ')
        parsed = wrapper.parse_decode_response(item.get('raw_response'), ids, count)
        require(parsed == item.get('parsed') and item.get('output_ids') == item['raw_response']['output_ids'],
                'Stored logprob parsing/output differs from raw response')
        require(all(type(item.get(k)) in (int,float) and math.isfinite(item[k]) for k in
                    ['sent_unix','sent_monotonic','finished_unix','finished_monotonic'])
                and item['finished_monotonic'] >= item['sent_monotonic'], 'Invalid request timing')
        item.update(path=str(path), sha256=sha(path), request_sha256=digest(payload),
                    raw_response_sha256=digest(item['raw_response']))
        all_requests.append(item)
        return item

    rng = random.Random(730031)
    long_ids = [128000] + [rng.randrange(128000) for _ in range(31662)]
    request(rawdir / 'request-long-prewarm.json', long_ids, 8, 'long_prewarm')
    cell_evidence = []
    for declared, cell in zip(expected, summary['cells']):
        require(all(cell.get(key) == val for key, val in declared.items())
                and cell.get('status') == 'diagnostic_complete' and cell.get('cache_flush_acknowledged') is True,
                'Cell conditions/flush/completion differ')
        lengths = declared['input_lengths']
        require(cell.get('completed_requests') == len(lengths)
                and len(cell.get('measured_files', [])) == len(lengths)
                and len(cell.get('warmup_files', [])) == (len(lengths) if declared['cache'] == 'warm' else 0),
                'Cell request inventory differs')
        folder = rawdir / declared['id']
        for i, path in enumerate(cell['warmup_files']):
            require(Path(path).resolve() == folder / f'request-warmup-{i:02d}.json', 'Wrong warmup file mapping')
            request(path, prompts[str(lengths[i])], 1, 'cache_warmup')
        captured = []
        for i, path in enumerate(cell['measured_files']):
            require(Path(path).resolve() == folder / f'request-measured-{i:02d}.json', 'Wrong measured file mapping')
            item = request(path, prompts[str(lengths[i])], wrapper.TOKENS, 'measured')
            key = (declared['repeat'], declared['cache'], declared['mode'], lengths[i])
            require(key not in measured, 'Duplicate measured condition')
            measured[key] = item
            captured.append(item)
        caches = {str(i): item['parsed']['cached_tokens'] for i, item in enumerate(captured)}
        spread = max(item['sent_monotonic'] for item in captured) - min(item['sent_monotonic'] for item in captured)
        require(cell.get('actual_cached_tokens') == caches and cell.get('send_spread_sec') == spread
                and cell.get('focus_output_ids') == captured[lengths.index(wrapper.FOCUS)]['output_ids'],
                'Cell cache/timing/focus summary differs from raw evidence')
        cell_evidence.append({**declared, 'actual_cached_tokens': caches, 'send_spread_sec': spread,
                              'cache_flush_unix': cell.get('cache_flush_unix')})
    require(seen_paths == {p.resolve() for p in rawdir.rglob('request-*.json')}
            and len(seen_paths) == 55 and len(measured) == 36, 'Extra or missing raw request files')
    require(summary.get('warmup_requests') == 19 and summary.get('warmup_output_tokens') == 26
            and summary.get('measured_completed_requests') == 36, 'Warmup/measured coverage differs')
    return dict(directory=str(directory), family=plan['family'], measured=measured, raw_requests=all_requests,
                cells=cell_evidence, prompts=prompts, metadata=run,
                evidence_sha256={str(p): sha(p) for p in [*paths.values(),*configs.values()]},
                server_before=server,server_final=final_server,
                request_evidence=[dict(path=r['path'], sha256=r['sha256'], scope=r['scope'],
                                      request_sha256=r['request_sha256'], raw_response_sha256=r['raw_response_sha256'],
                                      input_tokens=len(r['request']['input_ids']), output_tokens=len(r['output_ids']),
                                      cached_tokens=r['parsed']['cached_tokens']) for r in all_requests])


def token_summary(run):
    tokens = [t for r in run['measured'].values() for t in r['parsed']['tokens']]
    return dict(tokens=len(tokens), reported_top1_matches=sum(t['reported_top1_match'] for t in tokens),
                chosen_equals_reported_max=sum(t['chosen_equals_reported_max'] for t in tokens),
                canonical_top5_matches=sum(t['canonical_top5_match'] for t in tokens),
                tokens_with_reported_max_ties=sum(len(t['reported_max_tie_ids']) > 1 for t in tokens),
                top5_tie_sets_may_be_incomplete=sum(not t['all_max_ties_in_top5'] for t in tokens),
                positive_regret_tokens=sum(t['regret'] > 0 for t in tokens),
                negative_regret_tokens=sum(t['regret'] < 0 for t in tokens),
                maximum_regret=max(t['regret'] for t in tokens),
                mean_regret=sum(t['regret'] for t in tokens)/len(tokens))


def analyze_pair(plain_path, mux_path):
    left, right = load_run(plain_path), load_run(mux_path)
    require((left['family'], right['family']) == ('plain', 'muxpp'), 'Expected ordinary/MuxPP pair')
    require(left['prompts'] == right['prompts'], 'Cross-system prompt fixtures differ')
    a, b = left['metadata'], right['metadata']
    require(a['runtime_source_sha256'] == b['runtime_source_sha256'], 'Neural-network runtime source differs')
    require(a['model_revision'] == b['model_revision'] and a['model_provenance'] == b['model_provenance'],
            'Model revision/provenance differs')
    differences = [key for key in set(a['source_sha256']) | set(b['source_sha256'])
                   if a['source_sha256'].get(key) != b['source_sha256'].get(key)]
    # The reviewed additive overlap flag is the only historical source
    # transition accepted here, and both endpoints must remain overlap off.
    source_relation = 'identical_serving_and_runtime_source'
    if differences:
        spec = importlib.util.spec_from_file_location('_actual_compat', HERE / 'serving_compatibility.py')
        compat = importlib.util.module_from_spec(spec); spec.loader.exec_module(compat)
        require(differences == [compat.TARGET] and
                {a['source_sha256'][compat.TARGET], b['source_sha256'][compat.TARGET]} == {compat.BASE, compat.CANDIDATE},
                'Serving source differs beyond reviewed overlap-off transition')
        source_relation = 'known_overlap_off_runner_transition_no_neural_network_change'
    comparisons = []

    def compare(kind, lrun, lkey, rrun, rkey):
        comparisons.append(dict(kind=kind, left_system=lrun['family'], right_system=rrun['family'],
                                left_condition=list(lkey), right_condition=list(rkey),
                                **compare_records(lrun['measured'][lkey], rrun['measured'][rkey])))

    for key in left['measured']:
        compare('cross_system', left, key, right, key)
    for run in [left, right]:
        for cache in ['cold', 'warm']:
            for mode in ['serial', 'concurrent8']:
                for length in ([wrapper.FOCUS] if mode == 'serial' else wrapper.LENGTHS):
                    compare('repeat', run, (1,cache,mode,length), run, (2,cache,mode,length))
        for repeat in [1, 2]:
            for mode in ['serial', 'concurrent8']:
                for length in ([wrapper.FOCUS] if mode == 'serial' else wrapper.LENGTHS):
                    compare('cache', run, (repeat,'cold',mode,length), run, (repeat,'warm',mode,length))
            for cache in ['cold', 'warm']:
                compare('serial_vs_concurrent', run, (repeat,cache,'serial',wrapper.FOCUS),
                        run, (repeat,cache,'concurrent8',wrapper.FOCUS))
    require(len(comparisons) == 116, 'Unexpected comparison coverage')
    groups = {}
    for kind in ['cross_system', 'repeat', 'cache', 'serial_vs_concurrent']:
        rows = [r for r in comparisons if r['kind'] == kind]
        groups[kind] = dict(pairs=len(rows), exact_pairs=sum(r['exact_match'] for r in rows),
                            mismatching_pairs=sum(not r['exact_match'] for r in rows),
                            first_divergence_positions=dict(Counter(str(r['first_divergence']['position'])
                                for r in rows if r['first_divergence'] is not None)))
    by_system = {run['family']: {kind: dict(pairs=len(rows), exact_pairs=sum(r['exact_match'] for r in rows))
                              for kind in ['repeat','cache','serial_vs_concurrent']
                              for rows in [[r for r in comparisons if r['kind']==kind and r['left_system']==run['family']]]}
                 for run in [left,right]}
    return dict(schema_version=1, status='completed_diagnostic_analysis', diagnostic_only=True, measurement_eligible=False,
                coverage=dict(deployments=2,cells=16,measured_requests=72,measured_tokens=2304,raw_request_files=110,
                              warmup_requests=38,warmup_tokens=52,comparison_pairs=116),
                source_relation=source_relation, runtime_source_sha256=a['runtime_source_sha256'],
                model_revision=a['model_revision'],model_provenance=a['model_provenance'],
                runs=[dict(directory=run['directory'],family=run['family'],evidence_sha256=run['evidence_sha256'],
                           request_evidence=run['request_evidence'],cells=run['cells'],token_summary=token_summary(run),
                           server_before=run['server_before'],server_final=run['server_final'],effective_args=run['metadata']['args'],
                           configuration_sha256=run['metadata']['configuration_sha256'])
                      for run in [left,right]],
                comparison_summary=groups, within_system_summary=by_system, comparisons=comparisons,
                margin_labels=dict(exact_tie='Exact equality in reported logprobs, not a claim about full-precision logits.',
                    near_0_125='Absolute difference from0.125<=1e-6 is a descriptive category only; no correctness tolerance.',
                    above_0_5='A reporting threshold for larger gaps, not a correctness or KV-corruption decision.'),
                limitations=['Only the first divergent output position has the same actual prefix; later-position logprobs are not compared.',
                    'API logprob/top5 reporting can change execution shape/path. This bounded diagnostic does not substitute for ordinary inference or an independent reference model.',
                    'Cache state, batch/order, padding and arithmetic path can covary; observed cache counts and send spreads are recorded, not isolated causal proof.',
                    'Pair comparisons reuse the same measured requests across condition contrasts; counts are not independent statistical samples.',
                    'Top5 truncation leaves absent alternatives bounded only and may omit other tied tokens. Quantization-sized gaps alone do not identify a kernel or numerical cause.',
                    'No epsilon, tie or small-regret category changes strict token mismatch status. Resource/functional and final performance eligibility remain separate gates.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plain', type=Path, required=True)
    parser.add_argument('--muxpp', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = analyze_pair(args.plain, args.muxpp)
    result['analysis_script'] = dict(path=str(Path(__file__).resolve()), sha256=sha(__file__))
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(out=str(args.out),sha256=sha(args.out),coverage=result['coverage'],summary=result['comparison_summary'])))


if __name__ == '__main__':
    main()
