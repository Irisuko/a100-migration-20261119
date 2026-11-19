"""CPU-only actual-decode evidence and first-divergence analysis tests."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


target = load('_actual_analysis', HERE / 'analyze_actual_decode.py') if (HERE / 'analyze_actual_decode.py').exists() else None
fixture = load('_actual_fixtures', HERE / 'test_actual_decode_audit.py')
wrapper = fixture.audit


def record(ids, tokens):
    body = fixture.response(ids, len(tokens))
    body['output_ids'] = tokens
    for i, token in enumerate(tokens):
        body['meta_info']['output_token_logprobs'][i][1] = token
        body['meta_info']['output_top_logprobs'][i][0][1] = token
    return dict(request={'input_ids': ids}, output_ids=tokens,
                parsed=wrapper.parse_decode_response(body, ids, len(tokens)), path='synthetic', sha256='a'*64)


class ActualAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(target, 'actual-decode analysis implementation missing')

    def test_first_divergence_uses_common_prefix_only(self):
        left, right = record([128000,42], [20000,20001,20002]), record([128000,42], [20000,20009,20008])
        result = target.compare_records(left, right)
        self.assertFalse(result['exact_match'])
        self.assertEqual(result['first_divergence']['position'], 1)
        self.assertEqual(result['first_divergence']['common_output_prefix'], [20000])
        self.assertEqual(result['first_divergence']['left']['chosen'], 20001)
        self.assertEqual(result['first_divergence']['right']['chosen'], 20009)
        self.assertNotIn('later_position_logprob_comparison', result)

    def test_reported_tie_never_converts_strict_mismatch_to_pass(self):
        left, right = record([128000], [20000]), record([128000], [20001])
        for side, a, b in [(left,20000,20001),(right,20001,20000)]:
            t = side['parsed']['tokens'][0]
            t.update(top5_ids=[a,b,100,101,102],top5_logprobs=[-.25,-.25,-1,-2,-3],
                     top_two_margin=0,reported_max_tie_ids=sorted([a,b]),all_max_ties_in_top5=True)
        result = target.compare_records(left, right)
        self.assertFalse(result['exact_match'])
        self.assertEqual(result['first_divergence']['left']['other_choice_logprob'], -.25)
        self.assertEqual(result['first_divergence']['left']['chosen_minus_other'], 0)
        self.assertEqual(result['first_divergence']['left']['margin_description'], 'exact_tie_in_reported_logprobs')
        self.assertNotIn('correctness_pass', result)

    def test_absent_other_choice_is_a_bound_not_fabricated_logprob(self):
        result = target.compare_records(record([128000],[20000]), record([128000],[20001]))
        left = result['first_divergence']['left']
        self.assertIsNone(left['other_choice_logprob'])
        self.assertIsNone(left['chosen_minus_other'])
        self.assertEqual(left['other_choice_regret_lower_bound'], 2.75)

    def test_unequal_prompts_are_not_compared(self):
        with self.assertRaisesRegex(ValueError, 'prompt'):
            target.compare_records(record([128000,1],[20000]), record([128000,2],[20000]))

    def prepare(self, out, family, runtime='a'):
        serving = out/'serving';worker=serving/'worker0';worker.mkdir(parents=True)
        model = out.parent/'Llama-3.3-70B-Instruct'
        runner = fixture.FakeRunner(model, family)
        wrapper.run_preflight(runner, 32100, model, worker, family)
        summary_path=worker/'actual-decode-summary.json'
        summary=json.loads(summary_path.read_text())
        summary['server_before']['disable_overlap_schedule']=True
        summary_path.write_text(json.dumps(summary))
        (worker/'server-final.json').write_text(json.dumps(summary['server_before']))
        plan = wrapper.make_plan(['--model','70b','--family',family],out)
        source = {'source_sha256':{'python/sglang/synthetic.py':runtime*64,'scripts/a100_gpu0123/server_run.py':'b'*64},
                  'runtime_source_sha256':{'python/sglang/synthetic.py':runtime*64}}
        plan.update(wrapper_path=str(HERE/'actual_decode_audit.py'),wrapper_sha256=wrapper.sha(HERE/'actual_decode_audit.py'),source_provenance=source)
        (out/'plan.json').write_text(json.dumps(plan))
        args=dict(model='70b',family=family,tp=1,pp=4,replicas=1,overlap='off',suite='smoke')
        (serving/'spatial.yml').write_text('fixture: true\n')
        run=dict(status='execution_complete',args=args,**source,model_revision='fixture',model_provenance={'config.json':'c'*64},
                 configuration_sha256={'spatial':wrapper.sha(serving/'spatial.yml')},
                 overlap_policy={'requested':'off','effective':'off','enabled':False,'supported':False})
        if family=='muxpp':
            run['configuration_paths']={'worker0.pdmux':str((serving/'spatial.yml').resolve())}
            run['configuration_sha256']['worker0.pdmux']=wrapper.sha(serving/'spatial.yml')
        (serving/'run.json').write_text(json.dumps(run))
        wrapper.mark_diagnostic(out,plan)
        (out/'diagnostic.json').write_text(json.dumps(dict(status='diagnostic_complete',diagnostic_only=True,measurement_eligible=False,
          full_token_coverage=True,measured_requests=36,measured_output_tokens=1152,plan_sha256=wrapper.sha(out/'plan.json'))))

    def test_full_pair_has_116_measurement_pairs_and_all_110_raw_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.prepare(root/'plain','plain');self.prepare(root/'mux','muxpp')
            report=target.analyze_pair(root/'plain',root/'mux')
            self.assertEqual(report['coverage']['measured_requests'],72)
            self.assertEqual(report['coverage']['raw_request_files'],110)
            self.assertEqual(report['coverage']['measured_tokens'],2304)
            self.assertEqual(len(report['comparisons']),116)
            self.assertEqual(report['comparison_summary']['cross_system']['exact_pairs'],36)
            self.assertTrue(report['diagnostic_only'])
            self.assertFalse(report['measurement_eligible'])

    def test_reader_rejects_tampered_prompt_logprobs_fixture_or_cells(self):
        for mutation in ['prompt','logprobs','fixture','cells','extra_request','configuration','server']:
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp)/'plain';self.prepare(out,'plain');worker=out/'serving/worker0'
                req=worker/'actual-decode/repeat1-cold-serial/request-measured-00.json'
                if mutation in ['prompt','logprobs']:
                    data=json.loads(req.read_text())
                    if mutation=='prompt':data['request']['input_ids'][1]=1
                    else:data['parsed']['tokens'][0]['chosen_logprob']=-9
                    req.write_text(json.dumps(data))
                elif mutation=='fixture':
                    (worker/'actual-decode/prompts.json').write_text('{}')
                elif mutation=='cells':
                    p=worker/'actual-decode-summary.json';data=json.loads(p.read_text());data['cells'].pop();p.write_text(json.dumps(data))
                elif mutation=='extra_request':(worker/'actual-decode/request-extra.json').write_text('{}')
                elif mutation=='configuration':(out/'serving/spatial.yml').write_text('changed: true\n')
                else:
                    p=worker/'actual-decode-summary.json';data=json.loads(p.read_text());data['server_before']['enable_pdmux']=True;p.write_text(json.dumps(data))
                with self.assertRaises(ValueError):target.load_run(out)

    def test_different_runtime_source_fails_pair_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.prepare(root/'plain','plain');self.prepare(root/'mux','muxpp',runtime='d')
            with self.assertRaisesRegex(ValueError,'source'):
                target.analyze_pair(root/'plain',root/'mux')

    def test_actual_before_and_final_overlap_flag_must_be_true(self):
        for location in ['before','final']:
            for value in [False,None]:
                with self.subTest(location=location,value=value),tempfile.TemporaryDirectory() as tmp:
                    out=Path(tmp)/'plain';self.prepare(out,'plain');worker=out/'serving/worker0'
                    path=worker/('actual-decode-summary.json' if location=='before' else 'server-final.json')
                    data=json.loads(path.read_text());snapshot=data['server_before'] if location=='before' else data
                    snapshot['disable_overlap_schedule']=value;path.write_text(json.dumps(data))
                    with self.assertRaisesRegex(ValueError,'overlap'):
                        target.load_run(out)

    def test_muxpp_requires_its_effective_config_path_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'mux';self.prepare(out,'muxpp')
            original=out/'serving/run.before-diagnostic-marking.json'
            run=json.loads(original.read_text());run.pop('configuration_paths')
            run['configuration_sha256'].pop('worker0.pdmux')
            original.write_text(json.dumps(run));(out/'serving/run.json').write_text(json.dumps(run))
            wrapper.mark_diagnostic(out,json.loads((out/'plan.json').read_text()))
            with self.assertRaisesRegex(ValueError,'PDMux'):
                target.load_run(out)


if __name__ == '__main__':unittest.main()
