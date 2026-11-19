"""CPU-only tests for the bounded scoreless MuxPP diagnostic wrapper."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE=Path(__file__).resolve().parent


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


target=load('_tokens_target',HERE/'actual_decode_tokens_audit.py') if (HERE/'actual_decode_tokens_audit.py').exists() else None
fixture=load('_tokens_fixture',HERE/'test_actual_decode_audit.py')


class TokenRunner(fixture.FakeRunner):
    def __init__(self,model):super().__init__(model,'muxpp')

    def request(self,port,endpoint,payload=None,timeout=600):
        value=super().request(port,endpoint,payload,timeout)
        if endpoint=='/get_server_info':value['disable_overlap_schedule']=True
        else:
            for key in ['return_logprob','top_logprobs_num','logprob_start_len']:
                if key in payload:raise AssertionError('Scoreless request included '+key)
            value['meta_info'].pop('output_token_logprobs')
            value['meta_info'].pop('output_top_logprobs')
        return value


class TokensAuditTests(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(target,'Separate tokens-only wrapper is missing')

    def test_conditions_preserve_eight_cells_and_36_measured_requests(self):
        self.assertEqual(target.condition_plan(),fixture.audit.condition_plan())
        self.assertEqual(len(target.condition_plan()),8)
        self.assertEqual(sum(len(x['input_lengths']) for x in target.condition_plan()),36)

    def test_parser_distinguishes_output_coverage_from_missing_scores(self):
        runner=TokenRunner(Path('/model'))
        body=runner.request(1,'/generate',{'input_ids':[128000,42],'sampling_params':{'max_new_tokens':32}})
        parsed=target.parse_token_response(body,[128000,42],32)
        self.assertTrue(parsed['full_output_token_coverage'])
        self.assertFalse(parsed['full_logprob_coverage'])
        self.assertEqual(parsed['actual_logprob_tokens'],0)
        self.assertEqual(parsed['audited_output_tokens'],32)
        self.assertNotIn('full_token_coverage',parsed)
        self.assertNotIn('tokens',parsed)

    def test_parser_rejects_wrong_count_ids_cache_or_unexpected_scores(self):
        for mutation in ['count','ids','cache','scores']:
            with self.subTest(mutation=mutation):
                body=TokenRunner(Path('/model')).request(1,'/generate',{'input_ids':[128000,42],'sampling_params':{'max_new_tokens':32}})
                if mutation=='count':body['meta_info']['completion_tokens']=31
                elif mutation=='ids':body['output_ids'][0]=True
                elif mutation=='cache':body['meta_info']['cached_tokens']=-1
                else:body['meta_info']['output_token_logprobs']=[[-.25,20000,None]]
                with self.assertRaises(ValueError):target.parse_token_response(body,[128000,42],32)

    def test_full_preflight_preserves_all_raw_and_never_requests_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);runner=TokenRunner(out/'model')
            result=target.run_preflight(runner,32100,out/'model',out)
            self.assertEqual(result['status'],'tokens_only_diagnostic_complete')
            self.assertEqual(result['measured_requests'],36)
            self.assertEqual(result['measured_output_tokens'],1152)
            self.assertEqual((result['warmup_requests'],result['warmup_output_tokens']),(19,26))
            self.assertEqual(len(runner.calls),55);self.assertEqual(len(runner.flushes),10)
            self.assertEqual(runner.phases,['initial','final'])
            files=list((out/'actual-decode-tokens').rglob('request-*.json'))
            self.assertEqual(len(files),55)
            self.assertTrue(result['full_output_token_coverage'])
            self.assertFalse(result['full_logprob_coverage']);self.assertEqual(result['actual_logprob_tokens'],0)
            self.assertNotIn('full_token_coverage',result)
            for p in files:
                data=json.loads(p.read_text())
                self.assertIn('raw_response',data);self.assertEqual(data['actual_logprob_tokens'],0)
                self.assertFalse(data['measurement_eligible'])
            self.assertEqual(sum(r['sampling_params']['max_new_tokens'] for r in runner.calls),1178)

    def test_failed_request_preserves_raw_response_and_incomplete_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);runner=TokenRunner(out/'model');real=runner.request
            def bad(port,endpoint,payload=None,timeout=600):
                result=real(port,endpoint,payload,timeout)
                if payload and payload['sampling_params']['max_new_tokens']==32:result['meta_info']['completion_tokens']=31
                return result
            runner.request=bad
            with self.assertRaises(ValueError):target.run_preflight(runner,32100,out/'model',out)
            summary=json.loads((out/'actual-decode-tokens-summary.json').read_text())
            self.assertFalse(summary['full_output_token_coverage'])
            self.assertFalse(summary['full_logprob_coverage'])
            raw=json.loads((out/'actual-decode-tokens/repeat1-cold-serial/request-measured-00.json').read_text())
            self.assertIn('raw_response',raw)

    def test_plan_is_muxpp_70b_only_and_explicitly_scoreless(self):
        plan=target.make_plan(['--model','70b','--family','muxpp'],Path('/tmp/token-only'))
        self.assertEqual(plan['diagnostic_mode'],'tokens_only')
        self.assertFalse(plan['logprobs_requested']);self.assertFalse(plan['full_logprob_coverage'])
        self.assertEqual(plan['actual_logprob_tokens'],0)
        self.assertFalse(plan['measurement_eligible']);self.assertEqual(plan['measured_requests'],36)
        self.assertNotIn('full_token_coverage',plan)
        for argv in [['--model','8b','--family','muxpp'],['--model','70b','--family','plain']]:
            with self.assertRaises(ValueError):target.make_plan(argv,Path('/tmp/token-only'))

    def test_unowned_wrapper_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'unowned';server=SimpleNamespace(require_active_controller=mock.Mock(side_effect=RuntimeError('no controller')))
            with mock.patch.dict(target.sys.modules,{'server_run':server}),mock.patch.object(target.sys,'argv',
                ['tokens.py','--out',str(out),'--','--model','70b','--family','muxpp']):
                with self.assertRaisesRegex(RuntimeError,'controller'):target.main()
            self.assertFalse(out.exists())

    def test_failure_restores_preflight_and_marks_only_output_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'controlled';original=object()
            server=SimpleNamespace(require_active_controller=lambda:None,preflight=original,
                                   source_provenance=lambda repo:{'source_sha256':{'fixture':'hash'}})
            def main():
                self.assertIsNot(server.preflight,original)
                (out/'serving').mkdir();(out/'serving/run.json').write_text(json.dumps({'status':'failed'}))
                raise RuntimeError('synthetic')
            server.main=main
            with mock.patch.dict(target.sys.modules,{'server_run':server}),mock.patch.object(target.sys,'argv',
                ['tokens.py','--out',str(out),'--','--model','70b','--family','muxpp']):
                with self.assertRaisesRegex(RuntimeError,'synthetic'):target.main()
            self.assertIs(server.preflight,original)
            meta=json.loads((out/'serving/run.json').read_text())
            self.assertTrue(meta['diagnostic_only']);self.assertFalse(meta['measurement_eligible'])
            self.assertFalse(meta['full_output_token_coverage']);self.assertFalse(meta['full_logprob_coverage'])
            self.assertEqual(meta['actual_logprob_tokens'],0);self.assertNotIn('full_token_coverage',meta)


if __name__=='__main__':unittest.main()
