import unittest
from unittest import mock
import tempfile
from pathlib import Path
import sys
import os
import json
import server_run
from server_run import validate_topology, graph_sizes, worker_layout, router_command, pd_pairs, CONTEXT_LENGTH, CORRECTNESS_LENGTHS

class TopologyTests(unittest.TestCase):
    def test_exact_four_gpu_budget(self):
        for topology in [(1,1,4),(2,1,2),(4,1,1),(1,2,2),(1,4,1),(2,2,1)]:
            validate_topology('chunked',*topology)
        for topology in [(1,1,1),(4,2,1),(2,2,2),(0,4,1)]:
            with self.assertRaises(ValueError): validate_topology('chunked',*topology)
    def test_original_mux_never_pp(self):
        with self.assertRaises(ValueError): validate_topology('muxwise',1,4,1)
    def test_muxpp_requires_pp(self):
        with self.assertRaises(ValueError): validate_topology('muxpp',4,1,1)
    def test_capture_sizes_respect_configured_limit(self):
        self.assertEqual(graph_sizes(3), [1, 3])
        self.assertEqual(graph_sizes(8), [1, 4, 8])
        self.assertEqual(graph_sizes(32), [1, 4, 8, 16, 32])
        with self.assertRaises(ValueError): graph_sizes(0)
    def test_pd_uses_four_gpus_and_requires_both_roles(self):
        for tp, replicas, prefills in [(1,4,1),(1,4,2),(1,4,3),(2,2,1)]:
            validate_topology('pd',tp,1,replicas,prefills)
        for args in [(1,2,2,1),(4,1,1,1),(1,1,4,0),(1,1,4,4),(2,1,2,2)]:
            with self.assertRaises(ValueError): validate_topology('pd',*args)
    def test_dp_router_contains_every_replica(self):
        workers=worker_layout('chunked',2,1,2,0,32100)
        cmd,env=router_command(workers,32100,'chunked')
        self.assertIn('--worker-urls',cmd)
        self.assertIn('http://127.0.0.1:32101',cmd)
        self.assertIn('http://127.0.0.1:32102',cmd)
        self.assertEqual(len({gpu for worker in workers for gpu in worker['gpus']}),4)
        self.assertNotIn('--mini-lb',cmd)
    def test_pd_router_and_warmup_pairs_cover_all_workers(self):
        for prefills in (1,2,3):
            workers=worker_layout('pd',1,1,4,prefills,32100)
            cmd,env=router_command(workers,32100,'pd')
            self.assertIn('--mini-lb',cmd)
            self.assertEqual(cmd.count('--prefill'),prefills)
            self.assertEqual(cmd.count('--decode'),4-prefills)
            pairs=pd_pairs(workers)
            self.assertEqual({w['index'] for pair in pairs for w in pair},set(range(4)))
            self.assertIn('sgl-router/py_src',env['PYTHONPATH'])
    def test_max_trace_boundary_has_context_headroom(self):
        self.assertEqual(CONTEXT_LENGTH,33792)
        self.assertIn(32704,CORRECTNESS_LENGTHS)
        self.assertEqual(max(CORRECTNESS_LENGTHS)+64,32768)
        self.assertLess(32768,CONTEXT_LENGTH)
    def test_paired_pd_request_shares_bootstrap_identity_and_preserves_prefill_cache(self):
        workers=worker_layout('pd',2,1,2,1,32100)
        observed=[]
        def response(port,path,payload):
            observed.append((port,path,payload))
            return {'output_ids':[1], 'meta_info':{'cached_tokens':8000 if port==32101 else 0}}
        with mock.patch.object(server_run,'request',side_effect=response):
            result=server_run.paired_request(workers[0],workers[1],{'input_ids':[128000],'stream':False})
        self.assertEqual({row[0] for row in observed},{32101,32102})
        self.assertEqual(observed[0][2]['bootstrap_room'],observed[1][2]['bootstrap_room'])
        self.assertEqual(observed[0][2]['bootstrap_port'],33100)
        self.assertEqual(result['_prefill_meta_info']['cached_tokens'],8000)
    def test_standalone_runner_honors_suspension_before_creating_output_or_launching(self):
        import control
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);marker=root/'RESERVATION_DISABLED.json';marker.write_text('{}')
            args=['server_run.py','--model','8b','--family','plain','--tp','4','--pp','1','--out',str(root/'never-created')]
            with mock.patch.object(sys,'argv',args),mock.patch.object(control,'DEFAULT_ROOT',root),mock.patch.object(control,'GLOBAL_SUSPENSION',marker),mock.patch('subprocess.Popen',side_effect=AssertionError('no subprocess allowed')):
                with self.assertRaisesRegex(RuntimeError,'suspended'):server_run.main()
            self.assertFalse((root/'never-created').exists())
    def test_runner_requires_live_controller_token_without_holder(self):
        import control
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            parent=control.identity(os.getpid())
            state={'active':{'token':'expected-token','controller':parent},'holder':None}
            def write(value):
                (root/'controller.json').write_text(json.dumps(value))
            with mock.patch.dict(os.environ,{'A100_CAMPAIGN_ROOT':str(root),'A100_CONTROL_TOKEN':'expected-token'}),mock.patch.object(control,'GLOBAL_SUSPENSION',root/'disabled'),mock.patch('subprocess.Popen',side_effect=AssertionError('CPU test cannot launch')):
                with self.assertRaisesRegex(RuntimeError,'active controller'):server_run.require_active_controller()
                write(state)
                self.assertEqual(server_run.require_active_controller().root,root)
                with mock.patch.dict(os.environ,{'A100_CONTROL_TOKEN':'forged'}):
                    with self.assertRaisesRegex(RuntimeError,'active controller'):server_run.require_active_controller()
                with mock.patch.dict(os.environ,{'A100_CONTROL_TOKEN':''}):
                    with self.assertRaisesRegex(RuntimeError,'active controller'):server_run.require_active_controller()
                for changed in ({'starttime':parent['starttime']+1},{'pid':999999999},{'uid':-1}):
                    stale={**state,'active':{**state['active'],'controller':{**parent,**changed}}}
                    write(stale)
                    with self.assertRaisesRegex(RuntimeError,'active controller'):server_run.require_active_controller()
                write({**state,'holder':{'identity':parent}})
                with self.assertRaisesRegex(RuntimeError,'active controller'):server_run.require_active_controller()

if __name__=='__main__': unittest.main()
