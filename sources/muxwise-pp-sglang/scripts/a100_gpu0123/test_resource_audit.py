"""CPU-only regressions: parse evidence and extract scheduler methods via AST."""
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
MODULE = Path(__file__).with_name('resource_audit.py')


def scheduler_methods():
    tree = ast.parse((ROOT / 'python/sglang/srt/managers/scheduler.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Scheduler')
    names = {'check_memory', '_log_a100_resource_audit'}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(methods) == 2, 'opt-in resource audit method is missing'
    messages = []
    clock = NS(now=100.0)
    env = {'SGLANG_A100_RESOURCE_AUDIT': '1'}
    memory = NS(memory_allocated=lambda: 10, memory_reserved=lambda: 20,
                max_memory_allocated=lambda: 30, max_memory_reserved=lambda: 40)
    scope = {'json': json, 'os': NS(environ=env), 'time': NS(monotonic=lambda: clock.now, time=lambda: clock.now),
             'torch': NS(cuda=memory), 'logger': NS(info=lambda fmt, *args: messages.append(fmt % args)),
             'DisaggregationMode': NS(DECODE='decode')}
    exec(compile(ast.Module(body=methods, type_ignores=[]), '<scheduler-audit-methods>', 'exec'), scope)
    scheduler = NS(is_hybrid=False, max_total_num_tokens=100, disaggregation_mode='null',
                   req_to_token_pool=NS(size=4, free_slots=list(range(4))), enable_metrics=False,
                   tree_cache=NS(protected_size=lambda: 0), pp_rank=1, tp_rank=0, dp_rank=None,
                   gpu_id=0, device='cuda', forward_ct=9,
                   _publish_kv_events=lambda: None,
                   _get_token_info=lambda: (0, 0, 80, 20))
    scheduler._log_a100_resource_audit = lambda *a: scope['_log_a100_resource_audit'](scheduler, *a)
    return scope, scheduler, messages, clock, env


class SchedulerEvidenceTests(unittest.TestCase):
    def test_opt_in_records_verified_counts_and_memory_without_importing_runtime(self):
        scope, scheduler, messages, _, _ = scheduler_methods()
        scheduler._pp_receive_pool = NS(free=[0, 1], buffers=[None, None])
        scope['check_memory'](scheduler)
        record = json.loads(messages[0].split('A100 resource audit ', 1)[1])
        self.assertEqual(record['free_kv'] + record['evictable'], record['capacity'])
        self.assertEqual(record['free_reqs'], record['request_capacity'])
        self.assertEqual(record['free_receive_slots'], record['receive_slot_capacity'])
        self.assertEqual(record['cuda_memory']['max_allocated'], 30)

    def test_existing_leak_assertion_prevents_positive_log(self):
        scope, scheduler, messages, _, _ = scheduler_methods()
        scheduler._get_token_info = lambda: (0, 0, 79, 20)
        with self.assertRaisesRegex(ValueError, 'memory leak'):
            scope['check_memory'](scheduler)
        self.assertEqual(messages, [])

    def test_disabled_and_throttled_logging_does_not_query_cuda(self):
        scope, scheduler, messages, clock, env = scheduler_methods()
        env.clear()
        scope['torch'] = None
        scope['check_memory'](scheduler)
        self.assertEqual(messages, [])
        env['SGLANG_A100_RESOURCE_AUDIT'] = '1'
        scheduler._a100_resource_audit_last = clock.now
        scope['check_memory'](scheduler)
        self.assertEqual(messages, [])

    def test_decode_preallocation_is_included_in_request_capacity(self):
        scope, scheduler, messages, _, _ = scheduler_methods()
        scheduler.disaggregation_mode = 'decode'
        scheduler.req_to_token_pool.pre_alloc_size = 2
        scheduler.req_to_token_pool.free_slots = list(range(6))
        scope['check_memory'](scheduler)
        record = json.loads(messages[0].split('A100 resource audit ', 1)[1])
        self.assertEqual(record['request_capacity'], 6)


class RunEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(MODULE.exists(), 'resource_audit.py is missing')
        spec = importlib.util.spec_from_file_location('a100_resource_audit', MODULE)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'run.json').write_text(json.dumps({'args': {'tp': 2, 'pp': 2, 'replicas': 1,
                                                               'family': 'muxpp'}, 'status': 'execution_complete'}))
        (self.root / 'worker0').mkdir()
        (self.root / 'worker0/resource-phase-markers.json').write_text(json.dumps([
            {'phase': 'after-cancel', 'start_unix': 100, 'end_unix': 102},
            {'phase': 'after-cancel-pre-flush', 'start_unix': 100, 'end_unix': 102},
            {'phase': 'run-final-pre-flush', 'start_unix': 200, 'end_unix': 202},
            {'phase': 'run-final', 'start_unix': 200, 'end_unix': 202}]))
        (self.root / 'worker0/server-final.json').write_text(json.dumps({'tokenizer_pending_requests': 0}))
        self.records = [dict(time_unix=stamp, pp_rank=pp, tp_rank=tp, capacity=100,
                             free_kv=80, evictable=20, protected=0, free_reqs=4,
                             request_capacity=4, free_receive_slots=2 if pp else 0,
                             receive_slot_capacity=2 if pp else 0,
                             cuda_memory={'allocated': 10, 'reserved': 20,
                                          'max_allocated': 30, 'max_reserved': 40})
                        for stamp in (101, 201) for pp in range(2) for tp in range(2)]

    def audit(self, suffix=''):
        log = '\n'.join('[2026-11-14 10:00:00] A100 resource audit ' + json.dumps(r)
                        for r in self.records) + '\n' + suffix
        (self.root / 'server-0.log').write_text(log)
        return self.module.audit_run(self.root)

    def test_complete_four_rank_evidence_passes_and_preserves_memory_scope(self):
        result = self.audit('[2026-11-14 10:00:00 TP0 PP0] Capture cuda graph end. Time elapsed: 1.0 s. mem usage=0.75 GB. avail mem=1.0 GB.')
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(len(result['workers'][0]['ranks']), 4)
        rank = result['workers'][0]['ranks'][0]
        self.assertEqual(rank['cuda_memory_peak_bytes']['max_allocated'], 30)
        self.assertEqual(rank['graph_capture_gb'], .75)

    def test_missing_tp_rank_never_passes(self):
        self.records = [r for r in self.records if r['tp_rank'] == 0]
        result = self.audit()
        self.assertEqual(result['status'], 'incomplete')
        self.assertTrue(any('tp=1' in x for x in result['missing_evidence']))

    def test_final_kv_request_or_slot_leaks_fail(self):
        for key in ('free_kv', 'free_reqs', 'free_receive_slots'):
            with self.subTest(key=key):
                self.records[-1][key] -= 1
                self.assertEqual(self.audit()['status'], 'failed')
                self.records[-1][key] += 1

    def test_startup_only_records_cannot_certify_final_recovery(self):
        self.records = self.records[:4]
        result = self.audit()
        self.assertEqual(result['status'], 'incomplete')
        self.assertTrue(any('final' in x for x in result['missing_evidence']))

    def test_api_capacity_snapshot_is_not_conservation_evidence(self):
        self.records = []
        (self.root / 'worker0/server-after.json').write_text(json.dumps({
            'internal_states': [{'memory_usage': {'token_capacity': 100, 'graph': .75}}]}))
        result = self.audit()
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(result['snapshots'][0]['memory_usage'][0]['token_capacity'], 100)

    def test_oom_watchdog_and_malformed_audit_are_failures(self):
        for text in ('torch.OutOfMemoryError: CUDA out of memory',
                     'Watchdog timeout (self.watchdog_timeout=180)',
                     'A100 resource audit {bad json}',
                     'A100 resource audit {"pp_rank":0,"tp_rank":0,"time_unix":201}'):
            with self.subTest(text=text):
                self.assertEqual(self.audit(text)['status'], 'failed')

    def test_nccl_error_handling_configuration_is_not_a_runtime_error(self):
        result = self.audit('NCCL INFO NCCL_ASYNC_ERROR_HANDLING=1')
        self.assertEqual(result['status'], 'passed')

    def test_pd_pair_markers_cover_both_worker_logs(self):
        phasepath = self.root / 'worker0/resource-phase-markers.json'
        phases = phasepath.read_text()
        phasepath.unlink()
        (self.root / 'pair0').mkdir()
        (self.root / 'pair0/resource-phase-markers.json').write_text(phases)
        (self.root / 'pair0/pair.json').write_text(json.dumps({'prefill': {'index': 0}, 'decode': {'index': 1}}))
        self.assertEqual(self.audit()['status'], 'passed')

    def test_pd_pair_and_worker_phases_are_both_required(self):
        (self.root / 'pair0').mkdir()
        (self.root / 'pair0/pair.json').write_text(json.dumps({'prefill': {'index': 0}, 'decode': {'index': 1}}))
        (self.root / 'pair0/resource-phase-markers.json').write_text(json.dumps([
            {'phase': 'after-stress', 'start_unix': 150, 'end_unix': 152}]))
        result = self.audit()
        self.assertEqual(result['status'], 'incomplete')
        self.assertTrue(any('pair0/after-stress' in p for p in result['missing_evidence']))

    def test_noncompleted_or_trace_without_post_trace_phase_is_incomplete(self):
        path = self.root / 'run.json'
        metadata = json.loads(path.read_text())
        metadata['status'] = 'running'
        path.write_text(json.dumps(metadata))
        self.assertEqual(self.audit()['status'], 'incomplete')
        metadata['status'] = 'execution_complete'
        metadata['args']['suite'] = 'trace'
        path.write_text(json.dumps(metadata))
        marker = self.root / 'worker0/resource-phase-markers.json'
        marker.write_text(json.dumps([p for p in json.loads(marker.read_text()) if p['phase'] != 'run-final']))
        self.assertEqual(self.audit()['status'], 'incomplete')

    def test_initial_only_phase_is_not_final_recovery(self):
        path = self.root / 'worker0/resource-phase-markers.json'
        path.write_text(json.dumps([{'phase': 'initial', 'start_unix': 100, 'end_unix': 102}]))
        self.assertEqual(self.audit()['status'], 'incomplete')

    def test_shutdown_pending_requests_fail_unless_they_subsequently_drain(self):
        suffix = '\n'.join('Gracefully exiting... remaining number of requests 11' for _ in range(4))
        result = self.audit(suffix)
        self.assertEqual(result['status'], 'failed')
        self.assertTrue(any('tokenizer' in p.lower() for p in result['errors']))
        self.assertEqual(self.audit(suffix + '\nGracefully exiting... remaining number of requests 0')['status'], 'passed')

    def test_final_tokenizer_api_count_must_be_zero(self):
        path = self.root / 'worker0/server-final.json'
        path.write_text(json.dumps({'tokenizer_pending_requests': 11}))
        self.assertEqual(self.audit()['status'], 'failed')
        path.write_text('{}')
        self.assertEqual(self.audit()['status'], 'incomplete')

    def test_post_flush_only_cannot_certify_cancel_stress_or_final_recovery(self):
        path = self.root / 'run.json'
        metadata = json.loads(path.read_text())
        metadata['args'].update(suite='correctness', stress200=True)
        path.write_text(json.dumps(metadata))
        marker = self.root / 'worker0/resource-phase-markers.json'
        marker.write_text(json.dumps([{'phase': p, 'start_unix': 100, 'end_unix': 102}
                                     for p in ('after-cancel', 'after-stress', 'run-final')]))
        result = self.audit()
        self.assertEqual(result['status'], 'incomplete')
        for phase in ('after-cancel-pre-flush', 'after-stress-pre-flush', 'run-final-pre-flush'):
            self.assertTrue(any(phase in p for p in result['missing_evidence']), phase)

    def test_gpu_monitor_peak_is_sampled_memory_not_allocator_peak(self):
        (self.root / 'gpu-monitor.csv').write_text('timestamp, index, uuid, memory.used [MiB]\n'
                                                 '2026/11/14 00:00:00, 0, GPU-a, 200 MiB\n'
                                                 '2026/11/14 00:00:01, 0, GPU-a, 350 MiB\n')
        result = self.audit()
        self.assertEqual(result['gpu_memory_peak_mib'], {'GPU-a': 350.0})


if __name__ == '__main__':
    unittest.main()
