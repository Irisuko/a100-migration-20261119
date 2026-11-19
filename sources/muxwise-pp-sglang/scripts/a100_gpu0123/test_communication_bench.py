"""CPU checks for communication plans and the absence of host waits in capture."""
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).with_name('communication_bench.py')


class CommunicationBenchTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(PATH.exists(), 'communication_bench.py is missing')
        spec = importlib.util.spec_from_file_location('communication_bench', PATH)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_topologies_use_only_fixed_four_ranks_and_cover_pp_boundaries(self):
        plans = self.module.plans()
        self.assertEqual(plans['tp4_allreduce'], {'kind': 'allreduce', 'groups': [[0, 1, 2, 3]]})
        self.assertEqual(plans['tp2_allreduce']['groups'], [[0, 1], [2, 3]])
        self.assertEqual(plans['pp2tp2_p2p']['groups'], [[0, 2], [1, 3]])
        for pair in ([0, 1], [1, 2], [2, 3]):
            self.assertIn({'kind': 'p2p', 'groups': [pair]}, plans.values())
        for plan in plans.values():
            ranks = [r for group in plan['groups'] for r in group]
            self.assertEqual(len(ranks), len(set(ranks)))
            self.assertTrue(set(ranks) <= {0, 1, 2, 3})

    def test_enqueue_dependency_uses_stream_synchronize_not_timed_host_wait(self):
        class Work:
            synced = False
            def synchronize(self): self.synced = True
            def wait(self, *args, **kwargs): raise AssertionError('host wait forbidden')
        work = Work()
        self.module.enqueue_dependency(work)
        self.assertTrue(work.synced)

    def test_gpu_bandwidth_and_ring_factor_use_measured_gpu_ms(self):
        row = self.module.summarize([1.0, 2.0, 3.0], 4_000_000, 4, 'allreduce')
        self.assertEqual(row['median_gpu_ms'], 2)
        self.assertEqual(row['algorithmic_GB_per_s'], 2)
        self.assertEqual(row['ring_bus_GB_per_s_estimate'], 3)
        self.assertEqual(row['samples'], 3)
        self.assertNotIn('ring_bus_GB_per_s_estimate', self.module.summarize([2.0], 4_000_000, 2, 'p2p'))

    def test_zero_negative_or_nonfinite_timing_is_rejected(self):
        for values in ([], [0.0], [-1], [float('nan')], [float('inf')]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.module.summarize(values, 8, 4, 'allreduce')

    def test_dry_configuration_rejects_other_gpu_masks_without_torch(self):
        env = dict(CUDA_VISIBLE_DEVICES='GPU-4,GPU-5,GPU-6,GPU-7', RANK='0', LOCAL_RANK='0',
                   WORLD_SIZE='4', LOCAL_WORLD_SIZE='4')
        with self.assertRaises(ValueError): self.module.launch_config(env)


if __name__ == '__main__':
    unittest.main()
