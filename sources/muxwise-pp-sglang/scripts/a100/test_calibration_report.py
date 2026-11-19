import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("calibration_report", Path(__file__).with_name("calibration_report.py"))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def run(name, partitions=(1,)):
    rows = []
    for partition in partitions:
        for stage in (0, 1):
            for batch in (1, 4):
                for context in (512, 1024):
                    rows.append(dict(stage=stage, partition=partition, prefill_sms=84,
                                     decode_sms=24, batch_size=batch, context_tokens=context,
                                     seconds=.005 if stage == 0 else .010))
    return {"name": name, "group": "same", "rows": rows,
            "settings": {"command_settings": {"--pp-size": "2"}}}


class CalibrationReportTests(unittest.TestCase):
    def test_upper_corner_monotonic_envelope_and_unknown_coverage(self):
        records = [(1, 512, .006), (1, 1024, .008), (4, 512, .010), (4, 1024, .007)]
        self.assertEqual(MODULE.envelope(records, 2, 800), .010)
        self.assertIsNone(MODULE.envelope(records, 5, 512))
        self.assertIsNone(MODULE.envelope(records, 1, 2048))

    def test_policy_gate_is_twice_slowest_stage_not_stage_sum(self):
        _, individual, _, safety = MODULE.aggregate([run("one")], contexts=(512,), batches=(1,), targets_ms=(22, 27))
        self.assertAlmostEqual(individual[0]["sum_stage_initial_seconds"], .020)
        self.assertAlmostEqual(individual[0]["both_stages_initial_gate_seconds"], .026)
        self.assertEqual(safety[0]["both_stages_safe_all_startups"], [])
        self.assertEqual(safety[1]["both_stages_safe_all_startups"], [1])

    def test_missing_partition_in_startup_cannot_be_all_startups_safe(self):
        _, _, _, safety = MODULE.aggregate([run("one", (1, 2)), run("two", (1,))], contexts=(512,), batches=(1,), targets_ms=(50,))
        self.assertEqual(safety[0]["both_stages_safe_all_startups"], [1])

    def test_duplicate_startup_has_zero_population_variation(self):
        measured, _, _, _ = MODULE.aggregate([run("one"), run("two")], contexts=(512,), batches=(1,))
        self.assertEqual(measured[0]["seconds_count"], 2)
        self.assertEqual(measured[0]["seconds_cv_population"], 0)

    def test_threshold_candidate_skips_unsafe_partition_without_removing_it(self):
        inputs = [{"group": "same", "settings": {
            "command_settings": {"--max-micro-batch-size": "16", "--context-length": "8192"},
            "sm_pairs": [[100, 8], [92, 16], [84, 24], [76, 32], [60, 48], [52, 56]]}}]
        safety = [{"group": "same", "context_tokens": 8192, "target_ms": 50,
                   "batch_size": b, "max_prefill_safe_partition_all_startups": 2 if b <= 4 else 3}
                  for b in (1, 4, 8, 16)]
        result = MODULE.fallback_candidates(inputs, safety)[0]
        self.assertEqual([row[2] for row in result["manual_divisions_candidate"]], [1, 1, 5, 17, 17, 17])
        for batch in range(1, 17):
            chosen = 1
            for index, (_, _, threshold) in enumerate(result["manual_divisions_candidate"], start=1):
                if batch >= threshold:
                    chosen = index
            self.assertEqual(chosen, 2 if batch <= 4 else 3)


if __name__ == "__main__":
    unittest.main()
