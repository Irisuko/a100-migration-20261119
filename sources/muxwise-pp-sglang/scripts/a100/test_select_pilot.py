import copy
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("select_pilot", Path(__file__).with_name("select_pilot.py"))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def valid_trial():
    names = [f"pilot-{i:02d}" for i in range(24)]
    fixture = {"suites": {"pilot": names}, "cases": {name: {"input_tokens": 8, "max_new_tokens": 2} for name in names}}
    rows = [{"fixture_id": name, "request_id": name, "error": None, "input_tokens": 8,
             "requested_output_tokens": 2, "output_ids": [1, 2], "actual_output_tokens": 2,
             "sse_done": True, "ttft_s": .5, "tpot_s": .02} for name in names]
    client = {"passed": True, "suite": "pilot", "fixture_sha256": "hash", "repeat": 1,
              "seed": 20261114, "request_rate": 0, "concurrency": 32, "order_mode": "fixture",
              "request_order": names, "cache_reset": True, "external_cache_reset": {"http_status": 200},
              "requests": rows, "aggregate": {"failed": 0, "completed": 24, "actual_output_tokens": 48,
                  "wall_s": 2, "output_tokens_per_s": 24, "ttft_s": {"p95": .5}, "tpot_s": {"p95": .02},
                  "slo": {"ttft_target_s": 2, "tpot_target_s": .05, "both_met_fraction": 1.0}}}
    return client, fixture


class PilotSelectionTests(unittest.TestCase):
    def test_slo_wins_over_faster_configuration_then_throughput_breaks_tie(self):
        def row(name, slo, speed):
            return {"run": name, "chunk_tokens": 1024, "slice_token_layer_budget": 8192,
                    "joint_slo_fraction": {"mean": slo}, "output_tokens_per_s": {"median": speed}}
        rows = [row("fast-but-slo-miss", .75, 2000), row("winner", .9, 900), row("slower", .9, 800)]
        self.assertEqual([r["run"] for r in sorted(rows, key=MODULE.ranking_key)], ["winner", "slower", "fast-but-slo-miss"])

    def test_truncated_or_failed_request_cannot_enter_ranking(self):
        for mutation in (lambda d: d["requests"][0].update(output_ids=[1]),
                         lambda d: d["requests"][0].update(error="timeout"),
                         lambda d: d["aggregate"]["slo"].update(both_met_fraction=.5)):
            client, fixture = valid_trial()
            mutation(client)
            with self.assertRaises(ValueError):
                MODULE.check_trial(client, fixture, "hash", 0)

    def test_complete_paired_trial_is_accepted(self):
        client, fixture = valid_trial()
        result = MODULE.check_trial(client, fixture, "hash", 0)
        self.assertEqual(result["joint_slo_fraction"], 1)
        self.assertEqual(result["output_tokens_per_s"], 24)
        with self.assertRaisesRegex(ValueError, "seed"):
            MODULE.check_trial(client, fixture, "hash", 1)

    def test_cohort_only_ignores_chunk_and_budget(self):
        launch = {"command": ["python", "--pdmux-config-path", "one.yml", "--pdmux-pp-prefill-chunk-size", "256", "--dtype", "bfloat16"],
                  "config": {"split_forward_token_budget": 2048, "manual_divisions": [[92, 16, 1]]},
                  "model": "8b", "environment": {"CUDA_VISIBLE_DEVICES": "6,7"},
                  "runtime_source_sha256": {"backend.py": "fixed"}, "repository_commit": "commit"}
        other = copy.deepcopy(launch)
        other["command"][2], other["command"][4] = "two.yml", "1024"
        other["config"]["split_forward_token_budget"] = 8192
        self.assertEqual(MODULE.cohort_signature(launch), MODULE.cohort_signature(other))
        other["runtime_source_sha256"]["backend.py"] = "unfixed"
        self.assertNotEqual(MODULE.cohort_signature(launch), MODULE.cohort_signature(other))


if __name__ == "__main__":
    unittest.main()
