"""CPU-only tests for evidence strength and incomplete experiment handling."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parent))
SPEC = importlib.util.spec_from_file_location(
    "mechanism_report", Path(__file__).with_name("mechanism_report.py")
)
reporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporter)


def stage(samples=(16, 15), finals=20, admitted=20):
    return {
        "sm_counts": [[108, 0], [92, 16], [0, 108]], "startup_count": 1,
        "idle_counter_max": {"prefill_slices": 40, "overlap_launches": 8},
        "latest_telemetry": {
            "samples": list(samples), "mape": [.1, .2],
            "by_partition": [[0, 0], [32, 99], [10, 0]],
            "slice_choices": [4, 0, 0],
            "timing": {"partitions": {"0": 2, "1": 4, "2": 4},
                       "slices": {"prefill_first": [10, .1],
                                  "prefill_final": [finals, .5]}},
            "admission": {"decisions": {"admitted": admitted},
                          "blocked": {}, "stops": {"inflight_tokens": 3}},
        },
    }


class MechanismReportTest(unittest.TestCase):
    def test_model_readiness_uses_batch_counts_and_does_not_imply_alp(self):
        result = reporter.summarize_stage("0", stage(), [], "static")
        self.assertTrue(result["rls"]["decode_ready"])
        self.assertFalse(result["rls"]["prefill_ready"])
        self.assertFalse(result["rls"]["adaptive_policy_selected"])
        self.assertEqual(result["partitions"]["mixed_wall_seconds"], 4)
        self.assertEqual(result["partitions"]["mixed_wall_fraction"], .4)
        self.assertEqual(result["layer_slicing"]["decode_aware_shortening"], "not_observed")
        self.assertFalse(result["coissued_launches"]["proves_kernel_overlap"])

    def test_missing_counters_remain_unknown_but_empty_admission_counter_is_zero(self):
        result = reporter.summarize_stage("0", stage(), [], "alp")
        self.assertEqual(result["admission"]["inflight_refill"]["blocked"], 0)
        self.assertEqual(result["admission"]["inflight_tokens"]["activation"], "observed")
        missing = reporter.summarize_stage("1", {}, [], "alp")
        self.assertIsNone(missing["rls"]["decode_ready"])
        self.assertIsNone(missing["coissued_launches"]["count"])
        self.assertEqual(missing["admission"]["inflight_tokens"]["activation"], "unknown")

    def test_chunk_count_requires_complete_drained_single_startup(self):
        observed = reporter.chunk_evidence(stage(finals=30), complete=True, drained=True)
        self.assertEqual(observed["activation"], "observed")
        self.assertEqual(observed["excess_final_slices"], 10)
        for complete, drained in [(False, True), (True, False)]:
            self.assertEqual(reporter.chunk_evidence(
                stage(finals=30), complete=complete, drained=drained
            )["activation"], "unknown")
        self.assertEqual(reporter.chunk_evidence(
            stage(), complete=True, drained=True)["activation"], "not_observed")

    def test_interior_snapshots_exclude_coarse_boundary_and_report_resets(self):
        summaries = [{"timestamp": f"2026-11-13 20:00:{second:02d}",
                      "line_number": second, "stages": {"0": {
                          "samples": [count, count], "timing": {"partitions": {"1": count}}
                      }}} for second, count in [(0, 0), (2, 4), (4, 8), (6, 12), (8, 16), (10, 20)]]
        start = reporter.timestamp_epoch("2026-11-13 20:00:00")
        result = reporter.interior_observations(summaries, "0", start, start + 10, 2)
        self.assertEqual(result["first_snapshot"]["line_number"], 2)
        self.assertEqual(result["last_snapshot"]["line_number"], 6)
        self.assertEqual(result["observed_deltas"]["rls.decode"], 8)
        self.assertFalse(result["exact_trial_attribution"])
        summaries[3]["stages"]["0"]["samples"] = [1, 1]
        result = reporter.interior_observations(summaries, "0", start, start + 10, 2)
        self.assertIsNone(result["observed_deltas"]["rls.decode"])
        self.assertIn("rls.decode", result["counter_resets"])

    def test_incomplete_matrix_preserves_missing_runs_and_baseline_not_applicable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol = {"main_runs": ["baseline", "missing"], "controlled_comparisons": {}}
            (root / "main-protocol.json").write_text(json.dumps(protocol))
            (root / "mechanism-jobs.json").write_text("[]")
            run = root / "baseline"
            run.mkdir()
            (run / "launch.json").write_text(json.dumps({"mode": "baseline", "command": []}))
            (run / "outcome.json").write_text(json.dumps({"passed": True, "runs": []}))
            (run / "server.log").write_text("normal baseline server\n")
            result = reporter.build_report(root)
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["runs"]["baseline"]["telemetry_status"], "not_applicable")
            self.assertEqual(result["runs"]["missing"]["status"], "missing")
            json.dumps(result, allow_nan=False)

    def test_configured_migration_zero_is_inactive_and_slo_uses_experiment_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "launch.json").write_text(json.dumps({"mode": "mux", "command": [
                "--pdmux-pp-slo-tpot", "0.01", "--pdmux-pp-policy", "alp"
            ], "config": {"pp_profile_decode": False}}))
            (root / "outcome.json").write_text(json.dumps({"passed": True, "runs": [{
                "suite": "pilot", "repeat": 0, "output": str(root / "pilot-0.json")
            }]}))
            (root / "pilot-0.json").write_text(json.dumps({
                "experiment_slo": {"tpot_target_s": .01, "tpot_met_fraction": .25},
                "aggregate": {"slo": {"tpot_target_s": .05, "tpot_met_fraction": 1.0}}
            }))
            (root / "scheduler.json").write_text(json.dumps({
                "metadata": {}, "stages": {"0": stage()}, "summaries": [],
                "calibration_rows": [], "calibration_coverage": {}, "idle_checks": [],
                "migrations": {"max": 0}
            }))
            result = reporter.summarize_run(root)
            self.assertEqual(result["migrations"]["activation"], "not_observed")
            self.assertTrue(result["configuration"]["rebalance"])
            self.assertEqual(result["calibration"]["expected_matrix_rows"], 0)
            self.assertEqual(result["measurements"][0]["experiment_slo"]["tpot_target_s"], .01)
            self.assertNotIn("slo", result["measurements"][0]["aggregate"]) 

    def test_missing_measurement_is_an_explicit_coverage_issue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "launch.json").write_text(json.dumps({"mode": "baseline", "command": []}))
            (root / "outcome.json").write_text(json.dumps({"passed": True, "runs": [
                {"suite": "perf", "repeat": 0, "output": str(root / "missing.json")}
            ]}))
            result = reporter.summarize_run(root)
            self.assertEqual(result["measurements"][0]["result_status"], "missing_or_unreadable")
            self.assertTrue(result["issues"])

    def test_actual_openloop_suite_keeps_arrival_and_http_metrics_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "launch.json").write_text(json.dumps({"mode": "baseline", "command": []}))
            (root / "outcome.json").write_text(json.dumps({"passed": True, "runs": [
                {"suite": "perf-openloop", "repeat": 0, "output": str(root / "perf-openloop-0.json")}
            ]}))
            (root / "perf-openloop-0.json").write_text(json.dumps({
                "experiment_slo": {"tpot_target_s": .01, "both_met_fraction": .8},
                "aggregate": {"arrival_ttft_s": {"p95": 3.0}, "queue_delay_s": {"p95": 2.0},
                              "slo": {"ttft_target_s": 2.0, "tpot_target_s": .05,
                                      "arrival_both_met_fraction": .5}}
            }))
            result = reporter.summarize_run(root)
            self.assertEqual(len(result["measurements"]), 1)
            measurement = result["measurements"][0]
            self.assertEqual(measurement["aggregate"]["arrival_ttft_s"]["p95"], 3.0)
            self.assertEqual(measurement["aggregate"]["queue_delay_s"]["p95"], 2.0)
            self.assertEqual(measurement["fixed_target_arrival_slo"]["tpot_target_s"], .05)
            self.assertEqual(measurement["experiment_slo"]["tpot_target_s"], .01)


if __name__ == "__main__":
    unittest.main()
