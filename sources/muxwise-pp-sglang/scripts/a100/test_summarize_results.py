"""CPU checks for observed stream gaps and existing trial filtering."""

import importlib.util
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    "summarize_results", Path(__file__).with_name("summarize_results.py"))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def record(events, count=None, error=None):
    count = sum(event[1] for event in events) if count is None else count
    return {
        "error": error, "output_ids": list(range(count)),
        "actual_output_tokens": count, "requested_output_tokens": count,
        "events": [{"elapsed_s": elapsed, "new_tokens": tokens}
                   for elapsed, tokens in events],
        "ttft_s": events[0][0] if events else None, "tpot_s": .1,
        "arrival_ttft_s": .2, "e2e_s": 1.0, "arrival_e2e_s": 1.1,
    }


class StreamGapTests(unittest.TestCase):
    def test_pool_within_request_gaps_ignores_metadata_and_prompt_lookback(self):
        first = record([(10.0, 1), (10.1, 1), (10.2, 0), (10.4, 1)])
        first["events"][0].update(raw_tokens=6, prompt_lookback_dropped=5)
        second = record([(.01, 1), (.03, 1), (.09, 1)])
        result = MODULE.client_stream_gaps([first, second])
        self.assertEqual(result["client_stream_token_gap_status"], "available")
        self.assertEqual(result["client_stream_token_gap_count"], 4)
        self.assertEqual(result["client_stream_generated_event_count"], 6)
        self.assertEqual(result["client_stream_token_gap_covered_requests"], 2)
        self.assertAlmostEqual(result["client_stream_token_gap_s_p50"], .08)
        self.assertAlmostEqual(result["client_stream_token_gap_s_p95"], .27)
        self.assertAlmostEqual(result["client_stream_token_gap_s_p99"], .294)

    def test_one_batched_frame_withholds_entire_trial_percentiles(self):
        result = MODULE.client_stream_gaps([
            record([(1.0, 1), (1.2, 2)]),
            record([(1.0, 1), (1.1, 1), (1.2, 1)]),
        ])
        self.assertEqual(result["client_stream_token_gap_status"], "unavailable")
        self.assertIn("batched_generated_events", result["client_stream_token_gap_unavailable_reasons"])
        self.assertEqual(result["client_stream_batched_event_count"], 1)
        self.assertEqual(result["client_stream_token_gap_covered_requests"], 1)
        self.assertEqual(result["client_stream_token_gap_expected_count"], 4)
        self.assertIsNone(result["client_stream_token_gap_count"])
        for p in (50, 95, 99):
            self.assertIsNone(result[f"client_stream_token_gap_s_p{p}"])

    def test_incomplete_failed_and_invalid_records_cannot_claim_full_coverage(self):
        cases = [
            ([], "missing_request_records"),
            ([record([(1.0, 1), (1.1, 1)], count=3)], "incomplete_token_events"),
            ([record([(1.0, 1), (.9, 1)])], "invalid_event_timing"),
            ([record([(1.0, 1), (float("nan"), 1)])], "invalid_event_timing"),
            ([record([(1.0, 1), (1.1, 1)], error="timeout")], "failed_requests"),
            ([record([(1.0, 1)])], "no_adjacent_token_gaps"),
            ([dict(record([(1.0, 1), (1.1, 1)]), events=None)], "missing_events"),
        ]
        for records, reason in cases:
            with self.subTest(reason=reason):
                result = MODULE.client_stream_gaps(records)
                self.assertIn(reason, result["client_stream_token_gap_unavailable_reasons"])
                self.assertIsNone(result["client_stream_token_gap_s_p50"])

    def test_trial_keeps_actual_open_loop_settings_and_performance_filter(self):
        base = {
            "run": "trial", "classification": "main8b", "run_complete": True,
            "run_passed": True, "profile": False, "profile_metadata_recorded": True,
            "configured_ttft_s": 2.0, "configured_tpot_s": .05,
            "main_ttft_s": 2.0, "main_tpot_s": .05,
            "request_rate": 0, "concurrency": 32,
        }
        client = {
            "suite": "perf", "request_rate": 8, "concurrency": 64, "passed": True,
            "aggregate": {"wall_s": 2.0},
        }
        records = [record([(.1, 1), (.3, 1), (.7, 1)])]
        issues = []
        row = MODULE.trial_row(base, client, Path("perf-openloop-1.json"), 1, {}, records, issues)
        self.assertEqual((row["request_rate"], row["concurrency"], row["repetition"]), (8, 64, 1))
        self.assertTrue(row["performance_plot_eligible"])
        self.assertEqual(row["tpot_s_p50"], .1)
        self.assertAlmostEqual(row["client_stream_token_gap_s_p50"], .3)
        self.assertIn("transport", row["client_stream_token_gap_scope"])
        self.assertFalse(issues)
        warmup = MODULE.trial_row(base, {**client, "suite": "warmup"}, Path("warmup.json"),
                                  0, {}, records, [])
        profile = MODULE.trial_row({**base, "profile": True}, client, Path("perf.json"),
                                   0, {}, records, [])
        self.assertFalse(warmup["performance_plot_eligible"])
        self.assertFalse(profile["performance_plot_eligible"])

    def test_perf_purpose_filters_main_tables_but_retains_trial_data(self):
        with tempfile.TemporaryDirectory() as directory:
            for purpose in (None, "main", "mechanism", "diagnostic", "correctness", "pilot"):
                with self.subTest(purpose=purpose):
                    launch = {"model": "Llama-3.1-8B-Instruct", "profile": False,
                              "experiment_purpose": purpose}
                    base, _, _ = MODULE.metadata(Path(directory), launch, {"passed": True}, {}, [])
                    base["run_complete"] = True
                    client = {"suite": "perf", "passed": True, "request_rate": 8,
                              "concurrency": 64, "aggregate": {"wall_s": 1.0}}
                    row = MODULE.trial_row(base, client, Path("perf-0.json"), 0, {},
                                           [record([(.1, 1), (.2, 1)])], [])
                    self.assertEqual(row["experiment_purpose"], purpose)
                    self.assertEqual(row["performance_plot_eligible"], purpose in (None, "main"))
                    self.assertEqual(row["actual_output_tokens"], 2)
                    self.assertEqual(row["client_stream_token_gap_status"], "available")

    def test_current_protocol_supersedes_old_main_without_dropping_trial_data(self):
        with tempfile.TemporaryDirectory() as directory:
            current_runs = ["v2-main-cpp"]
            for name, active in (("main-cpp", False), ("v2-main-cpp", True)):
                with self.subTest(run=name):
                    launch = {"model": "Llama-3.1-8B-Instruct", "profile": False,
                              "experiment_purpose": "main"}
                    base, _, _ = MODULE.metadata(Path(directory)/name, launch, {"passed": True}, {}, [],
                                                 current_main_runs=current_runs)
                    base["run_complete"] = True
                    client = {"suite": "perf", "passed": True, "aggregate": {"wall_s": 1.0}}
                    row = MODULE.trial_row(base, client, Path("perf-0.json"), 0, {},
                                           [record([(.1, 1), (.2, 1)])], [])
                    self.assertEqual(row["main_protocol_whitelisted"], active)
                    self.assertEqual(row["superseded"], not active)
                    self.assertEqual(row["performance_plot_eligible"], active)
                    self.assertEqual(row["actual_output_tokens"], 2)


if __name__ == "__main__":
    unittest.main()
