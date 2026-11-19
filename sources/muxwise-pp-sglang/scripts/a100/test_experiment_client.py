"""CPU-only tests for native SSE parsing and observed-token latency accounting."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiment_client import (
    StreamAccumulator, aggregate, iter_sse, percentile, run, validate_cache_requests,
)


class StreamTests(unittest.TestCase):
    def test_sse_comments_multiline_and_done(self):
        lines = [b": keepalive\n", b"event: message\n", b"data: {\"a\":\n",
                 b"data: 1}\n", b"\n", b"data: [DONE]\n", b"\n"]
        self.assertEqual(list(iter_sse(lines)), ['{"a":\n1}', "[DONE]"])

    def test_multiple_tokens_in_event_use_total_token_span(self):
        stream = StreamAccumulator()
        stream.feed({"output_ids": [10, 11], "meta_info": {"prompt_tokens": 5, "completion_tokens": 2}}, 0.1)
        stream.feed({"output_ids": [12, 13], "meta_info": {
            "completion_tokens": 4, "cached_tokens": 0,
            "finish_reason": {"type": "length", "length": 4}}}, 0.4)
        result = stream.finish(True, 0.5, 5, 4)
        self.assertAlmostEqual(result["ttft_s"], 0.1)
        self.assertAlmostEqual(result["tpot_s"], 0.1)
        self.assertAlmostEqual(result["e2e_s"], 0.5)
        self.assertEqual(result["output_ids"], [10, 11, 12, 13])
        self.assertEqual(result["events"][1]["new_tokens"], 2)

    def test_missing_done_and_short_output_fail(self):
        stream = StreamAccumulator()
        stream.feed({"output_ids": [10], "meta_info": {
            "prompt_tokens": 5, "completion_tokens": 1, "cached_tokens": 0,
            "finish_reason": {"type": "length"}}}, 0.1)
        with self.assertRaisesRegex(ValueError, "DONE"):
            stream.finish(False, 0.2, 5, 1)
        with self.assertRaisesRegex(ValueError, "output length"):
            stream.finish(True, 0.2, 5, 4)

    def test_nan_and_abort_fail(self):
        stream = StreamAccumulator()
        with self.assertRaisesRegex(ValueError, "time"):
            stream.feed({"output_ids": [1]}, float("nan"))
        with self.assertRaisesRegex(ValueError, "server error"):
            stream.feed({"error": {"message": "failed"}}, 0.2)

    def test_native_first_frame_validates_and_removes_prompt_lookback(self):
        prompt = [1, 2, 465, 22137, 29989, 29958, 13]
        stream = StreamAccumulator(prompt)
        stream.feed({"output_ids": prompt[-5:] + [1576], "meta_info": {
            "prompt_tokens": len(prompt), "completion_tokens": 1}}, 0.1)
        stream.feed({"output_ids": [2060], "meta_info": {
            "completion_tokens": 2, "cached_tokens": 0,
            "finish_reason": {"type": "length", "length": 2}}}, 0.2)
        result = stream.finish(True, 0.3, len(prompt), 2)
        self.assertEqual(result["output_ids"], [1576, 2060])
        self.assertEqual(result["raw_output_ids"], prompt[-5:] + [1576, 2060])
        self.assertEqual(result["prompt_lookback_dropped_ids"], prompt[-5:])
        self.assertEqual(result["prompt_lookback_dropped_count"], 5)
        self.assertEqual(result["events"][0]["new_tokens"], 1)
        self.assertEqual(result["events"][0]["raw_tokens"], 6)
        self.assertAlmostEqual(result["tpot_s"], 0.1)

    def test_wrong_lookback_and_later_token_delta_mismatches_fail(self):
        prompt = [1, 2, 3, 4, 5, 6]
        stream = StreamAccumulator(prompt)
        with self.assertRaisesRegex(ValueError, "prompt suffix"):
            stream.feed({"output_ids": [99, 3, 4, 5, 6, 10],
                         "meta_info": {"completion_tokens": 1}}, 0.1)
        for later_ids in ([], [11, 12]):
            with self.subTest(later_ids=later_ids):
                stream = StreamAccumulator(prompt)
                stream.feed({"output_ids": prompt[-5:] + [10],
                             "meta_info": {"completion_tokens": 1}}, 0.1)
                with self.assertRaisesRegex(ValueError, "completion delta"):
                    stream.feed({"output_ids": later_ids,
                                 "meta_info": {"completion_tokens": 2}}, 0.2)

    def test_percentile_interpolates(self):
        self.assertEqual(percentile([1, 2, 3, 4], 50), 2.5)
        self.assertAlmostEqual(percentile([1, 2, 3, 4], 95), 3.85)
        self.assertIsNone(percentile([], 50))

    def test_cache_checks_reject_superficial_hits(self):
        records = [
            {"fixture_id": "cache-base", "input_tokens": 2048, "output_ids": [10, 11],
             "cached_tokens": 0, "error": None},
            {"fixture_id": "cache-base", "input_tokens": 2048, "output_ids": [10, 11],
             "cached_tokens": 1, "error": None},
            {"fixture_id": "cache-branch", "input_tokens": 2048, "output_ids": [12, 13],
             "cached_tokens": 1791, "error": None},
        ]
        validate_cache_requests(records)
        self.assertIsNone(records[0]["error"])
        self.assertIn("2047", records[1]["error"])
        self.assertIn("1792", records[2]["error"])

    def test_cache_accepts_full_hits_but_requires_exact_repeated_outputs(self):
        records = [
            {"fixture_id": "cache-base", "input_tokens": 2048, "output_ids": [10, 11],
             "cached_tokens": 0, "error": None},
            {"fixture_id": "cache-base", "input_tokens": 2048, "output_ids": [10, 11],
             "cached_tokens": 2047, "error": None},
            {"fixture_id": "cache-branch", "input_tokens": 2048, "output_ids": [12, 13],
             "cached_tokens": 1792, "error": None},
            {"fixture_id": "cache-branch", "input_tokens": 2048, "output_ids": [12, 14],
             "cached_tokens": 2047, "error": None},
        ]
        validate_cache_requests(records)
        self.assertTrue(all(record["error"] is None for record in records[:3]))
        self.assertIn("changed output_ids", records[3]["error"])

    def test_slo_counts_failures_and_distinguishes_client_queue(self):
        success = {"error": None, "actual_output_tokens": 4, "input_tokens": 5,
                   "cached_tokens": 0, "ttft_s": 0.1, "tpot_s": 0.01, "e2e_s": 0.5,
                   "queue_delay_s": 3.0, "arrival_ttft_s": 3.1, "arrival_e2e_s": 3.5}
        result = aggregate([success, {"error": "timeout"}], 4)
        self.assertEqual(result["slo"]["ttft_met_fraction"], 0.5)
        self.assertEqual(result["slo"]["arrival_ttft_met_fraction"], 0.0)
        self.assertEqual(aggregate([{"error": "timeout"}], 4)["slo"]["both_met_fraction"], 0.0)

    def test_invocation_namespaces_preserve_trial_order_and_unique_cache_requests(self):
        ids = [1] * 2048
        case = {"input_ids": ids, "input_tokens": len(ids), "max_new_tokens": 2,
                "input_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
        fixture = {"schema_version": 1, "cases": {"cache-base": case, "cache-branch": case},
                   "suites": {"cache": ["cache-base", "cache-base", "cache-branch", "cache-branch", "cache-base"]}}

        def fake_request(base_url, requested_case, record, trial_start, timeout):
            return {**record, "error": None, "output_ids": [7, 8], "actual_output_tokens": 2,
                    "cached_tokens": 2047, "ttft_s": 0.1, "tpot_s": 0.1, "e2e_s": 0.2,
                    "queue_delay_s": 0.0, "arrival_ttft_s": 0.1, "arrival_e2e_s": 0.2}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixtures = root / "fixtures.json"
            fixtures.write_text(json.dumps(fixture))
            reports = []
            with patch("experiment_client.send_request", side_effect=fake_request), contextlib.redirect_stdout(io.StringIO()):
                for index in range(2):
                    args = SimpleNamespace(fixtures=fixtures, suite="cache", order="fixture", concurrency=16,
                                           base_url="http://unused", seed=42, repeat=2, request_rate=0,
                                           timeout=1, output=root / f"result-{index}.json")
                    self.assertEqual(run(args), 0)
                    reports.append(json.loads(args.output.read_text()))
            self.assertNotEqual(reports[0]["request_namespace"], reports[1]["request_namespace"])
            for report in reports:
                request_ids = [record["request_id"] for record in report["requests"]]
                self.assertEqual(len(request_ids), len(set(request_ids)))
                self.assertTrue(all(rid.startswith(report["request_namespace"] + ":") for rid in request_ids))
                self.assertEqual([record["repeat_index"] for record in report["requests"]], [0] * 5 + [1] * 5)
                self.assertEqual(report["request_order"], fixture["suites"]["cache"])
                self.assertEqual(len(report["client_source_sha256"]), 64)
            self.assertEqual([r["request_id"].split(":", 1)[1] for r in reports[0]["requests"]],
                             [r["request_id"].split(":", 1)[1] for r in reports[1]["requests"]])


if __name__ == "__main__":
    unittest.main()
