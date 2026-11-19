"""CPU-only diagnostic checks with native /generate response fixtures."""

import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

import diagnose_differences as diagnostic


def result_row(tokens, request_id="r0", error=None):
    return {"fixture_id": "case", "request_id": request_id, "output_ids": tokens,
            "requested_output_tokens": 2, "actual_output_tokens": len(tokens), "error": error}


def response():
    return {"text": "ignored", "meta_info": {
        "prompt_tokens": 4, "completion_tokens": 1,
        "finish_reason": {"type": "length", "length": 1},
        "input_token_logprobs": [[None, 20, None], [-0.7, 31, None], [-0.1, 41, None]],
        "input_top_logprobs": [None,
            [[-0.2, 30, None], [-0.7, 31, None], [-2.0, 32, None], [-3.0, 33, None], [-4.0, 34, None]],
            [[-0.1, 41, None], [-1.0, 40, None], [-2.0, 42, None], [-3.0, 43, None], [-4.0, 44, None]]],
        "output_token_logprobs": [[-0.1, 51, None]],
        "output_top_logprobs": [[[-0.1, 51, None]]],
    }}


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.module = diagnostic

    def test_scores_use_candidate_prefix_and_skip_only_anchor(self):
        parsed = self.module.parse_logprobs(response(), [10, 20], [31, 41])
        self.assertEqual([row["chosen"] for row in parsed["tokens"]], [31, 41])
        self.assertAlmostEqual(parsed["tokens"][0]["regret"], 0.5)
        self.assertAlmostEqual(parsed["tokens"][0]["top_two_margin"], 0.5)
        self.assertEqual(parsed["tokens"][0]["oracle_best_token"], 30)
        self.assertEqual(parsed["oracle_greedy_matches"], 1)
        self.assertEqual(parsed["first_oracle_divergence"], 0)
        self.assertFalse(parsed["all_tokens_oracle_greedy"])

    def test_misaligned_missing_and_nonfinite_scores_are_rejected(self):
        mutations = [
            lambda meta: meta["input_token_logprobs"][0].__setitem__(0, -1.0),
            lambda meta: meta["input_token_logprobs"][0].__setitem__(1, 99),
            lambda meta: meta["input_token_logprobs"][1].__setitem__(1, 99),
            lambda meta: meta["input_token_logprobs"].pop(),
            lambda meta: meta["input_token_logprobs"][1].__setitem__(0, float("nan")),
            lambda meta: meta["input_top_logprobs"][1].pop(),
            lambda meta: meta["input_top_logprobs"][1].reverse(),
            lambda meta: meta.pop("input_top_logprobs"),
            lambda meta: meta.__setitem__("prompt_tokens", 99),
            lambda meta: meta.__setitem__("finish_reason", {"type": "abort", "message": "unsupported"}),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                data = response()
                mutate(data["meta_info"])
                with self.assertRaises(ValueError):
                    self.module.parse_logprobs(data, [10, 20], [31, 41])

    def test_occurrences_match_reference_and_request_uses_actual_continuation(self):
        fixtures = {"cases": {"case": {"input_ids": [10, 20]}}}
        reference = {"requests": [result_row([31, 41]), result_row([30, 40], "r1")]}
        candidate = {"requests": [result_row([31, 41]), result_row([31, 41], "r1")]}
        calls = []

        def api(path, payload=None):
            calls.append((path, payload))
            return {"enable_pdmux": False, "pp_size": 2} if payload is None else response()

        report = self.module.diagnose(fixtures, reference, candidate, api, max_cases=12)
        self.assertEqual(len(report["checks"]), 1)
        self.assertEqual(report["checks"][0]["occurrence"], 1)
        self.assertEqual(report["checks"][0]["first_divergence"], 0)
        self.assertFalse(report["comparison"]["passed"])
        self.assertTrue(report["completed"])
        self.assertEqual(calls[1], ("/generate", {
            "input_ids": [10, 20, 31, 41], "return_logprob": True,
            "logprob_start_len": 1, "top_logprobs_num": 5,
            "sampling_params": {"max_new_tokens": 1, "ignore_eos": True, "temperature": 0},
            "stream": False,
        }))

    def test_missing_failed_and_truncated_coverage_stays_explicit(self):
        fixtures = {"cases": {"case": {"input_ids": [10, 20]}}}
        reference = {"requests": [result_row([30, 40], str(i)) for i in range(4)]}
        candidate = {"requests": [result_row([31, 41], "0"), result_row([31, 41], "1"),
                                  result_row([], "2", "timeout")]}
        report = self.module.diagnose(fixtures, reference, candidate,
            lambda path, payload=None: {"enable_pdmux": False} if payload is None else response(),
            max_cases=1)
        self.assertEqual(report["comparison"]["missing_fixtures"], {"case": 1})
        self.assertEqual(report["unselected_mismatch_count"], 1)
        self.assertEqual(len(report["invalid_cases"]), 1)
        self.assertFalse(report["full_mismatch_coverage"])
        self.assertFalse(report["completed"])

    def test_pdmux_is_rejected_before_generation(self):
        calls = []

        def api(path, payload=None):
            calls.append(path)
            return {"enable_pdmux": True}

        report = self.module.diagnose({"cases": {"case": {"input_ids": [10, 20]}}},
            {"requests": [result_row([30, 40])]}, {"requests": [result_row([31, 41])]}, api)
        self.assertEqual(calls, ["/get_server_info"])
        self.assertFalse(report["completed"])
        self.assertIn("PDMux", report["errors"][0])

    def test_hashes_are_checked_against_actual_fixture_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            fixtures = {"cases": {"case": {"input_ids": [10, 20], "input_tokens": 2,
                "input_sha256": hashlib.sha256(b"[10, 20]").hexdigest()}}}
            paths = [directory / name for name in ("fixtures.json", "ref.json", "candidate.json")]
            paths[0].write_text(json.dumps(fixtures))
            data = {"fixture_sha256": hashlib.sha256(paths[0].read_bytes()).hexdigest(),
                    "requests": [result_row([31, 41])]}
            for path in paths[1:]:
                path.write_text(json.dumps(data))
            self.module.load_inputs(*paths)
            data["requests"][0]["fixture_id"] = "unknown"
            paths[2].write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "Unknown fixture"):
                self.module.load_inputs(*paths)
            data["requests"][0]["fixture_id"] = "case"
            data["requests"][0]["output_ids"] = [31, float("nan")]
            paths[2].write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "output_ids"):
                self.module.load_inputs(*paths)
            data["requests"][0]["output_ids"] = [31, 41]
            paths[2].write_text(json.dumps(data))
            paths[0].write_text(paths[0].read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                self.module.load_inputs(*paths)

    def test_http_errors_keep_api_message_and_bypass_proxy_environment(self):
        error = urllib.error.HTTPError("http://localhost/generate", 400, "Bad Request", {},
                                      io.BytesIO(b'{"error":"PDMux does not support logprobs"}'))
        with patch.object(self.module.urllib.request, "build_opener") as build:
            build.return_value.open.side_effect = error
            with self.assertRaisesRegex(RuntimeError, "HTTP 400.*PDMux"):
                self.module.api_json("http://localhost", "/generate", {"input_ids": [1]})
            self.assertEqual(build.call_args.args[0].proxies, {})

    def test_cli_persists_hash_errors_without_contacting_server(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            fixtures, reference, candidate, output = [directory / name for name in (
                "fixtures.json", "reference.json", "candidate.json", "diagnostic.json")]
            fixtures.write_text('{"cases": {}}')
            for path in (reference, candidate):
                path.write_text('{"fixture_sha256": "wrong", "requests": []}')
            arguments = ["diagnose_differences.py", "--fixtures", str(fixtures),
                         "--reference", str(reference), "--candidate", str(candidate),
                         "--output", str(output), "--base-url", "http://localhost:1"]
            with patch("sys.argv", arguments), patch.object(self.module, "api_json", side_effect=AssertionError("network forbidden")):
                self.assertEqual(self.module.main(), 2)
            report = json.loads(output.read_text())
            self.assertFalse(report["completed"])
            self.assertIn("SHA256", report["errors"][0])


if __name__ == "__main__":
    unittest.main()
