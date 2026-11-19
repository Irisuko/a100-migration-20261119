"""CPU-only teacher-forcing audit contracts; HTTP is replaced at its boundary."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

PATH = Path(__file__).with_name("numerical_audit.py")
audit_module = None
if PATH.exists():
    spec = importlib.util.spec_from_file_location("test_numerical_audit_module", PATH)
    audit_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit_module)


def response(tie=False):
    return {"output_ids": [51], "meta_info": {
        "prompt_tokens": 4, "completion_tokens": 1,
        "finish_reason": {"type": "length", "length": 1},
        "input_token_logprobs": [[None, 20, None], [-.2 if tie else -.7, 31, None], [-.1, 41, None]],
        "input_top_logprobs": [None,
            [[-.2, 30, None], [-.2 if tie else -.7, 31, None], [-2., 32, None], [-3., 33, None], [-4., 34, None]],
            [[-.1, 41, None], [-1., 40, None], [-2., 42, None], [-3., 43, None], [-4., 44, None]]],
        "output_token_logprobs": [[-.1, 51, None]],
    }}


class NumericalAuditTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(audit_module, "Numerical audit implementation missing")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "concurrent.json"
        self.source.write_text(json.dumps([dict(input_ids=[10, 20], output_ids=[31, 41])]))
        self.cases = self.root / "cases.json"
        self.case = dict(label="candidate-12", input_ids=[10, 20], output_ids=[31, 41],
                         source=str(self.source), source_sha256=hashlib.sha256(self.source.read_bytes()).hexdigest(), first_mismatch=0)
        self.cases.write_text(json.dumps([self.case]))
        self.output = self.root / "audit.json"
        self.expected = dict(model_path="/models/Llama-3.1-8B-Instruct", tp_size=1, pp_size=4,
                             dtype="bfloat16", enable_pdmux=False)
        self.server = dict(self.expected, context_length=33792)
        self.calls = []

    def api(self, port, path, payload=None):
        self.calls.append((port, path, payload))
        return self.server if path == "/get_server_info" else response()

    def execute(self, api=None, expected=True):
        with mock.patch.object(audit_module, "request_json", side_effect=api or self.api):
            return audit_module.audit(32100, self.cases, self.output,
                                      expected_metadata=self.expected if expected else None)

    def test_actual_candidate_prefix_payload_and_full_raw_response_are_preserved(self):
        report = self.execute()
        self.assertEqual(self.calls[1], (32100, "/generate", {
            "input_ids": [10, 20, 31, 41], "return_logprob": True, "logprob_start_len": 1,
            "top_logprobs_num": 5, "sampling_params": {"max_new_tokens": 1, "ignore_eos": True, "temperature": 0}, "stream": False}))
        self.assertTrue(report["completed"])
        self.assertTrue(report["full_token_coverage"])
        self.assertEqual(report["audited_output_tokens"], 2)
        check = report["checks"][0]
        self.assertEqual(check["raw_response"], response())
        self.assertEqual(check["original_first_mismatch"], 0)
        self.assertAlmostEqual(check["tokens"][0]["regret"], .5)
        self.assertFalse(check["all_tokens_oracle_greedy"])
        self.assertNotIn("passed", report)
        self.assertEqual(json.loads(self.output.read_text())["checks"][0]["raw_response"], response())

    def test_exact_reported_tie_is_diagnostic_and_never_converts_mismatch_to_pass(self):
        report = self.execute(lambda port, path, payload=None: self.server if payload is None else response(tie=True))
        token = report["checks"][0]["tokens"][0]
        self.assertEqual(token["reported_max_tie_ids"], [30, 31])
        self.assertEqual(token["canonical_top5_token"], 30)
        self.assertTrue(token["chosen_in_reported_max_tie"])
        self.assertFalse(token["canonical_top5_match"])
        self.assertEqual(token["regret"], 0)
        self.assertFalse(report["checks"][0]["all_tokens_oracle_greedy"])
        self.assertTrue(report["completed"])

    def test_identity_precision_and_pdmux_mismatches_stop_before_generation(self):
        mutations = [dict(model_path="/models/other"), dict(tp_size=2), dict(pp_size=2),
                     dict(enable_pdmux=True), dict(dtype="float16")]
        for index, values in enumerate(mutations):
            with self.subTest(values=values):
                self.output = self.root / f"audit-{index}.json"
                self.calls.clear()
                self.server = dict(self.expected, **values)
                report = self.execute()
                self.assertFalse(report["completed"])
                self.assertEqual([call[1] for call in self.calls], ["/get_server_info"])
                self.assertTrue(report["errors"])

    def test_source_hash_mismatch_is_saved_without_http(self):
        self.source.write_text("changed")
        report = self.execute()
        self.assertFalse(report["completed"])
        self.assertEqual(self.calls, [])
        self.assertIn("hash", report["errors"][0].lower())

    def test_missing_expected_metadata_is_not_inferred_from_live_endpoint(self):
        report = self.execute(expected=False)
        self.assertFalse(report["completed"])
        self.assertEqual(self.calls, [])

    def test_expected_metadata_sidecar_is_supported(self):
        self.cases.with_suffix(".metadata.json").write_text(json.dumps(self.expected))
        self.assertTrue(self.execute(expected=False)["completed"])

    def test_bad_token_alignment_retains_full_response_and_incomplete_coverage(self):
        bad = response()
        bad["meta_info"]["input_token_logprobs"][1][1] = 99
        report = self.execute(lambda port, path, payload=None: self.server if payload is None else bad)
        self.assertFalse(report["completed"])
        self.assertFalse(report["full_token_coverage"])
        self.assertEqual(report["checks"][0]["raw_response"], bad)
        self.assertEqual(report["audited_output_tokens"], 0)
        self.assertIn("alignment", report["checks"][0]["error"])

    def test_partial_case_failure_never_hides_uncovered_tokens(self):
        self.cases.write_text(json.dumps([self.case, dict(self.case, label="second")]))
        count = 0
        def api(port, path, payload=None):
            nonlocal count
            if payload is None:
                return self.server
            count += 1
            data = response()
            if count == 2:
                data["meta_info"]["input_token_logprobs"].pop()
            return data
        report = self.execute(api)
        self.assertFalse(report["completed"])
        self.assertEqual(report["requested_output_tokens"], 4)
        self.assertEqual(report["audited_output_tokens"], 2)
        self.assertEqual(len(report["checks"]), 2)

    def test_existing_report_cannot_be_overwritten(self):
        self.output.write_text("original")
        with self.assertRaises(FileExistsError):
            self.execute()
        self.assertEqual(self.output.read_text(), "original")

    def test_duplicate_case_labels_are_rejected_before_http(self):
        self.cases.write_text(json.dumps([self.case, self.case]))
        report = self.execute()
        self.assertFalse(report["completed"])
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
