"""CPU-only tests: no Torch, server process, or real endpoint is used."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

PATH = Path(__file__).with_name("actual_decode_audit.py")
audit = None
if PATH.is_file():
    spec = importlib.util.spec_from_file_location("actual_decode_test_target", PATH)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)


def response(ids, n):
    tokens = [20000 + i for i in range(n)]
    return {"output_ids": tokens, "meta_info": {
        "prompt_tokens": len(ids), "completion_tokens": n, "cached_tokens": 0,
        "finish_reason": {"type": "length", "length": n},
        "output_token_logprobs": [[-.25, token, None] for token in tokens],
        "output_top_logprobs": [[[-.25, token, None], [-.5, 100, None], [-1., 101, None],
                                 [-2., 102, None], [-3., 103, None]] for token in tokens]}}


class FakeRunner:
    CORRECTNESS_LENGTHS = [100, 512, 2047, 2048, 2049, 4095, 4096, 4097, 8191, 8192, 8193, 12288, 31663, 32704]

    def __init__(self, model, family="plain", break_logprobs=False):
        self.model, self.family, self.break_logprobs = model, family, break_logprobs
        self.flushes, self.calls, self.phases = [], [], []

    def fixture_cases(self, model):
        return [[128000] + [42] * (n - 1) for n in self.CORRECTNESS_LENGTHS]

    def flush(self, port):
        self.flushes.append(port)

    def resource_phase(self, out, phase, ports):
        self.phases.append(phase)

    def request(self, port, endpoint, payload=None, timeout=600):
        if endpoint == "/get_server_info":
            return {"model_path": str(self.model), "tp_size": 1, "pp_size": 4, "dtype": "bfloat16",
                    "enable_pdmux": self.family == "muxpp", "chunked_prefill_size": 2048 if self.family == "plain" else -1}
        assert endpoint == "/generate"
        self.calls.append(payload)
        value = response(payload["input_ids"], payload["sampling_params"]["max_new_tokens"])
        if self.break_logprobs and payload["sampling_params"]["max_new_tokens"] == 32:
            value["meta_info"]["output_top_logprobs"].pop()
        return value


class DecodeAuditTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(audit, "actual_decode_audit.py is missing")

    def test_predeclared_eight_cells_and_exact_request_budget(self):
        cells = audit.condition_plan()
        self.assertEqual(len(cells), 8)
        self.assertEqual({(c["cache"], c["mode"], c["repeat"]) for c in cells},
                         {(cache, mode, repeat) for cache in ["cold", "warm"]
                          for mode in ["serial", "concurrent8"] for repeat in [1, 2]})
        self.assertEqual(sum(len(c["input_lengths"]) for c in cells), 36)
        self.assertEqual(sum(len(c["input_lengths"]) * 32 for c in cells), 1152)
        self.assertEqual(sum(len(c["input_lengths"]) for c in cells if c["cache"] == "warm"), 18)

    def test_parser_preserves_every_actual_decode_token_and_top5(self):
        body = response([128000, 42], 32)
        parsed = audit.parse_decode_response(body, [128000, 42], 32)
        self.assertEqual(parsed["audited_tokens"], 32)
        self.assertEqual([t["chosen"] for t in parsed["tokens"]], body["output_ids"])
        self.assertEqual(parsed["tokens"][0]["top_two_margin"], .25)

    def test_parser_rejects_missing_or_misaligned_or_nonfinite_logprobs(self):
        for mutation in ["missing", "alignment", "nonfinite", "unordered", "duplicate", "count"]:
            with self.subTest(mutation=mutation):
                body = response([128000, 42], 32)
                if mutation == "missing": body["meta_info"]["output_top_logprobs"].pop()
                elif mutation == "alignment": body["meta_info"]["output_token_logprobs"][0][1] = 99
                elif mutation == "nonfinite": body["meta_info"]["output_token_logprobs"][0][0] = float("nan")
                elif mutation == "unordered": body["meta_info"]["output_top_logprobs"][0].reverse()
                elif mutation == "duplicate": body["meta_info"]["output_top_logprobs"][0][1][1] = 20000
                else: body["meta_info"]["completion_tokens"] = 31
                with self.assertRaises(ValueError): audit.parse_decode_response(body, [128000, 42], 32)

    def test_reported_tie_does_not_become_exact_correctness_pass(self):
        body = response([128000, 42], 1)
        body["meta_info"]["output_top_logprobs"][0][1][0] = -.25
        parsed = audit.parse_decode_response(body, [128000, 42], 1)
        self.assertEqual(parsed["tokens"][0]["reported_max_tie_ids"], [100, 20000])
        self.assertEqual(parsed["tokens"][0]["canonical_top5_token"], 100)
        self.assertNotIn("correctness_pass", parsed)

    def test_full_mocked_preflight_is_bounded_and_retains_all_raw_responses(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory); model = out / "model"
            runner = FakeRunner(model)
            result = audit.run_preflight(runner, 32100, model, out, "plain")
            self.assertEqual(result["status"], "diagnostic_complete")
            self.assertTrue(result["full_token_coverage"])
            self.assertEqual(result["measured_requests"], 36)
            self.assertEqual(result["measured_output_tokens"], 1152)
            self.assertEqual((result["warmup_requests"], result["warmup_output_tokens"]), (19, 26))
            self.assertEqual(len(runner.calls), 55)
            self.assertEqual(len(runner.flushes), 10)  # prewarm before/after plus every cell
            self.assertEqual(runner.phases, ["initial", "final"])
            self.assertEqual(len(list((out / "actual-decode").rglob("request-*.json"))), 55)
            for request in runner.calls:
                self.assertTrue(request["return_logprob"])
                self.assertEqual(request["top_logprobs_num"], 5)
                self.assertEqual(request["logprob_start_len"], len(request["input_ids"]) - 1)
            self.assertTrue(result["diagnostic_only"])
            self.assertFalse(result["measurement_eligible"])

    def test_failed_coverage_keeps_raw_and_incomplete_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory); runner = FakeRunner(out / "model", break_logprobs=True)
            with self.assertRaises(ValueError): audit.run_preflight(runner, 32100, out / "model", out, "plain")
            summary = json.loads((out / "actual-decode-summary.json").read_text())
            self.assertFalse(summary["full_token_coverage"])
            self.assertEqual(summary["status"], "diagnostic_incomplete")
            files = list((out / "actual-decode").rglob("request-*.json"))
            self.assertTrue(any("raw_response" in json.loads(path.read_text()) for path in files))
            self.assertTrue(summary["errors"])

    def test_plan_forces_smoke_and_plain_chunk2048_without_trace_or_topology_override(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = audit.make_plan(["--model", "70b", "--family", "plain", "--mem", ".85"], Path(directory) / "new")
            self.assertEqual(plan["family"], "plain")
            self.assertEqual(plan["server_args"][-3:], ["--extra", "--chunked-prefill-size", "2048"])
            self.assertIn("smoke", plan["server_args"])
            self.assertTrue(plan["diagnostic_only"])
            for extra in [["--trace", "bad"], ["--oracle-cases", "bad"], ["--ora", "bad"],
                          ["--tp", "2"], ["--suite", "trace"]]:
                with self.subTest(extra=extra):
                    with self.assertRaises((ValueError, SystemExit)):
                        audit.make_plan(["--model", "70b", "--family", "plain", *extra], Path(directory) / "new")

    def test_unowned_wrapper_never_creates_output_or_calls_server_main(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "unowned"
            server = SimpleNamespace(require_active_controller=mock.Mock(side_effect=RuntimeError("no controller")),
                                     main=mock.Mock(side_effect=AssertionError("must not launch")))
            with mock.patch.dict(audit.sys.modules, {"server_run": server}), mock.patch.object(audit.sys, "argv",
                    [str(PATH), "--out", str(out), "--", "--model", "70b", "--family", "plain"]):
                with self.assertRaisesRegex(RuntimeError, "controller"):
                    audit.main()
            self.assertFalse(out.exists())
            server.main.assert_not_called()

    def test_wrapper_marks_metadata_and_restores_only_preflight_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "controlled"
            original = object()
            server = SimpleNamespace(require_active_controller=mock.Mock(), preflight=original,
                                     source_provenance=lambda repo: {"source_sha256": {"test": "hash"}})
            def failed_main():
                self.assertIsNot(server.preflight, original)
                serving = out / "serving"; serving.mkdir()
                (serving / "run.json").write_text(json.dumps({"status": "failed", "args": {"suite": "smoke"}}))
                raise RuntimeError("synthetic child failure")
            server.main = failed_main
            with mock.patch.dict(audit.sys.modules, {"server_run": server}), mock.patch.object(audit.sys, "argv",
                    [str(PATH), "--out", str(out), "--", "--model", "70b", "--family", "muxpp"]):
                with self.assertRaisesRegex(RuntimeError, "synthetic child failure"):
                    audit.main()
            self.assertIs(server.preflight, original)
            marked = json.loads((out / "serving/run.json").read_text())
            self.assertTrue(marked["diagnostic_only"])
            self.assertFalse(marked["measurement_eligible"])
            self.assertTrue((out / "serving/run.before-diagnostic-marking.json").is_file())
            self.assertEqual(json.loads((out / "diagnostic.json").read_text())["status"], "diagnostic_incomplete")


if __name__ == "__main__": unittest.main()
