"""CPU contracts preventing invalid or superseded measurements entering tables."""
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parent))
SPEC = importlib.util.spec_from_file_location(
    "final_performance_tables", Path(__file__).with_name("final_performance_tables.py"))
MODULE_EXISTS = SPEC is not None and Path(SPEC.origin).exists()
if MODULE_EXISTS:
    tables = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(tables)


def sample_client(rate=0, concurrency=32):
    names = [f"perf-{i:02d}" for i in range(64)]
    fixture = {"suites": {"perf": names, "warmup": names[:24]}, "cases": {
        name: {"input_tokens": 128, "max_new_tokens": 2} for name in names}}
    requests = [{"fixture_id": name, "request_id": f"namespace:{name}",
                 "input_tokens": 128, "requested_output_tokens": 2,
                 "output_ids": [10, 11], "actual_output_tokens": 2,
                 "error": None, "sse_done": True, "ttft_s": 1.0, "tpot_s": .02,
                 "e2e_s": 1.02, "arrival_ttft_s": 4.0, "arrival_e2e_s": 4.02,
                 "queue_delay_s": 3.0, "events": [
                     {"new_tokens": 1, "total_tokens": 1, "raw_tokens": 1,
                      "server_completion_tokens": 1, "elapsed_s": 1.0},
                     {"new_tokens": 1, "total_tokens": 2, "raw_tokens": 1,
                      "server_completion_tokens": 2, "elapsed_s": 1.02}]} for name in names]
    aggregate = {"requested": 64, "completed": 64, "failed": 0,
                 "actual_output_tokens": 128, "wall_s": 10,
                 "output_tokens_per_s": 12.8}
    client = {"suite": "perf", "passed": True, "repeat": 1, "seed": 20261114,
              "fixture_sha256": "fixture-hash", "client_source_sha256": "client-hash",
              "request_rate": rate, "concurrency": concurrency, "order_mode": "fixture",
              "request_order": names, "cache_reset": True,
              "external_cache_reset": {"http_status": 200, "success": True},
              "experiment_slo": {"ttft_target_s": 2.0, "tpot_target_s": .05},
              "requests": requests, "aggregate": aggregate,
              "trials": [{"repeat_index": 0, "aggregate": aggregate}]}
    return client, fixture


def write_matrix(root, *, fail_last_trial=False, reused_ids=False):
    names = ["v2-main-static", "v2-main-baseline-whole", "v2-main-no-prefetch",
             "v2-main-cpu-metadata", "v2-main-no-calibration", "v2-main-no-async",
             "v2-main-cpp", "v2-main-alp-whole", "v2-main-baseline-chunk2048"]
    def put(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    protocol = {"protocol_revision": 2, "main_runs": names}
    put(root / "main-protocol.json", protocol)
    source_hashes = {f"runtime{i}.py": str(i) for i in range(5)}
    put(root / "main-review.json", {"protocol_revision": 2, "passed": True,
        "protocol_sha256": hashlib.sha256((root / "main-protocol.json").read_bytes()).hexdigest(),
        "runtime_sources": {p: {"expected": value} for p, value in source_hashes.items()},
        "plan_snapshots": {name: {"mode": "mux", "policy": "static", "chunk_tokens": 0,
                                  "config": {}, "cpu_prefill_metadata": False,
                                  "shared_server_flags": {},
                                  "server_flags": {"--pdmux-pp-policy": "static"}} for name in names}})
    client, fixture = sample_client()
    fixture["model_path"] = "/models/Llama-3.1-8B-Instruct"
    put(root / "fixtures.json", fixture)
    fixture_hash = hashlib.sha256((root / "fixtures.json").read_bytes()).hexdigest()
    client_hash = hashlib.sha256((Path(__file__).with_name("experiment_client.py")).read_bytes()).hexdigest()
    directory = root / names[0]
    launch = {"mode": "mux", "model": fixture["model_path"], "command": ["--pdmux-pp-policy", "static"],
              "experiment_purpose": "main", "profile": False, "trace": False,
              "plan_reuse_check": False, "runtime_source_sha256": source_hashes}
    put(directory / "launch.json", launch)
    put(directory / "scheduler.json", {})
    put(directory / "shutdown.json", {"remaining": [], "forced": False})
    history = [("warmup", 0), ("perf", 0), ("perf", 1), ("perf", 2),
               ("serial", 0), ("cache", 0), ("concurrent", 0)]
    put(directory / "outcome.json", {"passed": True, "runs": [
        {"suite": suite, "repeat": repetition, "exit_code": 0} for suite, repetition in history]})
    warmup = copy.deepcopy(client)
    warmup.update(suite="warmup", concurrency=1, fixture_sha256=fixture_hash, client_source_sha256=client_hash,
                  request_order=fixture["suites"]["warmup"], requests=warmup["requests"][:24])
    for request in warmup["requests"]:
        request["request_id"] = "warmup:" + request["request_id"]
    warmup["aggregate"].update(requested=24, completed=24, actual_output_tokens=48, output_tokens_per_s=4.8)
    put(directory / "warmup-0.json", warmup)
    for repetition in range(3):
        row = copy.deepcopy(client)
        row.update(seed=20261114 + repetition, fixture_sha256=fixture_hash, client_source_sha256=client_hash)
        if not reused_ids:
            for request in row["requests"]:
                request["request_id"] = f"trial{repetition}:" + request["request_id"]
        if fail_last_trial and repetition == 2:
            row["passed"] = False
        put(directory / f"perf-{repetition}.json", row)
    # A complete-looking superseded directory must never enter the whitelist.
    put(root / "main-old" / "launch.json", launch)


class FinalPerformanceTablesTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(MODULE_EXISTS, "standalone final tables module is not implemented")

    def test_accepts_complete_saturation_and_real_perf_openloop_suite(self):
        for suite, rate, concurrency in (("perf", 0, 32), ("perf-openloop", 8, 64)):
            client, fixture = sample_client(rate, concurrency)
            tables.validate_client(client, fixture, "fixture-hash", 0, suite, "client-hash")

    def test_rejects_a_lost_sse_delta_even_when_aggregate_claims_success(self):
        client, fixture = sample_client()
        client["requests"][0]["events"][0]["server_completion_tokens"] = 2
        with self.assertRaisesRegex(ValueError, "SSE"):
            tables.validate_client(client, fixture, "fixture-hash", 0, "perf", "client-hash")

    def test_rejects_truncation_and_unpaired_seed_or_client_source(self):
        for field, value in (("seed", 99), ("client_source_sha256", "other")):
            client, fixture = sample_client(); client[field] = value
            with self.assertRaises(ValueError):
                tables.validate_client(client, fixture, "fixture-hash", 0, "perf", "client-hash")
        client, fixture = sample_client(); client["requests"][0]["output_ids"] = [10]
        with self.assertRaisesRegex(ValueError, "output"):
            tables.validate_client(client, fixture, "fixture-hash", 0, "perf", "client-hash")

    def test_aggregates_three_trial_statistics_without_pooling_request_quantiles(self):
        rows = [{"output_tokens_per_s": rate, "ttft_s_p95": ttft,
                 "main_both_met_fraction": slo} for rate, ttft, slo in
                ((100, 1, .25), (300, 9, .5), (200, 2, .75))]
        row = tables.aggregate_trials("v2-main-cpp", "perf", rows)
        self.assertEqual(row["output_tokens_per_s"], {"median": 200, "min": 100, "max": 300})
        self.assertEqual(row["ttft_s_p95"]["median"], 2)
        self.assertEqual(row["main_both_met_fraction"]["mean"], .5)
        with self.assertRaisesRegex(ValueError, "three"):
            tables.aggregate_trials("v2-main-cpp", "perf", rows[:2])

    def test_v2_whitelist_refuses_duplicate_or_superseded_names(self):
        names = [f"v2-main-{i}" for i in range(9)]
        self.assertEqual(tables.validate_whitelist({"protocol_revision": 2, "main_runs": names}), names)
        for changed in (names[:8] + [names[0]], ["main-old"] + names[1:]):
            with self.assertRaises(ValueError):
                tables.validate_whitelist({"protocol_revision": 2, "main_runs": changed})

    def test_existing_statistics_keep_arrival_slo_separate_from_http_slo(self):
        client, fixture = sample_client(8, 64)
        base = {"run": "v2-main-cpp", "configured_ttft_s": 2, "configured_tpot_s": .05,
                "main_ttft_s": 2, "main_tpot_s": .05, "classification": "main8b",
                "run_complete": True, "experiment_purpose": "main", "main_protocol_whitelisted": True,
                "run_passed": True, "profile": False, "profile_metadata_recorded": True}
        row = tables.trial_row(base, client, Path("perf-openloop-0.json"), 0,
                               client["trials"][0], client["requests"], [])
        self.assertEqual(row["main_both_met_fraction"], 1)
        self.assertEqual(row["main_arrival_both_met_fraction"], 0)

    def test_partial_matrix_includes_only_complete_cohorts_and_ignores_v1(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_matrix(root)
            result = tables.build_report(root)
            self.assertEqual(result["status"], "partial")
            self.assertEqual([r["run"] for r in result["tables"]["perf"]], ["v2-main-static"])
            self.assertNotIn("main-old", result["runs"])
            write_matrix(root, fail_last_trial=True)
            rejected = tables.build_report(root)
            self.assertEqual(rejected["tables"]["perf"], [])
            self.assertEqual(rejected["runs"]["v2-main-static"]["status"], "excluded")

    def test_reused_request_ids_between_trials_reject_entire_server_cohort(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_matrix(root, reused_ids=True)
            result = tables.build_report(root)
            self.assertEqual(result["tables"]["perf"], [])
            self.assertIn("request IDs", result["runs"]["v2-main-static"]["reason"])

    def test_terminal_failure_is_excluded_even_without_scheduler_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_matrix(root)
            (root / "v2-main-static" / "outcome.json").write_text(json.dumps({"passed": False, "error": "server failed"}))
            (root / "v2-main-static" / "scheduler.json").unlink()
            result = tables.build_report(root)
            self.assertEqual(result["runs"]["v2-main-static"]["status"], "excluded")

    def test_actual_config_change_cannot_enter_a_planned_single_factor_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_matrix(root)
            path = root / "v2-main-static" / "launch.json"
            launch = json.loads(path.read_text())
            launch["config"] = {"pp_prefetch": False}
            path.write_text(json.dumps(launch))
            result = tables.build_report(root)
            self.assertEqual(result["tables"]["perf"], [])
            self.assertIn("config", result["runs"]["v2-main-static"]["reason"])

    def test_inner_aggregate_cannot_override_validated_throughput(self):
        client, fixture = sample_client()
        client["trials"][0]["aggregate"] = dict(client["aggregate"], wall_s=.01)
        with self.assertRaisesRegex(ValueError, "aggregate"):
            tables.validate_client(client, fixture, "fixture-hash", 0, "perf", "client-hash")

    def test_missing_warmup_artifact_cannot_prove_equal_training_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_matrix(root)
            (root / "v2-main-static" / "warmup-0.json").unlink()
            result = tables.build_report(root)
            self.assertEqual(result["tables"]["perf"], [])
            self.assertIn("warmup", result["runs"]["v2-main-static"]["reason"])

    def test_unplanned_server_flag_cannot_change_the_measured_algorithm(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_matrix(root)
            path = root / "v2-main-static" / "launch.json"
            launch = json.loads(path.read_text())
            launch["command"] += ["--pdmux-pp-slo-tpot", "0.01"]
            path.write_text(json.dumps(launch))
            result = tables.build_report(root)
            self.assertEqual(result["tables"]["perf"], [])
            self.assertIn("flags", result["runs"]["v2-main-static"]["reason"])


if __name__ == "__main__":
    unittest.main()
