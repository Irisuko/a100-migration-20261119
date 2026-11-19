"""Selection must not learn from formal data or hide failed requests/repeats."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).with_name("select_development.py")
selection = None
if SCRIPT.exists():
    spec = importlib.util.spec_from_file_location("test_development_selector", SCRIPT)
    selection = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selection)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(selection, "Development selector is missing")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.trace = self.root / "dev_toolagent.jsonl"
        jsonl(self.trace, [dict(idx=i, arrival=i / 10, input_len=2, output_len=2, input_ids=[128000, 42]) for i in range(4)])
        write(self.trace.with_suffix(".stats.json"), dict(duration_sec=4, requests=4, start_sec=1200,
              source_duration_sec=4, stretch=1, tokenizer="fixture-tokenizer"))
        self.source = {"python/engine.py": hashlib.sha256(b"source").hexdigest()}
        self.builder = self.root / "build_trace.py"
        self.builder.write_text("# Explicit CPU fixture builder provenance\n")
        self.raw = self.root / "raw_toolagent.jsonl"
        jsonl(self.raw, [dict(timestamp=1200000 + i * 100, input_length=2, output_length=2, hash_ids=[i])
                         for i in range(4)])
        self.manifest = self.root / "manifest.json"
        self.tuning_manifest = self.root / "dev-tuning-manifest.json"
        self.manifest_record = dict(name="fixture_dev_toolagent", model="Llama-3.1-8B-Instruct",
            workload="toolagent", purpose="dev", path=str(self.trace), sha256=sha(self.trace),
            stats_path=str(self.trace.with_suffix(".stats.json")), stats_sha256=sha(self.trace.with_suffix(".stats.json")),
            source_start_sec=1200, source_end_exclusive_sec=1204, stretch=1, requests=4,
            base_arrival_window_sec=4, source_indices_sha256=selection.digest(list(range(4))),
            all_rows_and_hash_blocks_validated=True, length_clipping=False, padding_tokens=0)
        write(self.manifest, dict(schema_version=1, status="PASS", builder=dict(path=str(self.builder), sha256=sha(self.builder)),
              sources=[dict(workload="toolagent", path=str(self.raw), sha256=sha(self.raw))], records=[self.manifest_record]))
        for name, value in [("TRACE_MANIFEST", self.manifest), ("TUNING_TRACE_MANIFEST", self.tuning_manifest),
                            ("TRACE_BUILDER", self.builder)]:
            patch = mock.patch.object(selection, name, value, create=True)
            patch.start(); self.addCleanup(patch.stop)

    def run_fixture(self, name, family="muxpp", rate=1, ttfts=None, lag=0, phase="development",
                    unfinished=0, unsent=0, chunk=4096, source=None, tpot=.05, limit=None):
        task_dir = self.root / "runs" / name
        out = self.root / "measurements" / name
        task_dir.mkdir(parents=True)
        out.mkdir(parents=True)
        total = 4 if limit is None else limit
        planned = [dict(idx=i, arrival=i / 10 / rate, input_len=2, output_len=2) for i in range(total)]
        records = []
        for i, item in enumerate(planned[:total - unsent]):
            success = i < total - unsent - unfinished
            sent = item["arrival"] + lag
            ttft = (ttfts or [1] * total)[i]
            first = sent + ttft if success else None
            end = first + tpot if success else 300
            records.append(dict(**item, sent_at=sent, first_token_at=first, finished_at=end,
                                status="completed" if success else "unfinished", success=success,
                                output_tokens=2 if success else 0, ttft=ttft if success else None,
                                e2e=ttft + tpot if success else None, tpot=tpot if success else None,
                                send_lag=lag, cached_tokens=0, itl=[],
                                chunks=[dict(at=first, completion_tokens=1), dict(at=end, completion_tokens=2)] if success else []))
        jsonl(out / "trace_planned.jsonl", planned)
        jsonl(out / "trace_requests.jsonl", records)
        scale = 1 / rate
        window = 4 * scale if limit is None else planned[-1]["arrival"]
        args = dict(model="8b", family=family, tp=2, pp=2, replicas=1, chunk=chunk,
                    out=str(out), trace=str(self.trace), time_scale=scale, limit=limit, suite="trace")
        argv = ["--model", "8b", "--family", family, "--chunk", str(chunk), "--suite", "trace",
                "--trace", str(self.trace), "--time-scale", str(scale), "--out", str(out)]
        if limit is not None:
            argv += ["--limit", str(limit)]
        actual_source = source or self.source
        write(task_dir / "queue-task.json", dict(task=dict(id=name, phase=phase, workload="toolagent", argv=argv, timeout=600, dependencies=[]),
                                                  evidence=dict(source_sha256=actual_source, trace_sha256=sha(self.trace))))
        write(task_dir / "outcome.json", dict(status="completed", returncode=0))
        write(out / "run.json", dict(args=args, source_sha256=actual_source, start_unix=1))
        write(out / "trace_summary.json", dict(planned_requests=total, arrival_window_sec=window,
              cutoff_time=300, run_status="completed", measurement_eligible=True, limit=limit,
              trace=str(self.trace), time_scale=scale, interrupted=False,
              request_details_file=str(out / "trace_requests.jsonl"), planned_requests_file=str(out / "trace_planned.jsonl"),
              output_id_validation=dict(successful_requests_with_verified_ids=total - unfinished - unsent),
              slo={"ttft_le_5s_and_tpot_le_100ms": {"rate": 1}}))
        return task_dir

    def test_formal_artifacts_are_rejected_even_with_successful_results(self):
        run = self.run_fixture("formal", phase="formal_main")
        with self.assertRaisesRegex(ValueError, "phase"):
            selection.analyze([run])

    def test_formal_trace_cannot_be_relabelled_as_development(self):
        run = self.run_fixture("mislabelled")
        inventory = json.loads(self.manifest.read_text())
        inventory["records"].append(dict(sha256=sha(self.trace), purpose="formal"))
        write(self.manifest, inventory)
        with self.assertRaisesRegex(ValueError, "formal|heldout"):
            selection.analyze([run])

    def test_unknown_trace_hash_fails_closed_despite_execution_provenance(self):
        self.trace.write_text(self.trace.read_text().replace("128000, 42", "128000, 43"))
        run = self.run_fixture("unknown")
        with self.assertRaisesRegex(ValueError, "allowlist|audited development"):
            selection.analyze([run])

    def test_unknown_trace_is_rejected_even_when_measurement_artifacts_missing(self):
        self.trace.write_text(self.trace.read_text().replace("128000, 42", "128000, 43"))
        run = self.run_fixture("unknown_missing")
        (self.root / "measurements/unknown_missing/trace_summary.json").unlink()
        with self.assertRaisesRegex(ValueError, "allowlist|audited development"):
            selection.analyze([run])

    def test_trace_allowlist_matches_both_model_and_workload(self):
        run = self.run_fixture("wrong_identity")
        path = run / "queue-task.json"
        task = json.loads(path.read_text()); task["task"]["workload"] = "conversation"; write(path, task)
        with self.assertRaisesRegex(ValueError, "model|workload|allowlist"):
            selection.analyze([run])
        task["task"]["workload"] = "toolagent"
        task["task"]["argv"][1] = "70b"; write(path, task)
        metadata_path = self.root / "measurements/wrong_identity/run.json"
        metadata = json.loads(metadata_path.read_text()); metadata["args"]["model"] = "70b"; write(metadata_path, metadata)
        with self.assertRaisesRegex(ValueError, "model|workload|allowlist"):
            selection.analyze([run])

    def test_missing_or_failed_manifest_is_not_an_allowlist(self):
        run = self.run_fixture("no_allowlist")
        original = json.loads(self.manifest.read_text())
        self.manifest.unlink()
        with self.assertRaisesRegex(ValueError, "manifest|allowlist"):
            selection.analyze([run])
        write(self.manifest, {**original, "status": "failed"})
        with self.assertRaisesRegex(ValueError, "manifest|PASS"):
            selection.analyze([run])

    def test_allowlist_requires_known_unchanged_builder(self):
        run = self.run_fixture("changed_builder")
        self.builder.write_text("# changed builder\n")
        with self.assertRaisesRegex(ValueError, "builder|Builder"):
            selection.analyze([run])

    def test_declared_dev_record_cannot_contain_formal_raw_rows(self):
        run = self.run_fixture("bad_raw_window")
        raw = [json.loads(line) for line in self.raw.read_text().splitlines()]
        raw[0]["timestamp"] = 600000
        jsonl(self.raw, raw)
        inventory = json.loads(self.manifest.read_text())
        inventory["sources"][0]["sha256"] = sha(self.raw); write(self.manifest, inventory)
        with self.assertRaisesRegex(ValueError, "raw|window|source rows"):
            selection.analyze([run])

    def test_missing_stats_or_tampered_stats_cannot_change_window_provenance(self):
        run = self.run_fixture("changed_stats")
        self.trace.with_suffix(".stats.json").unlink()
        with self.assertRaisesRegex(ValueError, "Missing.*stats"):
            selection.analyze([run])
        write(self.trace.with_suffix(".stats.json"), dict(duration_sec=400, requests=4))
        with self.assertRaisesRegex(ValueError, "stats.*hash|hash.*stats"):
            selection.analyze([run])

    def test_tuning_manifest_cannot_claim_an_unpinned_parent(self):
        run = self.run_fixture("bad_tuning_parent")
        inventory = json.loads(self.manifest.read_text())
        inventory.update(parent_manifest=str(self.manifest), parent_manifest_sha256="0" * 64)
        write(self.tuning_manifest, inventory)
        with self.assertRaisesRegex(ValueError, "parent provenance"):
            selection.analyze([run])

    def test_successful_record_cannot_finish_before_its_first_token(self):
        run = self.run_fixture("bad_clock")
        path = self.root / "measurements/bad_clock/trace_requests.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        records[0]["finished_at"] = .5
        jsonl(path, records)
        with self.assertRaisesRegex(ValueError, "first token"):
            selection.analyze([run])

    def test_recompute_uses_all_planned_and_keeps_censored_p95(self):
        run = self.run_fixture("incomplete", unfinished=1, unsent=1)
        result = selection.analyze([run])
        row = result["measurements"][0]
        self.assertEqual(row["historical_main_slo"], .5)
        self.assertEqual(row["planned"], 4)
        self.assertEqual(row["ttft_p95_sec"], ">cutoff")
        self.assertFalse(row["selection_eligible"])
        self.assertEqual(result["draft_selections"], [])

    def test_ranking_uses_primary_then_ttft_then_throughput_at_equal_rate(self):
        bad = self.run_fixture("low_slo", ttfts=[1, 1, 1, 6], chunk=1024)
        slow = self.run_fixture("higher_ttft", ttfts=[2] * 4, chunk=2048)
        low_throughput = self.run_fixture("low_throughput", chunk=4096, tpot=.09)
        best = self.run_fixture("best", chunk=8192, tpot=.01)
        result = selection.analyze([bad, slow, low_throughput, best])
        selected = result["draft_selections"][0]
        self.assertEqual(selected["labels"], ["best"])
        self.assertEqual(selected["parameters"]["chunk"], 8192)
        self.assertEqual(selected["trace_sha256"], sha(self.trace))
        self.assertEqual(selected["source_sha256"], self.source)
        self.assertEqual(selected["absolute_requests_per_sec"], 1)
        self.assertFalse(result["freeze_created"])

    def test_rates_are_never_ranked_against_each_other(self):
        runs = [self.run_fixture("slow_load", rate=1), self.run_fixture("fast_load", rate=4)]
        result = selection.analyze(runs)
        self.assertEqual({r["absolute_requests_per_sec"] for r in result["draft_selections"]}, {1, 4})

    def test_source_changes_require_explicit_separation(self):
        a = self.run_fixture("a")
        b = self.run_fixture("b", source={"python/engine.py": hashlib.sha256(b"changed").hexdigest()})
        with self.assertRaisesRegex(ValueError, "source"):
            selection.analyze([a, b])
        result = selection.analyze([a, b], separate_sources=True)
        self.assertEqual(len(result["draft_selections"]), 2)

    def test_send_lag_flags_do_not_replace_historical_primary_ranking(self):
        run = self.run_fixture("late", lag=6)
        row = selection.analyze([run])["measurements"][0]
        self.assertEqual(row["historical_main_slo"], 1)
        self.assertEqual(row["arrival_main_slo"], 0)
        self.assertTrue(row["arrival_lag_flags"])
        self.assertTrue(row["selection_eligible"])

    def test_short_prefix_is_reported_but_cannot_choose_final_parameters(self):
        row = selection.analyze([self.run_fixture("partial", limit=2)])["measurements"][0]
        self.assertFalse(row["selection_eligible"])
        self.assertIn("partial_trace", row["exclusion_reasons"])

    def test_failed_repeat_disqualifies_config_instead_of_selecting_best_repeat(self):
        a = self.run_fixture("good_repeat", chunk=8192)
        b = self.run_fixture("bad_repeat", chunk=8192, unfinished=1)
        c = self.run_fixture("other", chunk=4096, ttfts=[2] * 4)
        result = selection.analyze([a, b, c])
        self.assertEqual(result["draft_selections"][0]["parameters"]["chunk"], 4096)

    def test_missing_artifact_repeat_disqualifies_same_configuration_and_rate(self):
        good = self.run_fixture("good_present", chunk=8192)
        failed = self.run_fixture("failed_absent", chunk=8192)
        other = self.run_fixture("other_present", chunk=4096, ttfts=[2] * 4)
        (self.root / "measurements/failed_absent/trace_summary.json").unlink()
        result = selection.analyze([good, failed, other])
        self.assertEqual(result["draft_selections"][0]["parameters"]["chunk"], 4096)
        config = next(r for r in result["configuration_aggregates"] if r["parameters"]["chunk"] == 8192)
        self.assertFalse(config["selection_eligible"])
        self.assertIn("missing_repeat_artifacts", config["exclusion_reasons"])

    def test_summary_window_must_match_trace_duration_or_explicit_task_window(self):
        run = self.run_fixture("wrong_window")
        path = self.root / "measurements/wrong_window/trace_summary.json"
        value = json.loads(path.read_text()); value["arrival_window_sec"] = 400
        write(path, value)
        with self.assertRaisesRegex(ValueError, "arrival.window"):
            selection.analyze([run])
        task_path = run / "queue-task.json"
        task = json.loads(task_path.read_text())
        task["task"]["argv"] += ["--arrival-window-sec", "400"]
        write(task_path, task)
        self.assertEqual(selection.analyze([run])["measurements"][0]["absolute_requests_per_sec"], .01)

    def test_rate_brackets_never_mix_different_arrival_intent_at_equal_average_rate(self):
        a = self.run_fixture("intent_a", family="muxpp", rate=1)
        b = self.run_fixture("intent_b", family="chunked", rate=2)
        task_path = b / "queue-task.json"
        task = json.loads(task_path.read_text())
        task["task"]["argv"] += ["--arrival-window-sec", "4"]
        write(task_path, task)
        summary_path = self.root / "measurements/intent_b/trace_summary.json"
        summary = json.loads(summary_path.read_text()); summary["arrival_window_sec"] = 4
        write(summary_path, summary)
        result = selection.analyze([a, b])
        self.assertEqual(result["rate_suggestions"][0]["common_measured_rates"], [])

    def test_missing_repeat_at_another_rate_does_not_disqualify_this_rate(self):
        good = self.run_fixture("present_rate", chunk=8192, rate=1)
        missing = self.run_fixture("missing_rate", chunk=8192, rate=2)
        (self.root / "measurements/missing_rate/run.json").unlink()
        result = selection.analyze([good, missing])
        self.assertEqual(result["draft_selections"][0]["labels"], ["present_rate"])

    def test_changed_configuration_bytes_form_separate_candidates(self):
        a = self.run_fixture("config_a", ttfts=[2] * 4)
        b = self.run_fixture("config_b", ttfts=[1] * 4)
        for name, value in [("config_a", "a" * 64), ("config_b", "b" * 64)]:
            path = self.root / "measurements" / name / "run.json"
            metadata = json.loads(path.read_text())
            metadata["configuration_sha256"] = {"spatial": value}
            write(path, metadata)
        result = selection.analyze([a, b])
        self.assertEqual(len(result["configuration_aggregates"]), 2)
        self.assertEqual(result["draft_selections"][0]["labels"], ["config_b"])
        self.assertEqual(result["draft_selections"][0]["configuration_sha256"], {"spatial": "b" * 64})

    def test_changed_model_provenance_cannot_be_compared(self):
        a = self.run_fixture("model_a")
        b = self.run_fixture("model_b")
        for name, revision in [("model_a", "revision-a"), ("model_b", "revision-b")]:
            path = self.root / "measurements" / name / "run.json"
            metadata = json.loads(path.read_text())
            metadata.update(model_revision=revision, model_provenance={"config.json": "a" * 64})
            write(path, metadata)
        with self.assertRaisesRegex(ValueError, "model provenance"):
            selection.analyze([a, b])

    def test_queued_runtime_hash_cannot_disappear_from_execution_evidence(self):
        run = self.run_fixture("missing_runtime_hash")
        path = run / "queue-task.json"
        task = json.loads(path.read_text())
        task["evidence"]["source_sha256"]["python/sglang/srt/managers/scheduler.py"] = "a" * 64
        write(path, task)
        with self.assertRaisesRegex(ValueError, "source.*missing|missing.*source"):
            selection.analyze([run])

    def test_optional_queued_audit_hash_is_not_required_in_runtime_subset(self):
        run = self.run_fixture("extra_audit_hash")
        path = run / "queue-task.json"
        task = json.loads(path.read_text())
        task["evidence"]["source_sha256"]["scripts/a100_gpu0123/resource_audit.py"] = "a" * 64
        write(path, task)
        self.assertTrue(selection.analyze([run])["measurements"][0]["selection_eligible"])

    def test_rate_suggestions_use_measured_common_absolute_rates(self):
        runs = []
        for family in ["muxpp", "chunked"]:
            for rate, ttfts in [(1, [1] * 4), (2, [1, 1, 1, 6]), (4, [1, 6, 6, 6])]:
                runs.append(self.run_fixture(f"{family}_{rate}", family=family, rate=rate, ttfts=ttfts, phase="rate_calibration"))
        suggestion = selection.analyze(runs)["rate_suggestions"][0]
        self.assertEqual(suggestion["light_requests_per_sec"], 1)
        self.assertEqual(suggestion["critical_requests_per_sec"], 2)
        self.assertEqual(suggestion["overload_requests_per_sec"], 4)
        self.assertEqual(suggestion["common_measured_rates"], [1, 2, 4])
        self.assertEqual(suggestion["missing_expected_systems"], ["muxwise", "pd"])

    def test_empty_measurements_never_invent_selections_or_rates(self):
        result = selection.analyze([])
        self.assertEqual(result["draft_selections"], [])
        self.assertEqual(result["rate_suggestions"], [])
        self.assertFalse(result["freeze_created"])

    def test_startup_failure_with_no_observations_remains_an_unranked_row(self):
        run = self.run_fixture("startup", unsent=4)
        summary_path = self.root / "measurements/startup/trace_summary.json"
        summary = json.loads(summary_path.read_text())
        summary.update(measurement_eligible=False, run_status="startup_failed")
        write(summary_path, summary)
        write(run / "outcome.json", dict(status="failed", returncode=1))
        result = selection.analyze([run])
        self.assertEqual(result["draft_selections"], [])
        self.assertEqual(result["measurements"][0]["unsent"], 4)
        self.assertIsNone(result["configuration_aggregates"][0]["output_tokens_per_sec"])

    def test_changed_trace_is_rejected_against_execution_provenance(self):
        run = self.run_fixture("changed")
        self.trace.write_text(self.trace.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "trace hash"):
            selection.analyze([run])

    def platform(self, label, transport="default", version="a"):
        return dict(status="verified", software=dict(
            loader_environment={"LD_LIBRARY_PATH": "/validated/lib", "RDMAV_DRIVERS": "/validated/lib/provider"},
            installed_libraries={"/validated/lib/libibverbs.so": {"sha256": version * 64, "resolved_path": "/validated/lib/libibverbs.so.1"}},
            packages_sha256={"/validated/packages/rdma.deb": version * 64}),
            configuration={"mooncake_transport": transport},
            public_environment={"LD_LIBRARY_PATH": "/validated/lib", "RDMAV_DRIVERS": "/validated/lib/provider",
                                "MC_INTRANODE_NVLINK": "1" if transport == "nvlink" else None},
            manifest_path=f"/campaign/queues/{label}/platform-launch.json", manifest_sha256=sha(self.trace))

    def test_platform_queue_ids_and_manifest_paths_do_not_split_repeats(self):
        a, b = self.run_fixture("platform_a"), self.run_fixture("platform_b")
        def reader(root, provenance):
            self.assertEqual(root, self.root)
            return self.platform(provenance["task"]["id"])
        with mock.patch.object(selection, "read_platform_provenance", side_effect=reader, create=True):
            result = selection.analyze([a, b])
        self.assertEqual(len(result["configuration_aggregates"]), 1)
        self.assertEqual(result["configuration_aggregates"][0]["measurement_count"], 2)
        self.assertEqual(result["measurements"][0]["platform_provenance"]["status"], "verified")

    def test_platform_transport_variants_compete_as_configs_not_source_cohorts(self):
        a = self.run_fixture("default_transport", ttfts=[2] * 4)
        b = self.run_fixture("nvlink_transport")
        def reader(root, provenance):
            label = provenance["task"]["id"]
            return self.platform(label, "nvlink" if label == "nvlink_transport" else "default")
        with mock.patch.object(selection, "read_platform_provenance", side_effect=reader, create=True):
            result = selection.analyze([a, b])
        self.assertEqual(len({r["source_fingerprint"] for r in result["measurements"]}), 1)
        self.assertEqual(len(result["configuration_aggregates"]), 2)
        self.assertEqual(result["draft_selections"][0]["platform_configuration"], {"mooncake_transport": "nvlink"})

    def test_platform_software_changes_require_explicit_source_separation(self):
        a, b = self.run_fixture("software_a"), self.run_fixture("software_b")
        def reader(root, provenance):
            label = provenance["task"]["id"]
            return self.platform(label, version="b" if label == "software_b" else "a")
        with mock.patch.object(selection, "read_platform_provenance", side_effect=reader, create=True):
            with self.assertRaisesRegex(ValueError, "source"):
                selection.analyze([a, b])
            result = selection.analyze([a, b], separate_sources=True)
        self.assertEqual(len(result["draft_selections"]), 2)

    def test_missing_repeat_does_not_cross_transport_or_platform_software(self):
        good = self.run_fixture("platform_good")
        missing = self.run_fixture("platform_missing")
        (self.root / "measurements/platform_missing/trace_summary.json").unlink()
        for transport, version in [("nvlink", "a"), ("default", "b")]:
            with self.subTest(transport=transport, version=version):
                def reader(root, provenance):
                    label = provenance["task"]["id"]
                    return self.platform(label, transport, version) if label == "platform_missing" else self.platform(label)
                with mock.patch.object(selection, "read_platform_provenance", side_effect=reader, create=True):
                    result = selection.analyze([good, missing])
                self.assertEqual(result["draft_selections"][0]["labels"], ["platform_good"])

    def test_inconsistent_platform_evidence_is_not_downgraded_to_legacy(self):
        run = self.run_fixture("tampered_platform")
        with mock.patch.object(selection, "read_platform_provenance", side_effect=ValueError("Platform hash mismatch"), create=True):
            with self.assertRaisesRegex(ValueError, "Platform hash"):
                selection.analyze([run])

    def test_legacy_platform_is_explicitly_unrecorded(self):
        result = selection.analyze([self.run_fixture("legacy_platform")])
        row = result["measurements"][0]
        self.assertEqual(row["platform_provenance"]["status"], "missing")
        self.assertIsNone(row["platform_software"])
        self.assertEqual(row["platform_configuration"], {"mooncake_transport": "unrecorded"})


class AdoptedTraceAuditTests(unittest.TestCase):
    """CPU integration against immutable campaign artifacts when available."""
    def setUp(self):
        if not selection.TRACE_MANIFEST.is_file() or not selection.TUNING_TRACE_MANIFEST.is_file():
            self.skipTest("Adopted campaign trace artifacts are not installed")
        self.parent = json.loads(selection.TRACE_MANIFEST.read_text())
        self.tuning = json.loads(selection.TUNING_TRACE_MANIFEST.read_text())

    def test_actual_full_dev_and_four_target_tokenizer_tuning_traces_are_allowed(self):
        full = next(r for r in self.parent["records"] if r["purpose"] == "dev"
                    and r["model"] == "Llama-3.1-8B-Instruct" and r["workload"] == "toolagent")
        cache = {}
        for row in [full, *self.tuning["records"]]:
            with self.subTest(name=row["name"]):
                model = "8b" if row["model"] == "Llama-3.1-8B-Instruct" else "70b"
                path = Path(row["path"]).resolve()
                result = selection.audit_development_trace(path, selection.trace_metadata(path, cache),
                                                           model, row["workload"], cache)
                self.assertEqual(result["purpose"], row["purpose"])
                self.assertEqual(result["base_arrival_window_sec"], 600 if row["purpose"] == "dev" else 150)
                self.assertEqual(result["checked_source_rows"], row["requests"])
                self.assertEqual(result["formal_or_heldout_source_rows"], 0)

    def test_renamed_derived_formal_and_heldout_subsets_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            for purpose in ["formal", "heldout"]:
                with self.subTest(purpose=purpose):
                    source = next(r for r in self.parent["records"] if r["purpose"] == purpose
                                  and r["model"] == "Llama-3.1-8B-Instruct" and r["workload"] == "toolagent")
                    path = Path(directory) / "innocently_named_dev_subset.jsonl"
                    with Path(source["path"]).open() as original:
                        path.write_text(original.readline() + original.readline())
                    cache = {}
                    with self.assertRaisesRegex(ValueError, "allowlist|audited development"):
                        selection.audit_development_trace(path, selection.trace_metadata(path, cache),
                                                          "8b", "toolagent", cache)


if __name__ == "__main__":
    unittest.main()
