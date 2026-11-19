"""Development queue generation reads evidence and never launches a process."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).with_name("build_development.py")
builder = None
if SCRIPT.exists():
    spec = importlib.util.spec_from_file_location("test_development_builder", SCRIPT)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DevelopmentBuilderTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(builder, "Development builder is missing")
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.repo = Path(tmp.name) / "repo"
        self.root = self.repo / "run/campaign"
        for name in ["python/sglang/engine.py", "scripts/a100_gpu0123/server_run.py"]:
            path = self.repo / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("# frozen\n")
        self.candidate = dict(id="8b-muxpp-pp4", model="8b", family="muxpp", tp=1, pp=4, replicas=1)
        write(self.root / "campaign.json", dict(candidates=[self.candidate]))
        self.label = "support-8b-muxpp-pp4-v2"
        self.run = self.root / "preflight" / self.label
        self.run.mkdir(parents=True)
        self.plan = self.root / "queues/support.json"
        write(self.plan, dict(tasks=[dict(id=self.label, candidate_id=self.candidate["id"],
                                         argv=["--out", str(self.run)])], excluded_equivalent_candidates=[]))
        model = self.repo.parent / "models/Llama-3.1-8B-Instruct"
        write(model / "config.json", dict(model_type="llama"))
        self.spatial = self.run / "spatial.yml"; self.spatial.write_text("sm_group_num: 8\n")
        self.metadata = dict(status="execution_complete", model_revision="frozen-revision",
            model_provenance={"config.json": sha(model / "config.json")},
            source_sha256=builder.current_sources(self.repo),
            configuration_sha256={"spatial": sha(self.spatial), "worker0.pdmux": sha(self.spatial)},
            configuration_paths={"worker0.pdmux": str(self.spatial)},
            args=dict(model="8b", family="muxpp", tp=1, pp=4, replicas=1, mem=.85, running=128,
                      graph=16, chunk=8192, baseline_chunk=2048, inflight=3, budget=32768,
                      custom_allreduce=False, extra=[]))
        write(self.run / "run.json", self.metadata)
        write(self.root / "runs" / self.label / "outcome.json", dict(status="completed", returncode=0))
        self.command = self.root / "runs" / self.label / "command.json"
        write(self.command, dict(argv=["taskset", "-c", "0-31,64-95", str(self.repo / ".venv/bin/python"),
            str(self.repo / "scripts/a100_gpu0123/server_run.py"), "--model", "8b", "--family", "muxpp",
            "--tp", "1", "--pp", "4", "--replicas", "1", "--out", str(self.run)]))
        write(self.run / "resource-audit.json", dict(schema_version=2, run_dir=str(self.run), status="passed", errors=[], missing_evidence=[]))
        self.manifest = self.root / "traces/manifest.json"
        records=[]
        for workload, count in [("toolagent", 4), ("conversation", 2)]:
            trace=self.root / "traces" / (workload+".jsonl")
            trace.parent.mkdir(parents=True, exist_ok=True)
            trace.write_text("".join(json.dumps(dict(idx=i,arrival=i,input_len=2,output_len=2,input_ids=[128000,42]))+"\n" for i in range(count)))
            stats=trace.with_suffix(".stats.json")
            write(stats,dict(start_sec=1200,source_duration_sec=240,stretch=2.5,duration_sec=600,requests=count))
            records.append(dict(model="Llama-3.1-8B-Instruct",workload=workload,purpose="dev",path=str(trace),stats_path=str(stats),
                sha256=sha(trace),stats_sha256=sha(stats),requests=count,base_arrival_window_sec=600,
                source_start_sec=1200,source_end_exclusive_sec=1440,source_indices_sha256="a"*64,
                all_rows_and_hash_blocks_validated=True,length_clipping=False,padding_tokens=0))
        write(self.manifest,dict(schema_version=1,status="PASS",records=records))
        patch=mock.patch.object(builder.resource_audit,"audit_run",return_value=dict(status="passed",errors=[],missing_evidence=[]))
        self.fresh=patch.start();self.addCleanup(patch.stop)
        patch=mock.patch("subprocess.Popen",side_effect=AssertionError("GPU/process execution forbidden"))
        patch.start();self.addCleanup(patch.stop)

    def build(self, **kwargs):
        return builder.build(self.root,self.plan,self.manifest,"test-v1",repo=self.repo,**kwargs)

    def test_only_supported_candidate_gets_two_full_trace_tasks(self):
        queue, protocol=self.build()
        self.assertEqual(len(queue["tasks"]),2)
        first=queue["tasks"][0]
        self.assertEqual(first["phase"],"development")
        self.assertEqual(first["workload"],"toolagent")
        self.assertEqual(first["timeout"],7000)
        self.assertNotIn("--limit",first["argv"])
        self.assertEqual(builder.option(first["argv"],"--time-scale"),"1.0")
        self.assertTrue(protocol["eligibility"][0]["runtime_supported"])
        self.assertIn(str(self.spatial),queue["source_sha256"])

    def test_generated_trace_commands_override_inherited_debug_with_common_info(self):
        import argparse

        self.metadata["args"]["extra"] = ["--log-level", "debug", "--enable-p2p-check"]
        write(self.run / "run.json", self.metadata)
        queue, protocol = self.build()
        self.assertEqual(protocol["performance_log_level"], "info")
        self.assertEqual(len(queue["tasks"]), 2)
        for task in queue["tasks"]:
            argv = task["argv"]
            self.assertEqual(builder.option(argv, "--suite"), "trace")
            extra = argv[argv.index("--extra") + 1:]
            self.assertEqual(extra[-2:], ["--log-level", "info"])
            self.assertIn("--enable-p2p-check", extra)
            self.assertIn("--pdmux-config-path", extra)
            parser = argparse.ArgumentParser()
            parser.add_argument("--log-level")
            effective, _ = parser.parse_known_args(["--log-level", "debug", *extra])
            self.assertEqual(effective.log_level, "info")

    def test_missing_or_failed_outcome_never_emits_task(self):
        path=self.root / "runs" / self.label / "outcome.json"
        for value in [None,dict(status="failed",returncode=1)]:
            if value is None:path.unlink()
            else:write(path,value)
            queue,protocol=self.build()
            self.assertEqual(queue["tasks"],[])
            self.assertTrue(protocol["eligibility"][0]["exclusion_reasons"])

    def test_missing_or_failed_stored_resource_audit_never_emits_task(self):
        path=self.run / "resource-audit.json"
        path.unlink();self.assertEqual(self.build()[0]["tasks"],[])
        write(path,dict(schema_version=2,run_dir=str(self.run),status="failed",errors=["leak"],missing_evidence=[]))
        self.assertEqual(self.build()[0]["tasks"],[])

    def test_completed_controller_label_must_launch_the_same_support_run(self):
        command=json.loads(self.command.read_text())
        command["argv"][-1]=str(self.root / "unrelated")
        write(self.command,command)
        self.assertEqual(self.build()[0]["tasks"],[])
        self.command.unlink()
        self.assertEqual(self.build()[0]["tasks"],[])

    def test_controller_cpu_binding_and_saved_topology_must_match(self):
        command=json.loads(self.command.read_text());command["argv"][2]="0-3"
        write(self.command,command)
        self.assertEqual(self.build()[0]["tasks"],[])
        command["argv"][2]="0-31,64-95"
        command["argv"][command["argv"].index("--tp")+1]="2"
        write(self.command,command)
        self.assertEqual(self.build()[0]["tasks"],[])

    def test_stale_pass_cannot_override_fresh_raw_evidence_failure(self):
        self.fresh.return_value=dict(status="failed",errors=["pending requests"],missing_evidence=[])
        queue,protocol=self.build()
        self.assertEqual(queue["tasks"],[])
        self.assertIn("fresh_resource_audit_not_passed",protocol["eligibility"][0]["exclusion_reasons"])

    def test_changed_source_or_configuration_is_not_reused(self):
        self.spatial.write_text("changed\n")
        self.assertEqual(self.build()[0]["tasks"],[])
        self.spatial.write_text("sm_group_num: 8\n")
        (self.repo / "python/sglang/engine.py").write_text("# different\n")
        self.assertEqual(self.build()[0]["tasks"],[])

    def test_tp_custom_requires_positive_marker_for_every_rank(self):
        self.candidate.update(id="8b-muxwise-tp4",family="muxwise",tp=4,pp=1)
        write(self.root / "campaign.json",dict(candidates=[self.candidate]))
        write(self.plan,dict(tasks=[dict(id=self.label,candidate_id=self.candidate["id"],argv=["--out",str(self.run)])]))
        self.metadata["args"].update(family="muxwise",tp=4,pp=1,custom_allreduce=True)
        write(self.run / "run.json",self.metadata)
        command=json.loads(self.command.read_text())
        for key,value in [("family","muxwise"),("tp","4"),("pp","1")]:
            command["argv"][command["argv"].index("--"+key)+1]=value
        write(self.command,command)
        log=self.run / "server-0.log"
        text="".join(f"[TP{i}] Custom allreduce enabled: rank={i} world_size=4 physical_device_ids=[0, 1, 2, 3] full_nvlink=True\n" for i in range(4))
        log.write_text(text.splitlines()[0]+"\n")
        self.assertEqual(self.build()[0]["tasks"],[])
        log.write_text(text)
        self.assertEqual(len(self.build()[0]["tasks"]),2)
        log.write_text(text+"Setup Custom allreduce failed with failure\n")
        self.assertEqual(self.build()[0]["tasks"],[])

    def test_70b_rate_is_independent_and_derived_from_selected_trace_count(self):
        self.candidate.update(id="70b-muxpp-pp4",model="70b")
        write(self.root / "campaign.json",dict(candidates=[self.candidate]))
        write(self.plan,dict(tasks=[dict(id=self.label,candidate_id=self.candidate["id"],argv=["--out",str(self.run)])]))
        model=self.repo.parent / "models/Llama-3.3-70B-Instruct"
        write(model / "config.json",dict(model_type="llama"))
        self.metadata["args"]["model"]="70b";write(self.run / "run.json",self.metadata)
        command=json.loads(self.command.read_text());command["argv"][command["argv"].index("--model")+1]="70b"
        write(self.command,command)
        manifest=json.loads(self.manifest.read_text())
        for row in manifest["records"]:row["model"]="Llama-3.3-70B-Instruct"
        write(self.manifest,manifest)
        queue,_=self.build()
        self.assertAlmostEqual(float(builder.option(queue["tasks"][0]["argv"],"--time-scale")),4/600/.5)
        self.assertEqual(float(builder.option(queue["tasks"][0]["argv"],"--arrival-window-sec")),8)

    def prepare_conditional_70b(self):
        self.candidate.update(id="70b-muxpp-pp2dp2",model="70b",pp=2,replicas=2)
        write(self.root / "campaign.json",dict(candidates=[self.candidate]))
        write(self.plan,dict(tasks=[dict(id=self.label,candidate_id=self.candidate["id"],argv=["--out",str(self.run)])]))
        model=self.repo.parent / "models/Llama-3.3-70B-Instruct"
        write(model / "config.json",dict(model_type="llama"))
        self.metadata["args"].update(model="70b",pp=2,replicas=2,suite="correctness",stress200=True)
        write(self.run / "run.json",self.metadata)
        command=json.loads(self.command.read_text())
        for key,value in [("model","70b"),("pp","2"),("replicas","2")]:
            command["argv"][command["argv"].index("--"+key)+1]=value
        command["argv"] += ["--suite","correctness","--stress200"]
        write(self.command,command)
        manifest=json.loads(self.manifest.read_text())
        for row in manifest["records"]:row["model"]="Llama-3.3-70B-Instruct"
        write(self.manifest,manifest)
        self.fresh.return_value=dict(status="passed",errors=[],missing_evidence=[],workers=[
            dict(index=i,status="passed",ranks=[dict(pp=p,tp=0) for p in range(2)]) for i in range(2)])

    def test_two_gpu_70b_smoke_or_missing_stress_is_not_development_eligible(self):
        self.prepare_conditional_70b()
        for suite,stress in [("smoke",True),("correctness",False)]:
            with self.subTest(suite=suite,stress=stress):
                self.metadata["args"].update(suite=suite,stress200=stress)
                write(self.run / "run.json",self.metadata)
                queue,protocol=self.build()
                self.assertEqual(queue["tasks"],[])
                self.assertIn("conditional_70b_requires_full_correctness_and_stress200",protocol["eligibility"][0]["exclusion_reasons"])

    def test_two_gpu_70b_full_gate_requires_command_and_all_replica_rank_evidence(self):
        self.prepare_conditional_70b()
        self.assertEqual(len(self.build()[0]["tasks"]),2)
        command=json.loads(self.command.read_text());command["argv"].remove("--stress200")
        write(self.command,command)
        self.assertEqual(self.build()[0]["tasks"],[])
        command["argv"] += ["--stress200"];write(self.command,command)
        self.fresh.return_value["workers"].pop()
        queue,protocol=self.build()
        self.assertEqual(queue["tasks"],[])
        self.assertIn("conditional_70b_full_audit_rank_coverage_incomplete",protocol["eligibility"][0]["exclusion_reasons"])

    def prepare_pd(self):
        self.candidate.update(id="8b-pd-2p2d",family="pd",pp=1,replicas=4,prefill_replicas=2)
        write(self.root/"campaign.json",dict(candidates=[self.candidate]))
        write(self.plan,dict(tasks=[dict(id=self.label,candidate_id=self.candidate["id"],argv=["--out",str(self.run)])]))
        self.metadata["args"].update(family="pd",pp=1,replicas=4,prefill_replicas=2,suite="correctness",stress200=True)
        write(self.run/"run.json",self.metadata)
        command=json.loads(self.command.read_text())
        for key,value in [("family","pd"),("pp","1"),("replicas","4")]:command["argv"][command["argv"].index("--"+key)+1]=value
        command["argv"] += ["--prefill-replicas","2","--suite","correctness","--stress200"]
        write(self.command,command)
        self.fresh.return_value=dict(status="passed",errors=[],missing_evidence=[],workers=[
            dict(index=i,status="passed",ranks=[dict(pp=0,tp=0)]) for i in range(4)])

    def test_8b_pd_smoke_and_missing_stress_cannot_supply_architecture_seed(self):
        self.prepare_pd()
        for suite,stress in [("smoke",True),("correctness",False)]:
            with self.subTest(suite=suite,stress=stress):
                self.metadata["args"].update(suite=suite,stress200=stress);write(self.run/"run.json",self.metadata)
                queue,protocol=self.build();self.assertEqual(queue["tasks"],[])
                self.assertIn("pd_requires_full_paired_correctness_and_stress200",protocol["eligibility"][0]["exclusion_reasons"])

    def test_8b_pd_full_pair_gate_requires_command_and_every_role_rank(self):
        self.prepare_pd();self.assertEqual(len(self.build()[0]["tasks"]),2)
        command=json.loads(self.command.read_text());command["argv"].remove("--stress200");write(self.command,command)
        self.assertEqual(self.build()[0]["tasks"],[])
        command["argv"] += ["--stress200"];write(self.command,command)
        self.fresh.return_value["workers"].pop()
        queue,protocol=self.build();self.assertEqual(queue["tasks"],[])
        self.assertIn("pd_full_audit_rank_coverage_incomplete",protocol["eligibility"][0]["exclusion_reasons"])

    def test_pd_candidate_role_split_and_recorded_controller_split_must_match(self):
        self.prepare_pd()
        original=json.loads(self.command.read_text())
        for observed,recorded in [(1,1),(3,3),(2,1),(2,3)]:
            with self.subTest(observed=observed,recorded=recorded):
                self.metadata["args"]["prefill_replicas"]=observed;write(self.run/"run.json",self.metadata)
                command=json.loads(json.dumps(original));command["argv"][command["argv"].index("--prefill-replicas")+1]=str(recorded)
                write(self.command,command)
                queue,protocol=self.build();self.assertEqual(queue["tasks"],[])
                self.assertIn("support_pd_role_layout_mismatch",protocol["eligibility"][0]["exclusion_reasons"])

    def test_audited_tuning_subwindow_allowed_but_formal_overlap_rejected(self):
        value=json.loads(self.manifest.read_text())
        for row in value["records"]:
            row.update(purpose="dev_tuning",source_end_exclusive_sec=1260,base_arrival_window_sec=150)
            stats=Path(row["stats_path"]);write(stats,dict(start_sec=1200,source_duration_sec=60,stretch=2.5,duration_sec=150,requests=row["requests"]))
            row["stats_sha256"]=sha(stats)
        path=self.root / "traces/tuning.json";write(path,value)
        queue,_=builder.build(self.root,self.plan,path,"test-v1",repo=self.repo)
        self.assertEqual(float(builder.option(queue["tasks"][0]["argv"],"--arrival-window-sec")),150)
        for row in value["records"]:row["source_start_sec"]=600
        write(path,value)
        with self.assertRaisesRegex(ValueError,"development.*window|window.*development"):
            builder.build(self.root,self.plan,path,"test-v1",repo=self.repo)

    def test_tampered_trace_or_stats_cannot_emit_a_queue(self):
        value=json.loads(self.manifest.read_text())
        Path(value["records"][0]["path"]).write_text("changed\n")
        with self.assertRaisesRegex(ValueError,"trace.*hash|hash.*trace"):
            self.build()

    def test_artifacts_are_immutable_and_generation_does_not_launch(self):
        queue,protocol=self.build()
        directory=self.root / "queues/draft"
        builder.write_artifacts(directory,queue,protocol)
        original=(directory / "tasks.json").read_bytes()
        with self.assertRaises(FileExistsError):builder.write_artifacts(directory,queue,protocol)
        self.assertEqual((directory / "tasks.json").read_bytes(),original)

    def test_explicit_known_runner_compatibility_is_pinned_and_uses_off_mux_seed(self):
        # Link immutable runtime bytes into the temporary fixture; never edit them.
        from _compatibility_test_fixtures import populate_runtime
        runtime=self.repo/'python/sglang'
        (runtime/'engine.py').unlink();runtime.rmdir()
        proposal=SCRIPT.parents[2]/'run/a100-gpu0123-20261114/proposals/native-overlap-v1'
        populate_runtime(self.repo,json.loads((proposal/'compatibility.json').read_text())['runtime_source_sha256'])
        (self.repo/'scripts/a100_gpu0123/server_run.py').write_bytes((proposal/'server_run.py.candidate').read_bytes())
        self.metadata['source_sha256']=builder.current_sources(self.repo)
        self.metadata['source_sha256'][builder.serving_compatibility.TARGET]=builder.serving_compatibility.BASE
        self.metadata['commands']=[{'command':['python','-m','sglang.launch_server','--disable-overlap-schedule']}]
        write(self.run/'run.json',self.metadata)
        write(self.run/'worker0/server-final.json',{'disable_overlap_schedule':True})
        queue,protocol=self.build(serving_compatibility_path=proposal/'compatibility.json')
        self.assertEqual(len(queue['tasks']),2)
        self.assertEqual(builder.option(queue['tasks'][0]['argv'],'--overlap'),'off')
        self.assertEqual(protocol['eligibility'][0]['support_source_compatibility']['kind'],'reviewed_default_off_compatibility')
        self.assertIn(str(proposal/'compatibility.json'),queue['source_sha256'])
        self.assertEqual(queue['serving_compatibility'],protocol['serving_compatibility'])
        self.assertEqual(self.build()[0]['tasks'],[]) # no opt-in means no source exception


if __name__=="__main__":unittest.main()
