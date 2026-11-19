"""Finite CPU tests: all receipt evidence and the delegated program are synthetic."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import pytest

PATH=Path(__file__).with_name("formal_readiness_gate.py")

def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj))
    return path

def ref(path):
    import hashlib
    return {"path":str(path.resolve()),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}

@pytest.fixture
def gate(tmp_path,monkeypatch):
    assert PATH.exists(), "readiness gate not implemented"
    spec=importlib.util.spec_from_file_location("readiness_test",PATH)
    g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
    g.REPO=tmp_path;g.ROOT=tmp_path/"campaign";g.PYTHON=Path(sys.executable)
    g.POLICY_PATH=g.ROOT/"automation/policies/both-models-ready-before-formal-v1.json"
    g.DELEGATE_PATH=g.ROOT/"proposals/global-formal-round-order-v1/order.py"
    g.DELEGATE_PATH.parent.mkdir(parents=True)
    calls=tmp_path/"calls.jsonl"
    g.DELEGATE_PATH.write_text("import json,sys\nwith open("+repr(str(calls))+",'a') as f:f.write(json.dumps(sys.argv[1:])+chr(10))\n")
    g.DELEGATE_SHA256=ref(g.DELEGATE_PATH)["sha256"]
    policy={"schema_version":1,"id":"both-models-ready-before-formal-v1","status":"ACTIVE_USER_INSTRUCTION","required_models":["8b","70b"],"required_workloads":["toolagent","conversation"],"gpu_budget_per_experiment":4,"precision":"bfloat16","max_total_tokens":32768,"readiness_paths":{m:str(g.ROOT/"automation/readiness"/(m+".json")) for m in ["8b","70b"]},"readiness_receipt_status":"ROOT_REVIEWED_MODEL_FORMAL_READINESS","required_readiness_references":["configuration_freeze","rate_freeze","formal_trace_freeze","final_reverification","prepared_formal_manifest","dry_run_evidence"],"formal_gate":{"entrypoint":str(PATH.resolve()),"delegate":str(g.DELEGATE_PATH),"delegate_sha256":g.DELEGATE_SHA256,"both_models_ready_required_for_every_formal_launch":True,"legacy_direct_formal_launch_not_authorized":True,"preparation_split_and_native_cpu_dryruns_allowed":True}}
    write(g.POLICY_PATH,policy);g.POLICY_SHA256=ref(g.POLICY_PATH)["sha256"]
    monkeypatch.setattr(g.os,"execv",lambda executable,argv:subprocess.run(argv,check=True).returncode)
    g.test_calls=calls;g.test_policy=policy;g.test_refs={}
    for model,pool in [("8b","gpu4567"),("70b","gpu0123")]:
        folder=g.ROOT/("pool-gpu4567" if pool=="gpu4567" else "main")/"ready-evidence";folder.mkdir(parents=True)
        selected=write(folder/"selected.json",{"status":"reviewed_selected_configurations","records":[{"candidate":{"model":model},"workload":w} for w in policy["required_workloads"]]})
        rates=write(folder/"rates.json",{"status":"frozen","selection_manifest_sha256":ref(selected)["sha256"],"evidence_phases":["rate_calibration"],"records":[{"model":model,"workload":w} for w in policy["required_workloads"]]})
        traces=write(folder/"traces.json",{"status":"PASS","purpose":"formal_frozen_rate","selection_manifest_sha256":ref(selected)["sha256"],"rates_manifest_sha256":ref(rates)["sha256"],"records":[{"model_key":model,"workload":w} for w in policy["required_workloads"]]})
        raw=write(folder/"actual-evidence.json",{"observed":"temporary test evidence"})
        source={str(raw):ref(raw)["sha256"]}
        tasks={}
        for w in policy["required_workloads"]:
            for r in [1,2,3]:
                label=model+"-"+w+"-"+str(r)
                tasks[label]={"id":label,"phase":"formal_main","argv":["--model",model,"--tp","4","--pp","1","--replicas","1"],"workload":w,"role":"main","independent_repetition":r,"required_independent_repetitions":3}
        freeze=write(folder/"freeze.json",{"status":"frozen","source_sha256":source,"tasks":tasks})
        taskfile=write(folder/"tasks.json",{"source_sha256":source,"tasks":list(tasks.values())})
        protocol=write(folder/"protocol.json",{"stage":"formal","root":str(g.ROOT/("pool-gpu4567" if pool=="gpu4567" else "")),"source_sha256":source,"selection_manifest_sha256":ref(selected)["sha256"],"selected_rate_manifest_sha256":ref(rates)["sha256"],"trace_manifest_sha256":ref(traces)["sha256"],"task_labels":list(tasks)})
        producer=write(folder/"producer.py",{"test":"source"})
        validation=write(folder/"producer-validation.json",{"status":"PASS","protocol_reference":ref(protocol),"freeze_reference":ref(freeze),"producer_reference":ref(producer),"evidence_sha256":source})
        bundle=write(folder/"root-bundle.json",{"status":"ROOT_REVIEWED_FROZEN_PERFORMANCE_BUNDLE","reviewer":"root","pool":pool,"producer_kind":"pool_formal" if pool=="gpu4567" else "main_formal","producer_validation_passed":True,"producer_reference":ref(producer),"producer_validation_reference":ref(validation),"protocol_reference":ref(protocol),"freeze_reference":ref(freeze)})
        final=write(folder/"final-reverification.json",{"status":"ROOT_REVIEWED_FINAL_REVERIFICATION","reviewer":"root","model":model,"workloads":policy["required_workloads"],"gpu_budget":4,"configuration_freeze_reference":ref(selected),"evidence_sha256":source})
        native=g.REPO/"scripts"/("a100_"+pool)/"run_campaign.py"
        argv=[str(g.PYTHON),str(native),"--tasks",str(taskfile),"--freeze",str(freeze),"--queue-id","fresh", "--dry-run"]
        command=write(folder/"dry-command.json",{"argv":argv,"returncode":0})
        result=write(folder/"dry-result.json",{"status":"PASS","gpu_processes_started":0,"command":[str(g.PYTHON),"queue.py","--tasks",str(taskfile),"--freeze",str(freeze)]})
        dry=write(folder/"dry-review.json",{"status":"ROOT_REVIEWED_NATIVE_FORMAL_DRY_RUN","reviewer":"root","model":model,"prepared_formal_manifest_reference":ref(bundle),"checks":[{"command_reference":ref(command),"result_reference":ref(result),"task_plan_reference":ref(taskfile)}]})
        references={"configuration_freeze":ref(selected),"rate_freeze":ref(rates),"formal_trace_freeze":ref(traces),"final_reverification":ref(final),"prepared_formal_manifest":ref(bundle),"dry_run_evidence":ref(dry)}
        receipt={"schema_version":1,"status":"ROOT_REVIEWED_MODEL_FORMAL_READINESS","reviewer":"root","model":model,"workloads":policy["required_workloads"],"gpu_budget":4,"precision":"bfloat16","max_total_tokens":32768,"policy_reference":ref(g.POLICY_PATH),"references":references,"additional_formal_manifests":[]}
        path=Path(policy["readiness_paths"][model]);write(path,receipt);g.test_refs[model]=(path,receipt)
    bundle_ref=g.test_refs["8b"][1]["references"]["prepared_formal_manifest"]
    schedule=write(tmp_path/"schedule.json",{"review_reference":bundle_ref,"freeze_reference":json.loads(Path(bundle_ref["path"]).read_text())["freeze_reference"],"pool":"gpu4567"})
    g.test_schedule=write(tmp_path/"schedule-ref.json",ref(schedule))
    g.test_args=["launch","--schedule-reference",str(g.test_schedule),"--repeat","1","--transport","default","--queue-id","fresh", "--receipt",str(tmp_path/"order-receipt.json")]
    return g


def test_missing_and_single_model_never_delegate(gate):
    Path(gate.test_policy["readiness_paths"]["70b"]).unlink()
    with pytest.raises((ValueError,OSError)):gate.main(gate.test_args+["--execute"])
    assert not gate.test_calls.exists()
    Path(gate.test_policy["readiness_paths"]["8b"]).unlink()
    with pytest.raises((ValueError,OSError)):gate.main(gate.test_args)
    assert not gate.test_calls.exists()


@pytest.mark.parametrize("field,value",[("model","70b"),("gpu_budget",8),("gpu_budget",True),("workloads",["toolagent"]),("precision","float16"),("status","ready")])
def test_wrong_receipt_identity_refused(gate,field,value):
    path,receipt=gate.test_refs["8b"];receipt[field]=value;write(path,receipt)
    with pytest.raises(ValueError):gate.main(gate.test_args)
    assert not gate.test_calls.exists()


def test_all_six_pins_required_and_hash_rechecked(gate):
    path,receipt=gate.test_refs["70b"]
    target=Path(receipt["references"]["final_reverification"]["path"]);target.write_text('{}')
    with pytest.raises(ValueError):gate.main(gate.test_args)
    assert not gate.test_calls.exists()


def test_both_ready_delegates_exact_launch_once_with_global_round_args(gate):
    global_ref=write(gate.REPO/"global-ref.json",{"path":"/test/inventory","sha256":"a"*64})
    args=gate.test_args+["--execute","--global-inventory-reference",str(global_ref)]
    assert gate.main(args)==0
    actual=[json.loads(x) for x in gate.test_calls.read_text().splitlines()]
    assert actual==[args]
    proof=Path(gate.REPO/"order-receipt.json.readiness.json")
    assert proof.is_file()
    doc=json.loads(proof.read_text());assert doc["policy_reference"]["sha256"]==gate.POLICY_SHA256
    assert set(doc["model_readiness"])=={"8b","70b"}
    with pytest.raises(FileExistsError):gate.main(args)
    assert len(gate.test_calls.read_text().splitlines())==1


def test_readonly_single_model_check_allows_preparation_without_70b(gate,capsys):
    Path(gate.test_policy["readiness_paths"]["70b"]).unlink()
    assert gate.main(["check","--model","8b"])==0
    assert json.loads(capsys.readouterr().out)["models_checked"]==["8b"]
    assert not gate.test_calls.exists()


def test_unrelated_schedule_bundle_refused(gate):
    schedule=Path(json.loads(gate.test_schedule.read_text())["path"])
    write(schedule,{"pool":"gpu4567","review_reference":ref(write(gate.REPO/"unrelated.json",{"status":"ready"}))})
    write(gate.test_schedule,ref(schedule))
    with pytest.raises(ValueError):gate.main(gate.test_args)
    assert not gate.test_calls.exists()


def test_policy_and_delegate_drift_refused(gate):
    gate.POLICY_PATH.write_text(gate.POLICY_PATH.read_text()+" ")
    with pytest.raises(ValueError):gate.main(gate.test_args)
    assert not gate.test_calls.exists()


def test_delegate_drift_refused(gate):
    gate.DELEGATE_PATH.write_text("raise SystemExit(0)")
    with pytest.raises(ValueError):gate.main(gate.test_args)
    assert not gate.test_calls.exists()


def test_rate_freeze_cannot_use_formal_evidence(gate):
    path,receipt=gate.test_refs["8b"];rate=Path(receipt["references"]["rate_freeze"]["path"])
    doc=json.loads(rate.read_text());doc["evidence_phases"]=["formal_main"];write(rate,doc)
    receipt["references"]["rate_freeze"]=ref(rate);write(path,receipt)
    with pytest.raises(ValueError):gate.main(gate.test_args)
    assert not gate.test_calls.exists()


def test_native_dryrun_must_be_complete_success_with_no_gpu_launch(gate):
    path,receipt=gate.test_refs["70b"];drypath=Path(receipt["references"]["dry_run_evidence"]["path"])
    dry=json.loads(drypath.read_text());resultpath=Path(dry["checks"][0]["result_reference"]["path"])
    result=json.loads(resultpath.read_text());result["gpu_processes_started"]=4;write(resultpath,result)
    dry["checks"][0]["result_reference"]=ref(resultpath);write(drypath,dry);receipt["references"]["dry_run_evidence"]=ref(drypath);write(path,receipt)
    with pytest.raises(ValueError):gate.main(gate.test_args)
    assert not gate.test_calls.exists()


def test_false_budget_in_frozen_task_cannot_pass_root_boolean(gate):
    path,receipt=gate.test_refs["8b"];bundlepath=Path(receipt["references"]["prepared_formal_manifest"]["path"])
    bundle=json.loads(bundlepath.read_text());fp=Path(bundle["freeze_reference"]["path"]);f=json.loads(fp.read_text());next(iter(f["tasks"].values()))["argv"][3]="8";write(fp,f)
    bundle["freeze_reference"]=ref(fp);write(bundlepath,bundle);receipt["references"]["prepared_formal_manifest"]=ref(bundlepath);write(path,receipt)
    with pytest.raises(ValueError):gate.main(gate.test_args)
    assert not gate.test_calls.exists()


@pytest.mark.parametrize("allowlisted",[False,True])
def test_only_explicit_auxiliary_bundle_is_allowed_and_keeps_real_formal_stage(gate,allowlisted):
    path,receipt=gate.test_refs["8b"]
    primary=receipt["references"]["prepared_formal_manifest"]
    aux=json.loads(Path(primary["path"]).read_text());aux["producer_kind"]="pool_auxiliary"
    auxpath=write(gate.REPO/"aux-root-review.json",aux)
    if allowlisted:
        receipt["additional_formal_manifests"]=[ref(auxpath)];write(path,receipt)
    schedule=Path(json.loads(gate.test_schedule.read_text())["path"])
    write(schedule,{"pool":"gpu4567","review_reference":ref(auxpath),"freeze_reference":aux["freeze_reference"]});write(gate.test_schedule,ref(schedule))
    if allowlisted:
        assert gate.main(gate.test_args)==0
        assert len(gate.test_calls.read_text().splitlines())==1
    else:
        with pytest.raises(ValueError):gate.main(gate.test_args)
        assert not gate.test_calls.exists()


def test_schedule_must_bind_same_full_freeze_even_with_allowed_root_review(gate):
    schedule=Path(json.loads(gate.test_schedule.read_text())["path"])
    doc=json.loads(schedule.read_text());doc["freeze_reference"]=ref(write(gate.REPO/"other-freeze.json",{"status":"frozen"}))
    write(schedule,doc);write(gate.test_schedule,ref(schedule))
    with pytest.raises(ValueError):gate.main(gate.test_args)
    assert not gate.test_calls.exists()
