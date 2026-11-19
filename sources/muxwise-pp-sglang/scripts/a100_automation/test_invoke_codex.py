"""CPU-only subprocess tests; the executable below never calls Codex or GPUs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

ADAPTER = Path(__file__).with_name("invoke_codex.py")

FAKE = r'''#!/usr/bin/env python3
import json, os, pathlib, subprocess, sys, time
args = sys.argv[1:]
schema = json.loads(pathlib.Path(args[args.index("--output-schema")+1]).read_text())
prompt = sys.stdin.read()
settings = json.loads(pathlib.Path(__file__).with_suffix(".json").read_text())
mode = settings.get("mode", "ok")
record = pathlib.Path(settings["record"])
with record.open("a") as f:
    f.write(json.dumps({"argv":args,"prompt":prompt,"env":{k:v for k,v in os.environ.items() if k.startswith("CODEX_") or k in ["CUDA_VISIBLE_DEVICES","SESSION_ID","TOOL_CALL_ID","INTERNAL_ORIGINATOR_OVERRIDE"]}})+"\n")
if mode == "sleep": time.sleep(30)
if mode == "detached":
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pathlib.Path(settings["child_pid"]).write_text(str(child.pid))
    time.sleep(30)
props = schema["properties"]
def fixed(name): return props[name].get("const", props[name].get("enum", [None])[0])
ack = {"batch_id":fixed("batch_id"), "invocation_nonce":fixed("invocation_nonce"), "event_ids":settings["event_ids"], "status":"blocked" if mode == "blocked" else "handled", "summary":"Accepted test batch"}
if mode == "batch": ack["batch_id"] = "different-batch"
if mode == "nonce": ack["invocation_nonce"] = "stale-nonce"
if mode == "ids": ack["event_ids"] = ["event-a"]
if mode == "ids_reverse": ack["event_ids"] = list(reversed(ack["event_ids"]))
if mode == "status": ack["status"] = "finished"
if mode == "summary": ack["summary"] = " "
if mode == "extra": ack["launch_anything"] = True
final = pathlib.Path(args[args.index("-o")+1])
if mode != "missing_final": final.write_text("not json" if mode == "malformed" else json.dumps(ack))
print(json.dumps({"type":"thread.started","thread_id":"fake"}), flush=True)
if mode == "tool": print(json.dumps({"type":"item.completed", "item":{"type":"command_execution", "command":"true"}}), flush=True)
if mode == "bad_jsonl": print("not json", flush=True)
if mode != "no_complete": print(json.dumps({"type":"turn.completed", "usage":{}}), flush=True)
if mode == "failed_turn": print(json.dumps({"type":"turn.failed", "error":{"message":"failed"}}), flush=True)
print("fake stderr retained", file=sys.stderr, flush=True)
sys.exit(7 if mode == "exit7" else 0)
'''


@pytest.fixture
def case(tmp_path):
    fake = tmp_path / "fake codex"
    fake.write_text(FAKE)
    fake.chmod(0o700)
    handoff = tmp_path / "HANDOFF.md"
    handoff.write_text("Handoff marker: use existing finite queues.")
    worker = tmp_path / "worker-prompt.md"
    worker.write_text("Worker marker: inspect original outcome evidence.")
    event = {"schema_version":1,"batch_id":"batch-1","event_ids":["event-a","event-b"],"events":[{"event_id":x,"kind":"terminal","source_id":"owned-queue","path":"/owned/status.json","key":x,"details":{"status":"complete"}} for x in ["event-a","event-b"]],"created_unix":1700000000,"reports":[]}
    event_file = tmp_path / "event.json"
    event_file.write_text(json.dumps(event))
    config = {"codex_cli":str(fake),"repo":str(tmp_path),"handoff":str(handoff),"worker_prompt":str(worker),"invocations_dir":str(tmp_path / "invocations"),"timeout_sec":3}
    config_file = tmp_path / "adapter.json"
    config_file.write_text(json.dumps(config))
    settings = {"record":str(tmp_path / "calls.jsonl"),"event_ids":event["event_ids"]}
    fake.with_suffix(".json").write_text(json.dumps(settings))
    return {"tmp":tmp_path,"fake":fake,"settings":settings,"event":event,"event_file":event_file,"config":config,"config_file":config_file,"ack":tmp_path / "ack.json"}


def run(case, mode="ok", *, test_mode=None, timeout=None, batch=None):
    case["settings"]["mode"] = mode
    case["fake"].with_suffix(".json").write_text(json.dumps(case["settings"]))
    if test_mode is not None: case["config"]["test_mode"] = test_mode
    if timeout is not None: case["config"]["timeout_sec"] = timeout
    case["config_file"].write_text(json.dumps(case["config"]))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="must-be-cleared", CODEX_THREAD_ID="stale", CODEX_SESSION_ID="stale", CODEX_TOOL_CALL_ID="stale", CODEX_INTERNAL_ORIGINATOR_OVERRIDE="stale", SESSION_ID="stale", TOOL_CALL_ID="stale", INTERNAL_ORIGINATOR_OVERRIDE="stale")
    return subprocess.run([sys.executable,str(ADAPTER),"--event-file",str(case["event_file"]),"--ack-file",str(case["ack"]),"--batch-id",batch or case["event"]["batch_id"],"--config",str(case["config_file"])],capture_output=True,text=True,env=env,timeout=12)


def calls(case):
    path = case["tmp"] / "calls.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def artifacts(case):
    dirs = list((case["tmp"] / "invocations").iterdir())
    assert len(dirs) == 1
    return dirs[0]


def test_success_requires_correlated_ack_and_retains_evidence(case):
    result = run(case)
    assert result.returncode == 0, result.stderr
    ack = json.loads(case["ack"].read_text())
    assert ack["batch_id"] == "batch-1"
    assert ack["event_ids"] == ["event-a","event-b"]
    assert ack["status"] == "handled"
    assert len(ack["invocation_nonce"]) >= 32
    invocation = artifacts(case)
    for name in ["event.json","prompt.txt","output-schema.json","invocation.json","stdout.jsonl","stderr.log","last-message.json","result.json"]:
        assert (invocation / name).is_file(), name
    receipt = json.loads((invocation / "result.json").read_text())
    assert receipt["status"] == "acknowledged"
    assert receipt["returncode"] == 0 and receipt["turn_completed"] is True
    assert "fake stderr retained" in (invocation / "stderr.log").read_text()
    call = calls(case)[0]
    assert call["argv"][:5] == ["-a","never","-s","danger-full-access","exec"]
    assert "-m" not in call["argv"] and "--model" not in call["argv"] and "-c" not in call["argv"]
    assert call["argv"][-1] == "-"
    assert call["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert not any(k in call["env"] for k in ["CODEX_THREAD_ID","CODEX_SESSION_ID","CODEX_TOOL_CALL_ID","CODEX_INTERNAL_ORIGINATOR_OVERRIDE","SESSION_ID","TOOL_CALL_ID","INTERNAL_ORIGINATOR_OVERRIDE"])
    assert "Handoff marker" in call["prompt"] and "Worker marker" in call["prompt"]


@pytest.mark.parametrize("mode", ["batch","nonce","ids","ids_reverse","status","summary","extra","missing_final","malformed","no_complete","exit7","bad_jsonl","failed_turn"])
def test_incomplete_or_mismatched_outcome_never_publishes_ack(case, mode):
    result = run(case, mode)
    assert result.returncode != 0
    assert not case["ack"].exists()
    receipt = json.loads((artifacts(case) / "result.json").read_text())
    assert receipt["status"] == "unknown"
    assert len(calls(case)) == 1


def test_blocked_is_valid_terminal_ack(case):
    assert run(case,"blocked").returncode == 0
    assert json.loads(case["ack"].read_text())["status"] == "blocked"


@pytest.mark.parametrize("first", ["ok","nonce","sleep"])
def test_claim_prevents_reinvocation_after_success_unknown_or_timeout(case, first):
    run(case,first,timeout=0.2 if first == "sleep" else 3)
    second = run(case)
    assert second.returncode != 0
    assert len(calls(case)) == 1


def test_preexisting_ack_is_not_overwritten(case):
    case["ack"].write_text('{"sentinel":true}')
    result = run(case)
    assert result.returncode != 0
    assert json.loads(result.stderr)["status"] == "rejected"
    assert json.loads(case["ack"].read_text()) == {"sentinel":True}
    assert not calls(case)


def test_bad_batch_does_not_invoke(case):
    result = run(case,batch="wrong-batch")
    assert result.returncode != 0
    assert json.loads(result.stderr)["status"] == "rejected"
    assert not calls(case) and not case["ack"].exists()


@pytest.mark.parametrize("mutation", ["duplicate_ids","missing_event","duplicate_event","invalid_version"])
def test_invalid_event_does_not_invoke(case, mutation):
    event = case["event"]
    if mutation == "duplicate_ids": event["event_ids"] = ["event-a","event-a"]
    if mutation == "missing_event": event["events"].pop()
    if mutation == "duplicate_event": event["events"][1] = event["events"][0]
    if mutation == "invalid_version": event["schema_version"] = True
    case["event_file"].write_text(json.dumps(event))
    result = run(case)
    assert result.returncode != 0
    assert json.loads(result.stderr)["status"] == "rejected"
    assert not calls(case)


def test_timeout_preserves_detached_work(case):
    child_file = case["tmp"] / "child.pid"
    case["settings"]["child_pid"] = str(child_file)
    try:
        result = run(case,"detached",timeout=0.3)
        assert result.returncode != 0
        assert not case["ack"].exists()
        child_pid = int(child_file.read_text())
        os.kill(child_pid,0)
        receipt = json.loads((artifacts(case) / "result.json").read_text())
        assert receipt["timed_out"] is True
        assert receipt["status"] == "unknown"
    finally:
        if child_file.exists():
            try: os.kill(int(child_file.read_text()),15)
            except ProcessLookupError: pass


def test_acceptance_uses_read_only_and_ack_only_prompt(case):
    assert run(case,test_mode="acceptance").returncode == 0
    call = calls(case)[0]
    assert call["argv"][:5] == ["-a","never","-s","read-only","exec"]
    assert "ACCEPTANCE" in call["prompt"]
    assert "Handoff marker" not in call["prompt"]


def test_acceptance_rejects_observed_tool_execution(case):
    assert run(case,"tool",test_mode="acceptance").returncode != 0
    assert not case["ack"].exists()


def test_bad_test_mode_does_not_invoke(case):
    result = run(case,test_mode="yes")
    assert result.returncode != 0
    assert json.loads(result.stderr)["status"] == "rejected"
    assert not calls(case)


def test_parallel_claims_launch_only_once(case):
    case["settings"]["mode"] = "sleep"
    case["fake"].with_suffix(".json").write_text(json.dumps(case["settings"]))
    case["config"]["timeout_sec"] = 0.3
    case["config_file"].write_text(json.dumps(case["config"]))
    command = [sys.executable,str(ADAPTER),"--event-file",str(case["event_file"]),"--ack-file",str(case["ack"]),"--batch-id","batch-1","--config",str(case["config_file"])]
    first = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    second = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    first.communicate(timeout=12)
    second.communicate(timeout=12)
    assert sorted([first.returncode, second.returncode]) == [1,2]
    assert len(calls(case)) == 1
    assert not case["ack"].exists()
    assert json.loads((artifacts(case) / "result.json").read_text())["timed_out"] is True


def test_os_launch_failure_retains_claim_and_unknown_evidence(case):
    case["fake"].write_text("#!/definitely/missing/interpreter\n")
    result = run(case)
    assert result.returncode == 1
    receipt = json.loads((artifacts(case) / "result.json").read_text())
    assert receipt["status"] == "unknown"
    assert receipt["returncode"] is None
    assert not case["ack"].exists()
    case["fake"].write_text(FAKE)
    assert run(case).returncode == 2
    assert not calls(case)
