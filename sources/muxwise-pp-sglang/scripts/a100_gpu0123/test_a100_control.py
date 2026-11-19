"""CPU-only safety and lifecycle tests; never initialize CUDA."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

SOURCE = Path(__file__).with_name("control.py")


@pytest.fixture(autouse=True)
def no_gpu_subprocesses(monkeypatch):
    original = subprocess.Popen
    def checked(command, *args, **kwargs):
        if not command or Path(command[0]).resolve() != Path("/bin/sleep").resolve():
            raise AssertionError("CPU controller tests only permit explicit /bin/sleep fixtures")
        return original(command, *args, **kwargs)
    monkeypatch.setattr(subprocess, "Popen", checked)


@pytest.fixture
def control(tmp_path):
    assert SOURCE.exists(), "The checked GPU controller has not been implemented"
    spec = importlib.util.spec_from_file_location("a100_control", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.GLOBAL_SUSPENSION = tmp_path / "global-disabled.json"
    return module


def test_environment_overrides_gpu_selection_and_removes_debug_variables(control, tmp_path):
    env = control.environment(tmp_path, {"CUDA_VISIBLE_DEVICES": "0", "SGLANG_PDMUX_PP_NUMERIC_TRACE": "bad",
                                        "NCCL_P2P_DISABLE": "1", "NCCL_SHM_USE_CUDA_MEMCPY": "1"})
    assert env["CUDA_VISIBLE_DEVICES"].split(",") == list(control.GPUS.values())
    assert "SGLANG_PDMUX_PP_NUMERIC_TRACE" not in env
    assert "NCCL_P2P_DISABLE" not in env
    assert "NCCL_SHM_USE_CUDA_MEMCPY" not in env
    assert env["CUDA_HOME"].endswith(".local/cuda-12.8")
    assert env["TORCH_CUDA_ARCH_LIST"] == "8.0"
    assert str(tmp_path) in env["FLASHINFER_WORKSPACE_BASE"]
    assert env["A100_GPU0123_CAMPAIGN_ROOT"] == str(tmp_path.resolve())


@pytest.mark.parametrize("label", ["../escape", "/absolute", "a/b", ".", "..", "bad label", ""])
def test_labels_cannot_escape_run_root(control, label):
    with pytest.raises(ValueError):
        control.validate_label(label)


def test_process_identity_rejects_pid_reuse_uid_and_command_changes(control):
    identity = control.identity(os.getpid())
    assert control.matches(identity)
    for field, value in [("starttime", -1), ("uid", -1), ("cmdline", ["different"] )]:
        changed = {**identity, field: value}
        assert not control.matches(changed)
        assert not control.signal_process(changed, 15)


def test_only_tagged_processes_are_stopped(control):
    token = "cpu-test-ownership"
    child = subprocess.Popen(["/bin/sleep", "30"],
                             env={**os.environ, "A100_GPU0123_CONTROL_TOKEN": token}, start_new_session=True)
    other = subprocess.Popen(["/bin/sleep", "30"], start_new_session=True)
    try:
        control.stop_owned(token)
        assert child.wait(timeout=3) != 0
        assert other.poll() is None
    finally:
        for process in (child, other):
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=3)


def test_holder_ready_identity_reconciles_launch_snapshot(control, tmp_path, monkeypatch):
    from types import SimpleNamespace
    current = control.identity(os.getpid())
    ready = tmp_path / 'ready.json'
    ready.write_text(json.dumps({'identity': current}))
    child = SimpleNamespace(pid=current['pid'], poll=lambda: None)
    monkeypatch.setattr(control, 'has_token', lambda pid, token: token == 'owned')
    launched = {**current, 'cmdline': []}
    assert control.confirm_holder_identity(child, launched, ready, 'owned') == current
    assert control.confirm_holder_identity(child, None, ready, 'owned') == current
    for field, value in [('starttime', -1), ('uid', -1), ('pid', -1)]:
        with pytest.raises(RuntimeError, match='Holder identity'):
            control.confirm_holder_identity(child, {**current, field: value}, ready, 'owned')
    with pytest.raises(RuntimeError, match='Holder identity'):
        control.confirm_holder_identity(child, launched, ready, 'foreign')
    child.poll = lambda: 0
    with pytest.raises(RuntimeError, match='Holder identity'):
        control.confirm_holder_identity(child, launched, ready, 'owned')


@pytest.mark.parametrize("code,timeout,status", [(0, 5, "completed"), (1, 5, "failed"), (None, .05, "timed_out")])
def test_run_records_outcomes_and_reacquires_reservation(control, tmp_path, monkeypatch, code, timeout, status):
    manager = control.Controller(tmp_path)
    events = []
    monkeypatch.setattr(manager, "release", lambda: events.append("release"))
    monkeypatch.setattr(manager, "reserve", lambda: events.append("reserve"))
    monkeypatch.setattr(control, "assert_clear", lambda *args: None)
    command = ["/bin/sleep", "30" if code is None else "0" if code == 0 else "invalid-duration"]
    outcome = manager.run("trial", timeout, command)
    assert outcome["status"] == status
    assert events == ["release", "reserve"]
    assert json.loads((tmp_path / "runs/trial/outcome.json").read_text()) == outcome
    assert json.loads((tmp_path / "controller.json").read_text())["active"] is None
    with pytest.raises(FileExistsError):
        manager.run("trial", timeout, command)


def test_foreign_gpu_context_aborts_before_launch(control, tmp_path, monkeypatch):
    manager = control.Controller(tmp_path)
    monkeypatch.setattr(manager, "release", lambda: None)
    monkeypatch.setattr(manager, "reserve", lambda: None)
    def occupied(*args):
        raise RuntimeError("Foreign GPU process")
    monkeypatch.setattr(control, "assert_clear", occupied)
    outcome = manager.run("blocked", 5, ["/bin/sleep", "0"])
    assert outcome["status"] == "failed"
    assert "Foreign GPU process" in outcome["error"]


def test_departed_gpu_pid_is_requeried_before_foreign_verdict(control, monkeypatch):
    row={'gpu_uuid':next(iter(control.GPUS.values())),'pid':999999999}
    readings=iter([[row],[]])
    monkeypatch.setattr(control,'gpu_processes',lambda:next(readings))
    monkeypatch.setattr(control,'has_token',lambda *args:False)
    monkeypatch.setattr(control,'identity',lambda pid:None)
    monkeypatch.setattr(control.time,'sleep',lambda seconds:None)
    control.assert_clear('owned-token')


def test_live_foreign_pid_is_never_excused_as_driver_lag(control, monkeypatch):
    row={'gpu_uuid':next(iter(control.GPUS.values())),'pid':12345}
    monkeypatch.setattr(control,'gpu_processes',lambda:[row])
    monkeypatch.setattr(control,'has_token',lambda *args:False)
    monkeypatch.setattr(control,'identity',lambda pid:{'pid':pid,'uid':-1,'starttime':10})
    monkeypatch.setattr(control.time,'sleep',lambda seconds:(_ for _ in ()).throw(AssertionError('live foreign must fail immediately')))
    with pytest.raises(RuntimeError,match='Foreign GPU process'):
        control.assert_clear('owned-token')


def test_own_gpu_identity_survives_environment_teardown_but_not_pid_reuse(control, monkeypatch):
    row={'gpu_uuid':next(iter(control.GPUS.values())),'pid':12345}
    current={'pid':12345,'uid':os.getuid(),'starttime':10,'cmdline':['own-rank']}
    monkeypatch.setattr(control,'gpu_processes',lambda:[row])
    monkeypatch.setattr(control,'has_token',lambda *args:True)
    monkeypatch.setattr(control,'identity',lambda pid:dict(current))
    control.assert_clear('owned-token')
    monkeypatch.setattr(control,'has_token',lambda *args:False)
    control.assert_clear('owned-token')
    current['starttime']=11
    with pytest.raises(RuntimeError,match='Foreign GPU process'):
        control.assert_clear('owned-token')


def test_persistent_departed_gpu_pid_is_not_silently_ignored(control, monkeypatch):
    row={'gpu_uuid':next(iter(control.GPUS.values())),'pid':999999999}
    monkeypatch.setattr(control,'gpu_processes',lambda:[row])
    monkeypatch.setattr(control,'has_token',lambda *args:False)
    monkeypatch.setattr(control,'identity',lambda pid:None)
    monkeypatch.setattr(control.time,'sleep',lambda seconds:None)
    with pytest.raises(RuntimeError,match='GPU'):
        control.assert_clear('owned-token')


def test_gpu_environment_teardown_race_rechecks_process_identity(control, monkeypatch):
    row={'gpu_uuid':next(iter(control.GPUS.values())),'pid':12345}
    readings=iter([[row],[]])
    identities=iter([{'pid':12345,'uid':os.getuid(),'starttime':10,'cmdline':['own-rank']},None])
    monkeypatch.setattr(control,'gpu_processes',lambda:next(readings))
    monkeypatch.setattr(control,'owned_processes',lambda token:[])
    monkeypatch.setattr(control,'has_token',lambda *args:False)
    monkeypatch.setattr(control,'identity',lambda pid:next(identities,None))
    monkeypatch.setattr(control.time,'sleep',lambda seconds:None)
    control.assert_clear('owned-token')


def test_retained_gpu_owner_uses_starttime_uid_not_mutable_process_title(control, monkeypatch):
    row={'gpu_uuid':next(iter(control.GPUS.values())),'pid':12345}
    current={'pid':12345,'uid':os.getuid(),'starttime':10,'cmdline':['python-rank']}
    monkeypatch.setattr(control,'gpu_processes',lambda:[row])
    monkeypatch.setattr(control,'owned_processes',lambda token:[])
    monkeypatch.setattr(control,'has_token',lambda *args:True)
    monkeypatch.setattr(control,'identity',lambda pid:dict(current))
    control.assert_clear('owned-token')
    monkeypatch.setattr(control,'has_token',lambda *args:False)
    current['cmdline']=['SGLang::Scheduler_TP0']
    control.assert_clear('owned-token')


def test_foreign_verdict_includes_identity_for_diagnosis(control, monkeypatch):
    row={'gpu_uuid':next(iter(control.GPUS.values())),'pid':12345}
    current={'pid':12345,'uid':-1,'starttime':777,'cmdline':['foreign-workload']}
    monkeypatch.setattr(control,'gpu_processes',lambda:[row])
    monkeypatch.setattr(control,'owned_processes',lambda token:[])
    monkeypatch.setattr(control,'has_token',lambda *args:False)
    monkeypatch.setattr(control,'identity',lambda pid:dict(current))
    with pytest.raises(RuntimeError,match='foreign-workload'):
        control.assert_clear('owned-token')


def test_gpu_worker_with_live_token_owned_group_survives_environment_teardown(control, monkeypatch):
    row={'gpu_uuid':next(iter(control.GPUS.values())),'pid':12345}
    worker={'pid':12345,'uid':os.getuid(),'starttime':10,'pgid':321,'session':321,'cmdline':['rank']}
    leader={'pid':321,'uid':os.getuid(),'starttime':5,'pgid':321,'session':321,'cmdline':['torchrun']}
    monkeypatch.setattr(control,'gpu_processes',lambda:[row])
    monkeypatch.setattr(control,'owned_processes',lambda token:[])
    monkeypatch.setattr(control,'has_token',lambda pid,token:pid==321)
    monkeypatch.setattr(control,'identity',lambda pid:dict(worker if pid==12345 else leader))
    control.assert_clear('owned-token')


def test_mutation_reads_controller_state_after_lock_acquisition(control, tmp_path, monkeypatch):
    root=tmp_path/'campaign';root.mkdir()
    state=root/'controller.json'
    state.write_text(json.dumps({'holder':None,'active':None,'generation':1}))
    def acquire(*args):
        state.write_text(json.dumps({'holder':None,'active':None,'generation':2}))
    def reserve(manager):
        assert manager.state['generation']==2
    monkeypatch.setattr(control,'REPO',tmp_path)
    monkeypatch.setattr(control.fcntl,'flock',acquire)
    monkeypatch.setattr(control.Controller,'reserve',reserve)
    monkeypatch.setattr(sys,'argv',['control.py','--root',str(root),'reserve'])
    assert control.main()==0


def test_interrupted_run_is_recovered_without_overwriting_existing_outcome(control, tmp_path, monkeypatch):
    manager = control.Controller(tmp_path)
    path = tmp_path / "runs/interrupted"
    path.mkdir(parents=True)
    manager.state["active"] = {"label": "interrupted", "token": "already-dead", "command": ["old"]}
    manager.save()
    manager.recover()
    outcome = json.loads((path / "outcome.json").read_text())
    assert outcome["status"] == "interrupted"
    assert manager.state["active"] is None
    manager.recover()
    assert json.loads((path / "outcome.json").read_text()) == outcome


def test_user_suspension_refuses_reserve_and_run_before_any_gpu_action(control, tmp_path, monkeypatch):
    manager = control.Controller(tmp_path)
    sentinel = tmp_path / "RESERVATION_DISABLED.json"
    sentinel.write_text('{"reason":"GPUs returned to other users"}')
    def unexpected(*args):
        raise AssertionError("GPU checks/changes must not occur while disabled")
    monkeypatch.setattr(control, "assert_clear", unexpected)
    for action in (manager.reserve, lambda: manager.run("forbidden", 5, [sys.executable, "-c", "pass"])):
        with pytest.raises(RuntimeError, match="suspended"):
            action()
    assert sentinel.exists()
    assert not (tmp_path / "runs/forbidden").exists()


def test_persistent_suspension_state_also_blocks_without_marker(control, tmp_path, monkeypatch):
    manager = control.Controller(tmp_path)
    manager.state["reservation_suspended"] = True
    def unexpected(*args):
        raise AssertionError("GPU checks/changes must not occur in CPU tests")
    monkeypatch.setattr(control, "assert_clear", unexpected)
    with pytest.raises(RuntimeError, match="suspended"):
        manager.reserve()


def test_changing_root_cannot_bypass_global_user_suspension(control, tmp_path):
    control.GLOBAL_SUSPENSION.write_text('{}')
    manager = control.Controller(tmp_path / "another-namespace")
    with pytest.raises(RuntimeError, match="suspended"):
        manager.reserve()
