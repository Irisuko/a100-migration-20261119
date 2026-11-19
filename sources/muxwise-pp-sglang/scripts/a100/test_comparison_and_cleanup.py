import json
import signal
from types import SimpleNamespace

import compare_outputs
import run_experiment


def row(key, ids, requested=2, error=None):
    return dict(fixture_id=key, request_id=key, output_ids=ids,
                requested_output_tokens=requested, actual_output_tokens=len(ids), error=error)


def test_missing_candidate_cannot_pass():
    reference = {"requests": [row("a", [1, 2]), row("b", [3, 4])]}
    candidate = {"requests": [row("a", [1, 2])]}
    result = compare_outputs.compare(reference, candidate)
    assert not result["passed"]
    assert result["missing_fixtures"] == {"b": 1}


def test_reordered_results_match_by_fixture_occurrence():
    reference = {"requests": [row("a", [1, 2]), row("b", [3, 4]), row("a", [1, 3])]}
    candidate = {"requests": [row("b", [3, 4]), row("a", [1, 2]), row("a", [1, 3])]}
    assert compare_outputs.compare(reference, candidate)["passed"]


def test_identically_truncated_or_failed_outputs_do_not_pass():
    truncated = {"requests": [row("a", [1])]}
    failed = {"requests": [row("a", [1, 2], error="aborted")]}
    assert not compare_outputs.compare(truncated, truncated)["passed"]
    assert not compare_outputs.compare(failed, failed)["passed"]


def test_cleanup_escalates_children_even_if_parent_already_exited(tmp_path, monkeypatch):
    args = SimpleNamespace(port=30100)
    server = run_experiment.Server(args, tmp_path)
    server.proc = SimpleNamespace(pid=43210, poll=lambda: 0, wait=lambda **kwargs: 0)
    sent = []
    alive = [True]

    def killpg(pgid, sig):
        assert pgid == 43210
        sent.append(sig)
        if sig == signal.SIGKILL:
            alive[0] = False

    clock = iter((0, 31, 32))
    monkeypatch.setattr(run_experiment.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(run_experiment, "active_group_members", lambda pgid: [43211] if alive[0] else [])
    monkeypatch.setattr(run_experiment.os, "killpg", killpg)
    monkeypatch.setattr(run_experiment, "gpu_snapshot", lambda: "GPU6 and7 free\n")
    server.stop()
    assert sent == [signal.SIGTERM, signal.SIGKILL]
    assert json.loads((tmp_path / "shutdown.json").read_text()) == {"remaining": [], "forced": True}
