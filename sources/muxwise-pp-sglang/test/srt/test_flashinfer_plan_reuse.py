"""CPU checks for diagnostics around FlashInfer's reused pinned plan buffer."""

from types import SimpleNamespace

import pytest


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_TORCH_COMPILE", "0")
    from sglang.srt.layers.attention import flashinfer_backend

    return flashinfer_backend


def test_pending_copy_is_reported_before_changed_host_plan_without_fence(backend, monkeypatch, caplog):
    helper = getattr(backend, "_checked_decode_plan", None)
    assert helper is not None, "Decode plan reuse diagnostic is missing"
    events, calls = [], []

    class Event:
        def __init__(self):
            self.done = False
            events.append(self)

        def record(self):
            calls.append("record")

        def query(self):
            calls.append("query")
            return self.done

        def synchronize(self):
            raise AssertionError("Observation-only diagnostic must not fence")

    monkeypatch.setattr(backend.torch.cuda, "Event", Event)
    wrapper = SimpleNamespace(device="cuda:1")

    def plan(value):
        calls.append(("plan", value))
        return value + 1

    assert helper(wrapper, plan, (0, 100, 200), 1) == 2
    assert helper(wrapper, plan, (0, 50, 200), 2) == 3
    assert calls == [("plan", 1), "record", "query", ("plan", 2), "record"]
    state = wrapper._pdmux_pp_plan_reuse
    assert state["calls"] == 2
    assert state["pending"] == 1
    assert state["pending_changed_indptr"] == 1
    assert state["pending_changed_batch"] == 0
    assert "previous_indptr=(0, 100, 200)" in caplog.text
    assert "next_indptr=(0, 50, 200)" in caplog.text


def test_retired_copy_and_equal_metadata_are_counted_separately(backend, monkeypatch):
    helper = getattr(backend, "_checked_decode_plan", None)
    assert helper is not None, "Decode plan reuse diagnostic is missing"

    class Event:
        done = False

        def record(self):
            pass

        def query(self):
            return self.done

    monkeypatch.setattr(backend.torch.cuda, "Event", Event)
    wrapper = SimpleNamespace(device="cuda:0")
    helper(wrapper, lambda: None, (0, 100))
    helper(wrapper, lambda: None, (0, 100))
    wrapper._pdmux_pp_plan_reuse["event"].done = True
    helper(wrapper, lambda: None, (0, 200, 300))
    state = wrapper._pdmux_pp_plan_reuse
    assert state["calls"] == 3 and state["pending"] == 1
    assert state["pending_changed_indptr"] == 0
    assert state["pending_changed_batch"] == 0


@pytest.mark.parametrize("diagnostic", [False, True])
def test_guard_finishes_previous_copy_before_host_buffer_is_rewritten(backend, monkeypatch, diagnostic):
    helper = getattr(backend, "_checked_decode_plan", None)
    assert helper is not None
    host_buffer, copied, ordering = [None], [], []

    class Event:
        def __init__(self):
            self.done = False

        def record(self):
            ordering.append("copy-enqueued")

        def query(self):
            return self.done

        def synchronize(self):
            copied.append(host_buffer[0])
            ordering.append("copy-finished")
            self.done = True

    monkeypatch.setattr(backend.torch.cuda, "Event", Event)
    wrapper = SimpleNamespace(device="cuda:1", _pdmux_pp_plan_guard=True)

    def plan(value):
        ordering.append(("host-write", value))
        host_buffer[0] = value
        return value

    assert helper(wrapper, plan, (0, 258) if diagnostic else None, 258) == 258
    assert helper(wrapper, plan, (0, 130) if diagnostic else None, 130) == 130
    assert copied == [258], "The first H2D must read its own plan, before the next CPU overwrite"
    assert ordering == [("host-write", 258), "copy-enqueued", "copy-finished",
                        ("host-write", 130), "copy-enqueued"]
    wrapper._pdmux_pp_plan_reuse["event"].done = True
    helper(wrapper, plan, (0, 131) if diagnostic else None, 131)
    assert copied == [258], "An already retired copy must not add a host wait"
