"""CPU-only checks for optional ranges on the actual launch/timing thread."""

import threading
from types import SimpleNamespace

import pytest

from sglang.srt.multiplex import multiplexing_pp as pp


@pytest.fixture
def scheduler(monkeypatch):
    for name in ("SGLANG_PDMUX_PP_TIMING_TRACE", "SGLANG_PDMUX_PP_ROUND_TRACE"):
        monkeypatch.delenv(name, raising=False)
    return SimpleNamespace(pp_rank=1, tp_rank=0, _pp_stream_idx=3,
                           server_args=SimpleNamespace())


def test_flag_off_preserves_timing_samples_without_entering_profiler(scheduler, monkeypatch):
    monkeypatch.delenv("SGLANG_PDMUX_PP_PROFILE_RANGES", raising=False)
    sample = object()
    ended = []
    monkeypatch.setattr(pp.LaneTimings, "begin", lambda *args, **kwargs: sample)
    monkeypatch.setattr(pp.LaneTimings, "end", lambda self, value: ended.append(value))
    monkeypatch.setattr(pp.torch.profiler, "record_function",
                        lambda name: pytest.fail("Disabled ranges entered the profiler"))
    factory = getattr(pp, "_pp_lane_timings", None)
    assert factory is not None, "Diagnostic range factory is missing"
    timer = factory(scheduler)
    token = timer.begin("decode", [1, 2, 3])
    timer.end(token)
    assert type(timer) is pp.LaneTimings
    assert token is sample and ended == [sample]


def test_nested_head_ranges_run_on_the_launch_thread_and_keep_partition(scheduler, monkeypatch):
    monkeypatch.setenv("SGLANG_PDMUX_PP_PROFILE_RANGES", "1")
    events, samples = [], []
    monkeypatch.setattr(pp.LaneTimings, "begin", lambda self, kind, *args, **kwargs: kind)
    monkeypatch.setattr(pp.LaneTimings, "end", lambda self, value: samples.append(value))

    class Range:
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            events.append(("enter", self.name, threading.get_ident()))

        def __exit__(self, *args):
            events.append(("exit", self.name, threading.get_ident()))

    monkeypatch.setattr(pp.torch.profiler, "record_function", Range)
    factory = getattr(pp, "_pp_lane_timings", None)
    assert factory is not None, "Diagnostic range factory is missing"
    timer = factory(scheduler)

    def launch():
        prefill = timer.begin("prefill", [], layers=4)
        head = timer.begin("prefill_head", [])
        timer.end(head)
        timer.end(prefill)
        scheduler._pp_stream_idx = 4
        decode = timer.begin("decode", [])
        timer.end(decode)

    thread = threading.Thread(target=launch)
    thread.start()
    thread.join()
    assert samples == ["prefill_head", "prefill", "decode"]
    assert {event[2] for event in events} == {thread.ident}
    assert [event[:2] for event in events] == [
        ("enter", "pdmux_pp|pp=1|tp=0|lane=prefill|partition=3|phase=prefill"),
        ("enter", "pdmux_pp|pp=1|tp=0|lane=prefill|partition=3|phase=prefill_head"),
        ("exit", "pdmux_pp|pp=1|tp=0|lane=prefill|partition=3|phase=prefill_head"),
        ("exit", "pdmux_pp|pp=1|tp=0|lane=prefill|partition=3|phase=prefill"),
        ("enter", "pdmux_pp|pp=1|tp=0|lane=decode|partition=4|phase=decode"),
        ("exit", "pdmux_pp|pp=1|tp=0|lane=decode|partition=4|phase=decode"),
    ]
