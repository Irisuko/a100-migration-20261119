"""CPU regressions for physically feasible startup calibration and sparse lookup."""

import math
from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest
import torch
import yaml

from sglang.srt.multiplex import pp_profile
from sglang.srt.multiplex.pdmux_context import load_pdmux_config
from sglang.srt.multiplex.pp_policy import calibrated_decode_time


@pytest.fixture
def cpu_profile(monkeypatch):
    """Keep the real profile loop/tensor mutations, replace CUDA execution only."""
    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            pass

        def synchronize(self):
            pass

        def elapsed_time(self, other):
            return 3.0

    monkeypatch.setattr(pp_profile.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(pp_profile.torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(pp_profile.torch.cuda, "Event", Event)

    def run(capacity=9, context=8, sizes=(1, 4), timeout=55, fail=False):
        maximum = max(sizes)
        mapping = torch.full((maximum, context), -19, dtype=torch.int32)
        original = mapping.clone()
        graph = NS(
            capture_bs=list(sizes),
            max_bs=maximum,
            seq_lens=torch.ones(maximum, dtype=torch.int32),
            req_pool_indices=torch.zeros(maximum, dtype=torch.int32),
            out_cache_loc=torch.zeros(maximum, dtype=torch.int64),
            positions=torch.zeros(maximum, dtype=torch.int64),
            num_token_non_padded=torch.zeros(1, dtype=torch.int32),
        )
        snapshots = []

        def replay(bs):
            lengths = graph.seq_lens[:bs].tolist()
            slots = [mapping[i, :n].tolist() for i, n in enumerate(lengths)]
            snapshots.append((lengths, slots, graph.out_cache_loc[:bs].tolist()))
            if fail:
                raise RuntimeError("synthetic replay failure")

        graph.graphs = {
            f"0_{bs}": NS(replay=lambda bs=bs: replay(bs)) for bs in sizes
        }
        backend = NS(
            init_forward_metadata_replay_cuda_graph=lambda *args, **kwargs: None
        )
        scheduler = NS(
            tp_worker=NS(
                model_runner=NS(
                    graph_runner=graph,
                    decode_attn_backend_group=[backend],
                    req_to_token_pool=NS(req_to_token=mapping),
                    token_to_kv_pool=NS(size=capacity, page_size=1),
                )
            ),
            model_config=NS(context_len=context),
            pdmux_config=NS(
                pp_decode_extra_contexts=[2], pp_decode_profile_max_seconds=timeout
            ),
            max_total_num_tokens=capacity,
            sm_counts=[(0, 108)],
            stream_groups=[(None, object())],
            tp_size=1,
            _pp_tp_all=lambda value: value,
        )
        before = {
            name: value.clone()
            for name, value in vars(graph).items()
            if isinstance(value, torch.Tensor)
        }
        try:
            table = pp_profile.profile_decode(scheduler)
        finally:
            assert torch.equal(mapping, original), "request mapping was not restored"
            for name, value in before.items():
                assert torch.equal(getattr(graph, name), value), name
        return table, snapshots

    return run


def test_profile_never_replays_more_unique_kv_than_physical_capacity(cpu_profile, caplog):
    table, snapshots = cpu_profile()
    assert all(sum(lengths) <= 9 for lengths, _, _ in snapshots)
    assert (1, 8) in [(bs, context) for bs, context, _ in table[0]]
    assert (4, 8) not in [(bs, context) for bs, context, _ in table[0]]
    assert "skip" in caplog.text.lower() and "capacity" in caplog.text.lower()


def test_feasible_short_batches_use_distinct_slots_and_write_the_last_token(cpu_profile):
    _, snapshots = cpu_profile()
    short_batches = [sample for sample in snapshots if sample[0] == [2, 2, 2, 2]]
    assert short_batches
    for _, rows, outputs in short_batches:
        assert rows == [[1, 2], [3, 4], [5, 6], [7, 8]]
        assert outputs == [2, 4, 6, 8]


def test_profile_adds_feasible_frontier_and_preserves_long_single_request(cpu_profile):
    table, snapshots = cpu_profile(capacity=40000, context=32768, sizes=(1, 4, 8, 16))
    pairs = {(bs, context) for bs, context, _ in table[0]}
    assert {(1, 32768), (4, 10000), (8, 5000), (16, 2500)} <= pairs
    for lengths, rows, outputs in snapshots:
        slots = [slot for row in rows for slot in row]
        assert len(slots) == len(set(slots)) == sum(lengths)
        assert min(slots) >= 1 and max(slots) <= 40000
        assert outputs == [row[-1] for row in rows]


def test_profile_restores_inputs_after_replay_failure(cpu_profile):
    with pytest.raises(RuntimeError, match="synthetic replay failure"):
        cpu_profile(fail=True)


def test_profile_uses_configured_timeout_instead_of_hardcoded_55(cpu_profile, monkeypatch):
    monkeypatch.setattr(
        pp_profile.time, "monotonic", iter([0] + [60] * 10000).__next__
    )
    table, _ = cpu_profile(timeout=600)
    assert table[0]


def test_profile_default_timeout_still_rejects_over_budget(cpu_profile, monkeypatch):
    monkeypatch.setattr(
        pp_profile.time, "monotonic", iter([0] + [60] * 10000).__next__
    )
    with pytest.raises(RuntimeError, match="time limit"):
        cpu_profile(timeout=55)


def test_sparse_lookup_never_invents_crossed_unmeasured_corner():
    rows = [(1, 2048, 0.001), (4, 4096, 0.005), (4, 8192, 0.020), (8, 6144, 0.015)]
    # Eligible corners are (4,8192) and (8,6144), never (4,6144).
    assert calibrated_decode_time(rows, 3, 5000) == 0.020
    assert math.isinf(calibrated_decode_time(rows, 9, 1))
    assert math.isinf(calibrated_decode_time(rows, 3, 9000))


def test_sparse_corner_keeps_conservative_prefix_envelope():
    rows = [(1, 2048, 0.001), (4, 4096, 0.025), (4, 8192, 0.020), (8, 6144, 0.015)]
    assert calibrated_decode_time(rows, 3, 5000) == 0.025


@pytest.mark.parametrize("timeout", [0, -1, True, "600", float("nan"), float("inf")])
def test_profile_timeout_rejects_unbounded_or_invalid_config(tmp_path, timeout):
    path = tmp_path / "config.yml"
    path.write_text(
        yaml.safe_dump({"sm_group_num": 8, "pp_decode_profile_max_seconds": timeout})
    )
    with pytest.raises(ValueError, match="pp_decode_profile_max_seconds"):
        load_pdmux_config(str(path))


def test_profile_timeout_defaults_and_override(tmp_path):
    path = tmp_path / "config.yml"
    path.write_text("sm_group_num: 8\n")
    assert load_pdmux_config(str(path)).pp_decode_profile_max_seconds == 55
    path.write_text("sm_group_num: 8\npp_decode_profile_max_seconds: 600\n")
    assert load_pdmux_config(str(path)).pp_decode_profile_max_seconds == 600
