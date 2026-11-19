"""CPU-only checks that PP profiler exports cannot overwrite another stage."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang.srt.managers import scheduler_profiler_mixin as profiler_module
from sglang.srt.managers.scheduler_profiler_mixin import SchedulerProfilerMixin
from sglang.srt.model_executor.forward_batch_info import ForwardMode


def scheduler(tmp_path, pp_size, pp_rank, activities, tp_rank=0):
    instance = SchedulerProfilerMixin()
    instance.init_profiler()
    instance.pp_size, instance.pp_rank = pp_size, pp_rank
    instance.tp_rank, instance.tp_cpu_group = tp_rank, object()
    instance.profile_id = "shared-profile"
    instance.torch_profiler_output_dir = str(tmp_path)
    instance.torch_profiler_with_stack = False
    instance.torch_profiler_record_shapes = False
    instance.profiler_activities = activities
    instance.profile_in_progress = True
    return instance


@pytest.fixture(autouse=True)
def no_devices(monkeypatch):
    monkeypatch.setattr(profiler_module, "_is_npu", False)
    monkeypatch.setattr(profiler_module.torch.distributed, "barrier", Mock())
    monkeypatch.setattr(profiler_module.time, "time", lambda: 1234.5)


@pytest.mark.parametrize("forward_stage", [None, ForwardMode.DECODE])
def test_torch_exports_preserve_both_pp_stages(tmp_path, forward_stage):
    exported = []
    for pp_rank in (0, 1):
        instance = scheduler(tmp_path, 2, pp_rank, ["CPU", "GPU"])
        instance.torch_profiler = Mock()

        def export(path, rank=pp_rank):
            exported.append(Path(path))
            Path(path).write_text(f"stage-{rank}")

        instance.torch_profiler.export_chrome_trace.side_effect = export
        assert instance.stop_profile(forward_stage).success
    assert len(set(exported)) == 2
    assert [path.read_text() for path in exported] == ["stage-0", "stage-1"]
    for pp_rank, path in enumerate(exported):
        assert f"-PP-{pp_rank}-" in path.name
        assert "-TP-0" in path.name
        if forward_stage is not None:
            assert "-ForwardMode.DECODE.trace.json.gz" in path.name


def test_torch_pp1_filename_is_unchanged(tmp_path):
    instance = scheduler(tmp_path, 1, 0, ["CPU"], tp_rank=2)
    instance.torch_profiler = Mock()
    exported = instance.torch_profiler.export_chrome_trace
    assert instance.stop_profile().success
    exported.assert_called_once_with(str(tmp_path / "shared-profile-TP-2.trace.json.gz"))


@pytest.mark.parametrize("pp_size", [1, 2])
def test_memory_exports_include_pp_rank_only_when_needed(tmp_path, monkeypatch, pp_size):
    dump = Mock()
    monkeypatch.setattr(profiler_module.torch.cuda.memory, "_dump_snapshot", dump)
    monkeypatch.setattr(profiler_module.torch.cuda.memory, "_record_memory_history", Mock())
    for pp_rank in range(pp_size):
        instance = scheduler(tmp_path, pp_size, pp_rank, ["MEM"])
        assert instance.stop_profile(ForwardMode.DECODE).success
    names = [Path(call.args[0]).name for call in dump.call_args_list]
    assert len(set(names)) == pp_size
    if pp_size == 1:
        assert names == ["1234.5-TP-0-memory-ForwardMode.DECODE.pickle"]
    else:
        assert all(f"-PP-{rank}-" in name for rank, name in enumerate(names))


@pytest.mark.parametrize("pp_size", [1, 2])
def test_rpd_export_paths_include_pp_rank_only_when_needed(tmp_path, monkeypatch, pp_size):
    # A nonzero TP rank avoids RPD's database setup; the tracer itself is mocked.
    # This checks output naming without requiring ROCm or launching a profiler.
    monkeypatch.setitem(
        sys.modules, "rpdTracerControl", SimpleNamespace(rpdTracerControl=Mock())
    )
    names = []
    for pp_rank in range(pp_size):
        instance = scheduler(tmp_path, pp_size, pp_rank, ["RPD"], tp_rank=1)
        assert instance.start_profile().success
        names.append(Path(instance.rpd_profile_path).name)
    assert len(set(names)) == pp_size
    if pp_size == 1:
        assert names == ["rpd-1234.5-TP-1.trace.json.gz"]
    else:
        assert all(f"-PP-{rank}-" in name for rank, name in enumerate(names))
