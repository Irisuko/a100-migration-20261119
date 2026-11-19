"""CPU AST regressions for UUID-bound custom all-reduce initialization."""
import ast
import os
from pathlib import Path
import re
from types import SimpleNamespace as NS
from typing import List, Optional, Union
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "python/sglang/srt/distributed/device_communicators"
UUIDS = ["GPU-01a10c02-8860-b872-3eeb-59004ed80ce2", "GPU-1c7d2944-1c3a-07a4-5810-ee0b6f269aaa",
         "GPU-dc982fa8-613f-9bb2-af81-fd679a5ac64a", "GPU-8bcee978-642e-88df-687f-bf5530de7edb"]


class Nvml:
    def __init__(self):
        self.events = []
        self.fail_init = False
        self.fail_lookup = False
        self.wrong_uuid = False

    def nvmlInit(self):
        self.events.append("init")
        if self.fail_init:
            raise RuntimeError("NVML initialization failed")

    def nvmlShutdown(self):
        self.events.append("shutdown")

    def nvmlDeviceGetHandleByUUID(self, value):
        if isinstance(value, bytes):
            value = value.decode()
        self.events.append(("lookup", value))
        if self.fail_lookup:
            raise RuntimeError("UUID lookup failed")
        return UUIDS.index(value)

    def nvmlDeviceGetUUID(self, handle):
        return UUIDS[(handle + 1) % len(UUIDS)] if self.wrong_uuid else UUIDS[handle]

    def nvmlDeviceGetIndex(self, handle):
        return handle


def resolver(nvml, hip=False):
    tree = ast.parse((BASE / "custom_all_reduce_utils.py").read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "resolve_physical_device_ids"]
    if not functions:
        return None
    scope = {"pynvml": nvml, "_is_hip": hip, "re": re, "Optional": Optional, "List": List}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "<uuid-resolver>", "exec"), scope)
    return scope["resolve_physical_device_ids"]


def constructor(nvml):
    tree = ast.parse((BASE / "custom_all_reduce.py").read_text())
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CustomAllreduce")
    body = [node for node in original.body if isinstance(node, ast.Assign)]
    body += [node for node in original.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"]
    cls = ast.ClassDef(name="CustomAllreduce", bases=[], keywords=[], body=body, decorator_list=[])
    class Device:
        def __init__(self, value):
            self.index = int(value.split(":")[-1])
    class Tensor:
        def __init__(self, value):
            self.value = value[0]
        def item(self):
            return self.value
    logs, p2p_calls = [], []
    def gather(items, own, group):
        for index, item in enumerate(items):
            item.value = index
    scope = dict(os=os, Union=Union, ProcessGroup=object, _is_cuda=True, _is_hip=False, custom_ar=True,
                 torch=NS(device=Device, tensor=lambda value, **kwargs: Tensor(value), int="int", uint8="uint8",
                          empty=lambda *args, **kwargs: object(), cuda=NS(device_count=lambda: 4)),
                 dist=NS(Backend=NS(NCCL="nccl"), get_backend=lambda group: "gloo", get_rank=lambda **kwargs: 0,
                         get_world_size=lambda **kwargs: 2, all_gather=gather),
                 ops=NS(meta_size=lambda: 0, init_custom_ar=lambda *args: 7, register_buffer=lambda *args: None),
                 in_the_same_node_as=lambda *args, **kwargs: [True, True],
                 is_full_nvlink=lambda ids, size: ids == [0, 1] and size == 2,
                 _can_p2p=lambda rank, size: p2p_calls.append((rank, size)) or True,
                 logger=NS(info=lambda fmt, *args: logs.append(fmt % args), warning=lambda *args: None),
                 resolve_physical_device_ids=resolver(nvml))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), "<custom-ar-init>", "exec"), scope)
    custom = scope["CustomAllreduce"]
    custom.create_shared_buffer = staticmethod(lambda *args, **kwargs: [111, 222])
    return custom, logs, p2p_calls


class UUIDMappingTests(unittest.TestCase):
    def test_uuid_bound_constructor_reaches_p2p_and_positive_enabled_marker(self):
        custom, logs, p2p = constructor(Nvml())
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ",".join(UUIDS)}):
            try:
                comm = custom(group=object(), device=0)
            except Exception as error:
                self.fail(f"UUID binding must initialize instead of silently falling back: {error}")
        self.assertFalse(comm.disabled)
        self.assertEqual(p2p, [(0, 2)])
        self.assertTrue(any("Custom allreduce enabled" in message for message in logs))

    def test_numeric_constructor_keeps_existing_path_without_nvml_lookup(self):
        nvml = Nvml()
        custom, _, p2p = constructor(nvml)
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}):
            comm = custom(group=object(), device=0)
        self.assertFalse(comm.disabled)
        self.assertEqual(nvml.events, [])
        self.assertEqual(p2p, [(0, 2)])

    def get_resolver(self, nvml, hip=False):
        function = resolver(nvml, hip)
        self.assertIsNotNone(function, "UUID resolver is missing")
        return function

    def test_full_uuid_order_maps_to_verified_physical_indices(self):
        nvml = Nvml()
        mapped = self.get_resolver(nvml)(",".join([UUIDS[2], UUIDS[0], UUIDS[3], UUIDS[1]]), 4)
        self.assertEqual(mapped, [2, 0, 3, 1])
        self.assertEqual(nvml.events[0], "init")
        self.assertEqual(nvml.events[-1], "shutdown")

    def test_numeric_and_unset_masks_do_not_touch_nvml_including_rocm(self):
        for hip in [False, True]:
            nvml = Nvml()
            function = self.get_resolver(nvml, hip)
            self.assertEqual(function("3, 1,0,2", 4), [3, 1, 0, 2])
            self.assertEqual(function(None, 4), [0, 1, 2, 3])
            self.assertEqual(function("", 4), [0, 1, 2, 3])
            self.assertEqual(nvml.events, [])

    def test_lookup_failure_releases_initialized_nvml(self):
        nvml = Nvml()
        nvml.fail_lookup = True
        with self.assertRaisesRegex(RuntimeError, "lookup"):
            self.get_resolver(nvml)(UUIDS[0], 1)
        self.assertEqual(nvml.events[-1], "shutdown")

    def test_init_failure_does_not_shutdown_uninitialized_nvml(self):
        nvml = Nvml()
        nvml.fail_init = True
        with self.assertRaisesRegex(RuntimeError, "initialization"):
            self.get_resolver(nvml)(UUIDS[0], 1)
        self.assertEqual(nvml.events, ["init"])

    def test_wrong_uuid_handle_cannot_enable_wrong_physical_gpu(self):
        nvml = Nvml()
        nvml.wrong_uuid = True
        with self.assertRaisesRegex(ValueError, "UUID"):
            self.get_resolver(nvml)(UUIDS[0], 1)
        self.assertEqual(nvml.events[-1], "shutdown")

    def test_abbreviated_uuid_and_rocm_uuid_are_rejected_without_nvml(self):
        for mask, hip in [("GPU-01a10c02", False), (UUIDS[0], True)]:
            nvml = Nvml()
            with self.assertRaises(ValueError):
                self.get_resolver(nvml, hip)(mask, 1)
            self.assertEqual(nvml.events, [])


if __name__ == "__main__":
    unittest.main()
