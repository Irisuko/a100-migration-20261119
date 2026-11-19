"""Synthetic Kineto traces test attribution separately from stream concurrency."""

import gzip
import json
from pathlib import Path
import tempfile
import unittest

import analyze_gpu_trace as analyzer


def event(name, category, start, duration, pid=100, tid=1, **args):
    return {"name": name, "cat": category, "ph": "X", "ts": start, "dur": duration,
            "pid": pid, "tid": tid, "args": args}


def lane(name, start, duration, phase=None):
    return event(f"pdmux_pp|pp=0|tp=0|lane={name}|partition=3|phase={phase or name}",
                 "user_annotation", start, duration)


def kernel(start, duration, stream, device=0, **args):
    return event("some_kernel", "kernel", start, duration, pid=device, tid=stream,
                 device=device, stream=stream, **args)


class TraceTests(unittest.TestCase):
    def setUp(self):
        self.module = analyzer

    def analyze(self, events):
        return self.module.analyze_trace({"traceEvents": events}, "test-PP-0-TP-0.trace.json.gz")

    def test_external_ids_measure_union_overlap_without_double_counting(self):
        result = self.analyze([
            lane("prefill", 0, 10), event("aten::mm", "cpu_op", 2, 2, **{"External id": 11}),
            lane("decode", 12, 8), event("aten::mm", "cpu_op", 13, 1, **{"External id": 12}),
            kernel(100, 20, 7, **{"External id": 11}), kernel(115, 20, 7, **{"External id": 11}),
            kernel(110, 20, 8, **{"External id": 12}), kernel(200, 10, 9),
        ])
        device = result["devices"]["0"]
        self.assertEqual(result["mapping"]["mapped_kernels"], 3)
        self.assertEqual(result["mapping"]["unmapped_kernels"], 1)
        self.assertEqual(device["lanes"]["prefill"]["kernel_sum_us"], 40)
        self.assertEqual(device["lanes"]["prefill"]["kernel_union_us"], 35)
        self.assertEqual(device["prefill_decode_overlap_us"], 20)
        self.assertEqual(device["generic_different_stream_overlap_us"], 20)

    def test_cuda_graph_launch_correlation_can_map_multiple_kernels(self):
        result = self.analyze([
            lane("decode", 0, 10),
            event("cudaGraphLaunch", "cuda_runtime", 2, 1, correlation=88),
            kernel(100, 10, 8, correlation=88), kernel(112, 10, 8, correlation=88),
        ])
        self.assertEqual(result["mapping"]["methods"]["correlation"], 2)
        self.assertEqual(result["devices"]["0"]["lanes"]["decode"]["kernel_union_us"], 20)

    def test_flow_maps_exact_endpoints_without_cross_device_overlap(self):
        result = self.analyze([
            lane("prefill", 0, 10), lane("decode", 20, 10),
            event("op", "cpu_op", 21, 1, **{"External id": 22}),
            {"name": "ac2g", "cat": "ac2g", "ph": "s", "id": 9, "pid": 100, "tid": 1, "ts": 3},
            {"name": "ac2g", "cat": "ac2g", "ph": "f", "id": 9, "pid": 0, "tid": 7, "ts": 100},
            kernel(100, 20, 7), kernel(105, 10, 8, device=1, **{"External id": 22}),
        ])
        self.assertEqual(result["mapping"]["methods"]["flow"], 1)
        self.assertIsNone(result["devices"]["0"]["prefill_decode_overlap_us"])
        self.assertIsNone(result["devices"]["1"]["prefill_decode_overlap_us"])

    def test_stream_overlap_without_lane_evidence_is_not_semantic_overlap(self):
        result = self.analyze([kernel(100, 10, 7), kernel(105, 10, 8)])
        device = result["devices"]["0"]
        self.assertEqual(device["generic_different_stream_overlap_us"], 5)
        self.assertFalse(device["semantic_overlap_available"])
        self.assertIsNone(device["prefill_decode_overlap_us"])
        self.assertTrue(result["limitations"])

    def test_unrelated_flows_cannot_assign_kernel_lanes(self):
        result = self.analyze([
            lane("prefill", 0, 10),
            {"name": "arbitrary", "cat": "dependency", "ph": "s", "id": 9, "pid": 100, "tid": 1, "ts": 3},
            {"name": "arbitrary", "cat": "dependency", "ph": "f", "id": 9, "pid": 0, "tid": 7, "ts": 100},
            kernel(100, 20, 7),
        ])
        self.assertEqual(result["mapping"]["mapped_kernels"], 0)

    def test_missing_device_or_stream_does_not_prove_same_gpu_overlap(self):
        for missing in ("device", "stream"):
            with self.subTest(missing=missing):
                first, second = kernel(100, 20, 7, **{"External id": 11}), kernel(105, 10, 8, **{"External id": 12})
                del first["args"][missing]
                del second["args"][missing]
                result = self.analyze([
                    lane("prefill", 0, 10), event("op", "cpu_op", 2, 1, **{"External id": 11}),
                    lane("decode", 20, 10), event("op", "cpu_op", 22, 1, **{"External id": 12}),
                    first, second,
                ])
                self.assertTrue(all(not device["semantic_overlap_available"] for device in result["devices"].values()))
                self.assertTrue(all(device["prefill_decode_overlap_us"] is None for device in result["devices"].values()))

    def test_conflicting_external_id_and_correlation_remain_ambiguous(self):
        result = self.analyze([
            lane("prefill", 0, 10), event("op", "cpu_op", 2, 1, **{"External id": 11}),
            lane("decode", 12, 8), event("cudaLaunchKernel", "cuda_runtime", 13, 1, correlation=88),
            kernel(100, 10, 7, correlation=88, **{"External id": 11}),
        ])
        self.assertEqual(result["mapping"]["ambiguous_kernels"], 1)
        self.assertEqual(result["mapping"]["mapped_kernels"], 0)

    def test_nested_head_uses_inner_phase_without_double_counting_lane(self):
        result = self.analyze([
            lane("prefill", 0, 20), lane("prefill", 5, 5, "prefill_head"),
            event("aten::mm", "cpu_op", 6, 1, **{"External id": 11}),
            kernel(100, 10, 7, **{"External id": 11}),
        ])
        self.assertEqual(result["devices"]["0"]["lanes"]["prefill"]["kernel_count"], 1)
        self.assertEqual(result["kernel_evidence"][0]["classification"]["phase"], "prefill_head")
        self.assertEqual(result["devices"]["0"]["lanes"]["prefill"].get("phase_stats"),
                         {"prefill_head": {"kernel_count": 1, "kernel_sum_us": 10, "kernel_union_us": 10}})

    def test_cli_reads_gzip_and_retains_each_pipeline_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            paths = []
            for stage in (0, 1):
                path = directory / f"profile-PP-{stage}-TP-0.trace.json.gz"
                with gzip.open(path, "wt") as target:
                    json.dump({"traceEvents": [kernel(100, 10, 7)]}, target)
                paths.append(str(path))
            output = directory / "analysis.json"
            self.assertEqual(self.module.main(["--trace", *paths, "--output", str(output)]), 0)
            report = json.loads(output.read_text())
            self.assertEqual([trace["pp_rank"] for trace in report["traces"]], [0, 1])
            self.assertTrue(report["completed"])


if __name__ == "__main__":
    unittest.main()
