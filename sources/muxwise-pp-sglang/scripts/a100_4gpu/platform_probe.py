#!/usr/bin/env python3
"""Bounded, four-rank A100 validation; run only on explicitly reserved GPUs.

CUDA_VISIBLE_DEVICES must contain four distinct GPU UUIDs. Example:
  CUDA_VISIBLE_DEVICES=GPU-...,GPU-...,GPU-...,GPU-... NCCL_DEBUG=INFO \
    NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P .venv/bin/python -m torch.distributed.run \
    --standalone --nproc-per-node=4 scripts/a100_4gpu/platform_probe.py \
    --output run/a100_4gpu/platform.json

Writes PATH.rank0.json through PATH.rank3.json as each stage completes, then
aggregates them into PATH. Failures preserve each rank's last completed stage.
This tests PyTorch's NCCL backend, not SGLang's custom all-reduce. Timings are
isolated diagnostic samples, not model throughput or overlap measurements.
The default workload is small enough to target <60 seconds on healthy hardware;
each distributed work item has a separate 60-second failure timeout.
"""

import argparse
import ctypes
from datetime import datetime, timedelta, timezone
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import sysconfig
import time
import traceback


MAX_BYTES = 32 * 1024 * 1024
DEFAULT_SIZES = "8,16384,262144,4194304,33554432"
TIMEOUT = timedelta(seconds=60)


def parse_sizes(value):
    try:
        sizes = [int(part.strip()) for part in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected comma-separated integer byte sizes") from error
    if not sizes or any(size < 2 or size > MAX_BYTES or size % 2 for size in sizes):
        raise argparse.ArgumentTypeError("BF16 byte sizes must be even and in [2, 33554432]")
    if len(set(sizes)) != len(sizes):
        raise argparse.ArgumentTypeError("Message sizes must be unique")
    return sorted(sizes)


def parse_green_pair(value):
    try:
        pair = [int(part.strip()) for part in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected SM counts such as 84,24") from error
    if len(pair) != 2 or min(pair) <= 0 or sum(pair) != 108:
        raise argparse.ArgumentTypeError("Two positive SM counts totaling 108 are required")
    return pair


def launch_config(environ):
    mask = environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(mask) != 4 or len(set(mask)) != 4 or any(not item.startswith("GPU-") for item in mask):
        raise ValueError("Set CUDA_VISIBLE_DEVICES to exactly four distinct reserved GPU UUIDs")
    rank = int(environ.get("RANK", "-1"))
    local_rank = int(environ.get("LOCAL_RANK", "-1"))
    world_size = int(environ.get("WORLD_SIZE", "0"))
    local_world_size = int(environ.get("LOCAL_WORLD_SIZE", "0"))
    if world_size != 4 or local_world_size != 4 or rank != local_rank or not 0 <= rank < 4:
        raise ValueError("Use a single-node torchrun launch with --nproc-per-node=4")
    return rank, local_rank, world_size, mask


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def check(condition, message):
    # Explicit checks remain active under python -O.
    if not condition:
        raise RuntimeError(message)


def wait(work):
    check(work.wait(timeout=TIMEOUT), "Distributed work did not complete")


def barrier(dist, local_rank):
    wait(dist.barrier(device_ids=[local_rank], async_op=True))


def timed_operation(torch, operation):
    """Exclude payload initialization, include synchronization in wall time."""
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    finish = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    operation()
    finish.record()
    finish.synchronize()
    return start.elapsed_time(finish), (time.perf_counter() - wall_start) * 1000


def timing_result(gpu_ms, wall_ms, nbytes):
    median = statistics.median(gpu_ms)
    return {
        "bytes": nbytes,
        "bf16_elements": nbytes // 2,
        "gpu_ms": gpu_ms,
        "wall_ms": wall_ms,
        "median_gpu_ms": median,
        "median_wall_ms": statistics.median(wall_ms),
        "payload_GB_per_s": nbytes / (median * 1e6) if median else None,
        "correct": True,
    }


def probe_collectives(torch, dist, rank, world_size, args, report, save):
    device = torch.device("cuda", rank)
    report["all_reduce"] = []
    report["pp_ring"] = []
    for nbytes in args.sizes:
        payload = torch.empty(nbytes // 2, dtype=torch.bfloat16, device=device)
        gpu_ms, wall_ms = [], []
        for iteration in range(args.warmup + args.iterations):
            offset = iteration % 4
            payload.fill_(rank + 1 + offset)
            elapsed, wall = timed_operation(
                torch, lambda: wait(dist.all_reduce(payload, async_op=True))
            )
            expected = world_size * (world_size + 1) // 2 + world_size * offset
            check(bool(torch.all(payload == expected).item()), f"All-reduce mismatch: {nbytes} bytes, rank {rank}")
            if iteration >= args.warmup:
                gpu_ms.append(elapsed)
                wall_ms.append(wall)
        row = timing_result(gpu_ms, wall_ms, nbytes)
        row["expected_last_value"] = expected
        row["algorithmic_GB_per_s"] = row.pop("payload_GB_per_s")
        row["ring_bus_GB_per_s_estimate"] = (
            row["algorithmic_GB_per_s"] * 2 * (world_size - 1) / world_size
            if row["algorithmic_GB_per_s"] is not None else None
        )
        report["all_reduce"].append(row)
        save()
        del payload

    # All four ranks participate: 0->1->2->3->0. Report simultaneous ring
    # exchanges separately from collectives; these are not isolated link tests.
    previous_rank, next_rank = (rank - 1) % world_size, (rank + 1) % world_size
    for nbytes in args.sizes:
        send = torch.empty(nbytes // 2, dtype=torch.bfloat16, device=device)
        receive = torch.empty_like(send)

        def exchange():
            requests = dist.batch_isend_irecv([
                dist.P2POp(dist.isend, send, next_rank),
                dist.P2POp(dist.irecv, receive, previous_rank),
            ])
            for request in requests:
                wait(request)

        gpu_ms, wall_ms = [], []
        for iteration in range(args.warmup + args.iterations):
            offset = iteration % 4
            send.fill_(rank + 1 + offset)
            receive.fill_(-1)
            elapsed, wall = timed_operation(torch, exchange)
            expected = previous_rank + 1 + offset
            check(bool(torch.all(receive == expected).item()), f"P2P mismatch: {nbytes} bytes, rank {rank}")
            if iteration >= args.warmup:
                gpu_ms.append(elapsed)
                wall_ms.append(wall)
        row = timing_result(gpu_ms, wall_ms, nbytes)
        row.update(send_to=next_rank, receive_from=previous_rank, expected_last_value=expected)
        report["pp_ring"].append(row)
        save()
        del send, receive


def probe_stream_graph(torch, stream):
    """Test actual BF16 add execution and changed-input graph replay."""
    a = torch.full((4096,), 1, device=stream.device, dtype=torch.bfloat16)
    b = torch.full_like(a, 2)
    output = torch.empty_like(a)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.add(a, b, out=output)
    stream.synchronize()
    check(bool(torch.all(output == 3).item()), "Stream add result mismatch")
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            torch.add(a, b, out=output)
        for value in (4, 7, 11):
            with torch.cuda.stream(stream):
                a.fill_(value)
                output.fill_(-1)
                graph.replay()
            stream.synchronize()
            check(bool(torch.all(output == value + 2).item()), "Changed-input graph replay mismatch")
    finally:
        stream.synchronize()
        graph.reset()
    return {"stream_add_correct": True, "graph_replays_correct": 3, "dtype": "bfloat16", "elements": 4096}


class GreenCleanup:
    """Recover opaque handles without changing the spatial extension's ABI."""

    def __init__(self):
        self.driver = ctypes.CDLL("libcuda.so.1")
        self.lookup = getattr(self.driver, "cuStreamGetGreenCtx", None)
        self.lookup_v2 = getattr(self.driver, "cuStreamGetCtx_v2", None)
        if self.lookup is None and self.lookup_v2 is None:
            raise RuntimeError("Driver lacks a green-handle query API; refusing uncleanable contexts")
        pointer = ctypes.c_void_p
        if self.lookup is not None:
            self.lookup.argtypes = [pointer, ctypes.POINTER(pointer)]
            self.lookup.restype = ctypes.c_int
        if self.lookup_v2 is not None:
            self.lookup_v2.argtypes = [pointer, ctypes.POINTER(pointer), ctypes.POINTER(pointer)]
            self.lookup_v2.restype = ctypes.c_int
        self.destroy_stream = self.driver.cuStreamDestroy_v2
        self.destroy_context = self.driver.cuGreenCtxDestroy
        for function in (self.destroy_stream, self.destroy_context):
            function.argtypes = [pointer]
            function.restype = ctypes.c_int

    def get_context(self, stream):
        green, primary = ctypes.c_void_p(), ctypes.c_void_p()
        if self.lookup is not None:
            result = self.lookup(ctypes.c_void_p(stream), ctypes.byref(green))
        else:
            result = self.lookup_v2(ctypes.c_void_p(stream), ctypes.byref(primary), ctypes.byref(green))
        check(result == 0 and bool(green.value), f"Green context lookup failed with CUDA result {result}")
        return green.value

    def close(self, streams, contexts):
        errors = []
        for stream in streams:
            result = self.destroy_stream(ctypes.c_void_p(stream))
            if result:
                errors.append(f"cuStreamDestroy_v2={result}")
        for context in set(contexts):
            result = self.destroy_context(ctypes.c_void_p(context))
            if result:
                errors.append(f"cuGreenCtxDestroy={result}")
        check(not errors, "; ".join(errors))


def probe_green(torch, rank, pair, result, save):
    cleanup = GreenCleanup()
    # Loading just the extension avoids unrelated sgl_kernel package imports.
    spec = importlib.util.find_spec("sgl_kernel")
    check(spec is not None and bool(spec.submodule_search_locations), "sgl_kernel package is missing")
    package = Path(next(iter(spec.submodule_search_locations)))
    preferred = package / f"spatial_ops{sysconfig.get_config_var('EXT_SUFFIX')}"
    libraries = [preferred] if preferred.is_file() else sorted(package.glob("spatial_ops*.so"))
    check(bool(libraries), "spatial_ops extension missing; see scripts/build_spatial_local.py")
    torch.ops.load_library(str(libraries[0]))
    result.update(extension=str(libraries[0]), requested_sm=pair, cleanup_complete=False)
    values = list(torch.ops.sgl_kernel.create_greenctx_stream_by_value(*pair, rank))
    check(len(values) >= 2, "Spatial op did not return two stream handles")
    streams, contexts = values[:2], []
    try:
        contexts.extend(cleanup.get_context(stream) for stream in streams)
        check(len(values) >= 4, "Spatial op did not report actual SM allocations")
        result["actual_sm"] = values[2:4]
        save()
        check(values[2:4] == pair, f"Requested SM split {pair}, got {values[2:4]}")
        result["streams"] = []
        for pointer in streams:
            stream = torch.cuda.ExternalStream(pointer, device=rank)
            result["streams"].append(probe_stream_graph(torch, stream))
            del stream  # ExternalStream does not own or destroy the native handle.
            save()
    finally:
        torch.cuda.synchronize(rank)
        cleanup.close(streams, contexts)
        result["cleanup_complete"] = len(contexts) == len(streams)
        save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sizes", type=parse_sizes, default=DEFAULT_SIZES, help="Comma-separated BF16 payload byte sizes (maximum 32 MiB)")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--green-pair", type=parse_green_pair, default="84,24")
    parser.add_argument("--skip-green", action="store_true", help="Explicitly skip spatial validation; report remains partial")
    parser.add_argument("--dry-run", action="store_true", help="Validate launch configuration without importing torch or initializing CUDA")
    args = parser.parse_args(argv)
    if not 0 <= args.warmup <= 10 or not 1 <= args.iterations <= 20:
        parser.error("Use 0..10 warmups and 1..20 measured iterations")
    try:
        rank, local_rank, world_size, mask = launch_config(os.environ)
    except ValueError as error:
        parser.error(str(error))
    if args.dry_run:
        print(json.dumps({"rank": rank, "visible_gpu_uuids": mask, "sizes": args.sizes, "green_pair": args.green_pair}))
        return

    # Importing/running this module for CPU validation never touches CUDA.
    import torch
    import torch.distributed as dist

    output = args.output.resolve()
    rank_path = output.with_name(output.name + f".rank{rank}.json")
    started = time.perf_counter()
    report = {
        "schema_version": 1,
        "run_id": os.environ.get("TORCHELASTIC_RUN_ID"),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "pid": os.getpid(),
        "cuda_visible_devices": mask,
        "selected_gpu_uuid": mask[local_rank],
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "environment": {key: value for key, value in os.environ.items() if key.startswith(("NCCL_", "TORCH_NCCL_"))},
        "communication_backend": "torch.distributed NCCL",
        "custom_all_reduce_validated": False,
        "distributed_timeout_seconds": int(TIMEOUT.total_seconds()),
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iterations,
        "status": "running",
        "stage": "device",
        "limitations": [
            "No custom all-reduce validation or inference workload is performed.",
            "Ring P2P exchanges are simultaneous and do not isolate individual physical links.",
            "CUDA graph checks cover local BF16 addition, not captured NCCL collectives.",
            "Bandwidth is diagnostic; timing includes stream waits and host launch gaps.",
        ],
    }

    def save():
        report["elapsed_seconds"] = time.perf_counter() - started
        write_json(rank_path, report)

    initialized = False
    try:
        save()
        if rank == 0:
            # Replace an earlier run's success before touching the hardware.
            write_json(output, {"schema_version": 1, "status": "running", "run_id": report["run_id"]})
        check(torch.cuda.device_count() == 4, "Expected exactly four visible GPUs")
        torch.cuda.set_device(local_rank)
        props = torch.cuda.get_device_properties(local_rank)
        report["device"] = {
            "name": props.name, "sm_count": props.multi_processor_count,
            "compute_capability": [props.major, props.minor], "total_memory_bytes": props.total_memory,
            "reported_uuid": str(getattr(props, "uuid", "unavailable")),
        }
        check("A100" in props.name and (props.major, props.minor) == (8, 0) and props.multi_processor_count == 108,
              f"Expected A100 sm80 with 108 SMs, got {report['device']}")
        report["peer_access"] = {
            str(peer): bool(torch.cuda.can_device_access_peer(local_rank, peer))
            for peer in range(world_size) if peer != local_rank
        }
        check(all(report["peer_access"].values()), "Not all GPU peers are accessible")
        check(torch.cuda.is_bf16_supported(including_emulation=False), "Native BF16 is unsupported")
        report["nccl_version"] = list(torch.cuda.nccl.version())
        save()

        report["stage"] = "nccl"
        save()
        dist.init_process_group("nccl", timeout=TIMEOUT, device_id=torch.device("cuda", local_rank))
        initialized = True
        barrier(dist, local_rank)
        probe_collectives(torch, dist, rank, world_size, args, report, save)
        barrier(dist, local_rank)

        report["stage"] = "cuda_graph"
        save()
        report["cuda_graph"] = probe_stream_graph(torch, torch.cuda.Stream(device=local_rank))
        save()
        report["stage"] = "green_context"
        report["green_context"] = {"status": "skipped" if args.skip_green else "running"}
        save()
        if not args.skip_green:
            probe_green(torch, local_rank, args.green_pair, report["green_context"], save)
            report["green_context"]["status"] = "passed"
        report["peak_torch_allocated_bytes"] = torch.cuda.max_memory_allocated(local_rank)
        report["status"] = "partial" if args.skip_green else "passed"
        report["stage"] = "complete"
        save()
        barrier(dist, local_rank)
        if rank == 0:
            ranks = [json.loads(output.with_name(output.name + f".rank{peer}.json").read_text()) for peer in range(world_size)]
            check(all(item["status"] == report["status"] for item in ranks), "Rank reports disagree")
            check(all(item["run_id"] == report["run_id"] for item in ranks), "Rank reports come from different launches")
            write_json(output, {"schema_version": 1, "status": report["status"], "custom_all_reduce_validated": False, "ranks": ranks})
            print(f"Platform probe {report['status']}: {output}", flush=True)
        barrier(dist, local_rank)
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        save()
        if rank == 0:
            write_json(output, {"schema_version": 1, "status": "failed", "run_id": report["run_id"], "rank0": report})
        print(f"Rank {rank} failed at {report['stage']}: {error}; evidence: {rank_path}", file=sys.stderr, flush=True)
        raise
    finally:
        if initialized:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
