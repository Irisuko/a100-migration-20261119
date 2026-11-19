"""Bounded startup calibration of decode graphs, before any request owns KV.

The table measures each partition directly, including long-context attention;
SM extrapolation is deliberately not used as a safety gate. No graph is captured
here and no request/cache/allocator state is allocated or published.
"""

import logging
import time

import torch
import torch.distributed as dist

from sglang.srt.model_executor.forward_batch_info import ForwardMode

logger = logging.getLogger(__name__)


def context_shapes(bs, mean_context, max_context):
    uniform = [mean_context] * bs
    skewed = [1] * bs
    remaining = bs * (mean_context - 1)
    for i in range(bs):
        extra = min(remaining, max_context - 1)
        skewed[i] += extra
        remaining -= extra
    assert remaining == 0
    return [uniform] if uniform == skewed else [uniform, skewed]


def decode_context_grid(max_context, extra_contexts=()):
    """Refine measured upper corners without removing the legacy coverage.

    Empty extras preserve the v4 grid exactly. Long contexts between 16k and
    32k can otherwise pay the full 32k estimate even when KV is scarce; extra
    measured points reduce that rounding without interpolation or extrapolation.
    """
    return sorted(
        {min(c, max_context) for c in (2048, 4096, 8192, 16384, 33792, *extra_contexts)}
    )


def profile_decode(scheduler, max_seconds=None):
    runner = scheduler.tp_worker.model_runner
    graph = runner.graph_runner
    if graph is None or not hasattr(runner, "decode_attn_backend_group"):
        raise ValueError("PP decode calibration requires partition CUDA graphs")
    if max_seconds is None:
        max_seconds = getattr(
            scheduler.pdmux_config, "pp_decode_profile_max_seconds", 55
        )
    started = time.monotonic()
    table = {}
    # Pad to captured sizes at runtime as well. Cover the physically feasible
    # context range, so a 31k request does not depend on SM/context extrapolation.
    sizes = sorted(
        set(
            [
                graph.capture_bs[0],
                *[b for b in graph.capture_bs if b in (4, 8, 16)],
                graph.max_bs,
            ]
        )
    )
    contexts = decode_context_grid(
        scheduler.model_config.context_len,
        scheduler.pdmux_config.pp_decode_extra_contexts,
    )
    mapping = runner.req_to_token_pool.req_to_token
    width = max(contexts)
    capacity = min(scheduler.max_total_num_tokens, runner.token_to_kv_pool.size)
    if capacity < 1:
        raise ValueError("PP decode calibration requires positive KV capacity")
    contexts_by_size = {}
    for bs in sizes:
        limit = min(width, capacity // bs)
        skipped = [context for context in contexts if context > limit]
        if skipped:
            logger.warning(
                "MuxWise-PP decode calibration skips infeasible contexts=%s "
                "batch_size=%s unique_KV_capacity=%s",
                skipped,
                bs,
                capacity,
            )
        # Keep the feasible frontier as a measured upper corner. In particular,
        # long single requests remain covered when larger batches cannot fit.
        contexts_by_size[bs] = sorted(
            {context for context in contexts if context <= limit}
            | ({limit} if limit > 0 else set())
        )
    saved_mapping = mapping[: graph.max_bs, :width].clone()
    saved = {
        name: getattr(graph, name).clone()
        for name in (
            "seq_lens",
            "req_pool_indices",
            "out_cache_loc",
            "positions",
            "num_token_non_padded",
        )
    }
    try:
        # Slot zero is padding; physical slots 1..capacity are unowned at
        # startup. Pack each shape separately so short rows never alias through
        # a maximum-context stride. Allocator/radix metadata stays untouched.
        locations = torch.arange(
            1,
            min(capacity, graph.max_bs * width) + 1,
            device=mapping.device,
            dtype=mapping.dtype,
        )
        graph.req_pool_indices.copy_(
            torch.arange(
                graph.max_bs, device=mapping.device, dtype=graph.req_pool_indices.dtype
            )
        )
        torch.cuda.synchronize()
        for idx, (_, decode_sm) in enumerate(scheduler.sm_counts):
            if not decode_sm:
                continue
            stream = scheduler.stream_groups[idx][1]
            backend = runner.decode_attn_backend_group[idx]
            records = []
            for bs in sizes:
                for context in contexts_by_size[bs]:
                    if not scheduler._pp_tp_all(
                        time.monotonic() - started <= max_seconds
                    ):
                        raise RuntimeError(
                            "PP decode calibration exceeded startup time limit"
                        )
                    seconds = 0.0
                    for shape in context_shapes(bs, context, width):
                        if scheduler.tp_size > 1:
                            dist.barrier(group=scheduler.tp_cpu_group)
                        with torch.cuda.stream(stream):
                            offset = 0
                            for row, length in enumerate(shape):
                                mapping[row, :length].copy_(
                                    locations[offset : offset + length]
                                )
                                # Decode writes the final token of each row,
                                # not the first token of its attention history.
                                graph.out_cache_loc[row] = offset + length
                                offset += length
                            graph.num_token_non_padded.fill_(bs)
                            cpu_lens = torch.tensor(shape, dtype=torch.int32)
                            graph.seq_lens[:bs].copy_(cpu_lens)
                            graph.positions[:bs].copy_(cpu_lens - 1)
                            backend.init_forward_metadata_replay_cuda_graph(
                                bs,
                                graph.req_pool_indices[:bs],
                                graph.seq_lens[:bs],
                                bs * context,
                                None,
                                ForwardMode.DECODE,
                                None,
                                seq_lens_cpu=cpu_lens,
                            )
                            replay = graph.graphs[f"{idx}_{bs}"].replay
                            replay()
                            start, end = torch.cuda.Event(
                                enable_timing=True
                            ), torch.cuda.Event(enable_timing=True)
                            start.record()
                            for _ in range(3):
                                replay()
                            end.record()
                        end.synchronize()
                        seconds = max(seconds, start.elapsed_time(end) / 3000)
                    records.append((bs, context, seconds))
            table[idx] = records
            logger.info(
                "MuxWise-PP decode calibration partition=%s SMs=%s samples=%s",
                idx,
                scheduler.sm_counts[idx],
                records,
            )
    finally:
        torch.cuda.synchronize()
        mapping[: graph.max_bs, :width].copy_(saved_mapping)
        for name, value in saved.items():
            getattr(graph, name).copy_(value)
        torch.cuda.synchronize()
    logger.info(
        "MuxWise-PP decode calibration elapsed=%.3fs", time.monotonic() - started
    )
    return table
