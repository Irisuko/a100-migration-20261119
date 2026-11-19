# A100 Calibration and Mechanism Experiment Plan

> **For agentic workers:** Execute the bounded experiment protocol below. Root owns GPU experiment scheduling; parallel workers may implement independent tooling and review results.

**Goal:** Recalibrate MuxWise-PP for this server and establish real-model correctness and measured effects of its main mechanisms.

**Architecture:** Run ordinary SGLang PP2 and PDMux PP2 sequentially on the same two A100 GPUs using identical Llama-3.1-8B-Instruct weights and frozen request fixtures. Use startup CUDA-event calibration for each real SM partition and independently collect client latency, output IDs, and scheduler evidence.

**Tech Stack:** Repository 638aa794, Python 3.10.22, Torch 2.8.0+cu128, FlashInfer 0.4.0rc1, local CUDA 12.8, BF16.

**Spec:** User request: calibrate the configuration for this A100 server, complete correctness validation and key mechanism experiments. User selected Llama-3.1-8B, dual-GPU PP2.

## Global Constraints

- GPUs 6 and 7 only; check availability before starting. Do not change or stop other workloads.
- Do not load partial model shards. Verify model SHA256 checksums when the existing download completes.
- TP1/PP2, same model/tokenizer revision, BF16, FlashInfer, seed, request IDs, input IDs, token budget, and graph sizes across comparisons.
- Context limit 8192. Decode calibration batches 1/4/8/16, contexts 128/512/1024/2048/4096/8192. KV capacity 135168 tokens, exceeding 16*8192+1 to prevent calibration aliasing.
- Max running requests 32; max decode microbatch 16; graph max batch 16. YAML and CLI SM-group counts must match.
- A100 has 108 SMs. Initial measured mixed partitions: 100/8, 92/16, 84/24, 76/32, 60/48, 52/56. These are candidates, not claimed optimal settings.
- Main latency targets: TTFT 2 seconds, TPOT 50 milliseconds. A separately labeled 10/20/50 ms TPOT sweep may be used to expose dynamic policy choices; do not substitute a favorable target after seeing results.
- Preserve raw logs, fixtures, per-request timings/output IDs, calibration tables, dependency versions, and exact launch arguments under run/a100/.
- Stop experiment services after completion, respecting the prior request to leave GPUs 6/7 free.

## Review Focus

- Exact token equality is the primary correctness result; token mismatches remain failures until diagnosed, not silently excused with arbitrary tolerances.
- Baseline self-repeat measures BF16/batching sensitivity; use teacher-forced oracle scoring or HF for mismatch diagnosis at the same actual prefix.
- PDMux rejects return_logprob/hidden-state requests. Collect output_ids directly; request logprobs only from the ordinary baseline or HF.
- Cold/warm prefix-cache cases, chunk boundaries, long context, mixed concurrency, and request order need explicit coverage.
- An ablation counts as exercised only if its scheduler counters show activity. Startup calibration alone does not establish end-to-end benefit.

## Tasks

- [x] Prepare CUDA12.8/FlashInfer JIT; verify small attention kernels against torch SDPA.
- [x] Build fixed-token correctness and performance fixtures plus a streaming measurement client. Record complete requested output lengths, finish reasons, and SSE termination.
- [x] Confirm complete 8B weights and archive hashes/environment/GPU topology.
- [x] Establish ordinary PP2 oracle with repeated serial and concurrent cases; validate a small independent HF subset.
- [x] Measure full A100 partition/context/batch decode table on both stages, retaining raw CUDA-event timings and uncertainty across repeated startups.
- [x] Run PDMux correctness first with static scheduling, then ALP and token chunks. Investigate mismatches before performance claims.
- [x] Pilot chunk sizes 256/1024/2048 and token-layer slice budgets 2048/8192/32768 on a fixed held-out pilot workload; select settings using stated latency/throughput criteria and report the full sweep. Source inspection shows 65536 and 32768 both allow all 16 local layers for these chunk sizes, so 2048 replaces the redundant 65536 candidate before measurement; skip and document mathematically identical combinations.
- [x] Compare ordinary PP2, static PDMux, calibrated ALP, final chunked configuration, no-prefetch, and no-async-decode; use identical measurement fixtures and at least three repetitions for final comparisons.
- [x] Run targeted policy/SLO and token-credit/admission experiments only with enough load to activate their controls; report inactivity explicitly.
- [x] Produce a Chinese report with correctness verdicts, calibration tables, latency/throughput distributions, mechanism evidence, limits, and reproducible commands. Stop all experiment-owned services and verify GPU release.
- [x] Fix profiler artifact filename collisions between PP ranks (preserve PP1 names), then obtain separate GPU traces in a diagnostic run excluded from throughput results.

## Workloads and acceptance

Correctness includes varied natural prompts, exact chunk-boundary lengths 255/256/257, 511/512/513, 1023/1024/1025 and 2047/2048/2049, long inputs up to 6144 tokens, repeated prefixes, and staggered mixed concurrency. Use ignore_eos and compare output_ids of the requested length. Save baseline and variant outputs separately.

Performance uses seeded heterogeneous short/long prompts and output lengths, cold cache per trial, a fixed warmup/training phase excluded from timing, and both saturation and controlled-arrival workloads. Report completed requests, actual tokens, TTFT, inter-token/TPOT, end-to-end latency, throughput, errors, cache hits, and scheduler counters. This two-GPU shared-server study does not establish cross-hardware speedup, PP4 scaling, or production SLO guarantees.

Completion note: all planned measurements and validation were performed. Strict token equality remains non-passing; the final report preserves the 0.5-regret refill case confirmed by both HF and ordinary PP2. Completed checkboxes indicate work performed, not a blanket correctness pass. See [the final report](../../../A100_EXPERIMENTS.md).
