# A100 Repository Cleanup Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement these tasks and review the combined changes before completion.

**Goal:** Make the repository's deployment entry points, configuration examples, and hardware test fixtures consistent with the current A100 server.

**Architecture:** Reuse the existing calibrated A100 configuration and Llama-3.1-8B TP1/PP2 launcher. Remove obsolete device-specific configurations and measurements, and preserve hardware-independent runtime code and existing A100 evidence.

**Tech Stack:** Python, Bash, YAML, PyTorch, SGLang, CUDA Green Contexts.

**Spec:** The user's request to adapt the repository to its current A100 server and remove obsolete hardware content.

## Constraints and review focus

- Current hardware: A100-SXM4-80GB, compute capability 8.0, 108 SMs; default deployment uses physical GPUs 6 and 7.
- Work in the requested checkout and preserve all pre-existing uncommitted runtime changes and A100 artifacts.
- Do not relabel old benchmark measurements as new hardware results.
- Preserve generic multiarchitecture support needed by upstream SGLang.
- Entry points must resolve repository/model paths independently of the caller's working directory and forward user arguments.
- Preserve existing correctness limitations in experiment documentation.

## Tasks

- [x] Remove obsolete device configs, tuning JSON files, and benchmark records; remove stale device mentions from comments and documentation.
- [x] Route deployment entry points to the existing A100 launcher and update benchmark defaults and local smoke configuration to the available model.
- [x] Update README, PDMux documentation, and run instructions; convert PDMux SM test fixtures to A100 partitions.
- [x] Run targeted runtime/tool tests, shell checks, YAML validation, hardware smoke checks, and a residual-content audit; obtain an independent review.

## Verification record

- Runtime/tool regression: 207 passed and 29 subtests passed; the two pre-existing environment warnings remain.
- All four A100 YAML files validated; seven Green Context partition pairs executed correctly on each of two A100 GPUs.
- Calibrated FlashInfer and local Triton launchers each passed all six HTTP/API smoke checks, including 16 concurrent requests.
- The benchmark entry point completed four requests and generated 64 tokens, using local synthetic input without dataset downloads.
- Shell syntax, Python compilation, local documentation links, residual-content search, and diff whitespace checks passed.
- The four pre-existing modified runtime files match their original A100 snapshot byte-for-byte.
- Independent review found two superseded root chunk scripts; both were removed. Baseline experiments remain available through the A100 experiment runner.
- Detailed evidence: `run/a100/repository-cleanup/verification.json`.

## Implementation decisions

- Reuse the measured A100 calibration; preserve raw numerical results and their correctness limitations.
- Use text decoded from synthetic token IDs for the quick benchmark. The installed benchmark path with raw NumPy token IDs is not JSON serializable; text requests passed the actual API run.
- Refresh the existing evidence archive's documentation and local entry points, retain its original experiment data, and regenerate its SHA256 receipt.
