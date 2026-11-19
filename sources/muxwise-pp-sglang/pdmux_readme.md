# MuxWise-PP on this A100 server

MuxWise-PP overlaps prefill and decode using CUDA Green Context SM partitions, pipeline prefetch, and asynchronous decode. The current deployment targets NVIDIA A100-SXM4-80GB GPUs with 108 SMs and compute capability 8.0. The server has eight GPUs connected by NVLink; the calibrated experiment uses physical GPUs **6 and 7**, TP1/PP2, and Llama-3.1-8B-Instruct in BF16.

## Environment and launch

The measured environment uses Python 3.10.22, PyTorch 2.8.0+cu128, FlashInfer 0.4.0rc1, SGLang 0.5.3rc0, sgl-kernel 0.3.12, and `$HOME/.local/cuda-12.8`. Follow [LOCAL_RUN.md](LOCAL_RUN.md) to prepare the environment and local model.

Run from the repository root:

```bash
scripts/run_a100.sh
```

The launcher uses [configs/pdmux_pp_a100.yml](configs/pdmux_pp_a100.yml), physical GPUs 6/7, FlashInfer attention, context 8192, and API model name `llama-3.1-8b`. It defaults to ALP whole-prefill (`PDMUX_POLICY=alp`, `PDMUX_CHUNK=0`). The root `start_pdmux.sh` forwards to this launcher.

The measured CPP setting uses a 2048-token chunk and the same 32768 token-layer budget:

```bash
PDMUX_CHUNK=2048 scripts/run_a100.sh
```

`MODEL_PATH`, `HOST`, `PORT`, `PDMUX_POLICY`, and `PDMUX_CHUNK` can be set through the environment; additional server arguments can be appended. The launcher checks that GPUs 6 and 7 have no active compute processes. Stop a foreground service with Ctrl-C.

## SM partitions and calibration

The [A100 YAML](configs/pdmux_pp_a100.yml) contains six mixed prefill/decode partitions: `100/8`, `92/16`, `84/24`, `76/32`, `60/48`, and `52/56`. The two endpoint groups are `108/0` and `0/108`, for eight groups in total. All partitions sum to the A100's 108 SMs.

Static fallback thresholds are `[1, 1, 5, 17, 17, 17]`: decode batch 1–4 uses `92/16`; batch 5–16 uses `84/24`. Repeated thresholds intentionally skip candidates in the static fallback, while ALP retains all candidates. These settings come from repeated decode calibration for the documented model, context range, and 50 ms TPOT target; they do not guarantee an end-to-end SLO. See [the configuration reference](docs/advanced_features/pd_multiplexing.md) and [the measured calibration](A100_EXPERIMENTS.md#校准).

## Validation and benchmarks

With the service running, exercise the HTTP interface and save a new result file:

```bash
.venv/bin/python scripts/smoke_local.py \
  --model llama-3.1-8b --output run/a100-current-smoke.json
bash bench_pdmux.sh
```

The benchmark uses a synthetic `random-ids` workload and the local model tokenizer, without downloading a dataset. It decodes the sampled IDs into text; server-side retokenization may change the input length slightly. It is a quick measurement; the complete frozen comparison protocol, calibration results, and mechanism experiments are in [A100_EXPERIMENTS.md](A100_EXPERIMENTS.md).

**Strict token equality did not pass for all evaluated cases.** Two runtime issues were fixed, but concurrent token mismatches and an independently confirmed numerical deviation remain documented in [the correctness report](docs/a100-correctness.md). HTTP smoke checks and benchmark completion do not replace those numerical checks. Existing raw A100 records under `run/a100/` remain the evidence for the published results.
