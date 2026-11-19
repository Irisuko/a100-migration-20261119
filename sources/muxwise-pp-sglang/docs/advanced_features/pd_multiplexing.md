# PD Multiplexing on A100

The current configuration targets A100-SXM4-80GB GPUs with 108 SMs and compute capability 8.0. The measured deployment uses two NVLink-connected GPUs, Llama-3.1-8B-Instruct BF16, and TP1/PP2. See [the local run guide](../../LOCAL_RUN.md) for environment setup and launch commands.

## Server arguments

| Argument | Purpose |
| --- | --- |
| `--enable-pdmux` | Enable prefill/decode multiplexing on Green Context streams. |
| `--pdmux-config-path <path>` | Load the SM partitions and pipeline settings from YAML. |
| `--sm-group-num 8` | Match the eight groups in the calibrated A100 YAML. |
| `--tp-size 1 --pp-size 2` | Use the measured tensor/pipeline layout. |
| `--pdmux-pp-policy alp` | Use the adaptive latency policy. |
| `--pdmux-pp-prefill-chunk-size 0` | Use whole-prefill; set `2048` for the measured CPP configuration. |

The [A100 launcher](../../scripts/run_a100.sh) supplies these arguments, FlashInfer attention, context 8192, and the calibrated token capacity and batch limits. Its default model path is `../models/Llama-3.1-8B-Instruct`, relative to the repository.

## YAML configuration

The deployment file is [configs/pdmux_pp_a100.yml](../../configs/pdmux_pp_a100.yml):

```yaml
# Six mixed groups plus the 108/0 and 0/108 endpoint groups.
sm_group_num: 8
# Entries are [prefill SMs, decode SMs, minimum decode batch].
# Every mixed partition sums to the A100's 108 SMs.
manual_divisions:
  - [100, 8, 1]
  - [92, 16, 1]
  - [84, 24, 5]
  - [76, 32, 17]
  - [60, 48, 17]
  - [52, 56, 17]
decode_bs_divisor: 32
split_forward_token_budget: 32768
pp_profile_decode: true
pp_decode_extra_contexts: [128, 512, 1024]
pp_prefetch: true
pp_prefetch_slots: 2
pp_async_decode: true
```

`sm_group_num` includes the two endpoint groups, so `manual_divisions` has `sm_group_num - 2` entries. Duplicate thresholds intentionally select `92/16` for static fallback batch 1–4 and `84/24` for batch 5–16, while retaining all candidates for ALP. Startup profiling measures the decode partitions; the extra contexts supplement the built-in context grid.

`decode_bs_divisor` controls default group selection when manual divisions are absent. `split_forward_token_budget` controls the number of prefill layers executed per split: `max(1, split_forward_token_budget // extend_num_tokens)`. Chunk size is a separate server argument.

The companion [calibration configuration](../../configs/pdmux_pp_a100_calibration.yml) retains the original candidate thresholds for experiments. [pdmux_pp_a100_cpp.yml](../../configs/pdmux_pp_a100_cpp.yml) has the same effective calibrated YAML values; the 2048-token CPP chunk is selected at launch. [pdmux_pp_a100_local.yml](../../configs/pdmux_pp_a100_local.yml) is the smaller `80/28` partition used by the lightweight Triton smoke launcher, and is not the calibrated performance setup.

The calibration applies to the measured model and workload. See [A100_EXPERIMENTS.md](../../A100_EXPERIMENTS.md) for the full protocol and results. **Strict token equality did not pass for all evaluated cases**; the remaining failures and numerical diagnostics are preserved in [the correctness report](../a100-correctness.md).
