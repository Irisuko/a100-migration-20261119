#!/usr/bin/env bash
# Calibrated Llama-3.1-8B-Instruct serving on this server's physical GPUs 6/7.
set -euo pipefail
A100_REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$A100_REPO_DIR"
A100_MODEL_PATH="${MODEL_PATH:-$A100_REPO_DIR/../models/Llama-3.1-8B-Instruct}"
if [[ ! -d "$A100_MODEL_PATH" ]]; then
  echo "Model directory does not exist: $A100_MODEL_PATH" >&2
  exit 1
fi
if compgen -G "$A100_MODEL_PATH/*.aria2" >/dev/null; then
  echo "Model download is incomplete: $A100_MODEL_PATH" >&2
  exit 1
fi
A100_ACTIVE_PROCESSES="$(nvidia-smi -i 6,7 --query-compute-apps=gpu_uuid,pid --format=csv,noheader)"
if [[ -n "$A100_ACTIVE_PROCESSES" ]]; then
  echo "GPUs 6/7 have active compute processes; stop the intended service before launching." >&2
  echo "$A100_ACTIVE_PROCESSES" >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES=6,7
export CUDA_HOME="$HOME/.local/cuda-12.8"
export PATH="$A100_REPO_DIR/.venv/bin:$CUDA_HOME/bin:$PATH"
export FLASHINFER_CUDA_ARCH_LIST=8.0
export FLASHINFER_WORKSPACE_BASE="$A100_REPO_DIR/run/a100/flashinfer-cache"
export OMP_NUM_THREADS=4
export MAX_JOBS=4
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NO_PROXY="localhost,127.0.0.1,::1${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
unset SGLANG_PDMUX_PP_NUMERIC_TRACE SGLANG_PDMUX_PP_TIMING_TRACE
unset SGLANG_PDMUX_PP_ROUND_TRACE SGLANG_PDMUX_PP_PROFILE_RANGES
unset SGLANG_PDMUX_PP_DECODE_PLAN_REUSE_CHECK
exec "$A100_REPO_DIR/.venv/bin/python" -m sglang.launch_server \
  --model-path "$A100_MODEL_PATH" --served-model-name llama-3.1-8b \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-30000}" \
  --tp-size 1 --pp-size 2 --dtype bfloat16 \
  --attention-backend flashinfer --sampling-backend pytorch \
  --context-length 8192 --max-total-tokens 135168 \
  --max-running-requests 32 --max-micro-batch-size 16 --cuda-graph-max-bs 16 \
  --max-prefill-tokens 8192 --mem-fraction-static 0.65 \
  --disable-overlap-schedule --chunked-prefill-size -1 --random-seed 20261114 \
  --enable-pdmux --sm-group-num 8 \
  --pdmux-config-path "$A100_REPO_DIR/configs/pdmux_pp_a100.yml" \
  --pdmux-pp-policy "${PDMUX_POLICY:-alp}" \
  --pdmux-pp-prefill-chunk-size "${PDMUX_CHUNK:-0}" \
  --pdmux-pp-slo-ttft 2 --pdmux-pp-slo-tpot 0.05 \
  --pdmux-pp-summary-interval 5 \
  "$@"
