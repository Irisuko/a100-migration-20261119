#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
export PATH="$REPO_DIR/.venv/bin:$PATH"
export NO_PROXY="localhost,127.0.0.1,::1${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec "$REPO_DIR/.venv/bin/python" -m sglang.launch_server \
  --model-path "${MODEL_PATH:-$REPO_DIR/../models/Llama-3.1-8B-Instruct}" \
  --served-model-name llama-3.1-8b \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-30000}" \
  --tp-size 1 --pp-size 2 \
  --enable-pdmux \
  --pdmux-config-path "$REPO_DIR/configs/pdmux_pp_a100_local.yml" \
  --chunked-prefill-size -1 --disable-overlap-schedule \
  --attention-backend triton --sampling-backend pytorch \
  --dtype bfloat16 --context-length 2048 \
  --mem-fraction-static 0.15 --max-total-tokens 8192 \
  --max-running-requests 16 --cuda-graph-max-bs 16 \
  --max-prefill-tokens 2048 --pdmux-pp-prefill-chunk-size 256 \
  --pdmux-pp-summary-interval 5 \
  "$@"
