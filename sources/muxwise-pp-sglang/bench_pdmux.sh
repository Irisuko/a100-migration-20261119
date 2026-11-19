#!/usr/bin/env bash
# Benchmark the local A100 service with synthetic prompts and a local tokenizer.
set -euo pipefail
A100_REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$A100_REPO_DIR"
export NO_PROXY="localhost,127.0.0.1,::1${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "$A100_REPO_DIR/run/a100"
exec "$A100_REPO_DIR/.venv/bin/python" -m sglang.bench_serving \
  --backend sglang --host "${HOST:-127.0.0.1}" --port "${PORT:-30000}" \
  --model "${MODEL_PATH:-$A100_REPO_DIR/../models/Llama-3.1-8B-Instruct}" \
  --tokenizer "${MODEL_PATH:-$A100_REPO_DIR/../models/Llama-3.1-8B-Instruct}" \
  --dataset-name random-ids --random-input-len 512 --random-output-len 128 \
  --random-range-ratio 1 \
  --num-prompts "${NUM_PROMPTS:-64}" --max-concurrency 32 \
  --request-rate "${REQUEST_RATE:-inf}" --flush-cache \
  --output-file "${OUTPUT_FILE:-$A100_REPO_DIR/run/a100/bench-serving.jsonl}" \
  "$@"
