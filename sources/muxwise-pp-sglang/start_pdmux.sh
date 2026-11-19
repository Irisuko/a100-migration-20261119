#!/usr/bin/env bash
# Launch the calibrated A100 TP1/PP2 service.
set -euo pipefail
A100_REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$A100_REPO_DIR/scripts/run_a100.sh" "$@"
