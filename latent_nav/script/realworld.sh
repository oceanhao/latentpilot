#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a model checkpoint directory}"
DEVICE="${DEVICE:-cuda:0}"
python streamvln/http_realworld_server.py --model_path "$MODEL_PATH" --num_future_steps "${NUM_FUTURE_STEPS:-4}" --num_frames "${NUM_FRAMES:-32}" --num_history "${NUM_HISTORY:-8}" --device "$DEVICE"
