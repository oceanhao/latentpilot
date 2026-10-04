#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a model checkpoint directory}"
HABITAT_CONFIG_PATH="${HABITAT_CONFIG_PATH:-config/vln_r2r.yaml}"
OUTPUT_PATH="${OUTPUT_PATH:-results/eval}"
NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-20001}"
MAX_LATENT_NUM="${MAX_LATENT_NUM:-1}"
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet NCCL_IGNORE_DISABLED_P2P=1
MAX_LATENT_NUM="$MAX_LATENT_NUM" torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" streamvln/streamvln_eval.py --output_path "$OUTPUT_PATH" --model_path "$MODEL_PATH" --habitat_config_path "$HABITAT_CONFIG_PATH"
