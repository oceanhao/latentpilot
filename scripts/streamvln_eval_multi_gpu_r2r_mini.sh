export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
MASTER_PORT=$((RANDOM % 101 + 20000))

CHECKPOINT="${MODEL_PATH:-./model_weight/mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3}"
echo "CHECKPOINT: ${CHECKPOINT}"
export NCCL_IGNORE_DISABLED_P2P=1
torchrun --nproc_per_node="${NUM_GPUS:-4}" --master_port="$MASTER_PORT" streamvln/streamvln_eval.py --output_path "${OUTPUT_PATH:-./results/mini_r2r_eval}" --model_path "$CHECKPOINT" --habitat_config_path "${HABITAT_CONFIG_PATH:-config/vln_r2r.yaml}"
