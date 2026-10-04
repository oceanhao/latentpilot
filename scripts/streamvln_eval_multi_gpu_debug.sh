export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
MASTER_PORT=$((RANDOM % 101 + 20000))

CHECKPOINT="./model_weight/mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3"
echo "CHECKPOINT: ${CHECKPOINT}"
# export NCCL_IGNORE_DISABLED_P2P=1s
export debug_flag=debug_input_dict
torchrun --nproc_per_node=1 --master_port=$MASTER_PORT streamvln/streamvln_eval.py --model_path $CHECKPOINT
