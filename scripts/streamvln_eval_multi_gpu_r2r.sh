export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
MASTER_PORT=$((RANDOM % 101 + 20000))

CHECKPOINT="./checkpoints/StreamVLN_Video_qwen_1_5_1epoch_196token_8history_16frame_stage_two_multitask"
echo "CHECKPOINT: ${CHECKPOINT}"
export NCCL_IGNORE_DISABLED_P2P=1
torchrun --nproc_per_node=4 --master_port=$MASTER_PORT streamvln/streamvln_eval.py --output_path ./results/Use_lossMask_val_unseen/stage_two_multitask_r2r_ckpt2248 --model_path $CHECKPOINT
