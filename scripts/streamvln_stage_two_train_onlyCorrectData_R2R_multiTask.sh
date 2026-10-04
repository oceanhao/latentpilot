#!/bin/bash
# 非SLURM版本的训练脚本 (适用于普通多机多卡环境)

# ========== 配置参数 ==========
# 分布式训练配置
NNODES=1                          # 节点数量（多机训练时修改）
NPROC_PER_NODE=3                  # 每个节点的GPU数量
NODE_RANK=0                       # 当前节点rank（多机时第一台为0，第二台为1...）
MASTER_ADDR="localhost"           # 主节点地址（多机时改为主节点IP）
MASTER_PORT=$((RANDOM % 101 + 20001))  # 主节点端口
gradient_accumulation_steps=32
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
# ========== 数据路径 ==========
VIDEO_FOLDER="data/trajectory_data/R2R","data/trajectory_data/RxR","data/dagger_data/R2R","data/dagger_data/RxR"
# MMC4_VIDEO_FOLDER="data/co-training_data/MMC4-core/images"
# SCANQA_VIDEO_FOLDER="data/co-training_data/ScanNet"
# QA_VIDEO_FOLDER="data/co-training_data/LLaVA-Video-178K"

# ========== 模型配置 ==========
LLM_VERSION="Qwen/Qwen2-7B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################
BASE_RUN_NAME="llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

############### Finetune ################
PROMPT_VERSION="qwen_1_5"
MID_RUN_NAME="StreamVLN_Video_${PROMPT_VERSION}_1epoch_196token_8history_16frame_stage_two_multitask"
PREV_STAGE_CHECKPOINT="./model_weight/mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3"
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"

# ========== 启动训练 ==========
torchrun --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    streamvln/streamvln_train.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --video_folder ${VIDEO_FOLDER} \
    \
    --num_history 8 \
    --num_future_steps 4 \
    --num_frames 16 \
    --data_augmentation True \
    \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio anyres_max_9 \
    --frames_upbound 16 \
    --force_sample True \
    --add_time_instruction True \
    --image_grid_pinpoints  "(1x1),...,(4x4)" \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir checkpoints/$MID_RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 5 \
    --learning_rate 5e-6 \
    --mm_vision_tower_lr 1e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 2 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --report_to wandb