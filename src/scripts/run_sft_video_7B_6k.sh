cd src/r1-v

export MODEL_DIR="/data/oss_bucket_0/yzl/Auto_Drive/AFile/models"
export DATASET_PATH="/mnt/xmap_nas_alg/yzl/Auto_Drive_Github/AData/sft_cot.json"
export PRETRAIN_MODEL_PATH="${MODEL_DIR}/Qwen2.5-VL-7B-Instruct"  # Path to pretrained model
export SAVE_PATH="${MODEL_DIR}/sft_video_7B_6k"                   # Absolute path to save checkpoints

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="${SAVE_PATH}/debug_log_2b.txt"

export WANDB_MODE="offline"

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12349" \
    src/open_r1/sft_video_final.py \
    --output_dir ${SAVE_PATH} \
    --model_name_or_path ${PRETRAIN_MODEL_PATH} \
    --dataset_name ${DATASET_PATH} \
    --deepspeed local_scripts/zero2.json \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-6 \
    --logging_steps 1 \
    --bf16 True \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name video-cot-sft \
    --save_steps 1000 \
    --max_grad_norm 5 \
    --save_only_model true \
    --report_to wandb \