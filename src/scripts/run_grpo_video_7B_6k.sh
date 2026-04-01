cd src/r1-v

export MODEL_DIR="/mnt/xmap_nas_alg/yzl/Auto_Drive/AFile/models"
export DATASET_PATH="/mnt/xmap_nas_alg/yzl/Auto_Drive_Github/AData/rl.json"
export PRETRAIN_MODEL_PATH="${MODEL_DIR}/sft_video_7B_6k"  # Path to pretrained model
export SAVE_PATH="${MODEL_DIR}/grpo_video_7B_6k"                   # Absolute path to save checkpoints

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
# export LOG_PATH="${SAVE_PATH}/debug_log_2b.txt"

# For resume training:  --resume_from_checkpoint Model_Path \
# Set temporal to choose between T-GRPO and GRPO, and len_control to enable or disable the length control reward.

# Qwen/Qwen2.5-VL-7B-Instruct

export WANDB_MODE="offline"

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12364" \
    src/open_r1/grpo_final.py \
    --output_dir ${SAVE_PATH} \
    --model_name_or_path ${PRETRAIN_MODEL_PATH} \
    --dataset_name ${DATASET_PATH} \
    --deepspeed local_scripts/zero3.json \
    --max_prompt_length 16384 \
    --max_completion_length 4096 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-7 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --fp16 \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --temporal true \
    --len_control true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --run_name Video-R1 \
    --save_steps 250 \
    --beta 0.04 \
    --max_grad_norm 5 \
    --report_to wandb \
    --save_only_model false \
    --temperature 0.8 \
    --num_generations 6

# number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance  
#   这个num_generations参数最后得调大才行
#   --resume_from_checkpoint ${SAVE_PATH}/checkpoint-420 \
    
