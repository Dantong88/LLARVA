#!/bin/bash
run_id=vision-action-instruction_tuning-rlbench-sweep_to_dustpan_of_size

export WANDB_NAME=$run_id
export WANDB_PROJECT=LLARVA

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 29500 llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path sim/data/anns/sweep_to_dustpan_of_size/notrace/train-95293.json::sim/data/anns/sweep_to_dustpan_of_size/notrace/train-95293.json \
    --image_folder path/to/sweep_to_dustpan_of_size/all_variations/episodes \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir sim/output/llava-lora-instruction-tuning-sweep_to_dustpan_of_size \
    --num_train_epochs 4 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps 10000000 \
    --save_strategy "steps" \
    --save_steps 745 \
    --save_total_limit 8 \
    --learning_rate 1e-4 \
    --weight_decay 0 \
    --warmup_ratio 0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
