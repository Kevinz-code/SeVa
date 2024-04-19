MODEL_VERSION=llava_loraft_dpo_our_ocrvqa8kfilter_diffu500_textvqa8kfilter_diffu500_r1024_a2048

OCR_DPO_DATA=data/step3/ocrvqa_dpo_8k_diffusion_step500.json
TEXT_DPO_DATA=data/step3/textvqa_dpo_8k_diffusion_step500.json

deepspeed seva/train_dpo_ours.py \
    --lora_enable True --lora_r 1024 --lora_alpha 2048 --mm_projector_lr 0 \
    --deepspeed seva/scripts/zero3.json \
    --model_name_or_path checkpoints/llava-v1.5-7b \
    --version v1 \
    --ocr_data_path ${OCR_DPO_DATA} \
    --ocr_image_path data/ocr_vqa/images/ \
    --textvqa_data_path ${TEXT_DPO_DATA} \
    --textvqa_image_path data/textvqa/train_images/ \
    --vision_tower ./checkpoints/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir checkpoints/${MODEL_VERSION} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-6 \
    --weight_decay 0. \
    --warmup_steps 0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name ${MODEL_VERSION} \
    --beta 0.1


