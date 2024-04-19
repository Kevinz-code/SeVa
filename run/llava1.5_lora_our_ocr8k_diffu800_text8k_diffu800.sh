MODEL_VERSION=llava_loraft_dpo_our_ocrvqa8kfilter_diffu800_textvqa8kfilter_diffu800_r1024_a2048
MODEL_BASE=./checkpoints/llava-v1.5-7b

OCR_DPO_DATA=data/step3/ocrvqa_dpo_8k_diffusion_step800.json
OCRVQA_IMAGES=data/ocr_vqa/images/
TEXTVQA_IMAGES=data/textvqa/train_images/
TEXT_DPO_DATA=data/step3/textvqa_dpo_8k_diffusion_step800.json


VISION_TOWER=./checkpoints/clip-vit-large-patch14-336

deepspeed seva/train_dpo_ours.py \
    --lora_enable True --lora_r 1024 --lora_alpha 2048 --mm_projector_lr 0 \
    --deepspeed seva/scripts/zero3.json \
    --model_name_or_path ${MODEL_BASE} \
    --version v1 \
    --ocr_data_path ${OCR_DPO_DATA} \
    --ocr_image_path ${OCRVQA_IMAGES} \
    --textvqa_data_path ${TEXT_DPO_DATA} \
    --textvqa_image_path ${TEXTVQA_IMAGES} \
    --vision_tower ${VISION_TOWER} \
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



torchrun --nproc_per_node 8 --master_port 29501 seva/pope_eval.py \
    --coco_path /data/hypertext/data/data/dataset/COCO/ \
    --pope_path /data/hypertext/zhuk/HA-DPO/ha_dpo/data/POPE/ \
    --model-path ./checkpoints/${MODEL_VERSION} \
    --model-base ./checkpoints/llava-v1.5-7b \
    --save_dir ./seva/pope_result/${MODEL_VERSION} \
    --set random


torchrun --nproc_per_node 8 --master_port 29501 seva/pope_eval.py \
    --coco_path /data/hypertext/data/data/dataset/COCO/ \
    --pope_path /data/hypertext/zhuk/HA-DPO/ha_dpo/data/POPE/ \
    --model-path ./checkpoints/${MODEL_VERSION} \
    --model-base ./checkpoints/llava-v1.5-7b \
    --save_dir ./seva/pope_result/${MODEL_VERSION} \
    --set popular


torchrun --nproc_per_node 8 --master_port 29501 seva/pope_eval.py \
    --coco_path /data/hypertext/data/data/dataset/COCO/ \
    --pope_path /data/hypertext/zhuk/HA-DPO/ha_dpo/data/POPE/ \
    --model-path ./checkpoints/${MODEL_VERSION} \
    --model-base ./checkpoints/llava-v1.5-7b \
    --save_dir ./seva/pope_result/${MODEL_VERSION} \
    --set adv