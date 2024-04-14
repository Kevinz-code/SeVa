torchrun --nproc_per_node 8 --master_port 29500 generate_with_aug.py \
    --model-path ../../checkpoints/llava-v1.5-7b \
    --image_file_list ../step1/ocrvqa_image_question_list_8k.json \
    --image_path /data/hypertext/zhuk/llava/LLaVA/data/ocr_vqa/images/ \
    --save_dir ./ \
    --res_file "ocrvqa_answer_file_8k_base.jsonl" \


