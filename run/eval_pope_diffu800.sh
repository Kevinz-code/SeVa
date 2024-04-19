MODEL_VERSION=llava_loraft_dpo_our_ocrvqa8kfilter_diffu800_textvqa8kfilter_diffu800_r1024_a2048

torchrun --nproc_per_node 8 --master_port 29501 seva/pope_eval.py \
    --coco_path data/coco2014 \
    --pope_path data/POPE/ \
    --model-path ./checkpoints/${MODEL_VERSION} \
    --model-base ./checkpoints/llava-v1.5-7b \
    --save_dir ./seva/pope_result/${MODEL_VERSION} \
    --set random


torchrun --nproc_per_node 8 --master_port 29501 seva/pope_eval.py \
    --coco_path data/coco2014 \
    --pope_path data/POPE/ \
    --model-path ./checkpoints/${MODEL_VERSION} \
    --model-base ./checkpoints/llava-v1.5-7b \
    --save_dir ./seva/pope_result/${MODEL_VERSION} \
    --set popular


torchrun --nproc_per_node 8 --master_port 29501 seva/pope_eval.py \
    --coco_path data/coco2014 \
    --pope_path data/POPE/ \
    --model-path ./checkpoints/${MODEL_VERSION} \
    --model-base ./checkpoints/llava-v1.5-7b \
    --save_dir ./seva/pope_result/${MODEL_VERSION} \
    --set adv


# calculate the final pope accuracy
python seva/pope_calculate.py --path ./seva/pope_result/${MODEL_VERSION}
