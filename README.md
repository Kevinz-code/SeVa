# :rabbit2:	Self-Supervised Visual Preference Alignment
We make the first attempt towards **unsupervised preference alignment** in Large Vision-Language Models.

*The whole pipeline without any GPT-4 or humman annotated labels*. [Paper](https://arxiv.org/abs/2404.10501) [Data](https://huggingface.co/kevinke/data/) [Models](https://huggingface.co/kevinke/)


![method](seva/utils/method.png) 




## Contents
- [Install](#Install)
- [Model Zoo](https://huggingface.co/kevinke/)
- [SeVa pipeline](data/README.md)
- [DPO Data](https://huggingface.co/kevinke/data/)
- [Training](#Training)


## Install
```
conda create -n seva python==3.9
```
Then in `seva' environment, install dependencies:
```
pip install torch==2.0.1 torchvision==0.15.2
pip install -e .
```


# Model
| Version | Augmentation | LLM | Schedule | Checkpoint | LLaVA-Bench | MM-Vet | MMB | MMB-CN | POPE| SEED | SHR (↓) | SQA | GQA |
|----------|------------|------|----------|------------|---|---|---|---|---|---|---|---|---|
| SeVa-7B | Diffusion500 | Vicuna-7B | lora_ft | [kevinke/seva-7b-diffu500](https://huggingface.co/kevinke/seva-7b-diffu500) | 70.7 | 35.5 | 64.7 | 58.8 | 86.8 | 65.8  | 32.7 | 67.4 | 61.1 |
| SeVa-7B | Diffusion800 | Vicuna-7B | lora_ft | [kevinke/seva-7b-diffu800](https://huggingface.co/kevinke/seva-7b-diffu800) | 72.2 | 37.2 | 65.6 | 59.2 | 86.7 | 65.8 | 34.9 | 67.5 | 60.7 |
| SeVa-7B | MOCO        | Vicuna-7B | lora_ft | [kevinke/seva-7b-moco](https://huggingface.co/kevinke/seva-7b-moco)      | 72.5 | 37.0 | 65.2 | 59.8 | 86.6 | 65.5 | 32.9 | 67.1 | 60.9| 
 


## Training
```
sh /data/hypertext/zhuk/github/upload/run/llava1.5_lora_our_ocrlv_8kfilter_diffu500_textvga_8kfilter_diffu500_r1024_a2048.sh
sh /data/hypertext/zhuk/github/upload/run/llava1.5_lora_our_ocrlv_8kfilter4k_diffu800_textvga_8kfilter6k_diffu800_r1024_a2048.sh
```


## Evaluation
Refer to [LLaVa-1.5](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md).
