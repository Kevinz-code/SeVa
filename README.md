# SeVa
This is the official code of paper **Self-Supervised Visual Preference Alignment**

![method](seva/utils/method.png)

# About SeVa
We make the first attempt towards **unsupervised preference alignment** in Large Vision-Language Models, and discuss its relations **contrastive learning**.

[Paper](now uploading) 

[Data](https://huggingface.co/kevinke/data/)

[Models](https://huggingface.co/kevinke/)

## Getting Started
```
conda create -n seva python==3.9
```
Then in `seva' environment, install dependencies:
pip install torch==2.0.1 torchvision==0.15.2

```
pip install -e .
```

## Training
```
sh /data/hypertext/zhuk/github/upload/run/llava1.5_lora_our_ocrlv_8kfilter_diffu500_textvga_8kfilter_diffu500_r1024_a2048.sh
sh /data/hypertext/zhuk/github/upload/run/llava1.5_lora_our_ocrlv_8kfilter4k_diffu800_textvga_8kfilter6k_diffu800_r1024_a2048.sh
```
