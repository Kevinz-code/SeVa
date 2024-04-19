The easiest way to obtain the DPO dataset constructed by SeVa is to download data from [kevinke/data](https://huggingface.co/kevinke/data)

In the meanwhile, you can follow SeVa pipeline to generate arbitrary number of DPO data with Step1-Step3.

### Step1: acquire dataset
First acquire textvqa and ocrvqa split from [llava665k](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json)
```
cd step1
python split_data.py
```
Then we extract 8k image-question pairs of ocrvqa and textvqa, respectively.
```
python make_ocrvqa_data.py --data-num 8k
python make_textvqa_data.py --data-num 8k
```
which will obtain the 'ocrvqa_image_question_list_8k.json' and 'textvqa_image_question_list_8k.json' in step1/ folder.
### Step2: Generate augmented response
In 'step2/' folder, generate choosen and rejected responses to ocrvqa image-question pairs. We recommand first choosing diffusion steps 500 for *your own data construction pipeline* since the results from steps 500 are more stable across multiple runs, although our SeVa-7B adopt DPO data from Diffusion steps 800.
```
sh llava1.5_base_gen_ocrvqa8k.sh
sh llava1.5_base_gen_ocrvqa8k_diffusion_step500.sh
```
generate the choosen and rejected responses to textvqa image-question pairs
```
sh llava1.5_base_gen_textvqa8k.sh
sh llava1.5_base_gen_textvqa8k_diffusion_step500.sh
```

### Step3: Simple Filtering
```
python make_pair_ocrvqa.py
python make_pair_textvqa.py
```
It will obtain 'textvqa_dpo_8k_{aug_name}.json', 'ocrvqa_dpo_8k_{aug_name}.json', which the **data for DPO training**. Finished!
