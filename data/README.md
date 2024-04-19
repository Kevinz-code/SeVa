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
In 'step2/' folder, generate choosen and rejected responses to ocrvqa image-question pairs. We recommand first choosing diffusion steps 500 for *your own data construction pipeline* since the results from steps 500 are more stable across multiple runs (although diffusion steps 800 could probably lead to better performance as used in SeVa-7B).

First, generate *chosen* and *rejected* responses from ocrvqa data:
```
sh llava1.5_base_gen_ocrvqa8k.sh
sh llava1.5_base_gen_ocrvqa8k_diffusion_step500.sh
```
Second, generate *chosen* and *rejected* responses from textvqa data:
```
sh llava1.5_base_gen_textvqa8k.sh
sh llava1.5_base_gen_textvqa8k_diffusion_step500.sh
```
We will then obtain 4 unfiltered answer files named {textvqa/ocrvqa}_answer_file_8k_{base/diffusion_step500}.json

### Step3: Simple Filtering
Finally, we filter **equal** responses in ocrvqa and textvqa below:
```
python make_pair_ocrvqa.py
python make_pair_textvqa.py
```
It will obtain 'textvqa_dpo_8k_{aug_name}.json', 'ocrvqa_dpo_8k_{aug_name}.json', which the **final data for DPO training**. Finished!
