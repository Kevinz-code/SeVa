### Step1: acquire dataset
First acquire all textvqa and ocrvqa dataset from llava665k
```
cd step1
python split_data.py
```
Then we extract 8k image-question pairs of ocrvqa and textvqa, respectively.
```
python make_ocrvqa_data.py
python make_textvqa_data.py
```
We will get the 'ocrvqa_image_question_list_8k.json' and 'textvqa_image_question_list_8k.json' in step1/ folder.
### Step2: Generate augmented response
In 'step2/' folder, generate choosen and rejected responses to ocrvqa image-question pairs:
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
