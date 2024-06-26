import json,random


# aug_name = "diffusion_step500"
aug_name = "diffusion_step800"

choosen_file = "../step2/textvqa_answer_file_8k_base.jsonl"

rejected_file = "../step2/textvqa_answer_file_8k_{}.jsonl".format(aug_name)

choosen_lines = open(choosen_file, "r").readlines()
rejected_lines = open(rejected_file, "r").readlines()


message = []
for cline, rline in zip(choosen_lines, rejected_lines):

    cline = json.loads(cline)
    rline = json.loads(rline)

    assert cline['image_id'] == rline['image_id']

    if cline['answer'] == rline['answer']:
        print(cline['answer'])
        print(cline['answer'])
        continue

    cans = cline['answer'].replace('</s>', '').replace('\n', '')
    rans = rline['answer'].replace('</s>', '').replace('\n', '')

    item = {}
    item['chosen'] = cans
    item['reject'] = rans
    item['question'] = cline['question']
    item['image_id'] = cline['image_id'] + ".jpg"

    message.append(item)





if aug_name == 'diffusion_step800':
    message = random.sample(message, min(6500, len(message))) # we downsample the instances in diffusion-step-800 to approximately align with the instances diffusion-step-500


json.dump(message, open("textvqa_dpo_8k_{}.json".format(aug_name), "w"))