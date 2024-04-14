import json
import random

# random.seed(0)

ocrvqa_file = json.load(open("llava_v1_5_mix665k_ocrvqa.json", "r"))



print("len file", len(ocrvqa_file))


message = []

for item in ocrvqa_file:
    conversation = item['conversations']

    info = {'text': []}
    info['image'] = item['image'].split('/')[-1]

    text_list = []

    for question in conversation:
        if question['from'] == 'gpt':
            continue
        text_list.append(question['value'].replace('<image>', '').replace('Answer the question using a single word or phrase.', '').replace('\n', ''))

    info['text'] = random.sample(text_list, min(len(text_list), 2))
    


    message.append(info)



sample_item = 4
sel_messge = random.sample(message, sample_item * 1000)


print("obtaining {}k ocrvqa image_question pairs".format(sample_item * 2)) # each sample item has 2 instances
print("saving as ./ocrvqa_image_question_list_{}k.json".format(sample_item * 2))

f = open("./ocrvqa_image_question_list_{}k.json".format(sample_item * 2), "w")

for info in sel_messge:

    for i in range(len(info['text'])):

        sel_info = {}
        sel_info['image'] = info['image']
        sel_info['text'] = info['text'][i]
    
        sel_info = json.dumps(sel_info)

        f.writelines(sel_info + "\n")



    
