import json


sft_root = "/data/hypertext/zhuk/llava/LLaVA/data/LLaVA-Instruct-150K/llava_v1_5_mix665k.json"
sft_file = json.load(open(sft_root, "r"))



message_ocrvqa = []
message_textvqa = []
for item in sft_file:
    if 'image' not in item.keys():
        continue
           
    prefix = item['image'].split('/')[0]

    if prefix == 'ocr_vqa':
        message_ocrvqa.append(item)
    if prefix == 'textvqa':
        message_textvqa.append(item)

print("ocrvqa data num", len(message_ocrvqa))
print("textvqa data num", len(message_textvqa))

json.dump(message_ocrvqa, open("llava_v1_5_mix665k_ocrvqa.json", "w"))
json.dump(message_textvqa, open("llava_v1_5_mix665k_textvqa.json", "w"))
