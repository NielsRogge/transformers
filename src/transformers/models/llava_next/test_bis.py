import torch
import requests
from PIL import Image

from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
processor.tokenizer.padding_side = "left"

model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="cuda",
)

# ! Chart and cat
cat_img = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
chart_img = Image.open(requests.get("https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true", stream=True).raw)


prompts = [
    "[INST] <image>\nWhat is shown in this image? [/INST]",
    "[INST] <image>\nWhat is shown in this image? [/INST]"
]
inputs = processor(prompts, [chart_img, cat_img], return_tensors='pt', padding=True).to("cuda")

for k,v in inputs.items():
    print(k,v.shape)

generated_ids = model.generate(**inputs, max_new_tokens=50, do_sample=False, pad_token_id=processor.tokenizer.pad_token_id)

print(processor.batch_decode(generated_ids, skip_special_tokens=True))