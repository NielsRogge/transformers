from transformers import LlavaNextProcessor
from PIL import Image
import requests

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
image1 = Image.open(requests.get(url, stream=True).raw)
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image2 = Image.open(requests.get(url, stream=True).raw)

images = [image1, image2]
prompts = ["[INST] <image>\nWhat is shown in this image? [/INST]", "[INST] <image>\nHow many cats are there? [/INST]"]

inputs = processor(images=images, text=prompts, padding=True, return_tensors="pt")

for k,v in inputs.items():
    print(k, v.shape)