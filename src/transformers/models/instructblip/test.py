from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
import requests

processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

device = "cuda" if torch.cuda.is_available() else "cpu"

url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
prompts = ["what is unusual about this image?", "where is this photo taken?"]

model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b", load_in_4bit=True, device_map={"":1})

model.config.qformer_config.pad_token_id = 0
model.config.text_config.pad_token_id = 0

inputs = processor(images=[image,image], text=prompts, padding=True, return_tensors="pt").to("cuda:1")

print(processor.decode(inputs.input_ids[0]))
print(processor.qformer_tokenizer.decode(inputs.qformer_input_ids[0]))

outputs = model.generate(
        **inputs,
        do_sample=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=1,
)

generated_text = processor.batch_decode(outputs, skip_special_tokens=True)
print(generated_text)