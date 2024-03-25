import torch
from huggingface_hub import hf_hub_download

from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor


processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

device = "cuda:0"

model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True
)
model.config.pad_token_id = 1
model.to(device)

# load from hub
filepath = hf_hub_download("nielsr/llava-batched-inference", filename="input_ids.pt", repo_type="dataset")
input_ids = torch.load(filepath).to(device)
filepath = hf_hub_download("nielsr/llava-batched-inference", filename="attention_mask.pt", repo_type="dataset")
attention_mask = torch.load(filepath).to(device)
filepath = hf_hub_download("nielsr/llava-batched-inference", filename="image_tensor.pt", repo_type="dataset")
image_tensor = torch.load(filepath)
pixel_values = torch.stack(image_tensor).to(device)
filepath = hf_hub_download("nielsr/llava-batched-inference", filename="image_sizes.pt", repo_type="dataset")
image_sizes = torch.load(filepath)
# reverse the list
image_sizes = [size[::-1] for size in image_sizes]
image_sizes = torch.tensor(image_sizes).to(device)

print("Input_ids:", input_ids)
print("Shape of input_ids:", input_ids.shape)
print("Shape of attention_mask:", attention_mask.shape)
print("Shape of pixel_values:", pixel_values.shape)
print("Shape of image_sizes:", image_sizes.shape)

# replace -200 by 32000 in input_ids
input_ids[input_ids == -200] = 32000

for i in input_ids:
    print(repr(processor.decode(i)))

# pixel_values are of shape (batch_size, num_patches, 3, patch_size, patch_size)
# join the first 2 dimensions into one
pixel_values = pixel_values.view(-1, *pixel_values.shape[2:])

# autoregressively complete prompt
generated_ids = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    pixel_values=pixel_values,
    image_sizes=image_sizes,
    max_new_tokens=100,
)

outputs = processor.batch_decode(generated_ids, skip_special_tokens=True)
outputs = [output.strip() for output in outputs]
print(outputs)
