import requests
import torch
from PIL import Image

from transformers import LlavaProcessor


processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")


image1 = Image.open(requests.get("https://llava-vl.github.io/static/images/view.jpg", stream=True).raw)
image2 = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

inputs = [
    [
        {"input": "image", "content": image1},
        {
            "input": "text",
            "content": "What are the things I should be cautious about when I visit this place? What should I bring with me?",
        },
    ],
    [
        {"input": "image", "content": image2},
        {"input": "text", "content": "What is this?"},
    ],
]

features = processor(inputs, padding=True, return_tensors="pt")


# assert
from huggingface_hub import hf_hub_download


filepath = hf_hub_download("nielsr/test-image", "llava_inputs.pt", repo_type="dataset")
original_inputs = torch.load(filepath)

for k, v in features.items():
    if k == "input_ids":
        for i in range(v.shape[0]):
            print(repr(processor.decode(v[i])))
            print(repr(processor.decode(original_inputs[k][i])))
    assert torch.allclose(v.float(), original_inputs[k].float())


# test non-batched

inputs = [
    [
        {"input": "image", "content": image1},
        {
            "input": "text",
            "content": "This is a cat",
        },
        {"input": "image", "content": image2},
        {
            "input": "text",
            "content": "This is a dog",
        },
        {"input": "image", "content": image1},
        {
            "input": "text",
            "content": "What is this?",
        },
    ],
]

features = processor(inputs, padding=True, return_tensors="pt")

# should create this:
"USER: <image>\nThis is a cat\n<image>\nThis is a dog\n<image>\nWhat is this?\nASSISTANT:"

for k, v in features.items():
    print(k, v.shape)

print(repr(processor.decode(features["input_ids"][0])))

# if k == "input_ids":
#     for i in range(v.shape[0]):
#         print(repr(processor.decode(v[i])))
#         print(repr(processor.decode(original_inputs[k][i])))
