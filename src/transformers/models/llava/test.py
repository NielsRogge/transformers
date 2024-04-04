import requests
from PIL import Image

from transformers import LlavaProcessor


processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")


image1 = Image.open(requests.get("https://llava-vl.github.io/static/images/view.jpg", stream=True).raw)
image2 = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

# inputs = [
#     [
#         {"input": "text", "content": "hello world"},
#         {"input": "image", "content": image1},
#     ],
# ]

prompts = [
    "USER: <image>\nWhat are the things I should be cautious about when I visit this place? What should I bring with me?\nASSISTANT:",
    "USER: <image>\nWhat is this?\nASSISTANT:",
]

inputs = [
    [
        {
            "input": "text",
            "content": "What are the things I should be cautious about when I visit this place? What should I bring with me?",
        },
        {"input": "image", "content": image1},
    ],
    [
        {"input": "text", "content": "What is this?"},
        {"input": "image", "content": image2},
    ],
]

features = processor(inputs, padding=True, return_tensors="pt")

for k, v in features.items():
    print(k, v.shape)
    # decode input_ids
    if k == "input_ids":
        for i in range(v.shape[0]):
            print(repr(processor.decode(v[i])))
