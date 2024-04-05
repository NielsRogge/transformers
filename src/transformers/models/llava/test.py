import requests
import torch
from PIL import Image

from transformers import LlavaProcessor


processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")


image1 = Image.open(requests.get("https://llava-vl.github.io/static/images/view.jpg", stream=True).raw)
image2 = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "content": image1},
            {
                "type": "text",
                "content": "What are the things I should be cautious about when I visit this place? What should I bring with me?",
            },
        ],
    },
]

features = processor.apply_chat_template(messages, return_tensors="pt")


# assert
from huggingface_hub import hf_hub_download


filepath = hf_hub_download("nielsr/test-image", "llava_inputs.pt", repo_type="dataset")
original_messages = torch.load(filepath)

for k, v in features.items():
    print("Key:", k)
    if k == "input_ids":
        print("Our input_ids:", v)
        print("Original input_ids:", original_messages[k])
        for i in range(v.shape[0]):
            print(repr(processor.decode(v[i])))
            print(repr(processor.decode(original_messages[k][i])))
    assert torch.allclose(v.float(), original_messages[k][0].float())


# test non-batched (few-shot prompting)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "content": image1},
            {"type": "text", "content": "this is a cat"},
            {"type": "image", "content": image2},
            {"type": "text", "content": "this is a dog"},
            {
                "type": "image",
                "content": image2,
            },
            {"type": "text", "content": "what is this?"},
        ],
    },
]


features = processor.apply_chat_template(messages, return_tensors="pt")

# should create this:
"USER: <image>\nThis is a cat\n<image>\nThis is a dog\n<image>\nWhat is this?\nASSISTANT:"

for k, v in features.items():
    print(k, v.shape)

print(repr(processor.decode(features["input_ids"][0])))

# if k == "input_ids":
#     for i in range(v.shape[0]):
#         print(repr(processor.decode(v[i])))
#         print(repr(processor.decode(original_messages[k][i])))
