import requests
from PIL import Image

from transformers import AutoProcessor

from datasets import load_dataset


dataset = load_dataset("")


processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b", do_image_splitting=False)

url = "https://upload.wikimedia.org/wikipedia/commons/f/f3/Zinedine_Zidane_by_Tasnim_03.jpg"
image = Image.open(requests.get(url, stream=True).raw)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Extract JSON."},
            {"type": "image"},
        ],
    },
]
text = processor.apply_chat_template(messages, add_generation_prompt=False).strip()

batch = processor(
    text=text,
    images=image,
    return_tensors="pt",
)

print(batch["input_ids"].shape)

print(processor.batch_decode(batch["input_ids"]))
