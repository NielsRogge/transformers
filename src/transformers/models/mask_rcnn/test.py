import torch
from PIL import Image

from transformers import MaskRCNNImageProcessor


image = Image.open("/Users/nielsrogge/Downloads/publaynet.jpeg").convert("RGB")

image_processor = MaskRCNNImageProcessor()

inputs = image_processor(images=image, return_tensors="pt")

for k, v in inputs.items():
    if isinstance(v, torch.Tensor):
        print(k, v.shape)
    else:
        print(k, v)
