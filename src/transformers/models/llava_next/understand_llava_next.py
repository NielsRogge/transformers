import torch
from transformers import AutoTokenizer

prompt = "hello <image> what is there"

tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
batch_size, sequence_length = input_ids.shape

print(input_ids)

# let's say we have image_features of shape (1, 577, 1024)
batch_size, num_image_patches, image_feature_dim = 1, 577, 1024
image_features = torch.randn(1, 577, 1024)

# now we want to create the final features for llava
left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(32001))
print(left_padding)

# 1. Create a mask to know where special image tokens are
special_image_token_mask = input_ids == 32000
print(special_image_token_mask)

num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
print(num_special_image_tokens)

# Compute the maximum embed dimension
max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length
batch_indices, non_image_indices = torch.where(input_ids != 32000)

print(max_embed_dim)
print("Batch indices:", batch_indices)
print("Non image indices:", non_image_indices)

# 2. Compute the positions where text should be written
# Calculate new positions for text tokens in merged image-text sequence.
# `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
# `torch.cumsum` computes how each image token shifts subsequent text token positions.
# - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
if left_padding:
    new_token_positions += nb_image_pad[:, None]  # offset for left padding
text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

print("New token positions:", new_token_positions)
print("Text to overwrite:", text_to_overwrite)