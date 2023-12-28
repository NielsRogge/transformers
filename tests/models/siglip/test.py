from transformers import AddedToken, SiglipTokenizer


tokenizer = SiglipTokenizer.from_pretrained("nielsr/siglip-base-patch16-224")

text = "Hey this is a SPECIAL token"

inputs = tokenizer.encode(text)

for token in inputs:
    print(tokenizer.decode([token]))

# step 2: adding tokens to the tokenizers
print("---------ADDING TOKENS---------")
added_tokens = [f"<extra_id_{i}>" for i in range(100)] + [
    AddedToken("SPECIAL", lstrip=False, rstrip=False, special=True, normalized=False)
]
tokenizer = SiglipTokenizer.from_pretrained("nielsr/siglip-base-patch16-224", additional_special_tokens=added_tokens)

inputs = tokenizer.encode(text)

for token in inputs:
    print(tokenizer.decode([token]))
