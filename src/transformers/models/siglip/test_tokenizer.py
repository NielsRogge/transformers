from transformers import SiglipTokenizer, SiglipTokenizerFast


slow_tokenizer = SiglipTokenizer.from_pretrained("nielsr/siglip-base-patch16-224")

fast_tokenizer = SiglipTokenizerFast.from_pretrained("nielsr/siglip-base-patch16-224")

text = "This is a test!?"

input_ids = slow_tokenizer(text).input_ids

input_ids_fast = fast_tokenizer(text).input_ids

print(slow_tokenizer.decode(input_ids))
print(fast_tokenizer.decode(input_ids_fast))
