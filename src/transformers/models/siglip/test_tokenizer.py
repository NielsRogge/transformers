from transformers import SiglipTokenizer, SiglipTokenizerFast


slow_tokenizer = SiglipTokenizer.from_pretrained("nielsr/siglip-base-patch16-224")

fast_tokenizer = SiglipTokenizerFast.from_pretrained("nielsr/siglip-base-patch16-224")

texts = [
    "an apple",
    "a picture of an apple",
    "an ipod",
    "granny smith",
    'an apple with a note saying "ipod"',
    "a cold drink on a hot day",
    "a hot drink on a cold day",
    "a photo of a cold drink on a hot day",
    "a photo of a hot drink on a cold day",
    #
    "a photo of two guys in need of caffeine",
    "a photo of two guys in need of water",
    "a photo of the SigLIP authors",
    "a photo of a rock band",
    "a photo of researchers at Google Brain",
    "a photo of researchers at OpenAI",
    #
    "a robot on a sign",
    "a photo of a robot on a sign",
    "an empty street",
    "autumn in Toronto",
    "a photo of autumn in Toronto",
    "a photo of Toronto in autumn",
    "a photo of Toronto in summer",
    "autumn in Singapore",
    #
    "cow",
    "a cow in a tuxedo",
    "a cow on the beach",
    "a cow in the prairie",
    #
    "the real mountain view",
    "Zürich",
    "San Francisco",
    "a picture of a laptop with the lockscreen on, a cup of cappucino, salt and pepper grinders. The view through the window reveals lake Zürich and the Alps in the background of the city.",
]

for text in texts:
    input_ids = slow_tokenizer(text).input_ids

    input_ids_fast = fast_tokenizer(text).input_ids

    assert input_ids == input_ids_fast

    print(slow_tokenizer.decode(input_ids))
    print(fast_tokenizer.decode(input_ids_fast))
