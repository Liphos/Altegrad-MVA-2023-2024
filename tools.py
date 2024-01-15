from transformers import AutoTokenizer


def load_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)
