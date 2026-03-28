import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_REGISTRY = {
    "llada": {
        "tokenizer": "GSAI-ML/LLaDA-8B-Instruct",
        "model": "GSAI-ML/LLaDA-8B-Instruct",
    },
    "llama": {
        "tokenizer": "meta-llama/Llama-3.1-8B-Instruct",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
    },
}


def get_tokenizer(alias: str):
    config = MODEL_REGISTRY[alias]
    return AutoTokenizer.from_pretrained(config["tokenizer"], trust_remote_code=True)


def get_model(
    alias: str,
    dtype: torch.dtype = torch.bfloat16,
):
    config = MODEL_REGISTRY[alias]
    model = AutoModelForCausalLM.from_pretrained(
        config["model"],
        trust_remote_code=True,
        dtype=dtype,
    )
    model.name = config["model"]

    return model
