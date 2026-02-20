from __future__ import annotations

import platform

import torch
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline


FALLBACK_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def _normalize_rope_scaling(config: object) -> object:
    rope_scaling = getattr(config, "rope_scaling", None)
    if isinstance(rope_scaling, dict) and "type" not in rope_scaling:
        rope_type = rope_scaling.get("rope_type") or rope_scaling.get("name") or "longrope"
        rope_scaling["type"] = rope_type
        setattr(config, "rope_scaling", rope_scaling)
    return config


def build_generation_llm(
    model_name: str,
    max_new_tokens: int = 350,
    temperature: float = 0.1,
    device_map: str = "auto",
) -> HuggingFacePipeline:
    use_cuda = torch.cuda.is_available()
    use_4bit = use_cuda and platform.system().lower() != "windows"
    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config = _normalize_rope_scaling(config)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            quantization_config=quantization_config,
            device_map=device_map if use_cuda else None,
            torch_dtype=torch.float16 if use_cuda else torch.float32,
            trust_remote_code=True,
        )
    except Exception:
        # Fallback for incompatible model configs/environments (common on Windows CPU).
        fallback_tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL)
        if fallback_tokenizer.pad_token_id is None:
            fallback_tokenizer.pad_token = fallback_tokenizer.eos_token
        tokenizer = fallback_tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            FALLBACK_MODEL,
            device_map=device_map if use_cuda else None,
            torch_dtype=torch.float16 if use_cuda else torch.float32,
        )

    text_generator = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        repetition_penalty=1.1,
        return_full_text=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return HuggingFacePipeline(pipeline=text_generator)
