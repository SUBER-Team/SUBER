import os

from guidance.llms.caches import DiskCache

# AutoGPTQ
from transformers import AutoTokenizer, pipeline, logging
from transformers.generation import GenerationMixin
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import argparse
from guidance.llms.transformers import Vicuna


def get_vicuna_7B_GPTQ(caching=False):
    model_name_or_path = "TheBloke/vicuna-7B-v1.5-GPTQ"
    model_basename = "model"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = AutoGPTQForCausalLM.from_quantized(
        model_name_or_path,
        model_basename=model_basename,
        use_safetensors=True,
        trust_remote_code=False,
        device="cuda:0",
        use_triton=False,
        quantize_config=None,
    )

    model = Vicuna(model=model, tokenizer=tokenizer, caching=caching)
    if caching:
        model._cache = DiskCache(model_basename)  # name for unique cache
    return model


def get_vicuna_13B_GPTQ(caching=False):
    """
    This 4bit model requires at least 12GB VRAM to load.
    """
    model_name_or_path = "TheBloke/vicuna-13B-v1.5-GPTQ"
    model_basename = "model"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = AutoGPTQForCausalLM.from_quantized(
        model_name_or_path,
        model_basename=model_basename,
        use_safetensors=True,
        trust_remote_code=False,
        device="cuda:0",
        use_triton=False,
        quantize_config=None,
    )

    model = Vicuna(model=model, tokenizer=tokenizer, caching=caching)
    if caching:
        model._cache = DiskCache(model_basename)  # name for unique cache
    return model


def get_vicuna_33B_GPTQ(caching=False):
    """
    This 4bit model requires at least 24GB VRAM to load. NOTE need to test
    """

    model_name_or_path = "TheBloke/vicuna-33B-GPTQ"
    model_basename = "model"

    use_triton = False
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = AutoGPTQForCausalLM.from_quantized(
        model_name_or_path,
        model_basename=model_basename,
        use_safetensors=True,
        trust_remote_code=False,
        device="cuda:0",
        # device="cuda",
        # device_map="balanced",
        # max_memory={0: "22GIB", 1: "22GIB"},
        use_triton=use_triton,
        quantize_config=None,
    )

    model = Vicuna(model=model, tokenizer=tokenizer, caching=caching)
    if caching:
        model._cache = DiskCache(model_basename)  # name for unique cache
    return model
