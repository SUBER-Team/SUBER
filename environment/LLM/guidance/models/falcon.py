from guidance.llms import Transformers
from guidance.llms.caches import DiskCache
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM


class Falcon(Transformers):
    llm_name: str = "falcon"

    default_system_prompt = (
        """A helpful assistant who helps the user with any questions asked."""
    )

    @staticmethod
    def role_start(role):
        if role == "user":
            return "User: "
        elif role == "assistant":
            return "Assistant: "
        else:
            return ""

    @staticmethod
    def role_end(role):
        if role == "user":
            return ""
        elif role == "assistant":
            return "</s>"
        else:
            return ""


def get_falcon_7B_GPTQ():
    """
    This 4bit model requires at least 8GB VRAM to load.
    """
    model_name_or_path = "TheBloke/falcon-7b-instruct-GPTQ"
    model_basename = "gptq_model-4bit-64g"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = AutoGPTQForCausalLM.from_quantized(
        model_name_or_path,
        model_basename=model_basename,
        use_safetensors=True,
        trust_remote_code=True,
        device="cuda:0",
        use_triton=False,
        quantize_config=None,
    )

    model = Falcon(model=model, tokenizer=tokenizer)
    model._cache = DiskCache("falcon_7B_GPTQ_4bit_64g")  # name for unique cache
    return model


def get_falcon_40B_GPTQ():
    """
    This 4bit model requires at least 35GB VRAM to load. It can be used on 40GB or 48GB cards, but notless.
    We run it on 2x GPUs with 24GB VRAM each.
    """
    model_name_or_path = "TheBloke/falcon-40b-instruct-GPTQ"
    model_basename = "gptq_model-4bit--1g"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = AutoGPTQForCausalLM.from_quantized(
        model_name_or_path,
        model_basename=model_basename,
        use_safetensors=True,
        trust_remote_code=True,
        device="cuda",
        device_map="balanced",
        max_memory={0: "22GIB", 1: "22GIB"},
        use_triton=False,
        quantize_config=None,
    )

    model = Falcon(model=model, tokenizer=tokenizer)
    model._cache = DiskCache("falcon_40B_GPTQ_4bit_-1g")  # name for unique cache
    return model


def get_falcon_40B_GPTQ_3Bits():
    """
    This 3bit model requires at least 24GB VRAM to load.
    """
    model_name_or_path = "TheBloke/falcon-40b-instruct-3bit-GPTQ"
    model_basename = "gptq_model-3bit--1g"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = AutoGPTQForCausalLM.from_quantized(
        model_name_or_path,
        model_basename=model_basename,
        use_safetensors=True,
        trust_remote_code=True,
        device="cuda:0",
        use_triton=False,
        quantize_config=None,
    )

    model = Falcon(model=model, tokenizer=tokenizer)
    model._cache = DiskCache("falcon_40B_GPTQ_3bit_-1g")
    return model
