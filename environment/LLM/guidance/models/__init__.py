from .falcon import get_falcon_40B_GPTQ, get_falcon_40B_GPTQ_3Bits, get_falcon_7B_GPTQ
from .vicuna import (
    get_vicuna_33B_GPTQ,
    get_vicuna_13B_GPTQ,
    get_vicuna_7B_GPTQ,
)


def get_model(model_name: str):
    if model_name == "vicuna7B_GPTQ":
        model = get_vicuna_7B_GPTQ()
    elif model_name == "vicuna13B_GPTQ":
        model = get_vicuna_13B_GPTQ()
    elif model_name == "vicuna33B_GPTQ":
        model = get_vicuna_33B_GPTQ()
    elif model_name == "falcon7B_GPTQ":
        model = get_falcon_7B_GPTQ()
    elif model_name == "falcon40B_GPTQ":
        model = get_falcon_40B_GPTQ()
    elif model_name == "falcon40B_GPTQ_3bit":
        model = get_falcon_40B_GPTQ_3Bits()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model
