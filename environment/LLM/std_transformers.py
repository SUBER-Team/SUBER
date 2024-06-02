from typing import Tuple
from auto_gptq import exllama_set_max_input_length
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    pipeline,
)

from .llm import LLM


class NumbersOnlyLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, model, options):
        self.tokenizer = tokenizer
        self.model = model
        # Precompute the mask for number tokens
        self.number_tokens_mask = self.create_number_tokens_mask(options)

    def create_number_tokens_mask(self, options):
        # Create a mask where number tokens are True and others are False
        mask = torch.full((self.model.config.vocab_size,), False, dtype=torch.bool)
        for token_id in range(self.model.config.vocab_size):
            token = self.tokenizer.decode([token_id])
            if token in options:
                mask[token_id] = True
        return mask

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        # Use the precomputed mask to zero out logits for non-number tokens
        scores[:, ~self.number_tokens_mask] = float("-inf")
        return scores


# Load model and tokenizer


class Transformers(LLM):
    def __init__(self, name, use_cache=True):
        super().__init__(name)
        self.tokenizer = AutoTokenizer.from_pretrained(name)

        model = AutoModelForCausalLM.from_pretrained(name, device_map="auto")
        if "gptq" in name.lower() and not "mixtral" in name.lower():
            model = exllama_set_max_input_length(model, max_input_length=4096)
        self.model = model

        self.pipe_request_rating_0_9 = self._create_pipeline(
            ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        )
        self.pipe_request_rating_1_5 = self._create_pipeline(["1", "2", "3", "4", "5"])
        self.pipe_request_rating_1_10 = self._create_pipeline(
            ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        )
        self.pipe_request_rating_text = self._create_pipeline(
            [
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
                "nine",
                "ten",
            ]
        )
        self.pipe_request_explanation = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            top_k=50,
        )

    def _create_pipeline(self, options):
        return pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=1,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            top_k=50,
            logits_processor=LogitsProcessorList(
                [
                    NumbersOnlyLogitsProcessor(
                        self.tokenizer,
                        self.model,
                        options,
                    )
                ]
            ),
        )

    def request_rating_0_9(self, system_prompt, dialog) -> Tuple[str, str]:
        prompt = self.encode(system_prompt, dialog)
        res = self.pipe_request_rating_0_9(prompt, max_new_tokens=1)
        return prompt, res[0]["generated_text"][len(prompt) :]

    def request_rating_1_5(self, system_prompt, dialog) -> Tuple[str, str]:
        prompt = self.encode(system_prompt, dialog)
        res = self.pipe_request_rating_1_5(prompt, max_new_tokens=1)
        return prompt, res[0]["generated_text"][len(prompt) :]

    def request_rating_1_10(self, system_prompt, dialog) -> Tuple[str, str]:
        prompt = self.encode(system_prompt, dialog)
        res = self.pipe_request_rating_1_10(prompt, max_new_tokens=1)
        return prompt, res[0]["generated_text"][len(prompt) :]

    def request_rating_text(self, system_prompt, dialog) -> Tuple[str, str]:
        prompt = self.encode(system_prompt, dialog)
        res = self.pipe_request_rating_text(prompt, max_new_tokens=1)
        return prompt, res[0]["generated_text"][len(prompt) :]

    def request_explanation(self, system_prompt, dialog) -> Tuple[str, str]:
        prompt = self.encode(system_prompt, dialog)
        res = self.pipe_request_explanation(prompt, max_new_tokens=512)
        return prompt, res[0]["generated_text"][len(prompt) :]
