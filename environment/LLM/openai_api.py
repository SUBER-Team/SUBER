from time import sleep
from typing import Tuple
from .llm import LLM
import openai


class OpenAIModelAPI(LLM):
    def __init__(self, name):
        super().__init__(name)
        self.name = name
        tokens_numeric_0_9 = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        self.map_logits_bias_0_9 = {k: 100 for k in tokens_numeric_0_9}

        tokens_numeric_1_10 = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 940]
        self.map_logits_bias_1_10 = {k: 100 for k in tokens_numeric_1_10}

        tokens_numeric_1_5 = [16, 17, 18, 19, 20]
        self.map_logits_bias_1_5 = {k: 100 for k in tokens_numeric_1_5}

    def request_rating_0_9(self, system_prompt, dialog) -> Tuple[str, str]:
        if system_prompt is None:
            system_prompt = "You are a helpful assistant."

        request = []
        input_text = ""
        if system_prompt:
            request.append({"role": "system", "content": system_prompt})
            input_text += system_prompt
        for d in dialog:
            role = d["role"]
            if role == "assistant_start":
                role = "assistant"
            input_text += "\n" + role + ": "
            input_text += d["content"]
            request.append({"role": role, "content": d["content"]})

        fail_count = 0
        while True:
            try_again = False
            try:
                out = openai.ChatCompletion.create(
                    model=self.name,
                    messages=request,
                    max_tokens=1,
                    logit_bias=self.map_logits_bias_0_9,
                )
            except (
                openai.error.RateLimitError,
                openai.error.ServiceUnavailableError,
                openai.error.APIError,
                openai.error.Timeout,
            ):
                sleep(10)
                try_again = True
                fail_count += 1

            if not try_again:
                break
        res = out["choices"][0]["message"]["content"]
        return input_text, res

    def request_rating_1_5(self, system_prompt, dialog) -> Tuple[str, str]:
        if system_prompt is None:
            system_prompt = "You are a helpful assistant."

        request = []
        input_text = ""
        if system_prompt:
            request.append({"role": "system", "content": system_prompt})
            input_text += system_prompt
        for d in dialog:
            role = d["role"]
            if role == "assistant_start":
                role = "assistant"
            input_text += "\n" + role + ": "
            input_text += d["content"]
            request.append({"role": role, "content": d["content"]})

        fail_count = 0
        while True:
            try_again = False
            try:
                out = openai.ChatCompletion.create(
                    model=self.name,
                    messages=request,
                    max_tokens=1,
                    logit_bias=self.map_logits_bias_1_5,
                )
            except (
                openai.error.RateLimitError,
                openai.error.ServiceUnavailableError,
                openai.error.APIError,
                openai.error.Timeout,
            ):
                sleep(10)
                try_again = True
                fail_count += 1

            if not try_again:
                break
        res = out["choices"][0]["message"]["content"]
        return input_text, res

    def request_rating_1_10(self, system_prompt, dialog) -> Tuple[str, str]:
        if system_prompt is None:
            system_prompt = "You are a helpful assistant."

        request = []
        input_text = ""
        if system_prompt:
            request.append({"role": "system", "content": system_prompt})
            input_text += system_prompt
        for d in dialog:
            role = d["role"]
            if role == "assistant_start":
                role = "assistant"
            input_text += "\n" + role + ": "
            input_text += d["content"]
            request.append({"role": role, "content": d["content"]})

        fail_count = 0
        while True:
            try_again = False
            try:
                out = openai.ChatCompletion.create(
                    model=self.name,
                    messages=request,
                    max_tokens=1,
                    logit_bias=self.map_logits_bias_1_10,
                )
            except (
                openai.error.RateLimitError,
                openai.error.ServiceUnavailableError,
                openai.error.APIError,
                openai.error.Timeout,
            ):
                sleep(30)
                try_again = True
                fail_count += 1

            if not try_again:
                break
        res = out["choices"][0]["message"]["content"]
        return input_text, res

    def request_rating_text(self, system_prompt, dialog) -> Tuple[str, str]:
        pass

    def request_explanation(self, system_prompt, dialog) -> Tuple[str, str]:
        if system_prompt is None:
            system_prompt = "You are a helpful assistant."

        request = []
        input_text = ""
        if system_prompt:
            request.append({"role": "system", "content": system_prompt})
            input_text += system_prompt
        for d in dialog:
            role = d["role"]
            if role == "assistant_start":
                role = "assistant"
            input_text += "\n" + role + ": "
            input_text += d["content"]
            request.append({"role": role, "content": d["content"]})

        fail_count = 0
        while True:
            try_again = False
            try:
                out = openai.ChatCompletion.create(
                    model=self.name, messages=request, max_tokens=512
                )
            except (
                openai.error.RateLimitError,
                openai.error.ServiceUnavailableError,
                openai.error.APIError,
                openai.error.Timeout,
            ):
                sleep(30)
                try_again = True
                fail_count += 1

            if not try_again:
                break
        res = out["choices"][0]["message"]["content"]
        return input_text, res
