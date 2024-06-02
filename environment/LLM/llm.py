from abc import ABC, abstractmethod
from typing import Tuple

"""
Dialog example
[
    {'role': 'user', 'content': 'How to go from Beijing to NY?'},
    {'role': 'assistant', 'content': 'Take a plane'},
    {'role': 'user', 'content': 'How long does it take?'}
    {'role': 'assistant_start', 'content': 'Take a' },
]
"""


class LLM(ABC):
    def __init__(self, name):
        if "vicuna" in name:
            self.conversation_template_name = "vicuna"
        elif "Llama" in name and ("Chat" in name or "chat" in name):
            self.conversation_template_name = "llama-2-chat"
        elif "Mistral" in name or "Mixtral" in name:
            self.conversation_template_name = "llama-2-chat"

    @abstractmethod
    def request_rating_0_9(self, system_prompt, dialog) -> Tuple[str, str]:
        pass

    @abstractmethod
    def request_rating_1_10(self, system_prompt, dialog) -> Tuple[str, str]:
        pass

    @abstractmethod
    def request_rating_text(self, system_prompt, dialog) -> Tuple[str, str]:
        pass

    @abstractmethod
    def request_explanation(self, system_prompt, dialog) -> Tuple[str, str]:
        pass

    def encode(self, system_prompt, dialog):
        if self.conversation_template_name == "vicuna":
            return self.encode_vicuna(system_prompt, dialog)
        elif self.conversation_template_name == "llama-2-chat":
            return self.encode_llama(system_prompt, dialog)
        elif self.conversation_template_name == "pretrained":
            return self.encode_pretrained(system_prompt, dialog)

    def encode_pretrained(self, system_prompt, dialog):
        ret = ""
        for i, d in enumerate(dialog):
            role = d["role"]
            message = d["content"]

            if i == 0:
                assert role == "user"
                if system_prompt is None:
                    ret += f"Q: {message}\n"
                else:
                    ret += f"{system_prompt}\nQ: {message}\n"
            else:
                if role == "user":
                    ret += f"Q: {message}\n"
                elif role == "assistant":
                    ret += f" A: {message}\n"
                elif role == "assistant_start":
                    ret += f" A: {message}"
        return ret

    def encode_vicuna(self, system_prompt, dialog):
        ret = ""
        for i, d in enumerate(dialog):
            role = d["role"]
            message = d["content"]

            if i == 0:
                assert role == "user"
                if system_prompt is None:
                    system_prompt = (
                        "A chat between a curious user and an artificial intelligence"
                        " assistant. The assistant gives helpful, detailed, and polite"
                        " answers to the user's questions."
                    )
                ret += f"{system_prompt} USER: {message}"
            else:
                if role == "user":
                    ret += f"USER: {message}"
                elif role == "assistant":
                    ret += f" ASSISTANT: {message}</s>"
                elif role == "assistant_start":
                    ret += f" ASSISTANT: {message}"
        return ret

    def encode_llama(self, system_prompt, dialog):
        ret = ""
        for i, d in enumerate(dialog):
            role = d["role"]
            message = d["content"]

            if i == 0:
                assert role == "user"
                if system_prompt is None:
                    system_prompt = (
                        "You are a helpful, respectful and honest assistant. Always"
                        " answer as helpfully as possible, while being safe.  Your"
                        " answers should not include any harmful, unethical, racist,"
                        " sexist, toxic, dangerous, or illegal content. Please ensure"
                        " that your responses are socially unbiased and positive in"
                        " nature. \n\nIf a question does not make any sense, or is not"
                        " factually coherent, explain why instead of answering"
                        " something not correct. If you don't know the answer to a"
                        " question, please don't share false information."
                    )
                ret += (
                    f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{message} [/INST]"
                )

            else:
                if role == "user":
                    ret += f"<s>[INST] {message} [/INST]"
                elif role == "assistant":
                    ret += f" {message} </s>"
                elif role == "assistant_start":
                    ret += f" {message}"
        return ret
