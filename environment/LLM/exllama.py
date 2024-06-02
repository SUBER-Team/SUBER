from typing import Tuple
import torch
from .llm import LLM
from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
from exllama.tokenizer import ExLlamaTokenizer
from exllama.generator import ExLlamaGenerator
import os, glob
from huggingface_hub import snapshot_download


class LLMExllama(LLM):
    def __init__(self, name, use_cache=True):
        super().__init__(name)
        self.use_cache = use_cache
        revision = "main"
        if name == "TheBloke/Llama-2-13B-chat-GPTQ":
            # 4-bit quantized model, with more quality
            revision = "gptq-4bit-32g-actorder_True"
        model_directory = snapshot_download(repo_id=name, revision=revision)

        # Locate files we need within that directory

        tokenizer_path = os.path.join(model_directory, "tokenizer.model")
        model_config_path = os.path.join(model_directory, "config.json")
        st_pattern = os.path.join(model_directory, "*.safetensors")
        model_path = glob.glob(st_pattern)[0]

        config = ExLlamaConfig(model_config_path)  # create config from config.json
        config.max_seq_len = 4096  # set max sequence length
        config.model_path = model_path  # supply path to model weights file

        self.model = ExLlama(config)  # create ExLlama instance and load the weights
        self.tokenizer = ExLlamaTokenizer(
            tokenizer_path
        )  # create tokenizer from tokenizer model file

        self.cache = ExLlamaCache(self.model)  # create cache for inference

        # Only digits are allowed in the rating
        only_numbers_0_9 = [self.tokenizer.eos_token_id]
        only_numbers_1_10 = []
        numbers = self.tokenizer.encode("0123456789")[0][1:]
        vocabulary_size = self.tokenizer.tokenizer.vocab_size()
        for t in range(vocabulary_size):
            if t not in numbers:
                only_numbers_0_9.append(t)

        for t in range(vocabulary_size):
            if t not in numbers and not t == self.tokenizer.eos_token_id:
                only_numbers_1_10.append(t)

        self.generator_rating_0_9 = ExLlamaGenerator(
            self.model, self.tokenizer, self.cache
        )  # create generator

        self.generator_rating_0_9.disallow_tokens(only_numbers_0_9)
        self.generator_rating_0_9.settings.temperature = 0.6
        self.generator_rating_0_9.settings.top_p = 0.9
        self.generator_rating_0_9.settings.top_k = 50

        self.generator_rating_1_10 = ExLlamaGenerator(
            self.model, self.tokenizer, self.cache
        )  # create generator

        self.generator_rating_1_10.disallow_tokens(only_numbers_1_10)
        self.generator_rating_1_10.settings.temperature = 0.6
        self.generator_rating_1_10.settings.top_p = 0.9
        self.generator_rating_1_10.settings.top_k = 50

        # Only 1 to 5
        only_numbers_1_5 = [self.tokenizer.eos_token_id]
        numbers = self.tokenizer.encode("12345")[0][1:]
        vocabulary_size = self.tokenizer.tokenizer.vocab_size()
        for t in range(vocabulary_size):
            if t not in numbers:
                only_numbers_1_5.append(t)

        self.generator_rating_1_5 = ExLlamaGenerator(
            self.model, self.tokenizer, self.cache
        )  # create generator

        self.generator_rating_1_5.disallow_tokens(only_numbers_1_5)
        self.generator_rating_1_5.settings.temperature = 0.6
        self.generator_rating_1_5.settings.top_p = 0.9
        self.generator_rating_1_5.settings.top_k = 50

        # Only one to ten
        only_numbers_text = [self.tokenizer.eos_token_id]
        numbers = self.tokenizer.encode(
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
        vocabulary_size = self.tokenizer.tokenizer.vocab_size()
        for t in range(vocabulary_size):
            if t not in numbers:
                only_numbers_text.append(t)

        self.generator_rating_text = ExLlamaGenerator(
            self.model, self.tokenizer, self.cache
        )  # create generator

        self.generator_rating_text.disallow_tokens(only_numbers_text)
        self.generator_rating_text.settings.temperature = 0.6
        self.generator_rating_text.settings.top_p = 0.9
        self.generator_rating_text.settings.top_k = 50

        self.generator_explanation = ExLlamaGenerator(
            self.model, self.tokenizer, self.cache
        )  # create generator

        self.generator_explanation.settings.temperature = 0.6
        self.generator_explanation.settings.top_p = 0.9
        self.generator_explanation.settings.top_k = 50

    def request_rating_0_9(self, system_prompt, dialog) -> Tuple[str, str]:
        prompt = self.encode(system_prompt, dialog)
        if self.use_cache:
            return prompt, self.generate_reuse_simple(
                self.generator_rating_0_9, prompt, max_new_tokens=1
            )
        else:
            return prompt, self.generator_rating_0_9.generate_simple(
                prompt, max_new_tokens=1
            )

    def request_rating_1_5(self, system_prompt, dialog) -> Tuple[str, str]:
        prompt = self.encode(system_prompt, dialog)
        if self.use_cache:
            return prompt, self.generate_reuse_simple(
                self.generator_rating_1_5, prompt, max_new_tokens=1
            )
        else:
            return prompt, self.generator_rating_1_5.generate_simple(
                prompt, max_new_tokens=1
            )

    def request_rating_1_10(
        self, system_prompt, dialog
    ) -> Tuple[str, str]:  # TODO need fix
        prompt = self.encode(system_prompt, dialog)
        if self.use_cache:
            return prompt, self.generate_reuse_simple(
                self.generator_rating_1_10, prompt, max_new_tokens=2
            )
        else:
            return prompt, self.generator_rating_1_10.generate_simple(
                prompt, max_new_tokens=2
            )

    def request_rating_text(self, system_prompt, dialog) -> Tuple[str, str]:
        prompt = self.encode(system_prompt, dialog)
        if self.use_cache:
            return prompt, self.generate_reuse_simple(
                self.generator_rating_text, prompt, max_new_tokens=1
            )
        else:
            return prompt, self.generator_rating_text.generate_simple(
                prompt, max_new_tokens=1
            )

    def request_explanation(self, system_prompt, dialog) -> Tuple[str, str]:
        prompt = self.encode(system_prompt, dialog)
        if self.use_cache:
            return prompt, self.generate_reuse_simple(
                self.generator_explanation, prompt, max_new_tokens=300
            )
        else:
            return prompt, self.generator_explanation.generate_simple(
                prompt, max_new_tokens=300
            )

    def generate_reuse_simple(
        self, generator: ExLlamaGenerator, prompt, max_new_tokens=128
    ):
        generator.end_beam_search()

        ids = generator.tokenizer.encode(prompt)
        generator.gen_begin_reuse(ids)

        max_new_tokens = min(
            max_new_tokens, generator.model.config.max_seq_len - ids.shape[1]
        )

        eos = torch.zeros((ids.shape[0],), dtype=torch.bool)
        for i in range(max_new_tokens):
            token = generator.gen_single_token()
            for j in range(token.shape[0]):
                if token[j, 0].item() == generator.tokenizer.eos_token_id:
                    eos[j] = True
            if eos.all():
                break

        text = generator.tokenizer.decode(
            generator.sequence[0][len(ids[0]) :]
            if generator.sequence.shape[0] == 1
            else generator.sequence
        )
        return text
