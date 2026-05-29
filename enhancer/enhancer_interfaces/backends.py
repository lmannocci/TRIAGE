from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import configparser
import os
from typing import Any, Dict, Optional, Tuple

from huggingface_hub import login
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from enhancer.enhancer_interfaces.tasks import PromptPayload
from src.medrag import MedRAG
from utils.log_manager.log_manager import LogManager


absolute_path = os.path.dirname(__file__)
corpus_path = os.path.join(absolute_path, f"..{os.sep}..{os.sep}corpus{os.sep}")
cache_path = os.path.join(absolute_path, f"..{os.sep}..{os.sep}..{os.sep}vast{os.sep}")
main_path = os.path.join(absolute_path, f"..{os.sep}..{os.sep}")


def login_huggingface_from_config() -> None:
    config = configparser.ConfigParser()
    config.read(f"{main_path}config.ini")

    token = config.get("huggingface", "token", fallback=None)
    if not token:
        raise ValueError("HuggingFace token not found in config.ini")

    login(token)


@dataclass
class ModelPrompt:
    prompt: str
    options: Optional[Dict[str, str]] = None
    forced_prefix: str = ""


class LLMBackend(ABC):
    provider_name: str

    def __init__(self, lm: LogManager, config: Dict[str, Any]):
        self.lm = lm
        self.config = config

    @classmethod
    def from_config(cls, config: Dict[str, Any], lm: LogManager) -> "LLMBackend":
        return cls(lm=lm, config=config)

    @abstractmethod
    def initialize(self) -> None:
        pass

    @abstractmethod
    def prepare_prompt(self, payload: PromptPayload) -> ModelPrompt:
        pass

    @abstractmethod
    def answer(self, model_prompt: ModelPrompt) -> Tuple[str, list, list]:
        pass


class MedRAGBackend(LLMBackend):
    provider_name = "medrag"

    def initialize(self) -> None:
        login_huggingface_from_config()
        self.medrag = MedRAG(
            llm_name=self.config["llm_name"],
            rag=self.config["rag"],
            retriever_name=self.config["retriever_name"],
            corpus_name=self.config["corpus_name"],
            corpus_cache=True,
            db_dir=self.config.get("db_dir", corpus_path),
            cache_dir=self.config.get("cache_dir", cache_path),
        )

    def prepare_prompt(self, payload: PromptPayload) -> ModelPrompt:
        if payload.prompt is None:
            raise ValueError("MedRAG tasks must provide a single prompt string.")
        return ModelPrompt(prompt=payload.prompt, options=payload.options)

    def answer(self, model_prompt: ModelPrompt) -> Tuple[str, list, list]:
        return self.medrag.answer(
            question=model_prompt.prompt,
            options=model_prompt.options,
            k=self.config["rag_K_documents"],
        )


class HuggingFaceBackend(LLMBackend):
    provider_name = "huggingface"

    def initialize(self) -> None:
        login_huggingface_from_config()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["llm_model"],
            use_fast=False,
            cache_dir=cache_path,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["llm_model"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cache_path,
        )

    def prepare_prompt(self, payload: PromptPayload) -> ModelPrompt:
        if payload.system_prompt is None or payload.user_prompt is None:
            raise ValueError("HuggingFace tasks must provide system_prompt and user_prompt.")

        self._ensure_chat_template()
        messages = [
            {"role": "system", "content": payload.system_prompt},
            {"role": "user", "content": payload.user_prompt},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt += payload.json_prefix
        return ModelPrompt(
            prompt=prompt,
            options=payload.options,
            forced_prefix=payload.json_prefix,
        )

    def answer(self, model_prompt: ModelPrompt) -> Tuple[str, list, list]:
        inputs = self.tokenizer(model_prompt.prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=2048,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.1,
        )

        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        if model_prompt.forced_prefix:
            generated_text = model_prompt.forced_prefix + generated_text
            answer = generated_text.split("}")[0] + "}"
        else:
            answer = generated_text

        return answer, [], []

    def _ensure_chat_template(self) -> None:
        if self.tokenizer.chat_template is not None:
            return

        self.tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "{{ '<<SYS>>\\n' + message['content'] + '\\n<</SYS>>\\n\\n' }}"
            "{% elif message['role'] == 'user' %}"
            "{{ '[INST] ' + message['content'] + ' [/INST]' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ message['content'] }}"
            "{% endif %}"
            "{% endfor %}"
        )
