from abc import ABC, abstractmethod
import pandas as pd
from enhancer.enhancer_interfaces.utils.template_prompts import *
from typing import Optional

class DatasetPrompt(ABC):
    # ---- MUST be defined by subclasses ----
    CONDITION_NAME: str
    POSITIVE_LABEL: str
    NEGATIVE_LABEL: str



    @abstractmethod
    def build_patient_description(self, row: pd.Series) -> str:
        pass

    def __build_medrag_prompt(self, description: str, return_options: bool):
        self.MEDRAG_PROMPT = self.BASE_INSTRUCTION.substitute(
            description=description,
            return_options=return_options
        )

    # MEDRAG_BASE_INSTRUCTION, MEDRAG_OUTPUT_FORMAT_TEMPLATE, MEDRAG_PROMPT_TEMPLATE
    def setting_prompt(self, enhancer_name: str, llm_name: str):
        if enhancer_name == "medrag":
            self.BASE_INSTRUCTION = MEDRAG_BASE_INSTRUCTION
            self.OUTPUT_FORMAT_TEMPLATE = MEDRAG_OUTPUT_FORMAT_TEMPLATE
            self.FINAL_PROMPT_TEMPLATE = MEDRAG_FINAL_PROMPT_TEMPLATE
        elif enhancer_name == "huggingface":
            self.SYSTEM_PROMPT = HUGGINGFACE_SYSTEM_PROMPT
        
        self.POSITIVE_LABEL = POSITIVE_LABEL
        self.NEGATIVE_LABEL = NEGATIVE_LABEL    

    def build_question(self, row: pd.Series, return_options: bool, enhancer_name: str, llm_name: str):
        """
        Build a MedRAG-compatible prompt for binary medical classification.
        """
        self.setting_prompt(enhancer_name, llm_name)

        description = self.build_patient_description(row)

        question_text = (
            f"Answer the question: **Does the patient have {self.CONDITION_NAME}?**\n"
            f"Based on the following information:\n{description}"
        )

        if return_options:
            # Options for MedRAG
            # options_dict = {"0=": "No", "1": "Yes"}
            # option_text = ", ".join(f"{k} = {v}" for k, v in options_dict.items())
            options_dict = {
                "A": "Yes",
                "B": "No"
            }
        else:
            options_dict = None

        if enhancer_name == "medrag":
            output_format = self.OUTPUT_FORMAT_TEMPLATE.substitute(
                positive_label=self.POSITIVE_LABEL,
                negative_label=self.NEGATIVE_LABEL,
            )
            final_prompt = self.FINAL_PROMPT_TEMPLATE.substitute(
                    base_instruction=self.BASE_INSTRUCTION,
                    question_text=question_text,
                    output_format=output_format,
                )
            return final_prompt, options_dict
        elif enhancer_name == "huggingface":
            system_prompt = self.SYSTEM_PROMPT.substitute(
                positive_label=self.POSITIVE_LABEL,
                negative_label=self.NEGATIVE_LABEL,
            )

            return system_prompt.strip(), question_text.strip(), options_dict

    def build_prompt(self, system_prompt: str, user_prompt: str) -> str:
        return (
            "<s>[INST] <<SYS>>\n"
            f"{system_prompt}\n"
            "<</SYS>>\n\n"
            f"{user_prompt}\n"
            "[/INST]\n"
        )
    # @abstractmethod
    # def build_question(self, row: pd.Series) -> str:
    #     pass

    # @abstractmethod
    # def extract_classification(self, answer: str):
    #     pass
