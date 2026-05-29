from typing import Dict

from enhancer.enhancer_interfaces.dataset_prompt_builder.pima_prompt import PimaPrompt
from enhancer.enhancer_interfaces.dataset_prompt_builder.diabetes_prompt import DiabetesPrompt
from enhancer.enhancer_interfaces.dataset_prompt_builder.stroke_prompt import StrokePrompt
from enhancer.enhancer_interfaces.dataset_prompt_builder.liver_prompt import LiverPrompt
from enhancer.enhancer_interfaces.dataset_prompt_builder.covid_prompt import CovidPrompt
from enhancer.enhancer_interfaces.utils.dataset_prompt import DatasetPrompt
# from dataset_prompts import (
#     PimaPrompt,
#     DiabetesPrompt,
#     StrokePrompt,
#     LiverPrompt,
#     CovidPrompt,
# )


class DatasetPromptFactory:

    _PROMPTS: Dict[str, type[DatasetPrompt]] = {
        "pima": PimaPrompt,
        "diabetes": DiabetesPrompt,
        "stroke": StrokePrompt,
        "liver": LiverPrompt,
        "covid": CovidPrompt,
    }

    @classmethod
    def create(cls, dataset_prefix: str) -> DatasetPrompt:
        try:
            return cls._PROMPTS[dataset_prefix]()
        except KeyError:
            raise ValueError(f"Unknown dataset prefix: {dataset_prefix}")
