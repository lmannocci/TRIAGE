from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import importlib
import json
import re
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from enhancer.enhancer_interfaces.utils.answer_processing import process_enhancer_answer
from enhancer.enhancer_interfaces.utils.dataset_prompt_factory import DatasetPromptFactory
from utils.common_variables import ind
from utils.log_manager.log_manager import LogManager


@dataclass
class PromptPayload:
    """Provider-neutral prompt request produced by a task."""

    provider_name: str
    prompt: Optional[str] = None
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    options: Optional[Dict[str, str]] = None
    json_prefix: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def text_for_debug(self) -> str:
        if self.prompt is not None:
            return self.prompt
        parts = [part for part in (self.system_prompt, self.user_prompt) if part]
        return "\n".join(parts)


class LLMTask(ABC):
    """
    A pluggable task definition for an LLM-backed enhancer.

    Subclasses decide how rows become prompts and how raw LLM answers become
    output rows. Backends only know how to run a provider such as MedRAG or
    HuggingFace.
    """

    name = "llm_task"

    @abstractmethod
    def build_prompt(self, row: pd.Series, provider_name: str) -> PromptPayload:
        pass

    @abstractmethod
    def process_answer(
        self,
        lm: LogManager,
        answer: str,
        snippets: list,
        scores: list,
        row: pd.Series,
        row_position: int,
        index_column: str = ind,
    ) -> Dict[str, Any]:
        pass

    def to_config(self) -> Dict[str, Any]:
        return {
            "task_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "params": self._config_params(),
        }

    def _config_params(self) -> Dict[str, Any]:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _config_params() to run in parallel."
        )


class DatasetClassificationTask(LLMTask):
    """Default binary classification task used by the current enhancer flows."""

    name = "dataset_classification"
    JSON_PREFIX = '{"step_by_step_thinking": "'

    def __init__(
        self,
        dataset_prefix: str,
        enhancer_name: str,
        llm_name: str,
        info: Dict[str, Any],
        return_options: bool,
    ):
        self.dataset_prefix = dataset_prefix
        self.enhancer_name = enhancer_name
        self.llm_name = llm_name
        self.info = info
        self.return_options = return_options

    def _config_params(self) -> Dict[str, Any]:
        return {
            "dataset_prefix": self.dataset_prefix,
            "enhancer_name": self.enhancer_name,
            "llm_name": self.llm_name,
            "info": self.info,
            "return_options": self.return_options,
        }

    def build_prompt(self, row: pd.Series, provider_name: str) -> PromptPayload:
        dataset_prompt = DatasetPromptFactory.create(self.dataset_prefix)

        if provider_name == "medrag":
            prompt, options = dataset_prompt.build_question(
                row,
                return_options=self.return_options,
                enhancer_name=provider_name,
                llm_name=self.llm_name,
            )
            return PromptPayload(
                provider_name=provider_name,
                prompt=prompt,
                options=options,
            )

        if provider_name == "huggingface":
            system_prompt, user_prompt, options = dataset_prompt.build_question(
                row,
                return_options=self.return_options,
                enhancer_name=provider_name,
                llm_name=self.llm_name,
            )
            return PromptPayload(
                provider_name=provider_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                options=options,
                json_prefix=self.JSON_PREFIX,
            )

        raise ValueError(f"Unsupported provider for dataset classification: {provider_name}")

    def process_answer(
        self,
        lm: LogManager,
        answer: str,
        snippets: list,
        scores: list,
        row: pd.Series,
        row_position: int,
        index_column: str = ind,
    ) -> Dict[str, Any]:
        answer_dict_column = (
            f"{self.enhancer_name}_answer_dict"
            if self.enhancer_name == "huggingface"
            else None
        )

        return process_enhancer_answer(
            lm=lm,
            enhancer_name=self.enhancer_name,
            info=self.info,
            answer=answer,
            snippets=snippets,
            scores=scores,
            row=row,
            index_column=index_column,
            row_position=row_position,
            answer_dict_column=answer_dict_column,
            replace_answer_with_dict=self.enhancer_name == "medrag",
        )


def create_task_from_config(config: Dict[str, Any]) -> LLMTask:
    task_type = config.get("task_type")
    params = config.get("params", {})

    if task_type in {None, "dataset_classification"}:
        return DatasetClassificationTask(**params)

    module_name, class_name = task_type.rsplit(".", 1)
    module = importlib.import_module(module_name)
    task_cls = getattr(module, class_name)
    return task_cls(**params)


class EnhancerValidationTask(LLMTask):
    """LLM-as-a-judge task for evaluating generated enhancer outputs."""

    name = "enhancer_validation"

    def __init__(self, judge_name: str = "llm_judge", evaluation_type: str = "explanation"):
        self.judge_name = judge_name
        self.evaluation_type = evaluation_type

    def _config_params(self) -> Dict[str, Any]:
        return {
            "judge_name": self.judge_name,
            "evaluation_type": self.evaluation_type,
        }

    def build_prompt(self, row: pd.Series, provider_name: str) -> PromptPayload:
        if self.evaluation_type == "retrieved_documents":
            prompt = self._build_retrieved_documents_prompt(row)
        else:
            prompt = self._build_explanation_prompt(row)

        if provider_name == "huggingface":
            return PromptPayload(
                provider_name=provider_name,
                system_prompt="You are a careful clinical evaluation judge. Return only valid JSON.",
                user_prompt=prompt,
            )

        if provider_name == "medrag":
            return PromptPayload(provider_name=provider_name, prompt=prompt)

        raise ValueError(f"Unsupported provider for enhancer validation: {provider_name}")

    def _build_explanation_prompt(self, row: pd.Series) -> str:
        return f"""
You are evaluating a clinical explanation generated by an AI system.

Evaluate the explanation according to these criteria:

1. feature_faithfulness
Does the explanation focus on the important features provided by the explainer?

2. prediction_consistency
Does the explanation support the enhancer prediction?

3. unsupported_claims
Does the explanation introduce unsupported or contradictory medical claims?
Score 5 means no unsupported or contradictory claims; score 1 means many severe unsupported or contradictory claims.

4. clarity
Is the explanation understandable for a clinician?

Assign an integer score from 1 (very poor) to 5 (excellent).
Set overall_score to the average of the four criterion scores.

Output ONLY valid JSON.

Patient information:
{row["patient_description"]}

Enhancer prediction:
{row["enhancer_prediction"]}

Important explainer features:
{row["explainer_ranking"]}

Generated explanation:
{row["generated_explanation"]}

Output format:
{{
    "feature_faithfulness": int,
    "prediction_consistency": int,
    "unsupported_claims": int,
    "clarity": int,
    "overall_score": float,
    "short_reason": "..."
}}
""".strip()

    def _build_retrieved_documents_prompt(self, row: pd.Series) -> str:
        return f"""
You are evaluating retrieved medical documents used by a RAG-based clinical explanation system.

Evaluate the following dimensions:

1. retrieval_relevance
Are the retrieved documents relevant to the generated explanation and prediction?

2. grounding
Is the generated explanation supported by the retrieved documents?

3. unsupported_claims
Does the explanation contain claims not supported by the retrieved documents?

Assign an integer score from 1 (very poor) to 5 (excellent).

Output ONLY valid JSON.

Enhancer prediction:
{row["enhancer_prediction"]}

Retrieved documents:
{row["retrieved_documents"]}

Generated explanation:
{row["generated_explanation"]}

Output format:
{{
    "retrieval_relevance": int,
    "grounding": int,
    "unsupported_claims": int,
    "overall_score": float,
    "short_reason": "..."
}}
""".strip()

    def process_answer(
        self,
        lm: LogManager,
        answer: str,
        snippets: list,
        scores: list,
        row: pd.Series,
        row_position: int,
        index_column: str = ind,
    ) -> Dict[str, Any]:
        if self.evaluation_type == "retrieved_documents":
            output_fields = ["retrieval_relevance", "grounding", "unsupported_claims"]
            row_fields = ["enhancer_prediction", "retrieved_documents", "generated_explanation"]
        else:
            output_fields = ["feature_faithfulness", "prediction_consistency", "unsupported_claims", "clarity"]
            row_fields = ["enhancer_prediction", "explainer_ranking", "generated_explanation"]

        result = {index_column: row[index_column]}
        for field in row_fields:
            result[field] = row.get(field, np.nan)

        result.update({
            f"{self.judge_name}_answer": answer,
            "overall_score": np.nan,
            "short_reason": np.nan,
            "parse_status": "failed",
            "parse_method": None,
        })
        for field in output_fields:
            result[field] = np.nan

        try:
            answer_dict = self._parse_json_answer(answer)
        except Exception as exc:
            lm.printl(f"Failed to parse judge answer patient {row_position}: {exc}")
            return result

        for field in output_fields:
            result[field] = self._score(answer_dict.get(field))

        result.update({
            "overall_score": self._overall_score(answer_dict, output_fields),
            "short_reason": answer_dict.get("short_reason", np.nan),
            f"{self.judge_name}_answer_dict": answer_dict,
            "parse_status": "success",
            "parse_method": "json",
        })
        return result

    @staticmethod
    def _parse_json_answer(answer: str) -> Dict[str, Any]:
        if isinstance(answer, dict):
            return answer

        match = re.search(r"\{.*\}", str(answer), re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in judge answer")

        return json.loads(match.group(0))

    @staticmethod
    def _score(value):
        try:
            score = int(value)
        except (TypeError, ValueError):
            return np.nan

        if score < 1 or score > 5:
            return np.nan
        return score

    def _overall_score(self, answer_dict: Dict[str, Any], score_fields: list[str]):
        try:
            return float(answer_dict.get("overall_score"))
        except (TypeError, ValueError):
            scores = [self._score(answer_dict.get(field)) for field in score_fields]
            scores = [score for score in scores if not pd.isna(score)]
            return float(np.mean(scores)) if scores else np.nan
