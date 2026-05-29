import re
import pandas as pd
from typing import Union

from enhancer.enhancer_interfaces.utils.dataset_prompt import DatasetPrompt
from enhancer.enhancer_interfaces.utils.prompt_utils import PromptUtils


class StrokePrompt(DatasetPrompt):
    CONDITION_NAME = "a stroke"
    POSITIVE_LABEL = "had a stroke"
    NEGATIVE_LABEL = "did not have a stroke"

    def build_patient_description(self, row: pd.Series) -> str:
        return (
            f"The patient "
            f"is {row['gender']} ('gender'), "
            f"has Age: {row['age']} ('age'), "
            f"{PromptUtils.describe_binary_feature(row['hypertension'], 'Hypertension', 'hypertension')}, "
            f"{PromptUtils.describe_binary_feature(row['heart_disease'], 'Heart Disease', 'heart_disease')}, "
            f"{'is married' if row['ever_married'] == 'Yes' else 'is not married'} ('ever_married'), "
            f"works as {row['work_type']} ('work_type'), "
            f"lives in a {row['residence_type']} area ('Residence_type'), "
            f"has average glucose level: {row['avg_glucose_level']} ('avg_glucose_level'), "
            f"has Body Mass Index: {row['bmi']} ('bmi'), "
            f"{'smoking status: ' + row['smoking_status'] if row['smoking_status'] != 'Unknown' else 'smoking status is unknown'} ('smoking_status'), "
        )

    # def build_question(self, row: pd.Series) -> str:
    #     description = self.build_patient_description(row)

    #     return f"""
    #         You are a helpful and reliable medical expert. Your task is to classify whether a patient had a stroke based on their medical data and the provided supporting documents.

    #         Below is a detailed description of the patient's information. For each health-related element, the name of the original feature is provided in parentheses. You must refer to these feature names in your response.

    #         Based on the following information:
    #         {description}

    #         Please answer the question: **Did the patient have a stroke?**

    #         Your response must be structured as a JSON object with the following fields:

    #         - "step_by_step_thinking"
    #         - "classification" (1 = had a stroke, 0 = did not have a stroke)
    #         - "feature_importance_ranking"
    #         - "rules_applied"

    #         Only include features that were relevant to the decision.
    #         """
    
    # def build_question(self, row: pd.Series):
    #     """
    #     Build the MedRAG-compatible prompt for a stroke patient row, 
    #     and return options_dict with string keys.
    #     """
    #     # Dataset-specific patient description
    #     description = self.build_patient_description(row)

    #     # Dataset-specific question text
    #     question_text = (
    #         f"Please answer the question: **Did the patient have a stroke?**\n\n"
    #         f"Patient info:\n{description}"
    #     )

    #     # Options for MedRAG (string keys)
    #     options_dict = {"0": "No", "1": "Yes"}
    #     option_text = ", ".join(f"{k} = {v}" for k, v in options_dict.items())

    #     # Fill the MedRAG template
    #     final_prompt = self.MEDRAG_PROMPT_TEMPLATE.substitute(
    #         base_instruction=self.BASE_INSTRUCTION,
    #         question_text=question_text,
    #         option_text=option_text,
    #         output_format=self.OUTPUT_FORMAT_INSTRUCTION
    #     )

    #     # Return prompt string and options dictionary
    #     return final_prompt, options_dict

    # def build_question(self, row: pd.Series):
    #     """
    #     Build the MedRAG-compatible prompt for a stroke patient row.
    #     """

    #     # Dataset-specific patient description
    #     description = self.build_patient_description(row)

    #     # Dataset-specific question text
    #     question_text = (
    #         f"Please answer the question: **Did the patient have {self.CONDITION_NAME}?**\n"
    #         f"Based on the following information:\n"
    #         f"{description}"
    #     )

    #     # Fill dynamic output-format section
    #     output_format = self.OUTPUT_FORMAT_TEMPLATE.substitute(
    #         positive_label=self.POSITIVE_LABEL,
    #         negative_label=self.NEGATIVE_LABEL,
    #     )

    #     # Final prompt
    #     final_prompt = self.MEDRAG_PROMPT_TEMPLATE.substitute(
    #         base_instruction=self.BASE_INSTRUCTION,
    #         question_text=question_text,
    #         output_format=output_format,
    #     )

    #     # Return prompt and options for MedRAG
    #     return final_prompt, None


