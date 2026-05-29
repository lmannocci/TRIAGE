import re
import pandas as pd
from typing import Union

from enhancer.enhancer_interfaces.utils.dataset_prompt import DatasetPrompt


class LiverPrompt(DatasetPrompt):
    CONDITION_NAME = "liver disease"
    POSITIVE_LABEL = "has liver disease"
    NEGATIVE_LABEL = "does not have liver disease"

    def build_patient_description(self, row: pd.Series) -> str:
        return (
            f"The patient is {row['age']} years old ('age'), "
            f"has Gender: {row['gender']} ('gender'), "
            f"has total bilirubin: {row['totalBilirubin']} ('totalBilirubin'), "
            f"has direct bilirubin: {row['directBilirubin']} ('directBilirubin'), "
            f"has alkaline phosphotase: {row['alkalinePhosphatase']} ('alkalinePhosphatase'), "
            f"has alanine aminotransferase: {row['alanineAminotransferase']} ('alanineAminotransferase'), "
            f"has aspartate aminotransferase: {row['aspartateAminotransferase']} ('aspartateAminotransferase'), "
            f"has total proteins: {row['totalProteins']} ('totalProteins'), "
            f"has albumin: {row['albumin']} ('albumin'), "
            f"has albumin and globulin ratio: {row['albuminGlobulinRatio']} ('albuminGlobulinRatio')."
        )

    # def build_question(self, row: pd.Series) -> str:
    #     description = self.build_patient_description(row)

    #     return f"""
    #         You are a helpful and reliable medical expert. Your task is to classify whether a patient has liver disease based on their medical data and the provided supporting documents.

    #         Below is a detailed description of the patient's information. For each health-related element, the name of the original feature is provided in parentheses. You must refer to these feature names in your response.

    #         Based on the following information:
    #         {description}

    #         Please answer the question: **Does the patient have liver disease?**

    #         Your response must be structured as a JSON object with the following fields:

    #         - "step_by_step_thinking"
    #         - "classification" (1 = has liver disease, 0 = does not have liver disease)
    #         - "feature_importance_ranking"
    #         - "rules_applied"

    #         Only include features that were relevant to the decision.
    #         """

    # def build_question(self, row: pd.Series):
    #     """
    #     Build the MedRAG-compatible prompt for a liver patient row, 
    #     and return options_dict with string keys.
    #     """
    #     # Dataset-specific patient description
    #     description = self.build_patient_description(row)

    #     # Dataset-specific question text
    #     question_text = (
    #         f"Please answer the question: **Does the patient have liver disease?**\n\n"
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
    #     Build the MedRAG-compatible prompt for a liver patient row.
    #     """

    #     # Dataset-specific patient description
    #     description = self.build_patient_description(row)

    #     # Dataset-specific question text
    #     question_text = (
    #         f"Please answer the question: **Does the patient have {self.CONDITION_NAME}?**\n"
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
