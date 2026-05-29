import re
import pandas as pd
from typing import Union
import json
from enhancer.enhancer_interfaces.utils.dataset_prompt import DatasetPrompt

class PimaPrompt(DatasetPrompt):
    # Dataset-specific constants (no options version)
    CONDITION_NAME = "diabetes"
    POSITIVE_LABEL = "has diabetes"
    NEGATIVE_LABEL = "does not have diabetes"

    # Method to create a textual description of each patient
    def build_patient_description(self, row: pd.Series):
        description = (f"The patient has Age: {row['Age']} ('Age'), "
                       f"has had Pregnancies: {row['Pregnancies']} ('Pregnancies'), "
                       f"has Glucose: {row['Glucose']} ('Glucose'), "
                       f"has BloodPressure: {row['BloodPressure']} ('BloodPressure'), "
                       f"has SkinThickness: {row['SkinThickness']} ('SkinThickness'), "
                       f"has Insulin: {row['Insulin']} ('Insulin'), "
                       f"has BMI: {row['BMI']} ('BMI'), "
                       f"has DiabetesPedigreeFunction: {row['DiabetesPedigreeFunction']} ('DiabetesPedigreeFunction').")
        return description

    # def build_question(self, row: pd.Series) -> str:
    #     description = self.build_patient_description(row)
    #     question = f"""
    #         You are a helpful and reliable medical expert. Your task is to classify whether a patient has diabetes based on their medical data and the provided supporting documents.

    #         Below is a detailed description of the patient's information. For each health-related element, the name of the original feature is provided in parentheses. You must refer to these feature names in your response.

    #         Based on the following information:
    #         {description}

    #         Please answer the question: **Does the patient have diabetes?**

    #         Your response must be structured as a JSON object with the following fields:

    #         - `"step_by_step_thinking"`: A clear, medically sound explanation of your reasoning process leading to the classification.
    #         - `"classification"`: An integer value, where `1` means the patient **has diabetes**, and `0` means the patient **does not have diabetes**.
    #         - `"feature_importance_ranking"`: A list of the most relevant features (by name) that influenced your decision, ordered from most to least important.
    #         - `"rules_applied"`: A dictionary mapping feature names to the specific medical rule or logic used in the decision-making process.

    #         Only include features that were relevant to the decision. If a feature did not influence your judgment, omit it from the ranking and rules.

    #         Your answer will be used for research purposes only, so please provide a well-justified and definite response.
    #     """
    #     return question

    # def build_question(self, row: pd.Series):
    #     # Dataset-specific patient description
    #     description = self.build_patient_description(row)

    #     # Dataset-specific question text
    #     question_text = f"Please answer the question: **Does the patient have diabetes?**\n\nPatient info:\n{description}"

    #     # Options for MedRAG
    #     options_dict = {"0": "No", "1": "Yes"}
    #     option_text = ", ".join(f"{k} = {v}" for k, v in options_dict.items())

    #     # Fill the template
    #     final_prompt = self.MEDRAG_PROMPT_TEMPLATE.substitute(
    #         base_instruction=self.BASE_INSTRUCTION,
    #         question_text=question_text,
    #         option_text=option_text,
    #         output_format=self.OUTPUT_FORMAT_INSTRUCTION
    #     )

    #     # Return prompt and options for MedRAG
    #     return final_prompt, options_dict
    

    # def build_question(self, row: pd.Series):
    #     # Dataset-specific patient description
    #     description = self.build_patient_description(row)

    #     # Dataset-specific question text
    #     question_text = f"Please answer the question: **Does the patient have {self.CONDITION_NAME}?**\nBased on the following information:\n{description}"

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

