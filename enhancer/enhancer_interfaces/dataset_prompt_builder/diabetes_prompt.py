import re
import pandas as pd
from typing import Union

from enhancer.enhancer_interfaces.utils.dataset_prompt import DatasetPrompt
from enhancer.enhancer_interfaces.utils.prompt_utils import PromptUtils


class DiabetesPrompt(DatasetPrompt):
    CONDITION_NAME = "diabetes"
    POSITIVE_LABEL = "has diabetes"
    NEGATIVE_LABEL = "does not have diabetes"
    def build_patient_description(self, row: pd.Series) -> str:
        return (
            f"The patient has Body Mass Index: {row['BMI']} ('BMI'), "
            f"number of days in which its mental health condition was not good in the past 30 days: "
            f"{row['MentHlth']} ('MentHlth'), "
            f"number of days in which its physical health condition was not good in the past 30 days: "
            f"{row['PhysHlth']} ('PhysHlth'), "
            f"has Age: {row['Age']} ('Age'), "
            f"{PromptUtils.describe_binary_feature(row['HighBP'], 'High Blood Pressure', 'HighBP', 'past')}, "
            f"{PromptUtils.describe_binary_feature(row['HighChol'], 'High Cholesterol', 'HighChol', 'past')}, "
            f"{PromptUtils.describe_binary_feature(row['CholCheck'], 'been checked for cholesterol levels in the last 5 years', 'CholCheck', 'past')}, "
            f"{PromptUtils.describe_binary_feature(row['Smoker'], 'smoked at least 100 cigarettes in its entire life', 'Smoker', 'past')}, "
            f"{PromptUtils.describe_binary_feature(row['Stroke'], 'a Stroke', 'Stroke', 'past')}, "
            f"{PromptUtils.describe_binary_feature(row['HeartDiseaseorAttack'], 'a coronary heart disease or myocardial infarction', 'HeartDiseaseorAttack', 'past')}, "
            f"{PromptUtils.describe_binary_feature(row['PhysActivity'], 'engaged in Physical Activity in the last 30 days', 'PhysActivity')}, "
            f"{'consumes Fruits 1 or more times per day' if row['Fruits'] == 1 else 'does not consume Fruits 1 or more times per day'} ('Fruits'), "
            f"{'consumes Vegetables 1 or more times per day' if row['Veggies'] == 1 else 'does not consume Vegetables 1 or more times per day'} ('Veggies'), "
            f"{PromptUtils.describe_binary_feature(row['HvyAlcoholConsump'], 'Heavy Alcohol Consumption', 'HvyAlcoholConsump')}, "
            f"{PromptUtils.describe_binary_feature(row['AnyHealthcare'], 'any kind of health care coverage, including health insurance', 'AnyHealthcare')}, "
            f"{'was there a time in the past 12 months when it needed to see a doctor but could not because of cost' if int(row['NoDocbcCost']) == 1 else 'was not there a time in the past 12 months when it needed to see a doctor but could not because of cost'} ('NoDocbcCost'), "
            f"on a scale from 1 to 5 in general its health is: {row['GenHlth']} ('GenHlth'), "
            f"{PromptUtils.describe_binary_feature(row['DiffWalk'], 'Difficulty in Walking', 'DiffWalk')}, "
            f"{'is male' if int(row['Sex']) == 1 else 'is female'} ('Sex'), "
            f"on a scale from 1 to 6 has Level of Education: {row['Education']} ('Education'), "
            f"on a scale from 1 to 8 has Income Level: {row['Income']} ('Income')."
        )

    # def build_question(self, row: pd.Series) -> str:
    #     description = self.build_patient_description(row)

    #     return f"""
    #         You are a helpful and reliable medical expert. Your task is to classify whether a patient has diabetes based on their medical data and the provided supporting documents.

    #         Below is a detailed description of the patient's information. For each health-related element, the name of the original feature is provided in parentheses. You must refer to these feature names in your response.

    #         Based on the following information:
    #         {description}

    #         Please answer the question: **Does the patient have diabetes?**

    #         Your response must be structured as a JSON object with the following fields:

    #         - "step_by_step_thinking"
    #         - "classification" (1 = diabetes, 0 = no diabetes)
    #         - "feature_importance_ranking"
    #         - "rules_applied"

    #         Only include features that were relevant to the decision.
    #         """

    # def build_question(self, row: pd.Series):
    #     """
    #     Build the MedRAG-compatible prompt for a diabetes patient row, 
    #     and return options_dict with string keys.
    #     """
    #     # Dataset-specific patient description
    #     description = self.build_patient_description(row)

    #     # Dataset-specific question text
    #     question_text = (
    #         f"Please answer the question: **Does the patient have diabetes?**\n\n"
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
    #     Build the MedRAG-compatible prompt for a diabetes patient row.
    #     """
    #     # Dataset-specific patient description
    #     description = self.build_patient_description(row)

    #     # Dataset-specific question text
    #     question_text = (
    #         f"Please answer the question: **Does the patient have {self.CONDITION_NAME}?**\n"
    #         f"Based on the following information:\n{description}"
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

    #     # Return prompt and no options (consistent with Pima)
    #     return final_prompt, None

