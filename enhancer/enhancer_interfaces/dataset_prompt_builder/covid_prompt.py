import re
import pandas as pd
from typing import Union

from enhancer.enhancer_interfaces.utils.dataset_prompt import DatasetPrompt
from enhancer.enhancer_interfaces.utils.prompt_utils import PromptUtils



class CovidPrompt(DatasetPrompt):
    CONDITION_NAME = "COVID-19"
    POSITIVE_LABEL = "has COVID-19"
    NEGATIVE_LABEL = "does not have COVID-19"

    def build_patient_description(self, row: pd.Series) -> str:
        return (
            f"The patient has age quantile: {row['age_quantile']} ('age_quantile'), "
            f"hematocrit level: {row['hematocrit']} ('hematocrit'), "
            f"hemoglobin level: {row['hemoglobin']} ('hemoglobin'), "
            f"platelet count: {row['platelets']} ('platelets'), "
            f"mean platelet volume: {row['mean_platelet_volume']} ('mean_platelet_volume'), "
            f"red blood cell count: {row['red_blood_cells']} ('red_blood_cells'), "
            f"lymphocyte count: {row['lymphocytes']} ('lymphocytes'), "
            f"MCHC (mean corpuscular hemoglobin concentration): {row['mchc']} ('mchc'), "
            f"leukocyte count: {row['leukocytes']} ('leukocytes'), "
            f"basophil count: {row['basophils']} ('basophils'), "
            f"MCH (mean corpuscular hemoglobin): {row['mch']} ('mch'), "
            f"eosinophil count: {row['eosinophils']} ('eosinophils'), "
            f"MCV (mean corpuscular volume): {row['mcv']} ('mcv'), "
            f"monocyte count: {row['monocytes']} ('monocytes'), "
            f"RDW (red cell distribution width): {row['rdw']} ('rdw'), "

            f"{PromptUtils.describe_binary_feature(row['respiratory_syncytial_virus'], 'a Respiratory Syncytial Virus infection', 'respiratory_syncytial_virus')}, "
            f"{PromptUtils.describe_binary_feature(row['influenza_a'], 'an Influenza A infection', 'influenza_a')}, "
            f"{PromptUtils.describe_binary_feature(row['influenza_b'], 'an Influenza B infection', 'influenza_b')}, "
            f"{PromptUtils.describe_binary_feature(row['parainfluenza_1'], 'a Parainfluenza 1 infection', 'parainfluenza_1')}, "
            f"{PromptUtils.describe_binary_feature(row['coronavirus_nl63'], 'a Coronavirus NL63 infection', 'coronavirus_nl63')}, "
            f"{PromptUtils.describe_binary_feature(row['rhinovirus_enterovirus'], 'a Rhinovirus/Enterovirus infection', 'rhinovirus_enterovirus')}, "
            f"{PromptUtils.describe_binary_feature(row['coronavirus_hku1'], 'a Coronavirus HKU1 infection', 'coronavirus_hku1')}, "
            f"{PromptUtils.describe_binary_feature(row['parainfluenza_3'], 'a Parainfluenza 3 infection', 'parainfluenza_3')}, "
            f"{PromptUtils.describe_binary_feature(row['chlamydophila_pneumoniae'], 'a Chlamydophila pneumoniae infection', 'chlamydophila_pneumoniae')}, "
            f"{PromptUtils.describe_binary_feature(row['adenovirus'], 'an Adenovirus infection', 'adenovirus')}, "
            f"{PromptUtils.describe_binary_feature(row['parainfluenza_4'], 'a Parainfluenza 4 infection', 'parainfluenza_4')}, "
            f"{PromptUtils.describe_binary_feature(row['coronavirus_229e'], 'a Coronavirus 229E infection', 'coronavirus_229e')}, "
            f"{PromptUtils.describe_binary_feature(row['coronavirus_oc43'], 'a Coronavirus OC43 infection', 'coronavirus_oc43')}, "
            f"{PromptUtils.describe_binary_feature(row['influenza_a_h1n1_2009'], 'an Influenza A H1N1 2009 infection', 'influenza_a_h1n1_2009')}, "
            f"{PromptUtils.describe_binary_feature(row['bordetella_pertussis'], 'a Bordetella pertussis infection', 'bordetella_pertussis')}, "
            f"{PromptUtils.describe_binary_feature(row['metapneumovirus'], 'a Metapneumovirus infection', 'metapneumovirus')}, "
            f"{PromptUtils.describe_binary_feature(row['parainfluenza_2'], 'a Parainfluenza 2 infection', 'parainfluenza_2')}, "

            f"neutrophil count: {row['neutrophils']} ('neutrophils'), "
            f"urea level: {row['urea']} ('urea'), "
            f"C-reactive protein level in mg/dL: {row['c_reactive_protein_mg_dl']} ('c_reactive_protein_mg_dl'), "
            f"creatinine level: {row['creatinine']} ('creatinine'), "
            f"potassium level: {row['potassium']} ('potassium'), "
            f"sodium level: {row['sodium']} ('sodium'), "
            f"{PromptUtils.describe_binary_feature(row['strepto_a'], 'a Strepto A infection', 'strepto_a')}."
        )

    # def build_question(self, row: pd.Series) -> str:
    #     description = self.build_patient_description(row)

    #     return f"""
    #         You are a helpful and reliable medical expert. Your task is to classify whether a patient has COVID-19 based on their medical data and the provided supporting documents.

    #         Below is a detailed description of the patient's information. For each health-related element, the name of the original feature is provided in parentheses. You must refer to these feature names in your response.

    #         Based on the following information:
    #         {description}

    #         Please answer the question: **Does the patient have COVID-19?**

    #         Your response must be structured as a JSON object with the following fields:

    #         - "step_by_step_thinking"
    #         - "classification" (1 = has COVID-19, 0 = does not have COVID-19)
    #         - "feature_importance_ranking"
    #         - "rules_applied"

    #         Only include features that were relevant to the decision.
    #         """

    # def build_question(self, row: pd.Series):
    #     """
    #     Build the MedRAG-compatible prompt for a COVID-19 patient row, 
    #     and return options_dict with string keys.
    #     """
    #     # Dataset-specific patient description
    #     description = self.build_patient_description(row)

    #     # Dataset-specific question text
    #     question_text = (
    #         f"Please answer the question: **Does the patient have COVID-19?**\n\n"
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
    #     Build the MedRAG-compatible prompt for a COVID-19 patient row.
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

    #     # Return prompt and no options (consistent with other datasets)
    #     return final_prompt, None
