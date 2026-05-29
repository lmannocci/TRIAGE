
from string import Template    
# BASE_INSTRUCTION = (
#     "You are a helpful and reliable medical expert. "
#     "Your task is to classify whether a patient has a medical condition "
#     "based on their medical data and the provided supporting documents."
# )

# OUTPUT_FORMAT_INSTRUCTION = """
#     Your response must be structured as a JSON object with the following fields:
#     - "step_by_step_thinking": A clear, medically sound explanation of your reasoning process leading to the classification.
#     - "answer_choice": Choose from the provided options: 0 = No, 1 = Yes
#     - "feature_importance_ranking": A list of the most relevant features (by name) that influenced your decision, ordered from most to least important.
#     - "rules_applied": A dictionary mapping feature names to the specific medical rule or logic used in the decision-making process.
#     Only include features that were relevant to the decision. If a feature did not influence your judgment, omit it from the ranking and rules.
#     Your answer will be used for research purposes only, so please provide a well-justified and definite response.
# """

# MEDRAG_PROMPT_TEMPLATE = Template("""$base_instruction
#     $question_text

#     Options:
#     $option_text

#     $output_format
# """)


# WITHOUT OPTIONS

# COMMON TEMPLATES
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
POSITIVE_LABEL: str = "has the condition"
NEGATIVE_LABEL: str = "does not have the condition"



# MEDRAG TEMPLATES
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

MEDRAG_BASE_INSTRUCTION = """
    You are a helpful and reliable medical expert. 
    Your task is to classify whether a patient has a medical condition based on their medical data 
    and the provided supporting documents.
"""


    # - "rules_applied": A dictionary mapping feature names to the specific medical rule or logic used in the decision-making process.
MEDRAG_OUTPUT_FORMAT_TEMPLATE = Template("""
    Your response must be structured as a JSON object with the following fields:

    - "step_by_step_thinking": A clear, medically sound explanation of your reasoning process leading to the classification.
    - "classification": An integer value, where 1 means the patient **$positive_label** and  0 means the patient **$negative_label**
    - "feature_importance_ranking": A list of the most relevant features (by name) that influenced your decision, ordered from most to least important.

    Only include features that were relevant to the decision.
    Your answer will be used for research purposes only, so please provide a well-justified and definite response.
    Output only the JSON object, without any additional commentary or text.
""")

MEDRAG_FINAL_PROMPT_TEMPLATE = Template("""
    $base_instruction
    $question_text
    $output_format
    """)

# HUGGINGFACE TEMPLATES
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
HUGGINGFACE_SYSTEM_PROMPT = Template("""
    You are a helpful and reliable medical expert. 
    Your task is to classify whether a patient has a medical condition based on their medical data 
    and the provided supporting documents.
            
    Your response must be structured as a JSON object with the following fields:
    - "step_by_step_thinking": A clear medically sound explanation
    - "classification": An integer value, where 1 means the patient **$positive_label** and  0 means the patient **$negative_label**
    - "feature_importance_ranking": A list of the most relevant features (by name) that influenced your decision, ordered from most to least important.

    Only include features that were relevant to the decision.
    Your answer will be used for research purposes only, so please provide a well-justified and definite response.
""")