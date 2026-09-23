from typing import List, Tuple, Dict, Union
import seaborn as sns
from scipy.signal import dfreqresp

dtype: Dict = {}
dpi = 300
ind = 'original_index'
level: Dict[str, int] = {
    'blackbox': 1,
    'explainer': 2,
    'enhancer': 3,
    'evaluator': 4,
    'bootstrap_confidence_intervals': 4,
    'explanation_topk_sensitivity': 4,
    'synthesizer': 5,
    'confidence_enhancer': 6,
    'selector': 7,
    'sensitivity_analysis': 7,
    'enhancer_validator': 8
}

av_datasets = ['pima', 'diabetes', 'stroke', 'liver', 'covid']
dataset_test_sizes_dict = {
    "covid": 216,
    "diabetes": 10000,
    "liver": 6137,
    "pima": 154,
    "stroke": 1022
}

av_models = ['random_forest', 'ebm', 'catboost', 'xgboost']
av_explainers = ['shap', 'lime', 'dalex', 'ebm']
av_models_explainer = {
    'random_forest': ['shap', 'lime', 'dalex'],
    'ebm': ['ebm'],
    'catboost': ['shap', 'lime', 'dalex'],
    'xgboost': ['shap', 'lime', 'dalex']
}
models_to_be_scaled = []

av_enhancers = ['medrag', 'huggingface']
av_enhancers_rag = {'medrag': [True, False],
                    'huggingface': [False]
                    }

av_enhancers_llm = {
    'medrag': ['mixtral', 'llama2'],
    'huggingface': ['meditron']
}

llms_map = {
    'mixtral': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
    'llama2': 'meta-llama/Llama-2-70b-chat-hf',
    'meditron': 'epfl-llm/meditron-70b'
}
llm_map_inverse = {v: k for k, v in llms_map.items()}

av_llms = ['mixtral', 'llama2', 'meditron']
av_corpus = ['StatPearls']
av_retrievers = ['MedCPT']


av_selectors = ['simple', 'abstain']
av_metrics_exp_ranking = ['rbo']

av_synthesizers = ['ctgan']

av_metrics_neighbors = ['cosine_similarity']

av_aggregation_methods = ['average', 'agreement']

df_info: Dict[str, Dict[str, Union[str, List[str]]]] = {
    'pima': {
        'columns': ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                    'DiabetesPedigreeFunction', 'Age', 'Outcome'],
        'x_columns': ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                      'DiabetesPedigreeFunction', 'Age'],
        'num_columns': ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                        'DiabetesPedigreeFunction', 'Age'],
        'cat_columns': [],
        'target': 'Outcome',
        'classes': [0, 1]
    },
    'diabetes' : {
        'columns': ['Outcome', 'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
        'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
        'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education',
        'Income'],
        'x_columns': ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
        'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
        'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education',
        'Income'],
        'num_columns': ['BMI', 'MentHlth', 'PhysHlth', 'Age'],
        'cat_columns': ['HighBP', 'HighChol', 'CholCheck', 'Smoker',
        'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'DiffWalk', 'Sex', 'Education', 'Income'],
        'target': 'Outcome',
        'classes': [0, 1]
    },
    'stroke': {
        'columns': ['id', 'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
        'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
        'smoking_status', 'Outcome'],
        'x_columns': ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
        'work_type', 'residence_type', 'avg_glucose_level', 'bmi',
        'smoking_status'],
        'cat_columns': ['gender', 'ever_married', 'work_type', 'residence_type', 'smoking_status'],
        'num_columns': ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi'],
        'target': 'Outcome',
        'classes': [0, 1]
    },
    'liver': {
        'columns': ['age', 'gender', 'totalBilirubin', 'directBilirubin','alkalinePhosphatase', 
        'alanineAminotransferase','aspartateAminotransferase', 'totalProteins', 'albumin',
        'albuminGlobulinRatio', 'Outcome'],
        'x_columns': ['age', 'gender', 'totalBilirubin', 'directBilirubin','alkalinePhosphatase', 
        'alanineAminotransferase','aspartateAminotransferase', 'totalProteins', 'albumin',
        'albuminGlobulinRatio'],
        'cat_columns': ['gender'],
        'num_columns': ['age', 'totalBilirubin', 'directBilirubin','alkalinePhosphatase', 
        'alanineAminotransferase','aspartateAminotransferase', 'totalProteins', 'albumin',
        'albuminGlobulinRatio'],
        'target': 'Outcome',
        'classes': [0, 1]


    },
    'covid': {
        'columns': ['age_quantile', 'Outcome', 'hematocrit', 'hemoglobin', 'platelets',
       'mean_platelet_volume', 'red_blood_cells', 'lymphocytes', 'mchc',
       'leukocytes', 'basophils', 'mch', 'eosinophils', 'mcv', 'monocytes',
       'rdw', 'respiratory_syncytial_virus', 'influenza_a', 'influenza_b',
       'parainfluenza_1', 'coronavirus_nl63', 'rhinovirus_enterovirus',
       'coronavirus_hku1', 'parainfluenza_3', 'chlamydophila_pneumoniae',
       'adenovirus', 'parainfluenza_4', 'coronavirus_229e', 'coronavirus_oc43',
       'influenza_a_h1n1_2009', 'bordetella_pertussis', 'metapneumovirus',
       'parainfluenza_2', 'neutrophils', 'urea', 'c_reactive_protein_mg_dl',
       'creatinine', 'potassium', 'sodium', 'strepto_a'],
        'x_columns': ['age_quantile', 'hematocrit', 'hemoglobin', 'platelets',
       'mean_platelet_volume', 'red_blood_cells', 'lymphocytes', 'mchc',
       'leukocytes', 'basophils', 'mch', 'eosinophils', 'mcv', 'monocytes',
       'rdw', 'respiratory_syncytial_virus', 'influenza_a', 'influenza_b',
       'parainfluenza_1', 'coronavirus_nl63', 'rhinovirus_enterovirus',
       'coronavirus_hku1', 'parainfluenza_3', 'chlamydophila_pneumoniae',
       'adenovirus', 'parainfluenza_4', 'coronavirus_229e', 'coronavirus_oc43',
       'influenza_a_h1n1_2009', 'bordetella_pertussis', 'metapneumovirus',
       'parainfluenza_2', 'neutrophils', 'urea', 'c_reactive_protein_mg_dl',
       'creatinine', 'potassium', 'sodium', 'strepto_a'],
        'cat_columns': ['respiratory_syncytial_virus', 'influenza_a',
       'influenza_b', 'parainfluenza_1', 'coronavirus_nl63',
       'rhinovirus_enterovirus', 'coronavirus_hku1', 'parainfluenza_3',
       'chlamydophila_pneumoniae', 'adenovirus', 'parainfluenza_4',
       'coronavirus_229e', 'coronavirus_oc43', 'influenza_a_h1n1_2009',
       'bordetella_pertussis', 'metapneumovirus', 'parainfluenza_2', 'strepto_a'],
        'num_columns': ['age_quantile', 'hematocrit', 'hemoglobin',
       'platelets', 'mean_platelet_volume', 'red_blood_cells', 'lymphocytes',
       'mchc', 'leukocytes', 'basophils', 'mch', 'eosinophils', 'mcv',
       'monocytes', 'rdw', 'neutrophils', 'urea', 'c_reactive_protein_mg_dl', 'creatinine',
       'potassium', 'sodium'],
        'target': 'Outcome',
        'classes': [0, 1]
    }
}

# ABBREVIATIONS
abbreviations: Dict[str, str] = {
    'blackbox': 'bb',
    'explainer': 'ex',
    'enhancer': 'en',
    'decision_system': 'ds',
    'random_forest': 'rf',
    'ebm': 'ebm',
    'catboost': 'cb',
    'xgboost': 'xgb',
    'huggingface': 'hf',
    'medrag': 'md'
}

pastel_palette = sns.color_palette("pastel")
enhancer_color_dict = {"mixtral RAG": pastel_palette[0],
              "mixtral LLM": pastel_palette[1],
              "llama2 RAG": pastel_palette[2],
              "llama2 LLM": pastel_palette[3],
              "meditron LLM": pastel_palette[5]}

output_color_dict = {"agree": pastel_palette[2],
                     "disagree_model": pastel_palette[5],
                     "disagree_enhancer": pastel_palette[3],
                     "abstain": pastel_palette[4],
                     "unknown": pastel_palette[7]
}


selector_explanation_color_dict = {"both": pastel_palette[2],
                     "explainer": pastel_palette[5],
                     "enhancer": pastel_palette[3],
                     "abstain": pastel_palette[4],
                     "unknown": pastel_palette[7]
}

selector_sensitivity_metric_color_dict = {
    "selector_f1-score_macro": pastel_palette[2],
    "selector_f1-score_1": pastel_palette[3],
    "coverage": pastel_palette[4],
}

selector_sensitivity_metric_label_dict = {
    "selector_f1-score_macro": "f1-score macro",
    "selector_f1-score_1": "f1-score class 1",
    "coverage": "coverage",
}

dataset_color_dict = {
    'pima': pastel_palette[2],
    'diabetes': pastel_palette[5],
    'stroke': pastel_palette[3],
    'liver': pastel_palette[4],
    'covid': pastel_palette[7]
}
