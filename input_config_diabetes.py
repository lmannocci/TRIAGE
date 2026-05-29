
dataset_prefix = 'diabetes'

model_name = 'random_forest' # ebm, # catboost , 'xgboost', 'random_forest'
model_prefix = 'balanced'

# 'random_forest': {
#         'n_estimators': [50, 100, 200, 500],  # Number of trees in the forest
#         'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
#         'min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
#         'min_samples_leaf': [1, 2, 4],  # Minimum samples required at each leaf node
#         'max_features': ['sqrt', 'log2', None],  # Number of features to consider when looking for the best split
#         'bootstrap': [True, False],  # Whether bootstrap samples are used when building trees
#         'criterion': ['gini', 'entropy'],  # Function to measure the quality of a split
#     },
    # 'ebm': {
    #     'max_bins': [128, 256],
    #     'interactions': [0, 10, 20],
    #     'learning_rate': [0.01, 0.05],
    #     'max_leaves': [3, 5]
    # },

param_grid = {
    'random_forest': {
        # 'n_estimators': [100, 200],
        # 'max_depth': [10, 20],
        # 'min_samples_split': [2, 5],
        # 'min_samples_leaf': [2, 4],
        # 'max_features': ['sqrt'],
        # 'bootstrap': [True],
        # 'criterion': ['gini'],
        # 'class_weight': [None, 'balanced']
        'n_estimators': [200],
        'max_depth': [10],
        'min_samples_split': [2],
        'min_samples_leaf': [4],
        'max_features': ['sqrt'],
        'bootstrap': [True],
        'criterion': ['gini'],
        'class_weight': [None]
    },
   'ebm': {
        # 'max_bins': [64, 128, 256],             # Try a lower bin number (64) for more smoothing
        # 'interactions': [0, 5, 10, 15],         # Smaller steps, including no interaction
        # 'learning_rate': [0.005, 0.01, 0.05],   # Add smaller learning rate to improve generalization
        # 'max_leaves': [3, 4, 5, 6],              # Slightly more granularity in tree size
        # 'max_rounds': [100, 200, 300],           # Number of boosting rounds; adjust if your library allows

        'max_bins': [64],
        'interactions': [15],
        'learning_rate': [0.01],
        'max_leaves': [3],
        'max_rounds': [200],
    },
    'catboost': {
        # 'iterations': [100, 200],
        # 'learning_rate': [0.01, 0.05],
        # 'depth': [3, 4, 5],
        # 'l2_leaf_reg': [5, 10]
        'iterations': [200],
        'learning_rate': [0.05],
        'depth': [5],
        'l2_leaf_reg': [5]
    },
    'xgboost': {
        # 'n_estimators': [100, 200, 300],
        # 'max_depth': [3, 5, 7],
        # 'learning_rate': [0.01, 0.03, 0.05],
        # 'subsample': [0.7, 0.8],
        # 'colsample_bytree': [0.7, 0.8],
        # 'gamma': [0, 0.1, 0.3],
        # 'reg_alpha': [0.1, 0.5, 1],
        # 'reg_lambda': [1, 1.5, 2]
        'n_estimators': [300],
        'max_depth': [7],
        'learning_rate': [0.03],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'gamma': [0],
        'reg_alpha': [0.1],
        'reg_lambda': [2]
    }
}

explainer_name = 'dalex'

# ENHANCER
# huggingface, medrag
enhancer_name = 'medrag'
# mixtral, meditron, llama2
llm_name = "llama2"
rag = True # If True, use RAG approach; if False, use standard LLM enhancement
retriever_name="MedCPT"
corpus_name="StatPearls"
return_options = True
rag_K_documents = 10

# EVALUATOR
top_k = 3
metric_exp_ranking = 'rbo'
ds_name = 'simple'
avg_text_y_shift_map = {
    "mixtral RAG": -0.002,
    "mixtral LLM": 0,
    "llama2 RAG": 0,
    "llama2 LLM": -0.018,
    "meditron LLM": -0.015,
}

avg_text_x_shift_map ={
    "mixtral RAG": 0,
    "mixtral LLM": 0,
    "llama2 RAG": 0,
    "llama2 LLM": 0,
    "meditron LLM": -3,
}

# SYNTHESIZER
synthesizer_name = "ctgan"
epochs = 2000
n_samples_synthesizer = 20000

# CONFIDENCE ENHANCER
n_neighbors = 10
metric_neighbors = 'cosine_similarity'
aggregation_method = 'agreement' # 'average', 'agreement'