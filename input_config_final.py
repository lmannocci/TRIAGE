
selector_name = "abstain" # "simple" or "abstain"

model_name = "catboost"
model_prefix = "balanced"

explainer_name = "shap"
top_k = 3
metric_exp_ranking = 'rbo'

enhancer_name = "medrag"
llm_name = "llama2"
rag = True
retriever_name = "MedCPT"
corpus_name = "StatPearls"
return_options = True
rag_K_documents = 10

synthesizer_name = "ctgan"


n_neighbors = 10
metric_neighbors = "cosine_similarity"
aggregation_method = "agreement"

metric_neighbors = "cosine_similarity"