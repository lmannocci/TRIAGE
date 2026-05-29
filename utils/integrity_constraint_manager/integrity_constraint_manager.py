from utils.common_variables import *
from utils.log_manager.log_manager import *
from utils.decorator_definition import *
import inspect
from sdv.single_table import CTGANSynthesizer
from typing import Dict, List, Union

class IntegrityConstraintManager:
    def __init__(self, lm: LogManager) -> None:
        self.lm = lm

    
    def check_dataset(self, dataset_prefix: str) -> None:
        if dataset_prefix not in av_datasets:
            self.lm.printl(f"dataset_name: {dataset_prefix} not in datasets: {av_datasets}")
            raise ValueError(f"dataset_name: {dataset_prefix} not in datasets: {av_datasets}")

    
    def check_model(self, model_name: str) -> None:
        if model_name not in av_models:
            self.lm.printl(f"model_name: {model_name} not in models: {av_models}")
            raise ValueError(f"model_name: {model_name} not in models: {av_models}")

    
    def check_explainer(self, explainer_name: str) -> None:
        if explainer_name not in av_explainers:
            self.lm.printl(f"explainer_name: {explainer_name} not in explainers: {av_explainers}")
            raise ValueError(f"explainer_name: {explainer_name} not in explainers: {av_explainers}")

    
    def check_model_explainer(self, model_name: str, explainer_name: str) -> None:
        if explainer_name not in av_models_explainer.get(model_name, []):
            self.lm.printl(f"explainer_name: {explainer_name} is not available for model: {model_name} explainers: {av_models_explainer.get(model_name, [])}")
            raise ValueError(f"explainer_name: {explainer_name} is not available for model: {model_name} explainers: {av_models_explainer.get(model_name, [])}")

    
    def check_enhancer(self, enhancer_name: str) -> None:
        if enhancer_name not in av_enhancers:
            self.lm.printl(f"enhancer_name: {enhancer_name} not in enhancers: {av_enhancers}")
            raise ValueError(f"enhancer_name: {enhancer_name} not in enhancers: {av_enhancers}")
        
    
    def check_llm(self, llm_name: str) -> None:
        if llm_name not in av_llms:
            self.lm.printl(f"llm_name: {llm_name} not in llms: {av_llms}")
            raise ValueError(f"llm_name: {llm_name} not in llms: {av_llms}")
    
    
    def check_rag(self, rag: bool) -> None:
        if not isinstance(rag, bool):
            self.lm.printl(f"rag: {rag} must be a boolean value")
            raise ValueError(f"rag: {rag} must be a boolean value")

    
    def check_enhancer_rag_combination(self, enhancer_name: str, rag: bool) -> None:
        if rag not in av_enhancers_rag.get(enhancer_name, []):
            self.lm.printl(f"RAG setting: {rag} is not valid for enhancer: {enhancer_name}. Valid RAG settings for this enhancer are: {av_enhancers_rag.get(enhancer_name, [])}")
            raise ValueError(f"RAG setting: {rag} is not valid for enhancer: {enhancer_name}. Valid RAG settings for this enhancer are: {av_enhancers_rag.get(enhancer_name, [])}")

    
    def check_retriever(self, retriever_name: str) -> None:
        if retriever_name not in av_retrievers:
            self.lm.printl(f"retriever_name: {retriever_name} not in retrievers: {av_retrievers}")
            raise ValueError(f"retriever_name: {retriever_name} not in retrievers: {av_retrievers}")
    
    
    def check_corpus(self, corpus_name: str) -> None:
        if corpus_name not in av_corpus:
            self.lm.printl(f"corpus_name: {corpus_name} not in corpus: {av_corpus}")
            raise ValueError(f"corpus_name: {corpus_name} not in corpus: {av_corpus}")
    
    
    def check_selector_name(self, selector_name: str) -> None:
        if selector_name not in av_selectors:
            self.lm.printl(f"selector_name: {selector_name} not in selectors: {av_selectors}")
            raise ValueError(f"selector_name: {selector_name} not in selectors: {av_selectors}")

    
    def check_top_k(self, top_k: int) -> None:
        if top_k <= 0:
            self.lm.printl(f"top_k: {top_k} must be greater than 0")
            raise ValueError(f"top_k: {top_k} must be greater than 0")
        
    
    def check_metric_exp_ranking(self, metric_exp_ranking: str) -> None:
        if metric_exp_ranking not in av_metrics_exp_ranking:
            self.lm.printl(f"metric_exp_ranking: {metric_exp_ranking} not in metrics: {av_metrics_exp_ranking}")
            raise ValueError(f"metric_exp_ranking: {metric_exp_ranking} not in metrics: {av_metrics_exp_ranking}")
        
    
    def check_synthesizer(self, synthesizer_name: str) -> None:
        if synthesizer_name not in av_synthesizers:
            self.lm.printl(f"synthesizer_name: {synthesizer_name} not in synthesizers: {av_synthesizers}")
            raise ValueError(f"synthesizer_name: {synthesizer_name} not in synthesizers: {av_synthesizers}")
        
    
    def check_synthesizer_epochs(self, epochs: int) -> None:
        if epochs <= 0:
            self.lm.printl(f"epochs: {epochs} must be greater than 0")
            raise ValueError(f"epochs: {epochs} must be greater than 0")

    def check_n_neighbors(self, n_neighbors: int) -> None:
        if n_neighbors <= 0:
            self.lm.printl(f"n_neighbors: {n_neighbors} must be greater than 0")
            raise ValueError(f"n_neighbors: {n_neighbors} must be greater than 0")
    
    
    def check_metric_neighbors(self, metric: str) -> None:
        if metric not in av_metrics_neighbors:
            self.lm.printl(f"metric: {metric} not in metrics: {av_metrics_neighbors}")
            raise ValueError(f"metric: {metric} not in metrics: {av_metrics_neighbors}")
        
    
    def check_aggregation_method(self, aggregation_method: str) -> None:
        if aggregation_method not in av_aggregation_methods:
            self.lm.printl(f"aggregation_method: {aggregation_method} not in valid methods: {av_aggregation_methods}")
            raise ValueError(f"aggregation_method: {aggregation_method} not in valid methods: {av_aggregation_methods}")