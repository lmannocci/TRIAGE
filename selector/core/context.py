from dataclasses import dataclass
from typing import Dict, List, Union, Optional, Any

@dataclass
class SelectorContext:
    dataset_prefix: str
    info: Dict[str, Union[str, List[str]]]
    selector_name: str
    model_name: str
    model_prefix: str
    explainer_name: str
    top_k: int
    metric_exp_ranking: str
    metric_complete_name: str
    enhancer_name: str
    llm_name: str
    rag: bool
    retriever_name: str
    corpus_name: str
    return_options: bool
    rag_K_documents: int
    synthesizer_name: str
    n_neighbors: int
    metric_neighbors: str
    aggregation_method: str
    th_model_conf: float
    th_enhancer_conf: float
    th_rbo: float

    @property
    def target_col(self) -> str:
        return self.info["target"]

    @property
    def model_pred_col(self) -> str:
        return f"{self.model_name}_pred"

    @property
    def model_conf_col(self) -> str:
        return f"{self.model_name}_conf"

    @property
    def enhancer_pred_col(self) -> str:
        return f"{self.enhancer_name}_pred"

    @property
    def enhancer_conf_col(self) -> str:
        return f"{self.enhancer_name}_conf"

    @property
    def selector_decision_col(self) -> str:
        return f"{self.selector_name}_decision"

    @property
    def selector_prediction_col(self) -> str:
        return f"{self.selector_name}_prediction"

    @property
    def selector_use_enhanced_expl_col(self) -> str:
        return f"{self.selector_name}_use_enhanced_explanation"

    @property
    def selector_use_enhancer_expl_col(self) -> str:
        return f"{self.selector_name}_use_enhancer_explanation"

    @property
    def selector_complete_case_col(self) -> str:
        return f"{self.selector_name}_complete_logging_case"

    @property
    def selector_case_col(self) -> str:
        return f"{self.selector_name}_logging_case"

    @property
    def selector_is_abstain_col(self) -> str:
        return f"{self.selector_name}_is_abstain"

    @property
    def selector_is_non_abstain_col(self) -> str:
        return f"{self.selector_name}_is_non_abstain"

    @property
    def selector_final_correct_col(self) -> str:
        return f"{self.selector_name}_final_correct"
