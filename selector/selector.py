import os
from typing import Optional

from utils.checkpoint.checkpoint import *
from utils.directory_manager.directory_manager import DirectoryManager
from utils.log_manager.log_manager import *
from utils.decorator_definition import *
from utils.common_variables import *
from utils.integrity_constraint_manager.integrity_constraint_manager import IntegrityConstraintManager

from selector.core.context import SelectorContext
from selector.core.decision_engine import SelectorDecisionEngine
from selector.core.analysis_service import SelectorAnalysisService

absolute_path = os.path.dirname(__file__)
results_path = os.path.join(absolute_path, f"..{os.sep}results{os.sep}")
data_path = os.path.join(absolute_path, f"..{os.sep}data{os.sep}")

class Selector:
    def __init__(
        self,
        ch: Checkpoint,
        lm: LogManager,
        dataset_prefix: str,
        selector_name: str,
        model_name: str,
        model_prefix: str,
        explainer_name: str,
        top_k: int,
        metric_exp_ranking: str,
        enhancer_name: str,
        llm_name: str,
        rag: bool,
        retriever_name: str,
        corpus_name: str,
        return_options: bool,
        rag_K_documents: int,
        synthesizer_name: str,
        epochs: int,
        n_samples_synthesizer: int,
        n_neighbors: int,
        metric_neighbors: str,
        aggregation_method: str,
        th_model_conf: float = 0.7,
        th_enhancer_conf: float = 0.7,
        th_rbo: float = 0.5,
        cuda_visible_devices: Optional[str] = "1,2,3",
        parallel: Optional[bool] = True,
    ):
        self.ch = ch
        self.lm = lm
        self.icm = IntegrityConstraintManager(lm)

        self._validate_inputs(
            dataset_prefix, model_name, explainer_name, top_k, metric_exp_ranking,
            enhancer_name, llm_name, rag, retriever_name, corpus_name,
            synthesizer_name, n_neighbors, metric_neighbors, aggregation_method, epochs
        )

        self.dm = DirectoryManager(
            lm, results_path, dataset_prefix=dataset_prefix,
            model_name=model_name, model_prefix=model_prefix,
            explainer_name=explainer_name, top_k=top_k, metric_exp_ranking=metric_exp_ranking,
            enhancer_name=enhancer_name, llm_name=llm_name, rag=rag,
            retriever_name=retriever_name, corpus_name=corpus_name,
            synthesizer_name=synthesizer_name, epochs=epochs,
            n_samples_synthesizer=n_samples_synthesizer,
            n_neighbors=n_neighbors, metric_neighbors=metric_neighbors,
            aggregation_method=aggregation_method
        )

        self.ctx = SelectorContext(
            dataset_prefix=dataset_prefix,
            info=df_info[dataset_prefix],
            selector_name=selector_name,
            model_name=model_name,
            model_prefix=model_prefix,
            explainer_name=explainer_name,
            top_k=top_k,
            metric_exp_ranking=metric_exp_ranking,
            metric_complete_name=f"{metric_exp_ranking}_{top_k}",
            enhancer_name=enhancer_name,
            llm_name=llm_name,
            rag=rag,
            retriever_name=retriever_name,
            corpus_name=corpus_name,
            return_options=return_options,
            rag_K_documents=rag_K_documents,
            synthesizer_name=synthesizer_name,
            n_neighbors=n_neighbors,
            metric_neighbors=metric_neighbors,
            aggregation_method=aggregation_method,
            th_model_conf=th_model_conf,
            th_enhancer_conf=th_enhancer_conf,
            th_rbo=th_rbo,
        )

        self.parallel = parallel
        self.cuda_visible_devices = cuda_visible_devices

        self._decision_engine = SelectorDecisionEngine(self.ctx)
        self._analysis = SelectorAnalysisService(ch, lm, self.ctx, self.dm)

    def _validate_inputs(
        self,
        dataset_prefix, model_name, explainer_name, top_k, metric_exp_ranking,
        enhancer_name, llm_name, rag, retriever_name, corpus_name,
        synthesizer_name, n_neighbors, metric_neighbors, aggregation_method, epochs
    ):
        self.icm.check_dataset(dataset_prefix)
        self.icm.check_model(model_name)
        self.icm.check_explainer(explainer_name)
        self.icm.check_model_explainer(model_name, explainer_name)
        self.icm.check_top_k(top_k)
        self.icm.check_metric_exp_ranking(metric_exp_ranking)
        self.icm.check_enhancer(enhancer_name)
        self.icm.check_llm(llm_name)
        self.icm.check_rag(rag)
        self.icm.check_retriever(retriever_name)
        self.icm.check_corpus(corpus_name)
        self.icm.check_synthesizer(synthesizer_name)
        self.icm.check_synthesizer_epochs(epochs)
        self.icm.check_n_neighbors(n_neighbors)
        self.icm.check_metric_neighbors(metric_neighbors)
        self.icm.check_aggregation_method(aggregation_method)

    def _selector_df_path(self) -> str:
        return (
            f"{self.dm.en_selector_path}"
            f"{self.ctx.model_name}_{self.ctx.explainer_name}_"
            f"{self.ctx.metric_complete_name}_{self.ctx.selector_name}_selector_df.csv"
        )

    @log_method
    def apply_selector(self):
        agreement_df = self.ch.read_dataframe(f"{self.dm.ev_agreement_csv_path}{self.ctx.model_name}_{self.ctx.enhancer_name}_{self.ctx.llm_name}_{str(self.ctx.rag)}_{self.ctx.retriever_name}_{self.ctx.corpus_name}_agreement_df.csv", dtype=dtype)
        evaluator_df = self.ch.read_dataframe(f"{self.dm.ev_explanations_csv_path}{self.ctx.model_name}_{self.ctx.explainer_name}_evaluator_df.csv", dtype=dtype)
        confidence_df = self.ch.read_dataframe(f"{self.dm.ce_enhancer_path}confidence_scores_{self.ctx.aggregation_method}.csv", dtype=dtype)

        df = agreement_df.merge(evaluator_df, on="original_index", how="inner").merge(confidence_df, on="original_index", how="inner")

        if self.ctx.selector_name == "simple":
            df = df.apply(self._decision_engine.apply_simple, axis=1)
        elif self.ctx.selector_name == "abstain":
            df = df.apply(self._decision_engine.apply_abstain, axis=1)

        columns = [
            ind,
            self.ctx.target_col,
            self.ctx.model_pred_col,
            self.ctx.model_conf_col,
            self.ctx.enhancer_pred_col,
            self.ctx.enhancer_conf_col,
            self.ctx.metric_complete_name,
            self.ctx.selector_decision_col,
            self.ctx.selector_prediction_col,
            self.ctx.selector_use_enhanced_expl_col,
            self.ctx.selector_use_enhancer_expl_col,
            self.ctx.selector_complete_case_col,
            self.ctx.selector_case_col,
            self.ctx.selector_is_abstain_col,
            self.ctx.selector_is_non_abstain_col,
            self.ctx.selector_final_correct_col,
        ]

        selector_df = df[[c for c in columns if c in df.columns]].copy()
        self.ch.save_dataframe(selector_df, self._selector_df_path())

    @log_method
    def analyze_selector(self):
        self._analysis.analyze_selector()

    @log_method
    def plot_selector_case_scatter(self):
        self._analysis.plot_selector_case_scatter()

    @log_method
    def save_selector_classification_report(self):
        self._analysis.save_selector_classification_report()

    @log_method
    def plot_selector_vs_bb_accepted_subset_dumbbell_per_dataset(self):
        self._analysis.plot_selector_vs_bb_accepted_subset_dumbbell_per_dataset()

    @log_method
    def plot_selector_vs_bb_matched_coverage_dumbbell_per_dataset(self):
        self._analysis.plot_selector_vs_bb_matched_coverage_dumbbell_per_dataset()
