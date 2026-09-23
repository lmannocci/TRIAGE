import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from selector.selector import Selector
from utils.checkpoint.checkpoint import Checkpoint
from utils.common_variables import dataset_test_sizes_dict, df_info, dtype, ind
from utils.decorator_definition import log_method
from utils.directory_manager.directory_manager import DirectoryManager
from utils.integrity_constraint_manager.integrity_constraint_manager import IntegrityConstraintManager
from utils.log_manager.log_manager import LogManager


absolute_path = os.path.dirname(__file__)
results_path = os.path.join(absolute_path, f"..{os.sep}results{os.sep}")


class ClassificationBootstrapCI:
    """
    Stratified bootstrap confidence intervals for test classification metrics.

    The class supports black-box, enhancer, and selector prediction files. The
    selector bootstrap also estimates uncertainty for abstention and coverage.
    """

    METRIC_COLUMNS = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1-score_macro",
        "precision_weighted",
        "recall_weighted",
        "f1-score_weighted",
        "precision_0",
        "recall_0",
        "f1-score_0",
        "precision_1",
        "recall_1",
        "f1-score_1",
    ]

    SELECTOR_EXTRA_METRIC_COLUMNS = [
        "coverage",
        "abstain_rate",
        "abstain_rate_0",
        "abstain_rate_1",
    ]

    def __init__(
        self,
        ch: Checkpoint,
        lm: LogManager,
        dataset_prefix: str,
        n_bootstrap: int = 2000,
        random_state: int = 42,
    ):
        self.ch = ch
        self.lm = lm
        self.icm = IntegrityConstraintManager(lm)
        self.icm.check_dataset(dataset_prefix)

        self.dataset_prefix = dataset_prefix
        self.info = df_info[dataset_prefix]
        self.n_bootstrap = int(n_bootstrap)
        self.random_state = int(random_state)
        self.dm = DirectoryManager(lm, results_path, dataset_prefix=dataset_prefix)

    @log_method
    def run_blackbox(
        self,
        model_name: str,
        model_prefix: str,
        save_global_evaluator: bool = False,
    ) -> pd.DataFrame:
        self.icm.check_model(model_name)
        dm = DirectoryManager(
            self.lm,
            results_path,
            dataset_prefix=self.dataset_prefix,
            model_name=model_name,
            model_prefix=model_prefix,
        )
        df = self.ch.read_dataframe(f"{dm.model_path}{model_name}_predicted.csv", dtype=dtype)
        target_col = self.info["target"]
        pred_col = f"{model_name}_pred"

        row = self._build_ci_row(
            df=df,
            target_col=target_col,
            pred_col=pred_col,
            system_type="blackbox",
            model_abbr=model_name,
            model_name=model_name,
            model_prefix=model_prefix,
            enhancer_name=None,
            llm_name=None,
            rag=None,
            retriever_name=None,
            corpus_name=None,
            error_rate=np.nan,
            errors=np.nan,
        )
        out_df = pd.DataFrame([row])
        self._save_report(out_df, save_global_evaluator)
        return out_df

    @log_method
    def run_selector(
        self,
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
        th_rbo: float = 0.7,
        save_global_evaluator: bool = False,
    ) -> pd.DataFrame:
        selector = Selector(
            ch=self.ch,
            lm=self.lm,
            dataset_prefix=self.dataset_prefix,
            selector_name=selector_name,
            model_name=model_name,
            model_prefix=model_prefix,
            explainer_name=explainer_name,
            top_k=top_k,
            metric_exp_ranking=metric_exp_ranking,
            enhancer_name=enhancer_name,
            llm_name=llm_name,
            rag=rag,
            retriever_name=retriever_name,
            corpus_name=corpus_name,
            return_options=return_options,
            rag_K_documents=rag_K_documents,
            synthesizer_name=synthesizer_name,
            epochs=epochs,
            n_samples_synthesizer=n_samples_synthesizer,
            n_neighbors=n_neighbors,
            metric_neighbors=metric_neighbors,
            aggregation_method=aggregation_method,
            th_model_conf=th_model_conf,
            th_enhancer_conf=th_enhancer_conf,
            th_rbo=th_rbo,
        )
        df = self.ch.read_dataframe(selector._selector_df_path(), dtype=dtype)
        row = self._build_selector_ci_row(
            df=df,
            selector_name=selector_name,
            model_name=model_name,
            model_prefix=model_prefix,
            explainer_name=explainer_name,
            top_k=top_k,
            metric_exp_ranking=metric_exp_ranking,
            enhancer_name=enhancer_name,
            llm_name=llm_name,
            rag=rag,
            retriever_name=retriever_name,
            corpus_name=corpus_name,
            synthesizer_name=synthesizer_name,
            epochs=epochs,
            n_samples_synthesizer=n_samples_synthesizer,
            n_neighbors=n_neighbors,
            metric_neighbors=metric_neighbors,
            aggregation_method=aggregation_method,
            th_model_conf=th_model_conf,
            th_enhancer_conf=th_enhancer_conf,
            th_rbo=th_rbo,
        )
        out_df = pd.DataFrame([row])
        self._save_report(out_df, save_global_evaluator)
        return out_df

    @log_method
    def run_selector_blackbox_baselines(
        self,
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
        th_rbo: float = 0.7,
        save_global_evaluator: bool = False,
    ) -> pd.DataFrame:
        """
        Bootstrap the two blackbox baselines used by selector reporting.

        The produced rows correspond to:
        - `accepted_subset`: blackbox performance on rows accepted by selector.
        - `matched_coverage`: blackbox performance on its highest-confidence
          rows, with the same number of rows accepted by the selector.
        """
        selector = Selector(
            ch=self.ch,
            lm=self.lm,
            dataset_prefix=self.dataset_prefix,
            selector_name=selector_name,
            model_name=model_name,
            model_prefix=model_prefix,
            explainer_name=explainer_name,
            top_k=top_k,
            metric_exp_ranking=metric_exp_ranking,
            enhancer_name=enhancer_name,
            llm_name=llm_name,
            rag=rag,
            retriever_name=retriever_name,
            corpus_name=corpus_name,
            return_options=return_options,
            rag_K_documents=rag_K_documents,
            synthesizer_name=synthesizer_name,
            epochs=epochs,
            n_samples_synthesizer=n_samples_synthesizer,
            n_neighbors=n_neighbors,
            metric_neighbors=metric_neighbors,
            aggregation_method=aggregation_method,
            th_model_conf=th_model_conf,
            th_enhancer_conf=th_enhancer_conf,
            th_rbo=th_rbo,
        )
        df_selector = self.ch.read_dataframe(selector._selector_df_path(), dtype=dtype)
        df_bb = self.ch.read_dataframe(f"{selector.dm.model_path}{model_name}_predicted.csv", dtype=dtype)

        abstain_col = f"{selector_name}_is_abstain"
        n_total = len(df_selector)
        n_abstain = int(self._as_bool(df_selector[abstain_col]).sum())
        n_non_abstain = n_total - n_abstain
        coverage = n_non_abstain / n_total if n_total > 0 else np.nan

        accepted_indices = df_selector.loc[~self._as_bool(df_selector[abstain_col]), ind].tolist()
        accepted_df = df_bb[df_bb[ind].isin(accepted_indices)].copy()
        matched_df = self._matched_coverage_blackbox_subset(
            df_selector=df_selector,
            df_bb=df_bb,
            model_conf_col=f"{model_name}_conf",
            n_non_abstain=n_non_abstain,
        )

        rows = [
            self._build_selector_blackbox_baseline_ci_row(
                df=accepted_df,
                baseline_type="accepted_subset",
                selector_name=selector_name,
                model_name=model_name,
                model_prefix=model_prefix,
                explainer_name=explainer_name,
                top_k=top_k,
                metric_exp_ranking=metric_exp_ranking,
                enhancer_name=enhancer_name,
                llm_name=llm_name,
                rag=rag,
                retriever_name=retriever_name,
                corpus_name=corpus_name,
                synthesizer_name=synthesizer_name,
                epochs=epochs,
                n_samples_synthesizer=n_samples_synthesizer,
                n_neighbors=n_neighbors,
                metric_neighbors=metric_neighbors,
                aggregation_method=aggregation_method,
                th_model_conf=th_model_conf,
                th_enhancer_conf=th_enhancer_conf,
                th_rbo=th_rbo,
                n_total=n_total,
                n_abstain=n_abstain,
                n_non_abstain=n_non_abstain,
                coverage=coverage,
            ),
            self._build_selector_blackbox_baseline_ci_row(
                df=matched_df,
                baseline_type="matched_coverage",
                selector_name=selector_name,
                model_name=model_name,
                model_prefix=model_prefix,
                explainer_name=explainer_name,
                top_k=top_k,
                metric_exp_ranking=metric_exp_ranking,
                enhancer_name=enhancer_name,
                llm_name=llm_name,
                rag=rag,
                retriever_name=retriever_name,
                corpus_name=corpus_name,
                synthesizer_name=synthesizer_name,
                epochs=epochs,
                n_samples_synthesizer=n_samples_synthesizer,
                n_neighbors=n_neighbors,
                metric_neighbors=metric_neighbors,
                aggregation_method=aggregation_method,
                th_model_conf=th_model_conf,
                th_enhancer_conf=th_enhancer_conf,
                th_rbo=th_rbo,
                n_total=n_total,
                n_abstain=n_abstain,
                n_non_abstain=n_non_abstain,
                coverage=coverage,
            ),
        ]
        out_df = pd.DataFrame(rows)
        self._save_selector_blackbox_baseline_report(out_df, save_global_evaluator)
        return out_df

    @log_method
    def run_enhancer(
        self,
        enhancer_name: str,
        llm_name: str,
        rag: bool,
        retriever_name: str = "MedCPT",
        corpus_name: str = "StatPearls",
        save_global_evaluator: bool = False,
    ) -> pd.DataFrame:
        self.icm.check_enhancer(enhancer_name)
        self.icm.check_llm(llm_name)
        self.icm.check_rag(rag)
        self.icm.check_retriever(retriever_name)
        self.icm.check_corpus(corpus_name)

        dm = DirectoryManager(
            self.lm,
            results_path,
            dataset_prefix=self.dataset_prefix,
            enhancer_name=enhancer_name,
            llm_name=llm_name,
            rag=rag,
            retriever_name=retriever_name,
            corpus_name=corpus_name,
        )
        df = self.ch.read_dataframe(f"{dm.enhancer_path}{enhancer_name}_cleaned_df.csv", dtype=dtype)
        removed_error_df = self.ch.read_dataframe(f"{dm.enhancer_path}{enhancer_name}_removed_error_idx.csv", dtype=dtype)

        target_col = self.info["target"]
        pred_col = f"{enhancer_name}_pred"
        n_test_size = dataset_test_sizes_dict[self.dataset_prefix]
        errors = len(removed_error_df)
        error_rate = (errors / n_test_size) if n_test_size > 0 else np.nan
        enhancer_label = f"{llm_name} RAG" if rag else f"{llm_name} LLM"

        row = self._build_ci_row(
            df=df,
            target_col=target_col,
            pred_col=pred_col,
            system_type="enhancer",
            model_abbr=enhancer_label,
            model_name=None,
            model_prefix=None,
            enhancer_name=enhancer_name,
            llm_name=llm_name,
            rag=rag,
            retriever_name=retriever_name,
            corpus_name=corpus_name,
            error_rate=error_rate,
            errors=errors,
        )
        out_df = pd.DataFrame([row])
        self._save_report(out_df, save_global_evaluator)
        return out_df

    def _build_ci_row(
        self,
        df: pd.DataFrame,
        target_col: str,
        pred_col: str,
        system_type: str,
        model_abbr: str,
        model_name: Optional[str],
        model_prefix: Optional[str],
        enhancer_name: Optional[str],
        llm_name: Optional[str],
        rag: Optional[bool],
        retriever_name: Optional[str],
        corpus_name: Optional[str],
        error_rate,
        errors,
    ) -> Dict:
        y_true = pd.to_numeric(df[target_col], errors="coerce")
        y_pred = pd.to_numeric(df[pred_col], errors="coerce")
        valid_mask = y_true.notna() & y_pred.notna()
        y_true = y_true.loc[valid_mask].astype(int).reset_index(drop=True)
        y_pred = y_pred.loc[valid_mask].astype(int).reset_index(drop=True)

        original_metrics = self._build_report_metrics(y_true, y_pred)
        bootstrap_metrics = self._stratified_bootstrap_metrics(y_true, y_pred)

        row = {
            "system_type": system_type,
            "dataset": self.dataset_prefix,
            "model_abbr": model_abbr,
            "model_name": model_name,
            "model_prefix": model_prefix,
            "enhancer_name": enhancer_name,
            "llm_name": llm_name,
            "rag": rag,
            "retriever_name": retriever_name,
            "corpus_name": corpus_name,
            "n_samples": int(len(y_true)),
            "n_bootstrap": self.n_bootstrap,
            "random_state": self.random_state,
            "error_rate": error_rate,
            "errors": errors,
        }

        for metric in self.METRIC_COLUMNS:
            values = pd.to_numeric(bootstrap_metrics[metric], errors="coerce")
            values = values[~np.isnan(values)]
            row[metric] = original_metrics.get(metric)
            row[f"{metric}_ci_lower"] = float(np.percentile(values, 2.5)) if len(values) else np.nan
            row[f"{metric}_ci_upper"] = float(np.percentile(values, 97.5)) if len(values) else np.nan

        return row

    def _build_selector_blackbox_baseline_ci_row(
        self,
        df: pd.DataFrame,
        baseline_type: str,
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
        synthesizer_name: str,
        epochs: int,
        n_samples_synthesizer: int,
        n_neighbors: int,
        metric_neighbors: str,
        aggregation_method: str,
        th_model_conf: float,
        th_enhancer_conf: float,
        th_rbo: float,
        n_total: int,
        n_abstain: int,
        n_non_abstain: int,
        coverage: float,
    ) -> Dict:
        target_col = self.info["target"]
        pred_col = f"{model_name}_pred"
        y_true = pd.to_numeric(df[target_col], errors="coerce")
        y_pred = pd.to_numeric(df[pred_col], errors="coerce")
        valid_mask = y_true.notna() & y_pred.notna()
        y_true = y_true.loc[valid_mask].astype(int).reset_index(drop=True)
        y_pred = y_pred.loc[valid_mask].astype(int).reset_index(drop=True)

        original_metrics = self._build_report_metrics(y_true, y_pred)
        bootstrap_metrics = self._stratified_bootstrap_metrics(y_true, y_pred)

        row = {
            "system_type": "selector_blackbox_baseline",
            "baseline_type": baseline_type,
            "dataset": self.dataset_prefix,
            "model_abbr": f"{model_name}_{baseline_type}",
            "model_name": model_name,
            "model_prefix": model_prefix,
            "enhancer_name": enhancer_name,
            "llm_name": llm_name,
            "rag": rag,
            "retriever_name": retriever_name,
            "corpus_name": corpus_name,
            "explainer_name": explainer_name,
            "selector_name": selector_name,
            "top_k": top_k,
            "metric_exp_ranking": metric_exp_ranking,
            "synthesizer_name": synthesizer_name,
            "epochs": epochs,
            "n_samples_synthesizer": n_samples_synthesizer,
            "n_neighbors": n_neighbors,
            "metric_neighbors": metric_neighbors,
            "aggregation_method": aggregation_method,
            "th_model_conf": th_model_conf,
            "th_enhancer_conf": th_enhancer_conf,
            "th_rbo": th_rbo,
            "n_total": n_total,
            "n_non_abstain": n_non_abstain,
            "n_abstain": n_abstain,
            "coverage": coverage,
            "n_samples": int(len(y_true)),
            "n_bootstrap": self.n_bootstrap,
            "random_state": self.random_state,
        }
        for metric in self.METRIC_COLUMNS:
            values = pd.to_numeric(bootstrap_metrics[metric], errors="coerce")
            values = values[~np.isnan(values)]
            row[metric] = original_metrics.get(metric)
            row[f"{metric}_ci_lower"] = float(np.percentile(values, 2.5)) if len(values) else np.nan
            row[f"{metric}_ci_upper"] = float(np.percentile(values, 97.5)) if len(values) else np.nan
        return row

    def _build_selector_ci_row(
        self,
        df: pd.DataFrame,
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
        synthesizer_name: str,
        epochs: int,
        n_samples_synthesizer: int,
        n_neighbors: int,
        metric_neighbors: str,
        aggregation_method: str,
        th_model_conf: float,
        th_enhancer_conf: float,
        th_rbo: float,
    ) -> Dict:
        target_col = self.info["target"]
        pred_col = f"{selector_name}_prediction"
        abstain_col = f"{selector_name}_is_abstain"

        work_df = df.copy()
        work_df[target_col] = pd.to_numeric(work_df[target_col], errors="coerce")
        work_df[pred_col] = pd.to_numeric(work_df[pred_col], errors="coerce")
        work_df[abstain_col] = self._as_bool(work_df[abstain_col])
        work_df = work_df[work_df[target_col].notna()].copy().reset_index(drop=True)
        work_df[target_col] = work_df[target_col].astype(int)

        original_metrics = self._build_selector_metrics(work_df, target_col, pred_col, abstain_col)
        bootstrap_metrics = self._stratified_bootstrap_selector_metrics(
            work_df=work_df,
            target_col=target_col,
            pred_col=pred_col,
            abstain_col=abstain_col,
        )

        selector_label = (
            f"{selector_name}_{model_name}_{explainer_name}_{enhancer_name}_"
            f"{llm_name}_{str(rag)}_{retriever_name}_{corpus_name}"
        )
        row = {
            "system_type": "selector",
            "dataset": self.dataset_prefix,
            "model_abbr": selector_label,
            "model_name": model_name,
            "model_prefix": model_prefix,
            "enhancer_name": enhancer_name,
            "llm_name": llm_name,
            "rag": rag,
            "retriever_name": retriever_name,
            "corpus_name": corpus_name,
            "explainer_name": explainer_name,
            "selector_name": selector_name,
            "top_k": top_k,
            "metric_exp_ranking": metric_exp_ranking,
            "synthesizer_name": synthesizer_name,
            "epochs": epochs,
            "n_samples_synthesizer": n_samples_synthesizer,
            "n_neighbors": n_neighbors,
            "metric_neighbors": metric_neighbors,
            "aggregation_method": aggregation_method,
            "th_model_conf": th_model_conf,
            "th_enhancer_conf": th_enhancer_conf,
            "th_rbo": th_rbo,
            "n_samples": int(len(work_df)),
            "n_bootstrap": self.n_bootstrap,
            "random_state": self.random_state,
            "error_rate": np.nan,
            "errors": np.nan,
        }

        for metric in self.METRIC_COLUMNS + self.SELECTOR_EXTRA_METRIC_COLUMNS:
            values = pd.to_numeric(bootstrap_metrics[metric], errors="coerce")
            values = values[~np.isnan(values)]
            row[metric] = original_metrics.get(metric)
            row[f"{metric}_ci_lower"] = float(np.percentile(values, 2.5)) if len(values) else np.nan
            row[f"{metric}_ci_upper"] = float(np.percentile(values, 97.5)) if len(values) else np.nan
        return row

    def _stratified_bootstrap_metrics(self, y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
        rng = np.random.default_rng(self.random_state)
        class_indices = {
            class_label: np.flatnonzero(y_true.to_numpy() == class_label)
            for class_label in sorted(y_true.unique())
        }

        rows = []
        for _ in range(self.n_bootstrap):
            sampled_indices = [
                rng.choice(indices, size=len(indices), replace=True)
                for indices in class_indices.values()
                if len(indices) > 0
            ]
            sample_idx = np.concatenate(sampled_indices)
            rows.append(self._build_report_metrics(y_true.iloc[sample_idx], y_pred.iloc[sample_idx]))
        return pd.DataFrame(rows)

    def _stratified_bootstrap_selector_metrics(
        self,
        work_df: pd.DataFrame,
        target_col: str,
        pred_col: str,
        abstain_col: str,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(self.random_state)
        y_true = work_df[target_col].to_numpy()
        class_indices = {
            class_label: np.flatnonzero(y_true == class_label)
            for class_label in sorted(pd.unique(y_true))
        }

        rows = []
        for _ in range(self.n_bootstrap):
            sampled_indices = [
                rng.choice(indices, size=len(indices), replace=True)
                for indices in class_indices.values()
                if len(indices) > 0
            ]
            sample_idx = np.concatenate(sampled_indices)
            rows.append(self._build_selector_metrics(work_df.iloc[sample_idx], target_col, pred_col, abstain_col))
        return pd.DataFrame(rows)

    def _build_selector_metrics(
        self,
        df: pd.DataFrame,
        target_col: str,
        pred_col: str,
        abstain_col: str,
    ) -> Dict:
        n_total = len(df)
        abstain = self._as_bool(df[abstain_col])
        accepted_df = df.loc[~abstain].copy()
        n_abstain = int(abstain.sum())
        n_non_abstain = len(accepted_df)

        y_true_all = pd.to_numeric(df[target_col], errors="coerce")
        mask_0 = y_true_all == 0
        mask_1 = y_true_all == 1
        n_true_0 = int(mask_0.sum())
        n_true_1 = int(mask_1.sum())

        y_true = pd.to_numeric(accepted_df[target_col], errors="coerce")
        y_pred = pd.to_numeric(accepted_df[pred_col], errors="coerce")
        valid_mask = y_true.notna() & y_pred.notna()
        metrics = self._build_report_metrics(
            y_true.loc[valid_mask].astype(int),
            y_pred.loc[valid_mask].astype(int),
        )
        metrics.update({
            "coverage": n_non_abstain / n_total if n_total > 0 else np.nan,
            "abstain_rate": n_abstain / n_total if n_total > 0 else np.nan,
            "abstain_rate_0": float((abstain & mask_0).sum() / n_true_0) if n_true_0 > 0 else np.nan,
            "abstain_rate_1": float((abstain & mask_1).sum() / n_true_1) if n_true_1 > 0 else np.nan,
        })
        return metrics

    def _matched_coverage_blackbox_subset(
        self,
        df_selector: pd.DataFrame,
        df_bb: pd.DataFrame,
        model_conf_col: str,
        n_non_abstain: int,
    ) -> pd.DataFrame:
        selector_indices = df_selector[ind].tolist()
        df_bb_scope = df_bb[df_bb[ind].isin(selector_indices)].copy()
        if len(df_bb_scope) != len(df_selector):
            missing = set(selector_indices) - set(df_bb_scope[ind].tolist())
            raise ValueError(
                f"Cannot build matched-coverage blackbox bootstrap baseline: "
                f"{len(missing)} selector rows are missing from blackbox predictions."
            )
        if model_conf_col not in df_bb_scope.columns:
            raise ValueError(f"Missing blackbox confidence column: {model_conf_col}")

        df_bb_scope["_matched_coverage_original_order"] = np.arange(len(df_bb_scope))
        df_bb_scope[model_conf_col] = pd.to_numeric(df_bb_scope[model_conf_col], errors="coerce")
        df_bb_scope[ind] = pd.to_numeric(df_bb_scope[ind], errors="coerce")
        ranked_df = df_bb_scope.sort_values(
            by=[model_conf_col, ind, "_matched_coverage_original_order"],
            ascending=[False, True, True],
            na_position="last",
            kind="mergesort",
        )
        return ranked_df.head(n_non_abstain).drop(columns=["_matched_coverage_original_order"])

    @staticmethod
    def _safe_class_metrics(report_dict: dict, class_label: int) -> dict:
        possible_keys = [str(class_label), str(float(class_label)), class_label, float(class_label)]
        cls = {}
        for key in possible_keys:
            if key in report_dict:
                cls = report_dict[key]
                break
        return {
            f"precision_{class_label}": cls.get("precision"),
            f"recall_{class_label}": cls.get("recall"),
            f"f1-score_{class_label}": cls.get("f1-score"),
        }

    def _build_report_metrics(self, y_true: pd.Series, y_pred: pd.Series) -> dict:
        if len(y_true) == 0:
            return {metric: np.nan for metric in self.METRIC_COLUMNS}

        report = classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0,
        )
        macro = report.get("macro avg", {})
        weighted = report.get("weighted avg", {})

        row = {
            "accuracy": report.get("accuracy"),
            "precision_macro": macro.get("precision"),
            "recall_macro": macro.get("recall"),
            "f1-score_macro": macro.get("f1-score"),
            "precision_weighted": weighted.get("precision"),
            "recall_weighted": weighted.get("recall"),
            "f1-score_weighted": weighted.get("f1-score"),
        }
        row.update(self._safe_class_metrics(report, 0))
        row.update(self._safe_class_metrics(report, 1))
        return row

    @staticmethod
    def _as_bool(series: pd.Series) -> pd.Series:
        if series.dtype == bool:
            return series
        return series.astype(str).str.lower().isin(["true", "1", "yes"])

    def _save_report(self, new_df: pd.DataFrame, save_global_evaluator: bool) -> None:
        if save_global_evaluator:
            path = f"{self.dm.global_evaluator_path}classification_bootstrap_ci.csv"
        else:
            path = f"{self.dm.ev_report_path}classification_bootstrap_ci.csv"

        if os.path.exists(path):
            report_df = self.ch.read_dataframe(path, dtype=dtype)
            report_df = pd.concat([report_df, new_df], ignore_index=True)
        else:
            report_df = new_df

        bb_key = ["system_type", "dataset", "model_name", "model_prefix", "n_bootstrap", "random_state"]
        enh_key = [
            "system_type",
            "dataset",
            "enhancer_name",
            "llm_name",
            "rag",
            "retriever_name",
            "corpus_name",
            "n_bootstrap",
            "random_state",
        ]

        bb_df = report_df[report_df["system_type"] == "blackbox"].copy()
        enh_df = report_df[report_df["system_type"] == "enhancer"].copy()
        selector_df = report_df[report_df["system_type"] == "selector"].copy()

        if not bb_df.empty:
            bb_df = bb_df.drop_duplicates(subset=bb_key, keep="last")
        if not enh_df.empty:
            enh_df = enh_df.drop_duplicates(subset=enh_key, keep="last")
        if not selector_df.empty:
            selector_key = [
                "system_type",
                "dataset",
                "model_name",
                "model_prefix",
                "explainer_name",
                "enhancer_name",
                "llm_name",
                "rag",
                "retriever_name",
                "corpus_name",
                "selector_name",
                "top_k",
                "metric_exp_ranking",
                "synthesizer_name",
                "epochs",
                "n_samples_synthesizer",
                "n_neighbors",
                "metric_neighbors",
                "aggregation_method",
                "th_model_conf",
                "th_enhancer_conf",
                "th_rbo",
                "n_bootstrap",
                "random_state",
            ]
            selector_df = selector_df.drop_duplicates(subset=selector_key, keep="last")

        report_df = pd.concat([bb_df, enh_df, selector_df], ignore_index=True)
        self.ch.save_dataframe(report_df, path)

    def _save_selector_blackbox_baseline_report(self, new_df: pd.DataFrame, save_global_evaluator: bool) -> None:
        if save_global_evaluator:
            path = f"{self.dm.selector_path}selector_blackbox_baseline_bootstrap_ci.csv"
        else:
            path = f"{self.dm.ev_report_path}selector_blackbox_baseline_bootstrap_ci.csv"

        if os.path.exists(path):
            report_df = self.ch.read_dataframe(path, dtype=dtype)
            report_df = pd.concat([report_df, new_df], ignore_index=True)
        else:
            report_df = new_df

        key_cols = [
            "system_type",
            "baseline_type",
            "dataset",
            "model_name",
            "model_prefix",
            "explainer_name",
            "enhancer_name",
            "llm_name",
            "rag",
            "retriever_name",
            "corpus_name",
            "selector_name",
            "top_k",
            "metric_exp_ranking",
            "synthesizer_name",
            "epochs",
            "n_samples_synthesizer",
            "n_neighbors",
            "metric_neighbors",
            "aggregation_method",
            "th_model_conf",
            "th_enhancer_conf",
            "th_rbo",
            "n_bootstrap",
            "random_state",
        ]
        report_df = report_df.drop_duplicates(subset=key_cols, keep="last")
        self.ch.save_dataframe(report_df, path)
