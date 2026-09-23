import itertools
import os
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from selector.core.metrics_helper import SelectorMetricsHelper
from selector.selector import Selector
from utils.checkpoint.checkpoint import Checkpoint
from utils.common_variables import (
    av_datasets,
    dpi,
    dtype,
    ind,
    selector_explanation_color_dict,
    selector_sensitivity_metric_color_dict,
    selector_sensitivity_metric_label_dict,
)
from utils.decorator_definition import log_method
from utils.log_manager.log_manager import LogManager


class SelectorSensitivityAnalysis(Selector):
    """
    Evaluate selector classification-report metrics over threshold grids.

    This class reuses the final selector inputs but avoids writing one selector
    dataframe per threshold setting. It computes the selector decisions in
    memory and saves compact sensitivity CSVs under global_evaluator/selector.

    The full-grid report is the canonical sensitivity result. The one-at-a-time
    and diagonal reports are filtered views of that full grid:
    - one_at_a_time: two thresholds fixed to baseline_threshold.
    - diagonal: all three thresholds set to the same value.
    """

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
        baseline_threshold: float = 0.7,
        threshold_values: Optional[Iterable[float]] = None,
        cuda_visible_devices: Optional[str] = "1,2,3",
        parallel: Optional[bool] = True,
    ):
        super().__init__(
            ch=ch,
            lm=lm,
            dataset_prefix=dataset_prefix,
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
            th_model_conf=baseline_threshold,
            th_enhancer_conf=baseline_threshold,
            th_rbo=baseline_threshold,
            cuda_visible_devices=cuda_visible_devices,
            parallel=parallel,
        )
        self.baseline_threshold = baseline_threshold
        self.threshold_values = list(threshold_values or np.round(np.arange(0.1, 1.0, 0.1), 1))
        self.metrics = SelectorMetricsHelper(self.ctx)

    @log_method
    def run(self) -> pd.DataFrame:
        source_df = self._load_selector_source_dataframe()
        bb_df = self.ch.read_dataframe(
            f"{self.dm.model_path}{self.ctx.model_name}_predicted.csv",
            dtype=dtype,
        )

        full_grid = self._run_full_grid(source_df=source_df, bb_df=bb_df)
        one_at_a_time = self._derive_one_at_a_time_view(full_grid)
        diagonal = self._derive_diagonal_view(full_grid)

        self._save_report(full_grid, "selector_sensitivity_full_grid_classification_reports.csv")
        self._save_report(one_at_a_time, "selector_sensitivity_one_at_a_time_classification_reports.csv")
        self._save_report(diagonal, "selector_sensitivity_diagonal_classification_reports.csv")

        combined = pd.concat([full_grid, one_at_a_time, diagonal], ignore_index=True)
        self._save_report(combined, "selector_sensitivity_classification_reports.csv")

        usage_full_grid = self._run_explanation_usage_full_grid(source_df=source_df)
        usage_one_at_a_time = self._derive_one_at_a_time_view(usage_full_grid)
        usage_diagonal = self._derive_diagonal_view(usage_full_grid)
        self._save_usage_report(usage_full_grid, "selector_sensitivity_full_grid_explanation_usage.csv")
        self._save_usage_report(usage_one_at_a_time, "selector_sensitivity_one_at_a_time_explanation_usage.csv")
        self._save_usage_report(usage_diagonal, "selector_sensitivity_diagonal_explanation_usage.csv")
        usage_combined = pd.concat([usage_full_grid, usage_one_at_a_time, usage_diagonal], ignore_index=True)
        self._save_usage_report(usage_combined, "selector_sensitivity_explanation_usage.csv")

        self._write_output_readme()
        return combined

    @log_method
    def run_ablation_study(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Evaluate selector ablations by switching reliability signals off.

        Thresholds set to 0.0 make the corresponding reliability signal always
        pass. The saved percentage-change columns compare each ablation against
        the full method configuration at 0.7/0.7/0.7.
        """
        source_df = self._load_selector_source_dataframe()
        bb_df = self.ch.read_dataframe(
            f"{self.dm.model_path}{self.ctx.model_name}_predicted.csv",
            dtype=dtype,
        )

        classification_rows = []
        usage_rows = []
        for ablation_rank, configuration, th_model_conf, th_enhancer_conf, th_rbo in self._ablation_settings():
            selector_df = self._apply_thresholds(
                source_df=source_df,
                th_model_conf=th_model_conf,
                th_enhancer_conf=th_enhancer_conf,
                th_rbo=th_rbo,
            )

            report_rows = self._build_report_rows(
                selector_df=selector_df,
                bb_df=bb_df,
                analysis_type="ablation",
                varied_threshold="fixed_ablation",
                th_model_conf=th_model_conf,
                th_enhancer_conf=th_enhancer_conf,
                th_rbo=th_rbo,
            )
            for row in report_rows:
                row["ablation_rank"] = ablation_rank
                row["configuration"] = configuration
            classification_rows.extend(report_rows)

            case_rows = self._build_explanation_usage_rows(
                selector_df=selector_df,
                analysis_type="ablation",
                varied_threshold="fixed_ablation",
                th_model_conf=th_model_conf,
                th_enhancer_conf=th_enhancer_conf,
                th_rbo=th_rbo,
            )
            for row in case_rows:
                row["ablation_rank"] = ablation_rank
                row["configuration"] = configuration
            usage_rows.extend(case_rows)

        classification_df = pd.DataFrame(classification_rows)
        usage_df = pd.DataFrame(usage_rows)
        classification_df = self._add_ablation_classification_changes_vs_full_method(classification_df)
        usage_df = self._add_ablation_usage_changes_vs_full_method(usage_df)

        self._save_ablation_report(classification_df, "selector_ablation_classification_reports.csv")
        self._save_ablation_usage_report(usage_df, "selector_ablation_explanation_usage.csv")
        self._write_output_readme()
        return classification_df, usage_df

    @log_method
    def plot(self, comparison_scope: str = "accepted_subset_bb") -> None:
        """
        Plot selector sensitivity line charts.

        Outputs one plot per dataset and varied threshold for the one-at-a-time
        view, plus one plot per dataset for the diagonal view. Each chart shows
        selector F1 macro, selector F1 for class 1, and coverage.
        """
        one_at_a_time_path = (
            f"{self.dm.selector_sensitivity_analysis_path}"
            "selector_sensitivity_one_at_a_time_classification_reports.csv"
        )
        diagonal_path = (
            f"{self.dm.selector_sensitivity_analysis_path}"
            "selector_sensitivity_diagonal_classification_reports.csv"
        )
        if not os.path.exists(one_at_a_time_path) or not os.path.exists(diagonal_path):
            self.lm.printl("[Warning] Missing selector sensitivity CSVs. Run sensitivity analysis first.")
            return

        one_at_a_time_df = self.ch.read_dataframe(one_at_a_time_path, dtype=dtype).copy()
        diagonal_df = self.ch.read_dataframe(diagonal_path, dtype=dtype).copy()

        metric_columns = ["selector_f1-score_macro", "selector_f1-score_1", "coverage"]
        for df in [one_at_a_time_df, diagonal_df]:
            for column in metric_columns + ["th_model_conf", "th_enhancer_conf", "th_rbo"]:
                df[column] = pd.to_numeric(df[column], errors="coerce")

        self._plot_one_at_a_time(one_at_a_time_df, comparison_scope, metric_columns)
        self._plot_diagonal(diagonal_df, comparison_scope, metric_columns)
        self._save_metric_legend()
        self.plot_explanation_usage()
        self._write_output_readme()

    @log_method
    def plot_explanation_usage(self) -> None:
        one_at_a_time_path = (
            f"{self.dm.selector_sensitivity_analysis_path}"
            "selector_sensitivity_one_at_a_time_explanation_usage.csv"
        )
        diagonal_path = (
            f"{self.dm.selector_sensitivity_analysis_path}"
            "selector_sensitivity_diagonal_explanation_usage.csv"
        )
        if not os.path.exists(one_at_a_time_path) or not os.path.exists(diagonal_path):
            self.lm.printl("[Warning] Missing selector sensitivity explanation-usage CSVs. Run sensitivity analysis first.")
            return

        one_at_a_time_df = self.ch.read_dataframe(one_at_a_time_path, dtype=dtype).copy()
        diagonal_df = self.ch.read_dataframe(diagonal_path, dtype=dtype).copy()
        for df in [one_at_a_time_df, diagonal_df]:
            for column in ["percentage", "th_model_conf", "th_enhancer_conf", "th_rbo"]:
                df[column] = pd.to_numeric(df[column], errors="coerce")

        self._plot_explanation_usage_one_at_a_time(one_at_a_time_df)
        self._plot_explanation_usage_diagonal(diagonal_df)
        self._save_explanation_usage_legend()

    def _run_full_grid(self, source_df: pd.DataFrame, bb_df: pd.DataFrame) -> pd.DataFrame:
        return self._run_threshold_grid(
            source_df=source_df,
            bb_df=bb_df,
            threshold_settings=self._full_grid_settings(),
            analysis_type="full_grid",
            varied_threshold="all_combinations",
        )

    def _load_selector_source_dataframe(self) -> pd.DataFrame:
        agreement_df = self.ch.read_dataframe(
            f"{self.dm.ev_agreement_csv_path}"
            f"{self.ctx.model_name}_{self.ctx.enhancer_name}_{self.ctx.llm_name}_"
            f"{str(self.ctx.rag)}_{self.ctx.retriever_name}_{self.ctx.corpus_name}_agreement_df.csv",
            dtype=dtype,
        )
        evaluator_df = self.ch.read_dataframe(
            f"{self.dm.ev_explanations_csv_path}{self.ctx.model_name}_{self.ctx.explainer_name}_evaluator_df.csv",
            dtype=dtype,
        )
        confidence_df = self.ch.read_dataframe(
            f"{self.dm.ce_enhancer_path}confidence_scores_{self.ctx.aggregation_method}.csv",
            dtype=dtype,
        )
        return agreement_df.merge(evaluator_df, on=ind, how="inner").merge(confidence_df, on=ind, how="inner")

    def _full_grid_settings(self) -> List[Tuple[float, float, float, Optional[str]]]:
        return [
            (model_th, enhancer_th, rbo_th, None)
            for model_th, enhancer_th, rbo_th in itertools.product(
                self.threshold_values,
                self.threshold_values,
                self.threshold_values,
            )
        ]

    def _run_threshold_grid(
        self,
        source_df: pd.DataFrame,
        bb_df: pd.DataFrame,
        threshold_settings: List[Tuple[float, float, float, Optional[str]]],
        analysis_type: str,
        varied_threshold: Optional[str] = None,
    ) -> pd.DataFrame:
        rows = []
        for th_model_conf, th_enhancer_conf, th_rbo, setting_varied_threshold in threshold_settings:
            selector_df = self._apply_thresholds(
                source_df=source_df,
                th_model_conf=th_model_conf,
                th_enhancer_conf=th_enhancer_conf,
                th_rbo=th_rbo,
            )
            rows.extend(
                self._build_report_rows(
                    selector_df=selector_df,
                    bb_df=bb_df,
                    analysis_type=analysis_type,
                    varied_threshold=setting_varied_threshold or varied_threshold,
                    th_model_conf=th_model_conf,
                    th_enhancer_conf=th_enhancer_conf,
                    th_rbo=th_rbo,
                )
            )
        return pd.DataFrame(rows)

    def _run_explanation_usage_full_grid(self, source_df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for th_model_conf, th_enhancer_conf, th_rbo, _ in self._full_grid_settings():
            selector_df = self._apply_thresholds(
                source_df=source_df,
                th_model_conf=th_model_conf,
                th_enhancer_conf=th_enhancer_conf,
                th_rbo=th_rbo,
            )
            rows.extend(
                self._build_explanation_usage_rows(
                    selector_df=selector_df,
                    analysis_type="full_grid",
                    varied_threshold="all_combinations",
                    th_model_conf=th_model_conf,
                    th_enhancer_conf=th_enhancer_conf,
                    th_rbo=th_rbo,
                )
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _ablation_settings() -> List[Tuple[int, str, float, float, float]]:
        return [
            (1, "Full method", 0.7, 0.7, 0.7),
            (2, "w/o black-box confidence", 0.0, 0.7, 0.7),
            (3, "w/o enhancer confidence", 0.7, 0.0, 0.7),
            (4, "w/o explanation agreement", 0.7, 0.7, 0.0),
            (5, "w/o black-box + enhancer confidence", 0.0, 0.0, 0.7),
            (6, "w/o black-box confidence + explanation agreement", 0.0, 0.7, 0.0),
            (7, "w/o enhancer confidence + explanation agreement", 0.7, 0.0, 0.0),
            (8, "w/o all reliability signals", 0.0, 0.0, 0.0),
        ]

    def _derive_one_at_a_time_view(self, full_grid: pd.DataFrame) -> pd.DataFrame:
        views = []
        threshold_columns = ["th_model_conf", "th_enhancer_conf", "th_rbo"]
        for varied_column in threshold_columns:
            fixed_columns = [column for column in threshold_columns if column != varied_column]
            mask = full_grid[fixed_columns[0]].eq(self.baseline_threshold) & full_grid[fixed_columns[1]].eq(
                self.baseline_threshold
            )
            view = full_grid.loc[mask].copy()
            view["analysis_type"] = "one_at_a_time"
            view["varied_threshold"] = varied_column
            views.append(view)
        return pd.concat(views, ignore_index=True)

    def _derive_diagonal_view(self, full_grid: pd.DataFrame) -> pd.DataFrame:
        mask = (
            full_grid["th_model_conf"].eq(full_grid["th_enhancer_conf"])
            & full_grid["th_model_conf"].eq(full_grid["th_rbo"])
        )
        view = full_grid.loc[mask].copy()
        view["analysis_type"] = "diagonal"
        view["varied_threshold"] = "all_equal"
        return view

    def _plot_one_at_a_time(
        self,
        df: pd.DataFrame,
        comparison_scope: str,
        metric_columns: List[str],
    ) -> None:
        threshold_label_map = {
            "th_model_conf": "model confidence threshold",
            "th_enhancer_conf": "enhancer confidence threshold",
            "th_rbo": "RBO threshold",
        }
        for dataset in self._dataset_order(df):
            for varied_threshold, xlabel in threshold_label_map.items():
                plot_df = df[
                    (df["dataset_prefix"].astype(str) == dataset)
                    & (df["comparison_scope"].astype(str) == comparison_scope)
                    & (df["varied_threshold"].astype(str) == varied_threshold)
                ].copy()
                if plot_df.empty:
                    continue

                plot_df = plot_df.sort_values(varied_threshold)
                out_name = f"{dataset}_selector_sensitivity_{varied_threshold}.png"
                self._save_metric_linechart(
                    plot_df=plot_df,
                    x_col=varied_threshold,
                    xlabel=xlabel,
                    out_name=out_name,
                    metric_columns=metric_columns,
                )

    def _plot_diagonal(
        self,
        df: pd.DataFrame,
        comparison_scope: str,
        metric_columns: List[str],
    ) -> None:
        for dataset in self._dataset_order(df):
            plot_df = df[
                (df["dataset_prefix"].astype(str) == dataset)
                & (df["comparison_scope"].astype(str) == comparison_scope)
            ].copy()
            if plot_df.empty:
                continue

            plot_df = plot_df.sort_values("th_model_conf")
            out_name = f"{dataset}_selector_sensitivity_diagonal.png"
            self._save_metric_linechart(
                plot_df=plot_df,
                x_col="th_model_conf",
                xlabel="shared threshold",
                out_name=out_name,
                metric_columns=metric_columns,
            )

    def _plot_explanation_usage_one_at_a_time(self, df: pd.DataFrame) -> None:
        threshold_label_map = {
            "th_model_conf": "model confidence threshold",
            "th_enhancer_conf": "enhancer confidence threshold",
            "th_rbo": "RBO threshold",
        }
        for dataset in self._dataset_order(df):
            for varied_threshold, xlabel in threshold_label_map.items():
                plot_df = df[
                    (df["dataset_prefix"].astype(str) == dataset)
                    & (df["varied_threshold"].astype(str) == varied_threshold)
                ].copy()
                if plot_df.empty:
                    continue

                out_name = f"{dataset}_selector_sensitivity_explanation_usage_{varied_threshold}.png"
                self._save_explanation_usage_linechart(
                    plot_df=plot_df,
                    x_col=varied_threshold,
                    xlabel=xlabel,
                    out_name=out_name,
                )

    def _plot_explanation_usage_diagonal(self, df: pd.DataFrame) -> None:
        for dataset in self._dataset_order(df):
            plot_df = df[df["dataset_prefix"].astype(str) == dataset].copy()
            if plot_df.empty:
                continue

            out_name = f"{dataset}_selector_sensitivity_explanation_usage_diagonal.png"
            self._save_explanation_usage_linechart(
                plot_df=plot_df,
                x_col="th_model_conf",
                xlabel="shared threshold",
                out_name=out_name,
            )

    def _save_metric_linechart(
        self,
        plot_df: pd.DataFrame,
        x_col: str,
        xlabel: str,
        out_name: str,
        metric_columns: List[str],
    ) -> None:
        fig, ax = plt.subplots(figsize=(6, 2.5), dpi=dpi)
        x = pd.to_numeric(plot_df[x_col], errors="coerce")

        for metric in metric_columns:
            ax.plot(
                x,
                pd.to_numeric(plot_df[metric], errors="coerce"),
                marker=self._metric_marker(metric),
                linewidth=2.5,
                color=self._metric_color(metric),
                label=self._metric_label(metric),
            )

        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xlim(min(self.threshold_values), max(self.threshold_values))
        ax.set_ylim(0, 1.02)
        ax.set_xticks(self.threshold_values)
        ax.set_xticklabels([f"{value:.1f}" for value in self.threshold_values], fontsize=12)
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.7)

        fig.tight_layout()
        out_path = f"{self.dm.selector_sensitivity_analysis_plot_path}{out_name}"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    def _save_explanation_usage_linechart(
        self,
        plot_df: pd.DataFrame,
        x_col: str,
        xlabel: str,
        out_name: str,
    ) -> None:
        fig, ax = plt.subplots(figsize=(6, 2.5), dpi=dpi)
        order_cases = ["both", "explainer", "enhancer", "abstain"]

        wide_df = (
            plot_df.pivot_table(
                index=x_col,
                columns="explanation_case",
                values="percentage",
                aggfunc="mean",
            )
            .reindex(columns=order_cases)
            .reset_index()
            .sort_values(x_col)
        )

        x = pd.to_numeric(wide_df[x_col], errors="coerce")
        for case in order_cases:
            ax.plot(
                x,
                pd.to_numeric(wide_df[case], errors="coerce"),
                marker=self._explanation_usage_marker(case),
                linewidth=2.5,
                color=selector_explanation_color_dict[case],
                label=case,
            )

        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xlim(min(self.threshold_values), max(self.threshold_values))
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.set_xticks(self.threshold_values)
        ax.set_xticklabels([f"{value:.1f}" for value in self.threshold_values], fontsize=12)
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.7)

        fig.tight_layout()
        out_path = f"{self.dm.selector_sensitivity_analysis_plot_path}{out_name}"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    def _save_metric_legend(self) -> None:
        metrics = ["selector_f1-score_macro", "selector_f1-score_1", "coverage"]
        legend_handles = [
            plt.Line2D(
                [0], [0],
                color=self._metric_color(metric),
                marker=self._metric_marker(metric),
                lw=3,
                label=self._metric_label(metric),
            )
            for metric in metrics
        ]

        legend_fig = plt.figure(figsize=(6, 0.5), dpi=dpi)
        legend_ax = legend_fig.add_subplot(111)
        legend_ax.legend(
            handles=legend_handles,
            loc="center",
            frameon=False,
            ncol=len(legend_handles),
            fontsize=14,
        )
        legend_ax.axis("off")
        legend_fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        legend_path = f"{self.dm.selector_sensitivity_analysis_plot_path}selector_sensitivity_metrics_legend.png"
        plt.savefig(legend_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
        plt.close(legend_fig)

    def _save_explanation_usage_legend(self) -> None:
        order_cases = ["both", "explainer", "enhancer", "abstain"]
        legend_handles = [
            plt.Line2D(
                [0], [0],
                color=selector_explanation_color_dict[case],
                marker=self._explanation_usage_marker(case),
                lw=3,
                label=case,
            )
            for case in order_cases
        ]

        legend_fig = plt.figure(figsize=(6, 0.5), dpi=dpi)
        legend_ax = legend_fig.add_subplot(111)
        legend_ax.legend(
            handles=legend_handles,
            loc="center",
            frameon=False,
            ncol=len(legend_handles),
            fontsize=14,
        )
        legend_ax.axis("off")
        legend_fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        legend_path = f"{self.dm.selector_sensitivity_analysis_plot_path}selector_sensitivity_explanation_usage_legend.png"
        plt.savefig(legend_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
        plt.close(legend_fig)

    @staticmethod
    def _metric_color(metric: str) -> str:
        return selector_sensitivity_metric_color_dict[metric]

    @staticmethod
    def _metric_marker(metric: str) -> str:
        marker_map = {
            "selector_f1-score_macro": "o",
            "selector_f1-score_1": "s",
            "coverage": "^",
        }
        return marker_map[metric]

    @staticmethod
    def _metric_label(metric: str) -> str:
        return selector_sensitivity_metric_label_dict[metric]

    @staticmethod
    def _explanation_usage_marker(case: str) -> str:
        marker_map = {
            "both": "o",
            "explainer": "s",
            "enhancer": "^",
            "abstain": "D",
        }
        return marker_map[case]

    @staticmethod
    def _dataset_order(df: pd.DataFrame) -> List[str]:
        present = set(df["dataset_prefix"].astype(str))
        return [dataset for dataset in av_datasets if dataset in present]

    def _apply_thresholds(
        self,
        source_df: pd.DataFrame,
        th_model_conf: float,
        th_enhancer_conf: float,
        th_rbo: float,
    ) -> pd.DataFrame:
        df = source_df.copy()
        target = self.ctx.target_col
        pred_col = self.ctx.selector_prediction_col
        abstain_col = self.ctx.selector_is_abstain_col
        use_enhanced_col = self.ctx.selector_use_enhanced_expl_col
        use_enhancer_col = self.ctx.selector_use_enhancer_expl_col

        agree = self._as_bool(df["agree"])
        model_conf = pd.to_numeric(df[self.ctx.model_conf_col], errors="coerce")
        enhancer_conf = pd.to_numeric(df[self.ctx.enhancer_conf_col], errors="coerce")
        rbo = pd.to_numeric(df[self.ctx.metric_complete_name], errors="coerce")
        model_pred = pd.to_numeric(df[self.ctx.model_pred_col], errors="coerce")
        enhancer_pred = pd.to_numeric(df[self.ctx.enhancer_pred_col], errors="coerce")

        if self.ctx.selector_name == "simple":
            accepted = agree
            prediction = np.where(accepted, model_pred, np.nan)
            use_enhanced = accepted.copy()
            use_enhancer = accepted.copy()
        elif self.ctx.selector_name == "abstain":
            model_high = model_conf >= th_model_conf
            enhancer_high = enhancer_conf >= th_enhancer_conf
            rbo_high = rbo >= th_rbo

            agree_both_high_rbo_high = agree & model_high & enhancer_high & rbo_high
            agree_both_high_rbo_low = agree & model_high & enhancer_high & (~rbo_high)
            agree_model_high_enhancer_low_rbo_high = agree & model_high & (~enhancer_high) & rbo_high
            agree_model_high_enhancer_low_rbo_low = agree & model_high & (~enhancer_high) & (~rbo_high)
            agree_model_low_enhancer_high_rbo_high = agree & (~model_high) & enhancer_high & rbo_high

            accepted_agree = (
                agree_both_high_rbo_high
                | agree_both_high_rbo_low
                | agree_model_high_enhancer_low_rbo_high
                | agree_model_high_enhancer_low_rbo_low
                | agree_model_low_enhancer_high_rbo_high
            )
            accepted_disagree_model = (~agree) & model_high & (~enhancer_high)
            accepted_disagree_enhancer = (~agree) & (~model_high) & enhancer_high
            accepted = accepted_agree | accepted_disagree_model | accepted_disagree_enhancer

            prediction = np.full(len(df), np.nan)
            prediction[accepted_agree | accepted_disagree_model] = model_pred[accepted_agree | accepted_disagree_model]
            prediction[accepted_disagree_enhancer] = enhancer_pred[accepted_disagree_enhancer]

            use_enhanced = (
                agree_both_high_rbo_high
                | agree_both_high_rbo_low
                | agree_model_high_enhancer_low_rbo_high
                | agree_model_high_enhancer_low_rbo_low
                | agree_model_low_enhancer_high_rbo_high
                | accepted_disagree_model
            )
            use_enhancer = (
                agree_both_high_rbo_high
                | agree_model_high_enhancer_low_rbo_high
                | agree_model_low_enhancer_high_rbo_high
                | accepted_disagree_enhancer
            )
        else:
            raise ValueError(f"Unknown selector_name: {self.ctx.selector_name}")

        df[pred_col] = prediction
        df[abstain_col] = ~accepted
        df[self.ctx.selector_is_non_abstain_col] = accepted
        df[use_enhanced_col] = use_enhanced
        df[use_enhancer_col] = use_enhancer
        df[self.ctx.selector_final_correct_col] = np.where(
            accepted & pd.notna(prediction) & pd.notna(df[target]),
            prediction == pd.to_numeric(df[target], errors="coerce"),
            np.nan,
        )
        return df

    @staticmethod
    def _as_bool(series: pd.Series) -> pd.Series:
        if series.dtype == bool:
            return series
        return series.astype(str).str.lower().isin(["true", "1", "yes"])

    def _build_report_rows(
        self,
        selector_df: pd.DataFrame,
        bb_df: pd.DataFrame,
        analysis_type: str,
        varied_threshold: Optional[str],
        th_model_conf: float,
        th_enhancer_conf: float,
        th_rbo: float,
    ) -> List[dict]:
        target = self.ctx.target_col
        selector_pred_col = self.ctx.selector_prediction_col
        selector_is_abstain_col = self.ctx.selector_is_abstain_col
        bb_pred_col = self.ctx.model_pred_col

        accepted_df = selector_df.loc[~selector_df[selector_is_abstain_col]].copy()
        n_total = len(selector_df)
        n_abstain = int(selector_df[selector_is_abstain_col].sum())
        n_non_abstain = len(accepted_df)

        selector_target_num = pd.to_numeric(selector_df[target], errors="coerce")
        mask_0 = selector_target_num == 0
        mask_1 = selector_target_num == 1
        n_true_0 = int(mask_0.sum())
        n_true_1 = int(mask_1.sum())

        y_true_selector = pd.to_numeric(accepted_df[target], errors="coerce")
        y_pred_selector = pd.to_numeric(accepted_df[selector_pred_col], errors="coerce")
        selector_valid = y_true_selector.notna() & y_pred_selector.notna()
        selector_metrics = self.metrics.build_report_metrics(
            y_true_selector.loc[selector_valid],
            y_pred_selector.loc[selector_valid],
        )

        y_true_bb = pd.to_numeric(bb_df[target], errors="coerce")
        y_pred_bb = pd.to_numeric(bb_df[bb_pred_col], errors="coerce")
        bb_valid = y_true_bb.notna() & y_pred_bb.notna()
        bb_metrics_full = self.metrics.build_report_metrics(
            y_true_bb.loc[bb_valid],
            y_pred_bb.loc[bb_valid],
        )

        accepted_indices = accepted_df[ind].tolist()
        bb_accepted_df = bb_df[bb_df[ind].isin(accepted_indices)].copy()
        y_true_bb_accepted = pd.to_numeric(bb_accepted_df[target], errors="coerce")
        y_pred_bb_accepted = pd.to_numeric(bb_accepted_df[bb_pred_col], errors="coerce")
        bb_accepted_valid = y_true_bb_accepted.notna() & y_pred_bb_accepted.notna()
        bb_metrics_accepted = self.metrics.build_report_metrics(
            y_true_bb_accepted.loc[bb_accepted_valid],
            y_pred_bb_accepted.loc[bb_accepted_valid],
        )

        return [
            self._build_report_row(
                selector_df=selector_df,
                selector_metrics=selector_metrics,
                bb_metrics=bb_metrics_full,
                comparison_scope="full_dataset_bb",
                analysis_type=analysis_type,
                varied_threshold=varied_threshold,
                th_model_conf=th_model_conf,
                th_enhancer_conf=th_enhancer_conf,
                th_rbo=th_rbo,
                n_total=n_total,
                n_non_abstain=n_non_abstain,
                n_abstain=n_abstain,
                n_true_0=n_true_0,
                n_true_1=n_true_1,
                mask_0=mask_0,
                mask_1=mask_1,
            ),
            self._build_report_row(
                selector_df=selector_df,
                selector_metrics=selector_metrics,
                bb_metrics=bb_metrics_accepted,
                comparison_scope="accepted_subset_bb",
                analysis_type=analysis_type,
                varied_threshold=varied_threshold,
                th_model_conf=th_model_conf,
                th_enhancer_conf=th_enhancer_conf,
                th_rbo=th_rbo,
                n_total=n_total,
                n_non_abstain=n_non_abstain,
                n_abstain=n_abstain,
                n_true_0=n_true_0,
                n_true_1=n_true_1,
                mask_0=mask_0,
                mask_1=mask_1,
            ),
        ]

    def _build_explanation_usage_rows(
        self,
        selector_df: pd.DataFrame,
        analysis_type: str,
        varied_threshold: Optional[str],
        th_model_conf: float,
        th_enhancer_conf: float,
        th_rbo: float,
    ) -> List[dict]:
        use_enhanced_col = self.ctx.selector_use_enhanced_expl_col
        use_enhancer_col = self.ctx.selector_use_enhancer_expl_col
        order_cases = ["both", "explainer", "enhancer", "abstain"]
        n_total = len(selector_df)

        tmp_df = selector_df.copy()
        tmp_df[use_enhanced_col] = self._as_bool(tmp_df[use_enhanced_col])
        tmp_df[use_enhancer_col] = self._as_bool(tmp_df[use_enhancer_col])

        both = tmp_df[use_enhanced_col] & tmp_df[use_enhancer_col]
        explainer = tmp_df[use_enhanced_col] & (~tmp_df[use_enhancer_col])
        enhancer = (~tmp_df[use_enhanced_col]) & tmp_df[use_enhancer_col]
        abstain = (~tmp_df[use_enhanced_col]) & (~tmp_df[use_enhancer_col])

        counts = {
            "both": int(both.sum()),
            "explainer": int(explainer.sum()),
            "enhancer": int(enhancer.sum()),
            "abstain": int(abstain.sum()),
        }

        rows = []
        for case in order_cases:
            count = counts[case]
            rows.append({
                "analysis_type": analysis_type,
                "varied_threshold": varied_threshold,
                "baseline_threshold": self.baseline_threshold,
                "th_model_conf": th_model_conf,
                "th_enhancer_conf": th_enhancer_conf,
                "th_rbo": th_rbo,
                "dataset_prefix": self.ctx.dataset_prefix,
                "model_name": self.ctx.model_name,
                "model_prefix": self.ctx.model_prefix,
                "explainer_name": self.ctx.explainer_name,
                "enhancer_name": self.ctx.enhancer_name,
                "llm_name": self.ctx.llm_name,
                "rag": self.ctx.rag,
                "retriever_name": self.ctx.retriever_name,
                "corpus_name": self.ctx.corpus_name,
                "selector_name": self.ctx.selector_name,
                "metric_complete_name": self.ctx.metric_complete_name,
                "explanation_case": case,
                "count": count,
                "percentage": float(100 * count / n_total) if n_total > 0 else np.nan,
                "n_total": n_total,
            })
        return rows

    def _build_report_row(
        self,
        selector_df: pd.DataFrame,
        selector_metrics: dict,
        bb_metrics: dict,
        comparison_scope: str,
        analysis_type: str,
        varied_threshold: Optional[str],
        th_model_conf: float,
        th_enhancer_conf: float,
        th_rbo: float,
        n_total: int,
        n_non_abstain: int,
        n_abstain: int,
        n_true_0: int,
        n_true_1: int,
        mask_0: pd.Series,
        mask_1: pd.Series,
    ) -> dict:
        metrics_for_comparison = [
            "accuracy", "precision_macro", "recall_macro", "f1-score_macro",
            "precision_weighted", "recall_weighted", "f1-score_weighted",
            "precision_0", "recall_0", "f1-score_0",
            "precision_1", "recall_1", "f1-score_1",
        ]
        selector_is_abstain_col = self.ctx.selector_is_abstain_col

        row = {
            "analysis_type": analysis_type,
            "comparison_scope": comparison_scope,
            "varied_threshold": varied_threshold,
            "baseline_threshold": self.baseline_threshold,
            "th_model_conf": th_model_conf,
            "th_enhancer_conf": th_enhancer_conf,
            "th_rbo": th_rbo,
            "dataset_prefix": self.ctx.dataset_prefix,
            "model_name": self.ctx.model_name,
            "model_prefix": self.ctx.model_prefix,
            "explainer_name": self.ctx.explainer_name,
            "enhancer_name": self.ctx.enhancer_name,
            "llm_name": self.ctx.llm_name,
            "rag": self.ctx.rag,
            "retriever_name": self.ctx.retriever_name,
            "corpus_name": self.ctx.corpus_name,
            "selector_name": self.ctx.selector_name,
            "metric_complete_name": self.ctx.metric_complete_name,
            "n_total": n_total,
            "n_non_abstain": n_non_abstain,
            "n_abstain": n_abstain,
            "coverage": n_non_abstain / n_total if n_total > 0 else np.nan,
            "abstain_rate": n_abstain / n_total if n_total > 0 else np.nan,
            "abstain_rate_0": (
                float((selector_df[selector_is_abstain_col] & mask_0).sum() / n_true_0)
                if n_true_0 > 0 else np.nan
            ),
            "abstain_rate_1": (
                float((selector_df[selector_is_abstain_col] & mask_1).sum() / n_true_1)
                if n_true_1 > 0 else np.nan
            ),
        }
        row.update({f"selector_{key}": value for key, value in selector_metrics.items()})
        row.update({f"bb_{key}": value for key, value in bb_metrics.items()})
        row.update({
            f"pct_change_{metric}": self.metrics.pct_change(
                selector_metrics.get(metric),
                bb_metrics.get(metric),
            )
            for metric in metrics_for_comparison
        })
        return row

    def _add_ablation_classification_changes_vs_full_method(self, df: pd.DataFrame) -> pd.DataFrame:
        out_df = df.copy()
        old_pct_cols = [column for column in out_df.columns if column.startswith("pct_change_")]
        out_df = out_df.drop(columns=old_pct_cols)

        metric_columns = [
            "coverage",
            "abstain_rate",
            "abstain_rate_0",
            "abstain_rate_1",
            "selector_accuracy",
            "selector_precision_macro",
            "selector_recall_macro",
            "selector_f1-score_macro",
            "selector_precision_weighted",
            "selector_recall_weighted",
            "selector_f1-score_weighted",
            "selector_precision_0",
            "selector_recall_0",
            "selector_f1-score_0",
            "selector_precision_1",
            "selector_recall_1",
            "selector_f1-score_1",
        ]
        group_cols = [
            "dataset_prefix",
            "model_name",
            "model_prefix",
            "explainer_name",
            "enhancer_name",
            "llm_name",
            "rag",
            "retriever_name",
            "corpus_name",
            "selector_name",
            "metric_complete_name",
            "comparison_scope",
        ]
        return self._add_changes_vs_full_method(
            out_df=out_df,
            group_cols=group_cols,
            metric_columns=metric_columns,
        )

    def _add_ablation_usage_changes_vs_full_method(self, df: pd.DataFrame) -> pd.DataFrame:
        group_cols = [
            "dataset_prefix",
            "model_name",
            "model_prefix",
            "explainer_name",
            "enhancer_name",
            "llm_name",
            "rag",
            "retriever_name",
            "corpus_name",
            "selector_name",
            "metric_complete_name",
            "explanation_case",
        ]
        return self._add_changes_vs_full_method(
            out_df=df.copy(),
            group_cols=group_cols,
            metric_columns=["percentage"],
        )

    def _add_changes_vs_full_method(
        self,
        out_df: pd.DataFrame,
        group_cols: List[str],
        metric_columns: List[str],
    ) -> pd.DataFrame:
        for metric in metric_columns:
            out_df[f"pct_change_vs_full_method_{metric}"] = np.nan

        for _, group_df in out_df.groupby(group_cols, dropna=False):
            full_rows = group_df[group_df["configuration"].astype(str).eq("Full method")]
            if full_rows.empty:
                continue

            full_row = full_rows.iloc[0]
            for idx in group_df.index:
                for metric in metric_columns:
                    out_df.loc[idx, f"pct_change_vs_full_method_{metric}"] = self.metrics.pct_change(
                        pd.to_numeric(out_df.loc[idx, metric], errors="coerce"),
                        pd.to_numeric(full_row[metric], errors="coerce"),
                    )
        return out_df

    def _save_report(self, new_df: pd.DataFrame, filename: str) -> None:
        path = f"{self.dm.selector_sensitivity_analysis_path}{filename}"
        if os.path.exists(path):
            old_df = self.ch.read_dataframe(path, dtype=dtype)
            out_df = pd.concat([old_df, new_df], ignore_index=True)
        else:
            out_df = new_df

        key_cols = [
            "analysis_type", "comparison_scope", "varied_threshold",
            "baseline_threshold", "th_model_conf", "th_enhancer_conf", "th_rbo",
            "dataset_prefix", "model_name", "model_prefix", "explainer_name",
            "enhancer_name", "llm_name", "rag", "retriever_name",
            "corpus_name", "selector_name", "metric_complete_name",
        ]
        out_df = out_df.drop_duplicates(subset=key_cols, keep="last")
        self.ch.save_dataframe(out_df, path)

    def _save_usage_report(self, new_df: pd.DataFrame, filename: str) -> None:
        path = f"{self.dm.selector_sensitivity_analysis_path}{filename}"
        if os.path.exists(path):
            old_df = self.ch.read_dataframe(path, dtype=dtype)
            out_df = pd.concat([old_df, new_df], ignore_index=True)
        else:
            out_df = new_df

        key_cols = [
            "analysis_type", "varied_threshold", "baseline_threshold",
            "th_model_conf", "th_enhancer_conf", "th_rbo",
            "dataset_prefix", "model_name", "model_prefix", "explainer_name",
            "enhancer_name", "llm_name", "rag", "retriever_name",
            "corpus_name", "selector_name", "metric_complete_name",
            "explanation_case",
        ]
        out_df = out_df.drop_duplicates(subset=key_cols, keep="last")
        self.ch.save_dataframe(out_df, path)

    def _save_ablation_report(self, new_df: pd.DataFrame, filename: str) -> None:
        path = f"{self.dm.selector_sensitivity_analysis_path}{filename}"
        if os.path.exists(path):
            old_df = self.ch.read_dataframe(path, dtype=dtype)
            out_df = pd.concat([old_df, new_df], ignore_index=True)
        else:
            out_df = new_df

        key_cols = [
            "analysis_type", "comparison_scope", "configuration",
            "baseline_threshold", "th_model_conf", "th_enhancer_conf", "th_rbo",
            "dataset_prefix", "model_name", "model_prefix", "explainer_name",
            "enhancer_name", "llm_name", "rag", "retriever_name",
            "corpus_name", "selector_name", "metric_complete_name",
        ]
        out_df = out_df.drop_duplicates(subset=key_cols, keep="last")
        out_df = out_df.sort_values(["dataset_prefix", "comparison_scope", "ablation_rank"])
        self.ch.save_dataframe(out_df, path)

    def _save_ablation_usage_report(self, new_df: pd.DataFrame, filename: str) -> None:
        path = f"{self.dm.selector_sensitivity_analysis_path}{filename}"
        if os.path.exists(path):
            old_df = self.ch.read_dataframe(path, dtype=dtype)
            out_df = pd.concat([old_df, new_df], ignore_index=True)
        else:
            out_df = new_df

        key_cols = [
            "analysis_type", "configuration",
            "baseline_threshold", "th_model_conf", "th_enhancer_conf", "th_rbo",
            "dataset_prefix", "model_name", "model_prefix", "explainer_name",
            "enhancer_name", "llm_name", "rag", "retriever_name",
            "corpus_name", "selector_name", "metric_complete_name",
            "explanation_case",
        ]
        out_df = out_df.drop_duplicates(subset=key_cols, keep="last")
        out_df = out_df.sort_values(["dataset_prefix", "ablation_rank", "explanation_case"])
        self.ch.save_dataframe(out_df, path)

    def _write_output_readme(self) -> None:
        path = f"{self.dm.selector_sensitivity_analysis_path}README.md"
        text = f"""# Selector Sensitivity Analysis

This directory contains threshold-sensitivity reports for the selector.

The selector has three thresholds:

- `th_model_conf`: black-box model confidence threshold.
- `th_enhancer_conf`: enhancer confidence threshold.
- `th_rbo`: explanation-ranking agreement threshold.

The baseline threshold used for the focused views is `{self.baseline_threshold}`.

## Files

- `selector_sensitivity_full_grid_classification_reports.csv`
  - Canonical result.
  - Contains every combination of `th_model_conf`, `th_enhancer_conf`, and `th_rbo`.
  - With the default values `0.1 ... 0.9`, this is `9 * 9 * 9 = 729` threshold settings per dataset.

- `selector_sensitivity_one_at_a_time_classification_reports.csv`
  - View derived from the full grid.
  - Two thresholds are fixed to `baseline_threshold`; the third varies.
  - Use `varied_threshold` to know which threshold is changing.

- `selector_sensitivity_diagonal_classification_reports.csv`
  - View derived from the full grid.
  - Keeps only rows where `th_model_conf == th_enhancer_conf == th_rbo`.

- `selector_sensitivity_classification_reports.csv`
  - Combined file containing the full grid plus the two derived views.
  - Use `analysis_type` to filter `full_grid`, `one_at_a_time`, or `diagonal`.

- `selector_sensitivity_full_grid_explanation_usage.csv`
  - Explanation-usage percentages for every threshold combination.

- `selector_sensitivity_one_at_a_time_explanation_usage.csv`
  - Explanation-usage view where two thresholds are fixed to `baseline_threshold`.

- `selector_sensitivity_diagonal_explanation_usage.csv`
  - Explanation-usage view where all thresholds are equal.

- `selector_sensitivity_explanation_usage.csv`
  - Combined explanation-usage file containing the full grid plus the two derived views.

- `selector_ablation_classification_reports.csv`
  - Fixed ablation study over the full-method threshold setting and the seven reliability-signal removals.
  - `configuration` identifies the ablation.
  - `pct_change_vs_full_method_*` columns compare each configuration against `Full method`.
  - Black-box-relative `pct_change_*` columns are intentionally omitted from this file.

- `selector_ablation_explanation_usage.csv`
  - Explanation-usage percentages for the same ablation configurations.
  - `pct_change_vs_full_method_percentage` compares each explanation-usage percentage against `Full method` for the same `explanation_case`.

- `selector_ablation_summary.csv`
  - Publication-oriented merged ablation table.
  - Contains one row per dataset and active reliability-signal configuration.
  - Keeps selector F1 macro, selector F1 for class 1, coverage, and explanation-usage percentages.

## Plots

Plots are saved under `plot/`.

- `<dataset>_selector_sensitivity_th_model_conf.png`
  - One-at-a-time view where `th_model_conf` varies and the other thresholds are fixed to `baseline_threshold`.

- `<dataset>_selector_sensitivity_th_enhancer_conf.png`
  - One-at-a-time view where `th_enhancer_conf` varies and the other thresholds are fixed to `baseline_threshold`.

- `<dataset>_selector_sensitivity_th_rbo.png`
  - One-at-a-time view where `th_rbo` varies and the other thresholds are fixed to `baseline_threshold`.

- `<dataset>_selector_sensitivity_diagonal.png`
  - Diagonal view where all three thresholds have the same value.

- `selector_sensitivity_metrics_legend.png`
  - Shared legend for all sensitivity line charts.

Each plot contains three lines: selector F1 macro, selector F1 for class 1, and coverage.
The plotting method uses `comparison_scope == "accepted_subset_bb"` by default to avoid duplicated selector rows.

- `<dataset>_selector_sensitivity_explanation_usage_th_model_conf.png`
  - Explanation-usage one-at-a-time view where `th_model_conf` varies.

- `<dataset>_selector_sensitivity_explanation_usage_th_enhancer_conf.png`
  - Explanation-usage one-at-a-time view where `th_enhancer_conf` varies.

- `<dataset>_selector_sensitivity_explanation_usage_th_rbo.png`
  - Explanation-usage one-at-a-time view where `th_rbo` varies.

- `<dataset>_selector_sensitivity_explanation_usage_diagonal.png`
  - Explanation-usage diagonal view where all thresholds have the same value.

- `selector_sensitivity_explanation_usage_legend.png`
  - Shared legend for explanation-usage line charts.

Each explanation-usage plot contains four percentages: both, explainer, enhancer, and abstain.

Each threshold setting has two comparison rows:

- `comparison_scope == "full_dataset_bb"` compares selector metrics on the accepted selector subset against black-box metrics on the full dataset.
- `comparison_scope == "accepted_subset_bb"` compares selector metrics against black-box metrics computed only on the same accepted subset.

## Important Columns

- `analysis_type`: `full_grid`, `one_at_a_time`, or `diagonal`.
- `comparison_scope`: black-box comparison scope.
- `varied_threshold`: threshold varied in the one-at-a-time view; `all_combinations` for the full grid; `all_equal` for the diagonal view.
- `dataset_prefix`: dataset name.
- `n_total`: total selector input rows.
- `n_non_abstain`: rows accepted by the selector.
- `n_abstain`: rows rejected by the selector.
- `coverage`: `n_non_abstain / n_total`.
- `abstain_rate`: `n_abstain / n_total`.
- `selector_*`: classification-report metric for selector predictions on accepted rows.
- `bb_*`: classification-report metric for the black-box baseline.
- `pct_change_*`: percentage change of selector metric relative to the selected black-box comparison.
- `explanation_case`: explanation-usage category in the explanation-usage CSVs.
- `percentage`: percentage of rows in each explanation-usage category.
"""
        with open(path, "w", encoding="utf-8") as file:
            file.write(text)
