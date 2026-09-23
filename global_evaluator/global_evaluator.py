import os
import re
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from utils.checkpoint.checkpoint import *
from utils.directory_manager.directory_manager import DirectoryManager
from utils.log_manager.log_manager import *
from utils.decorator_definition import *
from utils.common_variables import *

absolute_path = os.path.dirname(__file__)
results_path = os.path.join(absolute_path, f"..{os.sep}results{os.sep}")
data_path = os.path.join(absolute_path, f"..{os.sep}data{os.sep}")


class GlobalEvaluator:
    def __init__(self, ch: Checkpoint, lm: LogManager):
        self.ch = ch
        self.lm = lm
        self.dm = DirectoryManager(lm, results_path)

    @log_method
    def merge_ablation_results(self, comparison_scope: str = "accepted_subset_bb") -> pd.DataFrame:
        """
        Merge selector ablation predictive metrics and explanation usage.

        This is a global aggregation: it reads the cross-dataset ablation CSVs
        produced by SelectorSensitivityAnalysis and saves one compact summary
        table under global_evaluator/selector/sensitivity_analysis.
        """
        classification_path = (
            f"{self.dm.selector_sensitivity_analysis_path}"
            "selector_ablation_classification_reports.csv"
        )
        usage_path = (
            f"{self.dm.selector_sensitivity_analysis_path}"
            "selector_ablation_explanation_usage.csv"
        )
        if not os.path.exists(classification_path) or not os.path.exists(usage_path):
            self.lm.printl("[Warning] Missing ablation CSVs. Run SelectorSensitivityAnalysis.run_ablation_study first.")
            return pd.DataFrame()

        ablation_df = self.ch.read_dataframe(classification_path, dtype=dtype).copy()
        usage_df = self.ch.read_dataframe(usage_path, dtype=dtype).copy()
        ablation_df = ablation_df[ablation_df["comparison_scope"].astype(str).eq(comparison_scope)].copy()

        ablation_df["configuration_pub"] = ablation_df.apply(self._publication_configuration_from_thresholds, axis=1)
        usage_df["configuration_pub"] = usage_df.apply(self._publication_configuration_from_thresholds, axis=1)

        predictive_df = (
            ablation_df[
                [
                    "dataset_prefix",
                    "configuration_pub",
                    "selector_f1-score_macro",
                    "selector_f1-score_1",
                    "coverage",
                ]
            ]
            .drop_duplicates(subset=["dataset_prefix", "configuration_pub"])
            .rename(
                columns={
                    "configuration_pub": "configuration",
                    "selector_f1-score_macro": "selector_f1_macro",
                    "selector_f1-score_1": "selector_f1_1",
                }
            )
        )

        usage_wide_df = (
            usage_df[
                [
                    "dataset_prefix",
                    "configuration_pub",
                    "explanation_case",
                    "percentage",
                ]
            ]
            .pivot_table(
                index=["dataset_prefix", "configuration_pub"],
                columns="explanation_case",
                values="percentage",
                aggfunc="first",
            )
            .reset_index()
            .rename(
                columns={
                    "configuration_pub": "configuration",
                    "explainer": "explainer_only_pct",
                    "enhancer": "enhancer_only_pct",
                    "both": "both_pct",
                    "abstain": "abstain_pct",
                }
            )
        )

        summary_df = predictive_df.merge(
            usage_wide_df,
            on=["dataset_prefix", "configuration"],
            how="left",
        )

        output_columns = [
            "dataset_prefix",
            "configuration",
            "selector_f1_macro",
            "selector_f1_1",
            "coverage",
            "explainer_only_pct",
            "enhancer_only_pct",
            "both_pct",
            "abstain_pct",
        ]
        summary_df = summary_df[output_columns]
        summary_df = self._sort_ablation_summary(summary_df)
        summary_df = self._round_ablation_summary(summary_df)
        self._validate_ablation_summary(summary_df)

        out_path = f"{self.dm.selector_sensitivity_analysis_path}selector_ablation_summary.csv"
        self.ch.save_dataframe(summary_df, out_path)
        return summary_df

    @staticmethod
    def _publication_configuration_from_thresholds(row: pd.Series) -> str:
        model_active = pd.to_numeric(row["th_model_conf"], errors="coerce") > 0
        enhancer_active = pd.to_numeric(row["th_enhancer_conf"], errors="coerce") > 0
        exp_active = pd.to_numeric(row["th_rbo"], errors="coerce") > 0

        if model_active and enhancer_active and exp_active:
            return "Full"
        if model_active and enhancer_active:
            return "Model + Enhancer"
        if model_active and exp_active:
            return "Model + Explanation"
        if enhancer_active and exp_active:
            return "Enhancer + Explanation"
        if model_active:
            return "Model only"
        if enhancer_active:
            return "Enhancer only"
        if exp_active:
            return "Explanation only"
        return "No reliability signals"

    @staticmethod
    def _publication_configuration_order() -> list:
        return [
            "Full",
            "Model + Enhancer",
            "Model + Explanation",
            "Enhancer + Explanation",
            "Model only",
            "Enhancer only",
            "Explanation only",
            "No reliability signals",
        ]

    def _sort_ablation_summary(self, summary_df: pd.DataFrame) -> pd.DataFrame:
        out_df = summary_df.copy()
        out_df["configuration"] = pd.Categorical(
            out_df["configuration"],
            categories=self._publication_configuration_order(),
            ordered=True,
        )
        out_df["dataset_prefix"] = pd.Categorical(
            out_df["dataset_prefix"],
            categories=av_datasets,
            ordered=True,
        )
        out_df = out_df.sort_values(["dataset_prefix", "configuration"]).reset_index(drop=True)
        out_df["dataset_prefix"] = out_df["dataset_prefix"].astype(str)
        out_df["configuration"] = out_df["configuration"].astype(str)
        return out_df

    @staticmethod
    def _round_ablation_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
        out_df = summary_df.copy()
        predictive_columns = ["selector_f1_macro", "selector_f1_1", "coverage"]
        explanation_columns = ["explainer_only_pct", "enhancer_only_pct", "both_pct", "abstain_pct"]
        out_df[predictive_columns] = out_df[predictive_columns].apply(pd.to_numeric, errors="coerce").round(3)
        out_df[explanation_columns] = out_df[explanation_columns].apply(pd.to_numeric, errors="coerce").round(2)
        return out_df

    def _validate_ablation_summary(self, summary_df: pd.DataFrame) -> None:
        explanation_columns = ["explainer_only_pct", "enhancer_only_pct", "both_pct", "abstain_pct"]
        validation_df = summary_df.copy()
        validation_df["explanation_pct_sum"] = validation_df[explanation_columns].sum(axis=1)
        invalid_pct = validation_df[~validation_df["explanation_pct_sum"].between(99.9, 100.1)]
        if not invalid_pct.empty:
            self.lm.printl(
                "[Warning] Some ablation explanation percentages do not sum to 100: "
                f"{invalid_pct[['dataset_prefix', 'configuration', 'explanation_pct_sum']].to_dict('records')}"
            )

        validation_df["coverage_from_abstention"] = 1 - validation_df["abstain_pct"] / 100
        validation_df["coverage_difference"] = (
            pd.to_numeric(validation_df["coverage"], errors="coerce")
            - validation_df["coverage_from_abstention"]
        ).abs()
        invalid_coverage = validation_df[validation_df["coverage_difference"] > 0.02]
        if not invalid_coverage.empty:
            self.lm.printl(
                "[Warning] Coverage and abstention are not consistent in ablation summary: "
                f"{invalid_coverage[['dataset_prefix', 'configuration', 'coverage', 'abstain_pct', 'coverage_from_abstention']].to_dict('records')}"
            )

    def _plot_judge_metrics_from_path(
        self,
        path: str,
        out_name: str,
        legend_name: str,
        metric_order: list,
        metric_label_map: dict,
        out_dir: str = None,
    ):
        if not os.path.exists(path):
            return

        df = self.ch.read_dataframe(path, dtype=dtype).copy()

        dataset_order = ["pima", "diabetes", "stroke", "liver", "covid"]
        dataset_color_map = dataset_color_dict

        df = df[df["dataset_prefix"].isin(dataset_order)].copy()
        df["dataset_prefix"] = pd.Categorical(
            df["dataset_prefix"],
            categories=dataset_order,
            ordered=True,
        )
        df = df.sort_values("dataset_prefix")

        x = np.arange(len(metric_order))
        width = 0.15

        fig, ax = plt.subplots(figsize=(11, 5), dpi=dpi)

        for i, dataset in enumerate(dataset_order):
            row_df = df[df["dataset_prefix"].astype(str) == dataset]
            if row_df.empty:
                continue

            row = row_df.iloc[0]

            means = []
            stds = []

            for metric in metric_order:
                mean_col = f"{metric}_mean"
                std_col = f"{metric}_std"

                means.append(pd.to_numeric(row.get(mean_col, np.nan), errors="coerce"))
                stds.append(pd.to_numeric(row.get(std_col, np.nan), errors="coerce"))

            offset = (i - (len(dataset_order) - 1) / 2) * width

            ax.bar(
                x + offset,
                means,
                width,
                yerr=stds,
                capsize=4,
                color=dataset_color_map.get(dataset),
                label=dataset,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [metric_label_map[m] for m in metric_order],
            ha="center",
            fontsize=14,
        )

        ax.tick_params(axis="y", labelsize=14)
        ax.set_ylim(0, 6)
        ax.set_yticks(np.arange(0, 6, 1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.5))
        ax.grid(axis="y", which="major", linestyle="--", linewidth=0.7, alpha=0.7)
        ax.grid(axis="y", which="minor", linestyle="--", linewidth=0.4, alpha=0.4)

        fig.tight_layout()
        if out_dir is None:
            out_dir = self.dm.validator_global_path
        out_path = f"{out_dir}{out_name}"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        legend_handles = [
            plt.Line2D(
                [0], [0],
                color=dataset_color_map[dataset],
                lw=6,
                label=dataset,
            )
            for dataset in dataset_order
            if dataset in df["dataset_prefix"].astype(str).values
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

        legend_path = f"{out_dir}{legend_name}"

        plt.savefig(
            legend_path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close(legend_fig)

    @log_method
    def plot_global_explanation_usage_stacked(self):
        """
        Plot stacked bar chart of explanation usage percentages across datasets.
        - Adds percentage labels inside each stack
        - Removes legend from main plot
        - Saves legend separately

        Outputs:
            selector_explanation_usage_stacked.png
            selector_explanation_usage_stacked_legend.png
        """

        path = f"{self.dm.global_evaluator_path}selector_explanation_usage_summary.csv"
        if not os.path.exists(path):
            return

        df = self.ch.read_dataframe(path, dtype=dtype)

        out_path = f"{self.dm.global_evaluator_path}selector_explanation_usage_stacked.png"
        legend_path = f"{self.dm.global_evaluator_path}selector_explanation_usage_stacked_legend.png"

        axis_label_fontsize = 16
        tick_fontsize = 14

        order_cases = ["both", "explainer", "enhancer", "abstain"]
        order_datasets = av_datasets
        df["percentage"] = pd.to_numeric(df["percentage"], errors="coerce")

        plot_df = (
            df.pivot_table(
                index="dataset_prefix",
                columns="explanation_case",
                values="percentage",
                aggfunc="mean"
            )
            .reindex(columns=order_cases)
            .fillna(0)
        )
        plot_df = plot_df.reindex(order_datasets)

        fig, ax = plt.subplots(figsize=(10, 5), dpi=dpi)

        bottom = np.zeros(len(plot_df))
        palette = selector_explanation_color_dict

        for case in order_cases:
            values = plot_df[case].values

            bars = ax.bar(
                plot_df.index,
                values,
                bottom=bottom,
                color=palette.get(case),
            )

            # --- Add text inside each segment ---
            for i, (bar, val, btm) in enumerate(zip(bars, values, bottom)):
                if val > 1.6:  # avoid clutter for tiny segments
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        btm + val / 2,
                        f"{val:.1f}",
                        ha="center",
                        va="center",
                        fontsize=14,
                        color="black"
                    )

            bottom += values

        ax.set_xlabel("")
        ax.set_ylabel("percentage (%)", fontsize=axis_label_fontsize)
        ax.tick_params(axis="x", labelsize=tick_fontsize, rotation=30)
        ax.tick_params(axis="y", labelsize=tick_fontsize)
        ax.set_ylim(0, 100)
        

        # --- NO legend here ---
        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        # ---------------------------------------------------------
        # Separate legend figure
        # ---------------------------------------------------------
        label_map = {
            "both": "both",
            "explainer": "explainer",
            "enhancer": "enhancer",
            "abstain": "abstain",
        }

        legend_handles = [
            plt.Line2D(
                [0], [0],
                color=palette[case],
                lw=6,
                label=label_map.get(case, case)
            )
            for case in order_cases
            if case in plot_df.columns
        ]

        legend_fig = plt.figure(figsize=(6, 0.5), dpi=dpi)
        legend_ax = legend_fig.add_subplot(111)

        legend_ax.legend(
            handles=legend_handles,
            loc="center",
            frameon=False,
            ncol=len(legend_handles),
            fontsize=14
        )

        legend_ax.axis("off")
        legend_fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        plt.savefig(legend_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
        plt.close(legend_fig)

    @log_method
    def plot_aggregated_explanation_feature_statistics_linechart(self):
        """
        Plot the aggregated explanation feature statistics as a line chart.

        The chart uses one line per dataset and plots the mean value of each
        rank position (feature1_mean, feature2_mean, ...).

        Input:
            self.dm.global_evaluator_path/aggregated_explanation_feature_statistics.csv

        Output:
            self.dm.global_evaluator_path/aggregated_explanation_feature_statistics_linechart.png
            self.dm.global_evaluator_path/aggregated_explanation_feature_statistics_linechart_legend.png
        """

        path = f"{self.dm.global_evaluator_path}aggregated_explanation_feature_statistics.csv"
        if not os.path.exists(path):
            return

        df = self.ch.read_dataframe(path, dtype=dtype).copy()
        if df.empty or "dataset_prefix" not in df.columns:
            return

        dataset_order = [dataset for dataset in av_datasets if dataset in df["dataset_prefix"].astype(str).values]
        if not dataset_order:
            return

        feature_mean_cols = [
            col for col in df.columns
            if re.fullmatch(r"feature\d+_mean", str(col))
        ]
        if not feature_mean_cols:
            return

        feature_mean_cols = sorted(feature_mean_cols, key=lambda col: int(re.search(r"\d+", col).group()))

        plot_df = df.copy()
        plot_df["dataset_prefix"] = plot_df["dataset_prefix"].astype(str)

        numeric_cols = feature_mean_cols + [
            col for col in df.columns
            if re.fullmatch(r"feature\d+_std", str(col))
        ]
        for col in numeric_cols:
            plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

        grouped = plot_df.groupby("dataset_prefix", as_index=False)[feature_mean_cols].mean(numeric_only=True)

        x = np.arange(1, len(feature_mean_cols) + 1)
        fig, ax = plt.subplots(figsize=(11, 5), dpi=dpi)

        for dataset in dataset_order:
            row_df = grouped[grouped["dataset_prefix"] == dataset]
            if row_df.empty:
                continue

            row = row_df.iloc[0]
            y = row[feature_mean_cols].to_numpy(dtype=float)

            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2.5,
                color=dataset_color_dict.get(dataset),
                label=dataset,
            )

        ax.set_xlabel("feature rank", fontsize=14)
        ax.set_ylabel("mean explanation value", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in x], fontsize=12)
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.7)

        fig.tight_layout()
        out_path = f"{self.dm.global_evaluator_path}aggregated_explanation_feature_statistics_linechart.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        legend_handles = [
            plt.Line2D(
                [0], [0],
                color=dataset_color_dict[dataset],
                lw=6,
                label=dataset,
            )
            for dataset in dataset_order
            if dataset in grouped["dataset_prefix"].astype(str).values
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

        legend_path = f"{self.dm.global_evaluator_path}aggregated_explanation_feature_statistics_linechart_legend.png"
        plt.savefig(legend_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
        plt.close(legend_fig)

    @log_method
    def plot_llm_judge_metrics(self):
        """
        Plot LLM-judge explanation metrics aggregated by metric.

        X-axis:
            metrics

        Bars:
            datasets

        Input:
            self.dm.validator_global_path/aggregated_llm_judge_metrics.csv

        Output:
            self.dm.validator_global_path/llm_judge_metrics.png
            self.dm.validator_global_path/llm_judge_metrics_legend.png
        """
        metric_order = [
            "feature_faithfulness",
            "prediction_consistency",
            "unsupported_claims",
            "clarity",
        ]

        metric_label_map = {
            "feature_faithfulness": "feature\nfaithfulness",
            "prediction_consistency": "prediction\nconsistency",
            "unsupported_claims": "absent\nunsupported claims",
            "clarity": "clinical\nclarity",
        }
        self._plot_judge_metrics_from_path(
            path=f"{self.dm.validator_explanation_global_path}aggregated_llm_judge_metrics.csv",
            out_name="llm_judge_metrics.png",
            legend_name="llm_judge_metrics_legend.png",
            metric_order=metric_order,
            metric_label_map=metric_label_map,
            out_dir=self.dm.validator_explanation_global_path,
        )

    @log_method
    def plot_llm_rag_judge_metrics(self):
        """
        Plot LLM-judge metrics for retrieved documents aggregated by metric.

        Output:
            self.dm.validator_global_path/llm_rag_judge_metrics.png
            self.dm.validator_global_path/llm_rag_judge_metrics_legend.png
        """

        metric_order = [
            "retrieval_relevance",
            "grounding",
            "unsupported_claims",
        ]

        metric_label_map = {
            "retrieval_relevance": "retrieval\nrelevance",
            "grounding": "grounding",
            "unsupported_claims": "absent\nunsupported claims",
        }

        self._plot_judge_metrics_from_path(
            path=f"{self.dm.validator_rag_global_path}aggregated_llm_judge_metrics.csv",
            out_name="llm_rag_judge_metrics.png",
            legend_name="llm_rag_judge_metrics_legend.png",
            metric_order=metric_order,
            metric_label_map=metric_label_map,
            out_dir=self.dm.validator_rag_global_path,
        )

    @log_method
    def plot_llm_judge_metrics_starplot(self):
        """
        Create one radar/star plot with all datasets.
        Color distinguishes datasets.

        Input:
            self.dm.validator_global_path/aggregated_llm_judge_metrics.csv

        Output:
            self.dm.validator_global_path/llm_judge_metrics_starplot.png
            self.dm.validator_global_path/llm_judge_metrics_starplot_legend.png
        """

        path = f"{self.dm.validator_global_path}aggregated_llm_judge_metrics.csv"
        if not os.path.exists(path):
            return

        df = self.ch.read_dataframe(path, dtype=dtype).copy()

        dataset_order = ["pima", "diabetes", "stroke", "liver", "covid"]

        metric_order = [
            "feature_faithfulness",
            "prediction_consistency",
            "unsupported_claims",
            "clarity",
        ]

        metric_label_map = {
            "feature_faithfulness": "feature\nfaithfulness",
            "prediction_consistency": "prediction\nconsistency",
            "unsupported_claims": "no unsupported\nclaims",
            "clarity": "clarity",
        }

        dataset_color_map = dataset_color_dict

        df = df[df["dataset_prefix"].isin(dataset_order)].copy()
        df["dataset_prefix"] = pd.Categorical(
            df["dataset_prefix"],
            categories=dataset_order,
            ordered=True,
        )
        df = df.sort_values("dataset_prefix")

        num_vars = len(metric_order)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        labels = [metric_label_map[m] for m in metric_order]

        fig, ax = plt.subplots(
            figsize=(6, 6),
            subplot_kw=dict(polar=True),
            dpi=dpi,
        )

        for dataset in dataset_order:
            row_df = df[df["dataset_prefix"] == dataset]
            if row_df.empty:
                continue

            row = row_df.iloc[0]

            values = [
                float(pd.to_numeric(row[f"{metric}_mean"], errors="coerce"))
                for metric in metric_order
            ]
            values += values[:1]

            color = dataset_color_map[dataset]

            ax.fill(
                angles,
                values,
                color=color,
                alpha=0.25,
            )

            ax.plot(
                angles,
                values,
                color=color,
                linewidth=2.5,
            )

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([])

        label_radius = 6.5  # move labels outside the outer circle

        for angle, label in zip(angles[:-1], labels):
            ax.text(
                angle,
                label_radius,
                label,
                fontsize=15,
                ha="center",
                va="center",
            )

        ax.set_ylim(0, 5.5)

        # inner circles every 0.5
        ax.set_yticks(np.arange(0.5, 6.5, 1))

        # hide default radial labels
        ax.set_yticklabels([])

        ax.grid(color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.spines["polar"].set_visible(False)

        # radial labels only every 1
        for r in np.arange(1, 6, 1):
            ax.text(
                np.pi / 2 - 0.18,
                r + 0.12,
                f"{r:.0f}",
                color="gray",
                ha="center",
                va="center",
                fontsize=12,
            )

        fig.tight_layout()

        out_path = f"{self.dm.validator_global_path}llm_judge_metrics_starplot.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        # ---------------------------------------------------------
        # Separate legend figure
        # ---------------------------------------------------------
        legend_handles = [
            plt.Line2D(
                [0], [0],
                color=dataset_color_map[dataset],
                lw=6,
                label=dataset,
            )
            for dataset in dataset_order
            if dataset in df["dataset_prefix"].astype(str).values
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

        legend_path = f"{self.dm.validator_global_path}llm_judge_metrics_starplot_legend.png"

        plt.savefig(
            legend_path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close(legend_fig)
