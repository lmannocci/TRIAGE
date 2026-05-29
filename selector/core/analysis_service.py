

from utils.checkpoint import Checkpoint
from utils.directory_manager.directory_manager import DirectoryManager
from utils.log_manager.log_manager import *
from utils.common_variables import *


from selector.core.context import SelectorContext
from selector.core.decision_engine import SelectorDecisionEngine
from selector.core.plot_helper import SelectorPlotHelper
from selector.core.metrics_helper import SelectorMetricsHelper

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Optional

class SelectorAnalysisService:
    def __init__(self, ch: Checkpoint, lm: LogManager, ctx: SelectorContext, dm: DirectoryManager):
        self.ch = ch
        self.lm = lm
        self.ctx = ctx
        self.dm = dm
        self.metrics = SelectorMetricsHelper(ctx)
        self.plotter = SelectorPlotHelper(ch, lm, dm, ctx)

    def _selector_df_path(self) -> str:
        return (
            f"{self.dm.en_selector_path}"
            f"{self.ctx.model_name}_{self.ctx.explainer_name}_"
            f"{self.ctx.metric_complete_name}_{self.ctx.selector_name}_selector_df.csv"
        )

    def _save_simple_distribution(self, df: pd.DataFrame, col: str, out_csv: str):
        out = df[col].value_counts(dropna=False).rename_axis(col).reset_index(name="count")
        out["percentage"] = 100 * out["count"] / len(df) if len(df) > 0 else np.nan
        self.ch.save_dataframe(out, out_csv)

    def _save_global_summary(self, df: pd.DataFrame, out_csv: str):
        target_col = self.ctx.target_col
        is_abstain_col = self.ctx.selector_is_abstain_col
        is_non_abstain_col = self.ctx.selector_is_non_abstain_col

        n_total = len(df)
        n_abstain = int(df[is_abstain_col].sum())
        n_non_abstain = int(df[is_non_abstain_col].sum())
        coverage = n_non_abstain / n_total if n_total > 0 else np.nan
        abstain_rate = n_abstain / n_total if n_total > 0 else np.nan

        mask_0 = df[target_col] == 0
        mask_1 = df[target_col] == 1
        n_true_0 = int(mask_0.sum())
        n_true_1 = int(mask_1.sum())

        row = {
            "dataset_prefix": self.ctx.dataset_prefix,
            "model_name": self.ctx.model_name,
            "explainer_name": self.ctx.explainer_name,
            "selector_name": self.ctx.selector_name,
            "metric_complete_name": self.ctx.metric_complete_name,
            "n_total": n_total,
            "n_abstain": n_abstain,
            "n_non_abstain": n_non_abstain,
            "coverage": coverage,
            "abstain_rate": abstain_rate,
            "abstain_rate_0": float((df[is_abstain_col] & mask_0).sum() / n_true_0) if n_true_0 > 0 else np.nan,
            "abstain_rate_1": float((df[is_abstain_col] & mask_1).sum() / n_true_1) if n_true_1 > 0 else np.nan,
        }
        self.ch.save_dataframe(pd.DataFrame([row]), out_csv)

    def _save_f1_report(self, df: pd.DataFrame, out_csv: str, group_col: Optional[str] = None):
        target_col = self.ctx.target_col
        pred_col = self.ctx.selector_prediction_col
        is_non_abstain_col = self.ctx.selector_is_non_abstain_col

        rows = []
        groups = df.groupby(group_col, dropna=False) if group_col else [(None, df)]

        for group_value, group_df in groups:
            accepted_df = group_df[group_df[is_non_abstain_col]].copy()
            y_true = pd.to_numeric(accepted_df[target_col], errors="coerce")
            y_pred = pd.to_numeric(accepted_df[pred_col], errors="coerce")
            valid_mask = y_true.notna() & y_pred.notna()

            metrics = self.metrics.build_report_metrics(y_true.loc[valid_mask], y_pred.loc[valid_mask])

            row = {
                "n_rows_total_group": len(group_df),
                "n_rows_non_abstain_group": len(accepted_df),
                "n_valid_group": int(valid_mask.sum()),
                "coverage_group": len(accepted_df) / len(group_df) if len(group_df) > 0 else np.nan,
            }
            if group_col:
                row[group_col] = group_value
            else:
                row["scope"] = "global_non_abstain"
                row["n_rows"] = len(accepted_df)
                row["n_valid"] = int(valid_mask.sum())
                row.pop("n_rows_total_group")
                row.pop("n_rows_non_abstain_group")
                row.pop("coverage_group")
                row.pop("n_valid_group")

            row.update(metrics)
            rows.append(row)

        self.ch.save_dataframe(pd.DataFrame(rows), out_csv)

    def _save_explanation_usage_summary(self, df: pd.DataFrame) -> None:
        """
        Compute the percentage of explanation-usage cases for one dataset/configuration
        and save/update them in the global evaluator.

        Explanation cases:
            - both      : explainer explanation + enhancer explanation
            - explainer : only black-box/explainer explanation
            - enhancer  : only enhancer explanation
            - abstain   : no explanation
        """

        col_enhanced = f"{self.ctx.selector_name}_use_enhanced_explanation"
        col_enhancer = f"{self.ctx.selector_name}_use_enhancer_explanation"

        if col_enhanced not in df.columns or col_enhancer not in df.columns:
            return

        tmp_df = df.copy()

        tmp_df[col_enhanced] = tmp_df[col_enhanced].astype(bool)
        tmp_df[col_enhancer] = tmp_df[col_enhancer].astype(bool)

        def _map_case(row):
            if row[col_enhanced] and row[col_enhancer]:
                return "both"
            elif row[col_enhanced] and not row[col_enhancer]:
                return "explainer"
            elif not row[col_enhanced] and row[col_enhancer]:
                return "enhancer"
            else:
                return "abstain"

        tmp_df["explanation_case"] = tmp_df.apply(_map_case, axis=1)

        order = ["both", "explainer", "enhancer", "abstain"]

        counts = tmp_df["explanation_case"].value_counts().reindex(order).fillna(0)
        percentages = counts / len(tmp_df) * 100 if len(tmp_df) > 0 else counts * np.nan

        rows = []
        for case in order:
            rows.append({
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
                "count": int(counts.loc[case]),
                "percentage": float(percentages.loc[case]),
            })

        new_df = pd.DataFrame(rows)

        path = f"{self.dm.global_evaluator_path}selector_explanation_usage_summary.csv"

        key_cols = [
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

        if os.path.exists(path):
            global_df = self.ch.read_dataframe(path, dtype=dtype)
            global_df = pd.concat([global_df, new_df], ignore_index=True)
        else:
            global_df = new_df

        global_df = global_df.drop_duplicates(subset=key_cols, keep="last")

        self.ch.save_dataframe(global_df, path)


    def analyze_selector(self):
        df = self.ch.read_dataframe(self._selector_df_path(), dtype=dtype).copy()
        target_col = self.ctx.target_col
        pred_col = self.ctx.selector_prediction_col
        decision_col = self.ctx.selector_decision_col
        case_col = self.ctx.selector_case_col

        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
        df[pred_col] = pd.to_numeric(df[pred_col], errors="coerce")

        self._save_simple_distribution(df, decision_col, f"{self.dm.en_selector_csv_path}decision_distribution.csv")
        self._save_simple_distribution(df, case_col, f"{self.dm.en_selector_csv_path}logging_case_distribution.csv")
        self._save_global_summary(df, f"{self.dm.en_selector_csv_path}global_summary.csv")
        self._save_f1_report(df, f"{self.dm.en_selector_csv_path}f1_report_global.csv")
        self._save_f1_report(df, f"{self.dm.en_selector_csv_path}f1_report_by_decision.csv", decision_col)
        self._save_f1_report(df, f"{self.dm.en_selector_csv_path}f1_report_by_logging_case.csv", case_col)
        self._save_explanation_usage_summary(df)

        decision_df = self.ch.read_dataframe(f"{self.dm.en_selector_csv_path}decision_distribution.csv", dtype=dtype)
        case_df = self.ch.read_dataframe(f"{self.dm.en_selector_csv_path}logging_case_distribution.csv", dtype=dtype)

        self.plotter.save_distribution_plot(
            decision_df, decision_col, "percentage",
            f"{self.dm.en_selector_plot_path}decision_distribution.png",
            "Percentage"
        )
        self.plotter.save_distribution_plot(
            case_df, case_col, "percentage",
            f"{self.dm.en_selector_plot_path}logging_case_distribution.png",
            "Percentage"
        )

        for metric in ["f1-score_macro", "f1-score_1", "f1-score_0", "recall_0", "recall_1"]:
            suffix = metric.replace("-", "_")
            self.plotter.save_metric_by_group_plot(
                f"{self.dm.en_selector_csv_path}f1_report_by_logging_case.csv",
                case_col,
                metric,
                f"{self.dm.en_selector_plot_path}{suffix}_by_logging_case.png",
            )

        for metric in ["f1-score_macro", "f1-score_1", "f1-score_0"]:
            suffix = metric.replace("-", "_")
            self.plotter.save_metric_by_group_plot(
                f"{self.dm.en_selector_csv_path}f1_report_by_decision.csv",
                decision_col,
                metric,
                f"{self.dm.en_selector_plot_path}{suffix}_by_decision.png",
            )

        optional_cols = [
            self.ctx.model_conf_col,
            self.ctx.enhancer_conf_col,
            self.ctx.metric_complete_name,
        ]
        # self.plotter.save_histograms(df, optional_cols, self.dm.en_selector_plot_path)
        # self.plotter.save_kde_plots(df, optional_cols, self.dm.en_selector_plot_path)
        # self.plotter.save_distribution_plots(df, optional_cols, self.dm.en_selector_plot_path)

        self.plotter.plot_selector_case_scatter()

    def save_selector_classification_report(self):
        df_selector = self.ch.read_dataframe(self._selector_df_path(), dtype=dtype)
        df_bb = self.ch.read_dataframe(f"{self.dm.model_path}{self.ctx.model_name}_predicted.csv", dtype=dtype)

        target = self.ctx.target_col
        selector_pred_col = self.ctx.selector_prediction_col
        selector_is_abstain_col = self.ctx.selector_is_abstain_col
        bb_pred_col = self.ctx.model_pred_col

        path_full = f"{self.dm.selector_path}selector_classification_reports.csv"
        path_accepted = f"{self.dm.selector_path}selector_classification_reports_on_accepted_subset.csv"

        accepted_df = df_selector.loc[~df_selector[selector_is_abstain_col]].copy()
        n_total = len(df_selector)
        n_abstain = int(df_selector[selector_is_abstain_col].sum())
        n_non_abstain = len(accepted_df)

        coverage = n_non_abstain / n_total if n_total > 0 else np.nan
        abstain_rate = n_abstain / n_total if n_total > 0 else np.nan

        selector_target_num = pd.to_numeric(df_selector[target], errors="coerce")
        mask_0 = selector_target_num == 0
        mask_1 = selector_target_num == 1
        n_true_0 = int(mask_0.sum())
        n_true_1 = int(mask_1.sum())

        y_true_selector = pd.to_numeric(accepted_df[target], errors="coerce")
        y_pred_selector = pd.to_numeric(accepted_df[selector_pred_col], errors="coerce")
        valid_mask_selector = y_true_selector.notna() & y_pred_selector.notna()
        selector_metrics = self.metrics.build_report_metrics(
            y_true_selector.loc[valid_mask_selector],
            y_pred_selector.loc[valid_mask_selector],
        )

        y_true_bb = pd.to_numeric(df_bb[target], errors="coerce")
        y_pred_bb = pd.to_numeric(df_bb[bb_pred_col], errors="coerce")
        valid_mask_bb = y_true_bb.notna() & y_pred_bb.notna()
        bb_metrics_full = self.metrics.build_report_metrics(
            y_true_bb.loc[valid_mask_bb],
            y_pred_bb.loc[valid_mask_bb],
        )

        accepted_indices = accepted_df[ind].tolist()
        df_bb_accepted = df_bb[df_bb[ind].isin(accepted_indices)].copy()
        y_true_bb_accepted = pd.to_numeric(df_bb_accepted[target], errors="coerce")
        y_pred_bb_accepted = pd.to_numeric(df_bb_accepted[bb_pred_col], errors="coerce")
        valid_mask_bb_accepted = y_true_bb_accepted.notna() & y_pred_bb_accepted.notna()
        bb_metrics_accepted = self.metrics.build_report_metrics(
            y_true_bb_accepted.loc[valid_mask_bb_accepted],
            y_pred_bb_accepted.loc[valid_mask_bb_accepted],
        )

        metrics_for_comparison = [
            "accuracy", "precision_macro", "recall_macro", "f1-score_macro",
            "precision_weighted", "recall_weighted", "f1-score_weighted",
            "precision_0", "recall_0", "f1-score_0",
            "precision_1", "recall_1", "f1-score_1",
        ]

        def save_row(path: str, bb_metrics: dict):
            base_row = {
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
                "coverage": coverage,
                "abstain_rate": abstain_rate,
                "abstain_rate_0": float((df_selector[selector_is_abstain_col] & mask_0).sum() / n_true_0) if n_true_0 > 0 else np.nan,
                "abstain_rate_1": float((df_selector[selector_is_abstain_col] & mask_1).sum() / n_true_1) if n_true_1 > 0 else np.nan,
            }
            base_row.update({f"selector_{k}": v for k, v in selector_metrics.items()})
            base_row.update({f"bb_{k}": v for k, v in bb_metrics.items()})
            base_row.update({
                f"pct_change_{metric}": self.metrics.pct_change(selector_metrics.get(metric), bb_metrics.get(metric))
                for metric in metrics_for_comparison
            })

            new_df = pd.DataFrame([base_row])
            if os.path.exists(path):
                old_df = self.ch.read_dataframe(path, dtype=dtype)
                out_df = pd.concat([old_df, new_df], ignore_index=True)
            else:
                out_df = new_df

            key_cols = [
                "dataset_prefix", "model_name", "model_prefix", "explainer_name",
                "enhancer_name", "llm_name", "rag", "retriever_name",
                "corpus_name", "selector_name", "metric_complete_name",
            ]
            out_df = out_df.drop_duplicates(subset=key_cols, keep="last")
            self.ch.save_dataframe(out_df, path)

        save_row(path_full, bb_metrics_full)
        save_row(path_accepted, bb_metrics_accepted)

    def plot_selector_vs_bb_dumbbell_per_dataset(self):
        axis_label_fontsize = 16
        tick_fontsize = 14

        df = self.ch.read_dataframe(
            f"{self.dm.selector_path}selector_classification_reports_on_accepted_subset.csv",
            dtype=dtype
        )

        metric_pairs = [
            ("bb_precision_0", "selector_precision_0", "P0"),
            ("bb_recall_0", "selector_recall_0", "R0"),
            ("bb_f1-score_0", "selector_f1-score_0", "F1$_0$"),
            ("bb_precision_1", "selector_precision_1", "P1"),
            ("bb_recall_1", "selector_recall_1", "R1"),
            ("bb_f1-score_1", "selector_f1-score_1", "F1$_1$"),
            ("bb_f1-score_macro", "selector_f1-score_macro", "F1$_M$"),
            ("bb_accuracy", "selector_accuracy", "Acc"),
        ]

        row = df.loc[df["dataset_prefix"] == self.ctx.dataset_prefix].iloc[0]
        labels, bb_vals, sel_vals = [], [], []

        for bb_col, sel_col, label in metric_pairs:
            labels.append(label)
            bb_vals.append(float(row[bb_col]))
            sel_vals.append(float(row[sel_col]))

        y = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(11, 4), dpi=dpi)

        for i, (bb, sel) in enumerate(zip(bb_vals, sel_vals)):
            line_color = "#74c476" if sel >= bb else "#fb6a4a"
            ax.plot([bb, sel], [i, i], color=line_color, linewidth=2.5, zorder=1)
            ax.scatter(bb, i, color="#6b6b6b", s=55, zorder=2)
            ax.scatter(sel, i, color=line_color, s=55, zorder=3)
            delta = sel - bb
            sign = "+" if delta >= 0 else ""
            ax.text(max(bb, sel) + 0.015, i, f"{sign}{delta:.2f}", va="center", fontsize=axis_label_fontsize)

        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlim(0, 1.08)
        ax.grid(linestyle="--", alpha=0.3)
        ax.invert_yaxis()
        ax.tick_params(axis="x", labelsize=axis_label_fontsize)
        ax.tick_params(axis="y", labelsize=axis_label_fontsize)
        fig.tight_layout()

        out_path = f"{self.dm.selector_path}{self.ctx.dataset_prefix}_selector_vs_bb_dumbbell.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)