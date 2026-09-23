

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

        # optional_cols = [
        #     self.ctx.model_conf_col,
        #     self.ctx.enhancer_conf_col,
        #     self.ctx.metric_complete_name,
        # ]
        # self.plotter.save_histograms(df, optional_cols, self.dm.en_selector_plot_path)
        # self.plotter.save_kde_plots(df, optional_cols, self.dm.en_selector_plot_path)
        # self.plotter.save_distribution_plots(df, optional_cols, self.dm.en_selector_plot_path)

    def plot_selector_case_scatter(self):
        self.plotter.plot_selector_case_scatter()

    def save_selector_classification_report(self):
        """
        Save selector classification reports under the global selector folder.

        Three complementary comparisons are produced:

        1. `selector_classification_reports.csv`
           Compares selector metrics on its non-abstained predictions against
           blackbox metrics computed on the full test set. This answers whether
           the selective system improves over the original blackbox evaluated
           on all available test rows.

        2. `selector_classification_reports_on_accepted_subset.csv`
           Compares selector metrics against blackbox metrics computed on the
           exact same rows accepted by the selector. Here, the subset is chosen
           by the selector decision logic, so `pct_change_*` measures the
           selector improvement over the blackbox on the selector-accepted
           subset.

        3. `selector_classification_reports_matched_coverage_bb.csv`
           Compares selector metrics against a selective blackbox baseline with
           the same coverage as the selector. The blackbox subset is obtained
           by ranking selector-eligible test rows by blackbox confidence from
           highest to lowest and selecting exactly `n_non_abstain` rows. Ties
           at the confidence cutoff are resolved deterministically by smaller
           `original_index`, then original row order. This answers whether the
           selector improves over a confidence-based blackbox abstention policy
           at matched coverage.
        """
        # Load the selector output and the original blackbox predictions. The
        # selector dataframe contains abstention decisions; the blackbox file
        # contains predictions and confidence scores for the test set.
        df_selector = self.ch.read_dataframe(self._selector_df_path(), dtype=dtype)
        df_bb = self.ch.read_dataframe(f"{self.dm.model_path}{self.ctx.model_name}_predicted.csv", dtype=dtype)

        # Resolve column names from the selector context so this method remains
        # valid for different datasets/models/selectors.
        target = self.ctx.target_col
        selector_pred_col = self.ctx.selector_prediction_col
        selector_is_abstain_col = self.ctx.selector_is_abstain_col
        bb_pred_col = self.ctx.model_pred_col

        # Output files correspond to the three baselines described in the
        # docstring: full blackbox, blackbox on selector-accepted rows, and
        # confidence-based blackbox at matched coverage.
        path_full = f"{self.dm.selector_path}selector_classification_reports.csv"
        path_accepted = f"{self.dm.selector_path}selector_classification_reports_on_accepted_subset.csv"
        path_matched_coverage = f"{self.dm.selector_path}selector_classification_reports_matched_coverage_bb.csv"

        # Selector performance is defined only on rows where the selector does
        # not abstain. Coverage and abstention rates are computed on all rows.
        accepted_df = df_selector.loc[~df_selector[selector_is_abstain_col]].copy()
        n_total = len(df_selector)
        n_abstain = int(df_selector[selector_is_abstain_col].sum())
        n_non_abstain = len(accepted_df)

        coverage = n_non_abstain / n_total if n_total > 0 else np.nan
        abstain_rate = n_abstain / n_total if n_total > 0 else np.nan

        # Class-specific abstention rates need denominators for each true
        # class, so build class masks before creating the report rows.
        selector_target_num = pd.to_numeric(df_selector[target], errors="coerce")
        mask_0 = selector_target_num == 0
        mask_1 = selector_target_num == 1
        n_true_0 = int(mask_0.sum())
        n_true_1 = int(mask_1.sum())

        # Compute selector classification metrics on accepted rows only.
        y_true_selector = pd.to_numeric(accepted_df[target], errors="coerce")
        y_pred_selector = pd.to_numeric(accepted_df[selector_pred_col], errors="coerce")
        valid_mask_selector = y_true_selector.notna() & y_pred_selector.notna()
        selector_metrics = self.metrics.build_report_metrics(
            y_true_selector.loc[valid_mask_selector],
            y_pred_selector.loc[valid_mask_selector],
        )

        # Baseline 1: blackbox evaluated on the full test set.
        y_true_bb = pd.to_numeric(df_bb[target], errors="coerce")
        y_pred_bb = pd.to_numeric(df_bb[bb_pred_col], errors="coerce")
        valid_mask_bb = y_true_bb.notna() & y_pred_bb.notna()
        bb_metrics_full = self.metrics.build_report_metrics(
            y_true_bb.loc[valid_mask_bb],
            y_pred_bb.loc[valid_mask_bb],
        )

        # Baseline 2: blackbox evaluated on the exact same rows accepted by
        # the selector. This isolates whether the selector's chosen predictions
        # improve over blackbox predictions on the same cases.
        accepted_indices = accepted_df[ind].tolist()
        df_bb_accepted = df_bb[df_bb[ind].isin(accepted_indices)].copy()
        y_true_bb_accepted = pd.to_numeric(df_bb_accepted[target], errors="coerce")
        y_pred_bb_accepted = pd.to_numeric(df_bb_accepted[bb_pred_col], errors="coerce")
        valid_mask_bb_accepted = y_true_bb_accepted.notna() & y_pred_bb_accepted.notna()
        bb_metrics_accepted = self.metrics.build_report_metrics(
            y_true_bb_accepted.loc[valid_mask_bb_accepted],
            y_pred_bb_accepted.loc[valid_mask_bb_accepted],
        )

        # Baseline 3: blackbox evaluated on its highest-confidence rows, with
        # the number of rows matched to the selector coverage. This compares
        # the selector with a simple confidence-based abstention policy.
        df_bb_matched_coverage = self._matched_coverage_blackbox_subset(
            df_selector=df_selector,
            df_bb=df_bb,
            n_non_abstain=n_non_abstain,
        )
        y_true_bb_matched = pd.to_numeric(df_bb_matched_coverage[target], errors="coerce")
        y_pred_bb_matched = pd.to_numeric(df_bb_matched_coverage[bb_pred_col], errors="coerce")
        valid_mask_bb_matched = y_true_bb_matched.notna() & y_pred_bb_matched.notna()
        bb_metrics_matched_coverage = self.metrics.build_report_metrics(
            y_true_bb_matched.loc[valid_mask_bb_matched],
            y_pred_bb_matched.loc[valid_mask_bb_matched],
        )
        matched_coverage = len(df_bb_matched_coverage) / n_total if n_total > 0 else np.nan

        # Sanity checks: matched-coverage baseline must select exactly the same
        # number of rows, hence the same coverage, as the selector.
        if n_non_abstain != len(df_bb_matched_coverage):
            raise ValueError(
                f"Matched-coverage blackbox subset has {len(df_bb_matched_coverage)} rows, "
                f"expected {n_non_abstain}."
            )
        if not pd.isna(coverage) and not np.isclose(matched_coverage, coverage):
            raise ValueError(
                f"Matched-coverage blackbox coverage {matched_coverage} differs from selector coverage {coverage}."
            )

        # Metrics used for percentage-change columns. Support columns are
        # intentionally excluded because percentage changes on supports are not
        # performance measures.
        metrics_for_comparison = [
            "accuracy", "precision_macro", "recall_macro", "f1-score_macro",
            "precision_weighted", "recall_weighted", "f1-score_weighted",
            "precision_0", "recall_0", "f1-score_0",
            "precision_1", "recall_1", "f1-score_1",
        ]

        def save_row(path: str, bb_metrics: dict):
            # Shared metadata and selector abstention statistics are identical
            # across the three reports; only the blackbox baseline changes.
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
                "matched_coverage_bb_n_rows": len(df_bb_matched_coverage) if path == path_matched_coverage else np.nan,
                "matched_coverage_bb_coverage": matched_coverage if path == path_matched_coverage else np.nan,
            }

            # Prefix metric names to keep selector and blackbox values side by
            # side in the same row.
            base_row.update({f"selector_{k}": v for k, v in selector_metrics.items()})
            base_row.update({f"bb_{k}": v for k, v in bb_metrics.items()})

            # Positive pct_change means the selector metric is higher than the
            # selected blackbox baseline for that report.
            base_row.update({
                f"pct_change_{metric}": self.metrics.pct_change(selector_metrics.get(metric), bb_metrics.get(metric))
                for metric in metrics_for_comparison
            })

            # Append/update the relevant global CSV, keeping the latest row for
            # the same dataset/configuration.
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

        # Save the three comparisons described in the method docstring.
        save_row(path_full, bb_metrics_full)
        save_row(path_accepted, bb_metrics_accepted)
        save_row(path_matched_coverage, bb_metrics_matched_coverage)

    def _matched_coverage_blackbox_subset(
        self,
        df_selector: pd.DataFrame,
        df_bb: pd.DataFrame,
        n_non_abstain: int,
    ) -> pd.DataFrame:
        model_conf_col = self.ctx.model_conf_col

        # Match the selector evaluation universe exactly. The selector may be
        # built from inner joins of several files, so use only rows present in
        # the selector dataframe.
        selector_indices = df_selector[ind].tolist()
        df_bb_scope = df_bb[df_bb[ind].isin(selector_indices)].copy()
        if len(df_bb_scope) != len(df_selector):
            missing = set(selector_indices) - set(df_bb_scope[ind].tolist())
            raise ValueError(
                f"Cannot build matched-coverage blackbox baseline: "
                f"{len(missing)} selector rows are missing from blackbox predictions."
            )

        if model_conf_col not in df_bb_scope.columns:
            raise ValueError(f"Missing blackbox confidence column: {model_conf_col}")

        # Deterministic tie-breaking at the confidence cutoff:
        # 1. higher blackbox confidence,
        # 2. smaller original_index,
        # 3. original file order.
        df_bb_scope["_matched_coverage_original_order"] = np.arange(len(df_bb_scope))
        df_bb_scope[model_conf_col] = pd.to_numeric(df_bb_scope[model_conf_col], errors="coerce")
        df_bb_scope[ind] = pd.to_numeric(df_bb_scope[ind], errors="coerce")

        ranked_df = df_bb_scope.sort_values(
            by=[model_conf_col, ind, "_matched_coverage_original_order"],
            ascending=[False, True, True],
            na_position="last",
            kind="mergesort",
        )
        # Keep exactly as many rows as the selector accepted.
        return ranked_df.head(n_non_abstain).drop(columns=["_matched_coverage_original_order"])

    def plot_selector_vs_bb_accepted_subset_dumbbell_per_dataset(self):
        self._plot_selector_vs_bb_dumbbell_per_dataset(
            report_filename="selector_classification_reports_on_accepted_subset.csv",
            output_suffix="accepted_subset",
        )

    def plot_selector_vs_bb_matched_coverage_dumbbell_per_dataset(self):
        self._plot_selector_vs_bb_dumbbell_per_dataset(
            report_filename="selector_classification_reports_matched_coverage_bb.csv",
            output_suffix="matched_coverage",
        )

    def _plot_selector_vs_bb_dumbbell_per_dataset(self, report_filename: str, output_suffix: str):
        axis_label_fontsize = 16
        tick_fontsize = 14

        df = self.ch.read_dataframe(
            f"{self.dm.selector_path}{report_filename}",
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

        out_path = f"{self.dm.selector_path}{self.ctx.dataset_prefix}_selector_vs_bb_{output_suffix}_dumbbell.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
