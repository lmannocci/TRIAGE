


from matplotlib.patches import Patch
import matplotlib.colors as mcolors

from utils.checkpoint.checkpoint import Checkpoint
from utils.directory_manager.directory_manager import DirectoryManager
from utils.common_variables import *
from selector.core.context import SelectorContext
from utils.log_manager.log_manager import LogManager

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


class SelectorPlotHelper:
    def __init__(self, ch: Checkpoint, lm: LogManager, dm: DirectoryManager, ctx: SelectorContext):
        self.ch = ch
        self.lm = lm
        self.dm = dm
        self.ctx = ctx

    @staticmethod
    def _annotated_barplot(ax, labels, values, ylabel: str):
        bars = ax.bar(labels.astype(str), values)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=45)
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    def save_distribution_plot(self, plot_df: pd.DataFrame, x_col: str, y_col: str, out_path: str, ylabel: str):
        fig, ax = plt.subplots(figsize=(14, 6), dpi=dpi)
        self._annotated_barplot(ax, plot_df[x_col], plot_df[y_col], ylabel)
        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    def save_metric_by_group_plot(self, csv_path: str, group_col: str, metric_col: str, out_path: str):
        if not os.path.exists(csv_path):
            return
        plot_df = self.ch.read_dataframe(csv_path, dtype=dtype).copy()
        if metric_col not in plot_df.columns:
            return
        plot_df[metric_col] = pd.to_numeric(plot_df[metric_col], errors="coerce")
        plot_df = plot_df.dropna(subset=[metric_col])
        if plot_df.empty:
            return

        fig, ax = plt.subplots(figsize=(14, 6), dpi=dpi)
        self._annotated_barplot(ax, plot_df[group_col], plot_df[metric_col], metric_col)
        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    def save_histograms(self, df: pd.DataFrame, columns: List[str], out_dir: str):
        for col in columns:
            if col not in df.columns:
                continue
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if s.empty:
                continue

            fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)
            bins = np.linspace(0, 1, 21) if s.min() >= 0 and s.max() <= 1 else 20
            ax.hist(s.to_numpy(), bins=bins)
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            fig.tight_layout()
            safe_col = col.replace("/", "_")
            fig.savefig(f"{out_dir}{safe_col}_hist.png", dpi=dpi, bbox_inches="tight")
            plt.close(fig)
    
    def save_kde_plots(self, df: pd.DataFrame, columns: List[str], out_dir: str):
        for col in columns:
            if col not in df.columns:
                continue

            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if s.empty:
                continue

            fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)

            sns.kdeplot(
                s,
                fill=True,
                color="blue",
                ax=ax
            )

            ax.set_xlabel(col)
            ax.set_ylabel("Density")

            # constrain if confidence-like variable
            if s.min() >= 0 and s.max() <= 1:
                ax.set_xlim(0, 1)

            fig.tight_layout()

            safe_col = col.replace("/", "_")
            fig.savefig(f"{out_dir}{safe_col}_kde.png", dpi=dpi, bbox_inches="tight")

            plt.close(fig)
    
    def save_distribution_plots(self, df, columns, out_dir):
        for col in columns:
            if col not in df.columns:
                continue

            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if s.empty:
                continue

            fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)

            # histogram
            bins = np.linspace(0, 1, 21) if s.min() >= 0 and s.max() <= 1 else 20
            ax.hist(s, bins=bins, alpha=0.3, label="hist")

            # kde
            sns.kdeplot(s, fill=True, color="blue", ax=ax, label="kde")

            ax.set_xlabel(col)
            ax.set_ylabel("Density")

            if s.min() >= 0 and s.max() <= 1:
                ax.set_xlim(0, 1)

            ax.legend()
            fig.tight_layout()

            safe_col = col.replace("/", "_")
            fig.savefig(f"{out_dir}{safe_col}_dist.png", dpi=dpi, bbox_inches="tight")
            plt.close(fig)

    def plot_selector_case_scatter(self) -> None:
        """
        Build a scatterplot analysis for selector logging cases.

        x-axis:
            percentage of rows in each logging case (in [0, 1])

        y-axis:
            one of:
            - f1-score_1
            - f1-score_0
            - f1-score_macro

        color:
            classification output category:
            - agree
            - disagree_model
            - disagree_enhancer
            - abstain branches are omitted from the scatter legend because
              they have no non-abstained predictions from which F1 is computed.

        marker:
            confidence case:
            - both_high
            - both_low
            - model_high_enhancer_low
            - model_low_enhancer_high

        Output:
            one figure with 3 subplots side by side
            + one separate legend figure
        """

        case_col = self.ctx.selector_case_col

        path_dist = f"{self.dm.en_selector_csv_path}logging_case_distribution.csv"
        path_f1 = f"{self.dm.en_selector_csv_path}f1_report_by_logging_case.csv"
        out_path = f"{self.dm.global_evaluator_path}{self.ctx.dataset_prefix}_selector_case_scatter_f1.png"
        out_legend_path = f"{self.dm.global_evaluator_path}selector_case_scatter_f1_legend.png"

        if not os.path.exists(path_dist) or not os.path.exists(path_f1):
            self.lm.printl("[Warning] Missing logging-case distribution or F1 report file.")
            return

        dist_df = self.ch.read_dataframe(path_dist, dtype=dtype).copy()
        f1_df = self.ch.read_dataframe(path_f1, dtype=dtype).copy()

        if "abstain_logging_case" in dist_df.columns and case_col not in dist_df.columns:
            dist_df = dist_df.rename(columns={"abstain_logging_case": case_col})

        if "percentage" not in dist_df.columns:
            self.lm.printl("[Warning] Column 'percentage' not found in logging case distribution.")
            return

        dist_df["percentage"] = pd.to_numeric(dist_df["percentage"], errors="coerce") / 100.0
        f1_df["f1-score_1"] = pd.to_numeric(f1_df["f1-score_1"], errors="coerce")
        f1_df["f1-score_0"] = pd.to_numeric(f1_df["f1-score_0"], errors="coerce")
        f1_df["f1-score_macro"] = pd.to_numeric(f1_df["f1-score_macro"], errors="coerce")

        plot_df = dist_df.merge(f1_df, on=case_col, how="inner")

        def _map_output_type(logging_case: str) -> str:
            if pd.isna(logging_case):
                return "unknown"

            case = str(logging_case)
            if case.startswith("[ABSTAIN]"):
                return "abstain"
            if case.startswith("[MODEL_Y]"):
                return "disagree_model"
            if case.startswith("[ENHANCER_Y]"):
                return "disagree_enhancer"
            if case.startswith("[Y]"):
                return "agree"
            return "unknown"

        def _map_confidence_case(logging_case: str) -> str:
            if pd.isna(logging_case):
                return "unknown"

            case = str(logging_case)

            if "both_high" in case:
                return "both_high"
            if "both_low" in case:
                return "both_low"
            if "model_high_enhancer_low" in case:
                return "model_high_enhancer_low"
            if "model_low_enhancer_high" in case:
                return "model_low_enhancer_high"
            if case in ["[Y]agree", "[ABSTAIN]disagree"]:
                return "not_applicable"
            return "unknown"

        plot_df["output_type"] = plot_df[case_col].apply(_map_output_type)
        plot_df["confidence_case"] = plot_df[case_col].apply(_map_confidence_case)

        color_map = output_color_dict

        # Output type labels (colors)
        output_label_map = {
            "agree": "agree",
            "disagree_model": "disagree (model)",
            "disagree_enhancer": "disagree (enhancer)",
            "abstain": "abstain",
            "unknown": "unknown",
        }

        # Confidence case labels (markers)
        confidence_label_map = {
            "both_high": "both high",
            "both_low": "both low",
            "model_high_enhancer_low": "model high, enhancer low",
            "model_low_enhancer_high": "model low, enhancer high",
            "not_applicable": "n/a",
            "unknown": "unknown",
        }

        marker_map = {
            "both_high": "o",
            "both_low": "s",
            "model_high_enhancer_low": "^",
            "model_low_enhancer_high": "D",
            "not_applicable": "P",
            "unknown": "X",
        }

        subplot_info = [
            ("f1-score_1", "f1-score class 1"),
            ("f1-score_0", "f1-score class 0"),
            ("f1-score_macro", "f1-score macro"),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=dpi, sharex=True, sharey=True)

        title_fontsize = 18
        axis_label_fontsize = 18
        tick_fontsize = 14

        plotted_output_types = set()
        plotted_confidence_cases = set()

        for ax, (metric_col, title) in zip(axes, subplot_info):
            sub_df = plot_df.dropna(subset=["percentage", metric_col]).copy()

            for output_type in sub_df["output_type"].dropna().unique():
                for confidence_case in sub_df["confidence_case"].dropna().unique():
                    tmp = sub_df[
                        (sub_df["output_type"] == output_type) &
                        (sub_df["confidence_case"] == confidence_case)
                    ]
                    if tmp.empty:
                        continue

                    point_color = color_map.get(output_type, output_color_dict["unknown"])
                    ax.scatter(
                        tmp["percentage"],
                        tmp[metric_col],
                        c=[point_color],
                        marker=marker_map.get(confidence_case, "X"),
                        s=300,
                        # alpha=0.9,
                        edgecolors=[self._darken_color(point_color)],
                        linewidths=1.5,
                    )
                    plotted_output_types.add(output_type)
                    plotted_confidence_cases.add(confidence_case)

            ax.set_title(title, fontsize=title_fontsize)
            ax.set_xlabel("percentage of rows in case", fontsize=axis_label_fontsize)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(axis="both", linestyle="--", linewidth=0.5, alpha=0.5)
            ax.tick_params(axis="x", labelsize=tick_fontsize)
            ax.tick_params(axis="y", labelsize=tick_fontsize)

        axes[0].set_ylabel("f1-score", fontsize=axis_label_fontsize)

        fig.tight_layout()
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        self.lm.printl(f"Saved selector case scatter plot to {out_path}")

        # ---------------------------------------------------------
        # Separate legend figure
        # ---------------------------------------------------------
        
        output_handles = [
            plt.Line2D(
                [0], [0],
                color=color_map[label],
                lw=6,
                label=output_label_map.get(label, label),
            )
            for label in ["agree", "disagree_model", "disagree_enhancer"]
            if label in plotted_output_types
        ]

        confidence_handles = [
            plt.Line2D(
                [0], [0],
                marker=marker,
                color="black",
                markersize=10,
                linestyle="None",
                label=confidence_label_map.get(label, label),
            )
            for label, marker in marker_map.items()
            if label in plotted_confidence_cases
        ]

        legend_fig = plt.figure(figsize=(8, 1.4), dpi=dpi)
        legend_ax = legend_fig.add_subplot(111)

        legend1 = legend_ax.legend(
            handles=output_handles,
            # title="Output type",
            loc="upper center",
            bbox_to_anchor=(0.5, 0.95),
            frameon=False,
            ncol=len(output_handles),
            fontsize=14,
            title_fontsize=15,
        )
        legend_ax.add_artist(legend1)

        legend2 = legend_ax.legend(
            handles=confidence_handles,
            # title="Confidence case",
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            frameon=False,
            ncol=len(confidence_handles),
            fontsize=14,
            title_fontsize=15,
        )

        legend_ax.axis("off")
        legend_fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        plt.tight_layout()
        plt.savefig(out_legend_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
        plt.close(legend_fig)

    @staticmethod
    def _darken_color(color, factor: float = 0.65):
        rgb = np.array(mcolors.to_rgb(color))
        return tuple(np.clip(rgb * factor, 0, 1))

    # def plot_explanation_usage(self, df: pd.DataFrame, out_path: str):
    #     """
    #     Plot percentage of combinations of:
    #     - use_enhanced_explanation
    #     - use_enhancer_explanation
    #     """
    #     axis_label_fontsize = 16
    #     tick_fontsize = 14

    #     col_enhanced = f"{self.ctx.selector_name}_use_enhanced_explanation"
    #     col_enhancer = f"{self.ctx.selector_name}_use_enhancer_explanation"

    #     if col_enhanced not in df.columns or col_enhancer not in df.columns:
    #         return

    #     # Ensure boolean
    #     df[col_enhanced] = df[col_enhanced].astype(bool)
    #     df[col_enhancer] = df[col_enhancer].astype(bool)

    #     # Create combination labels
    #     def _map_case(row):
    #         if row[col_enhanced] and row[col_enhancer]:
    #             return "both"
    #         elif row[col_enhanced] and not row[col_enhancer]:
    #             return "explainer"
    #         elif not row[col_enhanced] and row[col_enhancer]:
    #             return "enhancer"
    #         else:
    #             return "abstain"

    #     df["explanation_case"] = df.apply(_map_case, axis=1)

    #     # Compute percentages
    #     counts = df["explanation_case"].value_counts()
    #     percentages = counts / len(df) * 100

    #     # Keep fixed order
    #     order = ["both", "explainer", "enhancer", "abstain"]
    #     percentages = percentages.reindex(order).fillna(0)

    #     # Plot
    #     fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)

    #     sns.barplot(
    #         x=percentages.index,
    #         y=percentages.values,
    #         palette="viridis",
    #         ax=ax
    #     )

    #     # Tick labels
    #     ax.set_xticklabels(
    #         ax.get_xticklabels(),
    #         rotation=30,
    #         ha="right",
    #         fontsize=tick_fontsize
    #     )

    #     # Axis labels
    #     ax.set_xlabel("", fontsize=axis_label_fontsize)
    #     ax.set_ylabel("percentage (%)", fontsize=axis_label_fontsize)

    #     # Tick size (IMPORTANT for y-axis)
    #     ax.tick_params(axis="y", labelsize=tick_fontsize)

    #     # Limits
    #     max_value = percentages.max()
    #     ax.set_ylim(0, max_value + 5)

    #     # Add values
    #     for i, value in enumerate(percentages.values):
    #         ax.text(
    #             i,
    #             value + 0.5,
    #             f"{value:.1f}%",
    #             ha="center",
    #             va="bottom",
    #             fontsize=16
    #         )

    #     fig.tight_layout()
    #     fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    #     plt.close(fig)
   
