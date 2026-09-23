import ast
import os
from typing import Iterable, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rbo import rbo

from utils.checkpoint.checkpoint import Checkpoint
from utils.common_variables import av_datasets, dataset_color_dict, df_info, dpi, dtype, ind
from utils.decorator_definition import log_method
from utils.directory_manager.directory_manager import DirectoryManager
from utils.integrity_constraint_manager.integrity_constraint_manager import IntegrityConstraintManager
from utils.log_manager.log_manager import LogManager


absolute_path = os.path.dirname(__file__)
results_path = os.path.join(absolute_path, f"..{os.sep}results{os.sep}")


class ExplanationTopKSensitivityAnalysis:
    """
    Evaluate explanation-ranking agreement while varying the top-k parameter.

    The class is intended for the final black-box, explainer, and enhancer
    configuration. It computes RBO directly from the saved full feature
    rankings, avoiding repeated calls to Evaluator.evaluate_explanations().
    """

    def __init__(
        self,
        ch: Checkpoint,
        lm: LogManager,
        dataset_prefix: str,
        model_name: str,
        model_prefix: str,
        explainer_name: str,
        enhancer_name: str,
        llm_name: str,
        rag: bool,
        retriever_name: str,
        corpus_name: str,
        top_k_values: Optional[Union[Iterable[int], str]] = None,
        metric_exp_ranking: str = "rbo",
    ):
        self.ch = ch
        self.lm = lm
        self.icm = IntegrityConstraintManager(lm)

        self.icm.check_dataset(dataset_prefix)
        self.icm.check_model(model_name)
        self.icm.check_explainer(explainer_name)
        self.icm.check_model_explainer(model_name, explainer_name)
        self.icm.check_enhancer(enhancer_name)
        self.icm.check_llm(llm_name)
        self.icm.check_rag(rag)
        self.icm.check_retriever(retriever_name)
        self.icm.check_corpus(corpus_name)
        self.icm.check_metric_exp_ranking(metric_exp_ranking)

        self.dataset_prefix = dataset_prefix
        self.model_name = model_name
        self.model_prefix = model_prefix
        self.explainer_name = explainer_name
        self.enhancer_name = enhancer_name
        self.llm_name = llm_name
        self.rag = rag
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        self.metric_exp_ranking = metric_exp_ranking
        self.info = df_info[dataset_prefix]
        self.complete_enhancer_name = (
            f"{self.enhancer_name}_{self.llm_name}_{str(self.rag)}_"
            f"{self.retriever_name}_{self.corpus_name}"
        )
        self.top_k_values = self._normalize_top_k_values(top_k_values)

        self.dm = DirectoryManager(
            lm,
            results_path,
            dataset_prefix=dataset_prefix,
            model_name=model_name,
            model_prefix=model_prefix,
            explainer_name=explainer_name,
            enhancer_name=enhancer_name,
            llm_name=llm_name,
            rag=rag,
            retriever_name=retriever_name,
            corpus_name=corpus_name,
            metric_exp_ranking=metric_exp_ranking,
            top_k=None,
        )

    @log_method
    def run(self) -> pd.DataFrame:
        if self.metric_exp_ranking != "rbo":
            raise NotImplementedError("Only RBO top-k sensitivity is currently implemented.")

        df = self._load_rankings()
        rows = []
        for top_k in self.top_k_values:
            row_scores = df.apply(lambda row: self._compute_row_rbo(row, top_k), axis=1)
            scores = pd.to_numeric(row_scores.apply(lambda item: item["score"]), errors="coerce")
            effective_ks = pd.to_numeric(row_scores.apply(lambda item: item["effective_k"]), errors="coerce")

            valid_scores = scores.dropna()
            valid_effective_ks = effective_ks.dropna()
            rows.append({
                "dataset_prefix": self.dataset_prefix,
                "model_name": self.model_name,
                "model_prefix": self.model_prefix,
                "explainer_name": self.explainer_name,
                "enhancer_name": self.enhancer_name,
                "llm_name": self.llm_name,
                "rag": self.rag,
                "retriever_name": self.retriever_name,
                "corpus_name": self.corpus_name,
                "metric_exp_ranking": self.metric_exp_ranking,
                "top_k": top_k,
                "n_total": len(df),
                "n_valid": int(valid_scores.shape[0]),
                "n_missing": int(scores.isna().sum()),
                "n_effective_k_less_than_top_k": int((effective_ks < top_k).sum()),
                "effective_k_mean": valid_effective_ks.mean(),
                "effective_k_std": valid_effective_ks.std(),
                "effective_k_min": valid_effective_ks.min(),
                "effective_k_max": valid_effective_ks.max(),
                "rbo_mean": valid_scores.mean(),
                "rbo_std": valid_scores.std(),
                "rbo_median": valid_scores.median(),
                "rbo_min": valid_scores.min(),
                "rbo_max": valid_scores.max(),
                "rbo_q25": valid_scores.quantile(0.25),
                "rbo_q75": valid_scores.quantile(0.75),
            })

        out_df = pd.DataFrame(rows)
        self._save_report(out_df)
        self._write_output_readme()
        return out_df

    @log_method
    def compute_random_rbo_baseline(
        self,
        n_simulations: int = 10000,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Estimate chance-level RBO for the dataset feature space.

        For each requested top-k, two independent ordered samples of unique
        features are drawn from the dataset feature set. The rankings may
        overlap with each other, but features are unique within each ranking.
        """
        if self.metric_exp_ranking != "rbo":
            raise NotImplementedError("Only RBO random baseline is currently implemented.")

        n_features = len(self.info["x_columns"])
        rng = np.random.default_rng(random_state)
        rows = []
        for top_k in self.top_k_values:
            if top_k > n_features:
                self.lm.printl(
                    f"[Warning] Skipping random RBO baseline for {self.dataset_prefix} "
                    f"top_k={top_k}, because n_features={n_features}."
                )
                continue

            values = self._simulate_random_rbo_values(
                n_features=n_features,
                top_k=top_k,
                n_simulations=n_simulations,
                rng=rng,
            )
            rows.append({
                "dataset_prefix": self.dataset_prefix,
                "n_features": n_features,
                "top_k": top_k,
                "n_simulations": int(n_simulations),
                "random_state": int(random_state),
                "metric_exp_ranking": self.metric_exp_ranking,
                "random_rbo_mean": float(np.mean(values)),
                "random_rbo_std": float(np.std(values, ddof=1)),
                "random_rbo_ci_lower": float(np.percentile(values, 2.5)),
                "random_rbo_ci_upper": float(np.percentile(values, 97.5)),
            })

        out_df = pd.DataFrame(rows)
        self._save_random_baseline_report(out_df)
        self._write_output_readme()
        return out_df

    @log_method
    def plot(self, max_top_k: Optional[int] = None) -> None:
        path = f"{self.dm.explanation_topk_sensitivity_path}explanation_topk_sensitivity_summary.csv"
        if not os.path.exists(path):
            self.lm.printl("[Warning] Missing top-k sensitivity CSV. Run ExplanationTopKSensitivityAnalysis.run first.")
            return

        df = self.ch.read_dataframe(path, dtype=dtype).copy()
        for column in ["top_k", "rbo_mean", "rbo_std"]:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        if max_top_k is not None:
            df = df[df["top_k"] <= int(max_top_k)].copy()
        if df.empty:
            self.lm.printl("[Warning] No top-k sensitivity rows available for the selected plotting range.")
            return

        fig, ax = plt.subplots(figsize=(7, 3.2), dpi=dpi)
        present_datasets = [dataset for dataset in av_datasets if dataset in df["dataset_prefix"].astype(str).values]

        for dataset in present_datasets:
            plot_df = df[df["dataset_prefix"].astype(str).eq(dataset)].copy()
            plot_df = plot_df.sort_values("top_k")
            x = plot_df["top_k"].to_numpy(dtype=float)
            mean = plot_df["rbo_mean"].to_numpy(dtype=float)
            std = plot_df["rbo_std"].fillna(0).to_numpy(dtype=float)
            color = dataset_color_dict[dataset]

            ax.plot(
                x,
                mean,
                marker="o",
                linewidth=2.5,
                color=color,
                label=dataset,
            )
            ax.fill_between(
                x,
                np.clip(mean - std, 0, 1),
                np.clip(mean + std, 0, 1),
                color=color,
                alpha=0.22,
                linewidth=0,
            )

        top_k_values = sorted(pd.to_numeric(df["top_k"], errors="coerce").dropna().unique())
        ax.set_xlabel("top-k", fontsize=14)
        ax.set_ylabel("rbo", fontsize=14)
        ax.set_ylim(0, 1.02)
        ax.set_xticks(top_k_values)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.7)

        fig.tight_layout()
        suffix = f"_top_{int(max_top_k)}" if max_top_k is not None else ""
        out_path = f"{self.dm.explanation_topk_sensitivity_plot_path}explanation_topk_sensitivity_rbo{suffix}.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        self._save_plot_legend(present_datasets)

    def _load_rankings(self) -> pd.DataFrame:
        enhancer_df = self.ch.read_dataframe(
            f"{self.dm.enhancer_path}{self.enhancer_name}_cleaned_df.csv",
            dtype=dtype,
        )
        explainer_df = self.ch.read_dataframe(
            f"{self.dm.explainer_path}{self.explainer_name}_df.csv",
            dtype=dtype,
        )

        c_explainer = f"{self.explainer_name}_feature_importance_ranking"
        c_enhancer = "feature_importance_ranking_clean"
        df = explainer_df[[ind, c_explainer]].merge(
            enhancer_df[[ind, c_enhancer]],
            on=ind,
            how="inner",
        )
        df[c_explainer] = df[c_explainer].apply(self._safe_parse_ranking)
        df[c_enhancer] = df[c_enhancer].apply(self._safe_parse_ranking)
        return df.rename(columns={c_explainer: "explainer_ranking", c_enhancer: "enhancer_ranking"})

    def _normalize_top_k_values(self, top_k_values: Optional[Union[Iterable[int], str]]) -> List[int]:
        if isinstance(top_k_values, str):
            if top_k_values not in {"dataset_max", "all_features", "auto"}:
                raise ValueError(
                    "top_k_values as a string must be one of: 'dataset_max', 'all_features', or 'auto'."
                )
            values = range(1, len(self.info["x_columns"]) + 1)
        else:
            values = list(top_k_values or range(1, 11))

        values = sorted({int(value) for value in values if int(value) > 0})
        if not values:
            raise ValueError("top_k_values must contain at least one positive integer.")
        return values

    @staticmethod
    def _safe_parse_ranking(value):
        if value is None:
            return np.nan
        if isinstance(value, float) and pd.isna(value):
            return np.nan
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return np.nan
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    return parsed
            except (ValueError, SyntaxError):
                return np.nan
        return np.nan

    def _compute_row_rbo(self, row: pd.Series, top_k: int) -> dict:
        explainer_rank = row["explainer_ranking"]
        enhancer_rank = row["enhancer_ranking"]

        if not isinstance(explainer_rank, list) or not isinstance(enhancer_rank, list):
            return {"score": np.nan, "effective_k": np.nan}

        effective_k = min(top_k, len(explainer_rank), len(enhancer_rank))
        if effective_k <= 0:
            return {"score": np.nan, "effective_k": np.nan}

        similarity = rbo.RankingSimilarity(
            explainer_rank[:effective_k],
            enhancer_rank[:effective_k],
        )
        return {"score": similarity.rbo(), "effective_k": effective_k}

    @staticmethod
    def _simulate_random_rbo_values(
        n_features: int,
        top_k: int,
        n_simulations: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        features = np.arange(n_features)
        values = np.empty(n_simulations, dtype=float)
        for idx in range(n_simulations):
            ranking_a = rng.choice(features, size=top_k, replace=False).tolist()
            ranking_b = rng.choice(features, size=top_k, replace=False).tolist()
            values[idx] = rbo.RankingSimilarity(ranking_a, ranking_b).rbo()
        return values

    def _save_report(self, new_df: pd.DataFrame) -> None:
        path = f"{self.dm.explanation_topk_sensitivity_path}explanation_topk_sensitivity_summary.csv"
        if os.path.exists(path):
            old_df = self.ch.read_dataframe(path, dtype=dtype)
            out_df = pd.concat([old_df, new_df], ignore_index=True)
        else:
            out_df = new_df

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
            "metric_exp_ranking",
            "top_k",
        ]
        out_df = out_df.drop_duplicates(subset=key_cols, keep="last")
        out_df["dataset_prefix"] = pd.Categorical(out_df["dataset_prefix"], categories=av_datasets, ordered=True)
        out_df["top_k"] = pd.to_numeric(out_df["top_k"], errors="coerce")
        out_df = out_df.sort_values(["dataset_prefix", "top_k"])
        out_df["dataset_prefix"] = out_df["dataset_prefix"].astype(str)
        self.ch.save_dataframe(out_df, path)

    def _save_random_baseline_report(self, new_df: pd.DataFrame) -> None:
        path = f"{self.dm.explanation_topk_sensitivity_path}random_rbo_baseline_summary.csv"
        if os.path.exists(path):
            old_df = self.ch.read_dataframe(path, dtype=dtype)
            out_df = pd.concat([old_df, new_df], ignore_index=True)
        else:
            out_df = new_df

        key_cols = [
            "dataset_prefix",
            "n_features",
            "top_k",
            "n_simulations",
            "random_state",
            "metric_exp_ranking",
        ]
        out_df = out_df.drop_duplicates(subset=key_cols, keep="last")
        out_df["dataset_prefix"] = pd.Categorical(out_df["dataset_prefix"], categories=av_datasets, ordered=True)
        out_df["top_k"] = pd.to_numeric(out_df["top_k"], errors="coerce")
        out_df = out_df.sort_values(["dataset_prefix", "top_k"])
        out_df["dataset_prefix"] = out_df["dataset_prefix"].astype(str)
        self.ch.save_dataframe(out_df, path)

    def _save_plot_legend(self, present_datasets: List[str]) -> None:
        legend_handles = [
            plt.Line2D(
                [0],
                [0],
                color=dataset_color_dict[dataset],
                marker="o",
                lw=3,
                label=dataset,
            )
            for dataset in present_datasets
        ]
        if not legend_handles:
            return

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

        legend_path = f"{self.dm.explanation_topk_sensitivity_plot_path}explanation_topk_sensitivity_rbo_legend.png"
        legend_fig.savefig(legend_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
        plt.close(legend_fig)

    def _write_output_readme(self) -> None:
        path = f"{self.dm.explanation_topk_sensitivity_path}README.md"
        text = """# Explanation Top-K Sensitivity

This directory contains the sensitivity analysis of explanation-ranking agreement when varying `top_k`.

## Files

- `explanation_topk_sensitivity_summary.csv`
  - One row per dataset, final configuration, metric, and requested `top_k`.
  - `rbo_mean`, `rbo_std`, `rbo_median`, `rbo_min`, `rbo_max`, `rbo_q25`, and `rbo_q75` summarize row-level RBO values.
  - `effective_k_*` columns summarize the actual ranking length used when either the explainer or enhancer ranking is shorter than the requested `top_k`.
  - `n_effective_k_less_than_top_k` counts rows where the requested `top_k` could not be fully used.

- `random_rbo_baseline_summary.csv`
  - Chance-level RBO baseline for each dataset and `top_k`.
  - Two independent random ordered feature rankings are sampled `n_simulations` times.
  - `random_rbo_ci_lower` and `random_rbo_ci_upper` are the empirical 2.5th and 97.5th percentiles.

## Plots

- `plot/explanation_topk_sensitivity_rbo.png`
  - Line chart of mean RBO by `top_k`, with one line per dataset and shaded standard deviation.
  - Calling `plot(max_top_k=N)` saves `plot/explanation_topk_sensitivity_rbo_top_N.png` and restricts the x-axis to `top_k <= N`.

- `plot/explanation_topk_sensitivity_rbo_legend.png`
  - Shared dataset legend.

Use `top_k_values="dataset_max"` to evaluate every value from 1 to the number of dataset features.
"""
        with open(path, "w", encoding="utf-8") as file:
            file.write(text)
