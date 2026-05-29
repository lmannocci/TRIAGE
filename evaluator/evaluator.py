from os import path

import numpy as np
import pandas as pd
import os
import ast
import re
from rbo import rbo
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from utils.checkpoint.checkpoint import *
from utils.directory_manager.directory_manager import DirectoryManager
from utils.log_manager.log_manager import *
from utils.decorator_definition import *
from utils.common_variables import *
from utils.integrity_constraint_manager.integrity_constraint_manager import IntegrityConstraintManager
from typing import List, Tuple, Dict, Union, Optional

absolute_path = os.path.dirname(__file__)
# file_name = os.path.splitext(os.path.basename(__file__))[0]
results_path = os.path.join(absolute_path, f"..{os.sep}results{os.sep}")
data_path = os.path.join(absolute_path, f"..{os.sep}data{os.sep}")


class Evaluator:
    def __init__(self, ch: Checkpoint, lm: LogManager, dataset_prefix: str, 
                 model_name: Optional[str] = None, model_prefix: Optional[str] = None,
                 enhancer_name: Optional[str] = None, llm_name: Optional[str] = None, rag: Optional[bool] = None,
                 top_k: Optional[int] = None, metric_exp_ranking:Optional[str] = 'rbo',
                 explainer_name: Optional[str] = None,
                 retriever_name: Optional[str] = 'MedCPT', corpus_name: Optional[str] = 'StatPearls'):
        self.en_model = None
        self.ch: Checkpoint = ch
        self.lm: LogManager = lm

        self.icm: IntegrityConstraintManager = IntegrityConstraintManager(lm)
        self.icm.check_dataset(dataset_prefix)
        if enhancer_name is not None:
            self.icm.check_enhancer(enhancer_name)
            self.icm.check_llm(llm_name)
            self.icm.check_rag(rag)
            self.icm.check_retriever(retriever_name)
            self.icm.check_corpus(corpus_name)
        if model_name is not None and model_prefix is not None:
            self.icm.check_model(model_name)
        if explainer_name is not None:
            self.icm.check_explainer(explainer_name)
            self.icm.check_model_explainer(model_name, explainer_name)
        if top_k is not None:
            self.icm.check_top_k(top_k)
        if metric_exp_ranking is not None:
            self.icm.check_metric_exp_ranking(metric_exp_ranking)

        self.dm: DirectoryManager = DirectoryManager(lm, results_path, dataset_prefix=dataset_prefix, model_name=model_name, model_prefix=model_prefix, 
                                                     explainer_name=explainer_name,enhancer_name=enhancer_name, llm_name=llm_name, rag=rag, 
                                                     retriever_name=retriever_name, corpus_name=corpus_name, metric_exp_ranking=metric_exp_ranking, top_k=top_k)

        self.dataset_prefix: str = dataset_prefix
        self.dataset: str = f"{self.dataset_prefix}_preprocessed.csv"
        self.info: Dict[str, Union[str, List[str]]] = df_info[self.dataset_prefix]

        self.model_name: str = model_name
        self.model_prefix: str = model_prefix
        self.explainer_name: str = explainer_name
        self.enhancer_name: str = enhancer_name

        self.llm_name: str = llm_name
        self.rag: bool = rag
        self.retriever_name: str = retriever_name
        self.corpus_name: str = corpus_name
        self.complete_enhancer_name: str = f"{self.enhancer_name}_{self.llm_name}_{str(self.rag)}_{self.retriever_name}_{self.corpus_name}"

        self.top_k: int = top_k
        self.metric_exp_ranking: str = metric_exp_ranking
        self.metric_complete_name = f"{self.metric_exp_ranking}_{str(self.top_k)}"
    

    # Define a function to extract feature names from the rule
    def __extract_lime_feature(self, rule):
        # Regular expression to match the feature name (before the comparison operator)
    #     feature_name = re.match(r'(?:[^\w\s]*\s*)([a-zA-Z0-9_]+)(?=\s*[<>=!])', rule)
        # Regex pattern to match the feature and number
        # pattern = r"([a-zA-Z]+)\s*(<=?|>=?)\s*(-?\d+(\.\d+)?)" # it does not work with features like residence_type, which contains underscore
        pattern = r"([a-zA-Z_][a-zA-Z0-9_-]*)\s*(<=?|>=?)\s*(-?\d+(?:\.\d+)?)"
        match = re.search(pattern, rule)
        if match:
            feature = match.group(1)
            operator = match.group(2)
            number = match.group(3)
            return feature
        return None

    # Define the function to extract the top k feature names from the list in the lime_exp_list column
    def __get_lime_top_k_features(self, lime_exp_list: str):
        # Convert the string representation of the list into an actual list
        feature_list = ast.literal_eval(lime_exp_list)
        
        # Check if k is greater than the number of available features
        if self.top_k > len(feature_list):
            raise ValueError(f"k={self.top_k} is greater than the number of features in the explanation list ({len(feature_list)}).")
        
        # Extract the top k features (sorted by importance)
        top_k_features = [self.__extract_lime_feature(feature) for feature, importance in feature_list[:self.top_k]]
        
        return top_k_features
    
    def __get_top_k_features(self, exp_dict):
        """
        Extracts the top-k features from an explanation dictionary,
        sorted by absolute importance value.

        Parameters:
            exp_dict (dict or str): A dictionary or a stringified dictionary of feature: importance pairs.

        Returns:
            List[str]: Top-k feature names sorted by absolute importance.
        """
        if isinstance(exp_dict, str):
            try:
                exp_dict = ast.literal_eval(exp_dict)
            except Exception as e:
                raise ValueError(f"Failed to parse explanation string: {e}")

        if not isinstance(exp_dict, dict):
            raise TypeError(f"Expected dict, got {type(exp_dict)}")

        return [
            feature for feature, _ in sorted(
                exp_dict.items(),
                key=lambda item: abs(item[1]),
                reverse=True
            )[:self.top_k]
        ]

    # Function to extract the top k features from the string representation of the list
    # def __select_topk_features_from_str(self, ranking_str):
    #     # Check if the value is NaN
    #     if pd.isna(ranking_str):
    #         return np.nan
    #     # Convert the string representation of the list into an actual list
    #     feature_list = eval(ranking_str)
    #     # Extract the top k features
    #     return feature_list[:self.top_k]

    def __extract_topk_explainer_features(self, df: pd.DataFrame):
        method_map = {
            'lime': self.__get_lime_top_k_features,
            'shap': self.__get_top_k_features,
            'dalex': self.__get_top_k_features,
            'ebm': self.__get_top_k_features,
        }

        if self.explainer_name in method_map:
            top_k_col = f'{self.explainer_name}_top_{self.top_k}'
            exp_list_col = f'{self.explainer_name}_exp_list'
            df[top_k_col] = df[exp_list_col].apply(method_map[self.explainer_name])
        
        return df
    
    # def __extract_topk_enhancer_features(self, df: pd.DataFrame):
    #     c = f'{self.enhancer_name}_top_{str(self.top_k)}'
    #     if c not in df.columns:
    #         df[c] = df['feature_importance_ranking'].apply(lambda x: self.__select_topk_features_from_str(x))
    #     return df
    
    def __safe_parse_ranking(self, x):
        """
        Convert a CSV-loaded ranking column into a Python list.
        Handles:
        - actual lists
        - stringified lists like "['a', 'b', 'c']"
        - nulls
        """
        if x is None:
            return np.nan

        if isinstance(x, float) and pd.isna(x):
            return np.nan

        if isinstance(x, list):
            return x

        if isinstance(x, str):
            x = x.strip()
            if not x:
                return np.nan
            try:
                parsed = ast.literal_eval(x)
                if isinstance(parsed, list):
                    return parsed
                return np.nan
            except (ValueError, SyntaxError):
                return np.nan

        return np.nan

    # Compute RBO for each row and create a new column 'rbo_k'
    def __compute_rbo(self, row, c_explainer, c_enhancer):
        explainer_rank = row[c_explainer]
        enhancer_rank = row[c_enhancer]

        if not isinstance(explainer_rank, list) or not isinstance(enhancer_rank, list):
            return np.nan

        if len(explainer_rank) == 0 or len(enhancer_rank) == 0:
            return np.nan

        a = explainer_rank[:self.top_k]
        b = enhancer_rank[:self.top_k]

        similarity = rbo.RankingSimilarity(a, b)
        return similarity.rbo()

    def __compute_similarity_exp(self, df: pd.DataFrame, c_explainer: str, c_enhancer: str) -> pd.DataFrame:
        if self.metric_exp_ranking == 'rbo':
            # Compute RBO for each row and create a new column 'rbo_k'
            df[self.metric_complete_name] = df.apply(lambda row: self.__compute_rbo(row, c_explainer, c_enhancer), axis=1)
        return df
    
    def __selector_simple(self, row):
        """
        Simple decision system:
        - agree → accept model prediction
        - disagree → abstain

        Output schema is identical to __ds_abstain
        """

        agree = row["agree"]

        model_pred = row[f"{self.model_name}_pred"]
        enhancer_pred = row[f"{self.enhancer_name}_pred"]

        if agree:
            return {
                "decision": "ok",
                "prediction": model_pred,
                "use_enhanced_explanation": True,
                "use_enhancer_explanation": True,
            }
        else:
            return {
                "decision": "abstain",
                "prediction": None,
                "use_enhanced_explanation": False,
                "use_enhancer_explanation": False,
            }


    def __selector_abstain(self, row):
        """
        Selector with abstention using breakdown & agreement columns.
        """

        model_pred = row[f"{self.model_name}_pred"]
        enhancer_pred = row[f"{self.enhancer_name}_pred"]

        model_conf = row[f"{self.model_name}_conf"]
        enhancer_conf = row[f"{self.enhancer_name}_conf"]

        rbo = row[self.metric_complete_name]

        # --- precomputed regions
        agree = row["agree"]

        # model_better = row["model_correct_enhancer_wrong"]
        # enhancer_better = row["enhancer_correct_model_wrong"]
        # both_correct = row["both_correct"]
        # both_wrong = row["both_wrong"]

        # --- thresholds
        model_high = model_conf >= self.th_model_conf
        enhancer_high = enhancer_conf >= self.th_enhancer_conf
        rbo_high = rbo >= self.th_rbo

        # =========================================================
        # CASE 1: model and enhancer AGREE
        # =========================================================
        if agree:

            y = model_pred  # same as enhancer_pred

            # --- safe region: both confident
            if model_high and enhancer_high:
                return {
                    "decision": "ok" if rbo_high else "ok_noRAGExp",
                    "prediction": y,
                    "use_enhanced_explanation": True,
                    "use_enhancer_explanation": rbo_high,
                }

            # --- asymmetric confidence
            if model_high or enhancer_high:
                if rbo_high:
                    return {
                        "decision": "ok",
                        "prediction": y,
                        "use_enhanced_explanation": True,
                        "use_enhancer_explanation": True,
                    }
                else:
                    return {
                        "decision": "abstain",
                        "prediction": None,
                        "use_enhanced_explanation": False,
                        "use_enhancer_explanation": False,
                    }

            # --- both low confidence
            return {
                "decision": "abstain",
                "prediction": None,
                "use_enhanced_explanation": False,
                "use_enhancer_explanation": False,
            }

        # =========================================================
        # CASE 2: model and enhancer DISAGREE
        # =========================================================
        else:

            # --- model dominates
            if model_high and not enhancer_high:
                return {
                    "decision": "ok_noRAGExp",
                    "prediction": model_pred,
                    "use_enhanced_explanation": True,
                    "use_enhancer_explanation": False,
                }

            # --- enhancer dominates
            if enhancer_high and not model_high:
                return {
                    "decision": "ok_noModelExp",
                    "prediction": enhancer_pred,
                    "use_enhanced_explanation": False,
                    "use_enhancer_explanation": True,
                }

            # --- both low confidence
            return {
                "decision": "abstain",
                "prediction": None,
                "use_enhanced_explanation": False,
                "use_enhancer_explanation": False,
            }


    @log_method
    def __plot_breakdown_agreement(self, df: pd.DataFrame):
        agreement_counts = df[
            ["both_correct", "both_wrong", "model_correct_enhancer_wrong", "enhancer_correct_model_wrong"]
        ].sum()
        agreement_percentages = agreement_counts / len(df) * 100

        fig, ax = plt.subplots(figsize=(8, 5))

        sns.barplot(
            x=agreement_percentages.index,
            y=agreement_percentages.values,
            palette="viridis",
            hue=agreement_percentages.index,
            legend=False,
            ax=ax
        )

        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        ax.set_xlabel("")
        ax.set_ylabel("percentage (%)")
        ax.set_title("Classifier agreement breakdown")

        max_value = agreement_percentages.max()
        ax.set_ylim(0, max_value + 5)

        for i, value in enumerate(agreement_percentages.values):
            ax.text(
                i,
                value + 0.5,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10
            )

        fig.tight_layout()

        path_plot = f"{self.dm.ev_agreement_plot_path}{self.model_name}_{self.complete_enhancer_name}_agreement_breakdown.png"
        fig.savefig(path_plot, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


    def __save_aggregated_agreement_df(self, df: pd.DataFrame, save_global_evaluator: bool) -> None:

        if save_global_evaluator:
            path_agg = f"{self.dm.global_evaluator_path}aggregated_agreement_df.csv"
        else:
            path_agg = f"{self.dm.ev_agreement_path}aggregated_agreement_df.csv"
        model_abbr = f"{self.llm_name} RAG" if self.rag else f"{self.llm_name} LLM"
        # columns to identify unique model-enhancer pairs (and associated metadata)
        key_cols = ["dataset_prefix", "model_name", "enhancer_name", "llm_name", "rag", "retriever_name", "corpus_name", "model_abbr"]

        n = len(df)
        sums = df[["both_correct", "both_wrong", "model_correct_enhancer_wrong", "enhancer_correct_model_wrong"]].sum()

        agreement = df["agree"].mean()

        new_row = {
            "dataset_prefix": self.dataset_prefix,
            "model_name": self.model_name,
            "enhancer_name": self.enhancer_name,
            "llm_name": self.llm_name,
            "rag": self.rag,
            "retriever_name": self.retriever_name,
            "corpus_name": self.corpus_name,
            "model_abbr": model_abbr,
            "agreement": agreement,
            "disagreement": 1 - agreement,
            "both_correct": sums["both_correct"] / n,
            "both_wrong": sums["both_wrong"] / n,
            "model_better": sums["model_correct_enhancer_wrong"] / n,
            "enhancer_better": sums["enhancer_correct_model_wrong"] / n,
        }

        # --- load or create
        if os.path.exists(path_agg):
            agreement_df = self.ch.read_dataframe(path_agg, dtype=dtype)
        else:
            agreement_df = pd.DataFrame(columns=new_row.keys())

        # --- append
        agreement_df = pd.concat([agreement_df, pd.DataFrame([new_row])], ignore_index=True)

        # --- overwrite duplicates (KEEP LAST = new row wins)
        agreement_df = agreement_df.drop_duplicates(subset=key_cols, keep="last")

        # --- save
        self.ch.save_dataframe(agreement_df, path_agg)


    def __read_explainer_enhancer_dfs(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        enhancer_df = self.ch.read_dataframe(f"{self.dm.enhancer_path}{self.enhancer_name}_cleaned_df.csv", dtype=dtype) # self.dm.enhancer_path contains the path to the enhancer directory. self.dm.ds_enhancer_path is for decision system enhancer directory
        if self.explainer_name is not None:
            explainer_df = self.ch.read_dataframe(f"{self.dm.explainer_path}{self.explainer_name}_df.csv", dtype=dtype)
        else:
            explainer_df = self.ch.read_dataframe(f"{self.dm.model_path}{self.model_name}_predicted.csv", dtype=dtype) # if no explainer, we can still compute agreement and confidence distribution using the model predictions and confidence scores
        # df = explainer_df.merge(enhancer_df, on=ind, how="inner")
        # df = df.drop(columns=f"{self.info['target']}_x")
        # df = df.rename(columns={f"{self.info["target"]}_y": self.info["target"]})

        target = self.info["target"]

        df = explainer_df.merge(enhancer_df.drop(columns=target), on=ind, how="inner")

        return df 

        
    # PUBLIC
    # --------------------------------------------------------------------------------------------------------------------

    @log_method
    def compute_breakdown_agreement(self, save_global_evaluator: bool = False) -> None:
        df = self.__read_explainer_enhancer_dfs()

        target_col = self.info["target"]
        model_pred_col = f"{self.model_name}_pred"
        enhancer_pred_col = f"{self.enhancer_name}_pred"
        model_conf_col = f"{self.model_name}_conf"

        select_columns = [ind, target_col, model_pred_col, model_conf_col, enhancer_pred_col]
        df = df[select_columns]

        # Masks
        model_correct = df[model_pred_col] == df[target_col]
        enhancer_correct = df[enhancer_pred_col] == df[target_col]
        
        df["both_correct"] = model_correct & enhancer_correct
        df["both_wrong"] = ~model_correct & ~enhancer_correct
        df[f"model_correct_enhancer_wrong"] = (model_correct & ~enhancer_correct)
        df[f"enhancer_correct_model_wrong"] = (~model_correct & enhancer_correct)
        df['agree'] = df[model_pred_col] == df[enhancer_pred_col]

        self.ch.save_dataframe(df, f"{self.dm.ev_agreement_csv_path}{self.model_name}_{self.complete_enhancer_name}_agreement_df.csv")
        df = self.ch.read_dataframe(f"{self.dm.ev_agreement_csv_path}{self.model_name}_{self.complete_enhancer_name}_agreement_df.csv", dtype=dtype)
        self.__save_aggregated_agreement_df(df, save_global_evaluator)
        self.__plot_breakdown_agreement(df)
    

    @log_method
    def evaluate_explanations(self):
        df = self.__read_explainer_enhancer_dfs()

        c_explainer = f'{self.explainer_name}_feature_importance_ranking'
        c_enhancer = 'feature_importance_ranking_clean'

         # Parse ranking columns back into Python lists
        df[c_explainer] = df[c_explainer].apply(self.__safe_parse_ranking)
        df[c_enhancer] = df[c_enhancer].apply(self.__safe_parse_ranking)

        select_columns = [ind, c_enhancer, c_explainer]
        # select_columns = [ind, self.info['target'], f"{self.model_name}_pred", f"{self.model_name}_conf", f"{self.enhancer_name}_pred", c_enhancer, c_explainer]
        df = df[select_columns]
        
        df = self.__compute_similarity_exp(df, c_explainer, c_enhancer) # compute similarity metric (e.g., RBO) between explainer and enhancer feature importance rankings, and store it in a new column (e.g., 'rbo_k').
        df.drop(columns=[c_explainer, c_enhancer], inplace=True) # drop the original ranking columns to save space

        self.ch.save_dataframe(df, f"{self.dm.ev_explanations_csv_path}{self.model_name}_{self.explainer_name}_evaluator_df.csv") # complete_enhancer_name is not necessary here, because I have already separated with the directories

    @log_method
    def save_aggregated_explanation_feature_statistics(self, save_global_evaluator: bool = True) -> None:
        if self.explainer_name is None:
            self.lm.printl("No explainer specified, skipping aggregated explanation feature statistics.")
            return

        df = self.ch.read_dataframe(f"{self.dm.explainer_path}{self.explainer_name}_df.csv", dtype=dtype)

        c_explainer = f'{self.explainer_name}_feature_importance_ranking'
        c_exp_dict = f'{self.explainer_name}_exp_dict'

        df[c_explainer] = df[c_explainer].apply(self.__safe_parse_ranking)
        df[c_exp_dict] = df[c_exp_dict].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

        ranked_rows = []
        for _, row in df.iterrows():
            ranked_features = row[c_explainer]
            exp_dict = row[c_exp_dict]

            if not isinstance(ranked_features, list) or not isinstance(exp_dict, dict):
                continue

            ranked_values = [exp_dict.get(feature, np.nan) for feature in ranked_features]
            ranked_rows.append(ranked_values)

        if not ranked_rows:
            self.lm.printl(f"No ranked explanation values found for {self.dataset_prefix} - {self.model_name} - {self.explainer_name}")
            return

        max_ranks = max(len(values) for values in ranked_rows)
        summary_row = {
            'dataset_prefix': self.dataset_prefix,
            'model_name': self.model_name,
            'model_prefix': self.model_prefix,
            'explainer_name': self.explainer_name,
            'n_samples': len(ranked_rows),
            'n_features': max_ranks,
        }

        for rank_idx in range(max_ranks):
            rank_values = [row[rank_idx] for row in ranked_rows if len(row) > rank_idx and pd.notna(row[rank_idx])]

            if not rank_values:
                continue

            rank_array = np.asarray(rank_values, dtype=float)
            rank_number = rank_idx + 1
            summary_row[f'feature{rank_number}_mean'] = float(np.mean(rank_array))
            summary_row[f'feature{rank_number}_std'] = float(np.std(rank_array, ddof=1)) if len(rank_array) > 1 else 0.0
            summary_row[f'feature{rank_number}_abs_mean'] = float(np.mean(np.abs(rank_array)))
            summary_row[f'feature{rank_number}_abs_std'] = float(np.std(np.abs(rank_array), ddof=1)) if len(rank_array) > 1 else 0.0
            summary_row[f'feature{rank_number}_n_samples'] = int(len(rank_array))

        summary_df = pd.DataFrame([summary_row])

        if save_global_evaluator:
            path_agg = f"{self.dm.global_evaluator_path}aggregated_explanation_feature_statistics.csv"
        else:
            path_agg = f"{self.dm.ev_explanations_path}{self.dataset_prefix}_aggregated_explanation_feature_statistics.csv"

        key_cols = ['dataset_prefix', 'model_name', 'model_prefix', 'explainer_name']

        if os.path.exists(path_agg):
            agg_df = self.ch.read_dataframe(path_agg, dtype=dtype)
            agg_df = pd.concat([agg_df, summary_df], ignore_index=True, sort=False)
        else:
            agg_df = summary_df

        agg_df = agg_df.drop_duplicates(subset=key_cols, keep='last')

        self.ch.save_dataframe(agg_df, path_agg)


    # SAVE AGGREGATED RESULTS (for easier cross-dataset comparison and plotting)
    #--------------------------------------------------------------------------------------------------------------------
    @log_method
    def save_aggregated_explanation_evaluation(self, save_global_evaluator: bool = False) -> None:

        evaluator_df = self.ch.read_dataframe(
            f"{self.dm.ev_explanations_csv_path}{self.model_name}_{self.explainer_name}_evaluator_df.csv",
            dtype=dtype
        )

        # Read agreement of model and enhancer, so that we can evaluate the
        # explanation similarity metric (e.g., RBO) in different agreement regions.
        agreement_df = self.ch.read_dataframe(
            f"{self.dm.ev_agreement_csv_path}{self.model_name}_{self.complete_enhancer_name}_agreement_df.csv",
            dtype=dtype
        )

        if len(evaluator_df) != len(agreement_df):
            raise ValueError(
                f"evaluator_df and agreement_df must have the same number of rows, "
                f"got {len(evaluator_df)} and {len(agreement_df)}."
            )
        if save_global_evaluator:
            path_agg = f"{self.dm.global_evaluator_path}aggregated_explanation_df.csv"
        else:
            path_agg = f"{self.dm.ev_explanations_path}aggregated_explanation_df.csv"

        # Metric column converted once
        s = pd.to_numeric(evaluator_df[self.metric_complete_name], errors="coerce")

        model_abbr = f"{self.llm_name} RAG" if self.rag else f"{self.llm_name} LLM"

        # Subsets to evaluate:
        # - all rows
        # - agree
        # - both_correct
        # - both_wrong
        subset_masks = {
            "all": pd.Series(True, index=evaluator_df.index),
            "agree": agreement_df["agree"].astype(bool),
            "both_correct": agreement_df["both_correct"].astype(bool),
            "both_wrong": agreement_df["both_wrong"].astype(bool),
        }

        new_rows = []

        for computation_type, mask in subset_masks.items():
            subset_s = s[mask]
            valid_s = subset_s.dropna()
            new_row = {
                "dataset_prefix": self.dataset_prefix,
                "model_name": self.model_name,
                "model_prefix": self.model_prefix,
                "enhancer_name": self.enhancer_name,
                "llm_name": self.llm_name,
                "rag": self.rag,
                "retriever_name": self.retriever_name,
                "corpus_name": self.corpus_name,
                "model_abbr": model_abbr,
                "explainer_name": self.explainer_name,
                "metric": self.metric_complete_name,
                "computation_type": computation_type,

                "n_rows": int(mask.sum()),
                "n_valid": int(valid_s.shape[0]),
                "n_missing": int(subset_s.isna().sum()),

                "mean": valid_s.mean() if not valid_s.empty else None,
                "std": valid_s.std() if not valid_s.empty else None,
                "min": valid_s.min() if not valid_s.empty else None,
                "q25": valid_s.quantile(0.25) if not valid_s.empty else None,
                "median": valid_s.median() if not valid_s.empty else None,
                "q75": valid_s.quantile(0.75) if not valid_s.empty else None,
                "max": valid_s.max() if not valid_s.empty else None,
            }

            new_rows.append(new_row)

        key_cols = [
            "dataset_prefix"
            "model_name",
            "model_prefix",
            "enhancer_name",
            "llm_name",
            "rag",
            "retriever_name",
            "corpus_name",
            "model_abbr",
            "explainer_name",
            "metric",
            "computation_type",
        ]

        if os.path.exists(path_agg):
            agg_df = self.ch.read_dataframe(path_agg, dtype=dtype)
        else:
            agg_df = pd.DataFrame(columns=list(new_rows[0].keys()))

        agg_df = pd.concat([agg_df, pd.DataFrame(new_rows)], ignore_index=True)
        agg_df = agg_df.drop_duplicates(subset=key_cols, keep="last")

        self.ch.save_dataframe(agg_df, path_agg)

    @log_method
    def save_classification_report(self, save_global_evaluator: bool = False) -> None:
        """
        Compute and save classification reports for black-box and/or enhancer.

        Per saved row:
        precision_0, precision_1,
        recall_0, recall_1,
        f1-score_0, f1-score_1,
        accuracy, errors, error_rate,
        support_0, support_1,
        precision_macro, recall_macro, f1-score_macro, support_macro,
        precision_weighted, recall_weighted, f1-score_weighted, support_weighted

        Uniqueness:
        - blackbox: dataset, model_name, model_prefix
        - enhancer: dataset, enhancer_name, llm_name, rag, retriever_name, corpus_name
        """

        if self.model_name is not None and self.model_prefix is not None:
            df = self.ch.read_dataframe(f"{self.dm.model_path}{self.model_name}_predicted.csv", dtype=dtype)
            error_rate = np.nan

        elif self.enhancer_name is not None:
            df = self.ch.read_dataframe(f"{self.dm.enhancer_path}{self.enhancer_name}_cleaned_df.csv", dtype=dtype)
            removed_error_df = self.ch.read_dataframe(f"{self.dm.enhancer_path}{self.enhancer_name}_removed_error_idx.csv", dtype=dtype)

            n_test_size = dataset_test_sizes_dict[self.dataset_prefix]
            errors = len(removed_error_df)
            error_rate = (errors / n_test_size) if n_test_size > 0 else np.nan

        else:
            self.lm.printl("[Warning] No model or enhancer specified for classification report. Skipping.")
            return

        target = self.info["target"]
        if save_global_evaluator:
            path_agg = f"{self.dm.global_evaluator_path}classification_reports.csv"
        else:
            path_agg = f"{self.dm.ev_report_path}classification_reports.csv"

        def _safe_class_metrics(report_dict: dict, class_label: int) -> dict:
            possible_keys = [
                str(class_label),          # "0"
                str(float(class_label)),   # "0.0"
                class_label,               # 0
                float(class_label),        # 0.0
            ]

            cls = {}

            for key in possible_keys:
                if key in report_dict:
                    cls = report_dict[key]
                    break

            return {
                f"precision_{class_label}": cls.get("precision"),
                f"recall_{class_label}": cls.get("recall"),
                f"f1-score_{class_label}": cls.get("f1-score"),
                f"support_{class_label}": cls.get("support"),
            }

        def _build_report_metrics(y_true, y_pred) -> dict:
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
                "support_macro": macro.get("support"),

                "precision_weighted": weighted.get("precision"),
                "recall_weighted": weighted.get("recall"),
                "f1-score_weighted": weighted.get("f1-score"),
                "support_weighted": weighted.get("support"),
            }

            row.update(_safe_class_metrics(report, 0))
            row.update(_safe_class_metrics(report, 1))
            return row

        rows = []

        # -------- black-box row --------
        if self.model_name is not None and self.model_prefix is not None:
            bb_metrics = _build_report_metrics(df[target], df[f"{self.model_name}_pred"])

            bb_row = {
                "system_type": "blackbox",
                "dataset": self.dataset_prefix,

                "model_abbr": self.model_name,
                "model_name": self.model_name,
                "model_prefix": self.model_prefix,

                "enhancer_name": None,
                "llm_name": None,
                "rag": None,
                "retriever_name": None,
                "corpus_name": None,

                "error_rate": np.nan,
                "errors": np.nan,
            }
            bb_row.update(bb_metrics)
            rows.append(bb_row)

        # -------- enhancer row --------
        if self.enhancer_name is not None:
            enhancer_metrics = _build_report_metrics(df[target], df[f"{self.enhancer_name}_pred"])

            enhancer_label = f"{self.llm_name} RAG" if self.rag else f"{self.llm_name} LLM"

            enh_row = {
                "system_type": "enhancer",
                "dataset": self.dataset_prefix,

                "model_abbr": enhancer_label,
                "model_name": None,
                "model_prefix": None,

                "enhancer_name": self.enhancer_name,
                "llm_name": self.llm_name,
                "rag": self.rag,
                "retriever_name": self.retriever_name,
                "corpus_name": self.corpus_name,

                "error_rate": error_rate,
                "errors": errors,
            }
            enh_row.update(enhancer_metrics)
            rows.append(enh_row)

        if not rows:
            return

        new_rows_df = pd.DataFrame(rows)

        if os.path.exists(path_agg):
            report_df = self.ch.read_dataframe(path_agg, dtype=dtype)
            report_df = pd.concat([report_df, new_rows_df], ignore_index=True)
        else:
            report_df = new_rows_df

        bb_key = ["system_type", "dataset", "model_name", "model_prefix"]
        enh_key = ["system_type", "dataset", "enhancer_name", "llm_name", "rag", "retriever_name", "corpus_name"]

        # deduplicate separately because blackbox and enhancer have different uniqueness rules
        bb_df = report_df[report_df["system_type"] == "blackbox"].copy()
        enh_df = report_df[report_df["system_type"] == "enhancer"].copy()

        if not bb_df.empty:
            bb_df = bb_df.drop_duplicates(subset=bb_key, keep="last")

        if not enh_df.empty:
            enh_df = enh_df.drop_duplicates(subset=enh_key, keep="last")

        report_df = pd.concat([bb_df, enh_df], ignore_index=True)

        self.ch.save_dataframe(report_df, path_agg)

    @log_method
    def save_aggregate_explanation_metrics(self, metrics_list: List[str], save_global_evaluator: bool = False) -> None:
        explainer_df = self.ch.read_dataframe(f"{self.dm.explainer_path}{self.explainer_name}_df.csv",dtype=dtype)
        if save_global_evaluator:
            path_agg = f"{self.dm.global_evaluator_path}aggregated_metrics.csv"
        else:
            path_agg = f"{self.dm.ev_explanations_path}aggregated_metrics.csv"

        aggregate_metrics = {
            "dataset_prefix": self.dataset_prefix,
            "model_name": self.model_name,
            "model_prefix": self.model_prefix,
            "explainer_name": self.explainer_name,
        }

        for metric in metrics_list:
            if metric in explainer_df.columns:
                aggregate_metrics[f"{metric}_mean"] = explainer_df[metric].mean()
                aggregate_metrics[f"{metric}_std"] = explainer_df[metric].std()

        new_row_df = pd.DataFrame([aggregate_metrics])

        key_cols = ["dataset_prefix", "model_name", "model_prefix", "explainer_name"]

        # Load existing file if present
        if os.path.exists(path_agg):
            agg_df = self.ch.read_dataframe(path_agg, dtype=dtype)
            agg_df = pd.concat([agg_df, new_row_df], ignore_index=True)
        else:
            agg_df = new_row_df

        # Remove duplicates (keep latest run)
        agg_df = agg_df.drop_duplicates(subset=key_cols, keep="last")

        self.ch.save_dataframe(agg_df, path_agg)

    @log_method
    def plot_aggregated_explanations_statistics(self, avg_text_x_shift_map=None, avg_text_y_shift_map=None, avg_text_x_default=None):
        """
        Plot aggregated explanation statistics in one figure.

        Expected columns in df:
        - model_abbr
        - explainer_name
        - mean
        """
        path_agg = f"{self.dm.ev_explanations_path}aggregated_explanation_df.csv"
        df = self.ch.read_dataframe(path_agg, dtype=dtype)
        plot_df = df.copy()
        plot_df["mean"] = pd.to_numeric(plot_df["mean"], errors="coerce")
        plot_df = plot_df.dropna(subset=["model_abbr", "explainer_name", "mean"])

        avg_text_x_shift_map = avg_text_x_shift_map or {}
        avg_text_y_shift_map = avg_text_y_shift_map or {}

        computation_types = sorted(plot_df["computation_type"].dropna().unique())

        for computation_type in computation_types:
            sub_plot_df = plot_df[plot_df["computation_type"] == computation_type].copy()

            if sub_plot_df.empty:
                continue

            path_plot = (
                f"{self.dm.ev_explanations_path}"
                f"aggregated_explanation_statistics_{computation_type}.png"
            )

            model_order = sorted(sub_plot_df["model_abbr"].unique())

            xpos = []
            colors = []
            values = []
            block_centers = {}
            counter = 0

            for model_abbr in model_order:
                model_df = (
                    sub_plot_df[sub_plot_df["model_abbr"] == model_abbr]
                    .sort_values("mean", ascending=False)
                    .reset_index(drop=True)
                )

                start = counter

                for _, row in model_df.iterrows():
                    xpos.append(counter)
                    colors.append(enhancer_color_dict[model_abbr])
                    values.append(row["mean"])
                    counter += 1

                if len(model_df) > 0:
                    block_centers[model_abbr] = np.mean(range(start, start + len(model_df)))

                counter += 1  # space between model groups

            fig, ax = plt.subplots(figsize=(16, 6), dpi=dpi)
            ax.set_facecolor("white")

            # bars
            ax.bar(x=xpos, height=values, color=colors, width=0.8, zorder=3)

            # update limits after plotting
            x_min, x_max = ax.get_xlim()

            # full-width dashed mean line + annotation per model_abbr
            for model_abbr in model_order:
                model_df = (
                    sub_plot_df[sub_plot_df["model_abbr"] == model_abbr]
                    .sort_values("mean", ascending=False)
                    .reset_index(drop=True)
                )

                if model_df.empty:
                    continue

                mean_val = model_df["mean"].mean()
                color = enhancer_color_dict[model_abbr]

                ax.axhline(
                    y=mean_val,
                    color=color,
                    linestyle="--",
                    linewidth=2,
                    zorder=2
                )

                x_shift = avg_text_x_shift_map.get(model_abbr, 0.0)
                y_shift = avg_text_y_shift_map.get(model_abbr, 0.0)

                text_x = (x_max - 0.5) if avg_text_x_default is None else avg_text_x_default
                text_x = text_x + x_shift

                ax.text(
                    text_x,
                    mean_val + y_shift,
                    f"{mean_val:.3f}",
                    ha="right",
                    va="bottom",
                    fontsize=16,
                    color=color,
                    zorder=4
                )

            # horizontal grid only
            ax.set_axisbelow(True)
            ax.grid(axis="y", linestyle="--", linewidth=0.5, color="lightgray")

            ax.set_xticks([block_centers[m] for m in model_order])
            ax.set_xticklabels(model_order, fontsize=16)
            ax.tick_params(axis="y", labelsize=16)

            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title("")

            # full border around the plot
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)
                spine.set_color("black")

            plt.tight_layout()
            plt.savefig(path_plot, dpi=dpi, bbox_inches="tight")
            plt.show()
            plt.close(fig)