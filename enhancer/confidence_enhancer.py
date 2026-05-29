import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from utils.checkpoint.checkpoint import *
from utils.directory_manager.directory_manager import DirectoryManager
from utils.log_manager.log_manager import *
from utils.decorator_definition import *
from typing import Optional
from utils.common_variables import *
from utils.integrity_constraint_manager.integrity_constraint_manager import IntegrityConstraintManager

from enhancer.enhancer import Enhancer

absolute_path = os.path.dirname(__file__)
# file_name = os.path.splitext(os.path.basename(__file__))[0]
results_path = os.path.join(absolute_path, f"..{os.sep}results{os.sep}")
data_path = os.path.join(absolute_path, f"..{os.sep}data{os.sep}")


class ConfidenceEnhancer:
    def __init__(self, ch: Checkpoint, lm: LogManager, dataset_prefix: str, enhancer_name: str,
                 llm_name: str, rag: bool, retriever_name: Optional[str] = 'MedCPT', corpus_name: Optional[str] = 'StatPearls', 
                 rag_K_documents: Optional[int] = 10, return_options: Optional[bool] = False,
                 synthesizer_name: Optional[str] = "ctgan", epochs: Optional[int] = 300, n_samples_synthesizer: Optional[int] = 1000,
                 n_neighbors: int = 3, metric_neighbors: str = 'cosine_similarity', aggregation_method: str = 'average',
                 cuda_visible_devices: Optional[str] = "1,2,3", parallel: Optional[bool] = True):
        self.en_model = None
        self.ch: Checkpoint = ch
        self.lm: LogManager = lm

        self.icm: IntegrityConstraintManager = IntegrityConstraintManager(lm)
        self.icm.check_dataset(dataset_prefix)
        self.icm.check_enhancer(enhancer_name)
        self.icm.check_llm(llm_name)
        self.icm.check_rag(rag)
        self.icm.check_retriever(retriever_name)
        self.icm.check_corpus(corpus_name)

        self.icm.check_synthesizer(synthesizer_name)

        self.icm.check_n_neighbors(n_neighbors)
        self.icm.check_metric_neighbors(metric_neighbors)
        self.icm.check_aggregation_method(aggregation_method)

        self.dm: DirectoryManager = DirectoryManager(lm, results_path, dataset_prefix=dataset_prefix, 
                                                     enhancer_name=enhancer_name, llm_name=llm_name, rag=rag, 
                                                     retriever_name=retriever_name, corpus_name=corpus_name,
                                                     synthesizer_name=synthesizer_name, epochs=epochs, n_samples_synthesizer=n_samples_synthesizer,
                                                     n_neighbors=n_neighbors, metric_neighbors=metric_neighbors, aggregation_method=aggregation_method)

        self.dataset_prefix: str = dataset_prefix
        self.dataset: str = f"{self.dataset_prefix}_preprocessed.csv"
        self.info: Dict[str, Union[str, List[str]]] = df_info[self.dataset_prefix]

        self.enhancer_name: str = enhancer_name
        self.llm_name: str = llm_name
        self.rag: bool = rag
        self.retriever_name: str = retriever_name
        self.corpus_name: str = corpus_name
        self.return_options: bool = return_options
        self.rag_K_documents: int = rag_K_documents

        self.synthesizer_name: str = synthesizer_name

        self.n_neighbors: int = n_neighbors
        self.metric_neighbors: str = metric_neighbors
        self.aggregation_method: str = aggregation_method

        self.parallel = parallel
        self.cuda_visible_devices = cuda_visible_devices


    def __fit_standard_scaler(self, df: pd.DataFrame):
        self._scaler = StandardScaler()
        self._scaler.fit(df[self.info["num_columns"]])

    def __transform_with_scaler(self, df: pd.DataFrame) -> pd.DataFrame:
        df_num_scaled = pd.DataFrame(
            self._scaler.transform(df[self.info["num_columns"]]),
            columns=self.info["num_columns"],
            index=df.index
        )

        df_cat = df[self.info["cat_columns"]]

        return pd.concat([df_num_scaled, df_cat], axis=1)


    @log_method
    def __preprocess_real_data(self, df: Optional[pd.DataFrame] = None):
        if df is None:
            df: pd.DataFrame = self.ch.read_dataframe(f"{data_path}{self.dataset}", dtype=dtype)


        train_idx_path = f"{self.dm.dataset_path}train_idx.csv"
        test_idx_path = f"{self.dm.dataset_path}test_idx.csv"


        train_idx = self.ch.read_dataframe(train_idx_path, dtype=dtype)[ind]
        test_idx = self.ch.read_dataframe(test_idx_path, dtype=dtype)[ind]

        # Define train and test sets
        train_df = df[~df[ind].isin(test_idx)].copy()
        test_df = df[df[ind].isin(test_idx)].copy()
         # Only if the dataset is diabetes, we need to filter the test set based on sampled indices
        if self.dataset_prefix == 'diabetes':
            sampled_idx_path = f"{self.dm.dataset_path}sampled_test_idx.csv"
            sampled_test_idx = self.ch.read_dataframe(sampled_idx_path, dtype=dtype)[ind]
            # Filter test_df to only include sampled indices
            test_df = test_df[test_df[ind].isin(sampled_test_idx)].copy()

        # ---- metadata ----
        test_meta = test_df[["original_index"]].copy()

        # Separate features and target
        # train_df e test_df already have 'index', target, model predictions and confidence, while X_train and X_test do not have these columns
        X_train = train_df[self.info['x_columns']]
        y_train = train_df[self.info['target']]
        X_test = test_df[self.info['x_columns']]
        y_test = test_df[self.info['target']]

        self.__fit_standard_scaler(X_train)

        X_train_scaled = self.__transform_with_scaler(X_train)
        X_test_scaled  = self.__transform_with_scaler(X_test)


        # Attach original_index AFTER scaling
        X_test_scaled["original_index"] = test_meta["original_index"]

        return X_test_scaled, y_test
    
    @log_method
    def __preprocess_synthetic_data(self, df: Optional[pd.DataFrame] = None):
        if df is None:
            df: pd.DataFrame = self.ch.read_dataframe(f"{self.dm.synthesizer_path}{self.dataset_prefix}_synthetic.csv", dtype=dtype)


        # Separate metadata, features, target
        meta_synth = df[["original_index"]].copy()

        # Separate features and target
        X_synth = df[self.info['x_columns']]
        y_synth = df[self.info['target']]

        # Standardize numerical columns
        X_synth_scaled = self.__transform_with_scaler(X_synth)

        # Attach original_index for downstream joins
        X_synth["original_index"] = meta_synth["original_index"]
        X_synth_scaled["original_index"] = meta_synth["original_index"]

        # I need both scaled and unscaled versions of synthetic data.
        # The scaled version is used to find the nearest neighbors, while the unscaled version is used to feed the enhancer model.
        return X_synth, X_synth_scaled, y_synth

    def __find_nearest_neighbors(self, X_test_scaled, X_synth_scaled, X_synth):
        """
        For each test sample, find the top-n nearest synthetic neighbors
        according to the specified metric.

        Returns a dataframe with:
        - test_original_index
        - synth_original_index
        - similarity
        - rank
        """

        # Compute similarity matrix
        if self.metric_neighbors == 'cosine_similarity':
            sim_matrix = cosine_similarity(X_test_scaled, X_synth_scaled)

        neighbors = []

        for test_pos, sims in enumerate(sim_matrix):
            # Indices of top-n neighbors (highest similarity)
            top_idx = np.argsort(sims)[-self.n_neighbors:][::-1]

            test_original_index = X_test_scaled.iloc[test_pos]["original_index"]

            for rank, synth_pos in enumerate(top_idx, start=1):
                neighbors.append({
                    "neighbor_of_original_index": test_original_index,
                    "synth_original_index": X_synth.iloc[synth_pos]["original_index"],
                    "similarity": sims[synth_pos],
                    "rank": rank
                })

        neighbors_df = pd.DataFrame(neighbors)

        # --- Join with raw synthetic data ---
        neighbors_with_data = neighbors_df.merge(
            X_synth,
            how="left",
            left_on="synth_original_index",
            right_on="original_index",
            suffixes=("", "_synth")
        )

        # Optional: drop duplicate join key
        neighbors_with_data = neighbors_with_data.drop(columns=["original_index"])

        self.ch.save_dataframe(neighbors_with_data, f"{self.dm.neighbours_path}neighbors_df.csv")

        return neighbors_with_data

    
        

    @log_method
    def __run_enhancer_on_neighbors(self, neighbors_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the enhancer only once per unique synthetic neighbor
        and merge enhancer outputs back to neighbors_df.
        """

        # 1. Extract unique synthetic records
        synth_cols = ["synth_original_index"] + self.info["x_columns"]

        unique_synth_df = (
            neighbors_df[synth_cols]
            .drop_duplicates(subset=["synth_original_index"])
            .reset_index(drop=True)
        )
        unique_synth_df = unique_synth_df.rename(columns={"synth_original_index": "original_index"})
        self.lm.printl(f"Total neighbor records: {len(neighbors_df)}")
        self.lm.printl(f"Unique synthetic records to enhance: {len(unique_synth_df)}")
        # 2. Run enhancer ONCE on unique synthetic records
        output_path = self.dm.ce_enhancer_path
        en = Enhancer(self.ch, self.lm, self.dataset_prefix, self.enhancer_name, llm_name=self.llm_name, rag=self.rag, rag_K_documents=self.rag_K_documents, 
                      return_options=self.return_options, cuda_visible_devices=self.cuda_visible_devices, parallel=self.parallel,
                      confidence_computation=True, output_path=output_path)
        en.initialize_enhancer()
        en.run_enhancer(unique_synth_df)
        enhancer_cleaned_df = en.clean_enhancer_df()
        enhancer_cleaned_df = en.clean_feature_importance_ranking()

        enhancer_cleaned_df = self.ch.read_dataframe(f"{self.dm.ce_enhancer_path}{self.enhancer_name}_cleaned_df.csv", dtype=dtype)
        # 3. Merge enhancer outputs back
        final_df = neighbors_df.merge(
            enhancer_cleaned_df,
            how="left",
            left_on="synth_original_index",
            right_on="original_index",
            suffixes=("", "_enh")
        )
        final_df = final_df.drop(columns=self.info["x_columns"])   

        self.ch.save_dataframe(final_df, f"{self.dm.ce_enhancer_path}neighbors_enhanced_df.csv") 
        return final_df

    def __compute_confidence_scores(self, neighbors_enhanced_df: pd.DataFrame, y_pred_df: pd.DataFrame,enhancer_pred_col: str):
        """
        Compute confidence scores for each test sample
        based on enhancer outputs of its synthetic neighbors.

        y_pred_df:
            contains columns [ind, enhancer_pred_col]
        """

        confidence_scores = []

        # build fast lookup: ind → prediction
        y_pred_map = dict(zip(y_pred_df[ind], y_pred_df[enhancer_pred_col]))

        # Group by test sample
        grouped = neighbors_enhanced_df.groupby("neighbor_of_original_index")

        for test_original_index, group in grouped:
            preds = group[f"{self.enhancer_name}_pred"].dropna().tolist()
            n = len(preds)

            if n == 0:
                confidence = np.nan

            elif self.aggregation_method == 'average':
                confidence = np.mean(preds)

            elif self.aggregation_method == 'agreement':
                # enhancer prediction on real sample
                y_enh = y_pred_map.get(test_original_index, None)

                if y_enh is None:
                    confidence = np.nan
                else:
                    matches = sum(p == y_enh for p in preds)

                    # simple version
                    confidence = matches / n

                    # optional robust version:
                    # confidence = (matches + 1) / (n + 2)

            confidence_scores.append({
                ind: test_original_index,
                f"{self.enhancer_name}_conf": confidence
            })

        confidence_df = pd.DataFrame(confidence_scores)

        # Save confidence scores
        self.ch.save_dataframe(confidence_df, f"{self.dm.ce_enhancer_path}confidence_scores_{self.aggregation_method}.csv")

        return confidence_df


    # PUBLIC
    # ------------------------------------------------------------------------------------------------------------------
    @log_method
    def find_nearest_neighbors(self):
        X_test_scaled, y_test = self.__preprocess_real_data()
        X_synth, X_synth_scaled, y_synth = self.__preprocess_synthetic_data()

        neighbors_df = self.__find_nearest_neighbors(X_test_scaled, X_synth_scaled, X_synth)

    @log_method
    def analyze_neighbors_distribution(self):
        """
        Analyze how synthetic neighbors are used across queries (test samples).

        neighbors_df is expected to have one row per (query, neighbor) pair with:
        - neighbor_of_original_index : the real/test sample id (the query)
        - synth_original_index       : the synthetic sample id selected as neighbor
        - similarity                 : cosine similarity (higher = closer)
        - rank                       : neighbor rank (1..k), where k == self.n_neighbors

        This method helps you answer questions like:
        - Are the same synthetic samples reused for many queries? (mode concentration)
        - How concentrated is the neighbor selection? (top10/top50 dominance)
        - Are the selected neighbors actually close? (similarity distributions, especially rank k)

        Outputs saved in: self.dm.neighbors_analysis_path
        CSVs:
            - neighbors_summary.csv         : one-row global summary
            - synth_reuse_counts.csv        : reuse count for every synthetic id
            - synth_reuse_top20.csv         : top 20 most reused synthetic ids
            - rank_similarity_stats.csv     : describe() of similarity for each rank
            - rank1_top20.csv               : top 20 most frequent rank-1 synthetic ids
        Plots:
            - similarity_hist.png
            - similarity_by_rank_box.png
            - synth_reuse_hist.png
        """

        neighbors_df = self.ch.read_dataframe(f"{self.dm.neighbours_path}neighbors_df.csv", dtype=dtype)
        # ---------------------------------------------------------------------
        # 0) Minimal type safety:
        #    - rank should be integer
        #    - similarity should be float
        #    This avoids subtle bugs when CSVs are read with object/string dtypes.
        # ---------------------------------------------------------------------
        df = neighbors_df.copy()
        df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
        df["similarity"] = pd.to_numeric(df["similarity"], errors="coerce")
        df = df.dropna(subset=["neighbor_of_original_index", "synth_original_index", "rank", "similarity"])
        df["rank"] = df["rank"].astype(int)

        # ---------------------------------------------------------------------
        # 1) Basic counts:
        #    - n_queries: how many real/test samples you queried
        #    - total_selected: total (query,neighbor) pairs = n_queries * k (ideally)
        #    - unique_synth: how many distinct synthetic ids were ever selected
        # ---------------------------------------------------------------------
        n_queries = df["neighbor_of_original_index"].nunique()
        total_selected = len(df)
        unique_synth = df["synth_original_index"].nunique()

        # Ratio close to 1 => neighbors are diverse (little reuse)
        # Ratio close to 0 => heavy reuse (same synthetic points used repeatedly)
        reuse_ratio = float(unique_synth / total_selected) if total_selected else float("nan")

        # ---------------------------------------------------------------------
        # 2) Synthetic reuse distribution:
        #    For each synthetic id: how many times it appears in all neighbor selections.
        #    This detects "dominant" synthetic points (modes) that serve many queries.
        # ---------------------------------------------------------------------
        reuse_counts = (
            df.groupby("synth_original_index")
            .size()
            .rename("times_selected")
            .sort_values(ascending=False)
            .reset_index()
        )

        max_reuse = int(reuse_counts["times_selected"].max()) if len(reuse_counts) else 0
        mean_reuse = float(reuse_counts["times_selected"].mean()) if len(reuse_counts) else float("nan")
        median_reuse = float(reuse_counts["times_selected"].median()) if len(reuse_counts) else float("nan")

        # Concentration metric: what fraction of all selections is explained by top X synth points?
        # High values => a small set of synthetic points dominates many neighborhoods.
        def top_share(x: int) -> float:
            if total_selected == 0 or len(reuse_counts) == 0:
                return float("nan")
            return float(reuse_counts["times_selected"].head(x).sum() / total_selected)

        top10_share = top_share(10)
        top50_share = top_share(50)

        # ---------------------------------------------------------------------
        # 3) Similarity statistics overall and by rank:
        #    - Overall describes how close neighbors tend to be.
        #    - By-rank describes how similarity decays with rank.
        #    - Rank-k (k=self.n_neighbors) is especially important: it's the "worst" neighbor you include in your confidence computation.
        # ---------------------------------------------------------------------
        rank_similarity_stats = (
            df.groupby("rank")["similarity"]
            .describe()
            .reset_index()
        )

        sim_desc = df["similarity"].describe()
        sim_mean = float(sim_desc["mean"])
        sim_std = float(sim_desc["std"]) if not pd.isna(sim_desc.get("std", np.nan)) else float("nan")
        sim_min = float(sim_desc["min"])
        sim_max = float(sim_desc["max"])

        k = int(self.n_neighbors)
        if (df["rank"] == k).any():
            sim_k_desc = df.loc[df["rank"] == k, "similarity"].describe()
            sim_k_mean = float(sim_k_desc["mean"])
            sim_k_median = float(sim_k_desc["50%"])
            sim_k_min = float(sim_k_desc["min"])
        else:
            # If rank k is missing, something is off in how neighbors were computed/stored.
            sim_k_mean = float("nan")
            sim_k_median = float("nan")
            sim_k_min = float("nan")

        # ---------------------------------------------------------------------
        # 4) Rank-1 dominance:
        #    If the same synthetic id is rank-1 for many queries, it’s a sign of
        #    strong clustering / mode dominance (or insufficient synthetic size).
        # ---------------------------------------------------------------------
        rank1 = df[df["rank"] == 1]
        rank1_top = (
            rank1["synth_original_index"].value_counts()
                .rename_axis("synth_original_index")
                .reset_index(name="rank1_count")
                .head(20)
        )
        rank1_unique = int(rank1["synth_original_index"].nunique()) if len(rank1) else 0

        # ---------------------------------------------------------------------
        # 5) One-row summary:
        #    This is the quick dashboard you can log/run-to-run compare.
        # ---------------------------------------------------------------------
        summary = {
            # Key columns to identify the run:
            "dataset_prefix": self.dataset_prefix,
            "n_neighbors": self.n_neighbors,
            "metric_neighbors": self.metric_neighbors,
            "aggregation_method": self.aggregation_method,

            # Size / diversity
            "n_queries": int(n_queries),
            "total_selected_neighbors": int(total_selected),
            "unique_synth_neighbors": int(unique_synth),
            "reuse_ratio_unique_over_selected": reuse_ratio,

            # Reuse / concentration
            "synth_reuse_max": int(max_reuse),
            "synth_reuse_mean": mean_reuse,
            "synth_reuse_median": median_reuse,
            "top10_synth_share_of_all_selections": top10_share,
            "top50_synth_share_of_all_selections": top50_share,
            "rank1_unique_synth": int(rank1_unique),

            # Similarity (overall)
            "similarity_mean": sim_mean,
            "similarity_std": sim_std,
            "similarity_min": sim_min,
            "similarity_max": sim_max,

            # Similarity at rank k (most relevant for your kNN-based confidence)
            "similarity_rank_k_mean": sim_k_mean,
            "similarity_rank_k_median": sim_k_median,
            "similarity_rank_k_min": sim_k_min,
        }
        summary_df = pd.DataFrame([summary])

        # ---------------------------------------------------------------------
        # 6) Save CSV outputs
        # ---------------------------------------------------------------------
        self.ch.save_dataframe(summary_df, os.path.join(self.dm.neighbours_analysis_path, "neighbors_summary.csv"))
        self.ch.save_dataframe(reuse_counts, os.path.join(self.dm.neighbours_analysis_path, "synth_reuse_counts.csv"))
        self.ch.save_dataframe(reuse_counts.head(20), os.path.join(self.dm.neighbours_analysis_path, "synth_reuse_top20.csv"))
        self.ch.save_dataframe(rank_similarity_stats, os.path.join(self.dm.neighbours_analysis_path, "rank_similarity_stats.csv"))
        self.ch.save_dataframe(rank1_top, os.path.join(self.dm.neighbours_analysis_path, "rank1_top20.csv"))
        
        # Save neighbors_summary.csv in a global path that aggregates all runs, to facilitate longitudinal analysis across runs.
        global_path = os.path.join(self.dm.global_evaluator_path, "neighbors_summary.csv")
        key_cols = [
            "dataset_prefix",
            "n_neighbors",
            "metric_neighbors",
            "aggregation_method",
        ]

        if os.path.exists(global_path):
            global_df = self.ch.read_dataframe(global_path, dtype=dtype)
            global_df = pd.concat([global_df, summary_df], ignore_index=True)
        else:
            global_df = summary_df.copy()
        global_df = global_df.drop_duplicates(subset=key_cols, keep="last")
        self.ch.save_dataframe(global_df, global_path)

        # ---------------------------------------------------------------------
        # 7) Save plots
        #    These plots help you visually detect:
        #      - poor neighbor quality (low similarity mass)
        #      - similarity decay by rank
        #      - heavy-tailed reuse (few synth points selected many times)
        # ---------------------------------------------------------------------

        # 7.1 Similarity histogram (all selected neighbors)
        plt.figure()
        plt.hist(df["similarity"].to_numpy(), bins=50)
        plt.title("Distribution of cosine similarity (all selected neighbors)")
        plt.xlabel("Similarity")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(self.dm.neighbours_analysis_path, "similarity_hist.png"), dpi=dpi)
        plt.close()

        # 7.2 Similarity by rank (boxplot)
        ranks_sorted = sorted(df["rank"].unique())
        data_by_rank = [df.loc[df["rank"] == r, "similarity"].to_numpy() for r in ranks_sorted]
        plt.figure()
        plt.boxplot(data_by_rank, labels=[str(r) for r in ranks_sorted], showfliers=False)
        plt.title("Cosine similarity by neighbor rank")
        plt.xlabel("Rank")
        plt.ylabel("Similarity")
        plt.tight_layout()
        plt.savefig(os.path.join(self.dm.neighbours_analysis_path, "similarity_by_rank_box.png"), dpi=dpi)
        plt.close()

        # 7.3 Synthetic reuse histogram
        plt.figure()
        plt.hist(reuse_counts["times_selected"].to_numpy(), bins=50)
        plt.title("Distribution of synthetic neighbor reuse counts")
        plt.xlabel("Times a synthetic row was selected")
        plt.ylabel("Number of synthetic rows")
        plt.tight_layout()
        plt.savefig(os.path.join(self.dm.neighbours_analysis_path, "synth_reuse_hist.png"), dpi=dpi)
        plt.close()

        return summary_df


    @log_method
    def run_enhancer_on_neighbors(self):
        neighbors_df = self.ch.read_dataframe(f"{self.dm.neighbours_path}neighbors_df.csv", dtype=dtype)
        final_df = self.__run_enhancer_on_neighbors(neighbors_df)


    @log_method
    def compute_confidence_scores(self):
        enhancer_df = self.ch.read_dataframe(f"{self.dm.enhancer_path}{self.enhancer_name}_cleaned_df.csv", dtype=dtype) # self.dm.enhancer_path contains the path to the enhancer directory. self.dm.ds_enhancer_path is for decision system enhancer directory
        enhancer_pred_col = f"{self.enhancer_name}_pred"
        y_pred_df = enhancer_df[[ind, enhancer_pred_col]]

        neighbors_enhanced_df = self.ch.read_dataframe(f"{self.dm.ce_enhancer_path}neighbors_enhanced_df.csv", dtype=dtype)

        self.__compute_confidence_scores(neighbors_enhanced_df, y_pred_df, enhancer_pred_col)

    @log_method
    def plot_confidence_enhancer_results(self):
        confidence_df = self.ch.read_dataframe(f"{self.dm.ce_enhancer_path}confidence_scores_{self.aggregation_method}.csv", dtype=dtype)

        plt.figure()
        plt.hist(confidence_df[f"{self.enhancer_name}_conf"].to_numpy(), bins=20)
        # plt.title("Distribution of confidence scores")
        # plt.xlabel("Confidence score")
        # plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(f"{self.dm.ce_enhancer_analysis_path}confidence_scores_{self.aggregation_method}_distribution.png", dpi=dpi)
        self.lm.printl(f"Saved confidence score distribution plot to {self.dm.ce_enhancer_analysis_path}confidence_scores_{self.aggregation_method}_distribution.png")

        plt.close()