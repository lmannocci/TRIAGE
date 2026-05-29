
    # def __get_shap_top_k_features(self, shap_dict):
    #     """
    #     Extracts the top-k features from a SHAP explanation dictionary,
    #     sorted by absolute SHAP value.

    #     Parameters:
    #         shap_dict (dict or str): A dictionary or a stringified dictionary of feature: shap_value pairs.

    #     Returns:
    #         List[str]: Top-k feature names sorted by absolute SHAP importance.
    #     """
    #     if isinstance(shap_dict, str):
    #         try:
    #             shap_dict = ast.literal_eval(shap_dict)
    #         except Exception as e:
    #             raise ValueError(f"Failed to parse shap_dict string: {e}")

    #     if not isinstance(shap_dict, dict):
    #         raise TypeError(f"Expected shap_dict to be a dict, got {type(shap_dict)}")

    #     # Get the top k features sorted by absolute SHAP value
    #     top_features = sorted(
    #         shap_dict.items(),
    #         key=lambda item: abs(item[1]),
    #         reverse=True
    #     )[:self.top_k]

    #     return [feature for feature, _ in top_features]
    

    # def __get_dalex_top_k_features(self, dalex_dict):
    #     """
    #     Extracts the top-k features from a DALEX explanation dictionary,
    #     sorted by absolute SHAP value.

    #     Parameters:
    #         dalex_dict (dict or str): A dictionary or a stringified dictionary of feature: shap_value pairs.

    #     Returns:
    #         List[str]: Top-k feature names sorted by absolute SHAP importance.
    #     """
    #     if isinstance(dalex_dict, str):
    #         try:
    #             dalex_dict = ast.literal_eval(dalex_dict)
    #         except Exception as e:
    #             raise ValueError(f"Failed to parse shap_dict string: {e}")

    #     if not isinstance(dalex_dict, dict):
    #         raise TypeError(f"Expected shap_dict to be a dict, got {type(dalex_dict)}")

    #     # Get the top k features sorted by absolute SHAP value
    #     top_features = sorted(
    #         dalex_dict.items(),
    #         key=lambda item: abs(item[1]),
    #         reverse=True
    #     )[:self.top_k]

    #     return [feature for feature, _ in top_features]

    @log_method
    def compute_info_selector(self, extract_features_explainer: bool = True, extract_features_enhancer: bool = True):
        # original_df = self.ch.read_dataframe(f"{self.dm.dataset_path}{self.dataset}", dtype=dtype)
        enhancer_df = self.ch.read_dataframe(f"{self.dm.enhancer_path}{self.enhancer_name}_cleaned_df.csv", dtype=dtype) # self.dm.enhancer_path contains the path to the enhancer directory. self.dm.ds_enhancer_path is for decision system enhancer directory
        explainer_df = self.ch.read_dataframe(f"{self.dm.explainer_path}{self.explainer_name}_df.csv", dtype=dtype)


        df = explainer_df.merge(enhancer_df, on=ind, how="inner")
        df = df.drop(columns=f'{self.info['target']}_x')
        df = df.rename(columns={f"{self.info["target"]}_y": self.info["target"]})
        desired_order = [ind, self.info['target'], f"{self.model_name}_pred", f"{self.model_name}_conf", f"{self.enhancer_name}_pred"]
        # Reorder the DataFrame while keeping other columns unchanged
        df = df[desired_order + [col for col in df.columns if col not in desired_order]]

        if extract_features_explainer:
            df = self.__extract_explainer_features(df)
        if extract_features_enhancer:
            df = self.__extract_enhancer_features(df)

        df = self.__compute_similarity_exp(df)

        # Save the DataFrame to a CSV file
        self.ch.save_dataframe(df, f"{self.dm.ds_csv_path}{self.model_name}_{self.explainer_name}_{self.ds_name}_{self.metric}_selector_df.csv", dtype=dtype)

    @log_method
    def compute_breakdown_agreement(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        target_col = self.info["target"]
        model_pred_col = f"{self.model_name}_pred"
        enhancer_pred_col = f"{self.enhancer_name}_pred"

        # Masks
        model_correct = df[model_pred_col] == df[target_col]
        enhancer_correct = df[enhancer_pred_col] == df[target_col]
        
        df["both_correct"] = model_correct & enhancer_correct
        df["both_wrong"] = ~model_correct & ~enhancer_correct
        df[f"model_correct_enhancer_wrong"] = (model_correct & ~enhancer_correct)
        df[f"enhancer_correct_model_wrong"] = (~model_correct & enhancer_correct)
        df['agree'] = df[model_pred_col] == df[enhancer_pred_col]

        self.ch.save_dataframe(df, f"{self.dm.ds_csv_path}{self.model_name}_{self.explainer_name}_{self.ds_name}_{self.metric}_selector.csv", dtype=dtype)
        
        self.__update_agreement_df(df)
        self.__plot_agreement_breakdown(df)
        self.__plot_confidence_distribution(df)
        return df


    def __save_classification_report(self, df: pd.DataFrame):
        target = self.info['target']
        # Save Classification report for blakcbox and enhancer
        report_bb = classification_report(df[target], df[f"{self.model_name}_pred"])
        report_enhancer = classification_report(df[target], df[f"{self.enhancer_name}_pred"])
        self.ch.save_txt(str(report_bb), f"{self.dm.ds_analysis_report_path}{self.model_name}_{self.model_prefix}_report.txt")
        self.ch.save_txt(str(report_enhancer), f"{self.dm.ds_analysis_report_path}{self.enhancer_name}_{self.model_prefix}_report.txt")


    

    
    @log_method
    def __plot_agreement_breakdown(self, df: pd.DataFrame):
        agreement_counts = df[["both_correct", "both_wrong", "model_correct_enhancer_wrong", "enhancer_correct_model_wrong"]].sum()
        agreement_percentages = agreement_counts / len(df) * 100
        path_plot = f"{self.dm.ds_analysis_agreement_path}{self.model_name}_{self.enhancer_name}_{self.ds_name}_{self.metric}_agreement_breakdown.png"

        ax = sns.barplot(x=agreement_percentages.index, y=agreement_percentages.values, palette="viridis",  hue=agreement_percentages.index, legend=False)
        plt.xticks(rotation=30, ha='right')
        plt.xlabel("")
        plt.ylabel("percentage (%)")
        plt.title("Classifier agreement breakdown")
        plt.tight_layout()

        # Add percentage labels on top of each bar
        for i, (value, label) in enumerate(zip(agreement_percentages.values, agreement_percentages.index)):
            ax.text(i, value + 1, f"{value:.1f}%", ha='center', va='bottom', fontsize=10)
        # Extend y-axis limit to prevent label clipping
        max_value = agreement_percentages.max()
        plt.ylim(0, max_value + 5)

        # Save the figure
        plt.savefig(path_plot, dpi=dpi, bbox_inches="tight")
        plt.show()

    @log_method
    def __plot_confidence_distribution(self, df: pd.DataFrame):
        path_plot =  f"{self.dm.ds_analysis_confidence_path}{self.model_name}_{self.enhancer_name}_{self.ds_name}_{self.metric}_confidence_distribution.png"

        plt.figure(figsize=(7, 5))
        sns.kdeplot(df[df[f"{self.model_name}_pred"] == df[f"{self.info['target']}"]][f"{self.model_name}_conf"], fill=True, label="Correct Predictions", color="green")
        
        sns.kdeplot(df[df[f"{self.model_name}_pred"] != df[f"{self.info['target']}"]][f"{self.model_name}_conf"], fill=True, label="Incorrect Predictions", color="red")
        plt.xlabel(f"{self.model_name} Confidence")
        plt.ylabel("Density")
        plt.title("Confidence Distribution for Correct vs Incorrect Predictions")
        plt.legend()
        plt.savefig(path_plot, dpi=dpi, bbox_inches="tight")
        plt.show()

    def __update_agreement_df(self, df: pd.DataFrame):
        """
        Update agreement_df using precomputed breakdown columns.
        Stores a full decision breakdown per model/explainer.
        """

        n = len(df)

        # --- use breakdown columns directly
        both_correct = df["both_correct"].sum() / n
        both_wrong = df["both_wrong"].sum() / n
        model_better = df["model_correct_enhancer_wrong"].sum() / n
        enhancer_better = df["enhancer_correct_model_wrong"].sum() / n
        agreement = df["agree"].mean()
        disagreement = 1 - agreement

        self.lm.printl(
            f"[Agreement] model={self.model_name} enhancer={self.enhancer_name} | "
            f"agree={agreement:.3f} | disagree={disagreement:.3f}"
        )

        path = f"{self.dm.ds_analysis_agreement_path}agreement_df.csv"

        if os.path.exists(path):
            agreement_df = self.ch.read_dataframe(path, dtype=dtype)
        else:
            agreement_df = pd.DataFrame(
                columns=[
                    "model",
                    "enhancer",
                    "agreement",
                    "disagreement",
                    "both_correct",
                    "both_wrong",
                    "model_better",
                    "enhancer_better",
                ]
            )

        mask = (
            (agreement_df["model"] == self.model_name)
            & (agreement_df["enhancer"] == self.enhancer_name)
        )

        new_row = {
            "model": self.model_name,
            "enhancer": self.enhancer_name,
            "agreement": agreement,
            "disagreement": disagreement,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "model_better": model_better,
            "enhancer_better": enhancer_better,
        }

        if mask.any():
            agreement_df.loc[mask, list(new_row.keys())] = list(new_row.values())
        else:
            agreement_df = pd.concat([agreement_df, pd.DataFrame([new_row])], ignore_index=True)

        self.ch.save_dataframe(agreement_df, path)

    def __compute_row_confidence_enhancer(self, df: pd.DataFrame, mode: str = "set"):
        """
        Select rows for which confidence must be computed.

        mode="set"      → cumulative union of rows (pickle set)
        mode="summary"  → save number of rows per model/explainer (csv)
        """

        # ---------------------------------------------------
        # 1. COMMON LOGIC: decide rows to compute confidence
        # ---------------------------------------------------

        metric = f"{self.metric_exp_ranking}_{self.top_k}"

        # Case 1: model & enhancer agree BUT explanation unstable (low RBO)
        same_class_low_metric = df[df["agree"] & (df[metric] < self.th_rbo)]

        # Case 2: model & enhancer disagree
        diff_class_df = df[~df["agree"]]

        rows_to_compute = set(
            same_class_low_metric["original_index"].tolist()
            + diff_class_df["original_index"].tolist()
        )

        n_rows = len(rows_to_compute)

        # ---------------------------------------------------
        # 2. OUTPUT MODE
        # ---------------------------------------------------

        if mode == "set":

            path = f"{self.dm.dataset_path}rows_to_compute_confidence.pkl"

            if os.path.exists(path):
                row_compute_confidence = self.ch.read_object(path)
            else:
                row_compute_confidence = set()

            row_compute_confidence |= rows_to_compute
            self.ch.save_object(row_compute_confidence, path)

            self.lm.printl(
                f"{self.dataset_prefix}/{self.model_name}/{self.explainer_name} | "
                f"Total rows={len(row_compute_confidence)}"
            )

        elif mode == "summary":

            path = f"{self.dm.dataset_path}confidence_rows_summary.csv"

            if os.path.exists(path):
                summary_df = self.ch.read_dataframe(path, dtype=dtype)
            else:
                summary_df = pd.DataFrame(columns=["model", "explainer", "n_rows_confidence"])

            mask = (
                (summary_df["model"] == self.model_name)
                & (summary_df["explainer"] == self.explainer_name)
            )

            if mask.any():
                summary_df.loc[mask, "n_rows_confidence"] = n_rows
            else:
                new_summary_row = pd.DataFrame(
                            [{
                                "model": self.model_name,
                                "explainer": self.explainer_name,
                                "n_rows_confidence": n_rows
                            }]
                        )
                summary_df = pd.concat([summary_df, new_summary_row], ignore_index=True)

            self.ch.save_dataframe(summary_df, path)

            self.lm.printl(
                f"Dataset={self.dm.dataset_path} | "
                f"Model={self.model_name} | Explainer={self.explainer_name} | "
                f"Rows={n_rows}"
            )

        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'set' or 'summary'.")




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
    def __get_medrag_top_k_features_from_str(self, ranking_str):
        # Check if the value is NaN
        if pd.isna(ranking_str):
            return np.nan
        # Convert the string representation of the list into an actual list
        feature_list = eval(ranking_str)
        # Extract the top k features
        return feature_list[:self.top_k]

    def __extract_explainer_features(self, df: pd.DataFrame):
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
    
    def __extract_enhancer_features(self, df: pd.DataFrame):

        if self.enhancer_name == 'medrag':
            # Apply the function to each row and create a new column 'medrag_top_k'
            df[f'{self.enhancer_name}_top_{str(self.top_k)}'] = df['feature_importance_ranking'].apply(lambda x: self.__get_medrag_top_k_features_from_str(x))
        return df
    

    # Compute RBO for each row and create a new column 'rbo_k'
    def __compute_rbo(self, row):
        if pd.isna(row['feature_importance_ranking']):  # Check for None (null) values
            return np.nan
        a = row[f'{self.explainer_name}_top_{str(self.top_k)}']
        b = row[f'{self.enhancer_name}_top_{str(self.top_k)}']

        # Compute RBO using RankingSimilarity
        similarity = rbo.RankingSimilarity(a, b)
        return similarity.rbo()

    def __compute_similarity_exp(self, df: pd.DataFrame):
        if self.metric_exp_ranking == 'rbo':
            # Compute RBO for each row and create a new column 'rbo_k'
            df[self.metric] = df.apply(self.__compute_rbo, axis=1)
        return df