import numpy as np
import pandas as pd
import os
from typing import List, Tuple, Dict, Union, Optional
import matplotlib.pyplot as plt
import ast
import shap
import dalex as dx
import re

from sklearn.model_selection import train_test_split
from utils.checkpoint.checkpoint import *
from utils.directory_manager.directory_manager import DirectoryManager
from utils.log_manager.log_manager import *
from utils.decorator_definition import *
from utils.common_variables import *
from utils.integrity_constraint_manager.integrity_constraint_manager import IntegrityConstraintManager
from explainer.metrics import ExplainerMetrics

import lime
import lime.lime_tabular
from lime.lime_tabular import LimeTabularExplainer

absolute_path = os.path.dirname(__file__)
# file_name = os.path.splitext(os.path.basename(__file__))[0]
results_path = os.path.join(absolute_path, f"..{os.sep}results{os.sep}")
data_path = os.path.join(absolute_path, f"..{os.sep}data{os.sep}")


class Explainer:
    def __init__(self, ch: Checkpoint, lm: LogManager, dataset_prefix: str, 
                 model_name: str, model_prefix: str,
                 explainer_name: str):
        self.en_model = None
        self.ch: Checkpoint = ch
        self.lm: LogManager = lm

        self.icm: IntegrityConstraintManager = IntegrityConstraintManager(lm)
        self.icm.check_dataset(dataset_prefix)
        self.icm.check_model(model_name)
        self.icm.check_explainer(explainer_name)
        self.icm.check_model_explainer(model_name, explainer_name)

        self.dm: DirectoryManager = DirectoryManager(lm, results_path, dataset_prefix=dataset_prefix, model_name=model_name, model_prefix=model_prefix, explainer_name=explainer_name)

        self.dataset_prefix: str = dataset_prefix
        self.dataset: str = f"{self.dataset_prefix}_preprocessed.csv"
        self.info: Dict[str, Union[str, List[str]]] = df_info[self.dataset_prefix]

        self.model_name: str = model_name
        self.model_prefix: str = model_prefix

        self.explainer_name: str = explainer_name

        if self.model_name in models_to_be_scaled:
            self.scaler = self.ch.load_model(self.dm.model_path + f"standard_scaler.joblib")
        self.model = self.ch.load_model(f"{self.dm.model_path}{self.model_name}_{self.model_prefix}.joblib")


        self.lm.printl(f"Running model: {self.model_name} with prefix: {self.model_prefix} and explainer: {self.explainer_name}")

    # START OF UTILITY METHODS
    # ------------------------------------------------------------------------------------------------------------------

    def __lime_predict_wrapper(self, unscaled_input):
        # Step 1: Recreate DataFrame with proper column names
        df = pd.DataFrame(unscaled_input, columns=self.info['x_columns'])

        if self.model_name in models_to_be_scaled:
            # Step 2: Apply the same scaler only to numerical columns
            scaled_numeric = pd.DataFrame(
                self.scaler.transform(df[self.info['num_columns']]),
                columns=self.info['num_columns'],
                index=df.index
            )

            # Step 3: Keep categorical columns as-is
            df_cat = df[self.info['cat_columns']]

            # Step 4: Recombine and ensure correct column order
            df_processed = pd.concat([scaled_numeric, df_cat], axis=1)
        else:
            # If no scaling is needed, just use the original DataFrame
            df_processed = df

        # Step 5: Predict
        return self.model.predict_proba(df_processed)


    def __standard_scaler(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize the numerical features in the training and test sets.
        """
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.info['num_columns']]),
                                    columns=self.info['num_columns'], index=X.index)

        # Concatenate scaled numerical columns with untouched categorical columns
        X_final = pd.concat([X_scaled, X[self.info['cat_columns']]], axis=1)
    
        return X_final
    
    def __shap_ifexists_extract_positive_class(self, shap_output):
        if isinstance(shap_output, np.ndarray):
            # SHAP values for class 1 (shape: n_samples x n_features x 2)
            if shap_output.ndim == 3 and shap_output.shape[2] == 2:
                return shap_output[:, :, 1]
            elif shap_output.ndim == 2:
                return shap_output  # Already class-1 or regression
            else:
                raise ValueError("Unsupported ndarray shape.")
        else:
            raise ValueError("Unsupported SHAP output format.")

    def __shap_ifexists_extract_predicted_class(self, shap_output, X):
        """
        Extract SHAP values corresponding to the predicted class.

        Parameters
        ----------
        shap_output : np.ndarray
            Output of explainer.shap_values(X)
        X : np.ndarray or pd.DataFrame
            Input data used to compute SHAP values

        Returns
        -------
        np.ndarray
            SHAP values aligned with the predicted class for each sample
            shape: (n_samples, n_features)
        """

        preds = self.model.predict(X).astype(int)

        if isinstance(shap_output, np.ndarray):
            if shap_output.ndim == 3 and shap_output.shape[2] == 2:
                idx = np.arange(len(preds))
                return shap_output[idx, :, preds]

            elif shap_output.ndim == 2:
                return shap_output

            else:
                raise ValueError(f"Unsupported SHAP ndarray shape: {shap_output.shape}")

        else:
            raise ValueError("Unsupported SHAP output format.")
        
    @log_method
    def __save_lime_plot(self, exp_list, filename):
        # Extract feature importances
        features, importances = zip(*exp_list)
        
        pos_color="#4daf4a"
        neg_color="#e41a1c"
        # Assign colors: green for positive, red for negative
        colors = [pos_color if val >= 0 else neg_color for val in importances]
        
        # Plot
        plt.figure(figsize=(9, 3))
        bars = plt.barh(features, importances, color=colors, height=0.4)
        plt.xlabel('')
        # Increase x-tick label size
        plt.xticks(fontsize=12)
        
    #     plt.title('Local explanation for class Diabetes')
        
        # Compute x-axis limits
        max_val = max(abs(val) for val in importances)
        x_buffer = max_val * 0.2
        plt.xlim(-max_val - x_buffer, max_val + x_buffer)

        # Add value labels with proper alignment
        for bar, value in zip(bars, importances):
            width = bar.get_width()
            bar_center = bar.get_y() + bar.get_height() / 2
            label = f'{value:.3f}'

            if width >= 0:
                xpos = width + (0.01 * max_val)
                ha = 'left'
            else:
                xpos = width - (0.01 * max_val)
                ha = 'right'

            # In case of very large values, avoid sticking to the edge
            xpos = max(xpos, -max_val - x_buffer + 0.01) if width < 0 else min(xpos, max_val + x_buffer - 0.01)

            plt.text(xpos, bar_center, label, va='center', ha=ha, fontsize=12)
            
            
        # Set the background color to white
    #     plt.gca().set_facecolor('white')

    #     # Customize grid lines to be gray
    #     plt.grid(True, color='gray', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(filename, dpi=dpi)
        plt.close()

    # END OF UTILITY METHODS
    # ------------------------------------------------------------------------------------------------------------------


    # START EXPLAINER METHODS
    # ------------------------------------------------------------------------------------------------------------------

    @log_method
    def __lime_explainer(self, train_df: pd.DataFrame, test_df: pd.DataFrame, X_train: pd.DataFrame, X_test: pd.DataFrame):
        explainer = LimeTabularExplainer(X_train.values, feature_names=self.info['x_columns'], 
                                         discretize_continuous=True, class_names=['No Diabetes', 'Diabetes'], 
                                         mode='classification')

        explanations_list = []
        exp_object_dict = {}
        lime_explanations_faithfulness = []
        for i, row in test_df.iterrows():
            # X_test already contains the x_columns, but we need also the information about the original index and the prediction to include in the final DataFrame
            # so I will use the row from test_df, which contains all the necessary information
            row_x = row[self.info['x_columns']].values
            explanation = explainer.explain_instance(row_x,
                                                     self.__lime_predict_wrapper,
                                                     num_features=len(self.info['x_columns']), 
                                                     top_labels=1)
            # Get the correct label index for the explained class
            label_index = explanation.top_labels[0]

            # -----------------------------
            # Human-readable interval explanation
            # -----------------------------
            lime_interval_list = explanation.as_list(label=label_index)

            # -----------------------------
            # For display / saved dataframe
            # -----------------------------
            # Get explanations for that class
            exp_data = list(dict(explanation.as_list(label=label_index)).keys())

            # -----------------------------
            # Faithfulness-ready explanation
            # feature_index -> weight  --> feature_name -> weight
            # -----------------------------
            lime_map = explanation.as_map()[label_index]
            lime_exp_dict = {
                self.info['x_columns'][feat_idx]: weight
                for feat_idx, weight in lime_map
            }

            exp_dict = {}
            exp_dict[ind] = row[ind]  # index of the test set
            exp_dict[self.info['target']] = row[self.info['target']]  # target value
            exp_dict[self.model_name + '_pred'] = row[self.model_name + '_pred']  # prediction of the bb
            exp_dict[self.model_name + '_conf'] = row[self.model_name + '_conf']  # confidence of the bb

            # save interval string
            for j, value in enumerate(exp_data):
                exp_dict[f'{self.explainer_name}_feature_{j}'] = value  # lime feature interval
           
            # save full interval-based explanation
            exp_dict[f"{self.explainer_name}_exp_list_interval"] = lime_interval_list

            # save faithfulness-ready dict
            exp_dict[f"{self.explainer_name}_exp_dict"] = lime_exp_dict
            
            # Add row
            explanations_list.append(exp_dict)

            exp_object_dict[row[ind]] = explanation # lime explanation object, must be saved in a dictionary as object

            self.lm.printK(i, self.K, f"Explained instance {str(i + 1)}/{len(test_df)}")

        explanations_df = pd.DataFrame(explanations_list)

        self.ch.save_dataframe(explanations_df, f"{self.dm.explainer_path}{self.explainer_name}_df.csv")
        self.ch.save_object(exp_object_dict, f"{self.dm.explainer_path}{self.explainer_name}_exp_object_dict.pkl")

    @log_method
    def __shap_explainer(self, test_df: pd.DataFrame, X_test: pd.DataFrame):
        # 1. Create SHAP explainer
        explainer = shap.TreeExplainer(self.model)

        # Use the same scaler on test data
        if self.model_name in models_to_be_scaled:
            X_test = self.__standard_scaler(X_test)

        # 2. Compute SHAP values for test set
        shap_values = explainer.shap_values(X_test)

        # 3. Create a DataFrame to store SHAP values for class 1
        try:
            # shap_values_class1 = self.__shap_ifexists_extract_positive_class(shap_values)
            shap_values_predicted_class = self.__shap_ifexists_extract_predicted_class(shap_values, X_test)
        except ValueError as e:
            self.lm.printl(f"Error extracting SHAP values for class 1: {e}")
            return

        # Build DataFrame of SHAP values with renamed columns
        df_shap = pd.DataFrame(
            shap_values_predicted_class,
            columns=[f"shap_{col}" for col in self.info['x_columns']]
        )

        # Add column with full shap explanation as a dictionary
        df_shap[f"{self.explainer_name}_exp_dict"] = pd.DataFrame(shap_values_predicted_class, columns=self.info['x_columns']).apply(lambda row: row.to_dict(), axis=1)

        # Reset index to ensure alignment with test_df (optional but safe)
        df_shap.reset_index(drop=True, inplace=True)

        # Add selected columns from test_df
        required_columns = [ind, self.info['target'], f'{self.model_name}_pred', f'{self.model_name}_conf']
        df_shap[required_columns] = test_df[required_columns]

        self.ch.save_dataframe(df_shap, f"{self.dm.explainer_path}{self.explainer_name}_df.csv")
        self.ch.save_object(shap_values, f"{self.dm.explainer_path}{self.explainer_name}_exp_object_dict.pkl")

    def __predict_predicted_class_proba(self, modelBB, X):
        proba = modelBB.predict_proba(X)
        pred = modelBB.predict(X).astype(int)
        return proba[np.arange(len(pred)), pred]



    @log_method
    def __dalex_explainer(self, train_df:pd.DataFrame, test_df: pd.DataFrame, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series):
        
        if self.model_name in models_to_be_scaled:
            X_train = self.__standard_scaler(X_train)
            X_test = self.__standard_scaler(X_test)

        # Step1: Create DALEX explainer
        # explainer = dx.Explainer(self.model, X_train, y_train, label=self.model_name) # explain class 1 by default, but I want to explain the predicted class probability, so I will create a custom predict function that extracts the predicted class probability from the output of model.predict_proba() and pass it to the explainer, so that the explainer will explain the predicted class probability instead of the default behavior of explaining the output of model.predict()
        explainer = dx.Explainer(self.model, X_train, y_train, label=self.model_name, predict_function=self.__predict_predicted_class_proba) # explain the predicted class probability instead of the default behavior of explaining the output of model.predict()
        # Step 2: Compute values for test set with progress print
        explanations = []

        for i in range(len(test_df)):
            self.lm.printK(i, self.K, f"Processing row {i}/{len(X_test)}")
            row = X_test.iloc[[i]] # Pass as DataFrame
            explanation = explainer.predict_parts(row, type='break_down', N=500)
            explanations.append(explanation)
        
        # Step 3: Convert DALEX explanations to a list of dictionaries (one per instance)
        explanations_list_dict = [
            {row['variable_name']: row['contribution'] for _, row in expl.result.iterrows() if row['variable_name'] not in ['intercept','prediction']}
            for expl in explanations
        ]

        # Step 4: Create a DataFrame with a column for each feature
        df_explanation = pd.DataFrame([
            {f'dalex_{feat}': instance_dict.get(feat, 0.0) for feat in self.info['x_columns']}
            for instance_dict in explanations_list_dict
        ])

        # Step 5: Add the full explanation dictionary for each instance
        df_explanation[f"{self.explainer_name}_exp_dict"] = explanations_list_dict

        # Step 6: Add the index and other relevant columns from test_df
        # Reset index to ensure alignment with test_df (optional but safe)
        df_explanation.reset_index(drop=True, inplace=True)
  
        # Add selected columns from test_df
        columns_to_keep = [ind, self.info['target'], f'{self.model_name}_pred', f'{self.model_name}_conf']
        df_explanation[columns_to_keep] = test_df[columns_to_keep]

        self.ch.save_dataframe(df_explanation, f"{self.dm.explainer_path}{self.explainer_name}_df.csv")
        self.ch.save_object(explanations, f"{self.dm.explainer_path}{self.explainer_name}_exp_object_dict.pkl")

    @log_method
    def __ebm_explainer(self, test_df: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.Series = None):
        # EBM does not require a separate explainer, it is already interpretable
        self.lm.printl("EBM is already interpretable, no need for a separate explainer. We will load the model and save the explanations.")
        
        if self.model_name in models_to_be_scaled:
            # Use the same scaler on test data
            X_test = pd.DataFrame(self.scaler.transform(X_test), columns=self.info['x_columns'])

        # Compute EBM local explanations
        self.lm.printl("Computing EBM local explanations...")
        ebm_local = self.model.explain_local(X_test, y_test)
        self.lm.printl("EBM local explanations computed.")

        preds = self.model.predict(X_test).astype(int)

        # Convert EBM explanations to a list of dictionaries (one per instance)
        explanations_list_dict = []
        feature_names = ebm_local.feature_names
        for i in range(len(X_test)):
            # contributions per feature for i-th instance
            contribs = ebm_local.data(i)['scores']  # list of floats

            # If predicted class is 0 → flip sign: I want to explain the predicted class, so if the predicted class is 0, the contribution of each feature to the predicted class 0 is the opposite of the contribution to class 1, so I flip the sign of the contributions
            if preds[i] == 0:
                contribs = [-c for c in contribs]

            instance_dict = dict(zip(feature_names, contribs))
            explanations_list_dict.append(instance_dict)
        
        # Create DataFrame with prefixed feature columns
        df_explanation = pd.DataFrame([
            {f'ebm_{k}': v for k, v in instance.items()}
            for instance in explanations_list_dict
        ])

        # Add full explanation dict as a new column
        df_explanation[f'{self.explainer_name}_exp_dict'] = explanations_list_dict

        # Step 6: Add the index and other relevant columns from test_df
        df_explanation.reset_index(drop=True, inplace=True)

        # Add selected columns from test_df
        columns_to_keep = [ind, self.info['target'], f'{self.model_name}_pred', f'{self.model_name}_conf']
        df_explanation[columns_to_keep] = test_df[columns_to_keep]

        self.ch.save_dataframe(df_explanation, f"{self.dm.explainer_path}{self.explainer_name}_df.csv")
        self.ch.save_object(ebm_local, f"{self.dm.explainer_path}{self.explainer_name}_exp_object_dict.pkl")

    # END EXPLAINER METHODS
    # ------------------------------------------------------------------------------------------------------------------

    # START METHODS FOR DATA AND MODEL
    # ------------------------------------------------------------------------------------------------------------------

    def __merge_test_set_with_predictions(self):
        # Read original full dataset and predicted test set
        df_original = self.ch.read_dataframe(f"{data_path}{self.dataset}", dtype=dtype)
        df_predicted = self.ch.read_dataframe(f"{self.dm.model_path}{self.model_name}_predicted.csv", dtype=dtype)


        # Load saved train/test split indices
        train_idx = self.ch.read_dataframe(f"{self.dm.dataset_path}train_idx.csv", dtype=dtype)[ind]
        test_idx = self.ch.read_dataframe(f"{self.dm.dataset_path}test_idx.csv", dtype=dtype)[ind]

         # Extract test_df and train_df from original data
        test_df = df_original[df_original[ind].isin(test_idx)].copy()
        train_df = df_original[df_original[ind].isin(train_idx)].copy()

        if self.dataset_prefix == "diabetes":
            # For diabetes, we sampled 10k rows from the test set to compute explanations, so we need to merge the predicted test set with the original test set to keep only the sampled rows
            sampled_idx_path = f"{self.dm.dataset_path}sampled_test_idx.csv"
            sampled_test_idx = self.ch.read_dataframe(sampled_idx_path, dtype=dtype)[ind]
            test_df = test_df[test_df[ind].isin(sampled_test_idx)].copy()

        # target_col = self.info['target'] # target column is already in df_original
        pred_col = f"{self.model_name}_pred"
        conf_col = f"{self.model_name}_conf"
        required_columns = [ind, pred_col, conf_col]
        test_df_merged = pd.merge(df_predicted[required_columns], test_df, on=ind, how='left')
        
        return train_df, test_df_merged
    
    def __get_data_model(self) -> Tuple[pd.DataFrame, pd.DataFrame]:

        train_df, test_df = self.__merge_test_set_with_predictions()
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        # Separate features and target
        # train_df e test_df already have 'index', target, model predictions and confidence, while X_train and X_test do not have these columns
        X_train = train_df[self.info['x_columns']]
        y_train = train_df[self.info['target']]
        X_test = test_df[self.info['x_columns']]
        y_test = test_df[self.info['target']]
        
        return train_df, test_df, X_train, X_test, y_train, y_test

    # END METHODS FOR DATA AND MODEL
    # ------------------------------------------------------------------------------------------------------------------


    # EXTRACT FEATURES FROM EXPLANATIONS
    # ------------------------------------------------------------------------------------------------------------------
    # Define a function to extract feature names from the rule
    def __match_lime_feature(self, rule):
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
    def __extract_lime_features(self, lime_exp_list: str):
        # Convert the string representation of the list into an actual list of tuples (feature, importance)
        feature_importance_list = ast.literal_eval(lime_exp_list)
        
        # Extract features (sorted by importance)
        features_ranking_list = [self.__match_lime_feature(feature) for feature, importance in feature_importance_list]
        return features_ranking_list

    def __eval_features_from_str(self, exp_dict):
        """
        Extracts the top-k features from an explanation dictionary,
        sorted by absolute importance value.

        Parameters:
            exp_dict (dict or str): A dictionary or a stringified dictionary of feature: importance pairs.

        Returns:
            List[str]: Feature names sorted by absolute importance.
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
            )
        ]


    # END EXTRACT FEATURES FROM EXPLANATIONS
    # ------------------------------------------------------------------------------------------------------------------
    
    # PUBLIC
    # ------------------------------------------------------------------------------------------------------------------
    @log_method
    def explain(self, n_row_sampling: Optional[int] = 10000):
        train_df, test_df, X_train, X_test, y_train, y_test = self.__get_data_model()
        self.K = int(0.1* len(test_df))  # K is the number of instances to explain, set to 10% of the test set size
        if self.explainer_name == 'lime':
            self.__lime_explainer(train_df, test_df, X_train, X_test)
        elif self.explainer_name == 'shap':
            self.__shap_explainer(test_df, X_test)
        elif self.explainer_name == 'dalex':
            self.__dalex_explainer(train_df, test_df, X_train, X_test, y_train, y_test)
        elif self.explainer_name == 'ebm':
            self.__ebm_explainer(test_df, X_test, y_test)
            
    
    @log_method
    def lime_plot(self, index: int):
        explainer_df = self.ch.read_dataframe(f"{self.dm.explainer_path}{self.explainer_name}_df.csv", dtype=dtype)
        
        exp_list = ast.literal_eval(explainer_df[explainer_df['index']==index]['lime_exp_dict'].values[0])
        self.__save_lime_plot(exp_list, f"lime_explanation_{str(index)}.png")

    @log_method
    def extract_explainer_features(self):
        explainer_df = self.ch.read_dataframe(f"{self.dm.explainer_path}{self.explainer_name}_df.csv", dtype=dtype)
        method_map = {
            'lime': self.__eval_features_from_str, # __extract_lime_features, # for lime we need a specific function to extract the feature names from the interval-based explanation list, while for shap, dalex and ebm we can use the same function to extract the feature names from the explanation dictionary
            'shap': self.__eval_features_from_str,
            'dalex': self.__eval_features_from_str,
            'ebm': self.__eval_features_from_str,
        }

        exp_list_col = f'{self.explainer_name}_exp_dict'
        explainer_df[f'{self.explainer_name}_feature_importance_ranking'] = explainer_df[exp_list_col].apply(method_map[self.explainer_name])
        # self.ch.save_dataframe(explainer_df, f"{self.dm.explainer_path}{self.explainer_name}_df_with_ranking.csv")
        self.ch.update_csv_inplace(explainer_df, f"{self.dm.explainer_path}{self.explainer_name}_df.csv")

    @log_method
    def compute_explainer_metrics(self, metrics_list: Optional[List[str]] = None):
        # Extract explanations from the DataFrame
        explainer_df = self.ch.read_dataframe(f"{self.dm.explainer_path}{self.explainer_name}_df.csv", dtype=dtype)
        # Convert the string representation of the explanation dictionary back to an actual dictionary if it's stored as a string in the DataFrame
        explainer_df[f"{self.explainer_name}_exp_dict"] = explainer_df[f"{self.explainer_name}_exp_dict"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        explanations = explainer_df[f'{self.explainer_name}_exp_dict'].tolist()

        train_df, test_df, X_train, X_test, y_train, y_test = self.__get_data_model()
        em: ExplainerMetrics = ExplainerMetrics(self.ch, self.lm, self.dm, self.icm, self.dataset_prefix, self.model_name, self.model_prefix, self.model, self.explainer_name)
        for metric in metrics_list:
            if metric == 'faithfulness':
                faithfulness_df = em.faithfulness_metric(X_test, test_df, explanations)
                explainer_df = pd.merge(explainer_df, faithfulness_df, on=ind, how='left')
        
        self.ch.update_csv_inplace(explainer_df, f"{self.dm.explainer_path}{self.explainer_name}_df.csv")