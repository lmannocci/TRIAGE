import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from utils.checkpoint.checkpoint import *
from utils.directory_manager.directory_manager import DirectoryManager
from utils.log_manager.log_manager import *
from utils.decorator_definition import *
from utils.common_variables import *
from utils.integrity_constraint_manager.integrity_constraint_manager import IntegrityConstraintManager

from typing import List, Tuple, Dict, Union, Optional


from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score, make_scorer, f1_score

from itertools import product
from interpret.glassbox import ExplainableBoostingClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import xgboost as xgb

absolute_path = os.path.dirname(__file__)
# file_name = os.path.splitext(os.path.basename(__file__))[0]
results_path = os.path.join(absolute_path, f"..{os.sep}results{os.sep}")
data_path = os.path.join(absolute_path, f"..{os.sep}data{os.sep}")


class Blackbox:
    def __init__(self, ch: Checkpoint, lm: LogManager, dataset_prefix: str, model_name: str, model_prefix: str,
                 param_grid: dict):
        self.ch: Checkpoint = ch
        self.lm: LogManager = lm

        self.icm: IntegrityConstraintManager = IntegrityConstraintManager(lm)
        self.icm.check_dataset(dataset_prefix)
        self.icm.check_model(model_name)

        self.dm: DirectoryManager = DirectoryManager(lm, results_path, dataset_prefix=dataset_prefix, model_name=model_name, model_prefix=model_prefix)

        self.dataset_prefix: str = dataset_prefix
        self.dataset: str = f"{self.dataset_prefix}_preprocessed.csv"
        self.info: Dict[str, Union[str, List[str]]] = df_info[self.dataset_prefix]

        self.model_name: str = model_name
        self.model_prefix: str = model_prefix
        self.param_grid: Dict[str, Union[List[float], None, str, int]] = param_grid

        self.lm.printl(f"Running model: {self.model_name} with prefix: {self.model_prefix}")

    

    def __standard_scaler(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Standardize the numerical features in the training and test sets.
        """
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train[self.info['num_columns']]),
                                      columns=self.info['num_columns'], index=X_train.index)
        
        # I use the same scaler to transform the test set, so I just call transform() and not fit_transform()
        X_test_scaled = pd.DataFrame(scaler.transform(X_test[self.info['num_columns']]),
                                     columns=self.info['num_columns'], index=X_test.index)

        # Concatenate scaled numerical columns with untouched categorical columns
        X_train_final = pd.concat([X_train_scaled, X_train[self.info['cat_columns']]], axis=1)
        X_test_final = pd.concat([X_test_scaled, X_test[self.info['cat_columns']]], axis=1)

        self.ch.save_model(scaler, self.dm.model_path + f"standard_scaler.joblib")
        
        return X_train_final, X_test_final

    def __sampler(self, type_sampler: str, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply oversampling to balance the dataset.
        """
        self.lm.printl(f"Train size: {str(len(X_train))}")
        self.lm.printl(f"Imbalanced dataset: {str(y_train.value_counts())}")
        if type_sampler == 'oversampling':
            # Apply oversampling only on the training set
            sampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
        elif type_sampler == 'undersampling':
            # Apply undersampling only on the training set
            sampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
        
        X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)

        self.lm.printl(f"Resampled train size: {str(len(X_train))}")
        self.lm.printl(f"Balanced dataset: {y_train_resampled.value_counts()}")

        return X_train_resampled, y_train_resampled

    def __sampling_test_set(self, test_df: pd.DataFrame, n_row_sampling: int) -> Tuple[pd.Series, pd.Series]:
        """
        Sample a subset of the test set for explanation.
        """
        if n_row_sampling >= len(test_df):
            self.lm.printl(f"Requested sampling size {n_row_sampling} is larger than the test set size {len(test_df)}. Using the entire test set.")
            raise ValueError("Sampling size is larger than the test set size.")
        
        sampled_idx_path = f"{self.dm.dataset_path}sampled_test_idx.csv"
        not_sampled_idx_path = f"{self.dm.dataset_path}not_sampled_test_idx.csv"
        
        if os.path.exists(sampled_idx_path) and os.path.exists(not_sampled_idx_path):
            sampled_test_idx = self.ch.read_dataframe(sampled_idx_path, dtype=dtype)[ind]
            not_sampled_test_idx = self.ch.read_dataframe(not_sampled_idx_path, dtype=dtype)[ind]
            self.lm.printl(f"Using pre-saved sampled test indices")
        else:
            self.lm.printl(f"Sampling {n_row_sampling} rows from the test set for {self.dataset_prefix} dataset")
            not_sampled_test_idx, sampled_test_idx = train_test_split(test_df[ind], test_size=n_row_sampling, stratify=test_df[self.info['target']], random_state=42)
            
            # Save indices as small dataframes
            self.ch.save_dataframe(pd.DataFrame({ind: sampled_test_idx}), sampled_idx_path)
            self.ch.save_dataframe(pd.DataFrame({ind: not_sampled_test_idx}), not_sampled_idx_path)
        
        # Filter test_df to only include sampled indices
        test_df = test_df[test_df[ind].isin(sampled_test_idx)].copy()
        self.lm.printl(f"Test set target distribution after sampling:\n{test_df[self.info['target']].value_counts()}")
        return test_df, sampled_test_idx, not_sampled_test_idx

    @log_method
    def __preprocessing_classification(self, df: pd.DataFrame, n_row_sampling: Optional[int] = None):
        if df is None:
            df: pd.DataFrame = self.ch.read_dataframe(f"{data_path}{self.dataset}", dtype=dtype)


        train_idx_path = f"{self.dm.dataset_path}train_idx.csv"
        test_idx_path = f"{self.dm.dataset_path}test_idx.csv"

        # Load or create train/test indices
        if os.path.exists(train_idx_path) and os.path.exists(test_idx_path):
            train_idx = self.ch.read_dataframe(train_idx_path, dtype=dtype)[ind]
            test_idx = self.ch.read_dataframe(test_idx_path, dtype=dtype)[ind]
        else:
            train_idx, test_idx = train_test_split(df[ind], test_size=0.2, random_state=42, stratify=df[self.info['target']])
            # Save indices as small dataframes
            self.ch.save_dataframe(pd.DataFrame({ind: train_idx}), train_idx_path)
            self.ch.save_dataframe(pd.DataFrame({ind: test_idx}), test_idx_path)


        # Define train and test sets
        train_df = df[~df[ind].isin(test_idx)].copy()
        test_df = df[df[ind].isin(test_idx)].copy()
        if self.dataset_prefix == 'diabetes' and n_row_sampling is not None:
            test_df, sampled_test_idx, not_sampled_test_idx = self.__sampling_test_set(test_df, n_row_sampling)

        # Separate features and target
        # train_df e test_df already have 'index', target, model predictions and confidence, while X_train and X_test do not have these columns
        X_train = train_df[self.info['x_columns']]
        y_train = train_df[self.info['target']]
        X_test = test_df[self.info['x_columns']]
        y_test = test_df[self.info['target']]

        if self.dataset_prefix == 'pima':
            if self.model_name in models_to_be_scaled:
                X_train, X_test = self.__standard_scaler(X_train, X_test)
            X_train_resampled, y_train_resampled = self.__sampler('oversampling', X_train, y_train)
            return df, train_idx, test_idx, train_df, test_df, X_train_resampled, X_test, y_train_resampled, y_test
        elif self.dataset_prefix == 'diabetes':
            if self.model_name in models_to_be_scaled:
                X_train, X_test = self.__standard_scaler(X_train, X_test)
            X_train_resampled, y_train_resampled = self.__sampler('undersampling', X_train, y_train)
            return df, train_idx, test_idx, train_df, test_df, X_train_resampled, X_test, y_train_resampled, y_test
        elif self.dataset_prefix == 'stroke':
            if self.model_name in models_to_be_scaled:
                X_train, X_test = self.__standard_scaler(X_train, X_test)
            X_train_resampled, y_train_resampled = self.__sampler('oversampling', X_train, y_train)
            return df, train_idx, test_idx, train_df, test_df, X_train_resampled, X_test, y_train_resampled, y_test
        elif self.dataset_prefix == 'liver':
            if self.model_name in models_to_be_scaled:
                X_train, X_test = self.__standard_scaler(X_train, X_test)
            X_train_resampled, y_train_resampled = self.__sampler('oversampling', X_train, y_train)
            return df, train_idx, test_idx, train_df, test_df, X_train_resampled, X_test, y_train_resampled, y_test
        elif self.dataset_prefix == 'covid':
            if self.model_name in models_to_be_scaled:
                X_train, X_test = self.__standard_scaler(X_train, X_test)
            X_train_resampled, y_train_resampled = self.__sampler('oversampling', X_train, y_train)
            return df, train_idx, test_idx, train_df, test_df, X_train_resampled, X_test, y_train_resampled, y_test

    @log_method
    def __grid_search(self, X_train, y_train):
        if self.model_name == 'random_forest':
            bb = RandomForestClassifier(random_state=42)
        elif self.model_name == 'ebm':
            bb = ExplainableBoostingClassifier(random_state=42)
        elif self.model_name == 'catboost':
            bb = CatBoostClassifier(verbose=0, random_state=42)
        elif self.model_name == 'xgboost':
            bb = XGBClassifier(eval_metric='logloss', tree_method = "hist", device='cuda', random_state=42) # tree_method='gpu_hist', predictor='gpu_predictor',
        else:
            raise ValueError(f"Model {self.model_name} not supported.")

        num_combinations = len(list(product(*self.param_grid.values())))
        self.lm.printl(f"Start gridsearch for {self.model_name}. Number of combinations: {str(num_combinations)}")


        # Create a custom F1 scorer that focuses on stroke=1
        f1_class = make_scorer(f1_score, pos_label=1)

        grid_search = GridSearchCV(bb, self.param_grid, cv=3, n_jobs=-1, error_score='raise', scoring=f1_class)
        grid_search.fit(X_train, y_train)

        best_bb = grid_search.best_estimator_
        self.lm.printl(f"Best parameters: {grid_search.best_params_}")
        self.ch.save_txt(str(grid_search.best_params_), self.dm.model_path + f"{self.model_name}_{self.model_prefix}_params.txt")
        
        self.ch.save_model(best_bb, self.dm.model_path + f"{self.model_name}_{self.model_prefix}.joblib")

        return best_bb


    def __plot_roc_curve(self, y_true: pd.Series, y_proba: np.ndarray, save_path: str) -> None:
        """
        Plot and save the ROC curve.
        """

        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(save_path)
        plt.close()

    # PUBLIC
    # ------------------------------------------------------------------------------------------------------------------
    @log_method
    def construct_bb(self, df: Optional[pd.DataFrame] = None, compute_model: Optional[bool]= False, n_row_sampling: Optional[int] = None) -> None:
        df, train_idx, test_idx, train_df, test_df, X_train, X_test, y_train, y_test = self.__preprocessing_classification(df, n_row_sampling)

        if compute_model or not os.path.exists(self.dm.model_path + f"{self.model_name}_{self.model_prefix}.joblib"):
            with measure_time(self.lm, label=f"grid_search {self.model_name}"):
                best_bb = self.__grid_search(X_train, y_train)
        else:
            best_bb = self.ch.load_model(self.dm.model_path + f"{self.model_name}_{self.model_prefix}.joblib")
            
        # Evaluate on validation set
        y_train_pred = best_bb.predict(X_train)
        y_train_pred = y_train_pred.astype(float)  # convert string predictions to float (USEFUL for EBM)
        train_report = classification_report(y_train, y_train_pred)

        # Final evaluation on the test set
        y_test_pred = best_bb.predict(X_test)
        y_test_pred = y_test_pred.astype(float)  # convert string predictions to float (USEFUL for EBM)
        test_report = classification_report(y_test, y_test_pred)
        self.lm.printl(f"\nTrain Classification Report:\n {str(train_report)}")
        self.lm.printl(f"\nTest Classification Report:\n {str(test_report)}")

        # Compute AUC
        y_train_proba = best_bb.predict_proba(X_train)[:, 1]
        auc_value_train = roc_auc_score(y_train, y_train_proba)
        self.lm.printl(f"Train AUC: {auc_value_train:.4f}")

        y_test_proba = best_bb.predict_proba(X_test)[:, 1]
        auc_value_test = roc_auc_score(y_test, y_test_proba)
        self.lm.printl(f"AUC: {auc_value_test:.4f}")

        # Save classification report *with* AUC appended
        train_report_with_auc = str(train_report) + f"\nAUC: {auc_value_train:.4f}\n"
        test_report_with_auc = str(test_report) + f"\nAUC: {auc_value_test:.4f}\n"

        self.ch.save_txt(train_report_with_auc, self.dm.model_path + f"{self.model_name}_{self.model_prefix}_train.txt")
        self.ch.save_txt(test_report_with_auc, self.dm.model_path + f"{self.model_name}_{self.model_prefix}_test.txt")

        self.__plot_roc_curve(y_test, y_test_proba, self.dm.model_path + f"{self.model_name}_{self.model_prefix}_roc_curve.png")

        # Get prediction probabilities (confidence scores)
        train_confidence = best_bb.predict_proba(X_train).max(axis=1)  # Max probability for the predicted class
        test_confidence = best_bb.predict_proba(X_test).max(axis=1)

        # Store predictions and the confidence in the original DataFrame
        # train_df.loc[self.model_name+'_pred'] = y_train_pred
        test_df[self.model_name+'_pred'] = y_test_pred
        # train_df[self.model_name+'_conf'] = train_confidence
        test_df[self.model_name+'_conf'] = test_confidence

        test_info_df = test_df[[ind, self.info['target'], self.model_name+'_pred', self.model_name+'_conf']].copy()

        # Save the modified DataFrame with prediction column for the test set
        self.ch.save_dataframe(test_info_df, f"{self.dm.model_path}{self.model_name}_predicted.csv")

    
    @log_method
    def plot_confidence_distribution(self, threshold: Optional[float] = None):
        df = self.ch.read_dataframe(
            f"{self.dm.model_path}{self.model_name}_predicted.csv",
            dtype=dtype
        )

        conf_col = f"{self.model_name}_conf"
        pred_col = f"{self.model_name}_pred"
        target_col = self.info["target"]

        # Ensure numeric
        df[conf_col] = pd.to_numeric(df[conf_col], errors="coerce")

        df_correct = df[df[pred_col] == df[target_col]]
        df_incorrect = df[df[pred_col] != df[target_col]]

        # ---------------------------------------------------------
        # 1. Correct vs Incorrect distribution
        # ---------------------------------------------------------
        plt.figure(figsize=(7, 5))

        sns.kdeplot(
            df_correct[conf_col].dropna(),
            fill=True,
            label="correct predictions",
            color="green"
        )

        sns.kdeplot(
            df_incorrect[conf_col].dropna(),
            fill=True,
            label="incorrect predictions",
            color="red"
        )

        plt.xlabel(f"confidence")
        plt.ylabel("density")
        # plt.title("Confidence Distribution: Correct vs Incorrect")
        plt.legend(loc="upper left")
        plt.xlim(0, 1)

        path_plot = f"{self.dm.model_path}{self.model_name}_breakdown_confidence_distribution.png"
        plt.savefig(path_plot, dpi=dpi, bbox_inches="tight")
        plt.show()
        plt.close()

        # ---------------------------------------------------------
        # 2. Overall confidence distribution
        # ---------------------------------------------------------
        plt.figure(figsize=(7, 5))

        sns.kdeplot(
            df[conf_col].dropna(),
            fill=True,
            color="blue"
        )

        plt.xlabel("confidence")
        plt.ylabel("density")
        # plt.title("Overall Confidence Distribution")
        plt.xlim(0, 1)

        path_plot_all = f"{self.dm.model_path}{self.model_name}_confidence_distribution.png"
        plt.savefig(path_plot_all, dpi=dpi, bbox_inches="tight")
        plt.show()
        plt.close()

        # ---------------------------------------------------------
        # 3. Threshold analysis (SAVE TO GLOBAL DF)
        # ---------------------------------------------------------
        if threshold is not None:
            self.lm.printl(f"[INFO] Confidence threshold = {threshold}")

            def compute_pct(series):
                series = series.dropna()
                if len(series) == 0:
                    return np.nan
                return 100 * (series > threshold).mean()

            total_pct = compute_pct(df[conf_col])
            correct_pct = compute_pct(df_correct[conf_col])
            incorrect_pct = compute_pct(df_incorrect[conf_col])

            self.lm.printl(f"[TOTAL] % confidence > {threshold}: {total_pct:.2f}%")
            self.lm.printl(f"[CORRECT] % confidence > {threshold}: {correct_pct:.2f}%")
            self.lm.printl(f"[INCORRECT] % confidence > {threshold}: {incorrect_pct:.2f}%")

            # -----------------------------------------------------
            # Save to global evaluator dataframe
            # -----------------------------------------------------
            path_global = f"{self.dm.global_evaluator_path}confidence_threshold_analysis.csv"

            new_row = {
                "dataset_prefix": self.dataset_prefix,
                "model_name": self.model_name,
                "model_prefix": self.model_prefix,
                "threshold": threshold,

                "pct_total": total_pct,
                "pct_correct": correct_pct,
                "pct_incorrect": incorrect_pct,
            }

            key_cols = ["dataset_prefix", "model_name", "model_prefix", "threshold"]

            if os.path.exists(path_global):
                df_global = self.ch.read_dataframe(path_global, dtype=dtype)
                df_global = pd.concat([df_global, pd.DataFrame([new_row])], ignore_index=True)
            else:
                df_global = pd.DataFrame([new_row])

            df_global = df_global.drop_duplicates(subset=key_cols, keep="last")

            self.ch.save_dataframe(df_global, path_global)