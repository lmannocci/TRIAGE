import importlib

import numpy as np
import pandas as pd
import os
from typing import List, Tuple, Dict, Union, Optional
from huggingface_hub import login
from sklearn.metrics import classification_report
import ast
from rapidfuzz import utils, fuzz

from utils.checkpoint.checkpoint import *
from utils.directory_manager.directory_manager import DirectoryManager
from utils.log_manager.log_manager import *
from utils.decorator_definition import *
from utils.common_variables import *
from utils.integrity_constraint_manager.integrity_constraint_manager import IntegrityConstraintManager

from enhancer.enhancer_interfaces.medrag_interface import MedRagInterface
from enhancer.enhancer_interfaces.huggingface_interface import HuggingFaceInterface
from enhancer.enhancer_interfaces.tasks import LLMTask


absolute_path = os.path.dirname(__file__)
# file_name = os.path.splitext(os.path.basename(__file__))[0]
results_path = os.path.join(absolute_path, f"..{os.sep}results{os.sep}")
data_path = os.path.join(absolute_path, f"..{os.sep}data{os.sep}")


class Enhancer:
    def __init__(self, ch: Checkpoint, lm: LogManager, dataset_prefix: str, enhancer_name: str, llm_name: str, rag: bool, # mandatory arguments
                 retriever_name: Optional[str] = 'MedCPT', corpus_name: Optional[str] = 'StatPearls',
                 rag_K_documents: Optional[int] = 10, return_options: Optional[bool] = False,
                 output_path: Optional[str] = None, confidence_computation: Optional[bool] = False, 
                 cuda_visible_devices: Optional[str] = "1,2,3", parallel: Optional[bool] = True,
                 task: Optional[LLMTask] = None):
        self.en_model = None
        self.ch: Checkpoint = ch
        self.lm: LogManager = lm

        self.icm: IntegrityConstraintManager = IntegrityConstraintManager(lm)
        self.icm.check_dataset(dataset_prefix)
        # self.icm.check_model(model_name)
        self.icm.check_enhancer(enhancer_name)
        self.icm.check_llm(llm_name)
        self.icm.check_rag(rag)
        self.icm.check_enhancer_rag_combination(enhancer_name, rag)
        self.icm.check_retriever(retriever_name)
        self.icm.check_corpus(corpus_name)


        self.dm: DirectoryManager = DirectoryManager(lm, results_path, dataset_prefix=dataset_prefix, enhancer_name=enhancer_name, 
                                                     llm_name=llm_name, rag=rag, retriever_name=retriever_name, corpus_name=corpus_name)

        self.dataset_prefix: str = dataset_prefix
        self.dataset: str = f"{self.dataset_prefix}_preprocessed.csv"
        self.info: Dict[str, Union[str, List[str]]] = df_info[self.dataset_prefix]

        self.enhancer_name: str = enhancer_name
        self.llm_name: str = llm_name
        self.rag: bool = rag
        self.retriever_name: str = retriever_name
        self.corpus_name: str = corpus_name

        self.rag_K_documents: int = rag_K_documents
        self.return_options: bool = return_options

        if output_path is None:
            self.output_path = self.dm.enhancer_path
        else:
            self.output_path = output_path
        self.confidence_computation = confidence_computation
        self.cuda_visible_devices = cuda_visible_devices
        self.parallel: bool = parallel
        self.task = task

    def __get_test_and_select_columns(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        # If df is provided, use it directly (after selecting only necessary columns). Otherwise, read the original dataset and extract the test set based on saved indices.
        if df is not None:
            test_df = df
        else:
            # Read original full dataset and predicted test set
            df_original = self.ch.read_dataframe(f"{data_path}{self.dataset}", dtype=dtype)
            # Load saved test split indices
            test_idx = self.ch.read_dataframe(f"{self.dm.dataset_path}test_idx.csv", dtype=dtype)[ind]
            # Extract test_df
            test_df = df_original[df_original[ind].isin(test_idx)].copy()

            # Only if the dataset is diabetes, we need to filter the test set based on sampled indices
            if self.dataset_prefix == 'diabetes':
                sampled_idx_path = f"{self.dm.dataset_path}sampled_test_idx.csv"
                sampled_test_idx = self.ch.read_dataframe(sampled_idx_path, dtype=dtype)[ind]
                # Filter test_df to only include sampled indices
                test_df = test_df[test_df[ind].isin(sampled_test_idx)].copy()

        # Round numeric columns to 2 digits
        num_cols = test_df.select_dtypes(include="number").columns
        test_df[num_cols] = test_df[num_cols].round(2)

        if self.dataset_prefix == 'stroke':
            # Read LabelEncoders
            label_encoders = self.ch.read_object(f"{data_path}{self.dataset_prefix}_label_encoders.pkl")
            # Apply LabelEncoders to categorical columns
            for col, le in label_encoders.items():
                test_df[col] = le.inverse_transform(test_df[col])
        elif self.dataset_prefix == 'liver':
            # Read LabelEncoders
            label_encoders = self.ch.read_object(f"{data_path}{self.dataset_prefix}_label_encoders.pkl")
            # Apply LabelEncoders to categorical columns
            for col, le in label_encoders.items():
                test_df[col] = le.inverse_transform(test_df[col])
        test_df = test_df.reset_index(drop=True)

        return test_df 
    
    def set_to_update(self, to_update: bool):
        self.to_update = to_update

    @log_method
    def __filter_rows_to_process(self, df: pd.DataFrame, filter_list_index: Optional[List[int]] = None) -> List[Tuple[int, pd.Series]]:
          
        # ============================
        # 1. PREPARATION (shared)
        # ============================

        # Load checkpoint
        if os.path.exists(f"{self.output_path}{self.enhancer_name}_df.csv"):
            medrag_df = self.ch.read_dataframe(f"{self.output_path}{self.enhancer_name}_df.csv", dtype=dtype)
            processed_indices = set(medrag_df['original_index'].tolist())
        else:
            processed_indices = set()

        # Build work list
        rows_to_process = []
        for i, row in df.iterrows():
            original_index = row["original_index"]

            if original_index in processed_indices:
                continue

            if filter_list_index is not None and original_index not in filter_list_index:
                continue

            rows_to_process.append((i, row))

        if len(rows_to_process) == 0:
            self.lm.printl("No rows to process.")
        self.lm.printl(f"Number of rows to process: {len(rows_to_process)}")
        return rows_to_process


    def __clean_df(self, df: pd.DataFrame):
        # 1. empty_ranking_and_rules (feature_importance_ranking == '[]')
        mask_empty = df['feature_importance_ranking'] == '[]'
        removed_empty = pd.DataFrame({
            ind: df[mask_empty][ind],
            "motivation": "ranking_null"
        })
        df = df[~mask_empty]  # drop them

        # 2. ranking_null (feature_importance_ranking is NaN)
        mask_null = df['feature_importance_ranking'].isnull()
        removed_null = pd.DataFrame({
            ind: df[mask_null][ind],
            "motivation": "ranking_null"
        })
        df = df[~mask_null]  # drop them

        # 3. invalid predicted target
        valid_values = self.info['classes']
        mask_invalid_pred = ~df[f'{self.enhancer_name}_pred'].isin(valid_values)
        removed_invalid = pd.DataFrame({
            ind: df[mask_invalid_pred][ind],
            "motivation": "invalid_predicted_target"
        })
        df = df[~mask_invalid_pred]  # drop them

        # 4. Concatenate all removed indices
        removed_all = pd.concat([removed_empty, removed_null, removed_invalid], ignore_index=True)

        # 5. Save them
        self.ch.save_dataframe(removed_all, f"{self.output_path}{self.enhancer_name}_removed_error_idx.csv")

        # Debug info
        self.lm.printl(f"Removed {len(removed_empty)} rows for empty_ranking_and_rules")
        self.lm.printl(f"Removed {len(removed_null)} rows for ranking_null")
        self.lm.printl(f"Removed {len(removed_invalid)} rows for invalid_predicted_target")

        return df


    # PUBLIC
    # ------------------------------------------------------------------------------------------------------------------
    @log_method
    def initialize_enhancer(self, intialize_model: bool = True):
        self.lm.printl(f"Initializing enhancer: {self.enhancer_name} with LLM: {self.llm_name} and RAG: {self.rag}")
        if self.enhancer_name == 'medrag':
            self.en_model = MedRagInterface(self.lm, self.ch, self.icm, self.dm, self.dataset_prefix, 
                                            self.enhancer_name, self.llm_name, self.rag, self.retriever_name, self.corpus_name, 
                                            self.rag_K_documents, self.return_options, self.output_path, 
                                            self.confidence_computation, self.cuda_visible_devices, self.parallel,
                                            task=self.task)
            # Initialize MedRag model only if specified (to avoid re-initialization if it is not necessary). 
            # For instance if I want to run the clean_enhancer_df() method,
            # I must create the Enhancer object, but I do not need to initialize the MedRag model
            if intialize_model:
                self.en_model.initialize_medrag()
        elif self.enhancer_name == 'huggingface':
            self.en_model = HuggingFaceInterface(self.lm, self.ch, self.icm, self.dm, self.dataset_prefix, self.enhancer_name, self.llm_name, self.rag, 
                                                 self.return_options, self.output_path, 
                                                 self.confidence_computation, self.cuda_visible_devices, self.parallel,
                                                 task=self.task)
            if intialize_model:
                self.en_model.initialize_huggingface()
    
    @log_method
    def run_enhancer(self, df: Optional[pd.DataFrame] = None, filter_list_index: Optional[List[int]] = None):

        test_df = self.__get_test_and_select_columns(df)
        if filter_list_index is None:
            self.to_update = True
        else:
            self.to_update = False

        # Filter rows to process, setting to_update flag in the model
        self.en_model.set_to_update(self.to_update)
        rows_to_process = self.__filter_rows_to_process(test_df, filter_list_index)

        if self.enhancer_name == 'medrag':
            self.en_model.run_medrag(rows_to_process)
        elif self.enhancer_name == 'huggingface':
            self.en_model.run_huggingface(rows_to_process)
   
    @log_method
    def clean_enhancer_df(self):
        try:
            df = self.ch.read_dataframe(f"{self.output_path}{self.enhancer_name}_df.csv", dtype=dtype)
        except Exception:
            try:
                self.lm.printl(f"Warning: {self.output_path} corrupted. Trying to read with on_bad_lines='skip'")
                df = self.ch.read_dataframe(f"{self.output_path}{self.enhancer_name}_df.csv", dtype=dtype, on_bad_lines="skip")
            except Exception as e:
                self.lm.printl(f"Failed to read enhancer dataframe: {e}")
                raise e

        enhancer_cleaned_df = self.__clean_df(df)
        self.ch.save_dataframe(enhancer_cleaned_df, f"{self.output_path}{self.enhancer_name}_cleaned_df.csv")
    
        if not self.confidence_computation:
            # Compute classification report for the enhancer predictions (only test set)
            report_enhancer = classification_report(enhancer_cleaned_df[f"{self.info['target']}"], enhancer_cleaned_df[f"{self.enhancer_name}_pred"])
            self.ch.save_txt(str(report_enhancer), f"{self.output_path}{self.enhancer_name}_test.txt")
        return enhancer_cleaned_df

    @log_method
    def clean_feature_importance_ranking(self):
        """
        Cleaning of feature_importance_ranking:

        - Convert values to list (if stored as string)
        - Replace NaN with empty list []
        - Create a new cleaned column: 'feature_importance_ranking_clean'
        - If a feature is not in self.info['x_columns'], first try replacement through
        a manual mapping dictionary
        - If not found in the mapping, try fuzzy matching with
        rapidfuzz.fuzz.partial_ratio against all valid features
        - If the best similarity is >= 0.80, replace the feature with the best match
        - If no acceptable match is found, keep the original feature
        - Create a log column with the applied changes

        Status column:
        - 'correct'     : all features were already valid
        - 'replace_map' : one or more features were solved through the manual mapping
        - 'replaced'    : one or more features were solved through fuzzy matching
        - 'error'       : at least one feature could not be matched/replaced
        """

        enhancer_df = self.ch.read_dataframe(
            f"{self.dm.enhancer_path}{self.enhancer_name}_cleaned_df.csv",
            dtype=dtype
        )
        df = enhancer_df.copy()

        valid_features = list(self.info["x_columns"])
        valid_features_set = set(valid_features)
        similarity_threshold = 80.0  # rapidfuzz scores are in [0, 100]

        dataset_to_config = {
            "pima": "mapping_features_pima",
            "diabetes": "mapping_features_diabetes",
            "stroke": "mapping_features_stroke",
            "liver": "mapping_features_liver",
            "covid": "mapping_features_covid",
        }

        config_module = importlib.import_module(f"enhancer.enhancer_interfaces.utils.mapping_features.{dataset_to_config[self.dataset_prefix]}")
        # pull only what you actually need from the config
        feature_alias_map = config_module.feature_alias_map
        print(feature_alias_map)
        
        def to_list(value):
            # NaN -> []
            if pd.isna(value):
                return []

            # Already a list
            if isinstance(value, list):
                return value

            # String -> try parsing as list
            if isinstance(value, str):
                value = value.strip()
                if value == "":
                    return []
                try:
                    parsed = ast.literal_eval(value)
                    return parsed if isinstance(parsed, list) else [parsed]
                except Exception:
                    return [value]

            # Fallback
            return [value]

        def match_feature(feat):
            """
            Return:
                corrected_feats : list of final chosen features
                status         : 'correct', 'replace_map', 'replaced', or 'error'
                log_entry      : dict or None
            """
            feat = str(feat).strip()

            # 1. Exact match
            if feat in valid_features_set:
                return [feat], "correct", None

            # 2. Manual mapping
            if feat.lower() in feature_alias_map:
                mapped = feature_alias_map[feat.lower()]
                mapped_list = mapped if isinstance(mapped, list) else [mapped]

                # keep only valid mapped features
                mapped_valid = [x for x in mapped_list if x in valid_features_set]

                if len(mapped_valid) == len(mapped_list) and len(mapped_valid) > 0:
                    return mapped_valid, "replace_map", {
                        "original": feat,
                        "corrected": mapped_valid,
                        "method": "replace_map"
                    }

            # 3. Best fuzzy match
            best_feature = None
            best_score = -1.0

            for candidate in valid_features:
                score = fuzz.partial_ratio(feat, candidate, processor=utils.default_process)
                if score > best_score:
                    best_score = score
                    best_feature = candidate

            # Accept replacement if above threshold
            if best_score >= similarity_threshold:
                return [best_feature], "replaced", {
                    "original": feat,
                    "corrected": [best_feature],
                    "similarity": best_score,
                    "method": "fuzzy"
                }

            # 4. Unresolved error
            return [feat], "error", {
                "original": feat,
                "corrected": None,
                "similarity": best_score,
                "method": "unresolved"
            }

        # Convert original column to list
        df["feature_importance_ranking"] = df["feature_importance_ranking"].apply(to_list)

        cleaned_lists = []
        status_list = []
        change_logs = []

        for feature_list in df["feature_importance_ranking"]:
            cleaned = []
            row_log = []

            n_correct = 0
            n_replace_map = 0
            n_replaced = 0
            n_error = 0

            for feat in feature_list:
                corrected_feats, feat_status, log_entry = match_feature(feat)
                cleaned.extend(corrected_feats)

                if log_entry is not None:
                    row_log.append(log_entry)

                if feat_status == "correct":
                    n_correct += 1
                elif feat_status == "replace_map":
                    n_replace_map += 1
                elif feat_status == "replaced":
                    n_replaced += 1
                elif feat_status == "error":
                    n_error += 1

            # Remove duplicates preserving order
            cleaned_unique = []
            seen = set()
            for x in cleaned:
                if x not in seen:
                    cleaned_unique.append(x)
                    seen.add(x)

            # Row-level final status
            if n_error > 0:
                row_status = "error"
            elif n_replace_map > 0:
                row_status = "replace_map"
            elif n_replaced > 0:
                row_status = "replaced"
            else:
                row_status = "correct"

            cleaned_lists.append(cleaned_unique)
            status_list.append(row_status)
            change_logs.append(row_log)

        # Always create a new cleaned column
        df["feature_importance_ranking_clean"] = cleaned_lists
        df["feature_ranking_error"] = status_list
        df["feature_importance_ranking_log"] = change_logs

        self.ch.update_csv_inplace(df, f"{self.dm.enhancer_path}{self.enhancer_name}_cleaned_df.csv")

        return df
