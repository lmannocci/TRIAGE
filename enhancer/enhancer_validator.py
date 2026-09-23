import ast
import glob
import os
import numpy as np
from typing import List, Optional, Tuple

import pandas as pd

from enhancer.enhancer_interfaces.tasks import EnhancerValidationTask
from enhancer.enhancer_interfaces.utils.dataset_prompt_factory import DatasetPromptFactory

from utils.checkpoint.checkpoint import Checkpoint
from utils.common_variables import df_info, dtype, ind
from utils.decorator_definition import log_method
from utils.directory_manager.directory_manager import DirectoryManager
from utils.log_manager.log_manager import LogManager


class EnhancerJudgeValidator:
    """
    Build and run the LLM-as-a-judge validation dataset for final enhancer outputs.

    The validator intentionally lives outside Enhancer because its inputs come
    from multiple pipeline stages: patient data, explainer outputs, and cleaned
    enhancer outputs.
    """

    validator_name = "llm_judge"

    def __init__(
        self,
        ch: Checkpoint,
        lm: LogManager,
        results_path: str,
        dataset_prefix: str,
        model_name: str = "catboost",
        model_prefix: str = "balanced",
        explainer_name: str = "shap",
        enhancer_name: str = "medrag",
        enhancer_llm_name: str = "llama2",
        rag: bool = True,
        retriever_name: str = "MedCPT",
        corpus_name: str = "StatPearls",
        judge_enhancer_name: str = "huggingface",
        judge_llm_name: str = "mixtral",
        judge_rag: bool = False,
        judge_retriever_name: str = "MedCPT",
        judge_corpus_name: str = "StatPearls",
        judge_return_options: bool = False,
        judge_rag_K_documents: int = 10,
        top_k: Optional[int] = None,
        cuda_visible_devices: Optional[str] = None,
        parallel: bool = False,
        rerun_existing: bool = False,
    ):
        self.ch = ch
        self.lm = lm
        self.dm = DirectoryManager(
            lm=lm,
            results_path=results_path,
            dataset_prefix=dataset_prefix,
            model_name=model_name,
            model_prefix=model_prefix,
            explainer_name=explainer_name,
            enhancer_name=enhancer_name,
            llm_name=enhancer_llm_name,
            rag=rag,
            retriever_name=retriever_name,
            corpus_name=corpus_name,
            top_k=top_k,
            judge_enhancer_name=judge_enhancer_name,
            judge_llm_name=judge_llm_name,
            judge_rag=judge_rag,
            judge_retriever_name=judge_retriever_name,
            judge_corpus_name=judge_corpus_name,
        )
        self.dataset_prefix = dataset_prefix
        self.info = df_info[self.dataset_prefix]

        self.model_name = model_name
        self.model_prefix = model_prefix
        self.explainer_name = explainer_name
        self.enhancer_name = enhancer_name
        self.enhancer_llm_name = enhancer_llm_name
        self.rag = rag
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        self.judge_enhancer_name = judge_enhancer_name
        self.judge_llm_name = judge_llm_name
        self.judge_rag = judge_rag
        self.judge_retriever_name = judge_retriever_name
        self.judge_corpus_name = judge_corpus_name
        self.judge_return_options = judge_return_options
        self.judge_rag_K_documents = judge_rag_K_documents
        self.top_k = top_k
        self.cuda_visible_devices = cuda_visible_devices
        self.parallel = parallel
        self.rerun_existing = rerun_existing

        self.dataset_path = self.dm.dataset_path
        self.explainer_path = f"{self.dm.explainer_path}{self.explainer_name}_df.csv"
        self.enhancer_path = self.dm.enhancer_path
        self.enhancer_df_path = f"{self.enhancer_path}{self.enhancer_name}_cleaned_df.csv"

        # per-variant output directories inside the validator folder
        self.output_path_explanation = self.dm.explaination_validator_path
        self.output_path_rag = self.dm.rag_validator_path

    @log_method
    def run(self):
        return self._run_judge_variant("explanation")

    @log_method
    def run_rag_judge(self):
        return self._run_judge_variant("retrieved_documents")

    def _run_judge_variant(self, evaluation_type: str):
        output_file = self._judge_output_file(evaluation_type, "df")
        if self.rerun_existing:
            self._clear_judge_variant_outputs(evaluation_type)

        validation_df = self._build_validation_dataframe(evaluation_type)
        rows_to_process = self._filter_rows_to_process(validation_df, output_file)
        if len(rows_to_process) == 0:
            self.lm.printl("No judge validation rows to process.")
            return output_file

        task = EnhancerValidationTask(
            judge_name=self._judge_file_prefix(evaluation_type),
            evaluation_type=evaluation_type,
        )
        interface = self._create_judge_interface(task, evaluation_type)
        interface.set_to_update(True)
        if self.judge_enhancer_name == "huggingface":
            interface.initialize_huggingface()
            interface.run_huggingface(rows_to_process)
        elif self.judge_enhancer_name == "medrag":
            interface.initialize_medrag()
            interface.run_medrag(rows_to_process)
        else:
            raise ValueError(f"Unknown judge enhancer: {self.judge_enhancer_name}")
        return output_file

    def _create_judge_interface(self, task: EnhancerValidationTask, evaluation_type: str):
        if self.judge_enhancer_name == "huggingface":
            from enhancer.enhancer_interfaces.huggingface_interface import HuggingFaceInterface
            out_path = self.output_path_rag if evaluation_type == "retrieved_documents" else self.output_path_explanation
            return HuggingFaceInterface(
                lm=self.lm,
                ch=self.ch,
                icm=None,
                dm=self.dm,
                dataset_prefix=self.dataset_prefix,
                enhancer_name=self.validator_name,
                llm_name=self.judge_llm_name,
                rag=self.judge_rag,
                return_options=self.judge_return_options,
                output_path=out_path,
                confidence_computation=False,
                cuda_visible_devices=self.cuda_visible_devices,
                parallel=self.parallel,
                task=task,
            )

        if self.judge_enhancer_name == "medrag":
            from enhancer.enhancer_interfaces.medrag_interface import MedRagInterface

            out_path = self.output_path_rag if evaluation_type == "retrieved_documents" else self.output_path_explanation
            return MedRagInterface(
                lm=self.lm,
                ch=self.ch,
                icm=None,
                dm=self.dm,
                dataset_prefix=self.dataset_prefix,
                enhancer_name=self.validator_name,
                llm_name=self.judge_llm_name,
                rag=self.judge_rag,
                retriever_name=self.judge_retriever_name,
                corpus_name=self.judge_corpus_name,
                rag_K_documents=self.judge_rag_K_documents,
                return_options=self.judge_return_options,
                output_path=out_path,
                confidence_computation=False,
                cuda_visible_devices=self.cuda_visible_devices,
                parallel=self.parallel,
                task=task,
            )

        raise ValueError(f"Unknown judge enhancer: {self.judge_enhancer_name}")

    def _clear_judge_variant_outputs(self, evaluation_type: str) -> None:
        out_path = self.output_path_rag if evaluation_type == "retrieved_documents" else self.output_path_explanation
        patterns = [
            f"{out_path}{self._judge_file_prefix(evaluation_type)}_*.csv",
            f"{out_path}{self.validator_name}_gpu*.csv",
            f"{out_path}{self.validator_name}_gpu*_timing.csv",
            f"{out_path}{self.validator_name}_timing.csv",
            f"{out_path}{self.validator_name}_timing_metrics.csv",
        ]

        removed = 0
        for pattern in patterns:
            for path in glob.glob(pattern):
                if os.path.isfile(path):
                    os.remove(path)
                    removed += 1

        self.lm.printl(f"Cleared {removed} existing judge output files for {evaluation_type}.")

    def _build_validation_dataframe(self, evaluation_type: str = "explanation") -> pd.DataFrame:
        patient_df = self._load_patient_dataframe()
        explainer_df = self.ch.read_dataframe(self.explainer_path, dtype=dtype)
        enhancer_df = self.ch.read_dataframe(self.enhancer_df_path, dtype=dtype)

        explainer_ranking_column = f"{self.explainer_name}_feature_importance_ranking"
        enhancer_pred_column = f"{self.enhancer_name}_pred"

        if evaluation_type == "retrieved_documents":
            self._require_columns(explainer_df, [ind], self.explainer_path)
            self._require_columns(
                enhancer_df,
                [ind, enhancer_pred_column, "step_by_step_thinking", "snippets"],
                self.enhancer_df_path,
            )

            df = patient_df.merge(
                explainer_df[[ind]],
                on=ind,
                how="inner",
            )
            df = df.merge(
                enhancer_df[[ind, enhancer_pred_column, "step_by_step_thinking", "snippets"]],
                on=ind,
                how="inner",
            )

            dataset_prompt = DatasetPromptFactory.create(self.dataset_prefix)
            df["enhancer_prediction"] = df[enhancer_pred_column].apply(
                lambda value: self._format_prediction(value, dataset_prompt)
            )
            df["retrieved_documents"] = df["snippets"].apply(self._format_retrieved_documents)
            df["generated_explanation"] = df["step_by_step_thinking"].fillna("")
            self.lm.printl(
                f"Prepared {len(df)} retrieved-documents judge rows for {self.dataset_prefix}."
            )
            return df

        self._require_columns(explainer_df, [ind, explainer_ranking_column], self.explainer_path)
        self._require_columns(
            enhancer_df,
            [ind, enhancer_pred_column, "step_by_step_thinking"],
            self.enhancer_df_path,
        )

        df = patient_df.merge(
            explainer_df[[ind, explainer_ranking_column]],
            on=ind,
            how="inner",
        )
        df = df.merge(
            enhancer_df[[ind, enhancer_pred_column, "step_by_step_thinking"]],
            on=ind,
            how="inner",
        )

        dataset_prompt = DatasetPromptFactory.create(self.dataset_prefix)
        df["patient_description"] = df.apply(dataset_prompt.build_patient_description, axis=1)
        df["explainer_ranking"] = df[explainer_ranking_column].apply(self._format_explainer_ranking)
        df["enhancer_prediction"] = df[enhancer_pred_column].apply(
            lambda value: self._format_prediction(value, dataset_prompt)
        )
        df["generated_explanation"] = df["step_by_step_thinking"].fillna("")

        self.lm.printl(f"Prepared {len(df)} judge validation rows for {self.dataset_prefix}.")
        return df

    def _load_patient_dataframe(self) -> pd.DataFrame:
        dataset = f"{self.dataset_prefix}_preprocessed.csv"
        df_original = self.ch.read_dataframe(f"{self.dm.data_path}{dataset}", dtype=dtype)
        test_idx = self.ch.read_dataframe(f"{self.dataset_path}test_idx.csv", dtype=dtype)[ind]
        test_df = df_original[df_original[ind].isin(test_idx)].copy()

        if self.dataset_prefix == "diabetes":
            sampled_idx_path = f"{self.dataset_path}sampled_test_idx.csv"
            sampled_test_idx = self.ch.read_dataframe(sampled_idx_path, dtype=dtype)[ind]
            test_df = test_df[test_df[ind].isin(sampled_test_idx)].copy()

        num_cols = test_df.select_dtypes(include="number").columns
        test_df[num_cols] = test_df[num_cols].round(2)

        if self.dataset_prefix in {"stroke", "liver"}:
            label_encoders = self.ch.read_object(f"{self.dm.data_path}{self.dataset_prefix}_label_encoders.pkl")
            for col, le in label_encoders.items():
                if col in test_df.columns:
                    test_df[col] = le.inverse_transform(test_df[col])

        return test_df.reset_index(drop=True)

    def _filter_rows_to_process(self, df: pd.DataFrame, output_file: str) -> List[Tuple[int, pd.Series]]:
        if os.path.exists(output_file):
            judge_df = self.ch.read_dataframe(output_file, dtype=dtype)
            processed_indices = set(judge_df[ind].tolist())
        else:
            processed_indices = set()

        rows_to_process = [
            (i, row)
            for i, row in df.iterrows()
            if row[ind] not in processed_indices
        ]
        self.lm.printl(f"Number of judge validation rows to process: {len(rows_to_process)}")
        return rows_to_process

    def _format_explainer_ranking(self, value):
        ranking = self._parse_list(value)
        if self.top_k is not None:
            ranking = ranking[:self.top_k]
        return ranking

    def _format_retrieved_documents(self, snippets_value):
        snippets = self._parse_list(snippets_value)
        if not snippets:
            return ""

        formatted_documents = []
        for index, snippet in enumerate(snippets, start=1):
            if isinstance(snippet, dict):
                title = snippet.get("title") or snippet.get("doc_title") or snippet.get("source") or ""
                content = snippet.get("content") or snippet.get("text") or snippet.get("snippet") or ""
                if title and content:
                    formatted_documents.append(f"Document [{index}] (Title: {title}) {content}")
                elif content:
                    formatted_documents.append(f"Document [{index}] {content}")
                else:
                    formatted_documents.append(f"Document [{index}] {snippet}")
            else:
                formatted_documents.append(f"Document [{index}] {snippet}")

        return "\n".join(formatted_documents)

    @staticmethod
    def _parse_list(value):
        if isinstance(value, list):
            return value
        if pd.isna(value):
            return []
        if isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)
                return parsed if isinstance(parsed, list) else [parsed]
            except Exception:
                return [value]
        return [value]

    @staticmethod
    def _format_prediction(value, dataset_prompt) -> str:
        try:
            prediction = int(float(value))
        except (TypeError, ValueError):
            return str(value)

        if prediction == 1:
            return f"1 ({dataset_prompt.POSITIVE_LABEL})"
        if prediction == 0:
            return f"0 ({dataset_prompt.NEGATIVE_LABEL})"
        return str(value)

    @staticmethod
    def _require_columns(df: pd.DataFrame, columns: List[str], path: str) -> None:
        missing = [column for column in columns if column not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in {path}: {missing}")

    def _judge_file_prefix(self, evaluation_type: str) -> str:
        return self.validator_name

    def _judge_output_file(self, evaluation_type: str, suffix: str) -> str:
        out_path = self.output_path_rag if evaluation_type == "retrieved_documents" else self.output_path_explanation
        return f"{out_path}{self._judge_file_prefix(evaluation_type)}_{suffix}.csv"

    def _judge_metric_fields(self, evaluation_type: str) -> List[str]:
        if evaluation_type == "retrieved_documents":
            return ["retrieval_relevance", "grounding", "unsupported_claims", "overall_score"]
        return ["feature_faithfulness", "prediction_consistency", "unsupported_claims", "clarity", "overall_score"]

    def _judge_output_fields(self, evaluation_type: str) -> List[str]:
        if evaluation_type == "retrieved_documents":
            return ["retrieval_relevance", "grounding", "unsupported_claims", "overall_score", "short_reason"]
        return ["feature_faithfulness", "prediction_consistency", "unsupported_claims", "clarity", "overall_score", "short_reason"]

    def _judge_key_cols(self) -> List[str]:
        return [
            "dataset_prefix",
            "model_name",
            "model_prefix",
            "explainer_name",
            "enhancer_name",
            "enhancer_llm_name",
            "rag",
            "retriever_name",
            "corpus_name",
            "judge_enhancer_name",
            "judge_llm_name",
            "judge_rag",
            "judge_retriever_name",
            "judge_corpus_name",
        ]

    def _clean_judge_variant(self, evaluation_type: str) -> None:
        input_file = self._judge_output_file(evaluation_type, "df")

        if not os.path.exists(input_file):
            self.lm.printl(f"[Warning] Input file not found: {input_file}")
            return

        df = self.ch.read_dataframe(input_file, dtype=dtype)
        output_fields = self._judge_output_fields(evaluation_type)
        score_fields = [field for field in output_fields if field != "short_reason"]

        problem_rows = []
        clean_mask = pd.Series(True, index=df.index)
        df_copy = df.copy()

        for idx, row in df.iterrows():
            problematic_fields = []

            for field in output_fields:
                if field not in df.columns:
                    problematic_fields.append(f"{field}(missing_column)")
                    continue

                value = row[field]

                if pd.isna(value):
                    problematic_fields.append(field)
                    continue
                if isinstance(value, str) and value.strip() == "":
                    problematic_fields.append(field)
                    continue

                if field in score_fields:
                    try:
                        num = float(value)
                    except Exception:
                        continue

                    if num < 1:
                        df_copy.at[idx, field] = 1.0
                    elif num > 5:
                        df_copy.at[idx, field] = 5.0
                    else:
                        df_copy.at[idx, field] = float(num)

            if problematic_fields:
                clean_mask.at[idx] = False
                problem_rows.append({
                    ind: row[ind] if ind in df.columns else np.nan,
                    "problematic_fields": ", ".join(problematic_fields),
                })

        cleaned_df = df_copy[clean_mask].copy()
        cleaned_output = self._judge_output_file(evaluation_type, "cleaned_df")
        self.ch.save_dataframe(cleaned_df, cleaned_output)
        self.lm.printl(
            f"Saved {len(cleaned_df)} cleaned rows to {cleaned_output} "
            f"(removed {len(problem_rows)} rows with issues)"
        )

        if problem_rows:
            error_df = pd.DataFrame(problem_rows)
            error_output = self._judge_output_file(evaluation_type, "removed_error_idx")
            self.ch.save_dataframe(error_df, error_output)
            self.lm.printl(f"Saved {len(error_df)} removed indices to {error_output}")
        else:
            self.lm.printl("No rows removed - all data is clean.")

    def _summarize_judge_variant(self, evaluation_type: str, save_global_evaluator: bool = True) -> None:
        cleaned_file = self._judge_output_file(evaluation_type, "cleaned_df")
        original_file = self._judge_output_file(evaluation_type, "df")
        error_idx_file = self._judge_output_file(evaluation_type, "removed_error_idx")

        if not os.path.exists(cleaned_file):
            self.lm.printl(f"[Warning] Cleaned file not found: {cleaned_file}. Run the cleaner first.")
            return

        df = self.ch.read_dataframe(cleaned_file, dtype=dtype)
        if df.empty:
            self.lm.printl("[Warning] Cleaned dataframe is empty. Skipping aggregation.")
            return

        n_total_before_cleaning = 0
        if os.path.exists(original_file):
            original_df = self.ch.read_dataframe(original_file, dtype=dtype)
            n_total_before_cleaning = len(original_df)

        n_removed = 0
        if os.path.exists(error_idx_file):
            error_df = self.ch.read_dataframe(error_idx_file, dtype=dtype)
            n_removed = len(error_df)

        discard_ratio = (n_removed / n_total_before_cleaning) if n_total_before_cleaning > 0 else 0.0

        metric_fields = self._judge_metric_fields(evaluation_type)
        new_row = {
            "dataset_prefix": self.dataset_prefix,
            "model_name": self.model_name,
            "model_prefix": self.model_prefix,
            "explainer_name": self.explainer_name,
            "enhancer_name": self.enhancer_name,
            "enhancer_llm_name": self.enhancer_llm_name,
            "rag": self.rag,
            "retriever_name": self.retriever_name,
            "corpus_name": self.corpus_name,
            "judge_enhancer_name": self.judge_enhancer_name,
            "judge_llm_name": self.judge_llm_name,
            "judge_rag": self.judge_rag,
            "judge_retriever_name": self.judge_retriever_name,
            "judge_corpus_name": self.judge_corpus_name,
            "judge_variant": evaluation_type,
            "n_total_before_cleaning": n_total_before_cleaning,
            "n_removed": n_removed,
            "discard_ratio": discard_ratio,
            "n_total": len(df),
        }

        for field in metric_fields:
            if field not in df.columns:
                self.lm.printl(f"[Warning] Column {field} not found in cleaned data.")
                new_row[f"{field}_mean"] = None
                new_row[f"{field}_median"] = None
                new_row[f"{field}_std"] = None
                continue

            series = pd.to_numeric(df[field], errors="coerce")
            valid_series = series.dropna()
            new_row[f"{field}_mean"] = valid_series.mean() if not valid_series.empty else None
            new_row[f"{field}_median"] = valid_series.median() if not valid_series.empty else None
            new_row[f"{field}_std"] = valid_series.std() if not valid_series.empty else None

        key_cols = self._judge_key_cols()
        path_agg = (
            f"{self.dm.validator_rag_global_path}aggregated_llm_judge_metrics.csv"
            if evaluation_type == "retrieved_documents" and save_global_evaluator
            else f"{self.dm.validator_explanation_global_path}aggregated_llm_judge_metrics.csv"
            if save_global_evaluator
            else self._judge_output_file(evaluation_type, "aggregated_metrics")
        )

        if not os.path.exists(os.path.dirname(path_agg)):
            os.makedirs(os.path.dirname(path_agg), exist_ok=True)

        if os.path.exists(path_agg):
            agg_df = self.ch.read_dataframe(path_agg, dtype=dtype)
        else:
            agg_df = pd.DataFrame(columns=list(new_row.keys()))

        agg_df = pd.concat([agg_df, pd.DataFrame([new_row])], ignore_index=True)
        agg_df = agg_df.drop_duplicates(subset=key_cols, keep="last")
        self.ch.save_dataframe(agg_df, path_agg)
        self.lm.printl(f"Saved aggregated metrics to {path_agg}")

    @log_method
    def clean_llm_judge(self) -> None:
        self._clean_judge_variant("explanation")

    @log_method
    def clean_llm_rag_judge(self) -> None:
        self._clean_judge_variant("retrieved_documents")

    @log_method
    def summarize_llm_judge_metrics(self, save_global_evaluator: bool = True) -> None:
        self._summarize_judge_variant("explanation", save_global_evaluator=save_global_evaluator)

    @log_method
    def summarize_llm_rag_judge_metrics(self, save_global_evaluator: bool = True) -> None:
        self._summarize_judge_variant("retrieved_documents", save_global_evaluator=save_global_evaluator)
