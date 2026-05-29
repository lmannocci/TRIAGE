import os
import re
import ast
import time
import json
import glob
import numpy as np
import multiprocessing as mp
import configparser
from huggingface_hub import login
from src.medrag import MedRAG
import torch

from enhancer.enhancer_interfaces.utils.dataset_prompt_factory import DatasetPromptFactory

absolute_path = os.path.dirname(__file__)
results_path = os.path.join(absolute_path, f"..{os.sep}results{os.sep}")
data_path = os.path.join(absolute_path, f"..{os.sep}data{os.sep}")
corpus_path = os.path.join(absolute_path, f"..{os.sep}..{os.sep}corpus{os.sep}")
cache_path = os.path.join(absolute_path, f"..{os.sep}..{os.sep}..{os.sep}vast{os.sep}")
main_path = os.path.join(absolute_path, f"..{os.sep}..{os.sep}")

# Multiprocessing worker function
def medrag_worker(worker_id, gpu_group, rows, config):
    # ---- logging ----
    lm = LogManager(f"worker{worker_id}_gpus{'_'.join(map(str, gpu_group))}")
    lm.printl(f"Worker {worker_id} using GPUs {gpu_group} starting with {len(rows)} rows")

    # ---- set visible GPUs ----
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_group))

    # IMPORTANT: inside worker, GPU indices start from 0
    torch.cuda.set_device(0)
    # Compute K locally
    print_K:int = max(1, int(0.01 * len(rows)))


    medrag = MedRAG(**config["medrag_init"])
    dataset_prompt = DatasetPromptFactory.create(config["dataset_prefix"])

    for chunk_idx, (i, row) in enumerate(rows):
        with measure_time(lm, when_print='every_K', i=chunk_idx, K=print_K, label=f"[GPU {gpu_group}] Answered patient {str(chunk_idx+1)}/{str(len(rows))} - original_index: {row['original_index']}"):
            prompt, options = dataset_prompt.build_question(row, return_options=config['return_options'], enhancer_name=config['enhancer_name'], llm_name=config['medrag_init']['llm_name'])
            # lm.printl(f"[GPU {gpu_group}] Question: {question}")

            # ---- measure LLM latency (ONLY the model call) ----
            t0 = time.perf_counter()
            answer, snippets, scores = medrag.answer(prompt, options, k=config['rag_K_documents'])
            llm_seconds = time.perf_counter() - t0

            # lm.printl(f"[GPU {gpu_group}] Answer: {answer}")
            processed = process_medrag_answer(lm, config['info'], answer, snippets, scores, row, ind, i)

            df_out = pd.DataFrame([processed])
            gpu_file = f"{config['output_path']}{config['enhancer_name']}_gpu{worker_id}.csv"
            df_out.to_csv(gpu_file, mode="a", header=not os.path.exists(gpu_file), index=False)

            timing_row = {
                "worker_id": worker_id,
                "gpu_group": gpu_group,
                "chunk_idx": chunk_idx,
                "i": i,
                "original_index": row.get("original_index"),
                "llm_seconds": llm_seconds,
                # optional: makes later debugging easier
                "prompt_chars": len(prompt),
            }
            timing_df = pd.DataFrame([timing_row])
            timing_file = f"{config['output_path']}{config['enhancer_name']}_gpu{worker_id}_timing.csv"
            timing_df.to_csv(timing_file, mode="a", header=not os.path.exists(timing_file), index=False)
        

class MedRagInterface:
    def __init__(self, lm: LogManager, ch: Checkpoint, icm: IntegrityConstraintManager, dm: DirectoryManager,
                 dataset_prefix: str, enhancer_name: str, llm_name: str, rag: bool, retriever_name: str, corpus_name: str, rag_K_documents: int, return_options: bool,
                 output_path: str, confidence_computation: bool, cuda_visible_devices: str = None, parallel: bool = False):
        self.medrag: MedRAG = None
        self.ch: Checkpoint = ch
        self.lm: LogManager = lm

        self.icm: IntegrityConstraintManager = icm
        self.dm: DirectoryManager = dm

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
        self.to_update: bool = None # If True, update main dataframe; if False, save in temp path

        self.output_path: str = output_path
        self.confidence_computation: bool = confidence_computation
        self.cuda_visible_devices: str = cuda_visible_devices
        self.parallel: bool = parallel

        if not self.parallel and self.cuda_visible_devices is not None:
            # Select GPUs
            os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
            # Verify GPU selection
            self.lm.printl(f"Number of GPUs selected: {str(torch.cuda.device_count())}")
        
        self.cuda_devices = (
            [int(x) for x in self.cuda_visible_devices.split(",")]
            if self.cuda_visible_devices else [0]
        )

    @log_method
    def __login(self) -> None:
        config = configparser.ConfigParser()
        config.read(f"{main_path}config.ini")

        GITHUB_TOKEN = config.get("github", "token", fallback=None)

        if not GITHUB_TOKEN:
            raise ValueError("GitHub token not found in config.ini")
        login(GITHUB_TOKEN)

    # def __process_and_save_medrag_answer(self, answer: str, snippets: list, scores: list, row: pd.Series, ind: str, i: int, filter_list_index: Union[List[int], None]):
    #     """
    #     Process a MedRAG answer, map answer_choice -> classification,
    #     build the row dictionary, and save/update the dataframe.
        
    #     Parameters
    #     ----------
    #     answer : str
    #         The raw answer returned by MedRAG (usually a list with one string JSON).
    #     snippets : list
    #         The list of retrieved snippets/articles from MedRAG.
    #     scores : list
    #         The list of scores corresponding to the retrieved snippets.
    #     row : pd.Series
    #         The current patient row being processed.
    #     ind : str
    #         The index column name in the dataset.
    #     i : int
    #         Row index (used for logging).
    #     filter_list_index : list or None
    #         If processing a subset of rows, save in temp path; otherwise, update main dataframe.
        
    #     Returns
    #     -------
    #     new_row : dict
    #         Processed row dictionary with all relevant fields.
    #     """

    #     # Start with default row values
    #     new_row = {
    #         ind: row[ind],
    #         'medrag_pred': np.nan,
    #         self.info['target']: row[self.info['target']],
    #         'step_by_step_thinking': np.nan,
    #         'feature_importance_ranking': np.nan,
    #         'rules_applied': np.nan,
    #         'medrag_answer': answer,  # default: raw string
    #         'snippets': snippets,
    #         'scores': scores
    #     }
    #     answer_dict = None
    #     parse_method = None
    #     try:
    #         # Try strict / safe JSON-based parsing first
    #         answer_dict = self.__safe_parse_medrag_answer(answer)
    #         parse_method = "json_safe"

    #     except Exception:
    #         try:
    #             # Fallback to Python literal parsing
    #             answer_dict = ast.literal_eval(answer)
    #             parse_method = "literal_eval"

    #         except Exception:
    #             try:
    #                 # Attempt to preprocess common JSON-like issues (it should be used with LLAMA2 outputs)
    #                 answer_dict = self.__parse_medrag_string(answer)
    #                 parse_method = "parse_string"
    #             except Exception as e:
    #                 self.lm.printl(f"Failed to parse answer patient {i} - index: {row[ind]}: {e}")
    #                 new_row['parse_status'] = 'failed'
    #                 new_row['parse_method'] = None

    #     # ---------- common post-processing ----------
    #     if isinstance(answer_dict, dict):
    #         # Map MedRAG answer_choice -> classification
    #         if "answer_choice" in answer_dict:
    #             raw_choice = answer_dict.pop("answer_choice", None)
    #             answer_dict["classification"] = self.__normalize_answer_choice(raw_choice)
    #             # answer_dict["classification"] = int(answer_dict.pop("answer_choice")) # This is weak, it works only if the answer_choice is exactly "0" or "1"

    #         # Update new_row with actual values
    #         new_row.update({
    #             'medrag_pred': answer_dict.get('classification', np.nan),
    #             'step_by_step_thinking': answer_dict.get('step_by_step_thinking', np.nan),
    #             'feature_importance_ranking': answer_dict.get('feature_importance_ranking', np.nan),
    #             'rules_applied': answer_dict.get('rules_applied', np.nan),
    #             'medrag_answer': answer_dict,  # store the dictionary itself
    #             'parse_status': 'success',
    #             'parse_method': parse_method,
    #         })

    #     # Save / update dataframe
    #     if filter_list_index is None:
    #         self.ch.update_dataframe(new_row, f"{self.output_path}{self.enhancer_name}_df.csv")
    #     else:
    #         new_row_df = pd.DataFrame([new_row])
    #         self.ch.save_dataframe(
    #             new_row_df,
    #             f"{self.dm.temp_answer_path}index_{str(int(row[ind]))}_{self.enhancer_name}.csv"
    #         )

    #     return new_row


    # @log_method
    # def _run_sequential(self, df: pd.DataFrame, filter_list_index: Union[List[int], None]):
    #     self.print_K = int(0.01* len(df))  # K is the number of instances to explain, set to 10% of the test set size

    #     # Read already processed indices to avoid re-processing
    #     if os.path.exists(f"{self.output_path}{self.enhancer_name}_df.csv"):
    #         medrag_df = self.ch.read_dataframe(f"{self.output_path}{self.enhancer_name}_df.csv", dtype=dtype)
    #         processed_indices = medrag_df['original_index'].tolist() # indices already processed (in case of program interruption and restart)
    #     else:
    #         processed_indices = []
            
    #     # For each patient in the dataframe, if the patient is in the filter_list_index, ask MedRAG the answer
    #     for i, row in df.iterrows():
    #         original_index = row["original_index"]
            
    #         # Skip if already processed
    #         if original_index in processed_indices:
    #             continue
            
    #         # Skip if filter_list_index is set and this index is not included
    #         if filter_list_index is not None and original_index not in filter_list_index:
    #             continue

    #         # if filter_list_index is None or (filter_list_index is not None and row['original_index'] in filter_list_index) or (row['original_index'] not in processed_indices):
    #         with measure_time(self.lm, when_print='every_K', i=i, K=self.print_K, label=f"Answered patient {str(i+1)}/{str(len(df))} - original_index: {original_index}"):
    #             # question = self.__create_patient_question(row)
                
    #             self.dataset_prompt = DatasetPromptFactory.create(self.dataset_prefix)
    #             question, options = self.dataset_prompt.build_question(row, return_options=self.return_options)
    #             # print(question)
    #             """
    #             Ask MedRAG the answer, which has the following format. It is a tuple with 3 element:
    #             0: A string containing a dictionary with the keys required. Since it is a string, it can be evaluated with the ast.literal_eval(answer[1])
    #             1: A list of articles in dictionary format. Each article is composed by id, title and content
    #             2: A list of scores, each one assigned to the article (The number of article retrieved is equals to k=10)
    #             tuple(str(
    #                 {"step_by_step_thinking": 'xxx',
    #                  'classification': 1,
    #                  'feature_importance_ranking': ['glucose']
    #                  'rules_applied': {"Glucose": "Glucose> 99 mg/dL indicates diabetes." }
    #                 }), 
    #               List[dict['id': 'article-38794_10', 'title': 'Diabetes Mellitus Screening -- Issues of Concern', 'content': 'xxx']], 
    #               List[float])
    #             """
    #             answer, snippets, scores = self.medrag.answer(question=question, options=options, k=self.rag_K_documents)  # scores are given by the retrieval system
    #             # print(answer)
    #             # print(type(answer))
    #             self.__process_and_save_medrag_answer(answer, snippets, scores, row, ind, i, filter_list_index)



    def __export_worker_config(self):
        return {
            "dataset_prefix": self.dataset_prefix,
            "output_path": self.output_path,
            "enhancer_name": self.enhancer_name,
            "info": self.info,
            "output_path": self.output_path,
            "rag_K_documents": self.rag_K_documents,
            "return_options": self.return_options,
            'to_update': self.to_update,
            "medrag_init": {
                "llm_name": llms_map[self.llm_name],
                "rag": self.rag,
                "retriever_name": self.retriever_name,
                "corpus_name": self.corpus_name,
                "corpus_cache": True,
                "db_dir": corpus_path,
                "cache_dir": cache_path,
            }
        }
    
    def __compute_timing_metrics_non_parallel(self):
        timing_file = f"{self.output_path}{self.enhancer_name}_timing.csv"
        if not os.path.exists(timing_file):
            raise FileNotFoundError(f"Timing file not found: {timing_file}")

        df_t = self.ch.read_dataframe(timing_file, dtype=dtype)

        if "llm_seconds" not in df_t.columns:
            raise ValueError("Expected column 'llm_seconds' in timing file.")

        df_t["llm_seconds"] = pd.to_numeric(df_t["llm_seconds"], errors="coerce")
        df_t = df_t.dropna(subset=["llm_seconds"])

        x = df_t["llm_seconds"].to_numpy()
        n = int(x.size)

        metrics = {
            "rows": n,
            "total_llm_seconds": float(np.sum(x)) if n else 0.0,
            "mean_llm_seconds": float(np.mean(x)) if n else 0.0,
            "std_llm_seconds": float(np.std(x, ddof=1)) if n > 1 else 0.0,
            "min_llm_seconds": float(np.min(x)) if n else 0.0,
            "max_llm_seconds": float(np.max(x)) if n else 0.0,
            "samples_per_llm_second": (float(n / np.sum(x)) if n and np.sum(x) > 0 else 0.0),
        }

        metrics_file = f"{self.output_path}{self.enhancer_name}_timing_metrics.csv"
        pd.DataFrame([metrics]).to_csv(metrics_file, index=False)

    @log_method
    def __run_sequential(self, rows_to_process):
        print_K = int(0.01 * len(rows_to_process))
        for i, row in rows_to_process:
            original_index = row["original_index"]
            with measure_time(self.lm, when_print='every_K', i=i, K=print_K, label=f"Answered patient {str(i+1)}/{str(len(rows_to_process))} - original_index: {original_index}"):
                self.dataset_prompt = DatasetPromptFactory.create(self.dataset_prefix)
                prompt, options = self.dataset_prompt.build_question(row, self.return_options, self.enhancer_name, self.llm_name)
                t0 = time.perf_counter()
                answer, snippets, scores = self.medrag.answer(question=prompt, options=options, k=self.rag_K_documents)
                llm_seconds = time.perf_counter() - t0

                self.lm.printl(f"Original Index: {original_index}")
                self.lm.printl(f"Question: {prompt}")
                self.lm.printl(f"Answer: {answer}")
                self.lm.printl("---------------------------------")

                # save timing row (append)
                timing_row = {
                    "i": i,
                    "original_index": row.get("original_index", None),
                    "llm_seconds": llm_seconds,
                    "prompt_chars": len(prompt),   # optional but handy
                }
                timing_file = f"{self.output_path}{self.enhancer_name}_timing.csv"
                pd.DataFrame([timing_row]).to_csv(
                    timing_file,
                    mode="a",
                    header=not os.path.exists(timing_file),
                    index=False
                )

                new_row = process_medrag_answer(self.lm, self.info, answer, snippets, scores, row, ind, i)
                # Save / update dataframe
                if self.to_update:
                    self.ch.update_dataframe(new_row, f"{self.output_path}{self.enhancer_name}_df.csv")
                else:
                    new_row_df = pd.DataFrame([new_row])
                    self.ch.save_dataframe(new_row_df, f"{self.dm.temp_answer_path}index_{str(int(row[ind]))}_{self.enhancer_name}.csv")
        
        # After processing all rows, compute overall timing metrics
        self.__compute_timing_metrics_non_parallel()

    @staticmethod
    def __gpu_sort_key(filename: str) -> int:
        # Extract the GPU ID number from the filename
        match = re.search(r"_gpu(\d+)\.csv$", filename)
        return int(match.group(1)) if match else -1


    @log_method
    def __merge_parallel_outputs(self):
        files = glob.glob(f"{self.output_path}{self.enhancer_name}_gpu[0-9]*.csv")
        files = [f for f in files if not f.endswith("_timing.csv")]
        # Sort files by name (lexicographically)
        files = sorted(files, key=self.__gpu_sort_key)

        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        df.to_csv(f"{self.output_path}{self.enhancer_name}_df.csv", index=False)


    @log_method
    def __merge_parallel_timing_and_metrics(self):
        # 1) Find timing files
        timing_files = glob.glob(f"{self.output_path}{self.enhancer_name}_gpu*_timing.csv")
        timing_files = sorted(timing_files, key=self.__gpu_sort_key)

        if not timing_files:
            raise FileNotFoundError(
                f"No timing files found matching: {self.output_path}{self.enhancer_name}_gpu*_timing.csv"
            )

        # 2) Merge all timing files
        df_t = pd.concat([pd.read_csv(f) for f in timing_files], ignore_index=True)

        if "llm_seconds" not in df_t.columns:
            raise ValueError("Expected column 'llm_seconds' in timing files.")

        df_t["llm_seconds"] = pd.to_numeric(df_t["llm_seconds"], errors="coerce")
        df_t = df_t.dropna(subset=["llm_seconds"])

        x = df_t["llm_seconds"].to_numpy()
        n = int(x.size)

        # 3) Compute simple metrics (NO percentiles)
        metrics = {
            "rows": n,
            "total_llm_seconds": float(np.sum(x)) if n else 0.0,
            "mean_llm_seconds": float(np.mean(x)) if n else 0.0,
            "std_llm_seconds": float(np.std(x, ddof=1)) if n > 1 else 0.0,
            "min_llm_seconds": float(np.min(x)) if n else 0.0,
            "max_llm_seconds": float(np.max(x)) if n else 0.0,
        }

        # Throughput (model-time based)
        if metrics["total_llm_seconds"] > 0:
            metrics["samples_per_llm_second"] = n / metrics["total_llm_seconds"]
        else:
            metrics["samples_per_llm_second"] = 0.0

        # 4) Save everything in ONE CSV
        metrics_file = f"{self.output_path}{self.enhancer_name}_timing_metrics.csv"
        pd.DataFrame([metrics]).to_csv(metrics_file, index=False)

    @log_method
    def __run_parallel(self, rows_to_process):
        gpu_list = self.cuda_devices
        gpus_per_worker = 2

        # ---- build GPU groups ----
        gpu_groups = [
            gpu_list[i:i + gpus_per_worker]
            for i in range(0, len(gpu_list), gpus_per_worker)
        ]

        num_workers = len(gpu_groups)

        self.lm.printl(f"Running parallel on GPU groups: {gpu_groups}")

        # ---- split rows per worker ----
        chunks = [rows_to_process[i::num_workers] for i in range(num_workers)]

        processes = []
        for worker_id, (gpu_group, chunk) in enumerate(zip(gpu_groups, chunks)):
            p = mp.Process(target=medrag_worker, args=(worker_id, gpu_group, chunk, self.__export_worker_config()),
                name=f"Worker-{worker_id}-GPUs{'_'.join(map(str, gpu_group))}")
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            self.lm.printl(f"Process {p.name} has finished.")

        self.__merge_parallel_outputs()
        self.__merge_parallel_timing_and_metrics()



    # PUBLIC
    # ------------------------------------------------------------------------------------------------------------------
    def set_to_update(self, to_update: bool):
        self.to_update = to_update

    @log_method
    def initialize_medrag(self):
        if self.parallel:
            return  # workers initialize themselves
        # Login to HuggingFace
        self.__login()
        # Initialize MedRAG
        
        with measure_time(self.lm, label="initialize_medrag"):
            # "mistralai/Mixtral-8x7B-Instruct-v0.1"
            self.medrag: MedRAG = MedRAG(llm_name=llms_map[self.llm_name], rag=self.rag,
                                         retriever_name=self.retriever_name,
                                         corpus_name=self.corpus_name, corpus_cache=True, db_dir=corpus_path, cache_dir=cache_path)

    @log_method
    def run_medrag(self, rows_to_process: List[Tuple[int, pd.Series]]):
        # ============================
        # 1. PREPARATION (shared)
        # ============================

        # Load checkpoint
        # self.lm.printl(f"{self.output_path}{self.enhancer_name}_df.csv")
        # if os.path.exists(f"{self.output_path}{self.enhancer_name}_df.csv"):
        #     medrag_df = self.ch.read_dataframe(f"{self.output_path}{self.enhancer_name}_df.csv", dtype=dtype)
        #     processed_indices = set(medrag_df['original_index'].tolist())
        # else:
        #     processed_indices = set()

        # # Build work list
        # rows_to_process = []
        # for i, row in df.iterrows():
        #     original_index = row["original_index"]

        #     if original_index in processed_indices:
        #         continue

        #     if filter_list_index is not None and original_index not in filter_list_index:
        #         continue

        #     rows_to_process.append((i, row))

        # if len(rows_to_process) == 0:
        #     self.lm.printl("No rows to process.")
        #     return

        # ============================
        # 2. EXECUTION
        # ============================

        if not self.parallel:
            self.__run_sequential(rows_to_process)
        else:
            self.__run_parallel(rows_to_process)

    # @log_method
    # def clean_medrag_df(self, df: pd.DataFrame):
    #     # 1. empty_ranking_and_rules (feature_importance_ranking == '[]')
    #     mask_empty = df['feature_importance_ranking'] == '[]'
    #     removed_empty = pd.DataFrame({
    #         ind: df[mask_empty][ind],
    #         "motivation": "empty_ranking_and_rules"
    #     })
    #     df = df[~mask_empty]  # drop them

    #     # 2. ranking_null (feature_importance_ranking is NaN)
    #     mask_null = df['feature_importance_ranking'].isnull()
    #     removed_null = pd.DataFrame({
    #         ind: df[mask_null][ind],
    #         "motivation": "ranking_null"
    #     })
    #     df = df[~mask_null]  # drop them

    #     # 3. invalid predicted target
    #     valid_values = df['Outcome'].unique()
    #     mask_invalid_pred = ~df['medrag_pred'].isin(valid_values)
    #     removed_invalid = pd.DataFrame({
    #         ind: df[mask_invalid_pred][ind],
    #         "motivation": "invalid_predicted_target"
    #     })
    #     df = df[~mask_invalid_pred]  # drop them

    #     # 4. Concatenate all removed indices
    #     removed_all = pd.concat([removed_empty, removed_null, removed_invalid], ignore_index=True)

    #     # 5. Save them
    #     self.ch.save_dataframe(removed_all, f"{self.output_path}{self.enhancer_name}_removed_error_idx.csv")

    #     # Debug info
    #     self.lm.printl(f"Removed {len(removed_empty)} rows for empty_ranking_and_rules")
    #     self.lm.printl(f"Removed {len(removed_null)} rows for ranking_null")
    #     self.lm.printl(f"Removed {len(removed_invalid)} rows for invalid_predicted_target")

    #     return df
        
