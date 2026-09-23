from __future__ import annotations

from abc import ABC, abstractmethod
import glob
import multiprocessing as mp
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
import torch

from enhancer.enhancer_interfaces.backends import LLMBackend
from enhancer.enhancer_interfaces.tasks import (
    DatasetClassificationTask,
    LLMTask,
    create_task_from_config,
)
from utils.checkpoint.checkpoint import Checkpoint
from utils.common_variables import df_info, dtype, ind
from utils.decorator_definition import log_method, measure_time
from utils.directory_manager.directory_manager import DirectoryManager
from utils.integrity_constraint_manager.integrity_constraint_manager import IntegrityConstraintManager
from utils.log_manager.log_manager import LogManager


def _append_dataframe_row(row: Dict[str, Any], path: str) -> None:
    df = pd.DataFrame([row])
    if os.path.exists(path):
        existing_columns = pd.read_csv(path, nrows=0).columns.tolist()
        extra_columns = [column for column in df.columns if column not in existing_columns]
        if extra_columns:
            raise ValueError(
                f"Cannot append row with new columns to {path}: {extra_columns}. "
                "Initialize these columns with NaN in the task output schema."
            )
        df = df.reindex(columns=existing_columns)

    df.to_csv(
        path,
        mode="a",
        header=not os.path.exists(path),
        index=False,
    )


def _timing_metrics(df_t: pd.DataFrame) -> Dict[str, float]:
    if "llm_seconds" not in df_t.columns:
        raise ValueError("Expected column 'llm_seconds' in timing file.")

    df_t = df_t.copy()
    df_t["llm_seconds"] = pd.to_numeric(df_t["llm_seconds"], errors="coerce")
    df_t = df_t.dropna(subset=["llm_seconds"])

    x = df_t["llm_seconds"].to_numpy()
    n = int(x.size)
    total = float(np.sum(x)) if n else 0.0

    return {
        "rows": n,
        "total_llm_seconds": total,
        "mean_llm_seconds": float(np.mean(x)) if n else 0.0,
        "std_llm_seconds": float(np.std(x, ddof=1)) if n > 1 else 0.0,
        "min_llm_seconds": float(np.min(x)) if n else 0.0,
        "max_llm_seconds": float(np.max(x)) if n else 0.0,
        "samples_per_llm_second": float(n / total) if n and total > 0 else 0.0,
    }


def _gpu_sort_key(filename: str) -> int:
    match = re.search(r"_gpu(\d+)(?:_timing)?\.csv$", filename)
    return int(match.group(1)) if match else -1


def process_rows_with_backend(
    *,
    lm: LogManager,
    rows_to_process: List[Tuple[int, pd.Series]],
    task: LLMTask,
    backend: LLMBackend,
    output_path: str,
    enhancer_name: str,
    index_column: str = ind,
    ch: Optional[Checkpoint] = None,
    dm: Optional[DirectoryManager] = None,
    to_update: Optional[bool] = None,
    worker_id: Optional[int] = None,
    gpu_group: Optional[List[int]] = None,
) -> None:
    total_rows = len(rows_to_process)
    print_k = max(1, int(0.01 * total_rows))

    for chunk_idx, (i, row) in enumerate(rows_to_process):
        original_index = row.get("original_index", None)
        prefix = f"[GPU {gpu_group}] " if gpu_group is not None else ""
        label = (
            f"{prefix}Answered patient {chunk_idx + 1}/{total_rows} "
            f"- original_index: {original_index}"
        )

        with measure_time(lm, when_print="every_K", i=chunk_idx, K=print_k, label=label):
            payload = task.build_prompt(row, backend.provider_name)
            model_prompt = backend.prepare_prompt(payload)
            # lm.printl(f"Prompt: {model_prompt.prompt}")
            t0 = time.perf_counter()
            answer, snippets, scores = backend.answer(model_prompt)
            # lm.printl(f"Answer: {answer}")
            llm_seconds = time.perf_counter() - t0

            new_row = task.process_answer(
                lm=lm,
                answer=answer,
                snippets=snippets,
                scores=scores,
                row=row,
                row_position=i,
                index_column=index_column,
            )

            if worker_id is None:
                if ch is None or dm is None:
                    raise ValueError("Sequential execution requires checkpoint and directory managers.")

                if to_update:
                    ch.update_dataframe(new_row, f"{output_path}{enhancer_name}_df.csv")
                else:
                    ch.save_dataframe(
                        pd.DataFrame([new_row]),
                        f"{dm.temp_answer_path}index_{str(int(row[index_column]))}_{enhancer_name}.csv",
                    )

                timing_file = f"{output_path}{enhancer_name}_timing.csv"
                timing_row = {
                    "i": i,
                    "original_index": original_index,
                    "llm_seconds": llm_seconds,
                    "prompt_chars": len(model_prompt.prompt),
                }
                _append_dataframe_row(timing_row, timing_file)
            else:
                result_file = f"{output_path}{enhancer_name}_gpu{worker_id}.csv"
                _append_dataframe_row(new_row, result_file)

                timing_file = f"{output_path}{enhancer_name}_gpu{worker_id}_timing.csv"
                timing_row = {
                    "worker_id": worker_id,
                    "gpu_group": gpu_group,
                    "chunk_idx": chunk_idx,
                    "i": i,
                    "original_index": original_index,
                    "llm_seconds": llm_seconds,
                    "prompt_chars": len(model_prompt.prompt),
                }
                _append_dataframe_row(timing_row, timing_file)


def run_parallel_worker(
    worker_id: int,
    gpu_group: List[int],
    rows: List[Tuple[int, pd.Series]],
    config: Dict[str, Any],
    backend_cls: Type[LLMBackend],
) -> None:
    lm = LogManager(f"worker{worker_id}_gpus{'_'.join(map(str, gpu_group))}")
    lm.printl(f"Worker {worker_id} using GPUs {gpu_group} starting with {len(rows)} rows")

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_group))
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    backend = backend_cls.from_config(config["backend"], lm)
    backend.initialize()
    task = create_task_from_config(config["task"])

    process_rows_with_backend(
        lm=lm,
        rows_to_process=rows,
        task=task,
        backend=backend,
        output_path=config["output_path"],
        enhancer_name=config["enhancer_name"],
        index_column=config.get("index_column", ind),
        worker_id=worker_id,
        gpu_group=gpu_group,
    )


class BaseLLMInterface(ABC):
    provider_name: str
    backend_cls: Type[LLMBackend]
    worker_target: Callable
    gpus_per_worker = 2

    def __init__(
        self,
        lm: LogManager,
        ch: Checkpoint,
        icm: IntegrityConstraintManager,
        dm: DirectoryManager,
        dataset_prefix: str,
        enhancer_name: str,
        llm_name: str,
        return_options: bool,
        output_path: str,
        confidence_computation: bool,
        cuda_visible_devices: Optional[str] = None,
        parallel: bool = False,
        task: Optional[LLMTask] = None,
    ):
        self.backend: Optional[LLMBackend] = None
        self.ch = ch
        self.lm = lm
        self.icm = icm
        self.dm = dm

        self.dataset_prefix = dataset_prefix
        self.dataset = f"{self.dataset_prefix}_preprocessed.csv"
        self.info = df_info[self.dataset_prefix]

        self.enhancer_name = enhancer_name
        self.llm_name = llm_name
        self.return_options = return_options
        self.to_update = None

        self.output_path = output_path
        self.confidence_computation = confidence_computation
        self.cuda_visible_devices = cuda_visible_devices
        self.parallel = parallel

        self.task = task or DatasetClassificationTask(
            dataset_prefix=self.dataset_prefix,
            enhancer_name=self.enhancer_name,
            llm_name=self.llm_name,
            info=self.info,
            return_options=self.return_options,
        )

        if not self.parallel and self.cuda_visible_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
            self.lm.printl(f"Number of GPUs selected: {str(torch.cuda.device_count())}")

        self.cuda_devices = (
            [int(x) for x in self.cuda_visible_devices.split(",")]
            if self.cuda_visible_devices
            else [0]
        )

    @abstractmethod
    def _backend_config(self) -> Dict[str, Any]:
        pass

    def _create_backend(self) -> LLMBackend:
        return self.backend_cls.from_config(self._backend_config(), self.lm)

    def _export_worker_config(self) -> Dict[str, Any]:
        return {
            "dataset_prefix": self.dataset_prefix,
            "output_path": self.output_path,
            "enhancer_name": self.enhancer_name,
            "return_options": self.return_options,
            "to_update": self.to_update,
            "llm_name": self.llm_name,
            "index_column": ind,
            "task": self.task.to_config(),
            "backend": self._backend_config(),
        }

    @log_method
    def initialize_model(self) -> None:
        if self.parallel:
            return
        with measure_time(self.lm, label=f"initialize_{self.provider_name}"):
            self.backend = self._create_backend()
            self.backend.initialize()

    def set_to_update(self, to_update: bool) -> None:
        self.to_update = to_update

    @log_method
    def run_task(self, rows_to_process: List[Tuple[int, pd.Series]]) -> None:
        if len(rows_to_process) == 0:
            self.lm.printl("No rows to process.")
            return

        if self.parallel:
            self._run_parallel(rows_to_process)
        else:
            self._run_sequential(rows_to_process)

    @log_method
    def _run_sequential(self, rows_to_process: List[Tuple[int, pd.Series]]) -> None:
        if self.backend is None:
            self.initialize_model()

        process_rows_with_backend(
            lm=self.lm,
            rows_to_process=rows_to_process,
            task=self.task,
            backend=self.backend,
            output_path=self.output_path,
            enhancer_name=self.enhancer_name,
            index_column=ind,
            ch=self.ch,
            dm=self.dm,
            to_update=self.to_update,
        )

        self._compute_timing_metrics_non_parallel()

    @log_method
    def _run_parallel(self, rows_to_process: List[Tuple[int, pd.Series]]) -> None:
        gpu_groups = [
            self.cuda_devices[i:i + self.gpus_per_worker]
            for i in range(0, len(self.cuda_devices), self.gpus_per_worker)
        ]
        num_workers = len(gpu_groups)
        self.lm.printl(f"Running parallel on GPU groups: {gpu_groups}")

        chunks = [rows_to_process[i::num_workers] for i in range(num_workers)]
        assignments = [
            (worker_id, gpu_group, chunk)
            for worker_id, (gpu_group, chunk) in enumerate(zip(gpu_groups, chunks))
            if len(chunk) > 0
        ]

        processes = []
        worker_config = self._export_worker_config()
        for worker_id, gpu_group, chunk in assignments:
            p = mp.Process(
                target=self.worker_target,
                args=(worker_id, gpu_group, chunk, worker_config),
                name=f"Worker-{worker_id}-GPUs{'_'.join(map(str, gpu_group))}",
            )
            p.start()
            processes.append(p)

        failed_processes = []
        for p in processes:
            p.join()
            self.lm.printl(f"Process {p.name} has finished.")
            if p.exitcode != 0:
                failed_processes.append((p.name, p.exitcode))

        if failed_processes:
            raise RuntimeError(f"Parallel workers failed: {failed_processes}")

        self._merge_parallel_outputs()
        self._merge_parallel_timing_and_metrics()

    def _compute_timing_metrics_non_parallel(self) -> None:
        timing_file = f"{self.output_path}{self.enhancer_name}_timing.csv"
        if not os.path.exists(timing_file):
            raise FileNotFoundError(f"Timing file not found: {timing_file}")

        df_t = self.ch.read_dataframe(timing_file, dtype=dtype)
        metrics = _timing_metrics(df_t)
        metrics_file = f"{self.output_path}{self.enhancer_name}_timing_metrics.csv"
        pd.DataFrame([metrics]).to_csv(metrics_file, index=False)

    @log_method
    def _merge_parallel_outputs(self) -> None:
        files = glob.glob(f"{self.output_path}{self.enhancer_name}_gpu[0-9]*.csv")
        files = [f for f in files if not f.endswith("_timing.csv")]
        files = sorted(files, key=_gpu_sort_key)

        dfs = []
        for file in files:
            try:
                dfs.append(pd.read_csv(file))
            except pd.errors.ParserError as exc:
                raise RuntimeError(f"Failed to read parallel output file {file}: {exc}") from exc

        df = pd.concat(dfs, ignore_index=True)
        df.to_csv(f"{self.output_path}{self.enhancer_name}_df.csv", index=False)

    @log_method
    def _merge_parallel_timing_and_metrics(self) -> None:
        timing_files = glob.glob(f"{self.output_path}{self.enhancer_name}_gpu*_timing.csv")
        timing_files = sorted(timing_files, key=_gpu_sort_key)

        if not timing_files:
            raise FileNotFoundError(
                f"No timing files found matching: {self.output_path}{self.enhancer_name}_gpu*_timing.csv"
            )

        df_t = pd.concat([pd.read_csv(f) for f in timing_files], ignore_index=True)
        metrics = _timing_metrics(df_t)
        metrics_file = f"{self.output_path}{self.enhancer_name}_timing_metrics.csv"
        pd.DataFrame([metrics]).to_csv(metrics_file, index=False)
