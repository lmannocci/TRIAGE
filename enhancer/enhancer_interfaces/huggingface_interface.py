import glob
import numpy as np
import pandas as pd
import os
import multiprocessing as mp
from typing import List, Tuple, Dict, Union, Optional
from huggingface_hub import login

from prompt_toolkit import prompt
from utils.checkpoint.checkpoint import *
from utils.directory_manager.directory_manager import DirectoryManager
from utils.log_manager.log_manager import *
from utils.decorator_definition import *
from utils.common_variables import *
from utils.integrity_constraint_manager.integrity_constraint_manager import IntegrityConstraintManager
from enhancer.enhancer_interfaces.utils.dataset_prompt_factory import DatasetPromptFactory
from enhancer.enhancer_interfaces.utils.answer_processing import *



absolute_path = os.path.dirname(__file__)
results_path = os.path.join(absolute_path, f"..{os.sep}results{os.sep}")
data_path = os.path.join(absolute_path, f"..{os.sep}data{os.sep}")
corpus_path = os.path.join(absolute_path, f"..{os.sep}..{os.sep}corpus{os.sep}")
cache_path = os.path.join(absolute_path, f"..{os.sep}..{os.sep}..{os.sep}vast{os.sep}")
main_path = os.path.join(absolute_path, f"..{os.sep}..{os.sep}")

# Refactored implementation
# ------------------------------------------------------------------------------------------------------------------
from enhancer.enhancer_interfaces.base_interface import BaseLLMInterface, run_parallel_worker
from enhancer.enhancer_interfaces.backends import HuggingFaceBackend
from enhancer.enhancer_interfaces.tasks import LLMTask
from utils.common_variables import llms_map


def worker(worker_id, gpu_group, rows, config):
    run_parallel_worker(worker_id, gpu_group, rows, config, HuggingFaceBackend)


class HuggingFaceInterface(BaseLLMInterface):
    provider_name = "huggingface"
    backend_cls = HuggingFaceBackend
    worker_target = staticmethod(worker)

    def __init__(
        self,
        lm: LogManager,
        ch: Checkpoint,
        icm: IntegrityConstraintManager,
        dm: DirectoryManager,
        dataset_prefix: str,
        enhancer_name: str,
        llm_name: str,
        rag: bool,
        return_options: bool,
        output_path: str,
        confidence_computation: bool,
        cuda_visible_devices: str = None,
        parallel: bool = False,
        task: Optional[LLMTask] = None,
    ):
        self.rag = rag
        self.llm_model = llms_map[llm_name]

        super().__init__(
            lm=lm,
            ch=ch,
            icm=icm,
            dm=dm,
            dataset_prefix=dataset_prefix,
            enhancer_name=enhancer_name,
            llm_name=llm_name,
            return_options=return_options,
            output_path=output_path,
            confidence_computation=confidence_computation,
            cuda_visible_devices=cuda_visible_devices,
            parallel=parallel,
            task=task,
        )

    def _backend_config(self):
        return {
            "llm_model": self.llm_model,
        }

    @log_method
    def initialize_huggingface(self):
        self.initialize_model()

    @log_method
    def run_huggingface(self, rows_to_process: List[Tuple[int, pd.Series]]):
        self.run_task(rows_to_process)
