
import pandas as pd
from typing import List, Tuple, Dict, Union, Optional



from utils.checkpoint.checkpoint import *
from utils.directory_manager.directory_manager import DirectoryManager
from utils.log_manager.log_manager import *
from utils.decorator_definition import *
from utils.common_variables import *
from utils.integrity_constraint_manager.integrity_constraint_manager import IntegrityConstraintManager

from enhancer.enhancer_interfaces.utils.answer_processing import *


# Refactored implementation
# ------------------------------------------------------------------------------------------------------------------
from enhancer.enhancer_interfaces.base_interface import BaseLLMInterface, run_parallel_worker
from enhancer.enhancer_interfaces.backends import MedRAGBackend, cache_path, corpus_path
from enhancer.enhancer_interfaces.tasks import LLMTask
from utils.common_variables import llms_map


def medrag_worker(worker_id, gpu_group, rows, config):
    run_parallel_worker(worker_id, gpu_group, rows, config, MedRAGBackend)


class MedRagInterface(BaseLLMInterface):
    provider_name = "medrag"
    backend_cls = MedRAGBackend
    worker_target = staticmethod(medrag_worker)

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
        retriever_name: str,
        corpus_name: str,
        rag_K_documents: int,
        return_options: bool,
        output_path: str,
        confidence_computation: bool,
        cuda_visible_devices: str = None,
        parallel: bool = False,
        task: Optional[LLMTask] = None,
    ):
        self.rag = rag
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        self.rag_K_documents = rag_K_documents

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
            "llm_name": llms_map[self.llm_name],
            "rag": self.rag,
            "retriever_name": self.retriever_name,
            "corpus_name": self.corpus_name,
            "rag_K_documents": self.rag_K_documents,
            "db_dir": corpus_path,
            "cache_dir": cache_path,
        }

    @log_method
    def initialize_medrag(self):
        self.initialize_model()

    @log_method
    def run_medrag(self, rows_to_process: List[Tuple[int, pd.Series]]):
        self.run_task(rows_to_process)
