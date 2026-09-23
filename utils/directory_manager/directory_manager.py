from utils.common_variables import *
from utils.mainMethods import *
from utils.log_manager.log_manager import *
from utils.decorator_definition import *

from typing import List, Tuple, Dict, Union, Optional


class DirectoryManager:
    def __init__(self, lm: LogManager, results_path: str, dataset_prefix: Optional[str] = None,
                 model_name: Optional[str] = None, model_prefix: Optional[str] = None,
                 explainer_name: Optional[str] = None,
                 enhancer_name: Optional[str] = None, llm_name: Optional[str] = None, rag: Optional[bool] = None,
                 retriever_name: Optional[str] = None, corpus_name: Optional[str] = None, 
                 metric_exp_ranking: Optional[str] = None, top_k: Optional[int] = None,
                 synthesizer_name: Optional[str] = None, epochs: Optional[int] = None, n_samples_synthesizer: Optional[int] = None,
                 n_neighbors: Optional[int] = None, metric_neighbors: Optional[str] = None, aggregation_method: Optional[str] = None,
                 judge_enhancer_name: Optional[str] = None, judge_llm_name: Optional[str] = None,
                 judge_rag: Optional[bool] = None, judge_retriever_name: Optional[str] = None,
                 judge_corpus_name: Optional[str] = None
                 ) -> None:  
        self.lm = lm
        self.results_path: str = results_path
        self.project_path: str = f"{os.path.dirname(os.path.normpath(self.results_path))}{os.sep}"
        self.data_path: str = f"{self.project_path}data{os.sep}"
        self.dataset_prefix: str = dataset_prefix
        self.model_prefix: str = model_prefix
        self.model_name: str = model_name
        self.explainer_name: str = explainer_name

        self.enhancer_name: str = enhancer_name
        self.llm_name: str = llm_name
        self.rag: bool = rag
        self.retriever_name: str = retriever_name
        self.corpus_name: str = corpus_name

        self.synthesizer_name: str = synthesizer_name
        self.epochs: int = epochs
        self.n_samples_synthesizer: int = n_samples_synthesizer

        self.n_neighbors: int = n_neighbors
        self.metric_neighbors: str = metric_neighbors
        self.aggregation_method: str = aggregation_method
        self.judge_enhancer_name: str = judge_enhancer_name
        self.judge_llm_name: str = judge_llm_name
        self.judge_rag: bool = judge_rag
        self.judge_retriever_name: str = judge_retriever_name
        self.judge_corpus_name: str = judge_corpus_name

        self.metric_exp_ranking: str = metric_exp_ranking
        self.top_k: int = top_k
        self.metric_exp_k = f"{self.metric_exp_ranking}_{str(self.top_k)}"

        self.caller_filename = self._get_caller_filename()
        self.dataset_prefix: str = dataset_prefix

        self.__global_evaluator_directories()

        if self.dataset_prefix is not None:
            self.dataset_path = f"{self.results_path}{self.dataset_prefix}{os.sep}"
            self.__create_directory(self.dataset_path)
            if level[self.caller_filename] == 1:
                self.__blackbox_directories()
            elif level[self.caller_filename] == 2:
                self.__explainer_directories()
            elif level[self.caller_filename] == 3:
                self.__enhancer_directories()
            elif level[self.caller_filename] == 4:
                self.__evaluator_directories()
            elif level[self.caller_filename] == 5:
                self.__synthesizer_directories()
            elif level[self.caller_filename] == 6:
                self.__confidence_enhancer_directories()
            elif level[self.caller_filename] == 7:
                self.__selector_directories()
            elif level[self.caller_filename] == 8:
                self.__enhancer_validator_directories()

    def _get_caller_filename(self):
        """Returns the filename (without .py) of the script that instantiated this class."""
        stack = inspect.stack()
        caller_frame = stack[2]  # The instantiation happens at stack[2]
        caller_filename = os.path.splitext(os.path.basename(caller_frame.filename))[0]  # Remove .py extension
        return caller_filename

    def __create_directory(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
            os.chmod(path, 0o777)
            self.lm.printl(f"Created directory {path}")

    def __global_evaluator_directories(self):
        # Dedicated folder for global / cross-experiment outputs
        self.global_evaluator_path = f"{self.results_path}global_evaluator{os.sep}"
        self.__create_directory(self.global_evaluator_path)

        self.validator_global_path = f"{self.global_evaluator_path}validator{os.sep}"
        self.__create_directory(self.validator_global_path)

        self.validator_explanation_global_path = f"{self.validator_global_path}explanation{os.sep}"
        self.__create_directory(self.validator_explanation_global_path)

        self.validator_rag_global_path = f"{self.validator_global_path}rag{os.sep}"
        self.__create_directory(self.validator_rag_global_path)

        self.explanation_topk_sensitivity_path = f"{self.global_evaluator_path}explanation_topk_sensitivity{os.sep}"
        self.__create_directory(self.explanation_topk_sensitivity_path)

        self.explanation_topk_sensitivity_plot_path = f"{self.explanation_topk_sensitivity_path}plot{os.sep}"
        self.__create_directory(self.explanation_topk_sensitivity_plot_path)

        self.selector_path = f"{self.global_evaluator_path}selector{os.sep}"
        self.__create_directory(self.selector_path)

        self.selector_sensitivity_analysis_path = f"{self.selector_path}sensitivity_analysis{os.sep}"
        self.__create_directory(self.selector_sensitivity_analysis_path)

        self.selector_sensitivity_analysis_plot_path = f"{self.selector_sensitivity_analysis_path}plot{os.sep}"
        self.__create_directory(self.selector_sensitivity_analysis_plot_path)

    def __blackbox_directories(self):
        self.model_path = f"{self.dataset_path}{self.model_name}_{self.model_prefix}{os.sep}"
        self.__create_directory(self.model_path)

    def __explainer_directories(self):
        self.__blackbox_directories()
        self.explainer_path = f"{self.model_path}{self.explainer_name}{os.sep}"
        self.__create_directory(self.explainer_path)

    def __enhancer_directories(self):
        # self.__blackbox_directories()
        self.enhancer_path = f"{self.dataset_path}{self.enhancer_name}_{self.llm_name}_{str(self.rag)}_{self.retriever_name}_{self.corpus_name}{os.sep}"
        self.__create_directory(self.enhancer_path)

        self.temp_answer_path = f"{self.enhancer_path}temp_answer_path{os.sep}"
        self.__create_directory(self.temp_answer_path)


    def __evaluator_directories(self):
        # Actually these 3 lines are not needed to create the directories, as they likely already exist, but just to access them with the variables
        if self.model_name is not None and self.model_prefix is not None:
            self.__blackbox_directories()
        if self.explainer_name is not None:
            self.__explainer_directories()
        if self.enhancer_name is not None:
            self.__enhancer_directories()

        self.evaluator_path = f"{self.dataset_path}evaluator{os.sep}"
        self.__create_directory(self.evaluator_path)

        self.ev_report_path = f"{self.evaluator_path}report{os.sep}"
        self.__create_directory(self.ev_report_path)

        # Agreement
        # ------------------------------------------------------------
        self.ev_agreement_path = f"{self.evaluator_path}agreement{os.sep}"
        self.__create_directory(self.ev_agreement_path)

        self.ev_agreement_csv_path = f"{self.ev_agreement_path}csv{os.sep}"
        self.__create_directory(self.ev_agreement_csv_path)

        self.ev_agreement_plot_path = f"{self.ev_agreement_path}aggregated_plot{os.sep}"
        self.__create_directory(self.ev_agreement_plot_path)

        # ------------------------------------------------------------
        self.ev_explanations_path = f"{self.evaluator_path}explanations{os.sep}"
        self.__create_directory(self.ev_explanations_path)
        if self.top_k is not None and self.metric_exp_ranking is not None and self.enhancer_name is not None and self.llm_name is not None and self.rag is not None and self.retriever_name is not None and self.corpus_name is not None:
            self.ev_enhancer_path = f"{self.ev_explanations_path}{self.enhancer_name}_{self.llm_name}_{str(self.rag)}_{self.retriever_name}_{self.corpus_name}{os.sep}"
            self.__create_directory(self.ev_enhancer_path)

            self.ev_explanations_metric_path = f"{self.ev_enhancer_path}{self.metric_exp_k}{os.sep}"
            self.__create_directory(self.ev_explanations_metric_path)

            self.ev_explanations_csv_path = f"{self.ev_explanations_metric_path}csv{os.sep}"
            self.__create_directory(self.ev_explanations_csv_path)

    def __synthesizer_directories(self):
        self.synthesizer_path = f"{self.dataset_path}synth_{self.synthesizer_name}_epochs_{self.epochs}_samples_{self.n_samples_synthesizer}{os.sep}"
        self.__create_directory(self.synthesizer_path)

    def __confidence_enhancer_directories(self):
        self.__enhancer_directories()
        self.__synthesizer_directories()
        
        self.neighbours_path = f"{self.synthesizer_path}neighbors_{self.metric_neighbors}_neighbors_{str(self.n_neighbors)}{os.sep}"
        self.__create_directory(self.neighbours_path)

        self.neighbours_analysis_path = f"{self.neighbours_path}analysis{os.sep}"
        self.__create_directory(self.neighbours_analysis_path)

        self.confidence_enhancer_path = f"{self.neighbours_path}confidence_enhancer{os.sep}"
        self.__create_directory(self.confidence_enhancer_path)

        self.ce_enhancer_path = f"{self.confidence_enhancer_path}{self.enhancer_name}_{self.llm_name}_{str(self.rag)}_{self.retriever_name}_{self.corpus_name}{os.sep}"
        self.__create_directory(self.ce_enhancer_path)

        self.ce_enhancer_analysis_path = f"{self.ce_enhancer_path}analysis{os.sep}"
        self.__create_directory(self.ce_enhancer_analysis_path)
    
    def __selector_directories(self):
        self.__confidence_enhancer_directories()
        self.__evaluator_directories()

        self.en_selector_path = f"{self.enhancer_path}selector{os.sep}"
        self.__create_directory(self.en_selector_path)

        self.en_selector_analysis_path = f"{self.en_selector_path}analysis{os.sep}"
        self.__create_directory(self.en_selector_analysis_path)

        self.en_selector_csv_path = f"{self.en_selector_analysis_path}csv{os.sep}"
        self.__create_directory(self.en_selector_csv_path)

        self.en_selector_plot_path = f"{self.en_selector_analysis_path}plot{os.sep}"
        self.__create_directory(self.en_selector_plot_path)

        self.selector_path = f"{self.global_evaluator_path}selector{os.sep}"
        self.__create_directory(self.selector_path)

        self.selector_sensitivity_analysis_path = f"{self.selector_path}sensitivity_analysis{os.sep}"
        self.__create_directory(self.selector_sensitivity_analysis_path)

        self.selector_sensitivity_analysis_plot_path = f"{self.selector_sensitivity_analysis_path}plot{os.sep}"
        self.__create_directory(self.selector_sensitivity_analysis_plot_path)

        # self.dataset_selector_path = f"{self.selector_path}{self.dataset_prefix}{os.sep}"
        # self.__create_directory(self.dataset_selector_path)

    def __enhancer_validator_directories(self):
        if self.model_name is not None and self.model_prefix is not None and self.explainer_name is not None:
            self.__explainer_directories()

        if self.enhancer_name is not None and self.llm_name is not None and self.rag is not None and self.retriever_name is not None and self.corpus_name is not None:
            self.__enhancer_directories()
        
        self.enhancer_validator_path = f"{self.enhancer_path}validator{os.sep}"
        self.__create_directory(self.enhancer_validator_path)
        
        self.enhancer_validator_llm_path = (
            f"{self.enhancer_validator_path}{self.judge_enhancer_name}_{self.judge_llm_name}_"
            f"{str(self.judge_rag)}_{self.judge_retriever_name}_{self.judge_corpus_name}{os.sep}"
        )
        self.__create_directory(self.enhancer_validator_llm_path)
        

        self.explaination_validator_path = f"{self.enhancer_validator_llm_path}explanation{os.sep}"
        self.__create_directory(self.explaination_validator_path)

        self.rag_validator_path = f"{self.enhancer_validator_llm_path}rag{os.sep}"
        self.__create_directory(self.rag_validator_path)

        self.temp_answer_path = f"{self.enhancer_validator_llm_path}temp_answer_path{os.sep}"
        self.__create_directory(self.temp_answer_path)
