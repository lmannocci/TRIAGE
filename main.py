import importlib

import pandas as pd
import sys
import os

from blackblox.blackbox import Blackbox
from enhancer.enhancer import Enhancer
from enhancer.enhancer_validator import EnhancerJudgeValidator
from explainer.explainer import Explainer
from synthesizer.synthesizer import Synthesizer
from enhancer.confidence_enhancer import ConfidenceEnhancer
from global_evaluator.global_evaluator import GlobalEvaluator
from selector.selector import Selector
from utils.log_manager import LogManager

from utils.checkpoint.checkpoint import *
# from input_config_pima import *
# from input_config_diabetes import *
# from input_config_stroke import *
# from input_config_liver import *
# from input_config_covid import *
from utils.common_variables import *
from preprocessing.preprocessing import Preprocessing
from evaluator.evaluator import Evaluator

absolute_path = os.path.dirname(__file__)
results = os.path.join(absolute_path, f"results{os.sep}")
data_path = os.path.join(absolute_path, f"data{os.sep}")

# GPU AND PARALLEL SETTINGS
# cuda_visible_devices = "0,1,2,3,4,5,6,7"
cuda_visible_devices = "0,1,4,5,6,7"
parallel = True

dataset_to_config = {
    "pima": "input_config_pima",
    "diabetes": "input_config_diabetes",
    "stroke": "input_config_stroke",
    "liver": "input_config_liver",
    "covid": "input_config_covid",
}

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # available_GPUs()
    # dataset = f"{dataset_prefix}.csv"
    lm: LogManager = LogManager('main')
    ch: Checkpoint = Checkpoint()

    env_name = os.path.basename(sys.prefix)
    lm.printl(f"Environment: {env_name}")
    # lm.printl(f"Using dataset: {dataset_prefix}")

    # PREPROCESSING
    # ----------------------------------------------------------------------------------------------------------
    for dataset_prefix in ['pima', 'diabetes' , 'stroke', 'liver', 'covid']:    
        pr: Preprocessing = Preprocessing(ch, lm, dataset_prefix)
        df: pd.DataFrame = pr.preprocess_dataframe()
        pr.info_dataframe()

        dataset = f"{dataset_prefix}_preprocessed.csv"
        df: pd.DataFrame = ch.read_dataframe(f"{data_path}{dataset}", dtype=dtype)

    # BLACKBOX
    # ----------------------------------------------------------------------------------------------------------
    for dataset_prefix in ['pima', 'diabetes' , 'stroke', 'liver', 'covid']:
        config_module = importlib.import_module(dataset_to_config[dataset_prefix])
        param_grid = config_module.param_grid
        model_prefix = 'balanced'
        for model_name in av_models: # ['random_forest', 'ebm', 'catboost', 'xgboost']:
            lm.printl(f"Evaluating model: {model_name} for dataset: {dataset_prefix}")
            bb: Blackbox = Blackbox(ch, lm, dataset_prefix, model_name, model_prefix, param_grid[model_name])
            bb.construct_bb(compute_model=True, n_row_sampling=10000 if dataset_prefix == "diabetes" else None)  # Adjust the number of rows to sample as needed n_row_sampling=10000
            bb.plot_confidence_distribution(threshold=0.7)
            ev: Evaluator = Evaluator(ch, lm, dataset_prefix, model_name=model_name, model_prefix=model_prefix)
            for save_global_evaluator in [False, True]:  # First save without global evaluator, then with global evaluator to include in cross-dataset comparison
                ev.save_classification_report(save_global_evaluator=save_global_evaluator)  # Save to global evaluator as well for easier cross-dataset comparison
    
    # EXPLAINER
    # ----------------------------------------------------------------------------------------------------------
    for dataset_prefix in ['pima', 'diabetes' , 'stroke', 'liver', 'covid']:
        config_module = importlib.import_module(dataset_to_config[dataset_prefix])
        model_prefix = config_module.model_prefix
        for model_name, explainer_list in av_models_explainer.items():
            for explainer_name in explainer_list: # ['shap', 'lime', 'dalex']:
                if (explainer_name == 'shap' and env_name == 'shapEnv') or (explainer_name != 'shap' and env_name != 'shapEnv'):
                    lm.printl(f"Dataset {dataset_prefix} - Explaining model: {model_name} with explainer: {explainer_name}")
                    ex = Explainer(ch, lm, dataset_prefix, model_name, model_prefix, explainer_name)
                    ex.explain()
                    ex.extract_explainer_features()
                lm.printl(f"Dataset {dataset_prefix} - Explaining model: {model_name} with explainer: {explainer_name}")
                ex = Explainer(ch, lm, dataset_prefix, model_name, model_prefix, explainer_name)
                ex.compute_explainer_metrics(['faithfulness'])

    # ENHANCER single run
    # ----------------------------------------------------------------------------------------------------------
    # HUGGINGFACE ENHANCER
    for dataset_prefix in ['pima', 'diabetes' , 'stroke', 'liver', 'covid']:
        config_module = importlib.import_module(dataset_to_config[dataset_prefix])
        enhancer_name = config_module.enhancer_name
        llm_name = config_module.llm_name
        rag = config_module.rag
        return_options = config_module.return_options
        rag_K_documents = config_module.rag_K_documents
        en: Enhancer = Enhancer(ch, lm, dataset_prefix, enhancer_name, llm_name, rag, return_options=return_options, cuda_visible_devices=cuda_visible_devices, parallel=parallel)
        en.initialize_enhancer()
        en.run_enhancer()  # Run on the entire dataset
        
        en: Enhancer = Enhancer(ch, lm, dataset_prefix, enhancer_name, llm_name=llm_name, rag=rag, return_options=return_options,  rag_K_documents=rag_K_documents)
        en.initialize_enhancer(intialize_model=False)
        en.clean_enhancer_df()
    
    # ENHANCER+EVALUATOR: CLEAN ENHANCER RANKING AND SAVE CLASSIFICATION REPORT
    # ----------------------------------------------------------------------------------------------------------
    for dataset_prefix in ['pima', 'diabetes' , 'stroke', 'liver', 'covid']:
        config_module = importlib.import_module(dataset_to_config[dataset_prefix])

        return_options = config_module.return_options
        rag_K_documents = config_module.rag_K_documents
        for enhancer_name, llm_list in av_enhancers_llm.items(): # ['medrag', 'huggingface'], [['mixtral', 'llama2', 'meditron'], ['mixtral', 'llama2']]
            for llm_name in llm_list: # ['mixtral', 'llama2', 'meditron']:
                for rag in av_enhancers_rag[enhancer_name]: # [True, False]:
                    lm.printl(f"{dataset_prefix}:{enhancer_name} with llm {llm_name} and RAG {rag}")
                    en: Enhancer = Enhancer(ch, lm, dataset_prefix, enhancer_name, llm_name=llm_name, rag=rag, return_options=return_options,  rag_K_documents=rag_K_documents)
                    en.clean_feature_importance_ranking()
                    ev: Evaluator = Evaluator(ch, lm, dataset_prefix, enhancer_name=enhancer_name, llm_name=llm_name, rag=rag)
                    for save_global_evaluator in [False, True]:  # First save without global evaluator, then with global evaluator to include in cross-dataset comparison
                        ev.save_classification_report(save_global_evaluator=save_global_evaluator)  # Save to global evaluator as well for easier cross-dataset comparison
    

    # EVALUATOR - Breakdown agreement
    # ----------------------------------------------------------------------------------------------------------
    for dataset_prefix in ['pima', 'diabetes' , 'stroke', 'liver', 'covid']:
        config_module = importlib.import_module(dataset_to_config[dataset_prefix])
        model_prefix = config_module.model_prefix
        for model_name, explainer_list in av_models_explainer.items():
            for enhancer_name, llm_list in av_enhancers_llm.items(): # ['medrag', 'huggingface'], [['mixtral', 'llama2', 'meditron'], ['mixtral', 'llama2']]
                for llm_name in llm_list: # ['mixtral', 'llama2', 'meditron']:
                    for rag in av_enhancers_rag[enhancer_name]: # [True, False]:

                        lm.printl(f"{dataset_prefix}:{model_name} with enhancer {enhancer_name} with llm {llm_name} and RAG {rag}")
                        ev: Evaluator = Evaluator(ch, lm, dataset_prefix, model_name, model_prefix, enhancer_name,  llm_name, rag)
                        ev.compute_breakdown_agreement(save_global_evaluator=True)  # Save to global evaluator as well for easier cross-dataset comparison



    # EVALUATOR - Evaluate Explanations (compare explainer feature importance ranking with enhancer feature importance ranking)
    # ----------------------------------------------------------------------------------------------------------
    for dataset_prefix in ['pima', 'diabetes' , 'stroke', 'liver', 'covid']:
        config_module = importlib.import_module(dataset_to_config[dataset_prefix])
        model_prefix = config_module.model_prefix
        top_k = config_module.top_k
        metric_exp_ranking = config_module.metric_exp_ranking
        
        for model_name, explainer_list in av_models_explainer.items():
            for explainer_name in explainer_list: # ['shap', 'lime', 'dalex']:
                for enhancer_name, llm_list in av_enhancers_llm.items(): # ['medrag', 'huggingface'], [['mixtral', 'llama2', 'meditron'], ['mixtral', 'llama2']]
                    for llm_name in llm_list: # ['mixtral', 'llama2', 'meditron']:
                        for rag in av_enhancers_rag[enhancer_name]: # [True, False]:
                            lm.printl(f"{dataset_prefix}:{model_name} with explainer {explainer_name} and enhancer {enhancer_name} with llm {llm_name} and RAG {rag}")
                            ev: Evaluator = Evaluator(ch, lm, dataset_prefix, 
                                                      model_name=model_name, model_prefix=model_prefix, 
                                                      explainer_name=explainer_name, 
                                                      enhancer_name=enhancer_name, llm_name=llm_name, rag=rag, top_k=top_k, metric_exp_ranking=metric_exp_ranking)
                            ev.evaluate_explanations()
                            ev.save_aggregated_explanation_evaluation(save_global_evaluator=True)  # Save to global evaluator as well for easier cross-dataset comparison

    # EVALUATOR - Aggregate explanations statistics
    # ----------------------------------------------------------------------------------------------------------
    for dataset_prefix in ['pima', 'diabetes' , 'stroke', 'liver', 'covid']:
        config_module = importlib.import_module(dataset_to_config[dataset_prefix])
        model_prefix = config_module.model_prefix
        top_k = config_module.top_k
        metric_exp_ranking = config_module.metric_exp_ranking
        
        for model_name, explainer_list in av_models_explainer.items():
            for explainer_name in explainer_list: # ['shap', 'lime', 'dalex']:
                lm.printl(f"{dataset_prefix}:{model_name} with explainer {explainer_name}")
                ev: Evaluator = Evaluator(ch, lm, dataset_prefix, model_name=model_name, model_prefix=model_prefix, explainer_name=explainer_name)
                ev.save_aggregate_explanation_metrics(metrics_list=['faithfulness'])
    
    
    for dataset_prefix in ['pima', 'diabetes' , 'stroke', 'liver', 'covid']:
        config_module = importlib.import_module(dataset_to_config[dataset_prefix])
        avg_text_y_shift_map = config_module.avg_text_y_shift_map
        avg_text_x_shift_map = config_module.avg_text_x_shift_map
        ev: Evaluator = Evaluator(ch, lm, dataset_prefix, top_k=top_k, metric_exp_ranking=metric_exp_ranking)
        ev.plot_aggregated_explanations_statistics(avg_text_x_shift_map=avg_text_x_shift_map, avg_text_y_shift_map=avg_text_y_shift_map)

    # SYNTHESIZER
    # ----------------------------------------------------------------------------------------------------------
    for dataset_prefix in ['pima', 'diabetes' , 'stroke', 'liver', 'covid']: # 
        config_module = importlib.import_module(dataset_to_config[dataset_prefix])
        synthesizer_name = config_module.synthesizer_name
        epochs = config_module.epochs
        n_samples_synthesizer = config_module.n_samples_synthesizer

        sy: Synthesizer = Synthesizer(ch, lm, dataset_prefix, synthesizer_name, epochs, n_samples_synthesizer, cuda_visible_devices)
        sy.run_synthesizer()
        sy.generate_synthetic_data()
        sy.evaluate_synthetic_resemblance()


    # CONFIDENCE ENHANCER
    # ----------------------------------------------------------------------------------------------------------
    for dataset_prefix in ['pima' , 'diabetes', 'stroke', 'liver', 'covid']: #,'pima' , 'diabetes', 'stroke', 'liver', 'covid'
        config_module = importlib.import_module(dataset_to_config[dataset_prefix])

        enhancer_name = config_module.enhancer_name
        llm_name = config_module.llm_name
        rag = config_module.rag
        return_options = config_module.return_options
        rag_K_documents = config_module.rag_K_documents
        retriever_name = config_module.retriever_name
        corpus_name = config_module.corpus_name

        synthesizer_name = config_module.synthesizer_name
        epochs = config_module.epochs
        n_samples_synthesizer = config_module.n_samples_synthesizer

        n_neighbors = config_module.n_neighbors
        metric_neighbors = config_module.metric_neighbors
        aggregation_method = config_module.aggregation_method


        lm.printl(f"{dataset_prefix}:{enhancer_name} with llm {llm_name} and RAG {rag}")
        ce: ConfidenceEnhancer = ConfidenceEnhancer(ch, lm, dataset_prefix, 
                                                    enhancer_name,  llm_name, rag, 
                                                    retriever_name=retriever_name, corpus_name=corpus_name,
                                                    return_options=return_options, rag_K_documents=rag_K_documents,
                                                    synthesizer_name=synthesizer_name, epochs=epochs, n_samples_synthesizer=n_samples_synthesizer,
                                                    n_neighbors=n_neighbors, metric_neighbors=metric_neighbors, aggregation_method=aggregation_method,
                                                    cuda_visible_devices=cuda_visible_devices, parallel=parallel)
        ce.find_nearest_neighbors()
        ce.analyze_neighbors_distribution()
        ce.run_enhancer_on_neighbors()
        ce.compute_confidence_scores()
        ce.plot_confidence_enhancer_results()

    # SELECTOR
    # ----------------------------------------------------------------------------------------------------------
    final_config_module = importlib.import_module("input_config_final") 
    selector_name = final_config_module.selector_name
    model_name = final_config_module.model_name
    model_prefix = final_config_module.model_prefix
    explainer_name = final_config_module.explainer_name
    top_k = final_config_module.top_k
    metric_exp_ranking = final_config_module.metric_exp_ranking
    enhancer_name = final_config_module.enhancer_name
    llm_name = final_config_module.llm_name
    rag = final_config_module.rag
    retriever_name = final_config_module.retriever_name
    corpus_name = final_config_module.corpus_name
    return_options = final_config_module.return_options
    rag_K_documents = final_config_module.rag_K_documents
    synthesizer_name = final_config_module.synthesizer_name
    n_neighbors = final_config_module.n_neighbors
    metric_neighbors = final_config_module.metric_neighbors
    aggregation_method = final_config_module.aggregation_method

    for dataset_prefix in ['pima' , 'diabetes', 'stroke', 'liver', 'covid']: #,'pima' , 'diabetes', 'stroke', 'liver', 'covid'
        config_module = importlib.import_module(dataset_to_config[dataset_prefix])
        epochs = config_module.epochs
        n_samples_synthesizer = config_module.n_samples_synthesizer

        se: Selector = Selector(ch, lm, dataset_prefix, selector_name,
                                model_name, model_prefix,
                                explainer_name, top_k, metric_exp_ranking,
                                enhancer_name, llm_name, rag, retriever_name, corpus_name, return_options, rag_K_documents,
                                synthesizer_name, epochs, n_samples_synthesizer, n_neighbors, metric_neighbors, aggregation_method,
                                th_enhancer_conf=0.7, th_model_conf=0.7, th_rbo=0.7,
                                cuda_visible_devices=cuda_visible_devices, parallel=parallel
                            )
        se.apply_selector()
        se.analyze_selector()
        se.save_selector_classification_report()
        se.plot_selector_vs_bb_dumbbell_per_dataset()
    
    # Explanation TOP-K agreement analysis for selector vs enhancer vs blackbox
    for dataset_prefix in ['pima' , 'diabetes', 'stroke', 'liver', 'covid']: #,'pima' , 'diabetes', 'stroke', 'liver', 'covid'
        lm.printl(f"Dataset {dataset_prefix} - saving aggregated explanation feature statistics for final configuration")
        ev: Evaluator = Evaluator(
            ch,
            lm,
            dataset_prefix,
            model_name=model_name,
            model_prefix=model_prefix,
            explainer_name=explainer_name,
        )
        ev.save_aggregated_explanation_feature_statistics(save_global_evaluator=True)

    # GLOBAL EVALUATOR - Works on cross-dataset comparison and overall trends across datasets, not on individual dataset analysis (e.g. selector vs blackbox per dataset)
    # ----------------------------------------------------------------------------------------------------------
    ge: GlobalEvaluator = GlobalEvaluator(ch, lm)
    ge.plot_global_explanation_usage_stacked()
    ge.plot_aggregated_explanation_feature_statistics_linechart()

    # LLM AS A JUDGE
    # ----------------------------------------------------------------------------------------------------------
    final_config_module = importlib.import_module("input_config_final") 
    model_name = final_config_module.model_name
    model_prefix = final_config_module.model_prefix
    explainer_name = final_config_module.explainer_name
    top_k = final_config_module.top_k
    enhancer_name = final_config_module.enhancer_name
    llm_name = final_config_module.llm_name
    rag = final_config_module.rag
    retriever_name = final_config_module.retriever_name
    corpus_name = final_config_module.corpus_name
    judge_enhancer_name = "huggingface"
    judge_llm_name = "mixtral"
    judge_rag = False
    judge_retriever_name = "MedCPT"
    judge_corpus_name = "StatPearls"

    for dataset_prefix in ['diabetes', 'stroke', 'liver', 'covid']: #,'pima' , 'diabetes', 'stroke', 'liver', 'covid'
        validator = EnhancerJudgeValidator(
            ch=ch,
            lm=lm,
            results_path=results,
            dataset_prefix=dataset_prefix,
            model_name=model_name,
            model_prefix=model_prefix,
            explainer_name=explainer_name,
            enhancer_name=enhancer_name,
            enhancer_llm_name=llm_name,
            rag=rag,
            retriever_name=retriever_name,
            corpus_name=corpus_name,
            judge_enhancer_name=judge_enhancer_name,
            judge_llm_name=judge_llm_name,
            judge_rag=judge_rag,
            judge_retriever_name=judge_retriever_name,
            judge_corpus_name=judge_corpus_name,
            top_k=top_k,
            cuda_visible_devices=cuda_visible_devices,
            parallel=parallel,
        )
        validator.run()
        validator.clean_llm_judge()
        validator.summarize_llm_judge_metrics()

        # Evaluate RAG retrieved documents with LLM judge
        validator.run_rag_judge()
        validator.clean_llm_rag_judge()
        validator.summarize_llm_rag_judge_metrics()

    ge: GlobalEvaluator = GlobalEvaluator(ch, lm)
    ge.plot_llm_judge_metrics()
    ge.plot_llm_rag_judge_metrics()