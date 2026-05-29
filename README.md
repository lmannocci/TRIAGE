# TRIAGE: Trustworthy Retrieval-based Interpretability with Abstention for Grounded Explanations in Clinical Decision Support System

The ability to provide explanations for Artificial Intelligence systems is increasingly important in healthcare,
where predictive models support high‑stakes decisions but are often difficult for end users (e.g., clinicians) to interpret.
Traditional feature‑based explanation methods (for example, SHAP or LIME) produce useful, structured insights but
can remain hard to read and lack narrative context. Large Language Models (LLMs) can generate natural‑language
explanations that are more accessible to clinicians, but they also introduce risks such as hallucinations and
reduced traceability. Retrieval‑Augmented Generation (RAG) can help mitigate these risks by conditioning generation
on external medical knowledge and sources.

This repository implements a modular, abstention‑aware framework for explainable clinical decision support —
referred to in the paper as TRIAGE. The framework integrates traditional machine learning models and
feature‑based explainers with a retrieval‑augmented generative enhancer and a selector that governs the final
output. The selector is the framework’s core: it evaluates evidence quality and conflicts across sources, chooses
the most reliable prediction and explanation, or abstains when information is insufficient or contradictory.

We evaluate the framework on five public healthcare binary classification datasets, comparing black‑box models,
local explainers, and enhancer configurations. Our analysis shows that traditional machine learning pipelines
remain the most reliable source of predictions and explanations in most cases; the generative enhancer is selected
only for a limited subset of instances, and the abstention mechanism substantially contributes to safe,
traceable outputs. TRIAGE demonstrates how retrieval grounding and abstention governance can enable the
controlled integration of generative explanations into clinical workflows while prioritizing safety and reproducibility.

Table of contents
- Installation
- Quick start
- Configuration
- Project structure
- Results
- Datasets

Installation
------------
Prerequisites
- Python 3.9+ (recommended)
- CUDA-enabled GPU for LLM/accelerated model training (optional but recommended)

Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you need a specific GPU configuration, set `CUDA_VISIBLE_DEVICES` in your environment or adjust
the `cuda_visible_devices` variable in `main.py`.

Quick start
-----------
Run the full pipeline (preprocessing, blackbox training, explainers, enhancers, evaluators):

```bash
python main.py
```

By default `main.py` iterates over the built-in datasets: `pima`, `diabetes`, `stroke`, `liver`, and `covid`.
Configuration for each dataset is stored in the root-level `input_config_*.py` files. To change the final
experiment configuration, edit `input_config_final.py`.

Example: run a single dataset (by editing the loop in `main.py` or selecting the dataset in a config module)

Configuration
-------------
- `input_config_pima.py`, `input_config_diabetes.py`, `input_config_stroke.py`, `input_config_liver.py`,
	`input_config_covid.py`: dataset-specific parameters (models, explainers, enhancer choices, RAG, retriever, etc.).
- `input_config_final.py`: final selector/evaluation configuration used by the `Selector` and global runs.

Each input config exposes keys the pipeline expects, such as `param_grid`, `model_prefix`, `enhancer_name`,
`llm_name`, `rag`, `return_options`, and others used throughout the pipeline.

Project structure (high level)
------------------------------
- `blackblox/` - Black-box model training and evaluation wrapper.
- `enhancer/` - LLM-based enhancement pipeline, validators and interfaces for different backends.
- `explainer/` - Explanation extraction and metrics (SHAP / LIME / DALEX wrappers).
- `evaluator/` - Per-dataset evaluator utilities and reporting.
- `global_evaluator/` - Cross-dataset aggregation and visualization.
- `preprocessing/` - Data loading and preprocessing utilities.
- `synthesizer/` - Synthetic data generation and evaluation.
- `selector/` - Selection logic to compare enhancer outputs vs black-box baselines.
- `templates/` - Jinja templates for prompt construction.
- `data/` - Raw and preprocessed CSVs used by experiments.
- `results/` - Generated plots, CSVs, and evaluation reports (organized by dataset).

Usage notes
-----------
- To change which models or explainers are evaluated, update the corresponding `av_models`,
	`av_models_explainer`, and `av_enhancers_llm` lists in `utils/common_variables.py` or in input config files.
- Large LLM-based experiments may require API credentials or local model artifacts; configure the
	corresponding credentials within `config.ini` and the model paths in the enhancer interfaces under `enhancer/enhancer_interfaces/`.

Results
-------
Results and artifacts are written under the `results/` directory and are organized per dataset. Typical
outputs include model reports, explainer metrics, enhancer rankings, selector analyses and plots.

Datasets
--------
We evaluated the method on five publicly available healthcare binary classification datasets:

- **Pima Indians Diabetes Database (Pima)** — https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
- **Liver Disease (Liver)** — https://www.kaggle.com/datasets/abhi8923shriv/liver-disease-patient-dataset
- **Stroke Prediction Dataset (Stroke)** — https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
- **COVID-19 (Covid)** — https://www.kaggle.com/datasets/einsteindata4u/covid19
- **Diabetes Health Indicators (Diabetes)** — https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset

All datasets listed above are publicly available for research and analysis.
