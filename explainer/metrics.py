import numpy as np
import pandas as pd

from utils.directory_manager.directory_manager import DirectoryManager
from utils.integrity_constraint_manager.integrity_constraint_manager import IntegrityConstraintManager
from utils.log_manager.log_manager import LogManager
from utils.checkpoint.checkpoint import Checkpoint
from utils.common_variables import *
from utils.decorator_definition import *

class ExplainerMetrics:
    def __init__(self, lm: LogManager, ch: Checkpoint, icm: IntegrityConstraintManager, dm: DirectoryManager,
                 dataset_prefix: str, model_name: str, model_prefix: str, model, explainer_name: str):
        self.ch = ch
        self.lm = lm
        self.dm = dm
        self.icm = icm
        self.dataset_prefix = dataset_prefix
        self.model_name = model_name
        self.model_prefix = model_prefix
        self.model = model # model (fitted classifier, e.g., RandomForestClassifier) to explain, must implement predict() and predict_proba()
        self.explainer_name = explainer_name
    
    def __build_reference_values(self, X_reference: pd.DataFrame):
        ref = {}

        for col in X_reference.columns:
            if pd.api.types.is_numeric_dtype(X_reference[col]):
                ref[col] = X_reference[col].median()
            else:
                ref[col] = X_reference[col].mode().iloc[0]

        return ref

    # PUBLIC
    # --------------------------------------------------------------------------------------------------
    @log_method
    def faithfulness_metric(
        self,
        X_test: pd.DataFrame,
        test_df: pd.DataFrame,
        explanations: list[dict],
        reference_values: dict | None = None
    ) -> pd.DataFrame:
        """
        Compute correlation-based faithfulness metric.

        Returns a DataFrame with:
            - ind (global identifier)
            - faithfulness
        """
        if len(X_test) != len(explanations):
            raise ValueError("X_test and explanations must have the same length.")

        if reference_values is None:
            reference_values = self.__build_reference_values(X_test)

        preds = self.model.predict(X_test)
        probas = self.model.predict_proba(X_test)

        # map class labels to proba column indices
        class_to_index = {c: i for i, c in enumerate(self.model.classes_)}
        pred_idx = np.array([class_to_index[p] for p in preds])

        results = []

        for i in range(len(X_test)):
            row = X_test.iloc[i]
            row_ind = test_df[ind].values[i]
            explanation = explanations[i] if explanations[i] is not None else {}

            # keep only valid features
            features = [f for f in explanation.keys() if f in X_test.columns]
            coefs = np.array([abs(explanation[f]) for f in features], dtype=float)

            # degenerate case
            if len(features) < 2:
                faithfulness = np.nan
            else:
                perturbed_probs = []

                for f in features:
                    X_copy = pd.DataFrame([row.copy()])
                    X_copy.at[X_copy.index[0], f] = reference_values[f]

                    new_proba = self.model.predict_proba(X_copy)[0, pred_idx[i]]
                    perturbed_probs.append(new_proba)

                perturbed_probs = np.array(perturbed_probs, dtype=float)

                corr = np.corrcoef(coefs, perturbed_probs)[0, 1]
                faithfulness = -corr if not np.isnan(corr) else np.nan

            results.append({
                ind: row_ind,   # global identifier
                "faithfulness": faithfulness
            })

        return pd.DataFrame(results)