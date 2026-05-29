

from selector.core.context import SelectorContext


from sklearn.metrics import classification_report
import numpy as np
import pandas as pd


class SelectorMetricsHelper:
    def __init__(self, ctx: SelectorContext):
        self.ctx = ctx

    def safe_class_metrics(self, report_dict: dict, class_label: int) -> dict:
        possible_keys = [str(class_label), str(float(class_label)), class_label, float(class_label)]
        cls = {}
        for key in possible_keys:
            if key in report_dict:
                cls = report_dict[key]
                break
        return {
            f"precision_{class_label}": cls.get("precision"),
            f"recall_{class_label}": cls.get("recall"),
            f"f1-score_{class_label}": cls.get("f1-score"),
            f"support_{class_label}": cls.get("support"),
        }

    def build_report_metrics(self, y_true: pd.Series, y_pred: pd.Series) -> dict:
        if len(y_true) == 0:
            return {
                "accuracy": np.nan,
                "precision_macro": np.nan,
                "recall_macro": np.nan,
                "f1-score_macro": np.nan,
                "support_macro": 0,
                "precision_weighted": np.nan,
                "recall_weighted": np.nan,
                "f1-score_weighted": np.nan,
                "support_weighted": 0,
                "precision_0": np.nan,
                "recall_0": np.nan,
                "f1-score_0": np.nan,
                "support_0": 0,
                "precision_1": np.nan,
                "recall_1": np.nan,
                "f1-score_1": np.nan,
                "support_1": 0,
            }

        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        macro = report.get("macro avg", {})
        weighted = report.get("weighted avg", {})

        row = {
            "accuracy": report.get("accuracy"),
            "precision_macro": macro.get("precision"),
            "recall_macro": macro.get("recall"),
            "f1-score_macro": macro.get("f1-score"),
            "support_macro": macro.get("support"),
            "precision_weighted": weighted.get("precision"),
            "recall_weighted": weighted.get("recall"),
            "f1-score_weighted": weighted.get("f1-score"),
            "support_weighted": weighted.get("support"),
        }
        row.update(self.safe_class_metrics(report, 0))
        row.update(self.safe_class_metrics(report, 1))
        return row

    @staticmethod
    def pct_change(new_value, base_value):
        if pd.isna(new_value) or pd.isna(base_value) or base_value == 0:
            return np.nan
        return 100.0 * (new_value - base_value) / base_value

