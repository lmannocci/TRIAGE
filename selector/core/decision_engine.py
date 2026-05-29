from selector.core.context import SelectorContext

import numpy as np
import pandas as pd
from typing import Optional

class SelectorDecisionEngine:
    def __init__(self, ctx: SelectorContext):
        self.ctx = ctx

    def _write_result(
        self,
        row: pd.Series,
        decision: str,
        prediction: Optional[float],
        use_enhanced_explanation: bool,
        use_enhancer_explanation: bool,
        complete_logging_case: str,
        classification_logging_case: str,
    ) -> pd.Series:
        is_abstain = decision == "abstain"
        is_non_abstain = not is_abstain
        target_value = row[self.ctx.target_col]

        if is_non_abstain and pd.notna(prediction) and pd.notna(target_value):
            final_correct = prediction == target_value
        else:
            final_correct = np.nan

        row[self.ctx.selector_decision_col] = decision
        row[self.ctx.selector_prediction_col] = prediction
        row[self.ctx.selector_use_enhanced_expl_col] = use_enhanced_explanation
        row[self.ctx.selector_use_enhancer_expl_col] = use_enhancer_explanation
        row[self.ctx.selector_complete_case_col] = complete_logging_case
        row[self.ctx.selector_case_col] = classification_logging_case
        row[self.ctx.selector_is_abstain_col] = is_abstain
        row[self.ctx.selector_is_non_abstain_col] = is_non_abstain
        row[self.ctx.selector_final_correct_col] = final_correct
        return row

    def apply_simple(self, row: pd.Series) -> pd.Series:
        agree = row["agree"]
        model_pred = row[self.ctx.model_pred_col]

        if agree:
            return self._write_result(
                row, "ok", model_pred, True, True, "[Y]agree", "[Y]agree"
            )
        return self._write_result(
            row, "abstain", None, False, False, "[ABSTAIN]disagree", "[ABSTAIN]disagree"
        )

    def apply_abstain(self, row: pd.Series) -> pd.Series:
        model_pred = row[self.ctx.model_pred_col]
        enhancer_pred = row[self.ctx.enhancer_pred_col]
        model_conf = row[self.ctx.model_conf_col]
        enhancer_conf = row[self.ctx.enhancer_conf_col]
        rbo = row[self.ctx.metric_complete_name]
        agree = row["agree"]

        model_high = model_conf >= self.ctx.th_model_conf
        enhancer_high = enhancer_conf >= self.ctx.th_enhancer_conf
        rbo_high = rbo >= self.ctx.th_rbo

        if agree:
            y = model_pred
            if model_high and enhancer_high:
                if rbo_high:
                    return self._write_result(
                        row, "ok", y, True, True,
                        "[Y]agree__both_high__rbo_high",
                        "[Y]agree__both_high",
                    )
                return self._write_result(
                    row, "ok_noEnhancerExplanation", y, True, False,
                    "[Y]agree__both_high__rbo_low",
                    "[Y]agree__both_high",
                )

            if model_high and not enhancer_high:
                if rbo_high:
                    return self._write_result(
                        row, "ok", y, True, True,
                        "[Y]agree__model_high_enhancer_low__rbo_high",
                        "[Y]agree__model_high_enhancer_low",
                    )
                return self._write_result(
                    row, "ok_noEnhancerExplanation", y, True, False,
                    "[Y]agree__both_high__rbo_low",
                    "[Y]agree__both_high",
                )

            if (not model_high) and enhancer_high:
                if rbo_high:
                    return self._write_result(
                        row, "ok", y, True, True,
                        "[Y]agree__model_low_enhancer_high__rbo_high",
                        "[Y]agree__model_low_enhancer_high",
                    )
                return self._write_result(
                    row, "abstain", None, False, False,
                    "[ABSTAIN]agree__model_low_enhancer_high__rbo_low",
                    "[ABSTAIN]agree__model_low_enhancer_high",
                )

            return self._write_result(
                row, "abstain", None, False, False,
                "[ABSTAIN]agree__both_low",
                "[ABSTAIN]agree__both_low",
            )

        # Disagreement cases
        if model_high and not enhancer_high:
            return self._write_result(
                row, "ok_noEnhancerExplanation", model_pred, True, False,
                "[MODEL_Y]disagree__model_high_enhancer_low",
                "[MODEL_Y]disagree__model_high_enhancer_low",
            )

        if (not model_high) and enhancer_high:
            return self._write_result(
                row, "ok_noModelExplanation", enhancer_pred, False, True,
                "[ENHANCER_Y]disagree__model_low_enhancer_high",
                "[ENHANCER_Y]disagree__model_low_enhancer_high",
            )

        if model_high and enhancer_high:
            return self._write_result(
                row, "abstain", None, False, False,
                "[ABSTAIN]disagree__both_high",
                "[ABSTAIN]disagree__both_high",
            )

        return self._write_result(
            row, "abstain", None, False, False,
            "[ABSTAIN]disagree__both_low",
            "[ABSTAIN]disagree__both_low",
        )

