from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class AttrReport:
    protected_attr: str
    dp_diff: float
    eo_diff: float
    alert: bool
    by_group: pd.DataFrame

def _safe_rate(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0

def batch_fairness_report(
    y_true: pd.Series,
    y_pred: pd.Series,
    protected_df: pd.DataFrame,
    threshold: float = 0.10,
) -> list[AttrReport]:
    """Compute DP diff + EO diff per protected attribute.
    DP diff: max(selection_rate) - min(selection_rate)
    EO diff: max(max(|TPR_i-TPR_j|, |FPR_i-FPR_j|)) across groups
    """
    reports: list[AttrReport] = []
    y_true = pd.Series(y_true).astype(int)
    y_pred = pd.Series(y_pred).astype(int)

    for attr in protected_df.columns:
        s = protected_df[attr].astype(str).fillna("Unknown")

        rows = []
        for g in sorted(s.unique()):
            idx = (s == g)
            yt = np.asarray(y_true)[idx.to_numpy()]

            yp = np.asarray(y_pred)[idx.to_numpy()]


            tp = int(((yt == 1) & (yp == 1)).sum())
            tn = int(((yt == 0) & (yp == 0)).sum())
            fp = int(((yt == 0) & (yp == 1)).sum())
            fn = int(((yt == 1) & (yp == 0)).sum())

            selection_rate = float(yp.mean()) if len(yp) else 0.0
            tpr = _safe_rate(tp, tp + fn)
            fpr = _safe_rate(fp, fp + tn)
            acc = _safe_rate(tp + tn, tp + tn + fp + fn)

            rows.append({
                "group": g,
                "n": int(idx.sum()),
                "selection_rate": selection_rate,
                "tpr": tpr,
                "fpr": fpr,
                "accuracy": acc,
            })

        by_group = pd.DataFrame(rows).set_index("group")

        if len(by_group) >= 2:
            dp_diff = float(by_group["selection_rate"].max() - by_group["selection_rate"].min())
            tpr_diff = float(by_group["tpr"].max() - by_group["tpr"].min())
            fpr_diff = float(by_group["fpr"].max() - by_group["fpr"].min())
            eo_diff = float(max(tpr_diff, fpr_diff))
        else:
            dp_diff = 0.0
            eo_diff = 0.0

        alert = bool((dp_diff > threshold) or (eo_diff > threshold))
        reports.append(AttrReport(attr, dp_diff, eo_diff, alert, by_group))

    return reports
