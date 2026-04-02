from __future__ import annotations
import pandas as pd
import numpy as np

def dataset_health(df: pd.DataFrame, target: str | None = None) -> dict:
    """Simple 'Data Health' score like a mini Biasense:
    - missingness
    - duplicate rate
    - target balance (if provided)
    """
    n = len(df)
    if n == 0:
        return {"rows": 0, "missing_pct": 0.0, "dup_pct": 0.0, "target_pos_rate": None}

    missing_pct = float(df.isna().mean().mean() * 100)
    dup_pct = float(df.duplicated().mean() * 100)

    target_pos_rate = None
    if target and target in df.columns:
        try:
            y = df[target].astype(int)
            target_pos_rate = float(y.mean() * 100)
        except Exception:
            target_pos_rate = None

    return {
        "rows": int(n),
        "cols": int(df.shape[1]),
        "missing_pct": missing_pct,
        "dup_pct": dup_pct,
        "target_pos_rate": target_pos_rate,
    }
