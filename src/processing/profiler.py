from __future__ import annotations
import pandas as pd


def dataset_profile(df: pd.DataFrame, target: str | None = None) -> dict:
    """Backend profiling for scalability decisions."""
    n = len(df)

    if n == 0:
        return {
            "rows": 0,
            "cols": 0,
            "memory_mb": 0.0,
            "missing_pct": 0.0,
            "dup_pct": 0.0,
            "target_pos_rate": None,
        }

    missing_pct = float(df.isna().mean().mean() * 100)
    dup_pct = float(df.duplicated().mean() * 100)
    memory_mb = float(df.memory_usage(deep=True).sum() / (1024 ** 2))

    target_pos_rate = None
    if target and target in df.columns:
        try:
            y = pd.to_numeric(df[target], errors="coerce")
            if y.notna().any():
                target_pos_rate = float(y.mean() * 100)
        except Exception:
            target_pos_rate = None

    return {
        "rows": int(n),
        "cols": int(df.shape[1]),
        "memory_mb": memory_mb,
        "missing_pct": missing_pct,
        "dup_pct": dup_pct,
        "target_pos_rate": target_pos_rate,
    }
