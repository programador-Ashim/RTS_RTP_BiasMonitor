from __future__ import annotations
import pandas as pd

from src.preprocess import (
    normalize_column_names,
    coerce_datetime_cols,
    coerce_numeric_cols,
    normalize_categorical_values,
    fill_missing_values,
    cap_outliers_iqr,
    adapt_athlete_dataset,
    ensure_rts_rtp,
)


def clean_chunk(
    df: pd.DataFrame,
    target_cols: list[str] | None = None,
    protected_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Clean a chunk using the same logic as the main preprocessing pipeline,
    but in a chunk-friendly way for large CSV ingestion.
    """
    target_cols = target_cols or []
    protected_cols = protected_cols or []

    out = df.copy()
    out = normalize_column_names(out)
    out = coerce_datetime_cols(out)
    out = coerce_numeric_cols(out)
    out = normalize_categorical_values(out)
    out = adapt_athlete_dataset(out)
    out = fill_missing_values(out)

    exclude = set(target_cols) | set(protected_cols)
    out = cap_outliers_iqr(out, exclude_cols=exclude)
    out = ensure_rts_rtp(out)

    return out.reset_index(drop=True)
