from __future__ import annotations
import pandas as pd

from src.processing.scalable_cleaner import clean_chunk


def load_large_csv_in_chunks(
    file_obj,
    chunksize: int = 50_000,
    target_cols: list[str] | None = None,
    protected_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Read a CSV in chunks, clean each chunk, and combine the results.
    This is the scalable path for large uploaded CSV files.
    """
    if hasattr(file_obj, "seek"):
        file_obj.seek(0)

    chunks = []
    for chunk in pd.read_csv(file_obj, chunksize=chunksize):
        cleaned = clean_chunk(
            chunk,
            target_cols=target_cols,
            protected_cols=protected_cols,
        )
        chunks.append(cleaned)

    if not chunks:
        raise ValueError("No data found in CSV.")

    return pd.concat(chunks, ignore_index=True)
