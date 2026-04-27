from __future__ import annotations


def choose_processing_engine(
    n_rows: int | None = None,
    file_size_mb: float | None = None,
    row_threshold: int = 300_000,
    file_threshold_mb: float = 200.0,
) -> str:
    """
    Choose the processing mode for the dataset.

    Returns:
        'pandas'  -> normal in-memory processing
        'chunked' -> scalable chunk-based processing
        'spark'   -> reserved future distributed mode
    """
    if file_size_mb is not None and file_size_mb >= file_threshold_mb:
        return "chunked"

    if n_rows is not None and n_rows >= row_threshold:
        return "chunked"

    return "pandas"
