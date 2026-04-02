from __future__ import annotations
import pandas as pd

def stream_batches(df: pd.DataFrame, batch_size: int = 100):
    n = len(df)
    for start in range(0, n, batch_size):
        yield df.iloc[start:start+batch_size].copy()
