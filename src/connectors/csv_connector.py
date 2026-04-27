from __future__ import annotations
import pandas as pd
from .base_connector import BaseConnector


class CSVConnector(BaseConnector):
    def __init__(self, file_obj, **read_kwargs):
        self.file_obj = file_obj
        self.read_kwargs = read_kwargs

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.file_obj, **self.read_kwargs)
        if df.empty:
            raise ValueError("CSV file is empty.")
        return df
