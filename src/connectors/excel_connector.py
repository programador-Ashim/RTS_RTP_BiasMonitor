from __future__ import annotations
import pandas as pd
from .base_connector import BaseConnector


class ExcelConnector(BaseConnector):
    def __init__(self, file_obj, sheet_name=0, **read_kwargs):
        self.file_obj = file_obj
        self.sheet_name = sheet_name
        self.read_kwargs = read_kwargs

    def load_data(self) -> pd.DataFrame:
        df = pd.read_excel(self.file_obj, sheet_name=self.sheet_name, **self.read_kwargs)
        if df.empty:
            raise ValueError("Excel file is empty.")
        return df
