from __future__ import annotations
import pandas as pd
from sqlalchemy import create_engine
from .base_connector import BaseConnector


class OracleConnector(BaseConnector):
    def __init__(self, host, port, service_name, user, password, query):
        self.host = host
        self.port = port
        self.service_name = service_name
        self.user = user
        self.password = password
        self.query = query

    def load_data(self) -> pd.DataFrame:
        dsn = f"{self.host}:{self.port}/?service_name={self.service_name}"
        engine = create_engine(
            f"oracle+oracledb://{self.user}:{self.password}@{dsn}"
        )
        df = pd.read_sql(self.query, engine)
        if df.empty:
            raise ValueError("Oracle query returned no rows.")
        return df
