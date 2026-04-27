from __future__ import annotations
import pandas as pd
from sqlalchemy import create_engine
from .base_connector import BaseConnector


class PostgreSQLConnector(BaseConnector):
    def __init__(self, host, port, user, password, database, query):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.query = query

    def load_data(self) -> pd.DataFrame:
        engine = create_engine(
            f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        )
        df = pd.read_sql(self.query, engine)
        if df.empty:
            raise ValueError("PostgreSQL query returned no rows.")
        return df
