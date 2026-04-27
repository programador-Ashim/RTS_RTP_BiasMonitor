from __future__ import annotations
import pandas as pd
from sqlalchemy import create_engine
from .base_connector import BaseConnector


class SQLServerConnector(BaseConnector):
    def __init__(self, host, database, user, password, query, driver="ODBC Driver 17 for SQL Server"):
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.query = query
        self.driver = driver

    def load_data(self) -> pd.DataFrame:
        conn_str = (
            f"mssql+pyodbc://{self.user}:{self.password}@{self.host}/{self.database}"
            f"?driver={self.driver.replace(' ', '+')}"
        )
        engine = create_engine(conn_str)
        df = pd.read_sql(self.query, engine)
        if df.empty:
            raise ValueError("SQL Server query returned no rows.")
        return df
