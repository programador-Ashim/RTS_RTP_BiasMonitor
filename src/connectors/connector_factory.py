from __future__ import annotations
from .csv_connector import CSVConnector
from .excel_connector import ExcelConnector
from .mysql_connector import MySQLConnector
from .postgres_connector import PostgreSQLConnector
from .sqlserver_connector import SQLServerConnector
from .oracle_connector import OracleConnector


class ConnectorFactory:
    @staticmethod
    def get_connector(source_type: str, **kwargs):
        source_type = source_type.lower()

        if source_type == "csv":
            return CSVConnector(**kwargs)
        if source_type == "excel":
            return ExcelConnector(**kwargs)
        if source_type == "mysql":
            return MySQLConnector(**kwargs)
        if source_type == "postgresql":
            return PostgreSQLConnector(**kwargs)
        if source_type == "sqlserver":
            return SQLServerConnector(**kwargs)
        if source_type == "oracle":
            return OracleConnector(**kwargs)

        raise ValueError(f"Unsupported source type: {source_type}")
