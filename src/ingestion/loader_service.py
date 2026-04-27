from __future__ import annotations
import pandas as pd
from src.connectors.connector_factory import ConnectorFactory
from src.processing.engine_selector import choose_processing_engine
from src.processing.scalable_loader import load_large_csv_in_chunks


class DataLoaderService:
    @staticmethod
    def load_from_uploaded_file(uploaded_file) -> pd.DataFrame:
        if uploaded_file is None:
            raise ValueError("No uploaded file provided.")

        name = uploaded_file.name.lower()
        file_size_mb = uploaded_file.size / (1024 ** 2) if hasattr(uploaded_file, "size") else None

        if name.endswith(".csv"):
            engine = choose_processing_engine(file_size_mb=file_size_mb)

            if engine == "chunked":
                uploaded_file.seek(0)
                df = load_large_csv_in_chunks(
                    uploaded_file,
                    chunksize=50_000,
                    target_cols=["rts", "rtp", "ready"],
                    protected_cols=["gender", "race", "age", "age_group"],
                )
            else:
                connector = ConnectorFactory.get_connector("csv", file_obj=uploaded_file)
                df = connector.load_data()

        elif name.endswith(".xlsx") or name.endswith(".xls"):
            connector = ConnectorFactory.get_connector("excel", file_obj=uploaded_file)
            df = connector.load_data()
        else:
            raise ValueError("Unsupported file type. Please upload CSV or Excel.")

        return DataLoaderService._standardize(df)

    @staticmethod
    def load_from_database(source_type: str, **kwargs) -> pd.DataFrame:
        connector = ConnectorFactory.get_connector(source_type, **kwargs)
        df = connector.load_data()
        return DataLoaderService._standardize(df)

    @staticmethod
    def _standardize(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        return df