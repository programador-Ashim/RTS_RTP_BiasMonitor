from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd


class BaseConnector(ABC):
    """Common interface for all data connectors."""

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """Return loaded data as a pandas DataFrame."""
        raise NotImplementedError
