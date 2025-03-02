from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict, Any


class CandlestickPattern(ABC):
    """Base class for all candlestick patterns."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the pattern."""
        pass

    @abstractmethod
    def find_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Find all instances of this pattern in the dataframe.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            List of dictionaries with pattern details
        """
        pass