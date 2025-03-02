from typing import List, Dict, Any, Optional
import pandas as pd
import logging
from .patterns.hammer import HammerPattern


class CandlestickPatternFinder:
    """Finds candlestick patterns in price data."""

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger('candlestick_finder')

        # Register available patterns
        self.patterns = {
            'hammer': HammerPattern()
        }

        # As you add more patterns, you can register them here or use auto-discovery

    def find_patterns(self, df: pd.DataFrame,
                      selected_patterns: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Find selected patterns in the dataframe.

        Args:
            df: DataFrame with OHLCV data
            selected_patterns: List of pattern names to look for

        Returns:
            List of dictionaries with pattern details
        """
        if not selected_patterns:
            selected_patterns = list(self.patterns.keys())

        selected_patterns = [p.lower() for p in selected_patterns]

        all_patterns = []
        for pattern_name in selected_patterns:
            if pattern_name in self.patterns:
                pattern_obj = self.patterns[pattern_name]
                patterns = pattern_obj.find_patterns(df)
                all_patterns.extend(patterns)

        return all_patterns

    # Convenience methods for specific patterns
    def find_hammers(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find hammer patterns in the dataframe."""
        if 'hammer' in self.patterns:
            return self.patterns['hammer'].find_patterns(df)
        return []