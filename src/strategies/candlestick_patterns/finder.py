from typing import List, Dict, Any, Optional, Type
import pandas as pd
import logging
from .base_pattern import CandlestickPattern
from .patterns.hammer import HammerPattern
from .patterns.bullish_engulfing import BullishEngulfingPattern


class CandlestickPatternFinder:
    """Finds candlestick patterns in price data."""

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger('candlestick_finder')

        # Register available patterns
        self.patterns = {
            'hammer': HammerPattern(),
            'bullish_engulfing': BullishEngulfingPattern()
        }

        # Configuration for optional confirmations
        self.confirmations = {
            'bullish_engulfing': {
                'use_volume_confirmation': False,
                'use_prior_trend': False,
                'use_size_significance': False
            }
        }

    def set_pattern_confirmation(self, pattern_name: str, confirmation_name: str, enabled: bool):
        """
        Enable or disable a specific confirmation for a pattern.

        Args:
            pattern_name: Name of the pattern (e.g., 'bullish_engulfing')
            confirmation_name: Name of the confirmation (e.g., 'use_volume_confirmation')
            enabled: Whether the confirmation should be enabled
        """
        if pattern_name in self.confirmations and confirmation_name in self.confirmations[pattern_name]:
            self.confirmations[pattern_name][confirmation_name] = enabled

            # Update the pattern instance with new confirmation settings
            if pattern_name == 'bullish_engulfing' and pattern_name in self.patterns:
                # Create a new instance with updated settings
                self.patterns[pattern_name] = BullishEngulfingPattern(
                    use_volume_confirmation=self.confirmations[pattern_name]['use_volume_confirmation'],
                    use_prior_trend=self.confirmations[pattern_name]['use_prior_trend'],
                    use_size_significance=self.confirmations[pattern_name]['use_size_significance']
                )

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

    def find_bullish_engulfing(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find bullish engulfing patterns in the dataframe."""
        if 'bullish_engulfing' in self.patterns:
            return self.patterns['bullish_engulfing'].find_patterns(df)
        return []