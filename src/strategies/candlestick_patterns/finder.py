from typing import List, Dict, Any, Optional, Type
import pandas as pd
import logging
from .base_pattern import CandlestickPattern
from .patterns.hammer import HammerPattern
from .patterns.bullish_engulfing import BullishEngulfingPattern
from .patterns.piercing_pattern import PiercingPattern
from .patterns.morning_star import MorningStarPattern


class CandlestickPatternFinder:
    """Finds candlestick patterns in price data."""

    def __init__(self, logger=None):
        if logger is not None and callable(logger):
            self.logger = logger
        else:
            # Create a standard logger if not provided or not callable
            self.logger = logging.getLogger('candlestick_finder')

        # Register available patterns
        self.patterns = {
            'hammer': HammerPattern(),
            'bullish_engulfing': BullishEngulfingPattern(),
            'piercing': PiercingPattern(),
            'morning_star': MorningStarPattern()
        }

        # Configuration for optional confirmations
        self.confirmations = {
            'bullish_engulfing': {
                'use_volume_confirmation': False,
                'use_prior_trend': False,
                'use_size_significance': False
            },
            'piercing': {
                'use_volume_confirmation': False,
                'use_prior_trend': False
            },
            'morning_star': {
                'use_volume_confirmation': False,
                'use_prior_trend': False
            }
        }

    def set_pattern_confirmation(self, pattern_name: str, confirmation_name: str, enabled: bool):
        """
        Enable or disable a specific confirmation for a pattern.

        Args:
            pattern_name: Name of the pattern (e.g., 'bullish_engulfing', 'piercing', 'morning_star')
            confirmation_name: Name of the confirmation (e.g., 'use_volume_confirmation')
            enabled: Whether the confirmation should be enabled
        """
        if pattern_name in self.confirmations and confirmation_name in self.confirmations[pattern_name]:
            self.confirmations[pattern_name][confirmation_name] = enabled

            # Update the pattern instance with new confirmation settings
            if pattern_name == 'bullish_engulfing':
                # Create a new instance with updated settings
                self.patterns[pattern_name] = BullishEngulfingPattern(
                    use_volume_confirmation=self.confirmations[pattern_name]['use_volume_confirmation'],
                    use_prior_trend=self.confirmations[pattern_name]['use_prior_trend'],
                    use_size_significance=self.confirmations[pattern_name]['use_size_significance']
                )
            elif pattern_name == 'piercing':
                # Create a new instance with updated settings
                self.patterns[pattern_name] = PiercingPattern(
                    use_volume_confirmation=self.confirmations[pattern_name]['use_volume_confirmation'],
                    use_prior_trend=self.confirmations[pattern_name]['use_prior_trend']
                )
            elif pattern_name == 'morning_star':
                # Create a new instance with updated settings
                self.patterns[pattern_name] = MorningStarPattern(
                    use_volume_confirmation=self.confirmations[pattern_name]['use_volume_confirmation'],
                    use_prior_trend=self.confirmations[pattern_name]['use_prior_trend']
                )

            if hasattr(self, 'logger'):
                if callable(self.logger):
                    # If it's a function (like log_cs), call it
                    self.logger(f"Updated {pattern_name} with {confirmation_name}={enabled}")
                elif hasattr(self.logger, 'info'):
                    # If it's a Logger object, call the info method
                    self.logger.info(f"Updated {pattern_name} with {confirmation_name}={enabled}")

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

    def find_piercing(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find piercing patterns in the dataframe."""
        if 'piercing' in self.patterns:
            return self.patterns['piercing'].find_patterns(df)
        return []

    def find_morning_stars(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find morning star patterns in the dataframe."""
        if 'morning_star' in self.patterns:
            return self.patterns['morning_star'].find_patterns(df)
        return []


# Add a simple test at the end
if __name__ == "__main__":
    # Create a simple test dataframe
    import pandas as pd

    # Create a sample dataframe with a morning star pattern
    data = {
        'timestamp': [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-02'), pd.Timestamp('2023-01-03')],
        'open': [100, 85, 83],
        'high': [105, 87, 95],
        'low': [95, 80, 81],
        'close': [97, 83, 93],
        'volume': [1000, 800, 1500]
    }
    df = pd.DataFrame(data)

    # Create the pattern finder
    finder = CandlestickPatternFinder()

    # Find morning star patterns
    patterns = finder.find_morning_stars(df)

    print(f"Found {len(patterns)} morning star patterns")
    for pattern in patterns:
        print(f"Pattern at index {pattern['index']}, strength: {pattern['strength']:.2f}")