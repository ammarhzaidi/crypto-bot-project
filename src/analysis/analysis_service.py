"""
Centralized service for running different types of market analysis.
This service coordinates between data access and analysis algorithms.
"""

from typing import List, Dict, Any, Optional, NamedTuple, Callable
import time
from dataclasses import dataclass
from src.market_data.okx_client import OKXClient
from src.analysis.hh_hl_analyzer import HHHLAnalyzer
from src.strategies.candlestick_patterns import CandlestickPatternFinder

@dataclass
class HHHLAnalysisParams:
    """Parameters for HH/HL analysis."""
    symbols_count: int
    tp_percent: float
    sl_percent: float
    check_freshness: bool
    timeframe: str = "1h"
    candles_count: int = 48

@dataclass
class CandlestickAnalysisParams:
    """Parameters for candlestick pattern analysis."""
    symbols_count: int
    timeframe: str
    candles_count: int
    patterns: List[str]

@dataclass
class AnalysisResult:
    """Base class for analysis results."""
    execution_time: float

@dataclass
class HHHLResult(AnalysisResult):
    """Results from HH/HL analysis."""
    uptrends: List[Dict[str, Any]]
    downtrends: List[Dict[str, Any]]
    no_trends: List[str]

@dataclass
class CandlestickResult(AnalysisResult):
    """Results from candlestick pattern analysis."""
    patterns: List[Dict[str, Any]]

class AnalysisService:
    """
    Service for coordinating market analysis operations.
    """

    def __init__(self, client: Optional[OKXClient] = None, logger: Optional[Callable] = None):
        """
        Initialize the analysis service.

        Args:
            client: OKXClient instance or None to create a new one
            logger: Optional logger function for analysis progress
        """
        self.client = client or OKXClient()
        self.logger = logger

    def run_hhhl_analysis(self, params: HHHLAnalysisParams) -> HHHLResult:
        """
        Run Higher Highs/Higher Lows analysis.

        Args:
            params: Analysis parameters

        Returns:
            Analysis results
        """
        start_time = time.time()

        # Get top symbols
        symbols = self.client.get_top_volume_symbols(limit=params.symbols_count)

        if not symbols:
            if self.logger:
                self.logger("No symbols found. Analysis failed.")
            return HHHLResult(
                uptrends=[],
                downtrends=[],
                no_trends=[],
                execution_time=time.time() - start_time
            )

        # Create analyzer with freshness setting
        analyzer = HHHLAnalyzer(check_freshness=params.check_freshness, logger=self.logger)

        # Run analysis
        uptrends, downtrends, no_trends = analyzer.analyze_symbols(
            symbols,
            self.client,
            tp_percent=params.tp_percent,
            sl_percent=params.sl_percent,
            timeframe=params.timeframe,
            candles_count=params.candles_count
        )

        # Calculate execution time
        execution_time = time.time() - start_time

        # Return results
        return HHHLResult(
            uptrends=uptrends,
            downtrends=downtrends,
            no_trends=no_trends,
            execution_time=execution_time
        )

    def run_candlestick_analysis(self, params: CandlestickAnalysisParams) -> CandlestickResult:
        """
        Run candlestick pattern analysis.

        Args:
            params: Analysis parameters

        Returns:
            Analysis results
        """
        # We'll implement this later when we refactor the candlestick analysis
        start_time = time.time()

        # Placeholder for now
        return CandlestickResult(
            patterns=[],
            execution_time=time.time() - start_time
        )