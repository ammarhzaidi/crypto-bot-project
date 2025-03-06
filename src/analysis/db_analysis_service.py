"""
Database-enabled analysis service.
Extends analysis capabilities with database storage and retrieval.
"""

import pandas as pd
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

# Import existing service components and models
from src.analysis.analysis_service import AnalysisService, HHHLAnalysisParams, HHHLResult
from src.analysis.candlestick_analyzer import CandlestickAnalyzer

# Import database components
from src.utils.database import DatabaseManager
from src.repositories.market_repository import MarketRepository
from src.repositories.movers_repository import MoversRepository
from src.repositories.analysis_repository import AnalysisRepository


class DBAnalysisService:
    """
    Database-enabled analysis service.
    Works alongside the existing AnalysisService to add database storage capabilities.
    """

    def __init__(self, analysis_service: AnalysisService = None, db_path: str = "data/market_data.db"):
        """
        Initialize the database-enabled analysis service.

        Args:
            analysis_service: Existing analysis service instance or None to create a new one
            db_path: Path to SQLite database file
        """
        # Use provided analysis service or create a new one
        self.analysis_service = analysis_service or AnalysisService()

        # Set up logging
        self.logger = logging.getLogger('db_analysis_service')

        # Initialize database components
        self.db = DatabaseManager(db_path)
        self.market_repo = MarketRepository(self.db)
        self.movers_repo = MoversRepository(self.db)
        self.analysis_repo = AnalysisRepository(self.db)

        self.logger.info("Database-enabled analysis service initialized")

    def run_hhhl_analysis_with_storage(self, params: HHHLAnalysisParams) -> HHHLResult:
        """
        Run Higher Highs/Higher Lows analysis and store results in database.

        Args:
            params: Analysis parameters

        Returns:
            Analysis results
        """
        # Run the analysis using the existing service
        result = self.analysis_service.run_hhhl_analysis(params)

        # Store results in database
        if result:
            try:
                # Store market data if available
                market_data = []

                # Extract symbols from results
                symbols = []
                if result.uptrends:
                    symbols.extend([item['symbol'] for item in result.uptrends])
                if result.downtrends:
                    symbols.extend([item['symbol'] for item in result.downtrends])
                if result.no_trends:
                    symbols.extend(result.no_trends)

                # Try to get market data for these symbols
                client = self.analysis_service.client
                if hasattr(client, 'get_ticker'):
                    for symbol in symbols:
                        ticker = client.get_ticker(symbol)
                        if ticker and 'last_price' in ticker:
                            market_data.append({
                                'symbol': symbol,
                                'price': ticker['last_price'],
                                'change_24h': ticker.get('change_24h', 0),
                                'volume_24h': ticker.get('volume_24h', 0)
                            })

                    if market_data:
                        self.market_repo.save_market_snapshot(pd.DataFrame(market_data))

                # Store gainers and losers
                if result.uptrends or result.downtrends:
                    # Extract data for storage
                    gainers_data = []
                    for item in result.uptrends:
                        gainers_data.append({
                            'symbol': item['symbol'],
                            'price': item['price'],
                            'change_24h': item.get('change_24h', 0),
                            'volume_24h': item.get('volume', 0)
                        })

                    losers_data = []
                    for item in result.downtrends:
                        losers_data.append({
                            'symbol': item['symbol'],
                            'price': item['price'],
                            'change_24h': item.get('change_24h', 0),
                            'volume_24h': item.get('volume', 0)
                        })

                    # Convert to DataFrames
                    gainers_df = pd.DataFrame(gainers_data) if gainers_data else pd.DataFrame()
                    losers_df = pd.DataFrame(losers_data) if losers_data else pd.DataFrame()

                    # Store in database
                    if not gainers_df.empty or not losers_df.empty:
                        self.movers_repo.save_top_movers(gainers_df, losers_df)

                # Store as persistent movers if they meet criteria
                persistent_gainers = []
                for item in result.uptrends:
                    if item.get('strength', 0) >= 2:
                        persistent_gainers.append({
                            'symbol': item['symbol'],
                            'latest_price': item['price'],
                            'latest_change': item.get('change_24h', 0),
                            'appearances': item.get('strength', 2),
                            'trend_consistency': 75.0,  # Default value if not calculated
                            'acceleration': 0.0  # Default value if not calculated
                        })

                persistent_losers = []
                for item in result.downtrends:
                    if item.get('strength', 0) >= 2:
                        persistent_losers.append({
                            'symbol': item['symbol'],
                            'latest_price': item['price'],
                            'latest_change': item.get('change_24h', 0),
                            'appearances': item.get('strength', 2),
                            'trend_consistency': 75.0,  # Default value if not calculated
                            'acceleration': 0.0  # Default value if not calculated
                        })

                if persistent_gainers or persistent_losers:
                    self.analysis_repo.save_persistent_movers(persistent_gainers, persistent_losers)

                self.logger.info(f"Stored analysis results for {len(symbols)} symbols")

            except Exception as e:
                self.logger.error(f"Error storing analysis results: {str(e)}")

        return result

    def run_candlestick_analysis_with_storage(self, symbols: List[str],
                                              timeframe: str = "1h",
                                              candles_count: int = 100,
                                              patterns: List[str] = None) -> List[Dict]:
        """
        Run candlestick pattern analysis and store results.

        Args:
            symbols: List of symbols to analyze
            timeframe: Candlestick timeframe
            candles_count: Number of candles to analyze
            patterns: List of pattern names to look for

        Returns:
            List of detected patterns
        """
        # Get candlestick analyzer from existing service
        analyzer = None
        for field_name in dir(self.analysis_service):
            field = getattr(self.analysis_service, field_name)
            if isinstance(field, CandlestickAnalyzer):
                analyzer = field
                break

        if not analyzer:
            analyzer = CandlestickAnalyzer(logger=self.logger.info)

        # Run analysis
        patterns_result = analyzer.analyze_symbols(
            symbols,
            self.analysis_service.client,
            timeframe=timeframe,
            candles_count=candles_count,
            selected_patterns=patterns
        )

        # Store results
        if patterns_result:
            try:
                # Group patterns by symbol
                symbols_data = {}

                for pattern in patterns_result:
                    symbol = pattern['symbol']
                    if symbol not in symbols_data:
                        symbols_data[symbol] = {
                            'symbol': symbol,
                            'price': pattern['close'],
                            'change_24h': 0,  # Will be updated if available
                            'volume_24h': pattern.get('usd_volume', 0),
                            'patterns': []
                        }

                    # Add pattern to symbol's data
                    symbols_data[symbol]['patterns'].append(pattern['pattern_type'])

                # Store market data
                if symbols_data:
                    market_data = list(symbols_data.values())
                    self.market_repo.save_market_snapshot(pd.DataFrame(market_data), "candlestick_analysis")

                # Store bullish patterns as potential gainers
                gainers_data = []
                for symbol, data in symbols_data.items():
                    if any(p in data['patterns'] for p in ['Hammer', 'Bullish Engulfing', 'Morning Star', 'Piercing']):
                        gainers_data.append({
                            'symbol': symbol,
                            'price': data['price'],
                            'change_24h': data.get('change_24h', 0),
                            'volume_24h': data.get('volume_24h', 0)
                        })

                # Store as gainers if found
                if gainers_data:
                    gainers_df = pd.DataFrame(gainers_data)
                    losers_df = pd.DataFrame()  # Empty for candlestick analysis
                    self.movers_repo.save_top_movers(gainers_df, losers_df)

                self.logger.info(f"Stored candlestick analysis results for {len(symbols_data)} symbols")

            except Exception as e:
                self.logger.error(f"Error storing candlestick analysis results: {str(e)}")

        return patterns_result

    def get_historical_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """
        Get historical price data for a symbol.

        Args:
            symbol: Symbol to get data for
            days: Number of days to look back

        Returns:
            DataFrame with historical data
        """
        return self.market_repo.get_symbol_price_history(symbol, days)

    def get_frequent_symbols(self, days: int = 7, min_count: int = 2) -> List[Dict]:
        """
        Get symbols that frequently appear in top movers.

        Args:
            days: Number of days to look back
            min_count: Minimum number of appearances

        Returns:
            List of dictionaries with symbol stats
        """
        return self.movers_repo.get_frequent_symbols(days, min_count)

    def get_persistent_movers(self, days: int = 7, min_appearances: int = 3) -> Tuple[List[Dict], List[Dict]]:
        """
        Get historical data for persistent movers.

        Args:
            days: Number of days to look back
            min_appearances: Minimum number of appearances

        Returns:
            Tuple of (persistent_gainers, persistent_losers)
        """
        gainers_df, losers_df = self.analysis_repo.get_persistent_movers_history(days, min_appearances)

        # Convert to list of dictionaries
        persistent_gainers = gainers_df.to_dict('records') if not gainers_df.empty else []
        persistent_losers = losers_df.to_dict('records') if not losers_df.empty else []

        return persistent_gainers, persistent_losers

    def get_symbol_appearances(self, symbol: str, days: int = 7) -> Dict[str, Any]:
        """
        Get history of appearances for a specific symbol in top movers.

        Args:
            symbol: Symbol to check
            days: Number of days to look back

        Returns:
            Dictionary with appearance data
        """
        return self.movers_repo.get_symbol_appearances(symbol, days)