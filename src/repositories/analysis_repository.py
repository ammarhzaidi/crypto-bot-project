"""
Repository for market analysis operations.
Handles storing and retrieving persistent movers analysis.
"""

import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

from src.utils.database import DatabaseManager


class AnalysisRepository:
    """
    Repository for market analysis storage and retrieval.
    Focuses on persistent movers and trends.
    """

    def __init__(self, db_manager: DatabaseManager = None):
        """
        Initialize the analysis repository.

        Args:
            db_manager: Database manager instance or None to create a new one
        """
        self.db = db_manager or DatabaseManager()

    def save_persistent_movers(self, gainers: List[Dict], losers: List[Dict]) -> bool:
        """
        Save persistent movers analysis to the database.

        Args:
            gainers: List of persistent gainers dictionaries
            losers: List of persistent losers dictionaries

        Returns:
            Success status
        """
        try:
            timestamp = datetime.now().isoformat()

            # Prepare data for batch insertion
            gainers_data = []
            losers_data = []

            # Prepare gainers data
            for data in gainers:
                gainers_data.append((
                    data['symbol'],
                    timestamp,
                    'gainer',
                    data['latest_price'],
                    data['latest_change'],
                    data['appearances'],
                    data.get('trend_consistency', None),
                    data.get('acceleration', None)
                ))

            # Prepare losers data
            for data in losers:
                losers_data.append((
                    data['symbol'],
                    timestamp,
                    'loser',
                    data['latest_price'],
                    data['latest_change'],
                    data['appearances'],
                    data.get('trend_consistency', None),
                    data.get('acceleration', None)
                ))

            # Batch insert
            all_data = gainers_data + losers_data
            if all_data:
                self.db.execute_many(
                    """
                    INSERT OR REPLACE INTO persistent_movers
                    (symbol, timestamp, mover_type, price, change_24h, appearances, 
                     trend_consistency, acceleration)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    all_data
                )

            return True
        except Exception as e:
            self.db.logger.error(f"Error saving persistent movers: {str(e)}")
            return False

    def get_latest_persistent_movers(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Get the most recent persistent movers analysis.

        Returns:
            Tuple of (persistent_gainers, persistent_losers)
        """
        try:
            # Get latest timestamp
            latest_timestamp_result = self.db.execute_query(
                """
                SELECT timestamp FROM persistent_movers
                ORDER BY timestamp DESC LIMIT 1
                """
            )

            if not latest_timestamp_result:
                return [], []

            latest_timestamp = latest_timestamp_result[0][0]

            # Get persistent gainers at that timestamp
            gainers_result = self.db.execute_query(
                """
                SELECT symbol, price, change_24h, appearances, trend_consistency, acceleration
                FROM persistent_movers
                WHERE timestamp = ? AND mover_type = 'gainer'
                ORDER BY appearances DESC, change_24h DESC
                """,
                (latest_timestamp,)
            )

            # Get persistent losers at that timestamp
            losers_result = self.db.execute_query(
                """
                SELECT symbol, price, change_24h, appearances, trend_consistency, acceleration
                FROM persistent_movers
                WHERE timestamp = ? AND mover_type = 'loser'
                ORDER BY appearances DESC, change_24h ASC
                """,
                (latest_timestamp,)
            )

            # Convert to dictionaries
            persistent_gainers = []
            for row in gainers_result:
                persistent_gainers.append({
                    'symbol': row[0],
                    'latest_price': row[1],
                    'latest_change': row[2],
                    'appearances': row[3],
                    'trend_consistency': row[4],
                    'acceleration': row[5]
                })

            persistent_losers = []
            for row in losers_result:
                persistent_losers.append({
                    'symbol': row[0],
                    'latest_price': row[1],
                    'latest_change': row[2],
                    'appearances': row[3],
                    'trend_consistency': row[4],
                    'acceleration': row[5]
                })

            return persistent_gainers, persistent_losers
        except Exception as e:
            self.db.logger.error(f"Error getting latest persistent movers: {str(e)}")
            return [], []

    def get_persistent_movers_history(self, days: int = 7, min_appearances: int = 3) -> Tuple[
        pd.DataFrame, pd.DataFrame]:
        """
        Get historical data for persistent movers.

        Args:
            days: Number of days to look back
            min_appearances: Minimum number of appearances to include

        Returns:
            Tuple of (persistent_gainers_df, persistent_losers_df)
        """
        try:
            # Query for persistent gainers
            gainers_query = f"""
            SELECT symbol, timestamp, price, change_24h, appearances, trend_consistency, acceleration
            FROM persistent_movers
            WHERE mover_type = 'gainer'
            AND appearances >= ?
            AND timestamp >= datetime('now', '-{days} days')
            ORDER BY timestamp DESC, appearances DESC
            """

            gainers_result = self.db.execute_query(gainers_query, (min_appearances,))

            # Query for persistent losers
            losers_query = f"""
            SELECT symbol, timestamp, price, change_24h, appearances, trend_consistency, acceleration
            FROM persistent_movers
            WHERE mover_type = 'loser'
            AND appearances >= ?
            AND timestamp >= datetime('now', '-{days} days')
            ORDER BY timestamp DESC, appearances DESC
            """

            losers_result = self.db.execute_query(losers_query, (min_appearances,))

            # Convert to DataFrames
            gainers_df = pd.DataFrame(
                gainers_result,
                columns=['symbol', 'timestamp', 'price', 'change_24h', 'appearances',
                         'trend_consistency', 'acceleration']
            ) if gainers_result else pd.DataFrame()

            losers_df = pd.DataFrame(
                losers_result,
                columns=['symbol', 'timestamp', 'price', 'change_24h', 'appearances',
                         'trend_consistency', 'acceleration']
            ) if losers_result else pd.DataFrame()

            return gainers_df, losers_df
        except Exception as e:
            self.db.logger.error(f"Error getting persistent movers history: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()

    def get_symbol_persistence_history(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """
        Get persistent mover history for a specific symbol.

        Args:
            symbol: Symbol to check
            days: Number of days to look back

        Returns:
            Dictionary with persistence history
        """
        try:
            query = f"""
            SELECT timestamp, mover_type, price, change_24h, appearances, 
                   trend_consistency, acceleration
            FROM persistent_movers
            WHERE symbol = ?
            AND timestamp >= datetime('now', '-{days} days')
            ORDER BY timestamp DESC
            """

            result = self.db.execute_query(query, (symbol,))

            if not result:
                return {
                    'symbol': symbol,
                    'history': []
                }

            history = []
            for row in result:
                history.append({
                    'timestamp': row[0],
                    'mover_type': row[1],
                    'price': row[2],
                    'change_24h': row[3],
                    'appearances': row[4],
                    'trend_consistency': row[5],
                    'acceleration': row[6],
                })

            return {
                'symbol': symbol,
                'history': history
            }
        except Exception as e:
            self.db.logger.error(f"Error getting symbol persistence history: {str(e)}")
            return {
                'symbol': symbol,
                'history': []
            }