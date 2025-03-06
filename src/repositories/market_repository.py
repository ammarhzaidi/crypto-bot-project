"""
Repository for market data operations.
Handles storing and retrieving market snapshots and price data.
"""

import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Optional, Any

from src.utils.database import DatabaseManager


class MarketRepository:
    """
    Repository for market data storage and retrieval.
    Focuses on raw market data and price information.
    """

    def __init__(self, db_manager: DatabaseManager = None):
        """
        Initialize the market repository.

        Args:
            db_manager: Database manager instance or None to create a new one
        """
        self.db = db_manager or DatabaseManager()

    def save_market_snapshot(self, df: pd.DataFrame, snapshot_type: str = "full_market") -> bool:
        """
        Save a market data snapshot to the database.

        Args:
            df: DataFrame with market data
            snapshot_type: Type of snapshot (e.g., "full_market", "top_gainers", "top_losers")

        Returns:
            Success status
        """
        try:
            # Convert DataFrame to JSON string
            json_data = df.to_json(orient="records")

            # Save snapshot to database
            timestamp = datetime.now().isoformat()

            self.db.execute_query(
                "INSERT INTO market_snapshots (timestamp, snapshot_type, raw_data) VALUES (?, ?, ?)",
                (timestamp, snapshot_type, json_data)
            )

            # Update symbols table and save price data
            symbols_data = []
            price_data = []

            for _, row in df.iterrows():
                symbol = row['symbol']

                # Prepare symbols data
                symbols_data.append(
                    (symbol, timestamp, timestamp)
                )

                # Prepare price data
                price_data.append(
                    (
                        symbol,
                        timestamp,
                        row['price'],
                        row.get('change_24h', None),
                        row.get('volume_24h', None)
                    )
                )

            # Batch update symbols
            self.db.execute_many(
                """
                INSERT INTO symbols (symbol, first_seen, last_seen) 
                VALUES (?, ?, ?)
                ON CONFLICT(symbol) 
                DO UPDATE SET last_seen = ?
                """,
                [(s[0], s[1], s[2], s[2]) for s in symbols_data]
            )

            # Batch insert price data
            self.db.execute_many(
                """
                INSERT OR REPLACE INTO price_data 
                (symbol, timestamp, price, change_24h, volume_24h)
                VALUES (?, ?, ?, ?, ?)
                """,
                price_data
            )

            return True
        except Exception as e:
            self.db.logger.error(f"Error saving market snapshot: {str(e)}")
            return False

    def get_latest_snapshot(self, snapshot_type: str = "full_market") -> Optional[pd.DataFrame]:
        """
        Get the latest market snapshot of a specific type.

        Args:
            snapshot_type: Type of snapshot to retrieve

        Returns:
            DataFrame with market data or None if no data found
        """
        try:
            result = self.db.execute_query(
                """
                SELECT timestamp, raw_data
                FROM market_snapshots
                WHERE snapshot_type = ?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (snapshot_type,)
            )

            if not result:
                return None

            timestamp, raw_data = result[0]

            # Parse JSON data into DataFrame
            df = pd.read_json(raw_data)

            return df
        except Exception as e:
            self.db.logger.error(f"Error retrieving latest snapshot: {str(e)}")
            return None

    def get_historical_snapshots(self, snapshot_type: str = "full_market",
                                 limit: int = 24) -> List[Dict[str, Any]]:
        """
        Get historical market snapshots.

        Args:
            snapshot_type: Type of snapshot to retrieve
            limit: Maximum number of snapshots to retrieve

        Returns:
            List of dictionaries with timestamp and data
        """
        try:
            result = self.db.execute_query(
                """
                SELECT timestamp, raw_data
                FROM market_snapshots
                WHERE snapshot_type = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (snapshot_type, limit)
            )

            if not result:
                return []

            snapshots = []
            for timestamp, raw_data in result:
                df = pd.read_json(raw_data)
                snapshots.append({
                    'timestamp': timestamp,
                    'data': df
                })

            return snapshots
        except Exception as e:
            self.db.logger.error(f"Error retrieving historical snapshots: {str(e)}")
            return []

    def get_symbol_price_history(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """
        Get price history for a specific symbol.

        Args:
            symbol: Symbol to get history for
            days: Number of days to look back

        Returns:
            DataFrame with price history
        """
        try:
            query = f"""
            SELECT timestamp, price, change_24h, volume_24h
            FROM price_data
            WHERE symbol = ?
            AND timestamp >= datetime('now', '-{days} days')
            ORDER BY timestamp ASC
            """

            result = self.db.execute_query(query, (symbol,))

            if not result:
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(result, columns=['timestamp', 'price', 'change_24h', 'volume_24h'])

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            return df
        except Exception as e:
            self.db.logger.error(f"Error getting symbol price history: {str(e)}")
            return pd.DataFrame()

    def get_all_tracked_symbols(self) -> List[str]:
        """
        Get list of all tracked symbols.

        Returns:
            List of symbol strings
        """
        try:
            result = self.db.execute_query(
                "SELECT symbol FROM symbols ORDER BY symbol"
            )

            if not result:
                return []

            return [row[0] for row in result]
        except Exception as e:
            self.db.logger.error(f"Error getting tracked symbols: {str(e)}")
            return []