"""
Repository for top market movers operations.
Handles storing and retrieving data for top gainers and losers.
"""

import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

from src.utils.database import DatabaseManager


class MoversRepository:
    """
    Repository for top market movers storage and retrieval.
    Focuses on top gainers and losers data.
    """

    def __init__(self, db_manager: DatabaseManager = None):
        """
        Initialize the movers repository.

        Args:
            db_manager: Database manager instance or None to create a new one
        """
        self.db = db_manager or DatabaseManager()

    def save_top_movers(self, gainers: pd.DataFrame, losers: pd.DataFrame) -> bool:
        """
        Save top gainers and losers to the database.

        Args:
            gainers: DataFrame with top gainers
            losers: DataFrame with top losers

        Returns:
            Success status
        """
        try:
            timestamp = datetime.now().isoformat()

            # Prepare data for batch insertion
            gainers_data = []
            losers_data = []

            # Prepare gainers data
            for rank, (_, row) in enumerate(gainers.iterrows(), 1):
                gainers_data.append((
                    row['symbol'],
                    timestamp,
                    row['price'],
                    row['change_24h'],
                    row.get('volume_24h', None),
                    rank
                ))

            # Prepare losers data
            for rank, (_, row) in enumerate(losers.iterrows(), 1):
                losers_data.append((
                    row['symbol'],
                    timestamp,
                    row['price'],
                    row['change_24h'],
                    row.get('volume_24h', None),
                    rank
                ))

            # Batch insert gainers
            if gainers_data:
                self.db.execute_many(
                    """
                    INSERT OR REPLACE INTO top_gainers
                    (symbol, timestamp, price, change_24h, volume_24h, rank)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    gainers_data
                )

            # Batch insert losers
            if losers_data:
                self.db.execute_many(
                    """
                    INSERT OR REPLACE INTO top_losers
                    (symbol, timestamp, price, change_24h, volume_24h, rank)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    losers_data
                )

            return True
        except Exception as e:
            self.db.logger.error(f"Error saving top movers: {str(e)}")
            return False

    def get_latest_movers(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get the most recent top gainers and losers.

        Returns:
            Tuple of (gainers_df, losers_df)
        """
        try:
            # Get latest timestamp for gainers
            latest_timestamp_result = self.db.execute_query(
                """
                SELECT timestamp FROM top_gainers
                ORDER BY timestamp DESC LIMIT 1
                """
            )

            if not latest_timestamp_result:
                return pd.DataFrame(), pd.DataFrame()

            latest_timestamp = latest_timestamp_result[0][0]

            # Get gainers at that timestamp
            gainers_result = self.db.execute_query(
                """
                SELECT symbol, timestamp, price, change_24h, volume_24h, rank
                FROM top_gainers
                WHERE timestamp = ?
                ORDER BY rank
                """,
                (latest_timestamp,)
            )

            # Get losers at that timestamp
            losers_result = self.db.execute_query(
                """
                SELECT symbol, timestamp, price, change_24h, volume_24h, rank
                FROM top_losers
                WHERE timestamp = ?
                ORDER BY rank
                """,
                (latest_timestamp,)
            )

            # Convert to DataFrames
            gainers_df = pd.DataFrame(
                gainers_result,
                columns=['symbol', 'timestamp', 'price', 'change_24h', 'volume_24h', 'rank']
            )

            losers_df = pd.DataFrame(
                losers_result,
                columns=['symbol', 'timestamp', 'price', 'change_24h', 'volume_24h', 'rank']
            )

            return gainers_df, losers_df
        except Exception as e:
            self.db.logger.error(f"Error getting latest movers: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()

    def get_movers_history(self, days: int = 7) -> Tuple[List[Dict], List[Dict]]:
        """
        Get historical data for top movers.

        Args:
            days: Number of days to look back

        Returns:
            Tuple of (gainers_history, losers_history) where each is a list of timestamp -> dataframe mappings
        """
        try:
            # Get distinct timestamps for gainers within timeframe
            timestamps_result = self.db.execute_query(
                f"""
                SELECT DISTINCT timestamp FROM top_gainers
                WHERE timestamp >= datetime('now', '-{days} days')
                ORDER BY timestamp DESC
                """
            )

            if not timestamps_result:
                return [], []

            gainers_history = []
            losers_history = []

            # For each timestamp, get the gainers and losers
            for (timestamp,) in timestamps_result:
                # Get gainers at that timestamp
                gainers_result = self.db.execute_query(
                    """
                    SELECT symbol, price, change_24h, volume_24h, rank
                    FROM top_gainers
                    WHERE timestamp = ?
                    ORDER BY rank
                    """,
                    (timestamp,)
                )

                # Get losers at that timestamp
                losers_result = self.db.execute_query(
                    """
                    SELECT symbol, price, change_24h, volume_24h, rank
                    FROM top_losers
                    WHERE timestamp = ?
                    ORDER BY rank
                    """,
                    (timestamp,)
                )

                if gainers_result:
                    gainers_df = pd.DataFrame(
                        gainers_result,
                        columns=['symbol', 'price', 'change_24h', 'volume_24h', 'rank']
                    )
                    gainers_history.append({
                        'timestamp': timestamp,
                        'data': gainers_df
                    })

                if losers_result:
                    losers_df = pd.DataFrame(
                        losers_result,
                        columns=['symbol', 'price', 'change_24h', 'volume_24h', 'rank']
                    )
                    losers_history.append({
                        'timestamp': timestamp,
                        'data': losers_df
                    })

            return gainers_history, losers_history
        except Exception as e:
            self.db.logger.error(f"Error getting movers history: {str(e)}")
            return [], []

    def get_persistent_movers(self, min_days: int = 3) -> List[Dict]:
        """
        Get symbols that show persistent directional movement.

        Args:
            min_days: Minimum consecutive days required to qualify

        Returns:
            List of dictionaries containing persistent movers data
        """
        query = """
        WITH daily_trends AS (
            SELECT 
                symbol,
                DATE(timestamp) as date,
                AVG(change_24h) as avg_daily_change,
                AVG(price) as avg_daily_price,
                AVG(volume_24h) as avg_daily_volume,
                CASE WHEN AVG(change_24h) > 0 THEN 'up' ELSE 'down' END as trend
            FROM market_moves
            GROUP BY symbol, DATE(timestamp)
        ),
        trend_runs AS (
            SELECT 
                symbol,
                trend,
                COUNT(*) as consecutive_days,
                MIN(date) as trend_start,
                MAX(date) as trend_end,
                FIRST_VALUE(avg_daily_price) OVER (PARTITION BY symbol ORDER BY date) as start_price,
                LAST_VALUE(avg_daily_price) OVER (PARTITION BY symbol ORDER BY date) as current_price,
                AVG(avg_daily_volume) as avg_volume
            FROM daily_trends
            GROUP BY symbol, trend
            HAVING COUNT(*) >= ?
        )
        SELECT * FROM trend_runs
        ORDER BY consecutive_days DESC, avg_volume DESC
        """

        try:
            cursor = self.db.cursor()
            cursor.execute(query, (min_days,))
            results = cursor.fetchall()

            persistent_movers = []
            for row in results:
                total_change = ((row['current_price'] - row['start_price']) / row['start_price']) * 100
                persistent_movers.append({
                    'symbol': row['symbol'],
                    'trend': row['trend'],
                    'days': row['consecutive_days'],
                    'start_price': row['start_price'],
                    'current_price': row['current_price'],
                    'total_change': total_change,
                    'avg_volume': row['avg_volume']
                })

            return persistent_movers
        except Exception as e:
            self.logger.error(f"Error getting persistent movers: {str(e)}")
            return []

    def get_symbol_appearances(self, symbol: str, days: int = 7) -> Dict[str, Any]:
        """
        Get history of appearances for a specific symbol in top movers.

        Args:
            symbol: Symbol to check
            days: Number of days to look back

        Returns:
            Dictionary with appearance counts and details
        """
        try:
            # Count appearances in gainers
            gainers_count_result = self.db.execute_query(
                f"""
                SELECT COUNT(*) from top_gainers
                WHERE symbol = ?
                AND timestamp >= datetime('now', '-{days} days')
                """,
                (symbol,)
            )

            # Count appearances in losers
            losers_count_result = self.db.execute_query(
                f"""
                SELECT COUNT(*) from top_losers
                WHERE symbol = ?
                AND timestamp >= datetime('now', '-{days} days')
                """,
                (symbol,)
            )

            gainers_count = gainers_count_result[0][0] if gainers_count_result else 0
            losers_count = losers_count_result[0][0] if losers_count_result else 0

            # Get gainers appearances details
            gainers_details = []
            if gainers_count > 0:
                gainers_details_result = self.db.execute_query(
                    f"""
                    SELECT timestamp, price, change_24h, volume_24h, rank
                    FROM top_gainers
                    WHERE symbol = ?
                    AND timestamp >= datetime('now', '-{days} days')
                    ORDER BY timestamp DESC
                    """,
                    (symbol,)
                )

                for row in gainers_details_result:
                    gainers_details.append({
                        'timestamp': row[0],
                        'price': row[1],
                        'change_24h': row[2],
                        'volume_24h': row[3],
                        'rank': row[4]
                    })

            # Get losers appearances details
            losers_details = []
            if losers_count > 0:
                losers_details_result = self.db.execute_query(
                    f"""
                    SELECT timestamp, price, change_24h, volume_24h, rank
                    FROM top_losers
                    WHERE symbol = ?
                    AND timestamp >= datetime('now', '-{days} days')
                    ORDER BY timestamp DESC
                    """,
                    (symbol,)
                )

                for row in losers_details_result:
                    losers_details.append({
                        'timestamp': row[0],
                        'price': row[1],
                        'change_24h': row[2],
                        'volume_24h': row[3],
                        'rank': row[4]
                    })

            return {
                'symbol': symbol,
                'gainers_count': gainers_count,
                'losers_count': losers_count,
                'total_count': gainers_count + losers_count,
                'gainers_details': gainers_details,
                'losers_details': losers_details
            }
        except Exception as e:
            self.db.logger.error(f"Error getting symbol appearances: {str(e)}")
            return {
                'symbol': symbol,
                'gainers_count': 0,
                'losers_count': 0,
                'total_count': 0,
                'gainers_details': [],
                'losers_details': []
            }

    def get_frequent_symbols(self, days: int = 7, min_count: int = 3) -> List[Dict[str, Any]]:
        """
        Get symbols that frequently appear in top gainers or losers.

        Args:
            days: Number of days to look back
            min_count: Minimum number of appearances to include

        Returns:
            List of dictionaries with symbol stats
        """
        try:
            # Get gainers counts
            gainers_counts_result = self.db.execute_query(
                f"""
                SELECT symbol, COUNT(*) as count
                FROM top_gainers
                WHERE timestamp >= datetime('now', '-{days} days')
                GROUP BY symbol
                HAVING count >= ?
                """,
                (min_count,)
            )

            # Get losers counts
            losers_counts_result = self.db.execute_query(
                f"""
                SELECT symbol, COUNT(*) as count
                FROM top_losers
                WHERE timestamp >= datetime('now', '-{days} days')
                GROUP BY symbol
                HAVING count >= ?
                """,
                (min_count,)
            )

            # Combine the results
            symbol_stats = {}

            # Process gainers
            if gainers_counts_result:
                for symbol, count in gainers_counts_result:
                    symbol_stats[symbol] = {
                        'symbol': symbol,
                        'gainers_count': count,
                        'losers_count': 0,
                        'total_count': count
                    }

            # Process losers
            if losers_counts_result:
                for symbol, count in losers_counts_result:
                    if symbol in symbol_stats:
                        symbol_stats[symbol]['losers_count'] = count
                        symbol_stats[symbol]['total_count'] += count
                    else:
                        symbol_stats[symbol] = {
                            'symbol': symbol,
                            'gainers_count': 0,
                            'losers_count': count,
                            'total_count': count
                        }

            # Get latest price data for these symbols
            for symbol in symbol_stats:
                price_data_result = self.db.execute_query(
                    """
                    SELECT price, change_24h, volume_24h, timestamp
                    FROM price_data
                    WHERE symbol = ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                    """,
                    (symbol,)
                )

                if price_data_result:
                    price, change_24h, volume_24h, timestamp = price_data_result[0]
                    symbol_stats[symbol].update({
                        'latest_price': price,
                        'latest_change': change_24h,
                        'latest_volume': volume_24h,
                        'latest_timestamp': timestamp
                    })

            # Convert to list and sort by total count
            result = list(symbol_stats.values())
            result.sort(key=lambda x: x['total_count'], reverse=True)

            return result
        except Exception as e:
            self.db.logger.error(f"Error getting frequent symbols: {str(e)}")
            return []