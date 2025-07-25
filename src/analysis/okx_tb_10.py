import requests
import pandas as pd
import logging
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional


class OKXMarketAnalyzer:
    """
    Analyzes OKX market data to track top gainers and losers.
    Compares data across different time periods to identify acceleration.
    """

    def __init__(self, min_market_cap: float = 50000000, min_volume: float = 1000000):
        """
        Initialize the OKX market analyzer.

        Args:
            min_market_cap: Minimum market cap to consider (default: $50M)
            min_volume: Minimum 24h volume to consider (default: $1M)
        """
        self.base_urls = [
            "https://www.okx.com",
            "https://api.okx.com",
            "https://aws.okx.com"
        ]
        self.min_market_cap = min_market_cap
        self.min_volume = min_volume

        # Store historical data for comparison
        self.historical_data = {}

        # Setup logging
        self.logger = logging.getLogger('okx_market_analyzer')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def fetch_market_data(self) -> Optional[List[Dict]]:
        """
        Fetch market data from OKX.

        Returns:
            List of dictionaries containing market data for all symbols.
        """
        for base_url in self.base_urls:
            try:
                # Use the tickers endpoint which we know works
                url = f"{base_url}/api/v5/market/tickers"
                params = {"instType": "SPOT"}

                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'application/json'
                }

                self.logger.info(f"Fetching market data from {base_url}...")
                response = requests.get(url, params=params, headers=headers, timeout=15)

                if response.status_code == 200:
                    data = response.json()

                    if data.get("code") == "0" and data.get("data"):
                        # Process the data
                        processed_data = []
                        raw_count = len(data["data"])
                        self.logger.info(f"Raw data items received: {raw_count}")

                        usdt_count = 0
                        for item in data["data"]:
                            # Only include USDT pairs
                            if not item["instId"].endswith("-USDT"):
                                continue
                            usdt_count += 1

                            try:
                                # Extract current price
                                last_price = float(item.get("last", 0))

                                # Calculate 24h change using open24h price
                                change_24h = 0
                                if "open24h" in item and item["open24h"] and float(item["open24h"]) > 0:
                                    open_price = float(item["open24h"])
                                    change_24h = ((last_price - open_price) / open_price) * 100

                                # Extract volume - this is base volume, not quote (USDT) volume
                                volume_usd = 0
                                if "volCcy24h" in item:
                                    volume_usd = float(item["volCcy24h"])
                                elif "vol24h" in item:
                                    # vol24h is in base currency, multiply by price for USD value
                                    volume_usd = float(item["vol24h"]) * last_price

                                processed_data.append({
                                    "symbol": item["instId"].replace("-USDT", ""),
                                    "price": last_price,
                                    "change_24h": change_24h,
                                    "volume_24h": volume_usd,
                                    "timestamp": datetime.now().isoformat()
                                })
                            except (ValueError, KeyError) as e:
                                self.logger.error(f"Error processing data for {item.get('instId')}: {str(e)}")
                                continue

                        self.logger.info(
                            f"Found {usdt_count} USDT pairs, processed {len(processed_data)} symbols with valid data")
                        return processed_data
                    else:
                        self.logger.error(f"API returned error: {data.get('msg', 'Unknown error')}")
                else:
                    self.logger.error(f"HTTP error: {response.status_code}")

            except Exception as e:
                self.logger.error(f"Error fetching market data from {base_url}: {str(e)}")
                continue

        # If all URLs fail
        self.logger.error("Failed to fetch market data from any endpoint")
        return None

    def get_top_gainers_losers(self, limit: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get the top gainers and losers based on 24h price change.

        Args:
            limit: Number of top gainers/losers to return (default: 10)

        Returns:
            Tuple of (top_gainers, top_losers) as pandas DataFrames
        """
        data = self.fetch_market_data()

        if not data:
            self.logger.error("No data available to analyze")
            return pd.DataFrame(), pd.DataFrame()

        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(data)
        self.logger.info(f"DataFrame created with {len(df)} rows")

        # Apply volume filter
        filtered_df = df[df['volume_24h'] >= self.min_volume].copy()
        self.logger.info(f"After volume filter: {len(filtered_df)} rows remain")

        # If no data after filtering
        if filtered_df.empty:
            self.logger.warning("No data left after applying filters. Reducing minimum requirements.")
            # Try with lower requirements
            filtered_df = df.copy()
            self.logger.info(f"Using all data without filtering: {len(filtered_df)} rows")

        # Sort by change_24h
        filtered_df.sort_values('change_24h', ascending=False, inplace=True)

        # Enforce the actual limit from the parameter
        top_gainers = filtered_df.head(limit).copy() if len(filtered_df) > 0 else pd.DataFrame()
        top_losers = filtered_df.tail(limit).sort_values('change_24h', ascending=True).copy() if len(
            filtered_df) > 0 else pd.DataFrame()

        self.logger.info(f"Selected {len(top_gainers)} top gainers and {len(top_losers)} top losers")

        return top_gainers, top_losers

    def _cleanup_historical_data(self, hours: int = 24):
        """Remove historical data older than specified hours."""
        current_time = datetime.now()
        keys_to_remove = []

        for timestamp in self.historical_data.keys():
            try:
                data_time = datetime.fromisoformat(timestamp)
                time_diff = (current_time - data_time).total_seconds() / 3600

                if time_diff > hours:
                    keys_to_remove.append(timestamp)
            except ValueError:
                # If timestamp is not a valid ISO format
                keys_to_remove.append(timestamp)

        for key in keys_to_remove:
            del self.historical_data[key]

    def compare_with_previous(self, hours_ago: float = 4) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compare current top gainers/losers with previous period.
        """
        # Get fresh current data from OKX
        current_gainers, current_losers = self.get_top_gainers_losers()

        # Get previous data from database
        db_path = "data/market_data.db"
        from src.repositories.movers_repository import MoversRepository
        from src.utils.database import DatabaseManager

        db = DatabaseManager(db_path)
        movers_repo = MoversRepository(db)

        # Get latest stored data from database
        previous_gainers, previous_losers = movers_repo.get_latest_movers()

        # Process gainers comparison
        gainers_comparison = pd.DataFrame()
        if not current_gainers.empty:
            gainers_comparison = current_gainers[['symbol', 'price', 'change_24h', 'volume_24h']].copy()
            gainers_comparison = gainers_comparison.rename(columns={'change_24h': 'change_24h_current'})

            if not previous_gainers.empty:
                gainers_comparison = gainers_comparison.merge(
                    previous_gainers[['symbol', 'change_24h']],
                    on='symbol',
                    how='left'
                )
                gainers_comparison = gainers_comparison.rename(columns={'change_24h': 'change_24h_previous'})
            else:
                gainers_comparison['change_24h_previous'] = 'New'

        # Process losers comparison
        losers_comparison = pd.DataFrame()
        if not current_losers.empty:
            losers_comparison = current_losers[['symbol', 'price', 'change_24h', 'volume_24h']].copy()
            losers_comparison = losers_comparison.rename(columns={'change_24h': 'change_24h_current'})

            if not previous_losers.empty:
                losers_comparison = losers_comparison.merge(
                    previous_losers[['symbol', 'change_24h']],
                    on='symbol',
                    how='left'
                )
                losers_comparison = losers_comparison.rename(columns={'change_24h': 'change_24h_previous'})
            else:
                losers_comparison['change_24h_previous'] = 'New'

        # Calculate acceleration for both gainers and losers
        for comparison in [gainers_comparison, losers_comparison]:
            if not comparison.empty:
                comparison['is_new'] = comparison['change_24h_previous'] == 'New'
                comparison.loc[~comparison['is_new'], 'acceleration'] = \
                    comparison.loc[~comparison['is_new'], 'change_24h_current'] - \
                    comparison.loc[~comparison['is_new'], 'change_24h_previous']
                comparison.loc[comparison['is_new'], 'acceleration'] = 'New'

        # Store current data in database for next comparison
        movers_repo.save_top_movers(current_gainers, current_losers)

        return gainers_comparison, losers_comparison

    def _create_default_comparison(self, df):
        """Create a default comparison dataframe with default values."""
        if df.empty:
            return pd.DataFrame()

        result = df[['symbol', 'price', 'change_24h', 'volume_24h']].copy()
        result['change_24h_current'] = result['change_24h']
        result['change_24h_previous'] = result['change_24h']
        result['acceleration'] = "New"
        result['timestamp'] = None
        result.drop('change_24h', axis=1, inplace=True, errors='ignore')

        return result

    def _update_historical_data(self, gainers_df, losers_df):
        """
        Update historical data with current values for future comparison.

        Args:
            gainers_df: DataFrame of current gainers
            losers_df: DataFrame of current losers
        """
        # Record current timestamp
        current_time = datetime.now().isoformat()

        # Prepare dataframe with combined data
        combined_data = []

        # Process gainers
        if not gainers_df.empty:
            for _, row in gainers_df.iterrows():
                combined_data.append({
                    'symbol': row['symbol'],
                    'change_24h': row['change_24h'],
                    'timestamp': current_time
                })

        # Process losers
        if not losers_df.empty:
            for _, row in losers_df.iterrows():
                combined_data.append({
                    'symbol': row['symbol'],
                    'change_24h': row['change_24h'],
                    'timestamp': current_time
                })

        # Convert to DataFrame
        if combined_data:
            historical_df = pd.DataFrame(combined_data)

            # Store in historical data dictionary
            self.historical_data[current_time] = historical_df

            # Clean up old records to prevent memory issues
            self._cleanup_historical_data(hours=24)

    def print_top_tables(self):
        """Print tables of top gainers and losers to console."""
        top_gainers, top_losers = self.get_top_gainers_losers()

        # Print timestamp
        print(f"\n=== OKX Market Analysis at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")

        # Print top gainers
        print("\n🔼 TOP 10 GAINERS (24h) 🔼")
        print("=" * 65)
        print(f"{'Symbol':<10} | {'Price':<10} | {'Change 24h':<12} | {'Volume 24h (USDT)':<20}")
        print("-" * 65)

        for _, row in top_gainers.iterrows():
            print(
                f"{row['symbol']:<10} | ${row['price']:<9.4f} | {row['change_24h']:+<11.2f}% | ${row['volume_24h'] / 1000000:<19.2f}M")

        # Print top losers
        print("\n🔽 TOP 10 LOSERS (24h) 🔽")
        print("=" * 65)
        print(f"{'Symbol':<10} | {'Price':<10} | {'Change 24h':<12} | {'Volume 24h (USDT)':<20}")
        print("-" * 65)

        for _, row in top_losers.iterrows():
            print(
                f"{row['symbol']:<10} | ${row['price']:<9.4f} | {row['change_24h']:+<11.2f}% | ${row['volume_24h'] / 1000000:<19.2f}M")

    def print_comparison(self, hours_ago: float = 4):
        """Print comparison between current and previous data."""
        gainers_comparison, losers_comparison = self.compare_with_previous(hours_ago)

        if gainers_comparison.empty and losers_comparison.empty:
            print(f"\nNo comparison data available for {hours_ago} hours ago")
            return

        # Print timestamp
        print(f"\n=== Comparison with {hours_ago} hours ago ===")

        # Print gainers comparison
        print("\n🚀 TOP GAINERS MOVEMENT 🚀")
        print("=" * 85)
        print(
            f"{'Symbol':<10} | {'Price':<10} | {'Current %':<10} | {'Previous %':<10} | {'Acceleration':<12} | {'Volume 24h':<15}")
        print("-" * 85)

        # Sort by acceleration
        gainers_comparison = gainers_comparison.sort_values('acceleration', ascending=False)

        for _, row in gainers_comparison.iterrows():
            print(f"{row['symbol']:<10} | ${row['price']:<9.4f} | {row['change_24h_current']:+<9.2f}% | "
                  f"{row['change_24h_previous']:+<9.2f}% | {row['acceleration']:+<11.2f}% | "
                  f"${row['volume_24h'] / 1000000:<14.2f}M")

        # Print losers comparison
        print("\n📉 TOP LOSERS MOVEMENT 📉")
        print("=" * 85)
        print(
            f"{'Symbol':<10} | {'Price':<10} | {'Current %':<10} | {'Previous %':<10} | {'Acceleration':<12} | {'Volume 24h':<15}")
        print("-" * 85)

        # Sort by acceleration (most negative first for losers)
        losers_comparison = losers_comparison.sort_values('acceleration', ascending=True)

        for _, row in losers_comparison.iterrows():
            print(f"{row['symbol']:<10} | ${row['price']:<9.4f} | {row['change_24h_current']:+<9.2f}% | "
                  f"{row['change_24h_previous']:+<9.2f}% | {row['acceleration']:+<11.2f}% | "
                  f"${row['volume_24h'] / 1000000:<14.2f}M")


if __name__ == "__main__":
    # Demo usage
    analyzer = OKXMarketAnalyzer(min_volume=1000000)  # $1M volume minimum

    print("Fetching current market data...")
    analyzer.print_top_tables()

    # To simulate historical data for the demo, we'll fetch now and wait
    print("\nWaiting for 30 seconds to simulate time passing...")
    time.sleep(30)

    # Fetch again and compare
    print("\nFetching updated market data and comparing...")
    analyzer.print_comparison(hours_ago=0.008)  # 30 seconds in hours