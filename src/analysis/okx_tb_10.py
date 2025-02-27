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
                # Try markets/price-overview endpoint
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
                        for item in data["data"]:
                            # Only include USDT pairs
                            if not item["instId"].endswith("-USDT"):
                                continue

                            try:
                                change_24h = float(item.get("change24h", "0").replace("%", ""))

                                # Skip pairs without change data
                                if change_24h == 0 and "change24h" not in item:
                                    continue

                                # Extract volume
                                volume_24h = float(item.get("vol24h", 0))

                                # Extract current price
                                last_price = float(item.get("last", 0))

                                # We don't have market cap in the API response, so we'll fetch it separately
                                # or use volume as a proxy for now

                                processed_data.append({
                                    "symbol": item["instId"].replace("-USDT", ""),
                                    "price": last_price,
                                    "change_24h": change_24h,
                                    "volume_24h": volume_24h,
                                    "timestamp": datetime.now().isoformat()
                                })
                            except (ValueError, KeyError) as e:
                                self.logger.error(f"Error processing data for {item.get('instId')}: {str(e)}")
                                continue

                        self.logger.info(f"Successfully fetched data for {len(processed_data)} symbols")
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
            limit: Number of top gainers/losers to return

        Returns:
            Tuple of (top_gainers, top_losers) as pandas DataFrames
        """
        data = self.fetch_market_data()

        if not data:
            self.logger.error("No data available to analyze")
            return pd.DataFrame(), pd.DataFrame()

        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(data)

        # Apply filters if we have market cap data
        # Since we don't have it from the API, we'll just use volume for now
        filtered_df = df[df['volume_24h'] >= self.min_volume].copy()

        # If no data after filtering
        if filtered_df.empty:
            self.logger.warning("No data left after applying filters. Reducing minimum requirements.")
            # Try with lower requirements
            filtered_df = df.copy()

        # Sort by change_24h
        filtered_df.sort_values('change_24h', ascending=False, inplace=True)

        # Get top gainers and losers
        top_gainers = filtered_df.head(limit).copy()
        top_losers = filtered_df.tail(limit).sort_values('change_24h', ascending=True).copy()

        # Store the data with timestamp for later comparison
        timestamp = datetime.now().isoformat()

        # Store historical data for comparison
        self.historical_data[timestamp] = filtered_df

        # Clean up old data (keep only last 24 hours)
        self._cleanup_historical_data()

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

        Args:
            hours_ago: Compare with data from this many hours ago

        Returns:
            Tuple of (gainers_comparison, losers_comparison) as pandas DataFrames
        """
        current_gainers, current_losers = self.get_top_gainers_losers()

        # Find closest historical data point
        closest_timestamp = None
        closest_time_diff = float('inf')

        current_time = datetime.now()
        for timestamp in self.historical_data.keys():
            try:
                data_time = datetime.fromisoformat(timestamp)
                time_diff = abs((current_time - data_time).total_seconds() / 3600 - hours_ago)

                if time_diff < closest_time_diff:
                    closest_time_diff = time_diff
                    closest_timestamp = timestamp
            except ValueError:
                continue

        if not closest_timestamp or closest_time_diff > hours_ago * 0.5:  # If no data within 50% of requested time
            self.logger.warning(f"No historical data found from approximately {hours_ago} hours ago")
            return current_gainers, current_losers

        # Get historical data
        historical_df = self.historical_data[closest_timestamp]

        # Merge current and historical data
        gainers_symbols = current_gainers['symbol'].tolist()
        losers_symbols = current_losers['symbol'].tolist()

        # Extract historical data for current gainers/losers
        historical_gainers = historical_df[historical_df['symbol'].isin(gainers_symbols)]
        historical_losers = historical_df[historical_df['symbol'].isin(losers_symbols)]

        # Merge dataframes
        gainers_comparison = pd.merge(
            current_gainers[['symbol', 'price', 'change_24h', 'volume_24h']],
            historical_gainers[['symbol', 'change_24h']],
            on='symbol',
            suffixes=('_current', '_previous')
        )

        losers_comparison = pd.merge(
            current_losers[['symbol', 'price', 'change_24h', 'volume_24h']],
            historical_losers[['symbol', 'change_24h']],
            on='symbol',
            suffixes=('_current', '_previous')
        )

        # Calculate acceleration
        gainers_comparison['acceleration'] = gainers_comparison['change_24h_current'] - gainers_comparison[
            'change_24h_previous']
        losers_comparison['acceleration'] = losers_comparison['change_24h_current'] - losers_comparison[
            'change_24h_previous']

        return gainers_comparison, losers_comparison

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