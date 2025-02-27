import requests
import websocket
import json
import logging
from typing import List, Dict, Any, Optional
from collections import deque
import threading
import time


class OKXClient:
    """Client for interacting with OKX exchange for market data."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize OKX client with optional configuration."""
        # Default configuration with more alternative URLs
        self.base_urls = [
            "https://www.okx.com",
            "https://www.okx.cab",
            "https://aws.okx.com",  # AWS endpoint
            "https://okx.com",  # Root domain
            "https://api.okx.com"  # API specific endpoint
        ]
        self.ws_urls = [
            "wss://ws.okx.com:8443/ws/v5/public",
            "wss://wsaws.okx.com:8443/ws/v5/public",
            "wss://wspap.okx.com:8443/ws/v5/public"  # Asia-Pacific endpoint
        ]

        # Override with provided config if available
        if config:
            self.base_urls = config.get('api', {}).get('okx', {}).get('base_urls', self.base_urls)
            self.ws_urls = config.get('api', {}).get('okx', {}).get('ws_urls', self.ws_urls)

        # Initialize variables
        self.available_symbols = []
        self.prices = {}
        self.ws = None
        self.ws_thread = None
        self.price_queues = {}
        self.running = False
        self.connected = False

        # Configure logging
        self.logger = logging.getLogger('okx_client')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def get_all_symbols(self) -> List[str]:
        """Fetch all available USDT trading pairs from OKX."""
        for base_url in self.base_urls:
            try:
                url = f"{base_url}/api/v5/public/instruments"
                params = {"instType": "SPOT"}

                headers = {
                    'User-Agent': 'Mozilla/5.0',
                    'Accept': 'application/json'
                }

                self.logger.info(f"Attempting to fetch USDT pairs from {base_url}...")
                response = requests.get(url, params=params, headers=headers, timeout=10)

                if response.status_code == 200:
                    data = response.json()

                    if data.get("code") == "0":
                        valid_symbols = []
                        for instrument in data["data"]:
                            symbol = instrument["instId"]
                            if symbol.endswith("-USDT"):
                                valid_symbols.append(symbol)

                        self.logger.info(f"Successfully found {len(valid_symbols)} USDT trading pairs")
                        if valid_symbols:
                            self.logger.info(f"Sample USDT pairs: {valid_symbols[:5]}")

                        self.available_symbols = valid_symbols
                        return valid_symbols
            except Exception as e:
                self.logger.error(f"Error connecting to {base_url}: {str(e)}")
                continue

        # If all URLs fail
        self.logger.error("All OKX API endpoints failed")
        return []

    def get_top_volume_symbols(self, limit: int = 30) -> List[str]:
        """
        Fetch top trading pairs by 24h volume.

        Args:
            limit: Number of top volume symbols to return

        Returns:
            List of symbol strings
        """
        # First get all available symbols
        all_symbols = self.get_all_symbols()

        if not all_symbols:
            self.logger.error("Failed to fetch symbols")
            return []

        # Create a list to store symbols with their volumes
        symbol_volumes = []

        # Get ticker data for each symbol
        self.logger.info(f"Fetching volume data for {len(all_symbols)} symbols...")

        for symbol in all_symbols:
            try:
                ticker = self.get_ticker(symbol)

                # Skip if ticker data is incomplete
                if not ticker or "volume_24h" not in ticker:
                    continue

                volume = ticker["volume_24h"]
                symbol_volumes.append((symbol, volume))

            except Exception as e:
                self.logger.error(f"Error getting volume for {symbol}: {str(e)}")

        # Sort by volume (descending)
        symbol_volumes.sort(key=lambda x: x[1], reverse=True)

        # Get top symbols by volume
        top_symbols = [s[0] for s in symbol_volumes[:limit]]

        self.logger.info(f"Successfully identified top {len(top_symbols)} symbols by volume")

        return top_symbols

    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get current ticker data for a symbol."""
        for base_url in self.base_urls:
            try:
                url = f"{base_url}/api/v5/market/ticker"
                params = {"instId": symbol}

                headers = {
                    'User-Agent': 'Mozilla/5.0',
                    'Accept': 'application/json'
                }

                response = requests.get(url, params=params, headers=headers, timeout=10)

                if response.status_code == 200:
                    data = response.json()

                    if data.get("code") == "0" and data.get("data"):
                        ticker_data = data["data"][0]
                        result = {
                            "symbol": symbol,
                            "last_price": float(ticker_data.get("last", 0)),
                            "bid": float(ticker_data.get("bidPx", 0)),
                            "ask": float(ticker_data.get("askPx", 0)),
                            "timestamp": ticker_data.get("ts", "")
                        }

                        # Handle optional fields that might be missing
                        if "vol24h" in ticker_data:
                            result["volume_24h"] = float(ticker_data["vol24h"])

                        if "change24h" in ticker_data:
                            result["change_24h"] = float(ticker_data["change24h"])

                        return result
            except Exception as e:
                self.logger.error(f"Error fetching ticker for {symbol} from {base_url}: {str(e)}")
                continue

        return {}

    def get_klines(self, symbol: str, interval: str = "1h", limit: int = 100) -> List[Dict[str, Any]]:
        """Get historical klines/candlestick data."""
        for base_url in self.base_urls:
            try:
                # Try with different endpoint pattern
                url = f"{base_url}/api/v5/market/candles"

                # Map intervals to proper OKX format (1H, 4H, 1D, etc.)
                okx_interval = interval.upper()

                # Updated parameters
                params = {
                    "instId": symbol,
                    "bar": okx_interval,
                    "limit": str(limit)  # Convert to string explicitly
                }

                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'application/json'
                }

                self.logger.info(f"Fetching klines from {base_url}...")
                response = requests.get(url, params=params, headers=headers, timeout=15)

                self.logger.info(f"Response status: {response.status_code}")
                if response.status_code == 200:
                    data = response.json()
                    self.logger.info(f"Response code: {data.get('code')}, msg: {data.get('msg', 'None')}")

                    if data.get("code") == "0" and data.get("data"):
                        result = []
                        for candle in data["data"]:
                            try:
                                # OKX candles format: [ts, open, high, low, close, vol, volCcy]
                                result.append({
                                    "timestamp": int(candle[0]),
                                    "open": float(candle[1]),
                                    "high": float(candle[2]),
                                    "low": float(candle[3]),
                                    "close": float(candle[4]),
                                    "volume": float(candle[5]) if len(candle) > 5 else 0
                                })
                            except (IndexError, ValueError) as e:
                                self.logger.error(f"Error parsing candle data: {e}, Data: {candle}")
                                continue

                        # Reverse to get chronological order (oldest to newest)
                        result.reverse()
                        self.logger.info(f"Successfully fetched {len(result)} klines")
                        return result
            except Exception as e:
                self.logger.error(f"Error fetching klines for {symbol} from {base_url}: {str(e)}")
                continue

        self.logger.warning(f"Failed to fetch klines for {symbol} from any endpoint")
        return []

    # Rest of the class remains the same