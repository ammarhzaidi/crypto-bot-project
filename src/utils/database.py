"""
Core database management for the crypto trading bot.
Handles connections, transactions, and initialization.
"""

import sqlite3
import logging
import os
from typing import Optional


class DatabaseManager:
    """
    Core database manager that handles connections and initialization.
    This class focuses only on database connection management.
    """

    def __init__(self, db_path: str = "data/market_data.db"):
        """
        Initialize the database manager.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.logger = logging.getLogger('database_manager')

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Initialize the database schema
        self._init_db()

    def get_connection(self) -> sqlite3.Connection:
        """
        Get a database connection.

        Returns:
            SQLite connection object
        """
        try:
            conn = sqlite3.connect(self.db_path)
            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys = ON")
            return conn
        except Exception as e:
            self.logger.error(f"Error connecting to database: {str(e)}")
            raise

    def _init_db(self):
        """Initialize the database schema if it doesn't exist."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Create market snapshots table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                snapshot_type TEXT NOT NULL,
                raw_data TEXT NOT NULL
            )
            ''')

            # Create symbols table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS symbols (
                symbol TEXT PRIMARY KEY,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL
            )
            ''')

            # Create price data table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                price REAL NOT NULL,
                change_24h REAL,
                volume_24h REAL,
                UNIQUE(symbol, timestamp)
            )
            ''')

            # Create gainers and losers tables
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS top_gainers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                price REAL NOT NULL,
                change_24h REAL NOT NULL,
                volume_24h REAL,
                rank INTEGER NOT NULL,
                UNIQUE(symbol, timestamp)
            )
            ''')

            cursor.execute('''
            CREATE TABLE IF NOT EXISTS top_losers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                price REAL NOT NULL,
                change_24h REAL NOT NULL,
                volume_24h REAL,
                rank INTEGER NOT NULL,
                UNIQUE(symbol, timestamp)
            )
            ''')

            # Create persistent movers table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS persistent_movers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                mover_type TEXT NOT NULL,
                price REAL NOT NULL,
                change_24h REAL NOT NULL,
                appearances INTEGER NOT NULL,
                trend_consistency REAL,
                acceleration REAL,
                UNIQUE(symbol, timestamp, mover_type)
            )
            ''')

            conn.commit()
            self.logger.info("Database initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing database: {str(e)}")
            raise
        finally:
            if conn:
                conn.close()

    def execute_query(self, query: str, params: tuple = None) -> Optional[list]:
        """
        Execute a SQL query and return results.

        Args:
            query: SQL query to execute
            params: Query parameters

        Returns:
            List of results or None for queries with no results
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            if query.strip().upper().startswith(("SELECT", "PRAGMA")):
                return cursor.fetchall()
            else:
                conn.commit()
                return None
        except Exception as e:
            if conn:
                conn.rollback()
            self.logger.error(f"Query execution error: {str(e)}")
            raise
        finally:
            if conn:
                conn.close()

    def execute_many(self, query: str, params_list: list) -> None:
        """
        Execute a SQL query with multiple parameter sets.

        Args:
            query: SQL query to execute
            params_list: List of parameter tuples
        """
        if not params_list:
            return

        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            self.logger.error(f"Batch execution error: {str(e)}")
            raise
        finally:
            if conn:
                conn.close()