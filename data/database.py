import os
import sqlite3
import datetime
import logging
from contextlib import contextmanager
from typing import Dict, List, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('database')

class Database:
    """
    Minimal SQLite database manager for NBA Stats Predictor.
    Only stores today's games and lineups.
    """
    
    def __init__(self, db_path: str = "nba_stats.db"):
        """Initialize the database manager."""
        # For in-memory database
        if db_path == ":memory:":
            self.db_path = db_path
        else:
            # Ensure the data directory exists
            db_dir = os.path.dirname(os.path.abspath(db_path))
            os.makedirs(db_dir, exist_ok=True)
            self.db_path = db_path
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            yield conn
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()
    
    def initialize_database(self) -> None:
        """Create minimal tables for storing today's games and lineups."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create Games table (minimal version)
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS games (
                game_id TEXT PRIMARY KEY,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                game_date TEXT NOT NULL,
                game_time TEXT NOT NULL,
                venue TEXT
            )
            ''')
            
            # Create Players table (minimal version)
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS players (
                name TEXT NOT NULL,
                team TEXT NOT NULL,
                position TEXT,
                status TEXT DEFAULT 'active',
                game_id TEXT,
                FOREIGN KEY (game_id) REFERENCES games(game_id)
            )
            ''')
            
            conn.commit()
            logger.info("Database tables initialized successfully")