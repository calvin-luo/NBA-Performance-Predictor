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
    
    def get_game(self, game_id: str) -> Optional[Dict[str, Any]]:
        """Get a game by its ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM games WHERE game_id = ?", (game_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def insert_game(self, game: Dict[str, Any]) -> None:
        """Insert a new game."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO games (game_id, home_team, away_team, game_date, game_time, venue) VALUES (?, ?, ?, ?, ?, ?)",
                (game['game_id'], game['home_team'], game['away_team'], game['game_date'], game['game_time'], game.get('venue', ''))
            )
            conn.commit()

    def update_game(self, game_id: str, game: Dict[str, Any]) -> None:
        """Update an existing game."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE games SET home_team = ?, away_team = ?, game_date = ?, game_time = ?, venue = ? WHERE game_id = ?",
                (game['home_team'], game['away_team'], game['game_date'], game['game_time'], game.get('venue', ''), game_id)
            )
            conn.commit()

    def get_games_by_date(self, date: str) -> List[Dict[str, Any]]:
        """Get all games for a specific date."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM games WHERE game_date = ?", (date,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_lineups_by_date(self, date: str) -> List[Dict[str, Any]]:
        """Get all lineups for games on a specific date."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT p.* FROM players p JOIN games g ON p.game_id = g.game_id WHERE g.game_date = ?",
                (date,)
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        
    def get_team_lineup(self, team_name: str, game_id: str) -> List[Dict[str, Any]]:
        """Get the lineup for a specific team in a specific game."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM players WHERE team = ? AND game_id = ?",
                (team_name, game_id)
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
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