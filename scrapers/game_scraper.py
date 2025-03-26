import os
import re
import time
import logging
import datetime
import requests
from typing import List, Dict, Any, Optional, Tuple

from nba_api.live.nba.endpoints import scoreboard
from data.database import Database

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('scrapers.game_scraper')

# NBA team name mappings (official team name to abbreviation and common variants)
NBA_TEAMS = {
    "Atlanta Hawks": {"abbr": "ATL", "variants": ["hawks", "atlanta"]},
    "Boston Celtics": {"abbr": "BOS", "variants": ["celtics", "boston"]},
    "Brooklyn Nets": {"abbr": "BKN", "variants": ["nets", "brooklyn"]},
    "Charlotte Hornets": {"abbr": "CHA", "variants": ["hornets", "charlotte"]},
    "Chicago Bulls": {"abbr": "CHI", "variants": ["bulls", "chicago"]},
    "Cleveland Cavaliers": {"abbr": "CLE", "variants": ["cavaliers", "cavs", "cleveland"]},
    "Dallas Mavericks": {"abbr": "DAL", "variants": ["mavericks", "mavs", "dallas"]},
    "Denver Nuggets": {"abbr": "DEN", "variants": ["nuggets", "denver"]},
    "Detroit Pistons": {"abbr": "DET", "variants": ["pistons", "detroit"]},
    "Golden State Warriors": {"abbr": "GSW", "variants": ["warriors", "golden state", "golden"]},
    "Houston Rockets": {"abbr": "HOU", "variants": ["rockets", "houston"]},
    "Indiana Pacers": {"abbr": "IND", "variants": ["pacers", "indiana"]},
    "Los Angeles Clippers": {"abbr": "LAC", "variants": ["clippers", "la clippers"]},
    "Los Angeles Lakers": {"abbr": "LAL", "variants": ["lakers", "la lakers"]},
    "Memphis Grizzlies": {"abbr": "MEM", "variants": ["grizzlies", "memphis"]},
    "Miami Heat": {"abbr": "MIA", "variants": ["heat", "miami"]},
    "Milwaukee Bucks": {"abbr": "MIL", "variants": ["bucks", "milwaukee"]},
    "Minnesota Timberwolves": {"abbr": "MIN", "variants": ["timberwolves", "wolves", "minnesota"]},
    "New Orleans Pelicans": {"abbr": "NOP", "variants": ["pelicans", "new orleans"]},
    "New York Knicks": {"abbr": "NYK", "variants": ["knicks", "new york"]},
    "Oklahoma City Thunder": {"abbr": "OKC", "variants": ["thunder", "oklahoma city", "oklahoma"]},
    "Orlando Magic": {"abbr": "ORL", "variants": ["magic", "orlando"]},
    "Philadelphia 76ers": {"abbr": "PHI", "variants": ["76ers", "sixers", "philadelphia"]},
    "Phoenix Suns": {"abbr": "PHX", "variants": ["suns", "phoenix"]},
    "Portland Trail Blazers": {"abbr": "POR", "variants": ["trail blazers", "blazers", "portland"]},
    "Sacramento Kings": {"abbr": "SAC", "variants": ["kings", "sacramento"]},
    "San Antonio Spurs": {"abbr": "SAS", "variants": ["spurs", "san antonio"]},
    "Toronto Raptors": {"abbr": "TOR", "variants": ["raptors", "toronto"]},
    "Utah Jazz": {"abbr": "UTA", "variants": ["jazz", "utah"]},
    "Washington Wizards": {"abbr": "WAS", "variants": ["wizards", "washington"]}
}

# Create reverse mappings
TEAM_ABBR_TO_NAME = {team_data["abbr"]: team_name for team_name, team_data in NBA_TEAMS.items()}


class NBAApiScraper:
    """
    Scrapes NBA game schedules using the nba_api package.
    Focused solely on retrieving today's game schedule.
    """
    
    def __init__(self, db=None):
        """
        Initialize the NBA API scraper.
        
        Args:
            db: Database instance (optional, creates a new one if None)
        """
        # Initialize database (create a new one if not provided)
        self.db = db if db else Database()
    
    def _get_team_name_from_abbreviation(self, abbr):
        """
        Get the full team name from a team abbreviation.
        
        Args:
            abbr: Team abbreviation (e.g., "NYK")
            
        Returns:
            Full team name or the original abbreviation if not found
        """
        # First check the official NBA abbreviations
        if abbr.upper() in TEAM_ABBR_TO_NAME:
            return TEAM_ABBR_TO_NAME[abbr.upper()]
        
        return abbr
    
    def _normalize_team_name(self, team_name):
        """
        Make sure we have a consistent team name format.
        
        Args:
            team_name: Team name or abbreviation
            
        Returns:
            Normalized team name
        """
        # If it's an abbreviation, convert to full name
        if len(team_name) <= 3:
            return self._get_team_name_from_abbreviation(team_name)
        
        # Try to match with known team names
        for full_name in NBA_TEAMS.keys():
            if team_name.lower() in full_name.lower():
                return full_name
        
        return team_name
    
    def _parse_time(self, time_str):
        """
        Parse a time string into a consistent 24-hour format.
        
        Args:
            time_str: Time string (e.g., "19:30:00Z")
            
        Returns:
            Time in HH:MM format
        """
        try:
            # Extract the time part
            time_parts = time_str.split(':')
            return f"{time_parts[0]}:{time_parts[1]}"
        except Exception as e:
            logger.error(f"Error parsing time string '{time_str}': {str(e)}")
            return "19:00"  # Default to 7 PM as fallback
    
    def _generate_game_id(self, home_team, away_team, game_date):
        """
        Generate a unique game ID.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            game_date: Game date in YYYY-MM-DD format
            
        Returns:
            Unique game ID string
        """
        # Create a consistent game ID format: DATE_AWAYTEAM_HOMETEAM
        # Use team abbreviations for brevity
        home_abbr = NBA_TEAMS.get(home_team, {}).get("abbr", "UNK")
        away_abbr = NBA_TEAMS.get(away_team, {}).get("abbr", "UNK")
        date_no_sep = game_date.replace("-", "")
        
        return f"{date_no_sep}_{away_abbr}_{home_abbr}"
    
    def scrape_nba_schedule(self, days_ahead=0):
        """
        Scrape NBA game schedule using the nba_api.
        
        Args:
            days_ahead: Not used with NBA API as it returns today's games
            
        Returns:
            List of dictionaries containing game information
        """
        try:
            logger.info("Scraping NBA schedule using nba_api")
            
            # Get today's games
            games_today = scoreboard.ScoreBoard()
            games_dict = games_today.get_dict()
            games = games_dict.get('scoreboard', {}).get('games', [])
            
            # Format the game data for our database
            formatted_games = []
            for game in games:
                game_id = game.get('gameId')
                home_team_data = game.get('homeTeam', {})
                away_team_data = game.get('awayTeam', {})
                
                # Get team names
                home_team = self._normalize_team_name(home_team_data.get('teamCity', '') + " " + home_team_data.get('teamName', ''))
                away_team = self._normalize_team_name(away_team_data.get('teamCity', '') + " " + away_team_data.get('teamName', ''))
                
                # Create a datetime string in the format YYYY-MM-DD
                game_date = game.get('gameTimeUTC', '').split('T')[0]
                
                # Get game time in HH:MM format
                try:
                    game_time_parts = game.get('gameTimeUTC', '').split('T')[1].split(':')
                    game_time = f"{game_time_parts[0]}:{game_time_parts[1]}"
                except (IndexError, ValueError):
                    game_time = "19:00"  # Default to 7 PM as fallback
                
                # Get venue information
                venue = game.get('arena', {}).get('arenaName', 'Unknown')
                
                # Generate a custom game ID using your format
                custom_game_id = self._generate_game_id(home_team, away_team, game_date)
                
                # Format the game information for the database
                formatted_game = {
                    'game_id': custom_game_id,
                    'home_team': home_team,
                    'away_team': away_team,
                    'game_date': game_date,
                    'game_time': game_time,
                    'venue': venue
                }
                
                formatted_games.append(formatted_game)
                logger.info(f"Found game: {away_team} @ {home_team} on {game_date} at {game_time}")
            
            logger.info(f"Scraped {len(formatted_games)} games from NBA API")
            return formatted_games
            
        except Exception as e:
            logger.error(f"Error scraping NBA schedule: {str(e)}")
            return []
    
    def save_games_to_database(self, games):
        """
        Save scraped games to the database.
        
        Args:
            games: List of game dictionaries
            
        Returns:
            Number of games saved
        """
        saved_count = 0
        
        for game in games:
            try:
                # Check if game already exists
                existing_game = self.db.get_game(game['game_id'])
                
                if existing_game:
                    # Update existing game
                    self.db.update_game(game['game_id'], game)
                    logger.info(f"Updated game: {game['game_id']}")
                else:
                    # Insert new game
                    self.db.insert_game(game)
                    logger.info(f"Inserted game: {game['game_id']}")
                
                saved_count += 1
                
            except Exception as e:
                logger.error(f"Error saving game {game.get('game_id', 'unknown')}: {str(e)}")
                continue
        
        logger.info(f"Saved {saved_count} games to database")
        return saved_count
    
    def scrape_and_save_games(self, days_ahead=0):
        """
        Scrape games from NBA API and save them to the database.
        
        Args:
            days_ahead: Not used with NBA API as it returns today's games
            
        Returns:
            Number of games saved
        """
        # Initialize database if not already connected
        self.db.initialize_database()
        
        # Scrape from NBA API
        games = self.scrape_nba_schedule(days_ahead)
        
        # Save games to database
        saved_count = self.save_games_to_database(games)
        
        return saved_count


# Example usage
if __name__ == "__main__":
    # Set up scraper
    scraper = NBAApiScraper()
    
    # Scrape and save games for today (same-day prediction)
    num_games = scraper.scrape_and_save_games(days_ahead=0)
    
    print(f"Scraped and saved {num_games} games for today")