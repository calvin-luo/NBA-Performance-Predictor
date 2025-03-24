import os
import re
import time
import logging
import datetime
import requests
import pandas as pd
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional, Tuple, Set

from nba_api.live.nba.endpoints import scoreboard
from data.database import Database

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('scrapers.nba_api')

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

# Team abbreviation mapping for Rotowire
ROTOWIRE_ABBR_TO_NAME = {
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GS": "Golden State Warriors",  # Note: Rotowire uses GS instead of GSW
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NO": "New Orleans Pelicans",   # Note: Rotowire uses NO instead of NOP
    "NY": "New York Knicks",        # Note: Rotowire uses NY instead of NYK
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SA": "San Antonio Spurs",      # Note: Rotowire uses SA instead of SAS
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards"
}

# Player status categories
PLAYER_STATUSES = {
    'active': ['active', 'available', 'probable', 'expected', 'starting'],
    'questionable': ['questionable', 'game-time decision', 'day-to-day'],
    'out': ['out', 'inactive', 'injured', 'suspension', 'not with team'],
    'unknown': ['unknown']
}


class NBAApiScraper:
    """
    Scrapes NBA game schedules using the nba_api package and lineups from Rotowire.
    Replaces both NBAGamesScraper and NBAPlayersScraper classes.
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
        
        # Then check Rotowire's abbreviations
        if abbr.upper() in ROTOWIRE_ABBR_TO_NAME:
            return ROTOWIRE_ABBR_TO_NAME[abbr.upper()]
        
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
    
    def _normalize_player_status(self, status_text):
        """
        Normalize player status to a standard format.
        
        Args:
            status_text: Player status text
            
        Returns:
            Normalized status
        """
        if not status_text:
            return 'active'  # Default to active for starting lineups
        
        status_lower = status_text.lower().strip()
        
        # Match to standard status categories
        for status_category, keywords in PLAYER_STATUSES.items():
            if any(keyword in status_lower for keyword in keywords):
                return status_category
        
        # Default to active for players in starting lineups
        return 'active'
    
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
    
    def scrape_todays_lineups(self):
        """
        Scrape NBA lineups from Rotowire.com.
        
        Returns:
            List of dictionaries containing player information
        """
        logger.info("Scraping today's NBA lineups from Rotowire.com")
        
        url = "https://www.rotowire.com/basketball/nba-lineups.php"
        
        # Set up headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5'
        }
        
        try:
            # Make the request to Rotowire
            logger.info(f"Sending request to {url}")
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            # Parse the HTML content with BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all lineup containers 
            lineup_containers = soup.find_all(class_='lineup')
            
            if not lineup_containers:
                logger.warning("No lineup containers found. The page structure might have changed.")
                return []
            
            logger.info(f"Found {len(lineup_containers)} lineup containers")
            
            # List to store all players
            all_players = []
            
            # Process each lineup container (one per game)
            for container in lineup_containers:
                try:
                    # Get team abbreviations
                    team_abbrs = container.find_all(class_='lineup__abbr')
                    if len(team_abbrs) < 2:
                        logger.warning("Could not find team abbreviations in a container")
                        continue
                    
                    away_abbr = team_abbrs[0].text.strip()
                    home_abbr = team_abbrs[1].text.strip()
                    
                    # Convert abbreviations to full team names
                    away_team = self._get_team_name_from_abbreviation(away_abbr)
                    home_team = self._get_team_name_from_abbreviation(home_abbr)
                    
                    logger.info(f"Teams in matchup: {away_team} @ {home_team}")
                    
                    # Find all lineup boxes (one for each team)
                    lineup_boxes = container.find_all(class_='lineup__box')
                    
                    if len(lineup_boxes) < 2:
                        logger.warning(f"Could not find lineup boxes for {away_team} @ {home_team}")
                        continue
                    
                    # Process away team (first box)
                    away_box = lineup_boxes[0]
                    self._process_lineup_box(away_box, away_team, all_players)
                    
                    # Process home team (second box)
                    home_box = lineup_boxes[1]
                    self._process_lineup_box(home_box, home_team, all_players)
                    
                except Exception as e:
                    logger.error(f"Error processing lineup container: {str(e)}")
                    continue
            
            logger.info(f"Scraped {len(all_players)} players from Rotowire")
            return all_players
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making request to Rotowire: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return []
    
    def _process_lineup_box(self, lineup_box, team_name, all_players):
        """
        Process a lineup box to extract player information.
        
        Args:
            lineup_box: BeautifulSoup element containing the lineup box
            team_name: Team name
            all_players: List to append player data to
        """
        # Find all starting players (100% chance to play)
        starters = lineup_box.find_all(class_='is-pct-play-100')
        
        # Extract player information for starters
        for player_elem in starters:
            try:
                # Get player position
                position_elem = player_elem.find('div')
                position = position_elem.text.strip() if position_elem else ""
                
                # Get player name
                name_elem = player_elem.find('a')
                player_name = name_elem.get('title', '').strip() if name_elem else "Unknown"
                
                # Create player data for starters
                player_data = {
                    'name': player_name,
                    'team': team_name,
                    'position': position,
                    'status': 'active',
                    'status_detail': 'Starting'
                }
                
                all_players.append(player_data)
                logger.info(f"Added starter: {player_name} ({position}) - {team_name}")
            except Exception as e:
                logger.error(f"Error processing starter element: {str(e)}")
        
        # Find all questionable players
        questionable_players = lineup_box.find_all(class_=lambda c: c and 'is-pct-play-' in c and 'is-pct-play-100' not in c and 'is-pct-play-0' not in c)
        
        # Extract player information for questionable players
        for player_elem in questionable_players:
            try:
                # Get playing percentage
                play_pct_class = [c for c in player_elem.get('class', []) if 'is-pct-play-' in c][0]
                play_pct = play_pct_class.replace('is-pct-play-', '')
                
                # Get player position
                position_elem = player_elem.find('div')
                position = position_elem.text.strip() if position_elem else ""
                
                # Get player name
                name_elem = player_elem.find('a')
                player_name = name_elem.get('title', '').strip() if name_elem else "Unknown"
                
                # Create player data for questionable players
                player_data = {
                    'name': player_name,
                    'team': team_name,
                    'position': position,
                    'status': 'questionable',
                    'status_detail': f"Questionable ({play_pct}% chance to play)"
                }
                
                all_players.append(player_data)
                logger.info(f"Added questionable player: {player_name} ({position}) - {team_name}")
            except Exception as e:
                logger.error(f"Error processing questionable player element: {str(e)}")
        
        # Find all out players
        out_players = lineup_box.find_all(class_='is-pct-play-0')
        
        # Extract player information for out players
        for player_elem in out_players:
            try:
                # Get player position
                position_elem = player_elem.find('div')
                position = position_elem.text.strip() if position_elem else ""
                
                # Get player name
                name_elem = player_elem.find('a')
                player_name = name_elem.get('title', '').strip() if name_elem else "Unknown"
                
                # Get injury details
                injury_elem = player_elem.find(class_='lineup__injury')
                injury_detail = injury_elem.text.strip() if injury_elem else "Out"
                
                # Create player data for out players
                player_data = {
                    'name': player_name,
                    'team': team_name,
                    'position': position,
                    'status': 'out',
                    'status_detail': injury_detail
                }
                
                all_players.append(player_data)
                logger.info(f"Added out player: {player_name} ({position}) - {team_name} - {injury_detail}")
            except Exception as e:
                logger.error(f"Error processing out player element: {str(e)}")
    
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
    
    def save_players_to_database(self, players):
        """
        Save scraped players to the database.
        
        Args:
            players: List of player dictionaries
            
        Returns:
            Number of players saved
        """
        saved_count = 0
        
        for player in players:
            try:
                # Create player data for database
                player_data = {
                    'name': player['name'],
                    'team': player['team'],
                    'status': player['status']
                }
                
                # Insert player
                self.db.insert_player(player_data)
                saved_count += 1
                
            except Exception as e:
                logger.error(f"Error saving player {player.get('name', 'unknown')}: {str(e)}")
                continue
        
        logger.info(f"Saved {saved_count} players to database")
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
    
    def scrape_and_save_players(self):
        """
        Scrape players from Rotowire and save them to the database.
        
        Returns:
            Number of players saved
        """
        # Initialize database if not already connected
        self.db.initialize_database()
        
        # Scrape from Rotowire
        players = self.scrape_todays_lineups()
        
        # Save players to database
        saved_count = self.save_players_to_database(players)
        
        return saved_count
    
    # Methods from NBAPlayersScraper that should be maintained
    def get_team_active_players(self, team_name):
        """
        Get active players for a specific team.
        
        Args:
            team_name: Team name
            
        Returns:
            List of dictionaries containing active player information
        """
        try:
            # Get all players for the team
            all_players = self.db.get_players_by_team(team_name)
            
            # Filter for active players only
            active_players = [p for p in all_players if p['status'] == 'active']
            
            return active_players
        except Exception as e:
            logger.error(f"Error getting active players for team {team_name}: {str(e)}")
            return []
    
    def get_today_game_players(self, game_id):
        """
        Get active players for both teams in a game.
        
        Args:
            game_id: Game ID
            
        Returns:
            Dictionary mapping 'home_players' and 'away_players' to lists of player dictionaries
        """
        try:
            # Get game details
            game_details = self.db.get_game(game_id)
            if not game_details:
                logger.error(f"Game not found: {game_id}")
                return {'home_players': [], 'away_players': []}
            
            # Get players for each team
            home_players = self.get_team_active_players(game_details['home_team'])
            away_players = self.get_team_active_players(game_details['away_team'])
            
            return {
                'home_players': home_players,
                'away_players': away_players
            }
        except Exception as e:
            logger.error(f"Error getting players for game {game_id}: {str(e)}")
            return {'home_players': [], 'away_players': []}


# Example usage
if __name__ == "__main__":
    # Set up scraper
    scraper = NBAApiScraper()
    
    # Scrape and save games for today (same-day prediction)
    num_games = scraper.scrape_and_save_games(days_ahead=0)
    
    print(f"Scraped and saved {num_games} games for today")
    
    # Scrape and save player information
    players_saved = scraper.scrape_and_save_players()
    print(f"Saved {players_saved} players from today's lineups")