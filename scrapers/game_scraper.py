import logging
import datetime
import pytz

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

# Team home venue mapping (fallback for when API doesn't provide venue)
TEAM_HOME_VENUES = {
    "Atlanta Hawks": "State Farm Arena",
    "Boston Celtics": "TD Garden",
    "Brooklyn Nets": "Barclays Center",
    "Charlotte Hornets": "Spectrum Center",
    "Chicago Bulls": "United Center",
    "Cleveland Cavaliers": "Rocket Mortgage FieldHouse",
    "Dallas Mavericks": "American Airlines Center",
    "Denver Nuggets": "Ball Arena",
    "Detroit Pistons": "Little Caesars Arena",
    "Golden State Warriors": "Chase Center",
    "Houston Rockets": "Toyota Center",
    "Indiana Pacers": "Gainbridge Fieldhouse",
    "Los Angeles Clippers": "Crypto.com Arena",
    "Los Angeles Lakers": "Crypto.com Arena",
    "Memphis Grizzlies": "FedExForum",
    "Miami Heat": "Kaseya Center",
    "Milwaukee Bucks": "Fiserv Forum",
    "Minnesota Timberwolves": "Target Center",
    "New Orleans Pelicans": "Smoothie King Center",
    "New York Knicks": "Madison Square Garden",
    "Oklahoma City Thunder": "Paycom Center",
    "Orlando Magic": "Kia Center",
    "Philadelphia 76ers": "Wells Fargo Center",
    "Phoenix Suns": "Footprint Center",
    "Portland Trail Blazers": "Moda Center",
    "Sacramento Kings": "Golden 1 Center",
    "San Antonio Spurs": "Frost Bank Center",
    "Toronto Raptors": "Scotiabank Arena",
    "Utah Jazz": "Delta Center",
    "Washington Wizards": "Capital One Arena"
}


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
        
        # Use Eastern Time since that's the NBA's reference timezone
        self.timezone = pytz.timezone('US/Eastern')
    
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
    
    def _get_current_date(self):
        """
        Get current date in Eastern Time (NBA's reference timezone).
        
        Returns:
            Current date string in YYYY-MM-DD format
        """
        # Get current time in Eastern timezone (NBA's reference)
        eastern_now = datetime.datetime.now(self.timezone)
        
        # If it's before 6 AM ET, we're likely looking for yesterday's games
        # that are ongoing or about to finish
        if eastern_now.hour < 6:
            eastern_now = eastern_now - datetime.timedelta(days=1)
            
        return eastern_now.strftime("%Y-%m-%d")
    
    def _convert_utc_to_est(self, utc_str):
        """
        Convert UTC datetime string to Eastern Time date.
        
        Args:
            utc_str: UTC datetime string (e.g., "2025-03-26T23:30:00Z")
            
        Returns:
            Date in Eastern Time in YYYY-MM-DD format
        """
        try:
            # Parse the UTC string
            utc_dt = datetime.datetime.strptime(utc_str, "%Y-%m-%dT%H:%M:%SZ")
            utc_dt = pytz.utc.localize(utc_dt)
            
            # Convert to Eastern Time
            eastern_dt = utc_dt.astimezone(self.timezone)
            
            # Return just the date portion
            return eastern_dt.strftime("%Y-%m-%d")
        except Exception as e:
            logger.error(f"Error converting UTC time '{utc_str}' to EST: {str(e)}")
            return self._get_current_date()  # Default to current date as fallback
    
    def _get_venue_info(self, game_data, home_team):
        """
        Extract venue information from game data with fallbacks.
        
        Args:
            game_data: Game data dictionary from NBA API
            home_team: Home team name for fallback venue lookup
            
        Returns:
            Venue name string
        """
        # Try to get venue from arena object (primary method)
        if 'arena' in game_data and isinstance(game_data['arena'], dict):
            arena_name = game_data['arena'].get('arenaName')
            if arena_name and arena_name.strip():
                logger.info(f"Found venue from API: {arena_name}")
                return arena_name
        
        # Fallback 1: Look in arenaName directly (some API endpoints structure)
        arena_name = game_data.get('arenaName')
        if arena_name and isinstance(arena_name, str) and arena_name.strip():
            logger.info(f"Found venue from direct arenaName: {arena_name}")
            return arena_name
        
        # Fallback 2: Use home team's usual venue
        if home_team in TEAM_HOME_VENUES:
            venue = TEAM_HOME_VENUES[home_team]
            logger.info(f"Using home venue fallback for {home_team}: {venue}")
            return venue
        
        # Final fallback
        logger.warning(f"Could not determine venue for game with home team {home_team}")
        return "TBD"
    
    def scrape_nba_schedule(self, days_ahead=0, target_date=None):
        """
        Scrape NBA game schedule using the nba_api.
        
        Args:
            days_ahead: Number of days ahead (0 for today, 1 for tomorrow, etc.)
            target_date: Specific date to fetch (format: 'YYYY-MM-DD', overrides days_ahead)
            
        Returns:
            List of dictionaries containing game information
        """
        try:
            # Determine the target date (for logging only)
            if target_date:
                # Use provided date
                game_date = target_date
                logger.info(f"Scraping NBA schedule for specific date: {game_date}")
            else:
                # Calculate target date based on days_ahead
                current_date = self._get_current_date()
                target_dt = datetime.datetime.strptime(current_date, "%Y-%m-%d")
                target_dt = target_dt + datetime.timedelta(days=days_ahead)
                game_date = target_dt.strftime("%Y-%m-%d")
                logger.info(f"Scraping NBA schedule for date: {game_date} (days_ahead={days_ahead})")
            
            # Get games from NBA API
            games_today = scoreboard.ScoreBoard()
            games_dict = games_today.get_dict()
            all_games = games_dict.get('scoreboard', {}).get('games', [])
            
            # Log the raw structure of the first game for debugging
            if all_games:
                logger.debug(f"First game structure: {all_games[0]}")
            
            # Format the game data for our database
            formatted_games = []
            for game in all_games:
                game_id = game.get('gameId')
                home_team_data = game.get('homeTeam', {})
                away_team_data = game.get('awayTeam', {})
                
                # Get team names
                home_team = self._normalize_team_name(home_team_data.get('teamCity', '') + " " + home_team_data.get('teamName', ''))
                away_team = self._normalize_team_name(away_team_data.get('teamCity', '') + " " + away_team_data.get('teamName', ''))
                
                # Get game time in UTC
                game_time_utc = game.get('gameTimeUTC', '')
                
                # Convert UTC date to Eastern Time for proper date matching
                game_date_est = self._convert_utc_to_est(game_time_utc)
                
                # Log the actual date we're getting from the API
                logger.info(f"Found NBA API game: {away_team} @ {home_team} on {game_date_est}")
                
                # Create a datetime string in the format YYYY-MM-DD
                game_date_str = game_date_est
                
                # Get game time in HH:MM format
                try:
                    game_time_parts = game_time_utc.split('T')[1].split(':')
                    game_time = f"{game_time_parts[0]}:{game_time_parts[1]}"
                except (IndexError, ValueError):
                    game_time = "19:00"  # Default to 7 PM as fallback
                
                # Get venue information with improved extraction and fallbacks
                venue = self._get_venue_info(game, home_team)
                
                # Generate a custom game ID using your format
                custom_game_id = self._generate_game_id(home_team, away_team, game_date_str)
                
                # Format the game information for the database
                formatted_game = {
                    'game_id': custom_game_id,
                    'home_team': home_team,
                    'away_team': away_team,
                    'game_date': game_date_str,
                    'game_time': game_time,
                    'venue': venue
                }
                
                formatted_games.append(formatted_game)
            
            # Always accept whatever games the NBA API gives us
            if formatted_games:
                logger.info(f"Scraped {len(formatted_games)} games from NBA API")
            else:
                logger.warning("No games found in NBA API response. This could be due to an API limitation.")
                
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
    
    def scrape_and_save_games(self, days_ahead=0, target_date=None):
        """
        Scrape games from NBA API and save them to the database.
        
        Args:
            days_ahead: Number of days ahead (0 for today, 1 for tomorrow)
            target_date: Specific date to fetch (format: 'YYYY-MM-DD', overrides days_ahead)
            
        Returns:
            Number of games saved
        """
        # Initialize database if not already connected
        self.db.initialize_database()
        
        # Scrape from NBA API
        games = self.scrape_nba_schedule(days_ahead, target_date)
        
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