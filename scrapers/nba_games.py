import os
import re
import time
import logging
import datetime
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from data.database import Database

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('scrapers.nba_games')

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

# Create reverse mapping from abbreviation to full team name
TEAM_ABBR_TO_NAME = {team_data["abbr"]: team_name for team_name, team_data in NBA_TEAMS.items()}


class NBAGamesScraper:
    """
    Scrapes NBA game schedules and information from ESPN.
    Focused on scraping same-day games for prediction.
    """
    
    def __init__(self, db: Optional[Database] = None):
        """
        Initialize the NBA games scraper.
        
        Args:
            db: Database instance (optional, creates a new one if None)
        """
        # Initialize database (create a new one if not provided)
        self.db = db if db else Database()
        
        # User Agent to mimic a browser (helps avoid blocks from some websites)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        # Delay between requests to be respectful of the servers
        self.request_delay = 2  # seconds
    
    def _make_request(self, url: str) -> Optional[str]:
        """
        Make an HTTP request with error handling and retries.
        
        Args:
            url: URL to request
            
        Returns:
            HTML content of the page or None if failed
        """
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Making request to: {url}")
                response = requests.get(url, headers=self.headers, timeout=30)
                response.raise_for_status()  # Raise an exception for HTTP errors
                
                # Add a delay to avoid hammering the server
                time.sleep(self.request_delay)
                
                return response.text
            
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed (attempt {attempt+1}/{max_retries}): {str(e)}")
                
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Max retries reached. Giving up on URL: {url}")
                    return None
    
    def _normalize_team_name(self, team_name: str) -> str:
        """
        Normalize team name to the official NBA team name.
        
        Args:
            team_name: Team name to normalize
            
        Returns:
            Normalized team name
        """
        # Try direct match with team abbreviation
        team_name = team_name.strip()
        if team_name in TEAM_ABBR_TO_NAME:
            return TEAM_ABBR_TO_NAME[team_name]
        
        # Try to match with official team name
        for official_name in NBA_TEAMS:
            if team_name.lower() == official_name.lower():
                return official_name
        
        # Try to match with team variants
        team_name_lower = team_name.lower()
        for official_name, team_data in NBA_TEAMS.items():
            if any(variant in team_name_lower for variant in team_data["variants"]):
                return official_name
        
        # If no match found, return the original name
        logger.warning(f"Could not normalize team name: {team_name}")
        return team_name
    
    def _parse_time(self, time_str: str) -> str:
        """
        Parse a time string into a consistent 24-hour format.
        
        Args:
            time_str: Time string (e.g., "7:30 PM ET")
            
        Returns:
            Time in HH:MM format
        """
        try:
            # Extract the time part and AM/PM
            match = re.search(r'(\d+):?(\d*)\s*(AM|PM|am|pm)', time_str)
            if match:
                hour = int(match.group(1))
                minute = int(match.group(2)) if match.group(2) else 0
                am_pm = match.group(3).upper()
                
                # Convert to 24-hour format
                if am_pm == 'PM' and hour < 12:
                    hour += 12
                elif am_pm == 'AM' and hour == 12:
                    hour = 0
                
                return f"{hour:02d}:{minute:02d}"
            else:
                return "19:00"  # Default time if parsing fails
        except Exception as e:
            logger.error(f"Error parsing time string '{time_str}': {str(e)}")
            return "19:00"  # Default to 7 PM as fallback
    
    def _generate_game_id(self, home_team: str, away_team: str, game_date: str) -> str:
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
    
    def scrape_espn_schedule(self, days_ahead: int = 1) -> List[Dict[str, Any]]:
        """
        Scrape NBA game schedule from ESPN.
        
        Args:
            days_ahead: Number of days ahead to scrape (default 1 for same-day prediction)
            
        Returns:
            List of dictionaries containing game information
        """
        # ESPN Schedule URL
        base_url = "https://www.espn.com/nba/schedule"
        today = datetime.date.today()
        
        all_games = []
        
        # Scrape for each date
        for day_offset in range(days_ahead):
            target_date = today + datetime.timedelta(days=day_offset)
            date_param = target_date.strftime("%Y%m%d")
            
            # Only add the date parameter if we're not looking at today
            url = base_url if day_offset == 0 else f"{base_url}/_/date/{date_param}"
            
            html_content = self._make_request(url)
            if not html_content:
                continue
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find all tables that might contain schedule information
            schedule_tables = soup.find_all('table', class_=lambda c: c and ('schedule' in c.lower() or 'Table' in c))
            
            if not schedule_tables:
                logger.warning(f"No schedule tables found for date: {target_date}. Trying alternative parsing.")
                
                # Try to find game cards or game containers
                game_containers = soup.find_all('div', class_=lambda c: c and ('event' in c.lower() or 'game' in c.lower() or 'matchup' in c.lower()))
                
                if game_containers:
                    for game in game_containers:
                        try:
                            # Find team names
                            team_elements = game.find_all(['a', 'span', 'div'], class_=lambda c: c and ('team' in c.lower() or 'competitor' in c.lower()))
                            
                            # If specific team elements not found, try a more generic approach
                            if len(team_elements) < 2:
                                team_elements = game.find_all(['a', 'span', 'div'], string=lambda s: s and any(team in s.lower() for team_name, team_data in NBA_TEAMS.items() for team in team_data["variants"]))
                            
                            if len(team_elements) >= 2:
                                away_team = self._normalize_team_name(team_elements[0].text.strip())
                                home_team = self._normalize_team_name(team_elements[1].text.strip())
                                
                                # Find game time
                                time_element = game.find(['span', 'div'], string=lambda s: s and (':' in s and ('PM' in s or 'AM' in s)))
                                
                                if time_element:
                                    time_text = time_element.text.strip()
                                    game_time = self._parse_time(time_text)
                                else:
                                    # Try to find time in other formats
                                    time_element = game.find(['span', 'div'], string=lambda s: s and re.search(r'\d+:\d+', s))
                                    if time_element:
                                        game_time = self._parse_time(time_element.text.strip())
                                    else:
                                        game_time = "19:00"  # Default if time not found
                                
                                # Use target date for game date
                                game_date = target_date.strftime("%Y-%m-%d")
                                
                                # Generate a unique game ID
                                game_id = self._generate_game_id(home_team, away_team, game_date)
                                
                                # Create game data dictionary
                                game_data = {
                                    'game_id': game_id,
                                    'home_team': home_team,
                                    'away_team': away_team,
                                    'game_date': game_date,
                                    'game_time': game_time,
                                    'venue': "Unknown"  # ESPN might not show venue in this view
                                }
                                
                                all_games.append(game_data)
                                logger.info(f"Found game: {away_team} @ {home_team} at {game_time}")
                                
                        except Exception as e:
                            logger.error(f"Error parsing game container: {str(e)}")
                            continue
                else:
                    logger.warning(f"No games found for date: {target_date} using alternative parsing.")
            
            # Process schedule tables if found
            for table in schedule_tables:
                # Find all the game rows
                game_rows = table.find_all('tr')
                
                # Skip header row
                for row in game_rows[1:]:
                    try:
                        cells = row.find_all('td')
                        
                        # Skip rows that don't have enough cells
                        if len(cells) < 2:
                            continue
                        
                        # Extract teams
                        away_team_element = cells[0].find('a', {'name': 'awayTeam'}) or cells[0].find('a')
                        home_team_element = cells[1].find('a', {'name': 'homeTeam'}) or cells[1].find('a')
                        
                        if not away_team_element or not home_team_element:
                            logger.warning(f"Could not find team elements in row")
                            continue
                        
                        away_team = self._normalize_team_name(away_team_element.text.strip())
                        home_team = self._normalize_team_name(home_team_element.text.strip())
                        
                        # Extract game time
                        time_element = None
                        for cell in cells[2:]:
                            possible_time = cell.get_text().strip()
                            if ':' in possible_time and ('AM' in possible_time or 'PM' in possible_time):
                                time_element = possible_time
                                break
                        
                        if time_element:
                            game_time = self._parse_time(time_element)
                        else:
                            game_time = "19:00"  # Default to 7 PM if time not found
                        
                        # Extract venue if available
                        venue = "Unknown"
                        for cell in cells:
                            venue_text = cell.get_text().strip()
                            if "arena" in venue_text.lower() or "center" in venue_text.lower():
                                venue = venue_text
                                break
                        
                        # Use target date
                        game_date = target_date.strftime("%Y-%m-%d")
                        
                        # Generate a unique game ID
                        game_id = self._generate_game_id(home_team, away_team, game_date)
                        
                        # Create game data dictionary
                        game_data = {
                            'game_id': game_id,
                            'home_team': home_team,
                            'away_team': away_team,
                            'game_date': game_date,
                            'game_time': game_time,
                            'venue': venue
                        }
                        
                        all_games.append(game_data)
                        logger.info(f"Found game: {away_team} @ {home_team} at {game_time}")
                        
                    except Exception as e:
                        logger.error(f"Error parsing game row: {str(e)}")
                        continue
        
        logger.info(f"Scraped {len(all_games)} games from ESPN schedule for the next {days_ahead} days")
        return all_games
    
    def save_games_to_database(self, games: List[Dict[str, Any]]) -> int:
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
    
    def scrape_and_save_games(self, days_ahead: int = 1) -> int:
        """
        Scrape games from ESPN and save them to the database.
        Limited to same-day prediction (default days_ahead=1).
        
        Args:
            days_ahead: Number of days ahead to scrape
            
        Returns:
            Number of games saved
        """
        # Initialize database if not already connected
        self.db.initialize_database()
        
        # Scrape from ESPN
        games = self.scrape_espn_schedule(days_ahead)
        
        # Save games to database
        saved_count = self.save_games_to_database(games)
        
        return saved_count


# Example usage
if __name__ == "__main__":
    # Set up scraper
    scraper = NBAGamesScraper()
    
    # Scrape and save games for today (same-day prediction)
    num_games = scraper.scrape_and_save_games(days_ahead=1)
    
    print(f"Scraped and saved {num_games} games for today")