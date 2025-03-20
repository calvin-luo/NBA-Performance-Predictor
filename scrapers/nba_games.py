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

# Create reverse mappings
TEAM_ABBR_TO_NAME = {team_data["abbr"]: team_name for team_name, team_data in NBA_TEAMS.items()}
TEAM_ABBR_TO_SHORTNAME = {
    "ATL": "Hawks", "BOS": "Celtics", "BKN": "Nets", "CHA": "Hornets", 
    "CHI": "Bulls", "CLE": "Cavaliers", "DAL": "Mavericks", "DEN": "Nuggets",
    "DET": "Pistons", "GSW": "Warriors", "HOU": "Rockets", "IND": "Pacers",
    "LAC": "Clippers", "LAL": "Lakers", "MEM": "Grizzlies", "MIA": "Heat", 
    "MIL": "Bucks", "MIN": "Timberwolves", "NOP": "Pelicans", "NYK": "Knicks",
    "OKC": "Thunder", "ORL": "Magic", "PHI": "76ers", "PHX": "Suns",
    "POR": "Trail Blazers", "SAC": "Kings", "SAS": "Spurs", "TOR": "Raptors",
    "UTA": "Jazz", "WAS": "Wizards"
}


class NBAGamesScraper:
    """
    Scrapes NBA game schedules from the official NBA.com schedule page.
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
        
        # NBA.com schedule URL
        self.schedule_url = "https://www.nba.com/schedule"
    
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
    
    def _parse_team_from_text(self, team_text: str) -> str:
        """
        Parse and normalize team name from text.
        
        Args:
            team_text: Team name text to parse
            
        Returns:
            Normalized team name
        """
        # First, check if it's a city + name format (e.g., "New York Knicks")
        for team_name in NBA_TEAMS.keys():
            if team_name.lower() in team_text.lower():
                return team_name
        
        # Next, check for short name (e.g., "Knicks")
        for team_name, team_data in NBA_TEAMS.items():
            short_name = team_name.split()[-1]  # Get last word of team name
            if short_name.lower() in team_text.lower():
                return team_name
            
            # Check variants
            for variant in team_data["variants"]:
                if variant.lower() in team_text.lower():
                    return team_name
        
        # If still not found, return the original text
        logger.warning(f"Could not parse team name from: {team_text}")
        return team_text
    
    def _parse_team_from_abbr(self, abbr: str) -> str:
        """
        Convert team abbreviation to full team name.
        
        Args:
            abbr: Team abbreviation (e.g., "NYK")
            
        Returns:
            Full team name
        """
        return TEAM_ABBR_TO_NAME.get(abbr.upper(), abbr)
    
    def _parse_time(self, time_str: str) -> str:
        """
        Parse a time string into a consistent 24-hour format.
        
        Args:
            time_str: Time string (e.g., "7:00 PM ET")
            
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
    
    def scrape_nba_schedule(self, days_ahead: int = 1) -> List[Dict[str, Any]]:
        """
        Scrape NBA game schedule from the official NBA.com website.
        
        Args:
            days_ahead: Number of days ahead to scrape (default 1 for same-day prediction)
            
        Returns:
            List of dictionaries containing game information
        """
        html_content = self._make_request(self.schedule_url)
        if not html_content:
            logger.error("Failed to retrieve NBA.com schedule page")
            return []
        
        soup = BeautifulSoup(html_content, 'html.parser')
        all_games = []
        
        # Based on the screenshot, we need to find the schedule section for today/tomorrow
        # The screenshot shows games listed under "THURSDAY, MARCH 20"
        try:
            # Look for date headers (e.g., "THURSDAY, MARCH 20")
            date_headers = soup.find_all(['h3', 'h4', 'div'], string=lambda s: s and re.match(r'[A-Z]+DAY,\s+[A-Z]+\s+\d+', s))
            
            for date_header in date_headers[:days_ahead]:  # Limit to the specified days ahead
                date_text = date_header.text.strip()
                
                try:
                    # Parse date from header text (e.g., "THURSDAY, MARCH 20" -> 2025-03-20)
                    # Note: Need to add year since it might not be in the header
                    current_year = datetime.datetime.now().year
                    date_match = re.search(r'[A-Z]+DAY,\s+([A-Z]+)\s+(\d+)', date_text)
                    
                    if date_match:
                        month_name = date_match.group(1)
                        day = int(date_match.group(2))
                        
                        # Convert month name to number
                        month_names = ['JANUARY', 'FEBRUARY', 'MARCH', 'APRIL', 'MAY', 'JUNE',
                                      'JULY', 'AUGUST', 'SEPTEMBER', 'OCTOBER', 'NOVEMBER', 'DECEMBER']
                        month = month_names.index(month_name) + 1
                        
                        # Create date object
                        game_date = datetime.date(current_year, month, day)
                        game_date_str = game_date.strftime("%Y-%m-%d")
                        
                        # Find the list of games under this date header
                        # Look for game entries after this header until the next header
                        next_element = date_header.find_next_sibling()
                        
                        # Process all game entries under this date
                        while next_element and not (next_element.name in ['h3', 'h4'] and 
                                                   re.match(r'[A-Z]+DAY,\s+[A-Z]+\s+\d+', next_element.text.strip())):
                            
                            # Based on screenshot, each game has time, teams, and venue information
                            # Look for time elements (e.g., "7:00 PM ET")
                            time_element = next_element.find(['div', 'span', 'p'], string=lambda s: s and re.search(r'\d+:\d+\s+[AP]M\s+ET', s))
                            
                            if time_element:
                                game_time = self._parse_time(time_element.text.strip())
                                
                                # Find team names
                                # From the screenshot, team names appear as links
                                team_links = next_element.find_all('a', href=lambda h: h and '/team/' in h)
                                
                                if len(team_links) >= 2:  # Need at least home and away teams
                                    away_team_name = self._parse_team_from_text(team_links[0].text.strip())
                                    home_team_name = self._parse_team_from_text(team_links[1].text.strip())
                                    
                                    # Find venue information
                                    venue_element = next_element.find(['div', 'span', 'p'], string=lambda s: s and 'Center' in s)
                                    venue = venue_element.text.strip() if venue_element else "Unknown"
                                    
                                    # Generate game ID
                                    game_id = self._generate_game_id(home_team_name, away_team_name, game_date_str)
                                    
                                    # Create game data
                                    game_data = {
                                        'game_id': game_id,
                                        'home_team': home_team_name,
                                        'away_team': away_team_name,
                                        'game_date': game_date_str,
                                        'game_time': game_time,
                                        'venue': venue
                                    }
                                    
                                    all_games.append(game_data)
                                    logger.info(f"Found game: {away_team_name} @ {home_team_name} on {game_date_str} at {game_time}")
                            
                            # Move to the next element
                            next_element = next_element.find_next_sibling()
                
                except Exception as e:
                    logger.error(f"Error processing games for date {date_text}: {str(e)}")
                    continue
        
        except Exception as e:
            logger.error(f"Error parsing NBA.com schedule: {str(e)}")
        
        # Alternative approach based on the exact structure shown in the screenshot
        if not all_games:
            logger.info("Trying alternative parsing approach for NBA.com schedule")
            
            try:
                # Look for game rows with time, teams, and venue
                game_rows = soup.find_all('div', class_=lambda c: c and ('GameCard' in c or 'GameRow' in c or 'GameInfo' in c))
                
                today = datetime.date.today()
                today_str = today.strftime("%Y-%m-%d")
                
                for game_row in game_rows:
                    try:
                        # Find time element
                        time_element = game_row.find(['span', 'div', 'p'], string=lambda s: s and re.search(r'\d+:\d+\s+[AP]M', s))
                        
                        if time_element:
                            game_time = self._parse_time(time_element.text.strip())
                            
                            # Find team elements - based on the structure in the screenshot
                            # Look for team abbreviations and full names
                            team_elements = game_row.find_all(['a', 'span', 'div'], string=lambda s: s and any(abbr.lower() in s.lower() for abbr in TEAM_ABBR_TO_NAME.keys()))
                            
                            if len(team_elements) >= 2:
                                away_team = self._parse_team_from_text(team_elements[0].text.strip())
                                home_team = self._parse_team_from_text(team_elements[1].text.strip())
                                
                                # Find venue information
                                venue_element = game_row.find(['span', 'div', 'p'], string=lambda s: s and ('Center' in s or 'Arena' in s))
                                venue = venue_element.text.strip() if venue_element else "Unknown"
                                
                                # Generate game ID
                                game_id = self._generate_game_id(home_team, away_team, today_str)
                                
                                # Create game data
                                game_data = {
                                    'game_id': game_id,
                                    'home_team': home_team,
                                    'away_team': away_team,
                                    'game_date': today_str,
                                    'game_time': game_time,
                                    'venue': venue
                                }
                                
                                all_games.append(game_data)
                                logger.info(f"Found game using alternative approach: {away_team} @ {home_team} at {game_time}")
                    
                    except Exception as e:
                        logger.error(f"Error processing game row: {str(e)}")
                        continue
            
            except Exception as e:
                logger.error(f"Error with alternative parsing approach: {str(e)}")
        
        # If still no games, use direct pattern matching based on screenshot structure
        if not all_games:
            logger.info("Trying direct pattern matching for NBA schedule")
            
            try:
                # Look for patterns like "7:00 PM ET" close to team names
                time_elements = soup.find_all(['div', 'span', 'p'], string=lambda s: s and re.search(r'\d+:\d+\s+[AP]M\s+ET', s))
                
                today = datetime.date.today()
                today_str = today.strftime("%Y-%m-%d")
                
                for time_element in time_elements:
                    try:
                        game_time = self._parse_time(time_element.text.strip())
                        
                        # Look for nearby elements that might contain team information
                        parent = time_element.parent
                        
                        # Find team elements in the same container
                        team_links = parent.find_all('a', href=lambda h: h and '/team/' in h)
                        
                        # If no direct links, try looking for text that matches team names
                        if len(team_links) < 2:
                            team_text_elements = parent.find_all(['div', 'span', 'p', 'a'], string=lambda s: s and any(team.lower() in s.lower() for team in TEAM_ABBR_TO_SHORTNAME.values()))
                            
                            if len(team_text_elements) >= 2:
                                away_team = self._parse_team_from_text(team_text_elements[0].text.strip())
                                home_team = self._parse_team_from_text(team_text_elements[1].text.strip())
                                
                                # Look for venue in nearby elements
                                venue_element = parent.find(['div', 'span', 'p'], string=lambda s: s and ('Center' in s or 'Arena' in s))
                                venue = venue_element.text.strip() if venue_element else "Unknown"
                                
                                # Generate game ID
                                game_id = self._generate_game_id(home_team, away_team, today_str)
                                
                                # Create game data
                                game_data = {
                                    'game_id': game_id,
                                    'home_team': home_team,
                                    'away_team': away_team,
                                    'game_date': today_str,
                                    'game_time': game_time,
                                    'venue': venue
                                }
                                
                                all_games.append(game_data)
                                logger.info(f"Found game using pattern matching: {away_team} @ {home_team} at {game_time}")
                    
                    except Exception as e:
                        logger.error(f"Error processing time element: {str(e)}")
                        continue
            
            except Exception as e:
                logger.error(f"Error with pattern matching approach: {str(e)}")
        
        logger.info(f"Scraped {len(all_games)} games from NBA.com schedule")
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
        Scrape games from NBA.com and save them to the database.
        Limited to same-day prediction (default days_ahead=1).
        
        Args:
            days_ahead: Number of days ahead to scrape
            
        Returns:
            Number of games saved
        """
        # Initialize database if not already connected
        self.db.initialize_database()
        
        # Scrape from NBA.com
        games = self.scrape_nba_schedule(days_ahead)
        
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