import os
import re
import time
import logging
import datetime
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

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
    Scrapes NBA game schedules from the official NBA.com schedule page using Selenium.
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
        
        # User Agent to mimic a browser
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36"
        
        # Delay between requests to be respectful of the servers
        self.request_delay = 2  # seconds
        
        # NBA.com schedule URL
        self.schedule_url = "https://www.nba.com/schedule"
    
    def _get_page_with_selenium(self, url: str) -> Optional[str]:
        """
        Load a webpage with Selenium to handle JavaScript rendering.
        
        Args:
            url: URL to load
            
        Returns:
            HTML content of the page or None if failed
        """
        try:
            logger.info(f"Loading URL with Selenium: {url}")
            
            # Set up Chrome options
            options = Options()
            options.add_argument("--headless=new")  # Run in headless mode
            options.add_argument(f"user-agent={self.user_agent}")
            options.add_argument("--disable-gpu")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            
            # Initialize Chrome driver
            service = Service(ChromeDriverManager().install())
            browser = webdriver.Chrome(service=service, options=options)
            
            # Set page load timeout
            browser.set_page_load_timeout(30)
            
            # Load the page
            browser.get(url)
            
            # Wait for JavaScript to load content
            time.sleep(5)  # Adjust this wait time if needed
            
            # Get the fully rendered HTML
            html_content = browser.page_source
            
            # Clean up
            browser.quit()
            
            logger.info("Successfully loaded page with Selenium")
            return html_content
            
        except Exception as e:
            logger.error(f"Error loading page with Selenium: {str(e)}")
            return None
    
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
    
    def scrape_nba_schedule(self, days_ahead: int = 0) -> List[Dict[str, Any]]:
        """
        Scrape NBA game schedule from the official NBA.com website.
        
        Args:
            days_ahead: Number of days ahead to scrape (default 0 for today's games)
            
        Returns:
            List of dictionaries containing game information
        """
        html_content = self._get_page_with_selenium(self.schedule_url)
        if not html_content:
            logger.error("Failed to retrieve NBA.com schedule page")
            return []
        
        soup = BeautifulSoup(html_content, 'html.parser')
        all_games = []
        
        # Based on your screenshot showing "THURSDAY, MARCH 20" heading
        # Now with Selenium, we should have the complete rendered page
        try:
            # Find the date headers (e.g., "THURSDAY, MARCH 20")
            day_headers = soup.find_all(['h3', 'h4', 'div'], class_=lambda c: c and ('Day' in c or 'day' in str(c).lower() or 'Date' in c))
            
            # If no specific class, try text pattern match
            if not day_headers:
                day_headers = []
                for element in soup.find_all(['h3', 'h4', 'div']):
                    if element.text and re.search(r'[A-Z]+DAY,\s+[A-Z]+\s+\d+', element.text.strip()):
                        day_headers.append(element)
            
            logger.info(f"Found {len(day_headers)} day headers")
            
            # For each day header, find the games for that day
            for day_header in day_headers:
                day_text = day_header.text.strip()
                logger.info(f"Processing games for: {day_text}")
                
                # Parse the date from the header
                try:
                    date_match = re.search(r'([A-Z]+DAY),\s+([A-Z]+)\s+(\d+)', day_text, re.IGNORECASE)
                    if date_match:
                        month_name = date_match.group(2)
                        day_num = int(date_match.group(3))
                        
                        # Get current year
                        current_year = datetime.datetime.now().year
                        
                        # Convert month name to number
                        month_dict = {
                            'JANUARY': 1, 'FEBRUARY': 2, 'MARCH': 3, 'APRIL': 4,
                            'MAY': 5, 'JUNE': 6, 'JULY': 7, 'AUGUST': 8,
                            'SEPTEMBER': 9, 'OCTOBER': 10, 'NOVEMBER': 11, 'DECEMBER': 12
                        }
                        month_num = month_dict.get(month_name.upper(), 1)
                        
                        # Create date string
                        game_date_str = f"{current_year}-{month_num:02d}-{day_num:02d}"
                        
                        # Check if this is the date we want (today + days_ahead)
                        target_date = datetime.date.today() + datetime.timedelta(days=days_ahead)
                        game_date = datetime.date(current_year, month_num, day_num)
                        
                        if game_date == target_date:
                            logger.info(f"Found target date: {game_date_str}")
                            
                            # Find the game container for this day
                            # It could be a sibling or a child of the day header
                            games_container = day_header.find_next_sibling(['div', 'section', 'ul'])
                            
                            if not games_container:
                                # Try looking for a container with games
                                parent = day_header.parent
                                games_container = parent.find(['div', 'section', 'ul'], class_=lambda c: c and ('games' in str(c).lower() or 'Games' in str(c)))
                            
                            # If still can't find, just use the parent
                            if not games_container:
                                games_container = parent
                            
                            # Look for game rows/cards - these vary by site design
                            game_elements = games_container.find_all(['div', 'li', 'article'], class_=lambda c: c and ('game' in str(c).lower() or 'Game' in str(c) or 'matchup' in str(c).lower() or 'Matchup' in str(c)))
                            
                            # If no specific class, try structure matching
                            if not game_elements:
                                # Look for elements that might contain game data
                                # Try to find elements with team links or time elements
                                game_elements = []
                                
                                # Find all time elements (e.g., "7:00 PM ET")
                                time_elements = games_container.find_all(['div', 'span', 'p'], string=lambda s: s and re.search(r'\d+:\d+\s+[AP]M', s))
                                
                                for time_element in time_elements:
                                    # Find the parent element that likely contains the game info
                                    game_parent = time_element.parent
                                    while game_parent and game_parent != games_container:
                                        # If this element has team links, it's probably a game element
                                        team_links = game_parent.find_all('a', href=lambda h: h and '/team/' in h)
                                        if len(team_links) >= 2:
                                            if game_parent not in game_elements:
                                                game_elements.append(game_parent)
                                            break
                                        game_parent = game_parent.parent
                            
                            logger.info(f"Found {len(game_elements)} game elements")
                            
                            # Process each game element
                            for game_element in game_elements:
                                try:
                                    # Extract teams - look for team links
                                    team_links = game_element.find_all('a', href=lambda h: h and '/team/' in h)
                                    
                                    # If no links, try looking for team names or abbreviations
                                    if len(team_links) < 2:
                                        team_elements = game_element.find_all(['div', 'span', 'p'], string=lambda s: s and any(abbr.upper() in s.upper() for abbr in TEAM_ABBR_TO_NAME.keys()))
                                        
                                        if len(team_elements) >= 2:
                                            team_links = team_elements
                                    
                                    if len(team_links) >= 2:
                                        # Get team names
                                        away_text = team_links[0].text.strip()
                                        home_text = team_links[1].text.strip()
                                        
                                        # Check if these are abbreviations
                                        if away_text.upper() in TEAM_ABBR_TO_NAME:
                                            away_team = TEAM_ABBR_TO_NAME[away_text.upper()]
                                        else:
                                            # Try to find full team name in text
                                            away_team = next((team for team in NBA_TEAMS.keys() if team.lower() in away_text.lower()), away_text)
                                        
                                        if home_text.upper() in TEAM_ABBR_TO_NAME:
                                            home_team = TEAM_ABBR_TO_NAME[home_text.upper()]
                                        else:
                                            # Try to find full team name in text
                                            home_team = next((team for team in NBA_TEAMS.keys() if team.lower() in home_text.lower()), home_text)
                                        
                                        # Extract game time
                                        time_element = game_element.find(['div', 'span', 'p'], string=lambda s: s and re.search(r'\d+:\d+\s+[AP]M', s))
                                        
                                        if time_element:
                                            game_time = self._parse_time(time_element.text.strip())
                                        else:
                                            game_time = "19:00"  # Default to 7 PM
                                        
                                        # Extract venue information if available
                                        venue_element = game_element.find(['div', 'span', 'p'], string=lambda s: s and ('Center' in s or 'Arena' in s))
                                        
                                        if not venue_element:
                                            # Try another approach to find the venue
                                            venue_candidates = game_element.find_all(['div', 'span', 'p'])
                                            for vc in venue_candidates:
                                                if vc.text and ('Center' in vc.text or 'Arena' in vc.text) and not re.search(r'\d+:\d+\s+[AP]M', vc.text):
                                                    venue_element = vc
                                                    break
                                        
                                        venue = venue_element.text.strip() if venue_element else "Unknown"
                                        
                                        # Generate game ID
                                        game_id = self._generate_game_id(home_team, away_team, game_date_str)
                                        
                                        # Create game data dictionary
                                        game_data = {
                                            'game_id': game_id,
                                            'home_team': home_team,
                                            'away_team': away_team,
                                            'game_date': game_date_str,
                                            'game_time': game_time,
                                            'venue': venue
                                        }
                                        
                                        all_games.append(game_data)
                                        logger.info(f"Found game: {away_team} @ {home_team} on {game_date_str} at {game_time}")
                                
                                except Exception as e:
                                    logger.error(f"Error parsing game element: {str(e)}")
                                    continue
                except Exception as e:
                    logger.error(f"Error parsing date: {str(e)}")
                    continue
        
        except Exception as e:
            logger.error(f"Error scraping NBA schedule: {str(e)}")
        
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
    
    def scrape_and_save_games(self, days_ahead: int = 0) -> int:
        """
        Scrape games from NBA.com and save them to the database.
        Focused on same-day prediction (default days_ahead=0).
        
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
    num_games = scraper.scrape_and_save_games(days_ahead=0)
    
    print(f"Scraped and saved {num_games} games for today")