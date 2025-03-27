import os
import re
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import pytz

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
logger = logging.getLogger('scrapers.rotowire_scraper')

# Define NBA teams and abbreviations 
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

# Create mapping from abbreviation to team name
TEAM_ABBR_TO_NAME = {team_data["abbr"]: team_name for team_name, team_data in NBA_TEAMS.items()}

# Player status mapping
PLAYER_STATUS_MAP = {
    'OUT': 'out',
    'GTD': 'questionable',
    'PROB': 'probable',
    'OFS': 'out',  # Out for season
    'SUSP': 'out'  # Suspended
}


class RotowireScraper:
    """
    Scrapes NBA lineups from Rotowire.com.
    Provides game information, starting lineups, and player statuses.
    """
    
    def __init__(self, db: Optional[Database] = None):
        """
        Initialize the Rotowire scraper.
        
        Args:
            db: Database instance (optional, creates a new one if None)
        """
        # Initialize database (create a new one if not provided)
        self.db = db if db else Database()
        
        # User Agent to mimic a browser
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36"
        
        # Delay between requests to be respectful of the servers
        self.request_delay = 2  # seconds
        
        # Rotowire URLs
        self.lineups_url = "https://www.rotowire.com/basketball/nba-lineups.php"
        self.tomorrow_url = "https://www.rotowire.com/basketball/nba-lineups.php?date=tomorrow"
        
        # Use Eastern Time since that's the NBA's reference timezone
        self.timezone = pytz.timezone('US/Eastern')
    
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
    
    def _normalize_team_name(self, team_abbr: str) -> str:
        """
        Convert team abbreviation to full team name.
        
        Args:
            team_abbr: Team abbreviation (e.g., "LAL")
            
        Returns:
            Full team name
        """
        # Use the mapping from abbreviation to team name
        return TEAM_ABBR_TO_NAME.get(team_abbr.upper(), team_abbr)
    
    def _parse_player_status(self, status_text: str) -> str:
        """
        Normalize player status text.
        
        Args:
            status_text: Player status text from Rotowire
            
        Returns:
            Normalized status: 'active', 'probable', 'questionable', or 'out'
        """
        if not status_text:
            return 'active'
        
        # Normalize the status based on the mapping
        return PLAYER_STATUS_MAP.get(status_text.strip(), 'active')
    
    def _generate_game_id(self, home_team: str, away_team: str, game_date: str) -> str:
        """
        Generate a unique game ID in the format required by the database.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            game_date: Game date in YYYY-MM-DD format
            
        Returns:
            Unique game ID string
        """
        # Create a consistent game ID format: DATE_AWAYTEAM_HOMETEAM
        # Use team abbreviations for brevity
        home_abbr = None
        away_abbr = None
        
        # Find the abbreviations from the NBA_TEAMS dictionary
        for team_name, team_data in NBA_TEAMS.items():
            if team_name == home_team:
                home_abbr = team_data["abbr"]
            if team_name == away_team:
                away_abbr = team_data["abbr"]
        
        # Fallbacks if not found
        if not home_abbr:
            home_abbr = "UNK"
        if not away_abbr:
            away_abbr = "UNK"
        
        # Format date without separators
        date_no_sep = game_date.replace("-", "")
        
        return f"{date_no_sep}_{away_abbr}_{home_abbr}"
    
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
    
    def _get_current_date(self) -> str:
        """
        Get current date in Eastern Time (NBA's reference timezone).
        
        Returns:
            Current date string in YYYY-MM-DD format
        """
        # Get current time in Eastern timezone (NBA's reference)
        eastern_now = datetime.now(self.timezone)
        
        # If it's before 6 AM ET, we're likely looking for yesterday's games
        # that are ongoing or about to finish
        if eastern_now.hour < 6:
            eastern_now = eastern_now - timedelta(days=1)
            
        return eastern_now.strftime("%Y-%m-%d")
    
    def _extract_game_date(self, html_content: str) -> str:
        """
        Extract the current game date from the page.
        
        Args:
            html_content: HTML content of the page
            
        Returns:
            Date string in YYYY-MM-DD format
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find the date in the page title
            page_title = soup.find('div', class_='page-title__secondary')
            if page_title and page_title.text:
                date_text = page_title.text.strip()
                
                # Parse the date
                match = re.search(r'(\w+) (\d+), (\d{4})', date_text)
                if match:
                    month_name = match.group(1)
                    day = int(match.group(2))
                    year = int(match.group(3))
                    
                    # Convert month name to number
                    month_map = {
                        'January': 1, 'February': 2, 'March': 3, 'April': 4,
                        'May': 5, 'June': 6, 'July': 7, 'August': 8,
                        'September': 9, 'October': 10, 'November': 11, 'December': 12
                    }
                    month = month_map.get(month_name, 1)
                    
                    # Format date
                    return f"{year}-{month:02d}-{day:02d}"
            
            # Fallback to current date
            return self._get_current_date()
            
        except Exception as e:
            logger.error(f"Error extracting game date: {str(e)}")
            # Fallback to current date
            return self._get_current_date()
    
    def _extract_lineup_players(self, lineup_list: Any, team_name: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extract player information from a lineup list.
        
        Args:
            lineup_list: BeautifulSoup element containing the lineup list
            team_name: Name of the team
            
        Returns:
            Tuple of (starting_lineup, injured_players)
        """
        starting_lineup = []
        injured_players = []
        
        # Find all elements in order (both players and section headers)
        all_elements = lineup_list.find_all('li')
        
        # Flag to track when we're in the injury section
        in_injury_section = False
        
        # Process elements in the order they appear in the HTML
        for elem in all_elements:
            # Skip the status header (Confirmed Lineup)
            if 'lineup__status' in elem.get('class', []):
                continue
                
            # Check if this is the "MAY NOT PLAY" section header
            if 'lineup__title' in elem.get('class', []) and elem.text.strip() == 'MAY NOT PLAY':
                in_injury_section = True
                continue
            
            # Process player elements
            if 'lineup__player' in elem.get('class', []):
                # Get position
                pos_elem = elem.find('div', class_='lineup__pos')
                position = pos_elem.text.strip() if pos_elem else ""
                
                # Get player name
                player_link = elem.find('a')
                if not player_link:
                    continue
                    
                player_name = player_link.text.strip()
                
                # Get player ID from href if available
                player_id = None
                player_href = player_link.get('href', '')
                player_id_match = re.search(r'/player/([^/]+)-(\d+)$', player_href)
                if player_id_match:
                    player_id = player_id_match.group(2)
                
                # Get player status
                status_elem = elem.find('span', class_='lineup__inj')
                status_text = status_elem.text.strip() if status_elem else None
                status = self._parse_player_status(status_text)
                
                # Skip hidden OFS players
                if status_text == 'OFS' and 'hide' in elem.get('class', []):
                    continue
                
                # Create player data
                player_data = {
                    'name': player_name,
                    'position': position,
                    'status': status,
                    'player_id': player_id
                }
                
                # Add to appropriate list based on current section
                if in_injury_section:
                    injured_players.append(player_data)
                else:
                    # Check if the player has "is-pct-play-100" class
                    if 'is-pct-play-100' in elem.get('class', []):
                        starting_lineup.append(player_data)
                    else:
                        injured_players.append(player_data)
        
        return starting_lineup, injured_players
    
    def scrape_rotowire_lineups(self, date_type: str = 'today') -> List[Dict[str, Any]]:
        """
        Scrape NBA lineups from Rotowire.
        
        Args:
            date_type: 'today' or 'tomorrow' for which lineups to fetch
            
        Returns:
            List of dictionaries containing game and lineup information
        """
        # Choose the URL based on date_type
        url = self.lineups_url if date_type == 'today' else self.tomorrow_url
        
        # Get the page content
        html_content = self._get_page_with_selenium(url)
        if not html_content:
            logger.error(f"Failed to retrieve Rotowire NBA lineups page for {date_type}")
            return []
        
        soup = BeautifulSoup(html_content, 'html.parser')
        all_games = []
        
        # Extract the current game date from the page
        game_date = self._extract_game_date(html_content)
        
        # Verify if the date matches today
        today_date = self._get_current_date()
        if date_type == 'today' and game_date != today_date:
            logger.warning(f"Rotowire 'today' page shows date {game_date}, but current date is {today_date}")
            logger.info(f"Using current date: {today_date}")
            # Override with today's date
            game_date = today_date
        
        # Find all game lineup containers (excluding promo/ad containers)
        game_containers = soup.find_all('div', class_='lineup', attrs={'data-lnum': True})
        game_containers = [g for g in game_containers if 'is-nba' in g.get('class', []) and not 'is-picks' in g.get('class', [])]
        
        logger.info(f"Found {len(game_containers)} game containers")
        
        for game_container in game_containers:
            try:
                # Get game time
                game_time_elem = game_container.find('div', class_='lineup__time')
                game_time_str = game_time_elem.text.strip() if game_time_elem else "7:00 PM ET"
                game_time = self._parse_time(game_time_str)
                
                # Find the team boxes
                box = game_container.find('div', class_='lineup__box')
                if not box:
                    logger.warning("Could not find lineup box. Skipping game.")
                    continue
                
                # Get team abbreviations
                team_elements = box.find_all('div', class_='lineup__abbr')
                if len(team_elements) != 2:
                    logger.warning(f"Expected 2 teams, found {len(team_elements)}. Skipping game.")
                    continue
                
                away_abbr = team_elements[0].text.strip()
                home_abbr = team_elements[1].text.strip()
                
                # Get full team names
                away_team = self._normalize_team_name(away_abbr)
                home_team = self._normalize_team_name(home_abbr)
                
                # Generate game ID
                game_id = self._generate_game_id(home_team, away_team, game_date)
                
                # Extract records if available
                record_elements = box.find_all('span', class_='lineup__wl')
                away_record = record_elements[0].text.strip() if len(record_elements) > 0 else ""
                home_record = record_elements[1].text.strip() if len(record_elements) > 1 else ""
                
                # Get venue if available
                venue = "Unknown"  # Default venue
                
                # Find the main element containing lineups
                lineup_main = box.find('div', class_='lineup__main')
                if not lineup_main:
                    logger.warning("Could not find lineup main element. Skipping game.")
                    continue
                
                # Find lineup lists (away and home)
                away_list = lineup_main.find('ul', class_=lambda c: c and 'lineup__list' in c and 'is-visit' in c)
                home_list = lineup_main.find('ul', class_=lambda c: c and 'lineup__list' in c and 'is-home' in c)
                
                # Initialize team lineups
                away_lineup = []
                home_lineup = []
                away_injuries = []
                home_injuries = []
                
                # Extract away team lineup
                if away_list:
                    away_lineup, away_injuries = self._extract_lineup_players(away_list, away_team)
                    logger.info(f"Extracted {len(away_lineup)} players and {len(away_injuries)} injuries for {away_team}")
                else:
                    logger.warning(f"Could not find away team lineup list for {away_team}")
                
                # Extract home team lineup
                if home_list:
                    home_lineup, home_injuries = self._extract_lineup_players(home_list, home_team)
                    logger.info(f"Extracted {len(home_lineup)} players and {len(home_injuries)} injuries for {home_team}")
                else:
                    logger.warning(f"Could not find home team lineup list for {home_team}")
                
                # Extract referees if available
                refs_elem = box.find('div', class_='lineup__umpire')
                referees = []
                if refs_elem:
                    ref_links = refs_elem.find_all('a')
                    referees = [ref.text.strip() for ref in ref_links]
                
                # Extract odds if available
                odds_elem = box.find('div', class_='lineup__odds')
                game_odds = {
                    'line': None,
                    'spread': None,
                    'over_under': None
                }
                
                if odds_elem:
                    # Extract line (moneyline)
                    line_elems = odds_elem.find_all('span', class_='draftkings')
                    if len(line_elems) > 0:
                        game_odds['line'] = line_elems[0].text.strip()
                    
                    # Extract spread
                    if len(line_elems) > 1:
                        game_odds['spread'] = line_elems[1].text.strip()
                    
                    # Extract over/under
                    if len(line_elems) > 2:
                        game_odds['over_under'] = line_elems[2].text.strip()
                
                # Create full game data dictionary
                game_data = {
                    'game_id': game_id,
                    'home_team': home_team,
                    'away_team': away_team,
                    'game_date': game_date,
                    'game_time': game_time,
                    'venue': venue,
                    'away_record': away_record,
                    'home_record': home_record,
                    'referees': referees,
                    'odds': game_odds,
                    'home_lineup': home_lineup,
                    'away_lineup': away_lineup,
                    'home_injuries': home_injuries,
                    'away_injuries': away_injuries
                }
                
                all_games.append(game_data)
                logger.info(f"Processed game: {away_team} @ {home_team}")
                
            except Exception as e:
                logger.error(f"Error processing game container: {str(e)}")
                continue
        
        logger.info(f"Successfully scraped {len(all_games)} games")
        return all_games
    
    def save_lineups_to_database(self, games: List[Dict[str, Any]]) -> int:
        """
        Save scraped lineups to the database.
        
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
                    self.db.update_game(game['game_id'], {
                        'game_date': game['game_date'],
                        'game_time': game['game_time'],
                        'venue': game['venue']
                    })
                    logger.info(f"Updated game: {game['game_id']}")
                else:
                    # Insert new game
                    game_data = {
                        'game_id': game['game_id'],
                        'home_team': game['home_team'],
                        'away_team': game['away_team'],
                        'game_date': game['game_date'],
                        'game_time': game['game_time'],
                        'venue': game['venue']
                    }
                    self.db.insert_game(game_data)
                    logger.info(f"Inserted game: {game['game_id']}")
                
                # Process players from lineups
                self._process_players(game['home_team'], game['home_lineup'], game['game_id'])
                self._process_players(game['away_team'], game['away_lineup'], game['game_id'])
                
                # Process injured players
                self._process_players(game['home_team'], game['home_injuries'], game['game_id'])
                self._process_players(game['away_team'], game['away_injuries'], game['game_id'])
                
                saved_count += 1
                
            except Exception as e:
                logger.error(f"Error saving game {game.get('game_id', 'unknown')}: {str(e)}")
                continue
        
        logger.info(f"Saved {saved_count} games to database")
        return saved_count
    
    def _process_players(self, team: str, players: List[Dict[str, Any]], game_id: str) -> None:
        """
        Process and save players to the database.
        
        Args:
            team: Team name
            players: List of player dictionaries
            game_id: Game ID for linking players to games
        """
        for player in players:
            try:
                # Create player data for database
                player_data = {
                    'name': player['name'],
                    'team': team,
                    'position': player.get('position', ''),
                    'status': player.get('status', 'active'),
                    'game_id': game_id
                }
                
                # Insert player
                self.db.insert_player(player_data)
                
            except Exception as e:
                logger.error(f"Error saving player {player.get('name', 'unknown')}: {str(e)}")
    
    def scrape_and_save_lineups(self, date_type: str = 'today') -> int:
        """
        Scrape Rotowire lineups and save them to the database.
        
        Args:
            date_type: 'today' or 'tomorrow'
            
        Returns:
            Number of games saved
        """
        # Initialize database if not already connected
        self.db.initialize_database()
        
        # Scrape from Rotowire
        games = self.scrape_rotowire_lineups(date_type)
        
        # Save lineups to database
        saved_count = self.save_lineups_to_database(games)
        
        return saved_count