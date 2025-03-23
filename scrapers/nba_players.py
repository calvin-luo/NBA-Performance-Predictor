import os
import re
import time
import logging
import datetime
from typing import List, Dict, Any, Optional, Tuple, Set
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

from data.database import Database
from scrapers.nba_games import NBA_TEAMS, TEAM_ABBR_TO_NAME

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('scrapers.nba_players')

# Player status categories
PLAYER_STATUSES = {
    'active': ['active', 'available', 'probable', 'expected'],
    'questionable': ['questionable', 'game-time decision', 'day-to-day'],
    'out': ['out', 'inactive', 'injured', 'suspension', 'not with team'],
    'unknown': ['unknown']
}

# Position abbreviations
POSITIONS = {
    'PG': 'Point Guard',
    'SG': 'Shooting Guard',
    'SF': 'Small Forward',
    'PF': 'Power Forward',
    'C': 'Center'
}


class NBAPlayersScraper:
    """
    Scrapes NBA player information and lineups from NBA.com/players/todays-lineups using Selenium.
    Updated to target specific elements in the HTML structure.
    """
    
    def __init__(self, db: Optional[Database] = None):
        """
        Initialize the NBA players scraper.
        
        Args:
            db: Database instance (optional, creates a new one if None)
        """
        # Initialize database (create a new one if not provided)
        self.db = db if db else Database()
        
        # User Agent to mimic a browser
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36"
        
        # Delay between requests to be respectful of the servers
        self.request_delay = 2  # seconds
        
        # NBA.com lineups URL
        self.lineups_url = "https://www.nba.com/players/todays-lineups"
    
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
        Normalize team abbreviation to the official NBA team name.
        
        Args:
            team_abbr: Team abbreviation to normalize
            
        Returns:
            Normalized team name
        """
        # Look up full team name from abbreviation
        if team_abbr.upper() in TEAM_ABBR_TO_NAME:
            return TEAM_ABBR_TO_NAME[team_abbr.upper()]
        
        # If not found directly, try to match with team variants
        for team_name, team_data in NBA_TEAMS.items():
            if team_abbr.lower() == team_data["abbr"].lower():
                return team_name
            
            # Also check if the abbreviation is in variants
            if team_abbr.lower() in [v.lower() for v in team_data["variants"]]:
                return team_name
        
        # If no match found, return the original abbreviation
        logger.warning(f"Could not normalize team abbreviation: {team_abbr}")
        return team_abbr
    
    def _normalize_player_status(self, status_text: str) -> str:
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
        
        # Default to active for players in today's lineups
        return 'active'
    
    def scrape_todays_lineups(self) -> List[Dict[str, Any]]:
        """
        Scrape NBA lineups from NBA.com's Today's Lineups page using Selenium.
        Updated to target specific HTML elements based on the website's structure.
        
        Returns:
            List of dictionaries containing player information
        """
        html_content = self._get_page_with_selenium(self.lineups_url)
        if not html_content:
            logger.error("Failed to retrieve NBA.com lineups page")
            return []
        
        soup = BeautifulSoup(html_content, 'html.parser')
        players = []
        
        # Check if there are any games today
        # The page might show a message like "No lineups available today"
        no_games_messages = soup.find_all(string=lambda text: text and ("no lineups" in text.lower() or 
                                                                     "no games" in text.lower() or 
                                                                     "check back" in text.lower()))
        if no_games_messages:
            logger.info(f"No games today: {no_games_messages[0]}")
            return []
        
        # Look for the game containers
        # Find all divs that have "DailyLineup_dl" in their class
        game_containers = soup.find_all('div', class_=lambda c: c and "DailyLineup_dl" in c)
        logger.info(f"Found {len(game_containers)} game containers")
        
        for game_container in game_containers:
            try:
                # Find the game matchup title (e.g., "CHI vs LAL")
                matchup_title = game_container.find(['h1', 'h2', 'h3'], class_=lambda c: c and "Block_blockTitleText" in c)
                
                if not matchup_title or not matchup_title.text:
                    logger.warning("Could not find matchup title")
                    continue
                
                matchup_text = matchup_title.text.strip()
                logger.info(f"Processing matchup: {matchup_text}")
                
                # Extract team abbreviations from the matchup text
                teams_match = re.search(r'([A-Z]{3})\s+vs\s+([A-Z]{3})', matchup_text, re.IGNORECASE)
                if not teams_match:
                    logger.warning(f"Could not extract team abbreviations from matchup: {matchup_text}")
                    continue
                
                team1_abbr = teams_match.group(1).upper()
                team2_abbr = teams_match.group(2).upper()
                
                team1_name = self._normalize_team_name(team1_abbr)
                team2_name = self._normalize_team_name(team2_abbr)
                
                logger.info(f"Teams in matchup: {team1_name} vs {team2_name}")
                
                # Find the team selector buttons
                team_buttons = game_container.find_all('button', class_=lambda c: c and "ButtonGroup_btn" in c)
                
                # Process each team in the matchup
                for team_idx, (team_abbr, team_name) in enumerate([(team1_abbr, team1_name), (team2_abbr, team2_name)]):
                    # Find the button for this team (may be used for switching active team view)
                    team_button = None
                    for button in team_buttons:
                        if button.text.strip() == team_abbr:
                            team_button = button
                            break
                    
                    # If this is the second team, we might need to click the button to see its lineup
                    # However, in our HTML analysis, we can just look for the team name in player elements
                    
                    # Find the player list for this team
                    player_list = game_container.find('ul', class_=lambda c: c and "DailyLineup_dlList" in c)
                    
                    if not player_list:
                        logger.warning(f"Could not find player list for {team_name}")
                        continue
                    
                    # Find all player elements within the list
                    player_elements = player_list.find_all('li', class_=lambda c: c and "DailyLineup_dlPlayer" in c)
                    
                    if not player_elements:
                        logger.warning(f"No player elements found for {team_name}")
                        continue
                    
                    logger.info(f"Found {len(player_elements)} players for {team_name}")
                    
                    # Extract player information from each element
                    for player_elem in player_elements:
                        try:
                            # Get player name
                            name_elem = player_elem.find('span', class_=lambda c: c and "DailyLineup_dlName" in c)
                            
                            if not name_elem or not name_elem.text:
                                continue
                            
                            player_name = name_elem.text.strip()
                            
                            # Get player position
                            pos_elem = player_elem.find('span', class_=lambda c: c and "DailyLineup_dlPos" in c)
                            position = pos_elem.text.strip() if pos_elem else ""
                            
                            # Get player status
                            status = "active"  # Default to active
                            
                            # Check for lineup status in data attributes
                            lineup_status = player_elem.get('data-lineup-status', '')
                            roster_status = player_elem.get('data-roster-status', '')
                            
                            if lineup_status or roster_status:
                                combined_status = f"{lineup_status} {roster_status}".strip()
                                status = self._normalize_player_status(combined_status)
                            
                            # Create player data
                            player_data = {
                                'name': player_name,
                                'team': team_name,
                                'position': position,
                                'status': status,
                                'status_detail': f"{lineup_status} {roster_status}".strip() or "Starting Lineup"
                            }
                            
                            players.append(player_data)
                            logger.info(f"Added player: {player_name} ({position}) - {team_name}")
                            
                        except Exception as e:
                            logger.error(f"Error processing player element: {str(e)}")
                            continue
                    
                    # After processing the first team, we need a way to switch to the second team's lineup
                    # In our case, we'll assume we've processed all available players in the container
                    # The real implementation might need to simulate clicking the team button
            
            except Exception as e:
                logger.error(f"Error processing game container: {str(e)}")
                continue
        
        # If we still haven't found any players, try a more direct approach
        if not players:
            logger.info("No players found with primary approach. Trying secondary approach...")
            
            # Look for any player elements on the page
            all_player_elems = soup.find_all('li', class_=lambda c: c and "DailyLineup_dlPlayer" in c)
            
            for player_elem in all_player_elems:
                try:
                    # Get player name
                    name_elem = player_elem.find('span', class_=lambda c: c and "DailyLineup_dlName" in c)
                    if not name_elem or not name_elem.text:
                        continue
                    
                    player_name = name_elem.text.strip()
                    
                    # Get player position
                    pos_elem = player_elem.find('span', class_=lambda c: c and "DailyLineup_dlPos" in c)
                    position = pos_elem.text.strip() if pos_elem else ""
                    
                    # Try to determine the team
                    team_name = None
                    
                    # Look up the hierarchy to find team context
                    parent = player_elem.parent
                    while parent and parent.name != 'html':
                        # Look for team buttons near this player
                        team_buttons = parent.find_all('button', class_=lambda c: c and "ButtonGroup_btn" in c)
                        for button in team_buttons:
                            if button.get('data-active') == 'true':
                                team_abbr = button.text.strip()
                                team_name = self._normalize_team_name(team_abbr)
                                break
                        
                        if team_name:
                            break
                        
                        # Look for matchup title
                        matchup_title = parent.find(['h1', 'h2', 'h3'], class_=lambda c: c and "Block_blockTitleText" in c)
                        if matchup_title and 'vs' in matchup_title.text.lower():
                            teams_match = re.search(r'([A-Z]{3})\s+vs\s+([A-Z]{3})', matchup_title.text, re.IGNORECASE)
                            if teams_match:
                                # Assume it's the first team (active team)
                                team_abbr = teams_match.group(1).upper()
                                team_name = self._normalize_team_name(team_abbr)
                                break
                        
                        parent = parent.parent
                    
                    if not team_name:
                        # Without team context, we can't properly categorize this player
                        logger.warning(f"Could not determine team for player {player_name}")
                        continue
                    
                    # Get player status
                    status = "active"  # Default to active
                    
                    # Check for lineup status in data attributes
                    lineup_status = player_elem.get('data-lineup-status', '')
                    roster_status = player_elem.get('data-roster-status', '')
                    
                    if lineup_status or roster_status:
                        combined_status = f"{lineup_status} {roster_status}".strip()
                        status = self._normalize_player_status(combined_status)
                    
                    # Create player data
                    player_data = {
                        'name': player_name,
                        'team': team_name,
                        'position': position,
                        'status': status,
                        'status_detail': f"{lineup_status} {roster_status}".strip() or "Starting Lineup"
                    }
                    
                    players.append(player_data)
                    logger.info(f"Added player (secondary approach): {player_name} ({position}) - {team_name}")
                
                except Exception as e:
                    logger.error(f"Error in secondary approach: {str(e)}")
                    continue
        
        logger.info(f"Scraped {len(players)} players from NBA.com lineups")
        return players
    
    def save_players_to_database(self, players: List[Dict[str, Any]]) -> int:
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
    
    def get_team_active_players(self, team_name: str) -> List[Dict[str, Any]]:
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
    
    def get_today_game_players(self, game_id: str) -> Dict[str, List[Dict[str, Any]]]:
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
    
    def scrape_and_save_players(self) -> int:
        """
        Scrape players from NBA.com lineups and save them to the database.
        
        Returns:
            Number of players saved
        """
        # Initialize database if not already connected
        self.db.initialize_database()
        
        # Scrape today's lineups
        players = self.scrape_todays_lineups()
        
        # Save players to database
        saved_count = self.save_players_to_database(players)
        
        logger.info(f"Saved {saved_count} players from today's lineups")
        
        return saved_count


# Example usage
if __name__ == "__main__":
    # Set up scraper
    scraper = NBAPlayersScraper()
    
    # Scrape and save player information
    players_saved = scraper.scrape_and_save_players()
    
    print(f"Saved {players_saved} players from today's lineups")