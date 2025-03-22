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
    'active': ['active', 'available', 'probable'],
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
            
            try:
                # Find the main content div to ensure content has loaded
                root_div = browser.find_element(By.ID, "__next")
                html_content = root_div.get_attribute("outerHTML")
                html = f"<html><body>{html_content}</body></html>"
            except Exception as e:
                logger.warning(f"Could not find __next element, getting full page source: {str(e)}")
                html = browser.page_source
            
            # Clean up
            browser.quit()
            
            logger.info("Successfully loaded page with Selenium")
            return html
            
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
        
        Returns:
            List of dictionaries containing player information
        """
        html_content = self._get_page_with_selenium(self.lineups_url)
        if not html_content:
            logger.error("Failed to retrieve NBA.com lineups page")
            return []
        
        soup = BeautifulSoup(html_content, 'html.parser')
        players = []
        
        # Based on the screenshot, we need to look for matchup sections like "NYK VS CHA"
        try:
            # Look for matchup headers - now with Selenium, we should have the full rendered page
            matchup_headers = soup.find_all(['h2', 'h3', 'div'], string=lambda s: s and re.search(r'[A-Z]{3}\s+VS\s+[A-Z]{3}', s))
            
            # Alternative approach: look for matchup headers by class
            if not matchup_headers:
                matchup_headers = soup.find_all(['div', 'h2', 'h3'], class_=lambda c: c and ('matchup' in str(c).lower() or 'vs' in str(c).lower()))
            
            logger.info(f"Found {len(matchup_headers)} matchup headers")
            
            for header in matchup_headers:
                try:
                    # Extract team abbreviations from the header
                    header_text = header.text.strip()
                    teams_match = re.search(r'([A-Z]{3})\s+VS\s+([A-Z]{3})', header_text)
                    
                    if teams_match:
                        away_abbr = teams_match.group(1)
                        home_abbr = teams_match.group(2)
                    else:
                        # Try alternative parsing if the first regex doesn't match
                        # Look for any three-letter uppercase words that might be team abbreviations
                        abbrs = re.findall(r'\b[A-Z]{3}\b', header_text)
                        if len(abbrs) >= 2:
                            away_abbr = abbrs[0]
                            home_abbr = abbrs[1]
                        else:
                            logger.warning(f"Could not extract team abbreviations from: {header_text}")
                            continue
                    
                    # Convert abbreviations to full team names
                    away_team = self._normalize_team_name(away_abbr)
                    home_team = self._normalize_team_name(home_abbr)
                    
                    logger.info(f"Processing matchup: {away_team} @ {home_team}")
                    
                    # Find team sections for this matchup
                    # Based on the screenshot, each team has a column with players
                    
                    # The team sections could be in different structures:
                    # 1. As tabs/tabs under the matchup header
                    # 2. As columns under the matchup header
                    # 3. Direct player listings
                    
                    # First, look for tab elements with team abbreviations
                    team_tabs = []
                    parent = header.parent
                    
                    # Look for elements containing the team abbreviations
                    away_tab = parent.find(['div', 'button', 'span'], string=lambda s: s and away_abbr in s)
                    home_tab = parent.find(['div', 'button', 'span'], string=lambda s: s and home_abbr in s)
                    
                    if away_tab and home_tab:
                        team_tabs = [away_tab, home_tab]
                    
                    # Process each team
                    teams = [(away_abbr, away_team), (home_abbr, home_team)]
                    
                    for idx, (team_abbr, team_name) in enumerate(teams):
                        # Find the team section - it could be a tab content or a column
                        team_section = None
                        
                        # If we found tabs, look for content associated with this tab
                        if team_tabs and idx < len(team_tabs):
                            tab = team_tabs[idx]
                            # Try to find the content section for this tab
                            # It could be a sibling or a child element
                            team_section = tab.find_next(['div', 'section', 'ul'])
                        
                        # If no tab content found, look for columns or sections
                        if not team_section:
                            # Look for sections with team abbreviation or team name
                            team_section = parent.find(['div', 'section', 'ul'], string=lambda s: s and (team_abbr in s or team_name in s))
                        
                        # If still not found, just search broadly around the matchup header
                        if not team_section:
                            # Look in surrounding elements for player listings
                            container = parent
                            for _ in range(3):  # Check a few levels up
                                team_section = container.find(['div', 'section', 'ul'], class_=lambda c: c and ('team' in str(c).lower() or team_abbr.lower() in str(c).lower()))
                                if team_section:
                                    break
                                container = container.parent
                        
                        # If we can't find a specific section, use the parent container
                        if not team_section:
                            team_section = parent
                        
                        # Now find player elements within the team section
                        # Based on the screenshot, players have positions like SG, SF, etc.
                        player_elements = []
                        
                        # Look for elements containing position abbreviations
                        for position in POSITIONS.keys():
                            pos_elements = team_section.find_all(['span', 'div', 'p'], string=lambda s: s and s.strip() == position)
                            for pos_elem in pos_elements:
                                # Each position element should have a parent that represents the player
                                player_parent = pos_elem.parent
                                if player_parent and player_parent not in player_elements:
                                    player_elements.append(player_parent)
                        
                        # If no player elements found by position, look for player name patterns
                        if not player_elements:
                            # Look for name elements that match the pattern of a name (First Last)
                            name_elements = team_section.find_all(['span', 'div', 'p', 'a'], string=lambda s: s and re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+$', s.strip()))
                            for name_elem in name_elements:
                                player_parent = name_elem.parent
                                if player_parent and player_parent not in player_elements:
                                    player_elements.append(player_parent)
                        
                        # Process each player element
                        for player_elem in player_elements:
                            try:
                                # Extract player name - look for prominent text that's not a position
                                name_candidates = player_elem.find_all(['span', 'div', 'p', 'a'])
                                player_name = None
                                
                                for candidate in name_candidates:
                                    candidate_text = candidate.text.strip()
                                    # Skip position abbreviations
                                    if candidate_text in POSITIONS.keys():
                                        continue
                                    # Skip empty or very short text
                                    if not candidate_text or len(candidate_text) < 3:
                                        continue
                                    # This could be the player name
                                    player_name = candidate_text
                                    break
                                
                                if not player_name:
                                    # If we still don't have a name, use all text in the element
                                    full_text = player_elem.text.strip()
                                    # Remove position abbreviations
                                    for pos in POSITIONS.keys():
                                        full_text = full_text.replace(pos, '').strip()
                                    player_name = full_text
                                
                                # Extract position
                                position_element = player_elem.find(['span', 'div', 'p'], string=lambda s: s and s.strip() in POSITIONS.keys())
                                position = position_element.text.strip() if position_element else ""
                                
                                # Create player data (all players in today's lineups are active)
                                player_data = {
                                    'name': player_name,
                                    'team': team_name,
                                    'position': position,
                                    'status': 'active',
                                    'status_detail': 'Starting Lineup'
                                }
                                
                                players.append(player_data)
                                logger.info(f"Found player: {player_name} ({position}) - {team_name}")
                            
                            except Exception as e:
                                logger.error(f"Error processing player element: {str(e)}")
                                continue
                
                except Exception as e:
                    logger.error(f"Error processing matchup header: {str(e)}")
                    continue
        
        except Exception as e:
            logger.error(f"Error scraping lineups: {str(e)}")
        
        # If no players found with the main approach, try a more direct approach
        if not players:
            logger.info("No players found with primary approach. Trying secondary approach...")
            
            try:
                # Look directly for position markers and player names near them
                position_elements = []
                for position in POSITIONS.keys():
                    pos_elems = soup.find_all(['span', 'div', 'p'], string=lambda s: s and s.strip() == position)
                    position_elements.extend(pos_elems)
                
                logger.info(f"Found {len(position_elements)} position elements")
                
                for pos_elem in position_elements:
                    try:
                        position = pos_elem.text.strip()
                        
                        # Look for team context - position elements should be under a team section
                        team_context = None
                        element = pos_elem
                        for _ in range(5):  # Look up to 5 levels up
                            if not element or element.name == 'html':
                                break
                            
                            # Check if this element has a team abbreviation
                            element_text = element.text
                            for abbr in TEAM_ABBR_TO_NAME.keys():
                                if abbr in element_text:
                                    team_context = abbr
                                    break
                            
                            if team_context:
                                break
                            
                            element = element.parent
                        
                        if not team_context:
                            # If we can't find a team context, look for VS patterns
                            vs_elements = soup.find_all(['div', 'h2', 'h3', 'span'], string=lambda s: s and 'VS' in s)
                            for vs_elem in vs_elements:
                                if pos_elem in vs_elem.descendants or vs_elem in pos_elem.descendants:
                                    teams_match = re.search(r'([A-Z]{3})\s+VS\s+([A-Z]{3})', vs_elem.text)
                                    if teams_match:
                                        # Determine which team by checking proximity
                                        away_abbr = teams_match.group(1)
                                        home_abbr = teams_match.group(2)
                                        away_text = vs_elem.text.split('VS')[0]
                                        home_text = vs_elem.text.split('VS')[1]
                                        
                                        # Check which side our position element is closer to
                                        if position in away_text:
                                            team_context = away_abbr
                                        elif position in home_text:
                                            team_context = home_abbr
                                        else:
                                            # Default to away team
                                            team_context = away_abbr
                                        
                                        break
                        
                        if not team_context:
                            logger.warning(f"Could not determine team context for position {position}")
                            continue
                        
                        team_name = self._normalize_team_name(team_context)
                        
                        # Find the player name - check sibling elements
                        player_name = None
                        
                        # First check if there's a sibling with a name
                        siblings = pos_elem.parent.contents
                        for sibling in siblings:
                            if hasattr(sibling, 'text') and sibling != pos_elem and sibling.text.strip() and sibling.text.strip() not in POSITIONS.keys():
                                player_name = sibling.text.strip()
                                break
                        
                        # If no name found in siblings, check parent's text
                        if not player_name:
                            parent_text = pos_elem.parent.text.strip()
                            # Remove position abbreviation
                            for pos in POSITIONS.keys():
                                parent_text = parent_text.replace(pos, '').strip()
                            player_name = parent_text
                        
                        if player_name:
                            # Create player data
                            player_data = {
                                'name': player_name,
                                'team': team_name,
                                'position': position,
                                'status': 'active',
                                'status_detail': 'Starting Lineup'
                            }
                            
                            players.append(player_data)
                            logger.info(f"Found player (secondary approach): {player_name} ({position}) - {team_name}")
                    
                    except Exception as e:
                        logger.error(f"Error in secondary approach: {str(e)}")
                        continue
            
            except Exception as e:
                logger.error(f"Error in secondary scraping approach: {str(e)}")
        
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