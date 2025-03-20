import os
import re
import time
import logging
import datetime
from typing import List, Dict, Any, Optional, Tuple, Set
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

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
    Scrapes NBA player information and lineups exclusively from NBA.com/players/todays-lineups.
    """
    
    def __init__(self, db: Optional[Database] = None):
        """
        Initialize the NBA players scraper.
        
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
        
        # NBA.com lineups URL (the only source we'll use)
        self.lineups_url = "https://www.nba.com/players/todays-lineups"
    
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
        Scrape NBA lineups from NBA.com's Today's Lineups page.
        
        Returns:
            List of dictionaries containing player information
        """
        html_content = self._make_request(self.lineups_url)
        if not html_content:
            logger.error("Failed to retrieve NBA.com lineups page")
            return []
        
        soup = BeautifulSoup(html_content, 'html.parser')
        players = []
        
        # Based on the screenshot, we need to look for game matchups and player listings
        # The layout shows matchups like "NYK VS CHA", "BKN VS IND", etc.
        try:
            # Find all game matchup headers
            matchup_headers = soup.find_all(['h2', 'h3', 'div'], string=lambda s: s and re.search(r'[A-Z]{3}\s+VS\s+[A-Z]{3}', s))
            
            # If no direct headers, look for elements with matchup class or structure
            if not matchup_headers:
                matchup_headers = soup.find_all(['div', 'h2', 'h3'], class_=lambda c: c and ('matchup' in c.lower() or 'Matchup' in c))
            
            # If still not found, look for the exact structure in the screenshot
            if not matchup_headers:
                # Look for elements containing team abbreviations separated by VS
                matchup_headers = []
                for element in soup.find_all(['div', 'h2', 'h3', 'span']):
                    if element.text and re.search(r'[A-Z]{3}\s+VS\s+[A-Z]{3}', element.text.strip()):
                        matchup_headers.append(element)
            
            logger.info(f"Found {len(matchup_headers)} game matchups")
            
            for matchup in matchup_headers:
                try:
                    # Extract team abbreviations from matchup heading
                    matchup_text = matchup.text.strip()
                    teams_match = re.search(r'([A-Z]{3})\s+VS\s+([A-Z]{3})', matchup_text)
                    
                    if teams_match:
                        away_abbr = teams_match.group(1)
                        home_abbr = teams_match.group(2)
                        
                        away_team = self._normalize_team_name(away_abbr)
                        home_team = self._normalize_team_name(home_abbr)
                        
                        logger.info(f"Processing matchup: {away_team} @ {home_team}")
                        
                        # Find the team sections for this matchup
                        # Based on the screenshot, each matchup has two team sections with players
                        
                        # Find team container elements near the matchup
                        team_containers = []
                        
                        # First look in the parent container
                        parent = matchup.parent
                        team_elements = parent.find_all(['div', 'section'], class_=lambda c: c and ('team' in str(c).lower() or 'Team' in str(c)))
                        
                        if team_elements and len(team_elements) >= 2:
                            team_containers = team_elements[:2]  # Away and home team containers
                        else:
                            # Try looking at siblings or nearby elements
                            # Check if there are tab elements or sections labeled with the team abbreviations
                            away_element = parent.find(['div', 'section', 'tab'], string=lambda s: s and away_abbr in s)
                            home_element = parent.find(['div', 'section', 'tab'], string=lambda s: s and home_abbr in s)
                            
                            if away_element and home_element:
                                team_containers = [away_element, home_element]
                            else:
                                # Direct approach: look for the exact team abbreviations as standalone headers
                                next_element = matchup.find_next_sibling()
                                while next_element and len(team_containers) < 2:
                                    if next_element.text.strip() in [away_abbr, home_abbr]:
                                        team_containers.append(next_element)
                                    next_element = next_element.find_next_sibling()
                        
                        # If still not found, use direct reference to the screenshot's exact structure
                        if not team_containers or len(team_containers) < 2:
                            # Look for container elements with team tables
                            team_tables = parent.find_all('table')
                            if team_tables and len(team_tables) >= 2:
                                # Create containers from the tables
                                team_containers = team_tables[:2]
                        
                        # Process team containers
                        for idx, container in enumerate(team_containers[:2]):  # Process up to 2 teams
                            team_name = away_team if idx == 0 else home_team
                            
                            # Find player elements in this team container
                            # Based on the screenshot, players are listed with position (SG, SF, etc.)
                            
                            # Try to find player rows or elements
                            player_elements = container.find_all(['tr', 'div', 'li'], class_=lambda c: c and ('player' in str(c).lower() or 'Player' in str(c)))
                            
                            # If no player elements with class, look for elements containing position abbreviations
                            if not player_elements:
                                for pos in POSITIONS.keys():
                                    pos_elements = container.find_all(['span', 'div'], string=lambda s: s and s.strip() == pos)
                                    for pos_elem in pos_elements:
                                        # Find the parent element that represents the player
                                        player_parent = pos_elem.parent
                                        if player_parent and player_parent not in player_elements:
                                            player_elements.append(player_parent)
                            
                            # Direct approach: Look for orange blocks that indicate players (from screenshot)
                            if not player_elements:
                                orange_elements = container.find_all(['div', 'span'], style=lambda s: s and 'background-color: #f60' in s.lower())
                                for elem in orange_elements:
                                    player_parent = elem.parent
                                    if player_parent and player_parent not in player_elements:
                                        player_elements.append(player_parent)
                            
                            # Process each player element
                            for player_elem in player_elements:
                                try:
                                    # Extract player name
                                    # Based on the screenshot format, player names are prominent
                                    name_element = player_elem.find(['span', 'div', 'a'], class_=lambda c: c and ('name' in str(c).lower() or 'Name' in str(c)))
                                    
                                    # If no specific class, try to find by context (positioning near position marker)
                                    if not name_element:
                                        # Look for text that doesn't match position abbreviations
                                        for text_elem in player_elem.find_all(['span', 'div', 'a']):
                                            if text_elem.text.strip() and text_elem.text.strip() not in POSITIONS.keys():
                                                name_element = text_elem
                                                break
                                    
                                    # If still not found, check directly for a name pattern
                                    if not name_element:
                                        # Look for elements that match name patterns (First Last)
                                        name_pattern = re.compile(r'[A-Z][a-z]+\s+[A-Z][a-z]+')
                                        for text_elem in player_elem.find_all(['span', 'div', 'a']):
                                            if text_elem.text and name_pattern.match(text_elem.text.strip()):
                                                name_element = text_elem
                                                break
                                    
                                    # Extract position
                                    position_element = player_elem.find(['span', 'div'], string=lambda s: s and s.strip() in POSITIONS.keys())
                                    
                                    # If we have a name, create a player record
                                    if name_element:
                                        player_name = name_element.text.strip()
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
                    logger.error(f"Error processing matchup {matchup_text}: {str(e)}")
                    continue
        
        except Exception as e:
            logger.error(f"Error scraping NBA.com lineups: {str(e)}")
        
        # Last resort: Try to directly parse based on the exact structure in the screenshot
        if not players:
            logger.info("Trying direct structure parsing based on the screenshot")
            
            try:
                # Look for team abbreviation headings (NYK, CHA, etc.)
                abbr_headers = soup.find_all(['div', 'h3', 'span'], string=lambda s: s and s.strip() in TEAM_ABBR_TO_NAME.keys())
                
                for abbr_header in abbr_headers:
                    team_abbr = abbr_header.text.strip()
                    team_name = self._normalize_team_name(team_abbr)
                    
                    # Find nearby player elements
                    parent = abbr_header.parent
                    
                    # Look for orange player indicators (from screenshot)
                    player_indicators = parent.find_all(['div', 'span'], style=lambda s: s and 'background-color' in s.lower())
                    
                    for indicator in player_indicators:
                        try:
                            # Find the player name near this indicator
                            # Check nearby text elements for names
                            player_parent = indicator.parent
                            
                            # Look for text that might be a name
                            name_elements = player_parent.find_all(['span', 'div'], string=lambda s: s and len(s.strip()) > 3 and s.strip() not in POSITIONS.keys())
                            
                            if name_elements:
                                player_name = name_elements[0].text.strip()
                                
                                # Look for position marker
                                position_element = player_parent.find(['span', 'div'], string=lambda s: s and s.strip() in POSITIONS.keys())
                                position = position_element.text.strip() if position_element else ""
                                
                                # Create player record
                                player_data = {
                                    'name': player_name,
                                    'team': team_name,
                                    'position': position,
                                    'status': 'active',
                                    'status_detail': 'Starting Lineup'
                                }
                                
                                players.append(player_data)
                                logger.info(f"Found player (direct parsing): {player_name} ({position}) - {team_name}")
                        
                        except Exception as e:
                            logger.error(f"Error in direct parsing: {str(e)}")
                            continue
            
            except Exception as e:
                logger.error(f"Error with direct structure parsing: {str(e)}")
        
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