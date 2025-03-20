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


class NBAPlayersScraper:
    """
    Scrapes NBA player information and statuses from ESPN and NBA.com.
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
        
        # URLs for player data
        self.espn_injuries_url = "https://www.espn.com/nba/injuries"
        self.nba_lineups_url = "https://www.nba.com/players/todays-lineups"
    
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
    
    def _normalize_player_status(self, status_text: str) -> str:
        """
        Normalize player status to a standard format.
        
        Args:
            status_text: Player status text
            
        Returns:
            Normalized status
        """
        if not status_text:
            return 'unknown'
        
        status_lower = status_text.lower().strip()
        
        # Match to standard status categories
        for status_category, keywords in PLAYER_STATUSES.items():
            if any(keyword in status_lower for keyword in keywords):
                return status_category
        
        # Additional patterns for injury statuses
        if re.search(r'(injury|injured|hurt)', status_lower):
            return 'out'
        
        if re.search(r'(expected to play|likely)', status_lower):
            return 'active'
        
        # Default to unknown if no match found
        return 'unknown'
    
    def scrape_espn_injury_report(self) -> List[Dict[str, Any]]:
        """
        Scrape NBA injury report from ESPN.
        
        Returns:
            List of dictionaries containing player information
        """
        html_content = self._make_request(self.espn_injuries_url)
        if not html_content:
            return []
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        players = []
        
        # Find all team sections
        team_sections = soup.find_all('div', class_='ResponsiveTable')
        if not team_sections:
            team_sections = soup.find_all('div', class_=lambda c: c and ('TeamSection' in c or 'Team' in c or 'team' in c.lower()))
        
        for section in team_sections:
            try:
                # Try to extract team name
                team_header = section.find_previous(['h1', 'h2', 'h3', 'div'], class_=lambda c: c and ('title' in c.lower() or 'header' in c.lower() or 'TeamName' in c))
                
                if not team_header or not team_header.text.strip():
                    # Try alternative ways to get the team name
                    team_header = section.find(['div', 'span'], class_=lambda c: c and ('team' in c.lower()))
                    
                    # If still not found, look at the section id or class
                    if not team_header:
                        section_id = section.get('id', '')
                        if section_id and any(team_abbr.lower() in section_id.lower() for team_abbr in TEAM_ABBR_TO_NAME.keys()):
                            for team_abbr in TEAM_ABBR_TO_NAME.keys():
                                if team_abbr.lower() in section_id.lower():
                                    team_name = TEAM_ABBR_TO_NAME[team_abbr]
                                    break
                            else:
                                logger.warning("Could not find team name in injury report section")
                                continue
                        else:
                            logger.warning("Could not find team name in injury report section")
                            continue
                else:
                    team_name_text = team_header.text.strip()
                    # Remove common suffixes like "Injuries" from team name
                    team_name_text = re.sub(r'\s+(injuries|injury report)$', '', team_name_text, flags=re.IGNORECASE)
                    team_name = self._normalize_team_name(team_name_text)
                
                if team_name not in NBA_TEAMS:
                    continue
                
                # Find player rows - look for table rows or list items
                player_rows = section.find_all('tr')
                if not player_rows:
                    player_rows = section.find_all(['li', 'div'], class_=lambda c: c and ('player' in c.lower() or 'row' in c.lower() or 'Row' in c))
                
                for row in player_rows:
                    try:
                        # Skip header rows
                        if row.find('th'):
                            continue
                        
                        # Try to find player name and status
                        cells = row.find_all('td')
                        
                        if cells:
                            # Table format
                            name_cell = cells[0] if len(cells) > 0 else None
                            player_name = name_cell.text.strip() if name_cell else ""
                            
                            # Extract position (if available)
                            position = cells[1].text.strip() if len(cells) > 1 else ""
                            
                            # Extract status
                            status_cell = cells[2] if len(cells) > 2 else None
                            status_text = status_cell.text.strip() if status_cell else "Unknown"
                        else:
                            # Non-table format
                            name_elem = row.find(['span', 'div', 'a'], class_=lambda c: c and ('name' in c.lower() or 'player' in c.lower() or 'Name' in c))
                            status_elem = row.find(['span', 'div', 'p'], class_=lambda c: c and ('status' in c.lower() or 'injury' in c.lower() or 'Status' in c or 'Injury' in c))
                            
                            if name_elem:
                                player_name = name_elem.text.strip()
                                status_text = status_elem.text.strip() if status_elem else "Unknown"
                                position = ""  # Position might not be available
                            else:
                                # Try to extract from raw text
                                row_text = row.get_text().strip()
                                match = re.search(r'^([^:]+):\s*(.+)$', row_text)
                                if match:
                                    player_name = match.group(1).strip()
                                    status_text = match.group(2).strip()
                                    position = ""
                                else:
                                    continue  # Skip if can't parse
                        
                        # Skip empty player names
                        if not player_name:
                            continue
                        
                        # Normalize status
                        status = self._normalize_player_status(status_text)
                        
                        # Create player data
                        player_data = {
                            'name': player_name,
                            'team': team_name,
                            'position': position,
                            'status': status,
                            'status_detail': status_text
                        }
                        
                        players.append(player_data)
                        
                    except Exception as e:
                        logger.error(f"Error parsing player row: {str(e)}")
                        continue
                
            except Exception as e:
                logger.error(f"Error parsing team section: {str(e)}")
                continue
        
        logger.info(f"Scraped {len(players)} players from ESPN injury report")
        return players
    
    def scrape_nba_lineups(self) -> List[Dict[str, Any]]:
        """
        Scrape NBA lineups from NBA.com's Today's Lineups page.
        
        Returns:
            List of dictionaries containing player information
        """
        html_content = self._make_request(self.nba_lineups_url)
        if not html_content:
            return []
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        players = []
        
        # Try to find game containers
        game_containers = soup.find_all('div', class_=lambda c: c and ('game' in c.lower() or 'Game' in c or 'matchup' in c.lower() or 'Matchup' in c))
        
        if not game_containers:
            logger.warning("Could not find game containers on NBA lineups page")
            return players
        
        for game_container in game_containers:
            try:
                # Find team names in the game container
                team_elements = game_container.find_all(['h2', 'h3', 'div', 'span'], class_=lambda c: c and ('team' in c.lower() or 'TeamName' in c))
                
                if len(team_elements) < 2:
                    # Try alternative approach
                    team_elements = game_container.find_all(['h2', 'h3', 'div', 'span'], string=lambda s: s and any(team in s.lower() for team_name, team_data in NBA_TEAMS.items() for team in team_data["variants"]))
                
                if len(team_elements) < 2:
                    logger.warning("Couldn't find both teams in game container")
                    continue
                
                # Get team names
                team_names = [self._normalize_team_name(team.text.strip()) for team in team_elements[:2]]
                
                # Find player sections for each team
                player_sections = game_container.find_all('div', class_=lambda c: c and ('players' in c.lower() or 'roster' in c.lower() or 'lineup' in c.lower()))
                
                if len(player_sections) < 2:
                    logger.warning("Couldn't find player sections for both teams")
                    continue
                
                # Process up to 2 teams (home and away)
                for idx, (team_name, player_section) in enumerate(zip(team_names[:2], player_sections[:2])):
                    # Find all player elements
                    player_elements = player_section.find_all(['div', 'li', 'span'], class_=lambda c: c and ('player' in c.lower() or 'Player' in c or 'name' in c.lower()))
                    
                    for player_element in player_elements:
                        try:
                            # Extract player name
                            player_name_elem = player_element.find(['span', 'div', 'a'], class_=lambda c: c and ('name' in c.lower() or 'Name' in c)) or player_element
                            
                            player_name = player_name_elem.text.strip()
                            if not player_name:
                                continue
                            
                            # Look for status indicator
                            status_elem = player_element.find(['span', 'div'], class_=lambda c: c and ('status' in c.lower() or 'Status' in c or 'injury' in c.lower() or 'Injury' in c))
                            
                            status_text = status_elem.text.strip() if status_elem else "Active"  # Assume active if no status found
                            status = self._normalize_player_status(status_text)
                            
                            # Create player data
                            player_data = {
                                'name': player_name,
                                'team': team_name,
                                'position': "",  # Position might not be available
                                'status': status,
                                'status_detail': status_text
                            }
                            
                            players.append(player_data)
                            
                        except Exception as e:
                            logger.error(f"Error parsing player element: {str(e)}")
                            continue
            
            except Exception as e:
                logger.error(f"Error parsing game container: {str(e)}")
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
    
    def update_player_statuses_from_injuries(self, injury_players: List[Dict[str, Any]]) -> int:
        """
        Update player statuses based on injury reports.
        
        Args:
            injury_players: List of player dictionaries with injury information
            
        Returns:
            Number of player statuses updated
        """
        updated_count = 0
        
        # Create a lookup dictionary for faster matching
        player_status_lookup = {}
        for player in injury_players:
            key = f"{player['name'].lower()}_{player['team'].lower()}"
            player_status_lookup[key] = player['status']
        
        # Get all players from the database
        for team_name in NBA_TEAMS.keys():
            try:
                team_players = self.db.get_players_by_team(team_name)
                
                for player in team_players:
                    player_key = f"{player['name'].lower()}_{player['team'].lower()}"
                    
                    if player_key in player_status_lookup:
                        # Update player status
                        new_status = player_status_lookup[player_key]
                        
                        if player['status'] != new_status:
                            self.db.update_player_status(player['player_id'], new_status)
                            logger.info(f"Updated {player['name']} status to {new_status}")
                            updated_count += 1
                
            except Exception as e:
                logger.error(f"Error updating players for team {team_name}: {str(e)}")
                continue
        
        logger.info(f"Updated {updated_count} player statuses from injury reports")
        return updated_count
    
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
    
    def scrape_and_save_players(self) -> Tuple[int, int]:
        """
        Scrape players from multiple sources and save them to the database.
        
        Returns:
            Tuple of (players_saved, statuses_updated)
        """
        # Initialize database if not already connected
        self.db.initialize_database()
        
        # Get player data from both sources
        espn_players = self.scrape_espn_injury_report()
        nba_players = self.scrape_nba_lineups()
        
        # Combine player lists (prioritizing ESPN data for duplicates)
        all_players = {}
        
        # Add NBA.com players first
        for player in nba_players:
            player_key = f"{player['name'].lower()}_{player['team'].lower()}"
            all_players[player_key] = player
        
        # Add or update with ESPN players
        for player in espn_players:
            player_key = f"{player['name'].lower()}_{player['team'].lower()}"
            all_players[player_key] = player
        
        # Convert back to list
        combined_players = list(all_players.values())
        
        # Save players to database
        saved_count = self.save_players_to_database(combined_players)
        
        # Update player statuses from injury reports
        updated_count = self.update_player_statuses_from_injuries(espn_players)
        
        return (saved_count, updated_count)


# Example usage
if __name__ == "__main__":
    # Set up scraper
    scraper = NBAPlayersScraper()
    
    # Scrape and save player information
    players_saved, statuses_updated = scraper.scrape_and_save_players()
    
    print(f"Saved {players_saved} players and updated {statuses_updated} player statuses")