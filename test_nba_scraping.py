#!/usr/bin/env python3
"""
NBA Sentiment Predictor - Scraping Tester

This script tests the scraping capabilities of the NBA Scraper modules.
It displays today's games and the lineups for each team playing today.
Uses Selenium to handle JavaScript-rendered content.
"""

import os
import sys
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from prettytable import PrettyTable

from data.database import Database
from scrapers.nba_games import NBAGamesScraper
from scrapers.nba_players import NBAPlayersScraper

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_scraping')

def test_database_connection():
    """Test database connection and initialization."""
    logger.info("Testing database connection...")
    
    db = Database()
    try:
        db.initialize_database()
        logger.info("✓ Database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Database initialization failed: {e}")
        return False

def scrape_todays_games():
    """Scrape and return today's NBA games."""
    logger.info("Scraping today's NBA games...")
    
    db = Database()
    db.initialize_database()
    
    game_scraper = NBAGamesScraper(db=db)
    
    # Scrape games for today
    games = game_scraper.scrape_nba_schedule(days_ahead=0)
    
    if not games:
        logger.warning("No games found for today")
        return []
    
    # Save games to database
    game_scraper.save_games_to_database(games)
    
    logger.info(f"✓ Found {len(games)} games scheduled for today")
    return games

def scrape_todays_lineups():
    """Scrape and return today's NBA lineups."""
    logger.info("Scraping today's NBA lineups...")
    
    db = Database()
    db.initialize_database()
    
    player_scraper = NBAPlayersScraper(db=db)
    
    # Scrape today's lineups
    players = player_scraper.scrape_todays_lineups()
    
    if not players:
        logger.warning("No lineups found for today")
        return []
    
    # Save players to database
    player_scraper.save_players_to_database(players)
    
    logger.info(f"✓ Found {len(players)} players in today's lineups")
    return players

def get_team_lineups(team_name: str) -> List[Dict[str, Any]]:
    """Get lineup for a specific team from the database."""
    db = Database()
    
    # Get players for the team
    return db.get_players_by_team(team_name)

def test_scraping_functionality():
    """Test the entire scraping functionality."""
    logger.info("=" * 50)
    logger.info("Testing NBA Scraping Functionality")
    logger.info("=" * 50)
    
    # Step 1: Test database
    if not test_database_connection():
        logger.error("Database test failed. Exiting...")
        return False
    
    # Step 2: Scrape today's games
    games = scrape_todays_games()
    
    # Step 3: Scrape today's lineups
    players = scrape_todays_lineups()
    
    # Step 4: Display the results
    display_results(games, players)
    
    return True

def display_results(games: List[Dict[str, Any]], players: List[Dict[str, Any]]):
    """Display the scraping results in a readable format."""
    logger.info("=" * 50)
    logger.info(f"RESULTS - {datetime.now().strftime('%A, %B %d, %Y')}")
    logger.info("=" * 50)
    
    # Display games table
    if games:
        games_table = PrettyTable()
        games_table.field_names = ["Time", "Away Team", "Home Team", "Venue"]
        
        for game in games:
            games_table.add_row([
                game['game_time'],
                game['away_team'],
                game['home_team'],
                game.get('venue', 'Unknown')
            ])
        
        print("\nTODAY'S NBA GAMES:")
        print(games_table)
    else:
        print("\nNo games scheduled for today.")
    
    # Collect and display lineups by team
    if players:
        # Group players by team
        team_players = {}
        for player in players:
            team = player['team']
            if team not in team_players:
                team_players[team] = []
            team_players[team].append(player)
        
        # Get teams playing today from the games
        today_teams = set()
        for game in games:
            today_teams.add(game['home_team'])
            today_teams.add(game['away_team'])
        
        # Retrieve additional players from database for each team
        db = Database()
        for team in today_teams:
            if team not in team_players:
                team_players[team] = []
            
            # Get players from database
            db_players = get_team_lineups(team)
            
            # Add players not already in the list
            existing_player_names = [p['name'] for p in team_players[team]]
            for player in db_players:
                if player['name'] not in existing_player_names:
                    team_players[team].append(player)
        
        # Display lineups for each team playing today
        print("\nTODAY'S LINEUPS:")
        
        for team in sorted(today_teams):
            lineup = team_players.get(team, [])
            
            if lineup:
                print(f"\n{team} LINEUP:")
                
                lineup_table = PrettyTable()
                lineup_table.field_names = ["Name", "Position", "Status"]
                
                for player in sorted(lineup, key=lambda p: p.get('name', '')):
                    lineup_table.add_row([
                        player.get('name', 'Unknown'),
                        player.get('position', ''),
                        player.get('status', 'Unknown')
                    ])
                
                print(lineup_table)
            else:
                print(f"\n{team} LINEUP: No lineup information available")
    else:
        print("\nNo lineup information available.")

def main():
    """Main function to run the test."""
    try:
        test_scraping_functionality()
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    # Install PrettyTable if not already installed
    try:
        import prettytable
    except ImportError:
        print("Installing required package: prettytable")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "prettytable"])
    
    # Run the test
    sys.exit(main())