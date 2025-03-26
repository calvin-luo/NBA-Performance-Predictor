#!/usr/bin/env python3
"""
NBA Sentiment Predictor - Scraper Testing Script

This script tests the NBA API scraper and Rotowire scraper by:
1. Fetching today's NBA games from the NBA API
2. Fetching today's NBA lineups from Rotowire.com
3. Displaying the matched data in a readable format
"""

import os
import sys
import logging
from typing import Dict, List, Any

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.database import Database
from scrapers.nba_api_scraper import NBAApiScraper
from scrapers.rotowire_scraper import RotowireScraper

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('scrape_tester')


def print_lineup(team_name: str, lineup: List[Dict[str, Any]]) -> None:
    """Print team lineup in a simple format."""
    print(f"Lineup {team_name}:")
    
    if not lineup:
        print("  No lineup data available")
        return
    
    # Sort lineup by position
    position_order = {"PG": 1, "SG": 2, "SF": 3, "PF": 4, "C": 5}
    sorted_lineup = sorted(
        lineup, 
        key=lambda x: position_order.get(x.get('position', ''), 99)
    )
    
    # Print player names with positions
    for player in sorted_lineup:
        status = "" if player.get('status') == 'active' else f" ({player['status']})"
        print(f"  {player.get('position', 'N/A')} - {player.get('name', 'Unknown')}{status}")


def main():
    """Main function to test the scrapers."""
    print("\n=== NBA SCRAPER TEST ===")
    print("Testing NBA API and Rotowire scrapers for today's games and lineups")
    
    # Create a temporary database for testing
    db = Database(db_path=":memory:")
    db.initialize_database()
    
    # Initialize scrapers
    nba_scraper = NBAApiScraper(db=db)
    rotowire_scraper = RotowireScraper(db=db)
    
    # Get games from NBA API
    try:
        nba_games = nba_scraper.scrape_nba_schedule(days_ahead=0)
        print(f"\nFound {len(nba_games)} games from NBA API")
    except Exception as e:
        logger.error(f"Error scraping NBA API: {str(e)}")
        print(f"Error scraping NBA API: {str(e)}")
        nba_games = []
    
    # Get lineups from Rotowire
    try:
        rotowire_games = rotowire_scraper.scrape_rotowire_lineups(date_type='today')
        print(f"Found {len(rotowire_games)} games on Rotowire\n")
    except Exception as e:
        logger.error(f"Error scraping Rotowire: {str(e)}")
        print(f"Error scraping Rotowire: {str(e)}")
        rotowire_games = []
    
    # Display games and lineups from Rotowire (which has complete data)
    for i, game in enumerate(rotowire_games):
        print(f"\nGame {i+1}: {game['away_team']} vs {game['home_team']} ({game['game_time']})")
        
        # Print away team lineup
        print_lineup(game['away_team'], game['away_lineup'])
        
        # Print home team lineup
        print_lineup(game['home_team'], game['home_lineup'])
        
        # Print injuries if there are any
        if game.get('away_injuries') and len(game['away_injuries']) > 0:
            print(f"\n{game['away_team']} Injuries:")
            for player in game['away_injuries']:
                print(f"  {player.get('name', 'Unknown')} ({player.get('status', 'unknown')})")
                
        if game.get('home_injuries') and len(game['home_injuries']) > 0:
            print(f"\n{game['home_team']} Injuries:")
            for player in game['home_injuries']:
                print(f"  {player.get('name', 'Unknown')} ({player.get('status', 'unknown')})")
    
    # Check if game counts match
    if nba_games and rotowire_games:
        print("\n=== SUMMARY ===")
        print(f"NBA API games: {len(nba_games)}")
        print(f"Rotowire games: {len(rotowire_games)}")
        
        if len(nba_games) != len(rotowire_games):
            print("\nWARNING: Game count mismatch between NBA API and Rotowire!")
            
            # Identify mismatched games
            nba_game_ids = {game['game_id'] for game in nba_games}
            rotowire_game_ids = {game['game_id'] for game in rotowire_games}
            
            missing_from_nba = rotowire_game_ids - nba_game_ids
            missing_from_rotowire = nba_game_ids - rotowire_game_ids
            
            if missing_from_nba:
                print("\nGames on Rotowire but not found in NBA API:")
                for game_id in missing_from_nba:
                    game = next((g for g in rotowire_games if g['game_id'] == game_id), None)
                    if game:
                        print(f"- {game['away_team']} @ {game['home_team']}")
            
            if missing_from_rotowire:
                print("\nGames from NBA API but not on Rotowire:")
                for game_id in missing_from_rotowire:
                    game = next((g for g in nba_games if g['game_id'] == game_id), None)
                    if game:
                        print(f"- {game['away_team']} @ {game['home_team']}")
        else:
            print("\nSUCCESS: Game counts match between NBA API and Rotowire!")


if __name__ == "__main__":
    main()