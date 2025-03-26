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
from pprint import pprint

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


def print_separator(title: str = None) -> None:
    """Print a separator line with optional title."""
    if title:
        print("\n" + "=" * 30 + f" {title} " + "=" * 30)
    else:
        print("\n" + "=" * 80)


def print_game_info(game: Dict[str, Any]) -> None:
    """Print game information in a readable format."""
    print(f"{game['away_team']} ({game.get('away_record', '')}) @ {game['home_team']} ({game.get('home_record', '')})")
    print(f"Game ID: {game['game_id']}")
    print(f"Date: {game['game_date']} | Time: {game['game_time']}")
    
    # Print venue if available
    if game.get('venue') and game['venue'] != "Unknown":
        print(f"Venue: {game['venue']}")
    
    # Print referees if available
    if game.get('referees'):
        print(f"Referees: {', '.join(game['referees'])}")
    
    # Print odds if available
    if game.get('odds'):
        odds = game['odds']
        if odds.get('line'):
            print(f"Line: {odds['line']}")
        if odds.get('spread'):
            print(f"Spread: {odds['spread']}")
        if odds.get('over_under'):
            print(f"Over/Under: {odds['over_under']}")


def print_lineup(team_name: str, lineup: List[Dict[str, Any]]) -> None:
    """Print team lineup in a readable format."""
    print(f"\n{team_name} Starting Lineup:")
    print("------------------------------")
    if not lineup:
        print("No lineup data available")
        return
    
    # Sort lineup by position
    position_order = {"PG": 1, "SG": 2, "SF": 3, "PF": 4, "C": 5}
    sorted_lineup = sorted(
        lineup, 
        key=lambda x: position_order.get(x.get('position', ''), 99)
    )
    
    for player in sorted_lineup:
        status_indicator = ""
        if player.get('status') != 'active':
            status_indicator = f" ({player['status'].upper()})"
        
        print(f"{player.get('position', 'N/A'):2} | {player.get('name', 'Unknown')}{status_indicator}")


def print_injuries(team_name: str, injuries: List[Dict[str, Any]]) -> None:
    """Print team injuries in a readable format."""
    if not injuries:
        return
    
    print(f"\n{team_name} Injury Report:")
    print("---------------------------")
    for player in injuries:
        print(f"{player.get('position', 'N/A'):2} | {player.get('name', 'Unknown')} - {player.get('status', 'Unknown').upper()}")


def main():
    """Main function to test the scrapers."""
    print_separator("NBA SCRAPER TEST")
    print("Testing NBA API and Rotowire scrapers for today's games and lineups")
    
    # Create a temporary database for testing
    db = Database(db_path=":memory:")
    db.initialize_database()
    
    # Initialize scrapers
    nba_scraper = NBAApiScraper(db=db)
    rotowire_scraper = RotowireScraper(db=db)
    
    # Scrape NBA API games
    print_separator("NBA API GAMES")
    try:
        nba_games = nba_scraper.scrape_nba_schedule(days_ahead=0)
        print(f"Found {len(nba_games)} games from NBA API")
        
        # Print each game
        for i, game in enumerate(nba_games):
            print_separator(f"Game {i+1}")
            print_game_info(game)
    except Exception as e:
        logger.error(f"Error scraping NBA API: {str(e)}")
        print(f"Error scraping NBA API: {str(e)}")
    
    # Scrape Rotowire lineups
    print_separator("ROTOWIRE LINEUPS")
    try:
        rotowire_games = rotowire_scraper.scrape_rotowire_lineups(date_type='today')
        print(f"Found {len(rotowire_games)} games on Rotowire")
        
        # Print each game with lineups
        for i, game in enumerate(rotowire_games):
            print_separator(f"Game {i+1}")
            print_game_info(game)
            
            # Print lineups
            print_lineup(game['away_team'], game['away_lineup'])
            print_lineup(game['home_team'], game['home_lineup'])
            
            # Print injuries
            print_injuries(game['away_team'], game['away_injuries'])
            print_injuries(game['home_team'], game['home_injuries'])
    except Exception as e:
        logger.error(f"Error scraping Rotowire: {str(e)}")
        print(f"Error scraping Rotowire: {str(e)}")
    
    # Print summary
    print_separator("SUMMARY")
    print(f"NBA API games: {len(nba_games) if 'nba_games' in locals() else 0}")
    print(f"Rotowire games: {len(rotowire_games) if 'rotowire_games' in locals() else 0}")
    
    # Check if game counts match
    if 'nba_games' in locals() and 'rotowire_games' in locals():
        if len(nba_games) != len(rotowire_games):
            print("\nWARNING: Game count mismatch between NBA API and Rotowire!")
            
            # Try to identify mismatched games
            nba_game_ids = {game['game_id'] for game in nba_games}
            rotowire_game_ids = {game['game_id'] for game in rotowire_games}
            
            missing_from_nba = rotowire_game_ids - nba_game_ids
            missing_from_rotowire = nba_game_ids - rotowire_game_ids
            
            if missing_from_nba:
                print("\nGames on Rotowire but not found in NBA API:")
                for game_id in missing_from_nba:
                    game = next((g for g in rotowire_games if g['game_id'] == game_id), None)
                    if game:
                        print(f"- {game['away_team']} @ {game['home_team']} ({game['game_date']})")
            
            if missing_from_rotowire:
                print("\nGames from NBA API but not on Rotowire:")
                for game_id in missing_from_rotowire:
                    game = next((g for g in nba_games if g['game_id'] == game_id), None)
                    if game:
                        print(f"- {game['away_team']} @ {game['home_team']} ({game['game_date']})")
        else:
            print("\nSUCCESS: Game counts match between NBA API and Rotowire!")
            
    # Bonus: Try NBA API's combined scraper function
    print_separator("NBA API COMBINED SCRAPER (GAMES + LINEUPS)")
    try:
        # The NBAApiScraper also has functionality to scrape lineups
        player_data = nba_scraper.scrape_todays_lineups()
        print(f"Found {len(player_data)} players from NBA API's lineup scraper")
        
        # Display sample of first few players
        if player_data:
            print("\nSample players:")
            for i, player in enumerate(player_data[:5]):
                print(f"{i+1}. {player.get('name', 'Unknown')} ({player.get('team', 'Unknown')}) - {player.get('position', 'N/A')} - {player.get('status', 'Unknown')}")
            if len(player_data) > 5:
                print(f"... and {len(player_data) - 5} more players")
    except Exception as e:
        logger.error(f"Error using NBA API's lineup scraper: {str(e)}")
        print(f"Error using NBA API's lineup scraper: {str(e)}")


if __name__ == "__main__":
    main()