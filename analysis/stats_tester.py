#!/usr/bin/env python3
"""
NBA Stats Predictor - Player Stats Testing Script

This script tests the PlayerStatsCollector class by:
1. Fetching today's NBA games and lineups
2. Retrieving historical stats for players in the lineups
3. Displaying the stats in a readable format
"""

import os
import sys
import logging
from typing import Dict, List, Any

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import scrapers and player stats collector
from scrapers.player_scraper import RotowireScraper
from analysis.player_stats import PlayerStatsCollector

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('stats_tester')


def main():
    """Main function to test the player stats collector."""
    print("\n=== NBA PLAYER STATS TEST ===")
    
    # Initialize player stats collector
    stats_collector = PlayerStatsCollector()
    
    # Get games and lineups from Rotowire
    try:
        rotowire = RotowireScraper()
        games = rotowire.scrape_rotowire_lineups(date_type='today')
        print(f"Found {len(games)} games for today")
    except Exception as e:
        logger.error(f"Error scraping Rotowire: {str(e)}")
        print(f"Error scraping Rotowire: {str(e)}")
        games = []
    
    if not games:
        print("No games found. Testing with sample players instead.")
        test_sample_players(stats_collector)
        return
    
    # Process each game
    for game_idx, game in enumerate(games):
        print(f"\n=== Game {game_idx+1}: {game['away_team']} @ {game['home_team']} ===")
        
        # Process both teams
        for team_type, team_name, lineup in [
            ("Away", game['away_team'], game['away_lineup']),
            ("Home", game['home_team'], game['home_lineup'])
        ]:
            print(f"\n--- {team_name} Players ---")
            
            # Process each player in lineup
            for player in lineup:
                player_name = player.get('name')
                if not player_name:
                    continue
                
                print_player_stats(stats_collector, player_name)


def print_player_stats(stats_collector, player_name):
    """Print stats for a single player."""
    # Print player name
    print(f"\n{player_name}")
    
    try:
        # Get player stats
        stats_df = stats_collector.get_player_stats(player_name, num_games=10)
        
        if stats_df is not None and not stats_df.empty:
            # Get most recent game stats (first row)
            recent_game = stats_df.iloc[0]
            
            # Print key stats
            stats_to_show = [
                'FG_PCT', 'TS_PCT', 'PTS_PER_MIN', 'PLUS_MINUS', 
                'OFF_RATING', 'DEF_RATING', 'GAME_SCORE', 'USG_RATE', 
                'MINUTES_PLAYED', 'AST_TO_RATIO'
            ]
            
            for stat in stats_to_show:
                if stat in stats_df.columns:
                    value = recent_game[stat]
                    # Format as float with 2 decimal places if numeric
                    if isinstance(value, (int, float)):
                        print(f"{stat}: {value:.2f}")
                    else:
                        print(f"{stat}: {value}")
        else:
            print("No stats available")
            
    except Exception as e:
        logger.error(f"Error retrieving stats for {player_name}: {str(e)}")
        print(f"Error: {str(e)}")


def test_sample_players(stats_collector):
    """Test with sample players if no games are found."""
    sample_players = [
        "LeBron James",
        "Stephen Curry",
        "Giannis Antetokounmpo",
        "Kevin Durant",
        "Nikola Jokic"
    ]
    
    for player_name in sample_players:
        print_player_stats(stats_collector, player_name)


if __name__ == "__main__":
    main()