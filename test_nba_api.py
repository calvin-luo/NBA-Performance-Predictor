#!/usr/bin/env python3
"""
Test script for NBA API Scraper.
This script tests the updated NBA scraper implementation.
"""

import sys
import logging
from datetime import datetime

# Import our NBA API scraper
from scrapers.nba_api_scraper import NBAApiScraper

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_nba_api_scraper')

def print_games(games):
    """Print games in a formatted manner."""
    if not games:
        print("No games found.")
        return

    print(f"\n===== TODAY'S NBA GAMES ({len(games)}) =====")
    for i, game in enumerate(games, 1):
        print(f"\nGame {i}: {game['away_team']} @ {game['home_team']}")
        print(f"Time: {game['game_time']}")
        print(f"Venue: {game['venue']}")
        print(f"Game ID: {game['game_id']}")
        print("-" * 50)

def print_players_by_team(players):
    """Print players organized by team."""
    if not players:
        print("No players found.")
        return

    # Group players by team
    teams = {}
    for player in players:
        team = player['team']
        if team not in teams:
            teams[team] = []
        teams[team].append(player)

    print(f"\n===== TODAY'S NBA LINEUPS ({len(teams)} teams) =====")

    # Print each team's lineup
    for team, team_players in sorted(teams.items()):
        print(f"\n{team} ({len(team_players)} players):")
        
        # Group by status
        active = []
        questionable = []
        out = []
        
        for player in team_players:
            if player['status'] == 'active':
                active.append(player)
            elif player['status'] == 'questionable':
                questionable.append(player)
            else:
                out.append(player)
        
        # Print active players
        if active:
            print("\nStarters/Active:")
            for player in active:
                print(f"  {player['name']} ({player['position']})")
        
        # Print questionable players
        if questionable:
            print("\nQuestionable:")
            for player in questionable:
                print(f"  {player['name']} ({player['position']}) - {player['status_detail']}")
        
        # Print out players
        if out:
            print("\nOut/Inactive:")
            for player in out:
                print(f"  {player['name']} ({player['position']}) - {player['status_detail']}")

def main():
    """Main test function."""
    print(f"NBA API Scraper Test - {datetime.now().strftime('%A, %B %d, %Y')}")
    print("=" * 60)
    
    try:
        # Create a scraper instance
        scraper = NBAApiScraper()
        
        # Step 1: Test game scraping
        print("\nTesting game schedule scraping...")
        games = scraper.scrape_nba_schedule()
        print_games(games)
        
        # Step 2: Test lineup scraping
        print("\nTesting lineup scraping...")
        players = scraper.scrape_todays_lineups()
        print_players_by_team(players)
        
        print("\nTest completed successfully!")
        print(f"Found {len(games)} games and {len(players)} players.")
    
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())