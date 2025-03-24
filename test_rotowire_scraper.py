#!/usr/bin/env python3
"""
Test script for Rotowire lineup scraper.
This script tests scraping NBA lineups from Rotowire.com.
"""

import sys
import logging
from datetime import datetime
from tabulate import tabulate

# Import our NBA API scraper
from scrapers.nba_api_scraper import NBAApiScraper

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_rotowire_scraper')

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

    print(f"\n===== TODAY'S NBA LINEUPS FROM ROTOWIRE ({len(teams)} teams) =====")

    # Print each team's lineup
    for team, team_players in sorted(teams.items()):
        print(f"\n{team} ({len(team_players)} players):")
        
        # Group by status
        starters = []
        questionable = []
        out = []
        
        for player in team_players:
            if player['status'] == 'active':
                starters.append([player['name'], player['position'], player['status_detail']])
            elif player['status'] == 'questionable':
                questionable.append([player['name'], player['position'], player['status_detail']])
            else:
                out.append([player['name'], player['position'], player['status_detail']])
        
        # Print starters
        if starters:
            print("\nStarting Lineup:")
            print(tabulate(starters, headers=['Name', 'Position', 'Status'], tablefmt='simple'))
        
        # Print questionable players
        if questionable:
            print("\nQuestionable:")
            print(tabulate(questionable, headers=['Name', 'Position', 'Status'], tablefmt='simple'))
        
        # Print out players
        if out:
            print("\nOut/Injured:")
            print(tabulate(out, headers=['Name', 'Position', 'Status'], tablefmt='simple'))

def main():
    """Main test function."""
    print(f"Rotowire NBA Lineup Scraper Test - {datetime.now().strftime('%A, %B %d, %Y')}")
    print("=" * 80)
    
    try:
        # Create a scraper instance
        scraper = NBAApiScraper()
        
        # Step 1: Test game scraping
        print("\nTesting game schedule scraping using nba_api...")
        games = scraper.scrape_nba_schedule()
        print_games(games)
        
        # Step 2: Test lineup scraping from Rotowire
        print("\nTesting lineup scraping from Rotowire...")
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