"""
Test script to verify the NBA scraper functionality.
This script tests both the game scraper and player scraper.
Focuses on same-day prediction by limiting days_ahead to 1.
"""

import os
import re
import logging
from datetime import datetime

import requests
from bs4 import BeautifulSoup

from data.database import Database
from scrapers.nba_games import NBAGamesScraper
from scrapers.nba_players import NBAPlayersScraper

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_scrapers')

def test_game_scraper(days_ahead=1):
    """Test the NBA games scraper functionality."""
    logger.info("=== Testing NBA Games Scraper ===")
    
    # Create test database with a temporary name
    test_db_path = "test_nba_scrapers.db"
    db = Database(db_path=test_db_path)
    db.initialize_database()
    
    try:
        # Create scraper
        game_scraper = NBAGamesScraper(db=db)
        
        # Test ESPN schedule scraping
        logger.info("Testing ESPN schedule scraping...")
        espn_games = game_scraper.scrape_espn_schedule(days_ahead=days_ahead)
        logger.info(f"✓ Found {len(espn_games)} games from ESPN for today")
        
        if espn_games:
            # Display sample game
            sample_game = espn_games[0]
            logger.info(f"Sample game: {sample_game['away_team']} @ {sample_game['home_team']} on {sample_game['game_date']}")
        else:
            logger.warning("No games found from ESPN")
        
        # Test NBA.com schedule scraping
        logger.info("Testing NBA.com schedule scraping...")
        nba_games = game_scraper.scrape_nba_schedule(days_ahead=days_ahead)
        logger.info(f"✓ Found {len(nba_games)} games from NBA.com for today")
        
        if nba_games:
            # Display sample game
            sample_game = nba_games[0]
            logger.info(f"Sample game: {sample_game['away_team']} @ {sample_game['home_team']} on {sample_game['game_date']}")
        else:
            logger.warning("No games found from NBA.com")
        
        # Test game saving
        logger.info("Testing saving games to database...")
        saved_count = game_scraper.save_games_to_database(espn_games + nba_games)
        logger.info(f"✓ Saved {saved_count} games to database")
        
        # Test full scrape and save
        logger.info("Testing full scrape and save process...")
        saved_count = game_scraper.scrape_and_save_games(days_ahead=days_ahead)
        logger.info(f"✓ Scraped and saved {saved_count} games in full process")
        
        # Verify games in database
        db.connect()
        upcoming_games = db.get_upcoming_games(days_ahead=days_ahead)
        db.disconnect()
        
        logger.info(f"✓ Found {len(upcoming_games)} upcoming games in database")
        
        # Print all game details for same-day prediction
        for i, game in enumerate(upcoming_games):
            logger.info(f"Game {i+1}: {game['away_team']} @ {game['home_team']} on {game['game_date']} at {game['game_time']}")
        
        logger.info("NBA Games Scraper test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"NBA Games Scraper test failed: {str(e)}")
        return False
        
    finally:
        try:
            # Clean up - delete test database
            if os.path.exists(test_db_path):
                try:
                    os.remove(test_db_path)
                    logger.info(f"✓ Test database {test_db_path} removed")
                except:
                    logger.warning(f"Could not remove test database {test_db_path}")
        except:
            pass

def test_player_scraper():
    """Test the NBA players scraper functionality."""
    logger.info("=== Testing NBA Players Scraper ===")
    
    # Create test database with a temporary name
    test_db_path = "test_nba_scrapers.db"
    db = Database(db_path=test_db_path)
    db.initialize_database()
    
    try:
        # Create scraper
        player_scraper = NBAPlayersScraper(db=db)
        
        # Test ESPN injury report scraping
        logger.info("Testing ESPN injury report scraping...")
        espn_players = player_scraper.scrape_espn_injury_report()
        logger.info(f"✓ Found {len(espn_players)} players in ESPN injury report")
        
        if espn_players:
            # Display sample players
            for i, player in enumerate(espn_players[:5]):  # Show first 5 players
                logger.info(f"Player {i+1}: {player['name']} ({player['team']}) - Status: {player['status']}")
            
            if len(espn_players) > 5:
                logger.info(f"... and {len(espn_players) - 5} more players")
        else:
            logger.warning("No players found in ESPN injury report")
        
        # Test NBA.com injury report scraping
        logger.info("Testing NBA.com injury report scraping...")
        nba_players = player_scraper.scrape_nba_injury_report()
        logger.info(f"✓ Found {len(nba_players)} players in NBA.com injury report")
        
        if nba_players:
            # Display sample players
            for i, player in enumerate(nba_players[:5]):  # Show first 5 players
                logger.info(f"Player {i+1}: {player['name']} ({player['team']}) - Status: {player['status']}")
            
            if len(nba_players) > 5:
                logger.info(f"... and {len(nba_players) - 5} more players")
        else:
            logger.warning("No players found in NBA.com injury report")
        
        # Test team roster scraping (limit to a few teams for testing)
        logger.info("Testing team roster scraping (sampling 3 teams)...")
        
        # Create a custom function that only tests 3 teams
        def sample_team_rosters():
            # Create a version that only tests 3 teams
            from scrapers.nba_games import NBA_TEAMS
            # Get first 3 teams only
            teams_to_test = list(NBA_TEAMS.keys())[:3]
            
            players = []
            for team_name in teams_to_test:
                team_data = NBA_TEAMS[team_name]
                team_abbr = team_data["abbr"]
                
                # ESPN team roster URL
                url = f"https://www.espn.com/nba/team/roster/_/name/{team_abbr.lower()}"
                
                html_content = player_scraper._make_request(url)
                if not html_content:
                    continue
                
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Find the roster table
                roster_table = soup.find('table', class_=re.compile('Table'))
                
                if not roster_table:
                    logger.warning(f"Could not find roster table for {team_name}")
                    continue
                
                # Find all player rows
                player_rows = roster_table.find_all('tr')
                
                # Skip header row
                for row in player_rows[1:]:
                    try:
                        cells = row.find_all('td')
                        
                        if len(cells) < 2:
                            continue
                        
                        # Extract player name
                        name_cell = cells[1]
                        player_name = name_cell.text.strip()
                        
                        # Extract position
                        position = cells[2].text.strip() if len(cells) > 2 else ""
                        
                        # Create player data (assume active status for roster players)
                        player_data = {
                            'name': player_name,
                            'team': team_name,
                            'position': position,
                            'status': 'active',
                            'status_detail': 'Active Roster'
                        }
                        
                        players.append(player_data)
                        
                    except Exception as e:
                        logger.error(f"Error parsing player row for {team_name}: {str(e)}")
                        continue
                
                logger.info(f"Scraped {len(player_rows) - 1} players from {team_name} roster")
            
            return players
            
        # Replace the original method temporarily
        original_scrape_team_rosters = player_scraper.scrape_team_rosters
        player_scraper.scrape_team_rosters = sample_team_rosters
        
        # Call the replacement method
        roster_players = player_scraper.scrape_team_rosters()
        
        # Restore the original method
        player_scraper.scrape_team_rosters = original_scrape_team_rosters
        
        logger.info(f"✓ Found {len(roster_players)} players in team rosters (sample)")
        
        if roster_players:
            # Display sample players
            for i, player in enumerate(roster_players[:5]):  # Show first 5 players
                logger.info(f"Roster Player {i+1}: {player['name']} ({player['team']}) - Position: {player['position']}")
            
            if len(roster_players) > 5:
                logger.info(f"... and {len(roster_players) - 5} more players")
        else:
            logger.warning("No players found in team rosters")
        
        # Test player saving
        logger.info("Testing saving players to database...")
        saved_count = player_scraper.save_players_to_database(roster_players[:10])  # Save a subset
        logger.info(f"✓ Saved {saved_count} players to database")
        
        # Test updating statuses
        logger.info("Testing updating player statuses...")
        updated_count = player_scraper.update_player_statuses_from_injuries(espn_players[:5])  # Update a subset
        logger.info(f"✓ Updated {updated_count} player statuses")
        
        # Test get_today_game_players
        from scrapers.nba_games import NBA_TEAMS
        team_name = list(NBA_TEAMS.keys())[0]  # Get first team
        logger.info(f"Testing get_team_active_players for {team_name}...")
        active_players = player_scraper.get_team_active_players(team_name)
        logger.info(f"✓ Found {len(active_players)} active players for {team_name}")
        
        logger.info("NBA Players Scraper test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"NBA Players Scraper test failed: {str(e)}")
        return False
        
    finally:
        try:
            # Clean up - delete test database
            if os.path.exists(test_db_path):
                try:
                    os.remove(test_db_path)
                    logger.info(f"✓ Test database {test_db_path} removed")
                except:
                    logger.warning(f"Could not remove test database {test_db_path}")
        except:
            pass

def run_all_tests():
    """Run all scraper tests and summarize results."""
    logger.info("Starting NBA Scraper tests")
    logger.info(f"Test time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("Focusing on same-day prediction")
    
    # Run tests
    games_result = test_game_scraper(days_ahead=1)
    players_result = test_player_scraper()
    
    # Summarize results
    logger.info("\n=== Test Summary ===")
    logger.info(f"NBA Games Scraper Test: {'PASSED' if games_result else 'FAILED'}")
    logger.info(f"NBA Players Scraper Test: {'PASSED' if players_result else 'FAILED'}")
    
    overall = games_result and players_result
    logger.info(f"Overall Result: {'PASSED' if overall else 'FAILED'}")
    
    return overall

if __name__ == "__main__":
    run_all_tests()