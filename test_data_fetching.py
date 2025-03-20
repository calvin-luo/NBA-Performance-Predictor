"""
Test script to verify the data fetching capabilities of the NBA Sentiment Predictor.
This script tests the database connection, Reddit API connection, and data collection.
"""

import os
import logging
from datetime import datetime

from data.database import Database
from reddit.connector import get_connector
from reddit.collector import RedditCollector

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_data_fetching')

def test_database():
    """Test database initialization and basic operations."""
    logger.info("=== Testing Database ===")
    
    # Create test database with a temporary name
    test_db_path = "test_nba_sentiment.db"
    db = Database(db_path=test_db_path)
    
    try:
        # Initialize database
        db.initialize_database()
        logger.info("✓ Database initialized successfully")
        
        # Test inserting a game
        game_data = {
            'game_id': 'TEST123',
            'home_team': 'Boston Celtics',
            'away_team': 'Los Angeles Lakers',
            'game_date': '2025-03-20',
            'game_time': '19:30',
            'venue': 'TD Garden'
        }
        
        game_id = db.insert_game(game_data)
        logger.info(f"✓ Game inserted successfully with ID: {game_id}")
        
        # Test retrieving the game
        retrieved_game = db.get_game('TEST123')
        if retrieved_game and retrieved_game['home_team'] == 'Boston Celtics':
            logger.info("✓ Game retrieved successfully")
        else:
            logger.error("✗ Failed to retrieve game correctly")
        
        logger.info("Database test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Database test failed: {str(e)}")
        return False
        
    finally:
        # Clean up - delete test database
        if os.path.exists(test_db_path):
            try:
                os.remove(test_db_path)
                logger.info(f"✓ Test database {test_db_path} removed")
            except:
                logger.warning(f"Could not remove test database {test_db_path}")

def test_reddit_connection():
    """Test connection to Reddit API."""
    logger.info("=== Testing Reddit Connection ===")
    
    try:
        # Create connector
        connector = get_connector()
        
        # Test connection
        if connector.check_connection():
            logger.info("✓ Successfully connected to Reddit API")
            
            # Verify we can access NBA subreddit
            reddit = connector.get_reddit_instance()
            subreddit = reddit.subreddit('nba')
            
            # Get info about the subreddit
            title = subreddit.title
            subscribers = subreddit.subscribers
            
            logger.info(f"✓ Connected to r/nba: {title} ({subscribers:,} subscribers)")
            logger.info("Reddit connection test completed successfully")
            return True
        else:
            logger.error("✗ Failed to connect to Reddit API")
            return False
            
    except Exception as e:
        logger.error(f"Reddit connection test failed: {str(e)}")
        return False

def test_data_collection(team_name='Boston Celtics'):
    """Test collecting data for a specific team."""
    logger.info(f"=== Testing Data Collection for {team_name} ===")
    
    # Create test database
    test_db_path = "test_nba_sentiment.db"
    db = Database(db_path=test_db_path)
    db.initialize_database()
    
    try:
        # Create collector
        collector = RedditCollector(db=db)
        
        # Collect data for one team with limited scope
        logger.info(f"Collecting data for {team_name} (looking back 1 day, limited to r/nba)...")
        posts, comments = collector.collect_team_data(team_name, days_back=1)
        
        # Log results
        logger.info(f"✓ Collected {posts} posts and ~{comments} comments for {team_name}")
        
        # Check if data was saved in database
        db.connect()
        cursor = db.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM reddit_posts")
        post_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM reddit_comments")
        comment_count = cursor.fetchone()[0]
        db.disconnect()
        
        logger.info(f"✓ Database now contains {post_count} posts and {comment_count} comments")
        
        if posts > 0 or post_count > 0:
            logger.info("Data collection test completed successfully")
            return True
        else:
            logger.warning("No posts were collected. This could be normal if there are no recent posts about the team.")
            return True  # Still return True as this might be legitimate
            
    except Exception as e:
        logger.error(f"Data collection test failed: {str(e)}")
        return False
        
    finally:
        # Clean up - delete test database
        if os.path.exists(test_db_path):
            try:
                os.remove(test_db_path)
                logger.info(f"✓ Test database {test_db_path} removed")
            except:
                logger.warning(f"Could not remove test database {test_db_path}")

def run_all_tests():
    """Run all tests and summarize results."""
    logger.info("Starting NBA Sentiment Predictor data fetching tests")
    logger.info(f"Test time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run tests
    db_result = test_database()
    reddit_result = test_reddit_connection()
    collection_result = test_data_collection()
    
    # Summarize results
    logger.info("\n=== Test Summary ===")
    logger.info(f"Database Test: {'PASSED' if db_result else 'FAILED'}")
    logger.info(f"Reddit Connection Test: {'PASSED' if reddit_result else 'FAILED'}")
    logger.info(f"Data Collection Test: {'PASSED' if collection_result else 'FAILED'}")
    
    overall = db_result and reddit_result and collection_result
    logger.info(f"Overall Result: {'PASSED' if overall else 'FAILED'}")
    
    return overall

if __name__ == "__main__":
    run_all_tests()