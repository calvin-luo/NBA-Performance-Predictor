#!/usr/bin/env python3
"""
NBA Sentiment Predictor - Database Tester

This script tests the database functionality of the NBA Sentiment Predictor.
It verifies that all CRUD operations work as expected for each table.
"""

import os
import sys
import logging
import datetime
from typing import Dict, List, Any, Optional
import unittest

from data.database import Database

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_database')

class DatabaseTester(unittest.TestCase):
    """Test cases for the Database class."""
    
    def setUp(self):
        """Set up the test case."""
        # Create a temporary database for testing
        self.test_db_path = "test_nba_sentiment.db"
        self.db = Database(db_path=self.test_db_path)
        self.db.initialize_database()
    
    def tearDown(self):
        """Clean up after the test case."""
        # Delete the temporary database
        if os.path.exists(self.test_db_path):
            try:
                os.remove(self.test_db_path)
                logger.info(f"Removed test database: {self.test_db_path}")
            except Exception as e:
                logger.warning(f"Could not remove test database: {e}")
    
    def test_database_connection(self):
        """Test database connection functionality."""
        # Test connect and disconnect
        self.db.connect()
        self.assertIsNotNone(self.db.conn, "Connection should be established")
        self.assertIsNotNone(self.db.cursor, "Cursor should be created")
        
        self.db.disconnect()
        self.assertIsNone(self.db.conn, "Connection should be closed")
        self.assertIsNone(self.db.cursor, "Cursor should be None")
        
        # Test context manager
        with self.db.get_connection() as conn:
            self.assertIsNotNone(conn, "Connection should be established via context manager")
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            self.assertEqual(result[0], 1, "Simple query should work")
    
    def test_games_operations(self):
        """Test game-related database operations."""
        # Test game insertion
        game_data = {
            'game_id': 'TEST_LAL_BOS',
            'home_team': 'Boston Celtics',
            'away_team': 'Los Angeles Lakers',
            'game_date': '2025-03-20',
            'game_time': '19:30',
            'venue': 'TD Garden'
        }
        
        game_id = self.db.insert_game(game_data)
        self.assertEqual(game_id, 'TEST_LAL_BOS', "Game ID should be returned correctly")
        
        # Test game retrieval
        game = self.db.get_game('TEST_LAL_BOS')
        self.assertIsNotNone(game, "Game should be retrievable")
        self.assertEqual(game['home_team'], 'Boston Celtics', "Home team should match")
        self.assertEqual(game['away_team'], 'Los Angeles Lakers', "Away team should match")
        
        # Test game update
        update_data = {
            'venue': 'Updated Venue',
            'game_time': '20:00'
        }
        result = self.db.update_game('TEST_LAL_BOS', update_data)
        self.assertTrue(result, "Update should be successful")
        
        # Verify update worked
        updated_game = self.db.get_game('TEST_LAL_BOS')
        self.assertEqual(updated_game['venue'], 'Updated Venue', "Venue should be updated")
        self.assertEqual(updated_game['game_time'], '20:00', "Game time should be updated")
        
        # Test upcoming games retrieval
        upcoming_games = self.db.get_upcoming_games(days_ahead=7)
        self.assertGreaterEqual(len(upcoming_games), 1, "Should find at least one upcoming game")
    
    def test_players_operations(self):
        """Test player-related database operations."""
        # Test player insertion
        player_data = {
            'name': 'Test Player',
            'team': 'Boston Celtics',
            'status': 'active'
        }
        
        player_id = self.db.insert_player(player_data)
        self.assertGreater(player_id, 0, "Player ID should be a positive integer")
        
        # Test player status update
        result = self.db.update_player_status(player_id, 'questionable')
        self.assertTrue(result, "Status update should be successful")
        
        # Test players by team retrieval
        team_players = self.db.get_players_by_team('Boston Celtics')
        self.assertEqual(len(team_players), 1, "Should find one player for the team")
        self.assertEqual(team_players[0]['name'], 'Test Player', "Player name should match")
        self.assertEqual(team_players[0]['status'], 'questionable', "Updated status should be reflected")
    
    def test_reddit_data_operations(self):
        """Test Reddit data operations."""
        # Test Reddit post insertion
        post_data = {
            'reddit_id': 'test123',
            'subreddit': 'nba',
            'title': 'Test Post',
            'content': 'Test content mentioning the Celtics',
            'author': 'test_user',
            'created_utc': datetime.datetime.now().timestamp(),
            'score': 10,
            'team_mention': 'Boston Celtics'
        }
        
        post_id = self.db.insert_reddit_post(post_data)
        self.assertGreater(post_id, 0, "Post ID should be a positive integer")
        
        # Test Reddit comment insertion
        comment_data = {
            'reddit_id': 'comment123',
            'post_id': post_id,
            'content': 'Test comment',
            'author': 'comment_user',
            'created_utc': datetime.datetime.now().timestamp(),
            'score': 5
        }
        
        comment_id = self.db.insert_reddit_comment(comment_data)
        self.assertGreater(comment_id, 0, "Comment ID should be a positive integer")
        
        # Test Reddit posts by team retrieval
        team_posts = self.db.get_reddit_posts_by_team('Boston Celtics', days_back=1)
        self.assertEqual(len(team_posts), 1, "Should find one post for the team")
        self.assertEqual(team_posts[0]['title'], 'Test Post', "Post title should match")
    
    def test_sentiment_operations(self):
        """Test sentiment analysis operations."""
        # First create a Reddit post for the sentiment record
        post_data = {
            'reddit_id': 'sentiment_post',
            'subreddit': 'nba',
            'title': 'Sentiment Test',
            'content': 'Test content',
            'author': 'test_user',
            'created_utc': datetime.datetime.now().timestamp(),
            'score': 10,
            'team_mention': 'Boston Celtics'
        }
        
        post_id = self.db.insert_reddit_post(post_data)
        
        # Test sentiment analysis insertion
        sentiment_data = {
            'entity_type': 'team',
            'entity_id': 'Boston Celtics',
            'post_id': post_id,
            'sentiment_score': 0.75,
            'confidence': 0.85
        }
        
        sentiment_id = self.db.insert_sentiment_analysis(sentiment_data)
        self.assertGreater(sentiment_id, 0, "Sentiment ID should be a positive integer")
        
        # Test team sentiment retrieval
        avg_sentiment, count = self.db.get_team_sentiment('Boston Celtics', days_back=1)
        self.assertEqual(count, 1, "Should find one sentiment record")
        self.assertEqual(avg_sentiment, 0.75, "Average sentiment should match")
    
    def test_prediction_operations(self):
        """Test prediction operations."""
        # First create a game for the prediction
        game_data = {
            'game_id': 'PRED_TEST',
            'home_team': 'Boston Celtics',
            'away_team': 'Los Angeles Lakers',
            'game_date': '2025-03-20',
            'game_time': '19:30',
            'venue': 'TD Garden'
        }
        
        self.db.insert_game(game_data)
        
        # Test prediction insertion
        prediction_data = {
            'game_id': 'PRED_TEST',
            'home_team_sentiment': 0.65,
            'away_team_sentiment': 0.45,
            'home_win_probability': 0.70,
            'prediction_timestamp': datetime.datetime.now().isoformat()
        }
        
        prediction_id = self.db.insert_prediction(prediction_data)
        self.assertGreater(prediction_id, 0, "Prediction ID should be a positive integer")
        
        # Test latest prediction retrieval
        prediction = self.db.get_latest_prediction('PRED_TEST')
        self.assertIsNotNone(prediction, "Prediction should be retrievable")
        self.assertEqual(prediction['home_win_probability'], 0.70, "Win probability should match")
        
        # Test prediction result update
        result = self.db.update_prediction_results('PRED_TEST', 'home_win')
        self.assertTrue(result, "Result update should be successful")
        
        # Verify update worked
        updated_game = self.db.get_game('PRED_TEST')
        self.assertEqual(updated_game['actual_result'], 'home_win', "Game result should be updated")
    
    def test_complex_scenario(self):
        """Test a complex scenario with multiple operations."""
        # 1. Insert a game
        game_data = {
            'game_id': 'COMPLEX_TEST',
            'home_team': 'Boston Celtics',
            'away_team': 'Los Angeles Lakers',
            'game_date': '2025-03-20',
            'game_time': '19:30',
            'venue': 'TD Garden'
        }
        
        self.db.insert_game(game_data)
        
        # 2. Insert players for both teams
        players = [
            {'name': 'Player 1', 'team': 'Boston Celtics', 'status': 'active'},
            {'name': 'Player 2', 'team': 'Boston Celtics', 'status': 'active'},
            {'name': 'Player 3', 'team': 'Los Angeles Lakers', 'status': 'active'},
            {'name': 'Player 4', 'team': 'Los Angeles Lakers', 'status': 'questionable'},
        ]
        
        for player_data in players:
            self.db.insert_player(player_data)
        
        # 3. Insert Reddit posts mentioning both teams
        posts = [
            {
                'reddit_id': 'post1',
                'subreddit': 'nba',
                'title': 'Celtics looking good',
                'content': 'The Celtics are playing well lately',
                'author': 'user1',
                'created_utc': datetime.datetime.now().timestamp(),
                'score': 15,
                'team_mention': 'Boston Celtics'
            },
            {
                'reddit_id': 'post2',
                'subreddit': 'nba',
                'title': 'Lakers struggling',
                'content': 'The Lakers need to improve',
                'author': 'user2',
                'created_utc': datetime.datetime.now().timestamp(),
                'score': 8,
                'team_mention': 'Los Angeles Lakers'
            }
        ]
        
        post_ids = []
        for post_data in posts:
            post_id = self.db.insert_reddit_post(post_data)
            post_ids.append(post_id)
        
        # 4. Insert sentiment analysis for each post
        sentiments = [
            {
                'entity_type': 'team',
                'entity_id': 'Boston Celtics',
                'post_id': post_ids[0],
                'sentiment_score': 0.80,
                'confidence': 0.90
            },
            {
                'entity_type': 'team',
                'entity_id': 'Los Angeles Lakers',
                'post_id': post_ids[1],
                'sentiment_score': 0.30,
                'confidence': 0.85
            }
        ]
        
        for sentiment_data in sentiments:
            self.db.insert_sentiment_analysis(sentiment_data)
        
        # 5. Insert a prediction based on the sentiment
        prediction_data = {
            'game_id': 'COMPLEX_TEST',
            'home_team_sentiment': 0.80,
            'away_team_sentiment': 0.30,
            'home_win_probability': 0.75,
            'prediction_timestamp': datetime.datetime.now().isoformat()
        }
        
        self.db.insert_prediction(prediction_data)
        
        # 6. Verify we can get all the data back
        # Check game
        game = self.db.get_game('COMPLEX_TEST')
        self.assertIsNotNone(game, "Game should be retrievable")
        
        # Check players
        celtics_players = self.db.get_players_by_team('Boston Celtics')
        lakers_players = self.db.get_players_by_team('Los Angeles Lakers')
        self.assertEqual(len(celtics_players), 2, "Should have 2 Celtics players")
        self.assertEqual(len(lakers_players), 2, "Should have 2 Lakers players")
        
        # Check sentiment
        celtics_sentiment, celtics_count = self.db.get_team_sentiment('Boston Celtics', days_back=1)
        lakers_sentiment, lakers_count = self.db.get_team_sentiment('Los Angeles Lakers', days_back=1)
        self.assertEqual(celtics_sentiment, 0.80, "Celtics sentiment should match")
        self.assertEqual(lakers_sentiment, 0.30, "Lakers sentiment should match")
        
        # Check prediction
        prediction = self.db.get_latest_prediction('COMPLEX_TEST')
        self.assertEqual(prediction['home_win_probability'], 0.75, "Prediction probability should match")


def run_database_tests():
    """Run all database tests."""
    logger.info("=" * 50)
    logger.info("Running NBA Sentiment Predictor Database Tests")
    logger.info("=" * 50)
    
    # Run tests
    suite = unittest.TestLoader().loadTestsFromTestCase(DatabaseTester)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    
    # Report results
    logger.info("=" * 50)
    logger.info(f"Test Results: {result.testsRun} tests run")
    logger.info(f"Failures: {len(result.failures)}")
    logger.info(f"Errors: {len(result.errors)}")
    logger.info("=" * 50)
    
    return len(result.failures) == 0 and len(result.errors) == 0

def main():
    """Main function to run the database tests."""
    if run_database_tests():
        logger.info("Database tests passed successfully!")
        return 0
    else:
        logger.error("Some database tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())