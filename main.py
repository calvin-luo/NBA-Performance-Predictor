#!/usr/bin/env python3
"""
NBA Sentiment Predictor - Main Script

This script runs the entire data collection, analysis, and prediction pipeline.
It can be executed manually or scheduled via cron for automatic updates.
"""

import os
import sys
import logging
import argparse
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

from data.database import Database
from reddit.connector import get_connector
from reddit.collector import RedditCollector
from scrapers.nba_games import NBAGamesScraper
from scrapers.nba_players import NBAPlayersScraper
from analysis.sentiment import SentimentAnalyzer
from analysis.predictor import NBAPredictor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('nba_sentiment.log')
    ]
)
logger = logging.getLogger('main')


def initialize_components():
    """Initialize all required components."""
    db = Database()
    db.initialize_database()
    
    # Create instances of all components
    game_scraper = NBAGamesScraper(db=db)
    player_scraper = NBAPlayersScraper(db=db)
    reddit_connector = get_connector()
    reddit_collector = RedditCollector(connector=reddit_connector, db=db)
    sentiment_analyzer = SentimentAnalyzer(db=db)
    predictor = NBAPredictor(db=db, sentiment_analyzer=sentiment_analyzer)
    
    return {
        'db': db,
        'game_scraper': game_scraper,
        'player_scraper': player_scraper,
        'reddit_connector': reddit_connector,
        'reddit_collector': reddit_collector,
        'sentiment_analyzer': sentiment_analyzer,
        'predictor': predictor
    }


def scrape_games(components, days_ahead=1):
    """Scrape NBA games for the specified number of days ahead."""
    logger.info(f"Scraping games for the next {days_ahead} days...")
    game_scraper = components['game_scraper']
    
    game_count = game_scraper.scrape_and_save_games(days_ahead=days_ahead)
    logger.info(f"Scraped and saved {game_count} games")
    
    return game_count


def scrape_players(components):
    """Scrape NBA player information and injury reports."""
    logger.info("Scraping player information...")
    player_scraper = components['player_scraper']
    
    players_saved, statuses_updated = player_scraper.scrape_and_save_players()
    logger.info(f"Saved {players_saved} players and updated {statuses_updated} player statuses")
    
    return players_saved, statuses_updated


def collect_reddit_data(components, days_back=1):
    """Collect Reddit data for teams playing today."""
    logger.info(f"Collecting Reddit data (looking back {days_back} days)...")
    db = components['db']
    reddit_collector = components['reddit_collector']
    
    # Get teams playing today
    games = db.get_upcoming_games(days_ahead=0)
    today_teams = set()
    
    for game in games:
        today_teams.add(game['home_team'])
        today_teams.add(game['away_team'])
    
    logger.info(f"Teams playing today: {', '.join(today_teams)}")
    
    if not today_teams:
        logger.warning("No games found for today. Skipping Reddit data collection.")
        return {}
    
    # Collect data for today's teams
    results = reddit_collector.collect_today_teams_data(list(today_teams), days_back=days_back)
    
    total_posts = sum(p for p, _ in results.values())
    total_comments = sum(c for _, c in results.values())
    
    logger.info(f"Collected {total_posts} posts and {total_comments} comments for {len(today_teams)} teams")
    
    return results


def analyze_sentiment(components, hours_back=24):
    """Analyze sentiment for recent Reddit content."""
    logger.info(f"Analyzing sentiment for content from the past {hours_back} hours...")
    sentiment_analyzer = components['sentiment_analyzer']
    
    posts_analyzed, comments_analyzed = sentiment_analyzer.analyze_recent_content(hours_back=hours_back)
    logger.info(f"Analyzed {posts_analyzed} posts and {comments_analyzed} comments")
    
    return posts_analyzed, comments_analyzed


def make_predictions(components, hours_back=24):
    """Make predictions for today's games."""
    logger.info("Making predictions for today's games...")
    predictor = components['predictor']
    
    predictions = predictor.predict_today_games(hours_back=hours_back)
    
    if not predictions:
        logger.warning("No predictions generated. Check if there are games today.")
        return []
    
    logger.info(f"Generated {len(predictions)} predictions")
    
    # Log each prediction
    for pred in predictions:
        if 'error' in pred:
            logger.error(f"Prediction error: {pred['error']}")
            continue
            
        home_team = pred['home_team']
        away_team = pred['away_team']
        win_prob = pred['home_win_probability'] * 100
        
        winner = home_team if win_prob > 50 else away_team
        winner_prob = win_prob if win_prob > 50 else 100 - win_prob
        
        logger.info(f"Prediction: {away_team} @ {home_team} - {winner} to win ({winner_prob:.1f}%)")
    
    return predictions


def run_pipeline(args):
    """Run the full data pipeline or specified components."""
    start_time = time.time()
    logger.info("Starting NBA Sentiment Predictor pipeline")
    
    # Initialize components
    components = initialize_components()
    
    # Run selected components or all if none specified
    results = {}
    
    # Scrape games (always do this first to get today's games)
    if args.all or args.games:
        results['games'] = scrape_games(components, days_ahead=args.days_ahead)
    
    # Scrape player information
    if args.all or args.players:
        results['players'] = scrape_players(components)
    
    # Collect Reddit data
    if args.all or args.reddit:
        results['reddit'] = collect_reddit_data(components, days_back=args.days_back)
    
    # Analyze sentiment
    if args.all or args.sentiment:
        results['sentiment'] = analyze_sentiment(components, hours_back=args.hours_back)
    
    # Make predictions
    if args.all or args.predict:
        results['predictions'] = make_predictions(components, hours_back=args.hours_back)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Pipeline completed in {elapsed_time:.2f} seconds")
    
    return results


def main():
    """Parse arguments and run the pipeline."""
    parser = argparse.ArgumentParser(description='NBA Sentiment Predictor Pipeline')
    parser.add_argument('--all', action='store_true', help='Run the complete pipeline (default)')
    parser.add_argument('--games', action='store_true', help='Scrape NBA games')
    parser.add_argument('--players', action='store_true', help='Scrape NBA player information')
    parser.add_argument('--reddit', action='store_true', help='Collect Reddit data')
    parser.add_argument('--sentiment', action='store_true', help='Analyze sentiment')
    parser.add_argument('--predict', action='store_true', help='Make predictions')
    parser.add_argument('--days-ahead', type=int, default=1, help='Days ahead to scrape games (default: 1)')
    parser.add_argument('--days-back', type=int, default=1, help='Days to look back for Reddit data (default: 1)')
    parser.add_argument('--hours-back', type=int, default=24, help='Hours to look back for sentiment analysis (default: 24)')
    
    args = parser.parse_args()
    
    # If no specific component is selected, run all
    if not (args.games or args.players or args.reddit or args.sentiment or args.predict):
        args.all = True
    
    run_pipeline(args)


if __name__ == '__main__':
    main()