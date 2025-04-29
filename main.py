#!/usr/bin/env python3
"""
NBA Stats Predictor - Main Application Entry Point

This script serves as the entry point for the NBA Stats Predictor application.
It initializes the core components and starts the web application.

Usage:
    python main.py [--scrape] [--serve]

Options:
    --scrape   Scrape today's games and lineups
    --serve    Start the web application server
"""

import os
import sys
import logging
import argparse
from datetime import datetime

# Import application components
from data.database import Database
from scrapers.game_scraper import NBAApiScraper
from scrapers.player_scraper import RotowireScraper
from analysis.player_stats import PlayerStatsCollector
from analysis.time_series import PlayerTimeSeriesAnalyzer

# Flask app import (for serving the web application)
from app import app

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("nba_predictor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('main')


class StatLemonApplication:
    """
    Main application class for NBA Stats Predictor.
    Handles core components and data scraping.
    """
    
    def __init__(self):
        """Initialize the application components."""
        logger.info("Initializing Stat Lemon application")
        
        # Initialize database
        self.db = Database()
        self.db.initialize_database()
        
        # Initialize scrapers
        self.nba_scraper = NBAApiScraper(self.db)
        self.rotowire_scraper = RotowireScraper(self.db)
        
        # Initialize analysis components
        self.stats_collector = PlayerStatsCollector()
        self.time_series_analyzer = PlayerTimeSeriesAnalyzer(min_games=10)
        
        logger.info("Application components initialized")
    
    def scrape_data(self):
        """Scrape today's games and lineups."""
        logger.info("Starting data scraping process")
        
        try:
            # Scrape games from NBA API
            num_games = self.nba_scraper.scrape_and_save_games(days_ahead=0)
            logger.info(f"Scraped {num_games} games from NBA API")
            
            # Scrape lineups from Rotowire
            num_lineups = self.rotowire_scraper.scrape_and_save_lineups(date_type='today')
            logger.info(f"Scraped {num_lineups} lineups from Rotowire")
            
            # Get today's date
            today = datetime.now().strftime('%Y-%m-%d')
            
            # Retrieve scraped data from database
            today_games = self.db.get_games_by_date(today)
            
            logger.info(f"Retrieved {len(today_games)} games for today")
            
            return len(today_games) > 0
            
        except Exception as e:
            logger.error(f"Error scraping data: {str(e)}")
            return False
    
    def start_web_server(self):
        """Start the web application server."""
        logger.info("Starting web application server")
        app.run(host='0.0.0.0', port=5000)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='NBA Stats Predictor')
    parser.add_argument('--scrape', action='store_true', help='Scrape today\'s games and lineups')
    parser.add_argument('--serve', action='store_true', help='Start the web application server')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Create the application
    app_instance = StatLemonApplication()
    
    # Run specific tasks if requested
    if args.scrape:
        app_instance.scrape_data()
    
    # Start the web server if requested
    if args.serve:
        app_instance.start_web_server()
    
    # If no arguments provided, assume both scrape and serve
    if not (args.scrape or args.serve):
        app_instance.scrape_data()
        app_instance.start_web_server()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())