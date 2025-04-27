#!/usr/bin/env python3
"""
NBA Stats Predictor - Main Application Entry Point

This script serves as the entry point for the NBA Stats Predictor application.
It orchestrates the various components (scrapers, database, analysis) and 
sets up scheduled jobs for data collection and model updates.

Usage:
    python main.py [--scrape] [--analyze] [--predict] [--serve]

Options:
    --scrape   Scrape today's games and lineups
    --analyze  Analyze player stats for today's games
    --predict  Generate predictions for today's games
    --serve    Start the web application server
"""

import os
import sys
import time
import logging
import argparse
import schedule
import threading
from datetime import datetime, timedelta

# Import application components
from data.database import Database
from scrapers.game_scraper import NBAApiScraper
from scrapers.player_scraper import RotowireScraper
from analysis.player_stats import PlayerStatsCollector
from analysis.time_series import PlayerTimeSeriesAnalyzer, GamePredictor

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
    Orchestrates components and manages scheduled jobs.
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
        self.game_predictor = GamePredictor(min_games=10)
        
        # Storage for today's data
        self.todays_games = []
        self.todays_lineups = []
        self.team_players_stats = {}
        self.game_predictions = []
        
        # Flag to indicate if predictions are ready
        self.predictions_ready = False
        
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
            
            # Check if lineups were found
            if num_lineups == 0:
                logger.warning("No lineups found for today. Using NBA API data only.")
            
            # Get today's date
            today = datetime.now().strftime('%Y-%m-%d')
            
            # Retrieve scraped data from database
            self.todays_games = self.db.get_games_by_date(today)
            self.todays_lineups = self.db.get_lineups_by_date(today)
            
            logger.info(f"Retrieved {len(self.todays_games)} games and {len(self.todays_lineups)} lineups for today")
            
            return len(self.todays_games) > 0
            
        except Exception as e:
            logger.error(f"Error scraping data: {str(e)}")
            return False
    
    def analyze_player_stats(self):
        """Analyze player stats for today's games."""
        logger.info("Starting player stats analysis")
        
        try:
            # Reset team players stats
            self.team_players_stats = {}
            
            # Process each game
            for game in self.todays_games:
                home_team = game['home_team']
                away_team = game['away_team']
                
                # Get home team lineup
                home_lineup = self.db.get_team_lineup(home_team, game['game_id'])
                
                # Get away team lineup
                away_lineup = self.db.get_team_lineup(away_team, game['game_id'])
                
                logger.info(f"Analyzing players for {away_team} @ {home_team}")
                
                # Collect stats for home team players
                home_players_stats = {}
                for player in home_lineup:
                    player_name = player['name']
                    try:
                        # Skip injured players
                        if player.get('status') != 'active':
                            logger.info(f"Skipping injured player: {player_name}")
                            continue
                        
                        # Get player stats
                        stats = self.stats_collector.get_player_stats(player_name)
                        if stats is not None:
                            home_players_stats[player_name] = stats
                            logger.info(f"Collected stats for {player_name}")
                        else:
                            logger.warning(f"No stats found for {player_name}")
                    except Exception as e:
                        logger.error(f"Error analyzing {player_name}: {str(e)}")
                
                # Collect stats for away team players
                away_players_stats = {}
                for player in away_lineup:
                    player_name = player['name']
                    try:
                        # Skip injured players
                        if player.get('status') != 'active':
                            logger.info(f"Skipping injured player: {player_name}")
                            continue
                        
                        # Get player stats
                        stats = self.stats_collector.get_player_stats(player_name)
                        if stats is not None:
                            away_players_stats[player_name] = stats
                            logger.info(f"Collected stats for {player_name}")
                        else:
                            logger.warning(f"No stats found for {player_name}")
                    except Exception as e:
                        logger.error(f"Error analyzing {player_name}: {str(e)}")
                
                # Store team players stats
                self.team_players_stats[home_team] = home_players_stats
                self.team_players_stats[away_team] = away_players_stats
                
                logger.info(f"Analyzed {len(home_players_stats)} home players and {len(away_players_stats)} away players")
            
            return len(self.team_players_stats) > 0
            
        except Exception as e:
            logger.error(f"Error analyzing player stats: {str(e)}")
            return False
    
    def generate_predictions(self):
        """Generate predictions for today's games."""
        logger.info("Starting game prediction generation")
        
        try:
            # Reset game predictions
            self.game_predictions = []
            
            # Process each game
            for game in self.todays_games:
                home_team = game['home_team']
                away_team = game['away_team']
                
                logger.info(f"Predicting game: {away_team} @ {home_team}")
                
                # Get team players stats
                home_players_stats = self.team_players_stats.get(home_team, {})
                away_players_stats = self.team_players_stats.get(away_team, {})
                
                # Check if we have enough player stats
                if len(home_players_stats) < 3 or len(away_players_stats) < 3:
                    logger.warning(f"Not enough player stats for {away_team} @ {home_team}")
                    continue
                
                # Generate game prediction
                prediction = self.game_predictor.predict_game(
                    game,
                    {home_team: home_players_stats, away_team: away_players_stats}
                )
                
                if prediction and 'error' not in prediction:
                    self.game_predictions.append(prediction)
                    logger.info(f"Generated prediction for {away_team} @ {home_team}")
                else:
                    error_msg = prediction.get('error', 'Unknown error') if prediction else 'No prediction generated'
                    logger.warning(f"Failed to predict {away_team} @ {home_team}: {error_msg}")
            
            # Set predictions ready flag
            self.predictions_ready = len(self.game_predictions) > 0
            
            logger.info(f"Generated {len(self.game_predictions)} game predictions")
            
            # Save predictions to database
            self.save_predictions()
            
            return self.predictions_ready
            
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            self.predictions_ready = False
            return False
    
    def save_predictions(self):
        """Save predictions to database."""
        logger.info("Saving predictions to database")
        
        try:
            for prediction in self.game_predictions:
                self.db.save_prediction(prediction)
            
            logger.info(f"Saved {len(self.game_predictions)} predictions to database")
            return True
            
        except Exception as e:
            logger.error(f"Error saving predictions: {str(e)}")
            return False
    
    def run_daily_update(self):
        """Run the daily update process (scrape, analyze, predict)."""
        logger.info("Starting daily update process")
        
        # Scrape today's data
        if self.scrape_data():
            # Analyze player stats
            if self.analyze_player_stats():
                # Generate predictions
                self.generate_predictions()
        
        logger.info("Daily update process completed")
    
    def schedule_jobs(self):
        """Schedule regular jobs."""
        logger.info("Scheduling regular jobs")
        
        # Schedule daily update at 1:00 AM
        schedule.every().day.at("01:00").do(self.run_daily_update)
        
        # Schedule game schedule update at 10:00 AM
        schedule.every().day.at("10:00").do(self.scrape_data)
        
        # Schedule lineup update at 4:00 PM (closer to game time)
        schedule.every().day.at("16:00").do(self.scrape_data)
        
        # Schedule another lineup update at 6:00 PM (even closer to game time)
        schedule.every().day.at("18:00").do(self.scrape_data)
        
        logger.info("Jobs scheduled")
    
    def run_scheduler(self):
        """Run the scheduler in a separate thread."""
        logger.info("Starting scheduler thread")
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def start_scheduler(self):
        """Start the scheduler in a separate thread."""
        scheduler_thread = threading.Thread(target=self.run_scheduler)
        scheduler_thread.daemon = True
        scheduler_thread.start()
        logger.info("Scheduler thread started")
    
    def start_web_server(self):
        """Start the web application server."""
        logger.info("Starting web application server")
        app.run(host='0.0.0.0', port=5000)
    
    def run(self, scrape=False, analyze=False, predict=False, serve=False):
        """Run the application with specified options."""
        # Schedule regular jobs
        self.schedule_jobs()
        
        # Start the scheduler
        self.start_scheduler()
        
        # Run specific tasks if requested
        if scrape:
            self.scrape_data()
        
        if analyze and (scrape or self.todays_games):
            self.analyze_player_stats()
        
        if predict and (analyze or self.team_players_stats):
            self.generate_predictions()
        
        # Start the web server if requested
        if serve:
            self.start_web_server()
        
        return True


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='NBA Stats Predictor')
    parser.add_argument('--scrape', action='store_true', help='Scrape today\'s games and lineups')
    parser.add_argument('--analyze', action='store_true', help='Analyze player stats for today\'s games')
    parser.add_argument('--predict', action='store_true', help='Generate predictions for today\'s games')
    parser.add_argument('--serve', action='store_true', help='Start the web application server')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Create and run the application
    app = StatLemonApplication()
    app.run(
        scrape=args.scrape,
        analyze=args.analyze,
        predict=args.predict,
        serve=args.serve
    )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())