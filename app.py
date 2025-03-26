import os
import logging
from datetime import datetime
from typing import Dict, List, Any

from flask import Flask, render_template, jsonify, request, redirect, url_for

from data.database import Database
from analysis.sentiment import SentimentAnalyzer
from analysis.predictor import NBAPredictor
from reddit.collector import RedditCollector
from scrapers.game_scraper import NBAApiScraper

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('app')

# Initialize Flask application
app = Flask(__name__)

# Initialize database
db = Database()
db.initialize_database()

# Initialize required modules
sentiment_analyzer = SentimentAnalyzer(db=db)
predictor = NBAPredictor(db=db, sentiment_analyzer=sentiment_analyzer)
collector = RedditCollector(db=db)
nba_scraper = NBAApiScraper(db=db)

# Team color mapping for visualization
TEAM_COLORS = {
    'Atlanta Hawks': '#E03A3E',
    'Boston Celtics': '#007A33',
    'Brooklyn Nets': '#000000',
    'Charlotte Hornets': '#1D1160',
    'Chicago Bulls': '#CE1141',
    'Cleveland Cavaliers': '#860038',
    'Dallas Mavericks': '#00538C',
    'Denver Nuggets': '#0E2240',
    'Detroit Pistons': '#C8102E',
    'Golden State Warriors': '#1D428A',
    'Houston Rockets': '#CE1141',
    'Indiana Pacers': '#002D62',
    'Los Angeles Clippers': '#C8102E',
    'Los Angeles Lakers': '#552583',
    'Memphis Grizzlies': '#5D76A9',
    'Miami Heat': '#98002E',
    'Milwaukee Bucks': '#00471B',
    'Minnesota Timberwolves': '#0C2340',
    'New Orleans Pelicans': '#0C2340',
    'New York Knicks': '#F58426',
    'Oklahoma City Thunder': '#007AC1',
    'Orlando Magic': '#0077C0',
    'Philadelphia 76ers': '#006BB6',
    'Phoenix Suns': '#1D1160',
    'Portland Trail Blazers': '#E03A3E',
    'Sacramento Kings': '#5A2D81',
    'San Antonio Spurs': '#C4CED4',
    'Toronto Raptors': '#CE1141',
    'Utah Jazz': '#002B5C',
    'Washington Wizards': '#002B5C'
}

# Utility functions
def format_datetime(timestamp: str) -> str:
    """Format an ISO timestamp to a readable date/time string."""
    try:
        dt = datetime.fromisoformat(timestamp)
        return dt.strftime("%b %d, %Y %I:%M %p")
    except:
        return timestamp

def format_percent(value: float) -> str:
    """Format a probability as a percentage string."""
    return f"{value * 100:.1f}%"

def get_team_color(team_name: str) -> str:
    """Get the primary color for a team."""
    return TEAM_COLORS.get(team_name, '#888888')  # Default to gray if team not found

# Register custom template filters
app.jinja_env.filters['format_datetime'] = format_datetime
app.jinja_env.filters['format_percent'] = format_percent
app.jinja_env.filters['team_color'] = get_team_color

# Routes
@app.route('/')
def index():
    """Home page with today's game predictions."""
    # Get today's games and predictions
    games = db.get_upcoming_games(days_ahead=0)
    
    # Prepare game data with predictions
    game_data = []
    
    for game in games:
        game_id = game['game_id']
        prediction = db.get_latest_prediction(game_id)
        
        if prediction:
            win_probability = prediction['home_win_probability']
            predicted_winner = game['home_team'] if win_probability > 0.5 else game['away_team']
            confidence = abs(win_probability - 0.5) * 2  # Scale to 0-1
        else:
            win_probability = 0.5
            predicted_winner = None
            confidence = 0
        
        game_info = {
            'game_id': game_id,
            'home_team': game['home_team'],
            'away_team': game['away_team'],
            'game_time': game['game_time'],
            'win_probability': win_probability,
            'predicted_winner': predicted_winner,
            'confidence': confidence,
            'prediction_available': prediction is not None
        }
        
        game_data.append(game_info)
    
    # Sort games by start time
    game_data.sort(key=lambda g: g['game_time'])
    
    return render_template(
        'index.html',
        games=game_data,
        current_date=datetime.now().strftime("%A, %B %d, %Y")
    )

@app.route('/games')
def games():
    """Page showing upcoming games."""
    days_ahead = int(request.args.get('days', 7))
    games = db.get_upcoming_games(days_ahead=days_ahead)
    
    # Group games by date
    games_by_date = {}
    for game in games:
        game_date = game['game_date']
        if game_date not in games_by_date:
            games_by_date[game_date] = []
        games_by_date[game_date].append(game)
    
    # Sort dates
    sorted_dates = sorted(games_by_date.keys())
    
    return render_template(
        'games.html',
        games_by_date=games_by_date,
        dates=sorted_dates,
        days_ahead=days_ahead
    )

@app.route('/game/<game_id>')
def game_detail(game_id):
    """Page showing details for a specific game."""
    game = db.get_game(game_id)
    
    if not game:
        return render_template('error.html', message=f"Game not found: {game_id}")
    
    # Get the latest prediction
    prediction = db.get_latest_prediction(game_id)
    
    # Get player data for both teams
    home_team = game['home_team']
    away_team = game['away_team']
    
    home_players = db.get_players_by_team(home_team)
    away_players = db.get_players_by_team(away_team)
    
    # Get team sentiment data
    home_sentiment = sentiment_analyzer.get_team_sentiment(home_team, hours_back=24)
    away_sentiment = sentiment_analyzer.get_team_sentiment(away_team, hours_back=24)
    
    return render_template(
        'game_detail.html',
        game=game,
        prediction=prediction,
        home_players=home_players,
        away_players=away_players,
        home_sentiment=home_sentiment,
        away_sentiment=away_sentiment
    )

@app.route('/team/<team_name>')
def team_detail(team_name):
    """Page showing details for a specific team."""
    # Get team data
    players = db.get_players_by_team(team_name)
    
    # Get recent sentiment data
    sentiment = sentiment_analyzer.get_team_sentiment(team_name, hours_back=48)
    
    # Get recent posts mentioning this team
    posts = db.get_reddit_posts_by_team(team_name, days_back=2)
    
    return render_template(
        'team_detail.html',
        team_name=team_name,
        players=players,
        sentiment=sentiment,
        posts=posts
    )

@app.route('/refresh', methods=['POST'])
def refresh_data():
    """API endpoint to refresh data."""
    try:
        action = request.form.get('action', 'all')
        
        if action == 'games' or action == 'all':
            # Scrape and save games
            game_count = nba_scraper.scrape_and_save_games(days_ahead=1)
            logger.info(f"Scraped and saved {game_count} games")
        
        if action == 'players' or action == 'all':
            # Scrape and save player data
            players_saved = nba_scraper.scrape_and_save_players()
            logger.info(f"Saved {players_saved} players")
        
        if action == 'reddit' or action == 'all':
            # Get today's games to find teams playing today
            games = db.get_upcoming_games(days_ahead=0)
            today_teams = set()
            
            for game in games:
                today_teams.add(game['home_team'])
                today_teams.add(game['away_team'])
            
            # Collect Reddit data for today's teams
            results = collector.collect_today_teams_data(list(today_teams), days_back=1)
            
            total_posts = sum(p for p, _ in results.values())
            total_comments = sum(c for _, c in results.values())
            
            logger.info(f"Collected {total_posts} posts and {total_comments} comments")
        
        if action == 'sentiment' or action == 'all':
            # Analyze sentiment for recent content
            posts_analyzed, comments_analyzed = sentiment_analyzer.analyze_recent_content(hours_back=24)
            logger.info(f"Analyzed {posts_analyzed} posts and {comments_analyzed} comments")
        
        if action == 'predictions' or action == 'all':
            # Make predictions for today's games
            predictions = predictor.predict_today_games(hours_back=24)
            logger.info(f"Generated {len(predictions)} predictions")
        
        return redirect(url_for('index'))
    
    except Exception as e:
        logger.error(f"Error refreshing data: {str(e)}")
        return render_template('error.html', message=f"Error refreshing data: {str(e)}")

@app.route('/about')
def about():
    """About page with project information."""
    return render_template('about.html')

@app.route('/api/games/today')
def api_games_today():
    """API endpoint for today's games and predictions."""
    games = db.get_upcoming_games(days_ahead=0)
    
    game_data = []
    for game in games:
        game_id = game['game_id']
        prediction = db.get_latest_prediction(game_id)
        
        game_info = {
            'game_id': game_id,
            'home_team': game['home_team'],
            'away_team': game['away_team'],
            'game_time': game['game_time'],
            'venue': game['venue'],
            'prediction': prediction['home_win_probability'] if prediction else None,
            'prediction_timestamp': prediction['prediction_timestamp'] if prediction else None
        }
        
        game_data.append(game_info)
    
    return jsonify(games=game_data)

@app.route('/api/team/<team_name>/sentiment')
def api_team_sentiment(team_name):
    """API endpoint for team sentiment data."""
    hours = int(request.args.get('hours', 24))
    sentiment = sentiment_analyzer.get_team_sentiment(team_name, hours_back=hours)
    
    return jsonify(sentiment)

@app.errorhandler(404)
def page_not_found(e):
    """Custom 404 page."""
    return render_template('error.html', message='Page not found'), 404

@app.errorhandler(500)
def server_error(e):
    """Custom 500 page."""
    logger.error(f"Server error: {str(e)}")
    return render_template('error.html', message='Server error'), 500

if __name__ == '__main__':
    # Create template directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000)