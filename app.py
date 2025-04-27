import os
import json
import logging
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, redirect, url_for

from data.database import Database
from scrapers.game_scraper import NBAApiScraper
from scrapers.player_scraper import RotowireScraper
from analysis.player_stats import PlayerStatsCollector
from analysis.time_series import PlayerTimeSeriesAnalyzer, GamePredictor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('app')

# Initialize Flask app
app = Flask(__name__)

# Initialize components
db = Database()
stats_collector = PlayerStatsCollector()
time_series_analyzer = PlayerTimeSeriesAnalyzer(min_games=10)
game_predictor = GamePredictor(min_games=10)

# Cache for player stats to reduce API calls
player_stats_cache = {}
# Cache TTL in seconds (1 hour)
CACHE_TTL = 3600

# Global cache for predictions
prediction_cache = {}


def get_player_stats(player_name, force_refresh=False):
    """Get player stats with caching to reduce API calls."""
    current_time = datetime.now()
    
    # Check if we have cached data that's still valid
    if not force_refresh and player_name in player_stats_cache:
        cache_time, stats = player_stats_cache[player_name]
        if (current_time - cache_time).total_seconds() < CACHE_TTL:
            logger.info(f"Using cached stats for {player_name}")
            return stats
    
    # Get fresh stats from the collector
    logger.info(f"Fetching fresh stats for {player_name}")
    stats = stats_collector.get_player_stats(player_name)
    
    # Cache the results
    if stats is not None:
        player_stats_cache[player_name] = (current_time, stats)
    
    return stats


def get_player_prediction(player_name, days_ahead=1, opponent_team=None):
    """Get player performance prediction."""
    # Get player stats
    player_stats = get_player_stats(player_name)
    
    if player_stats is None:
        logger.warning(f"No stats available for {player_name}")
        return None
    
    # Get prediction
    prediction = time_series_analyzer.forecast_player_performance(
        player_name, player_stats, opponent_team)
    
    return prediction


@app.route('/')
def index():
    """Render the homepage."""
    return render_template('index.html')


@app.route('/about')
def about():
    """Render the about page."""
    return render_template('about.html')


@app.route('/player/<player_name>')
def player_profile(player_name):
    """Render a player's profile page."""
    # Get player stats
    player_stats = get_player_stats(player_name)
    
    if player_stats is None:
        return render_template('error.html', message=f"Player {player_name} not found")
    
    # Get player prediction
    prediction = get_player_prediction(player_name)
    
    return render_template(
        'player.html',
        player_name=player_name,
        stats=player_stats,
        prediction=prediction
    )


@app.route('/compare')
def compare_form():
    """Render the player comparison form."""
    return render_template('compare.html')


@app.route('/compare/results', methods=['POST'])
def compare_results():
    """Process player comparison and show results."""
    player1 = request.form.get('player1')
    player2 = request.form.get('player2')
    
    if not player1 or not player2:
        return render_template('error.html', message="Please provide two player names")
    
    # Get stats and predictions for both players
    player1_stats = get_player_stats(player1)
    player2_stats = get_player_stats(player2)
    
    if player1_stats is None or player2_stats is None:
        return render_template('error.html', message="One or both players not found")
    
    player1_prediction = get_player_prediction(player1)
    player2_prediction = get_player_prediction(player2)
    
    return render_template(
        'compare_results.html',
        player1=player1,
        player2=player2,
        player1_stats=player1_stats,
        player2_stats=player2_stats,
        player1_prediction=player1_prediction,
        player2_prediction=player2_prediction
    )


@app.route('/lineup_builder')
def lineup_builder():
    """Render the lineup builder page."""
    return render_template('lineup_builder.html')


@app.route('/api/search_player')
def search_player():
    """API endpoint for player name autocomplete."""
    query = request.args.get('q', '').strip()
    
    if not query or len(query) < 2:
        return jsonify([])
    
    # Simple in-memory search based on player names we've seen
    matching_players = []
    for player_name in player_stats_cache.keys():
        if query.lower() in player_name.lower():
            matching_players.append(player_name)
    
    # If no matches in cache, search using the player_stats collector
    if not matching_players:
        # Here we could search for players, but that would require
        # integrating with the NBA API's player search functionality
        pass
    
    return jsonify(matching_players[:10])  # Limit to 10 results


@app.route('/api/player_stats/<player_name>')
def api_player_stats(player_name):
    """API endpoint for player stats."""
    # Parse optional parameters
    force_refresh = request.args.get('refresh', 'false').lower() == 'true'
    
    # Get stats
    stats = get_player_stats(player_name, force_refresh=force_refresh)
    
    if stats is None:
        return jsonify({'error': f"Player {player_name} not found"}), 404
    
    # Convert to JSON-serializable format
    stats_json = stats.to_dict(orient='records')
    
    return jsonify({
        'player': player_name,
        'stats': stats_json
    })


@app.route('/api/player_prediction/<player_name>')
def api_player_prediction(player_name):
    """API endpoint for player performance prediction."""
    # Parse optional parameters
    days_ahead = int(request.args.get('days_ahead', '1'))
    opponent_team = request.args.get('opponent')
    
    # Get prediction
    prediction = get_player_prediction(
        player_name, days_ahead=days_ahead, opponent_team=opponent_team)
    
    if prediction is None:
        return jsonify({'error': f"Could not generate prediction for {player_name}"}), 404
    
    # Convert prediction to JSON-serializable format
    prediction_json = {}
    for metric, values in prediction.items():
        prediction_json[metric] = {
            'forecast': float(values['forecast']),
            'lower_bound': float(values['lower_bound']),
            'upper_bound': float(values['upper_bound']),
            'method': values['method'],
            'confidence': values['confidence']
        }
    
    return jsonify({
        'player': player_name,
        'prediction': prediction_json,
        'opponent_team': opponent_team,
        'days_ahead': days_ahead
    })


@app.route('/api/compare_players')
def api_compare_players():
    """API endpoint for comparing multiple players."""
    # Get player names from query parameter (comma-separated)
    player_names = request.args.get('players', '').split(',')
    player_names = [name.strip() for name in player_names if name.strip()]
    
    if not player_names:
        return jsonify({'error': "No players specified"}), 400
    
    # Get predictions for each player
    comparisons = {}
    for player_name in player_names:
        prediction = get_player_prediction(player_name)
        if prediction:
            comparisons[player_name] = {}
            for metric, values in prediction.items():
                comparisons[player_name][metric] = float(values['forecast'])
    
    return jsonify({
        'comparisons': comparisons
    })


@app.route('/api/team_prediction')
def api_team_prediction():
    """API endpoint for team performance prediction."""
    team_name = request.args.get('team')
    opponent_team = request.args.get('opponent')
    
    if not team_name or not opponent_team:
        return jsonify({'error': "Team and opponent must be specified"}), 400
    
    # This is a placeholder - in a real implementation, you would:
    # 1. Get the team's lineup from the database
    # 2. Collect stats for each player
    # 3. Use the game predictor to generate a team prediction
    
    return jsonify({
        'message': "Team prediction functionality not implemented in MVP"
    })


@app.route('/api/lineup_prediction', methods=['POST'])
def api_lineup_prediction():
    """API endpoint for custom lineup performance prediction."""
    try:
        # Get lineup data from request
        lineup_data = request.json
        
        if not lineup_data or 'players' not in lineup_data:
            return jsonify({'error': "Invalid lineup data"}), 400
        
        player_names = lineup_data.get('players', [])
        opponent_team = lineup_data.get('opponent')
        
        if not player_names:
            return jsonify({'error': "No players in lineup"}), 400
        
        # Get stats for each player
        player_stats = {}
        for player_name in player_names:
            stats = get_player_stats(player_name)
            if stats is not None:
                player_stats[player_name] = stats
        
        # Get individual predictions
        player_predictions = {}
        for player_name, stats in player_stats.items():
            prediction = time_series_analyzer.forecast_player_performance(
                player_name, stats, opponent_team)
            if prediction:
                player_predictions[player_name] = prediction
        
        # Aggregate metrics (simple average for MVP)
        aggregate_metrics = {}
        for metric in ['PTS_PER_MIN', 'FG_PCT', 'TS_PCT', 'GAME_SCORE', 'PLUS_MINUS']:
            values = []
            for player, preds in player_predictions.items():
                if metric in preds:
                    values.append(preds[metric]['forecast'])
            
            if values:
                aggregate_metrics[metric] = sum(values) / len(values)
        
        return jsonify({
            'lineup': player_names,
            'opponent': opponent_team,
            'player_predictions': player_predictions,
            'aggregate_metrics': aggregate_metrics
        })
        
    except Exception as e:
        logger.error(f"Error processing lineup prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/schedule')
def schedule():
    """Render the schedule page."""
    # Get today's date
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Get games for today (placeholder - would come from database in full implementation)
    games = []
    
    return render_template('schedule.html', games=games, date=today)


@app.route('/matchup_simulator')
def matchup_simulator():
    """Render the matchup simulator page."""
    return render_template('matchup_simulator.html')


@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    return render_template('error.html', message="Page not found"), 404


@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    return render_template('error.html', message="Server error"), 500


# Run the app if executed directly
if __name__ == '__main__':
    # Make sure the database is initialized
    db.initialize_database()
    
    # Start the Flask app in debug mode
    app.run(debug=True, host='0.0.0.0')