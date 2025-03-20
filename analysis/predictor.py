import logging
import datetime
from typing import List, Dict, Any, Optional, Tuple

from data.database import Database
from analysis.sentiment import SentimentAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('analysis.predictor')


class NBAPredictor:
    """
    Predicts NBA game outcomes based on sentiment analysis and team data.
    """
    
    def __init__(self, db: Optional[Database] = None, sentiment_analyzer: Optional[SentimentAnalyzer] = None):
        """
        Initialize the NBA game predictor.
        
        Args:
            db: Database instance (optional, creates a new one if None)
            sentiment_analyzer: SentimentAnalyzer instance (optional, creates a new one if None)
        """
        # Initialize database (create a new one if not provided)
        self.db = db if db else Database()
        
        # Initialize sentiment analyzer (create a new one if not provided)
        self.sentiment_analyzer = sentiment_analyzer if sentiment_analyzer else SentimentAnalyzer(db=self.db)
        
        # Model weights for prediction factors
        self.weights = {
            'sentiment': 0.7,     # Weight for fan sentiment score
            'player_status': 0.3  # Weight for player availability score
        }
    
    def _calculate_player_status_score(self, team: str) -> float:
        """
        Calculate a score based on player status for a team.
        
        Args:
            team: The team name
            
        Returns:
            Player status score between 0 and 1
        """
        try:
            # Get all players for the team
            players = self.db.get_players_by_team(team)
            
            if not players:
                return 0.5  # Neutral score if no player data
            
            # Count players by status
            status_counts = {'active': 0, 'questionable': 0, 'out': 0, 'unknown': 0}
            
            for player in players:
                status = player.get('status', 'unknown')
                status_counts[status] = status_counts.get(status, 0) + 1
            
            total_players = len(players)
            
            # Calculate availability percentage with weighted statuses
            weighted_available = (
                status_counts['active'] + 
                0.5 * status_counts['questionable'] + 
                0.0 * status_counts['out'] + 
                0.5 * status_counts['unknown']
            )
            
            availability_score = weighted_available / total_players if total_players > 0 else 0.5
            
            # Normalize to 0-1 range (0.5 is neutral)
            normalized_score = 0.5 + (availability_score - 0.5) * 0.5
            
            return normalized_score
            
        except Exception as e:
            logger.error(f"Error calculating player status score for {team}: {str(e)}")
            return 0.5  # Neutral score on error
    
    def _normalize_sentiment_score(self, sentiment_value: float) -> float:
        """
        Normalize a sentiment score to the 0-1 range.
        
        Args:
            sentiment_value: Raw sentiment score (-1 to 1)
            
        Returns:
            Normalized sentiment score (0 to 1)
        """
        # Convert from [-1, 1] to [0, 1]
        return (sentiment_value + 1) / 2
    
    def predict_game(self, game_id: str, hours_back: int = 24) -> Dict[str, Any]:
        """
        Predict the outcome of a specific game.
        
        Args:
            game_id: The ID of the game to predict
            hours_back: Number of hours to look back for sentiment data
            
        Returns:
            Dictionary containing prediction data
        """
        try:
            # Get game details
            game = self.db.get_game(game_id)
            
            if not game:
                logger.error(f"Game not found: {game_id}")
                return {'error': 'Game not found'}
            
            home_team = game['home_team']
            away_team = game['away_team']
            
            # Get sentiment data for both teams
            home_sentiment = self.sentiment_analyzer.get_team_sentiment(home_team, hours_back=hours_back)
            away_sentiment = self.sentiment_analyzer.get_team_sentiment(away_team, hours_back=hours_back)
            
            # Calculate player status scores
            home_player_score = self._calculate_player_status_score(home_team)
            away_player_score = self._calculate_player_status_score(away_team)
            
            # Get normalized sentiment scores (0-1 range)
            home_sentiment_norm = self._normalize_sentiment_score(home_sentiment.get('avg_sentiment', 0.0))
            away_sentiment_norm = self._normalize_sentiment_score(away_sentiment.get('avg_sentiment', 0.0))
            
            # Calculate combined scores with weights
            home_score = (
                self.weights['sentiment'] * home_sentiment_norm +
                self.weights['player_status'] * home_player_score
            )
            
            away_score = (
                self.weights['sentiment'] * away_sentiment_norm +
                self.weights['player_status'] * away_player_score
            )
            
            # Calculate home win probability
            # Use a logistic function to convert score difference to probability
            score_diff = home_score - away_score
            home_advantage = 0.05  # Small boost for home team
            
            # Apply logistic transformation to get probability
            import math
            home_win_prob = 1 / (1 + math.exp(-10 * (score_diff + home_advantage)))
            
            # Round to 4 decimal places for readability
            home_win_prob = round(home_win_prob, 4)
            
            # Create prediction data
            prediction_data = {
                'game_id': game_id,
                'home_team_sentiment': home_sentiment.get('avg_sentiment', 0.0),
                'away_team_sentiment': away_sentiment.get('avg_sentiment', 0.0),
                'home_win_probability': home_win_prob,
                'prediction_timestamp': datetime.datetime.now().isoformat(),
                'home_team': home_team,
                'away_team': away_team,
                'home_player_score': home_player_score,
                'away_player_score': away_player_score,
                'confidence': min(
                    home_sentiment.get('confidence', 0.0),
                    away_sentiment.get('confidence', 0.0)
                )
            }
            
            return prediction_data
            
        except Exception as e:
            logger.error(f"Error predicting game {game_id}: {str(e)}")
            return {'error': str(e)}
    
    def save_prediction(self, prediction_data: Dict[str, Any]) -> Optional[int]:
        """
        Save a prediction to the database.
        
        Args:
            prediction_data: Dictionary containing prediction data
            
        Returns:
            prediction_id if successful, None otherwise
        """
        try:
            # Extract required fields for database
            db_prediction = {
                'game_id': prediction_data['game_id'],
                'home_team_sentiment': prediction_data['home_team_sentiment'],
                'away_team_sentiment': prediction_data['away_team_sentiment'],
                'home_win_probability': prediction_data['home_win_probability'],
                'prediction_timestamp': prediction_data['prediction_timestamp']
            }
            
            # Insert prediction
            prediction_id = self.db.insert_prediction(db_prediction)
            
            # Also update the prediction in the games table
            self.db.update_game(prediction_data['game_id'], {
                'prediction': prediction_data['home_win_probability']
            })
            
            return prediction_id
            
        except Exception as e:
            logger.error(f"Error saving prediction: {str(e)}")
            return None
    
    def predict_and_save_game(self, game_id: str, hours_back: int = 24) -> Dict[str, Any]:
        """
        Predict a game outcome and save the result to the database.
        
        Args:
            game_id: The ID of the game to predict
            hours_back: Number of hours to look back for sentiment data
            
        Returns:
            Dictionary containing prediction data and save status
        """
        # Get prediction
        prediction = self.predict_game(game_id, hours_back)
        
        if 'error' in prediction:
            return prediction
        
        # Save prediction
        prediction_id = self.save_prediction(prediction)
        
        # Add save status to result
        if prediction_id:
            prediction['saved'] = True
            prediction['prediction_id'] = prediction_id
        else:
            prediction['saved'] = False
        
        return prediction
    
    def predict_today_games(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """
        Predict outcomes for all of today's games.
        
        Args:
            hours_back: Number of hours to look back for sentiment data
            
        Returns:
            List of dictionaries containing prediction data
        """
        # Get today's games
        today_games = self.db.get_upcoming_games(days_ahead=0)
        
        if not today_games:
            logger.warning("No games found for today")
            return []
        
        predictions = []
        
        # Predict each game
        for game in today_games:
            try:
                prediction = self.predict_and_save_game(game['game_id'], hours_back)
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Error predicting game {game['game_id']}: {str(e)}")
                continue
        
        return predictions


# Example usage
if __name__ == "__main__":
    # Set up predictor
    db = Database()
    db.initialize_database()
    
    predictor = NBAPredictor(db=db)
    
    # Make predictions for today's games
    predictions = predictor.predict_today_games()
    
    # Print predictions
    print(f"Predictions for today's games ({len(predictions)} games):")
    for pred in predictions:
        if 'error' in pred:
            print(f"Error: {pred['error']}")
            continue
            
        home_team = pred['home_team']
        away_team = pred['away_team']
        win_prob = pred['home_win_probability'] * 100
        
        # Format as percentage with 1 decimal place
        win_prob_str = f"{win_prob:.1f}%"
        
        # Show sentiment scores
        home_sentiment = pred['home_team_sentiment']
        away_sentiment = pred['away_team_sentiment']
        
        print(f"{away_team} @ {home_team}: {home_team} win probability = {win_prob_str}")
        print(f"  Sentiment: {home_team} = {home_sentiment:.3f}, {away_team} = {away_sentiment:.3f}")
        print(f"  Saved: {pred.get('saved', False)}")
        print()