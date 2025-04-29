import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import warnings

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('analysis.time_series')

# Suppress statsmodels warnings
warnings.filterwarnings("ignore", category=UserWarning, module='statsmodels')


class PlayerTimeSeriesAnalyzer:
    """
    Analyzes player statistics using time series methods (ARIMA/SARIMA).
    Makes predictions for upcoming games based on historical performance patterns.
    
    Key features:
    - Automatic parameter selection for SARIMA models
    - Handles seasonality in player performance
    - Accounts for team matchups and other contextual factors
    - Computes confidence intervals for predictions
    - Detects player-specific patterns and trends
    """
    
    def __init__(self, min_games: int = 8, max_games: int = 30):
        """Create a new analyzer.

        Parameters
        ----------
        min_games : int, default 8
            Minimum recent games required to fit a model (lowered from 10 so
            bench contributors aren’t ignored).
        max_games : int, default 30
            Maximum historical games considered.
        """
        self.min_games = min_games
        self.max_games = max_games

        # Default SARIMA hyper‑parameters (unchanged)
        self.default_order = (1, 1, 1)
        self.default_seasonal_order = (1, 0, 1, 5)

        # Cache of fitted models keyed by “<player>_<metric>”
        self.fitted_models: dict[str, Any] = {}

        # --------------------------------------------------------------
        # 𝐅𝐚𝐧𝐭𝐚𝐬𝐲‑𝐟𝐢𝐫𝐬𝐭 𝐦𝐞𝐭𝐫𝐢𝐜 𝐬𝐞𝐭 (Tasks 8 & 9)
        # minutes & volume → defence → shooting volume → efficiency/impact
        # --------------------------------------------------------------
        self.key_metrics: list[str] = [
            # volume / availability
            "MINUTES_PLAYED", "PTS", "REB", "AST", "STL", "BLK",
            # shooting volume
            "FGM", "FGA",
            # efficiency / impact
            "FG_PCT", "TS_PCT", "PLUS_MINUS", "GAME_SCORE",
            # keep ratings for team‑level work
            "OFF_RATING", "DEF_RATING",
        ]

        # Track model accuracy / evaluation
        self.model_metrics: dict[str, dict[str, float]] = {}
        # Store per‑player fallback averages
        self.player_averages: dict[str, dict[str, float]] = {}
    
    def preprocess_time_series(self, player_stats: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Preprocess player statistics for time series analysis.
        
        Args:
            player_stats: DataFrame with player game-by-game statistics
            
        Returns:
            Dictionary mapping metric names to processed time series
        """
        logger.info("Preprocessing player time series data")
        
        if player_stats is None or player_stats.empty:
            logger.warning("No player stats provided for preprocessing")
            return {}
        
        # Verify that we have a date column
        if 'GAME_DATE' not in player_stats.columns:
            logger.error("Game date column not found in player stats")
            return {}
        
        # Sort by date (earliest to latest) for proper time series analysis
        try:
            stats_df = player_stats.copy().sort_values('GAME_DATE', ascending=True)
        except Exception as e:
            logger.error(f"Error sorting player stats by date: {str(e)}")
            return {}
        
        # Create dictionary to store processed time series for each metric
        processed_series = {}
        
        # Process each key metric if it exists in the data
        for metric in self.key_metrics:
            if metric in stats_df.columns:
                try:
                    # Create time series with game date as index
                    metric_series = stats_df[metric].copy()
                    
                    # Convert to float to ensure numeric values
                    metric_series = pd.to_numeric(metric_series, errors='coerce')
                    
                    # Interpolate missing values (if any)
                    metric_series = metric_series.interpolate(method='linear', limit_direction='both')
                    
                    # Store in dictionary
                    processed_series[metric] = metric_series
                    
                    # Log basic stats
                    logger.info(f"Processed {metric} time series: mean={metric_series.mean():.3f}, std={metric_series.std():.3f}")
                    
                except Exception as e:
                    logger.error(f"Error processing time series for {metric}: {str(e)}")
            else:
                logger.warning(f"Metric {metric} not found in player stats")
        
        return processed_series
    
    def check_stationarity(self, time_series: pd.Series) -> Tuple[bool, float, Dict[str, float]]:
        """
        Check if a time series is stationary using the Augmented Dickey-Fuller test.
        
        Args:
            time_series: Time series to check
            
        Returns:
            Tuple of (is_stationary, p_value, test_results)
        """
        try:
            # Run Augmented Dickey-Fuller test
            result = adfuller(time_series.dropna())
            
            # Get test statistics
            test_stat = result[0]
            p_value = result[1]
            critical_values = result[4]
            
            # Check if stationary (p_value < 0.05)
            is_stationary = p_value < 0.05
            
            # Create results dictionary
            test_results = {
                'test_statistic': test_stat,
                'p_value': p_value,
                'critical_values': critical_values
            }
            
            if is_stationary:
                logger.info(f"Time series is stationary (p-value: {p_value:.4f})")
            else:
                logger.info(f"Time series is not stationary (p-value: {p_value:.4f})")
            
            return is_stationary, p_value, test_results
            
        except Exception as e:
            logger.error(f"Error testing stationarity: {str(e)}")
            return False, 1.0, {'error': str(e)}
    
    def auto_select_sarima_parameters(self, time_series: pd.Series, max_p: int = 3, max_d: int = 2, 
                                    max_q: int = 3, max_P: int = 2, max_D: int = 1, 
                                    max_Q: int = 2, max_s: int = 10) -> Tuple[Tuple, Tuple]:
        """
        Automatically select optimal SARIMA parameters using a grid search approach.
        Uses a simplified approach for computational efficiency.
        
        Args:
            time_series: Time series data
            max_p, max_d, max_q: Maximum values for non-seasonal parameters
            max_P, max_D, max_Q, max_s: Maximum values for seasonal parameters
            
        Returns:
            Tuple of (order, seasonal_order) with best parameters
        """
        # Check if we have enough data
        if len(time_series) < self.min_games:
            logger.warning(f"Not enough data for parameter selection (min {self.min_games} required)")
            return self.default_order, self.default_seasonal_order
        
        try:
            logger.info("Beginning automatic SARIMA parameter selection")
            
            # First, check stationarity and determine d
            is_stationary, _, _ = self.check_stationarity(time_series)
            d = 0 if is_stationary else 1
            
            # For computational efficiency, we'll use a smarter approach rather than a full grid search
            # 1. First, determine seasonality period (s) by looking at autocorrelation
            
            # Calculate ACF
            acf_values = acf(time_series.dropna(), nlags=min(max_s, len(time_series)//2))
            
            # Find local maxima in ACF (potential seasonal periods)
            s_candidates = [i for i in range(2, len(acf_values)) 
                         if i < max_s and acf_values[i] > acf_values[i-1] and acf_values[i] > acf_values[i+1] 
                         and acf_values[i] > 0.2]
            
            # Default to 5 (approximately weekly for NBA schedule) if no strong candidates
            s = 5 if not s_candidates else s_candidates[0]
            
            # 2. Start with sensible defaults for other parameters
            best_order = (1, d, 1)
            best_seasonal_order = (1, 0, 1, s)
            best_aic = float('inf')
            
            # 3. Try a few promising parameter combinations
            parameter_combinations = [
                ((1, d, 1), (0, 0, 0, 0)),  # Simple ARIMA
                ((1, d, 1), (1, 0, 1, s)),  # Basic SARIMA
                ((2, d, 1), (1, 0, 1, s)),  # Increased AR
                ((1, d, 2), (1, 0, 1, s)),  # Increased MA
            ]
            
            # Add more combinations if we have enough data
            if len(time_series) > 20:
                parameter_combinations.extend([
                    ((2, d, 2), (1, 0, 1, s)),
                    ((1, d, 1), (2, 0, 1, s)),
                    ((1, d, 1), (1, 0, 2, s))
                ])
            
            # Try each combination
            for order, seasonal_order in parameter_combinations:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        
                        # Fit model with current parameters
                        model = SARIMAX(
                            time_series.dropna(),
                            order=order,
                            seasonal_order=seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                        result = model.fit(disp=False, maxiter=200)
                        
                        # Check if this is better (lower AIC)
                        if result.aic < best_aic:
                            best_aic = result.aic
                            best_order = order
                            best_seasonal_order = seasonal_order
                            logger.info(f"Found better model: SARIMA{best_order}{best_seasonal_order} (AIC: {best_aic:.2f})")
                
                except Exception as e:
                    logger.debug(f"Error fitting SARIMA{order}{seasonal_order}: {str(e)}")
                    continue
            
            return best_order, best_seasonal_order
            
        except Exception as e:
            logger.error(f"Error in parameter selection: {str(e)}")
            return self.default_order, self.default_seasonal_order
    
    def fit_sarima_model(self, time_series: pd.Series, order: Tuple = None, 
                       seasonal_order: Tuple = None) -> Optional[Any]:
        """
        Fit a SARIMA model to the time series data.
        
        Args:
            time_series: Time series data
            order: ARIMA order (p, d, q)
            seasonal_order: Seasonal order (P, D, Q, s)
            
        Returns:
            Fitted SARIMA model or None if fitting fails
        """
        # Check if we have enough data
        if time_series is None or len(time_series) < self.min_games:
            logger.warning(f"Not enough data to fit SARIMA model (need {self.min_games}, got {len(time_series) if time_series is not None else 0})")
            return None
        
        # Auto-select parameters if not provided
        if order is None or seasonal_order is None:
            order, seasonal_order = self.auto_select_sarima_parameters(time_series)
        
        logger.info(f"Fitting SARIMA model with order={order}, seasonal_order={seasonal_order}")
        
        try:
            # Fit the SARIMA model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                model = SARIMAX(
                    time_series.dropna(),
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                
                fitted_result = model.fit(disp=False, maxiter=200)
                
                logger.info(f"Successfully fit SARIMA model (AIC: {fitted_result.aic:.2f})")
                return fitted_result
                
        except Exception as e:
            logger.error(f"Error fitting SARIMA model: {str(e)}")
            
            # Try a simpler ARIMA model as fallback
            try:
                logger.info("Trying simpler ARIMA model as fallback")
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    # Determine d based on stationarity
                    is_stationary, _, _ = self.check_stationarity(time_series)
                    d = 0 if is_stationary else 1
                    
                    # Simple ARIMA model
                    model = ARIMA(time_series.dropna(), order=(1, d, 1))
                    fitted_result = model.fit()
                    
                    logger.info(f"Successfully fit fallback ARIMA model (AIC: {fitted_result.aic:.2f})")
                    return fitted_result
            
            except Exception as fallback_error:
                logger.error(f"Error fitting fallback ARIMA model: {str(fallback_error)}")
                return None
    
    def compute_player_averages(self, player_stats: pd.DataFrame, player_name: str) -> Dict[str, float]:
        """
        Compute player averages for various metrics as a fallback.
        
        Args:
            player_stats: DataFrame with player statistics
            player_name: Name of the player
            
        Returns:
            Dictionary mapping metrics to their average values
        """
        averages = {}
        
        if player_stats is None or player_stats.empty:
            logger.warning(f"No stats available for {player_name}")
            return averages
        
        # Calculate averages for each key metric
        for metric in self.key_metrics:
            if metric in player_stats.columns:
                try:
                    # Get values, convert to numeric, and remove missing values
                    values = pd.to_numeric(player_stats[metric], errors='coerce').dropna()
                    
                    if len(values) > 0:
                        # Calculate average
                        avg = values.mean()
                        averages[metric] = avg
                        logger.info(f"{player_name} average {metric}: {avg:.3f}")
                        
                except Exception as e:
                    logger.error(f"Error calculating average for {metric}: {str(e)}")
        
        # Store in class variable for future reference
        self.player_averages[player_name] = averages
        
        return averages
    
    def analyze_player_vs_team(self, player_stats: pd.DataFrame, opponent_team: str) -> Dict[str, float]:
        """
        Analyze player performance against a specific opponent team.
        
        Args:
            player_stats: DataFrame with player statistics
            opponent_team: Name of the opponent team
            
        Returns:
            Dictionary with performance adjustments for each metric
        """
        # This is a placeholder for future implementation that would filter stats
        # by games against a specific opponent and calculate adjustments
        
        # For now, return a neutral adjustment (1.0 = no adjustment)
        return {metric: 1.0 for metric in self.key_metrics}
    
    def forecast_player_performance(self, player_name: str, player_stats: pd.DataFrame, 
                                  opponent_team: str = None, num_forecast: int = 1) -> Dict[str, Dict[str, Any]]:
        """
        Forecast player performance for upcoming games.
        
        Args:
            player_name: Name of the player
            player_stats: DataFrame with player's historical statistics
            opponent_team: Name of the opponent team (optional)
            num_forecast: Number of games to forecast
            
        Returns:
            Dictionary mapping metrics to forecast results
        """
        logger.info(f"Forecasting performance for {player_name} (against {opponent_team if opponent_team else 'any team'})")
        
        # Check if we have data
        if player_stats is None or player_stats.empty:
            logger.warning(f"No stats available for {player_name}")
            return {}
        
        # Compute player averages as fallback
        player_averages = self.compute_player_averages(player_stats, player_name)
        
        # Preprocess player stats into time series
        time_series_dict = self.preprocess_time_series(player_stats)
        
        # Get opponent-specific adjustments if an opponent is specified
        opponent_adjustments = {}
        if opponent_team:
            opponent_adjustments = self.analyze_player_vs_team(player_stats, opponent_team)
        
        # Dictionary to store forecast results
        forecasts = {}
        
        # Process each metric
        for metric in self.key_metrics:
            if metric not in time_series_dict:
                logger.warning(f"Metric {metric} not available for {player_name}")
                continue
            
            time_series = time_series_dict[metric]
            
            # Skip if not enough data
            if len(time_series) < self.min_games:
                logger.warning(f"Not enough data for {player_name} {metric} (need {self.min_games}, got {len(time_series)})")
                
                # Use player average as fallback if available
                if metric in player_averages:
                    avg_value = player_averages[metric]
                    adjustment = opponent_adjustments.get(metric, 1.0)
                    adjusted_value = avg_value * adjustment
                    
                    forecasts[metric] = {
                        'forecast': adjusted_value,
                        'lower_bound': adjusted_value * 0.85,  # Simple confidence interval
                        'upper_bound': adjusted_value * 1.15,
                        'method': 'average',
                        'confidence': 'low'
                    }
                
                continue
            
            # Generate a unique model key for this player and metric
            model_key = f"{player_name}_{metric}"
            
            # Check if we already have a fitted model
            if model_key in self.fitted_models:
                model = self.fitted_models[model_key]
                logger.info(f"Using existing model for {player_name} {metric}")
            else:
                # Fit a new model
                model = self.fit_sarima_model(time_series)
                
                # Store the model if fitting was successful
                if model is not None:
                    self.fitted_models[model_key] = model
            
            # Generate forecast if we have a model
            if model is not None:
                try:
                    # Get forecast with confidence intervals
                    forecast_result = model.get_forecast(steps=num_forecast)
                    predicted_mean = forecast_result.predicted_mean.values
                    confidence_intervals = forecast_result.conf_int()
                    
                    # Get the forecast for the next game
                    forecast_value = predicted_mean[0]
                    lower_bound = confidence_intervals.iloc[0, 0]
                    upper_bound = confidence_intervals.iloc[0, 1]
                    
                    # Apply opponent adjustment if available
                    if metric in opponent_adjustments:
                        adjustment = opponent_adjustments[metric]
                        forecast_value *= adjustment
                        lower_bound *= adjustment
                        upper_bound *= adjustment
                        
                        logger.info(f"Applied opponent adjustment for {metric}: {adjustment:.2f}")
                    
                    # Store forecast results
                    forecasts[metric] = {
                        'forecast': forecast_value,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound,
                        'method': 'sarima',
                        'confidence': 'high'
                    }
                    
                    logger.info(f"SARIMA forecast for {player_name} {metric}: {forecast_value:.3f} [{lower_bound:.3f}, {upper_bound:.3f}]")
                    
                except Exception as e:
                    logger.error(f"Error generating forecast for {player_name} {metric}: {str(e)}")
                    
                    # Use player average as fallback
                    if metric in player_averages:
                        avg_value = player_averages[metric]
                        adjustment = opponent_adjustments.get(metric, 1.0)
                        adjusted_value = avg_value * adjustment
                        
                        forecasts[metric] = {
                            'forecast': adjusted_value,
                            'lower_bound': adjusted_value * 0.85,
                            'upper_bound': adjusted_value * 1.15,
                            'method': 'average',
                            'confidence': 'low'
                        }
                        
                        logger.info(f"Using average for {player_name} {metric}: {adjusted_value:.3f}")
            
            else:
                # Use player average as fallback if model fitting failed
                if metric in player_averages:
                    avg_value = player_averages[metric]
                    adjustment = opponent_adjustments.get(metric, 1.0)
                    adjusted_value = avg_value * adjustment
                    
                    forecasts[metric] = {
                        'forecast': adjusted_value,
                        'lower_bound': adjusted_value * 0.85,
                        'upper_bound': adjusted_value * 1.15,
                        'method': 'average',
                        'confidence': 'low'
                    }
                    
                    logger.info(f"Using average for {player_name} {metric}: {adjusted_value:.3f}")
        
        return forecasts
    
    def evaluate_model_performance(self, player_name: str, player_stats: pd.DataFrame, 
                                metric: str, test_size: int = 5) -> Dict[str, float]:
        """
        Evaluate the performance of the time series model by comparing predictions to actual values.
        
        Args:
            player_name: Name of the player
            player_stats: DataFrame with player's historical statistics
            metric: The metric to evaluate
            test_size: Number of most recent games to use for testing
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating model performance for {player_name} ({metric})")
        
        # Check if we have enough data
        if player_stats is None or player_stats.empty or len(player_stats) < self.min_games + test_size:
            logger.warning(f"Not enough data for {player_name} to evaluate model")
            return {}
        
        try:
            # Preprocess data
            time_series_dict = self.preprocess_time_series(player_stats)
            
            if metric not in time_series_dict:
                logger.warning(f"Metric {metric} not available for {player_name}")
                return {}
            
            time_series = time_series_dict[metric]
            
            # Split into training and test sets
            train_data = time_series.iloc[:-test_size]
            test_data = time_series.iloc[-test_size:]
            
            # Fit model on training data
            model = self.fit_sarima_model(train_data)
            
            if model is None:
                logger.warning(f"Could not fit model for {player_name} ({metric})")
                return {}
            
            # Generate forecasts for test period
            forecast_result = model.get_forecast(steps=test_size)
            forecast_values = forecast_result.predicted_mean.values
            
            # Calculate error metrics
            mse = mean_squared_error(test_data, forecast_values)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(test_data, forecast_values)
            
            # Mean absolute percentage error (MAPE)
            mape = np.mean(np.abs((test_data.values - forecast_values) / test_data.values)) * 100 if np.any(test_data != 0) else np.nan
            
            # Store metrics
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'mape': mape
            }
            
            # Store in class variable
            model_key = f"{player_name}_{metric}"
            self.model_metrics[model_key] = metrics
            
            logger.info(f"Model evaluation for {player_name} ({metric}): RMSE={rmse:.3f}, MAPE={mape:.2f}%")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model for {player_name} ({metric}): {str(e)}")
            return {}
    
    def forecast_team_performance(self, team_name: str, team_players_stats: Dict[str, pd.DataFrame],
                                opponent_team: str) -> Dict[str, Dict[str, Any]]:
        """
        Forecast the collective performance of a team based on individual player forecasts.
        
        Args:
            team_name: Name of the team
            team_players_stats: Dictionary mapping player names to their stats DataFrames
            opponent_team: Name of the opponent team
            
        Returns:
            Dictionary with team performance metrics
        """
        logger.info(f"Forecasting team performance for {team_name} vs {opponent_team}")
        
        # Check if we have player data
        if not team_players_stats:
            logger.warning(f"No player stats available for {team_name}")
            return {}
        
        # Dictionary to store team performance metrics
        team_performance = {}
        
        # Collect player forecasts
        player_forecasts = {}
        for player_name, player_stats in team_players_stats.items():
            player_forecast = self.forecast_player_performance(
                player_name, player_stats, opponent_team)
            
            if player_forecast:
                player_forecasts[player_name] = player_forecast
        
        if not player_forecasts:
            logger.warning(f"Could not generate forecasts for any players on {team_name}")
            return {}
        
        # Metrics to aggregate across the team
        team_metrics = {
            'TEAM_POINTS': 0,
            'TEAM_FG_PCT': [],
            'TEAM_TS_PCT': [],
            'TEAM_PLUS_MINUS': 0,
            'TEAM_OFF_RATING': [],
            'TEAM_DEF_RATING': []
        }
        
        # Calculate team metrics based on individual player predictions
        for player_name, forecast in player_forecasts.items():
            try:
                # Get player stats
                player_stats = team_players_stats[player_name]
                
                # Get most recent game's minutes (or average if not available)
                if 'MINUTES_PLAYED' in player_stats.columns:
                    minutes = player_stats['MINUTES_PLAYED'].mean()
                else:
                    minutes = 20  # Default assumption
                
                # Estimate points based on PTS_PER_MIN * minutes
                if 'PTS_PER_MIN' in forecast:
                    pts_per_min = forecast['PTS_PER_MIN']['forecast']
                    player_points = pts_per_min * minutes
                    team_metrics['TEAM_POINTS'] += player_points
                
                # Collect shooting percentages (weighted by minutes)
                if 'FG_PCT' in forecast:
                    team_metrics['TEAM_FG_PCT'].append((forecast['FG_PCT']['forecast'], minutes))
                
                if 'TS_PCT' in forecast:
                    team_metrics['TEAM_TS_PCT'].append((forecast['TS_PCT']['forecast'], minutes))
                
                # Sum plus-minus
                if 'PLUS_MINUS' in forecast:
                    team_metrics['TEAM_PLUS_MINUS'] += forecast['PLUS_MINUS']['forecast']
                
                # Collect offensive and defensive ratings (weighted by minutes)
                if 'OFF_RATING' in forecast:
                    team_metrics['TEAM_OFF_RATING'].append((forecast['OFF_RATING']['forecast'], minutes))
                
                if 'DEF_RATING' in forecast:
                    team_metrics['TEAM_DEF_RATING'].append((forecast['DEF_RATING']['forecast'], minutes))
                
            except Exception as e:
                logger.error(f"Error aggregating forecast for {player_name}: {str(e)}")
                continue
        
        # Calculate weighted averages for percentages and ratings
        for metric in ['TEAM_FG_PCT', 'TEAM_TS_PCT', 'TEAM_OFF_RATING', 'TEAM_DEF_RATING']:
            if team_metrics[metric]:
                values, weights = zip(*team_metrics[metric])
                team_performance[metric] = {
                    'forecast': np.average(values, weights=weights),
                    'method': 'weighted_average'
                }
        
        # Simple sum for points and plus-minus
        team_performance['TEAM_POINTS'] = {
            'forecast': team_metrics['TEAM_POINTS'],
            'method': 'sum'
        }
        
        team_performance['TEAM_PLUS_MINUS'] = {
            'forecast': team_metrics['TEAM_PLUS_MINUS'],
            'method': 'sum'
        }
        
        # Log team forecast
        logger.info(f"Team forecast for {team_name}: Points={team_performance['TEAM_POINTS']['forecast']:.1f}, "
                   f"Plus/Minus={team_performance['TEAM_PLUS_MINUS']['forecast']:.1f}")
        
        return team_performance
    
    def predict_game_outcome(self, home_team: str, away_team: str, 
                           home_players_stats: Dict[str, pd.DataFrame],
                           away_players_stats: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Predict the outcome of a game based on team and player forecasts.
        
        Args:
            home_team: Name of the home team
            away_team: Name of the away team
            home_players_stats: Dictionary mapping home player names to their stats
            away_players_stats: Dictionary mapping away player names to their stats
            
        Returns:
            Dictionary with game prediction details
        """
        logger.info(f"Predicting game outcome: {away_team} @ {home_team}")
        
        # Check if we have player data for both teams
        if not home_players_stats:
            logger.warning(f"No player stats available for {home_team}")
            return {'error': f"No player stats for {home_team}"}
        
        if not away_players_stats:
            logger.warning(f"No player stats available for {away_team}")
            return {'error': f"No player stats for {away_team}"}
        
        # Forecast team performances
        home_forecast = self.forecast_team_performance(home_team, home_players_stats, away_team)
        away_forecast = self.forecast_team_performance(away_team, away_players_stats, home_team)
        
        if not home_forecast or not away_forecast:
            msg = f"Could not generate forecasts for one or both teams: {home_team}, {away_team}"
            logger.warning(msg)
            return {'error': msg}
        
        # Estimate point differentials
        try:
            # Direct point prediction from TEAM_POINTS
            home_points = home_forecast.get('TEAM_POINTS', {}).get('forecast', 0)
            away_points = away_forecast.get('TEAM_POINTS', {}).get('forecast', 0)
            direct_point_diff = home_points - away_points
            
            # Plus/minus based prediction
            home_plus_minus = home_forecast.get('TEAM_PLUS_MINUS', {}).get('forecast', 0)
            away_plus_minus = away_forecast.get('TEAM_PLUS_MINUS', {}).get('forecast', 0)
            plus_minus_diff = home_plus_minus - away_plus_minus
            
            # Rating based prediction
            home_off_rating = home_forecast.get('TEAM_OFF_RATING', {}).get('forecast', 100)
            home_def_rating = home_forecast.get('TEAM_DEF_RATING', {}).get('forecast', 100)
            away_off_rating = away_forecast.get('TEAM_OFF_RATING', {}).get('forecast', 100)
            away_def_rating = away_forecast.get('TEAM_DEF_RATING', {}).get('forecast', 100)
            
            # Calculate expected point differential based on ratings
            # Higher offensive rating is better, lower defensive rating is better
            home_advantage = 3.5  # Home court advantage in points
            ratings_diff = (home_off_rating - away_def_rating) - (away_off_rating - home_def_rating) + home_advantage
            
            # Combine predictions (weighted average)
            weights = [0.4, 0.3, 0.3]  # Weights for direct points, plus/minus, and ratings
            predicted_diff = (
                weights[0] * direct_point_diff +
                weights[1] * plus_minus_diff +
                weights[2] * ratings_diff
            )
            
            # Determine winner
            if predicted_diff > 0:
                predicted_winner = home_team
                win_probability = 0.5 + min(0.45, abs(predicted_diff) / 20)  # Convert point diff to probability
            else:
                predicted_winner = away_team
                win_probability = 0.5 + min(0.45, abs(predicted_diff) / 20)
            
            # Create prediction result
            prediction = {
                'home_team': home_team,
                'away_team': away_team,
                'predicted_home_points': home_points,
                'predicted_away_points': away_points,
                'predicted_point_diff': predicted_diff,
                'predicted_winner': predicted_winner,
                'win_probability': win_probability,
                'home_forecast': home_forecast,
                'away_forecast': away_forecast
            }
            
            logger.info(f"Game prediction: {away_team} {away_points:.1f} @ {home_team} {home_points:.1f}")
            logger.info(f"Predicted winner: {predicted_winner} (probability: {win_probability:.2f})")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting game outcome: {str(e)}")
            return {'error': str(e)}


class GamePredictor:
    """
    High-level class that orchestrates the prediction process for NBA games.
    Integrates player statistics collection and time series analysis.
    """
    
    def __init__(self, min_games: int = 8):
        # Use revised default min_games so defensive specialists aren’t skipped
        self.time_series_analyzer = PlayerTimeSeriesAnalyzer(min_games=min_games)
        self.prediction_history: list[dict[str, Any]] = []
    
    def predict_game(self, game_data: Dict[str, Any], team_players_stats: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """
        Predict the outcome of a game based on game data and player statistics.
        
        Args:
            game_data: Dictionary with game information
            team_players_stats: Dictionary mapping team names to player stats dictionaries
            
        Returns:
            Dictionary with game prediction details
        """
        # Extract game information
        home_team = game_data.get('home_team')
        away_team = game_data.get('away_team')
        game_date = game_data.get('game_date')
        game_id = game_data.get('game_id')
        
        logger.info(f"Predicting game {game_id}: {away_team} @ {home_team} on {game_date}")
        
        # Get home and away player stats
        home_players_stats = team_players_stats.get(home_team, {})
        away_players_stats = team_players_stats.get(away_team, {})
        
        # Predict game outcome
        prediction = self.time_series_analyzer.predict_game_outcome(
            home_team, away_team, home_players_stats, away_players_stats)
        
        # Add game info to prediction
        prediction.update({
            'game_id': game_id,
            'game_date': game_date,
            'prediction_time': datetime.now().isoformat()
        })
        
        # Store in prediction history
        self.prediction_history.append(prediction)
        
        return prediction
    
    def evaluate_predictions(self, actual_results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate the accuracy of past predictions.
        
        Args:
            actual_results: Dictionary mapping game IDs to actual game results
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.prediction_history:
            logger.warning("No predictions to evaluate")
            return {}
        
        # Count correct winner predictions
        correct_winners = 0
        total_evaluated = 0
        
        # Collect point differentials for RMSE calculation
        predicted_diffs = []
        actual_diffs = []
        
        for prediction in self.prediction_history:
            game_id = prediction.get('game_id')
            
            if game_id not in actual_results:
                logger.warning(f"No actual result for game {game_id}")
                continue
            
            actual = actual_results[game_id]
            
            # Extract actual game data
            actual_home_points = actual.get('home_points', 0)
            actual_away_points = actual.get('away_points', 0)
            actual_diff = actual_home_points - actual_away_points
            actual_winner = prediction['home_team'] if actual_diff > 0 else prediction['away_team']
            
            # Compare with prediction
            predicted_winner = prediction.get('predicted_winner')
            predicted_diff = prediction.get('predicted_point_diff', 0)
            
            if predicted_winner == actual_winner:
                correct_winners += 1
            
            # Collect differentials
            predicted_diffs.append(predicted_diff)
            actual_diffs.append(actual_diff)
            
            total_evaluated += 1
        
        # Calculate metrics
        if total_evaluated > 0:
            winner_accuracy = correct_winners / total_evaluated
            
            # Calculate RMSE if we have differentials
            if predicted_diffs and actual_diffs:
                diff_rmse = np.sqrt(mean_squared_error(actual_diffs, predicted_diffs))
            else:
                diff_rmse = None
            
            metrics = {
                'winner_accuracy': winner_accuracy,
                'diff_rmse': diff_rmse,
                'total_evaluated': total_evaluated
            }
            
            logger.info(f"Prediction evaluation: Winner accuracy={winner_accuracy:.3f}, Diff RMSE={diff_rmse if diff_rmse else 'N/A'}")
            
            return metrics
        
        return {}


# Example usage when run as script
if __name__ == "__main__":
    from analysis.player_stats import PlayerStatsCollector
    
    # Initialize the player stats collector
    stats_collector = PlayerStatsCollector()
    
    # Initialize the time series analyzer
    time_series_analyzer = PlayerTimeSeriesAnalyzer(min_games=10)
    
    # Test with a single player
    player_name = "LeBron James"
    player_stats = stats_collector.get_player_stats(player_name, num_games=30)
    
    if player_stats is not None:
        print(f"\nForecasting performance for {player_name}:")
        forecast = time_series_analyzer.forecast_player_performance(player_name, player_stats)
        
        # Print forecast results for key metrics
        for metric, result in forecast.items():
            print(f"{metric}: {result['forecast']:.3f} [{result['lower_bound']:.3f}, {result['upper_bound']:.3f}]")