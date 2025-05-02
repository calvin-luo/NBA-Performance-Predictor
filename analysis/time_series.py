import logging
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import warnings

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('analysis.time_series')

# Suppress statsmodels warnings
warnings.filterwarnings("ignore", category=UserWarning, module='statsmodels')

# Define categories and key metrics
CATEGORIES = {
    "fantasy":    ["THREES", "TWOS", "FTM", "REB", "AST", "BLK", "STL", "TOV"],
    "volume":     ["MIN", "FGM", "FGA"],
    "efficiency": ["FG_PCT", "TS_PCT", "TOV", "THREE_PCT"],
}
KEY_METRICS = sorted({m for v in CATEGORIES.values() for m in v})  # 14 unique metrics (TOV appears twice)


class PlayerTimeSeriesAnalyzer:
    """
    Analyzes player statistics using time series methods (ARIMA/SARIMA).
    Makes predictions for upcoming games based on historical performance patterns.
    
    Key features:
    - Uses fixed SARIMA parameters for consistent modeling
    - Handles seasonality in player performance
    - Computes confidence intervals for predictions
    """
    
    def __init__(self, min_games: int = 10, max_games: int = 30):
        """
        Initialize the time series analyzer.
        
        Args:
            min_games: Minimum number of games needed for reliable analysis
            max_games: Maximum number of games to consider for time series analysis
        """
        self.min_games = min_games
        self.max_games = max_games
        
        # Fixed SARIMA parameters - simplified from grid search
        self.fixed_order = (1, 1, 1)         # p, d, q (non-seasonal components)
        self.fixed_seasonal_order = (1, 0, 1, 5)  # P, D, Q, S (seasonal components)
        
        # Key metrics to analyze - now using KEY_METRICS from constants
        self.key_metrics = KEY_METRICS
        
        # Add MIN if it's not already in the key metrics
        if 'MIN' not in self.key_metrics:
            self.key_metrics.append('MIN')
        
        # Dictionary to track player averages for fallback predictions
        self.player_averages = {}
    
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
    
    def fit_sarima_model(self, time_series: pd.Series) -> Optional[Any]:
        """
        Fit a SARIMA model to the time series data using fixed parameters.
        
        Args:
            time_series: Time series data
            
        Returns:
            Fitted SARIMA model or None if fitting fails
        """
        # Check if we have enough data
        if time_series is None or len(time_series) < self.min_games:
            logger.warning(f"Not enough data to fit SARIMA model (need {self.min_games}, got {len(time_series) if time_series is not None else 0})")
            return None
        
        logger.info(f"Fitting SARIMA model with fixed parameters: order={self.fixed_order}, seasonal_order={self.fixed_seasonal_order}")
        
        try:
            # Fit the SARIMA model with fixed parameters
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                model = SARIMAX(
                    time_series.dropna(),
                    order=self.fixed_order,
                    seasonal_order=self.fixed_seasonal_order,
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
    
    def forecast_player_performance(self, player_name: str, player_stats: pd.DataFrame, 
                                  num_forecast: int = 1) -> Dict[str, Dict[str, Any]]:
        """
        Forecast player performance for upcoming games.
        
        Args:
            player_name: Name of the player
            player_stats: DataFrame with player's historical statistics
            num_forecast: Number of games to forecast
            
        Returns:
            Dictionary mapping metrics to forecast results
        """
        logger.info(f"Forecasting performance for {player_name}")
        
        # Check if we have data
        if player_stats is None or player_stats.empty:
            logger.warning(f"No stats available for {player_name}")
            return {}
        
        # Compute player averages as fallback
        player_averages = self.compute_player_averages(player_stats, player_name)
        
        # Preprocess player stats into time series
        time_series_dict = self.preprocess_time_series(player_stats)
        
        # Dictionary to store forecast results
        forecast_dict = {}
        
        # Prioritize forecasting fantasy metrics and minutes
        priority_metrics = ['MIN'] + CATEGORIES['fantasy']
        
        # First process priority metrics (Minutes and Fantasy categories)
        for metric in priority_metrics:
            self._forecast_metric(metric, time_series_dict, player_averages, forecast_dict)
        
        # Then process remaining metrics
        for metric in self.key_metrics:
            if metric not in priority_metrics:
                self._forecast_metric(metric, time_series_dict, player_averages, forecast_dict)
        
        return forecast_dict
    
    def _forecast_metric(self, metric, time_series_dict, player_averages, forecast_dict):
        """Helper method to forecast a single metric and add it to the forecast dictionary"""
        if metric not in time_series_dict:
            if metric in player_averages:
                # Use player average as fallback if time series not available
                avg_value = player_averages[metric]
                forecast_dict[metric] = {
                    'forecast': avg_value,
                    'lower_bound': avg_value * 0.85,  # Simple confidence interval
                    'upper_bound': avg_value * 1.15,
                    'method': 'average',
                    'confidence': 'low'
                }
            return
        
        time_series = time_series_dict[metric]
        
        # Skip if not enough data
        if len(time_series) < self.min_games:
            # Use player average as fallback if available
            if metric in player_averages:
                avg_value = player_averages[metric]
                forecast_dict[metric] = {
                    'forecast': avg_value,
                    'lower_bound': avg_value * 0.85,  # Simple confidence interval
                    'upper_bound': avg_value * 1.15,
                    'method': 'average',
                    'confidence': 'low'
                }
            return
        
        # Fit a model
        model = self.fit_sarima_model(time_series)
        
        # Generate forecast if we have a model
        if model is not None:
            try:
                # Get forecast with confidence intervals
                forecast_result = model.get_forecast(steps=1)
                predicted_mean = forecast_result.predicted_mean.values
                confidence_intervals = forecast_result.conf_int()
                
                # Get the forecast for the next game
                forecast_value = predicted_mean[0]
                lower_bound = confidence_intervals.iloc[0, 0]
                upper_bound = confidence_intervals.iloc[0, 1]
                
                # Store forecast results
                forecast_dict[metric] = {
                    'forecast': forecast_value,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'method': 'sarima',
                    'confidence': 'high'
                }
                
                logger.info(f"SARIMA forecast for {metric}: {forecast_value:.3f} [{lower_bound:.3f}, {upper_bound:.3f}]")
            except Exception as e:
                logger.error(f"Error generating forecast for {metric}: {str(e)}")
                # Use player average as fallback
                if metric in player_averages:
                    avg_value = player_averages[metric]
                    forecast_dict[metric] = {
                        'forecast': avg_value,
                        'lower_bound': avg_value * 0.85,
                        'upper_bound': avg_value * 1.15,
                        'method': 'average',
                        'confidence': 'low'
                    }
        else:
            # Use player average as fallback if model fitting failed
            if metric in player_averages:
                avg_value = player_averages[metric]
                forecast_dict[metric] = {
                    'forecast': avg_value,
                    'lower_bound': avg_value * 0.85,
                    'upper_bound': avg_value * 1.15,
                    'method': 'average',
                    'confidence': 'low'
                }