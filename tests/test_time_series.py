import os
import sys
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis.time_series import PlayerTimeSeriesAnalyzer, CATEGORIES, KEY_METRICS


class TestTimeSeriesModule(unittest.TestCase):
    """Test cases for the time series analysis module."""
    
    def setUp(self):
        """Set up test data and analyzer instance."""
        self.analyzer = PlayerTimeSeriesAnalyzer(min_games=5, max_games=30)
        
        # Create sample player stats dataframe
        dates = [
            (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
            for i in range(20)
        ]
        
        # Create sample data for all the metrics
        sample_data = {
            'GAME_DATE': dates,
            'THREES': np.random.randint(0, 8, 20),
            'TWOS': np.random.randint(2, 12, 20),
            'FTM': np.random.randint(0, 10, 20),
            'REB': np.random.randint(3, 15, 20),
            'AST': np.random.randint(1, 12, 20),
            'BLK': np.random.randint(0, 5, 20),
            'STL': np.random.randint(0, 5, 20),
            'TOV': np.random.randint(0, 7, 20),
            'MIN': np.random.randint(15, 40, 20),
            'FGM': np.random.randint(5, 15, 20),
            'FGA': np.random.randint(10, 25, 20),
            'FG_PCT': np.random.uniform(0.3, 0.6, 20),
            'TS_PCT': np.random.uniform(0.4, 0.7, 20),
            'THREE_PCT': np.random.uniform(0.3, 0.5, 20),
            'GAME_SCORE': np.random.uniform(5, 30, 20),
            'PLUS_MINUS': np.random.uniform(-15, 15, 20),
            'MINUTES_PLAYED': np.random.uniform(15, 40, 20),
            'PTS_PER_MIN': np.random.uniform(0.3, 0.8, 20)
        }
        
        self.sample_stats = pd.DataFrame(sample_data)
    
    def test_key_metric_count(self):
        """Test that we have exactly 15 key metrics defined."""
        self.assertEqual(len(KEY_METRICS), 15, "Should have exactly 15 key metrics")
    
    def test_categories_include_all_metrics(self):
        """Test that all 15 metrics are included in the categories."""
        # Extract all metrics from all categories
        all_category_metrics = []
        for metrics in CATEGORIES.values():
            all_category_metrics.extend(metrics)
        
        # Remove duplicates
        unique_category_metrics = set(all_category_metrics)
        
        # Check that count matches
        self.assertEqual(len(unique_category_metrics), len(KEY_METRICS), 
                        "Number of unique metrics in categories should match KEY_METRICS")
        
        # Check that every metric in KEY_METRICS is also in a category
        for metric in KEY_METRICS:
            self.assertIn(metric, unique_category_metrics, 
                        f"Metric {metric} from KEY_METRICS not found in any category")
        
    def test_preprocess_time_series(self):
        """Test the time series preprocessing function."""
        processed = self.analyzer.preprocess_time_series(self.sample_stats)
        
        # Check that we have processed series for each key metric
        for metric in KEY_METRICS:
            if metric in self.sample_stats.columns:
                self.assertIn(metric, processed, f"Missing processed time series for {metric}")
                self.assertEqual(len(processed[metric]), len(self.sample_stats),
                                f"Processed time series length mismatch for {metric}")
    
    def test_compute_player_averages(self):
        """Test computing player averages."""
        player_name = "Test Player"
        averages = self.analyzer.compute_player_averages(self.sample_stats, player_name)
        
        # Check that we have averages for each key metric
        for metric in KEY_METRICS:
            if metric in self.sample_stats.columns:
                self.assertIn(metric, averages, f"Missing average for {metric}")
                self.assertIsInstance(averages[metric], float, f"Average for {metric} should be a float")
        
        # Check that the averages were stored in the class
        self.assertIn(player_name, self.analyzer.player_averages, 
                    "Player averages should be stored in the class")
    
    def test_forecast_structure(self):
        """Test that the forecast function returns the expected structure."""
        player_name = "Test Player"
        forecast = self.analyzer.forecast_player_performance(player_name, self.sample_stats)
        
        # Check that the forecast has the expected structure
        self.assertIn("flat", forecast, "Forecast should have a 'flat' key")
        self.assertIn("by_category", forecast, "Forecast should have a 'by_category' key")
        
        # Check that by_category contains all categories
        for category in CATEGORIES:
            self.assertIn(category, forecast["by_category"], 
                        f"Category {category} missing from forecast")
            
            # Check that each category contains the right metrics
            for metric in CATEGORIES[category]:
                if metric in forecast["flat"]:
                    self.assertIn(metric, forecast["by_category"][category], 
                                f"Metric {metric} should be in category {category}")
    
    def test_forecast_content(self):
        """Test that forecast results contain the expected fields."""
        player_name = "Test Player"
        forecast = self.analyzer.forecast_player_performance(player_name, self.sample_stats)
        
        # Check some metrics from the forecast
        for metric in ['THREES', 'REB', 'FG_PCT']:
            if metric in forecast["flat"]:
                metric_forecast = forecast["flat"][metric]
                
                # Check required fields in the forecast
                self.assertIn("forecast", metric_forecast, f"Missing 'forecast' value for {metric}")
                self.assertIn("lower_bound", metric_forecast, f"Missing 'lower_bound' value for {metric}")
                self.assertIn("upper_bound", metric_forecast, f"Missing 'upper_bound' value for {metric}")
                self.assertIn("method", metric_forecast, f"Missing 'method' value for {metric}")
                self.assertIn("confidence", metric_forecast, f"Missing 'confidence' value for {metric}")
                
                # Check that the confidence interval makes sense
                self.assertLess(metric_forecast["lower_bound"], metric_forecast["forecast"], 
                            f"Lower bound should be less than forecast for {metric}")
                self.assertGreater(metric_forecast["upper_bound"], metric_forecast["forecast"], 
                                f"Upper bound should be greater than forecast for {metric}")


if __name__ == "__main__":
    unittest.main()