import os
import pytest
from unittest.mock import patch
import json

# Add the parent directory to the path so we can import the app
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

@pytest.fixture
def client():
    """Create a test client for the app"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index_page(client):
    """Test that the index page loads"""
    response = client.get('/')
    assert response.status_code == 200
    assert b'Welcome to' in response.data

def test_today_games_endpoint(client):
    """Test that the today_games API endpoint works with mocked data"""
    with patch('app.game_scraper.scrape_and_save_games', return_value=1):
        with patch('app.db.get_games_by_date', return_value=[
            {
                'game_time': '19:30',
                'away_team': 'Boston Celtics',
                'home_team': 'Los Angeles Lakers',
                'venue': 'Crypto.com Arena'
            }
        ]):
            response = client.get('/api/today_games')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'games' in data
            assert len(data['games']) == 1
            assert data['games'][0]['away_team'] == 'Boston Celtics'

def test_player_stats_endpoint(client):
    """Test that the player_stats API endpoint works with mocked data"""
    mock_stats = [{'GAME_DATE': '2025-04-01', 'PTS_PER_MIN': 0.8, 'FG_PCT': 0.55}]
    
    with patch('app.player_stats.get_player_stats', return_value=mock_stats):
        response = client.get('/api/player_stats/LeBron James')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'stats' in data
        assert len(data['stats']) == 1
        assert data['stats'][0]['PTS_PER_MIN'] == 0.8

def test_player_prediction_endpoint(client):
    """Test that the player_prediction API endpoint works with mocked data"""
    mock_stats = [{'GAME_DATE': '2025-04-01', 'PTS_PER_MIN': 0.8, 'FG_PCT': 0.55}]
    mock_forecast = {
        'PTS_PER_MIN': {
            'forecast': 0.85,
            'lower_bound': 0.75,
            'upper_bound': 0.95,
            'method': 'sarima',
            'confidence': 'high'
        }
    }
    
    with patch('app.player_stats.get_player_stats', return_value=mock_stats):
        with patch('app.series_analyzer.forecast_player_performance', return_value=mock_forecast):
            response = client.get('/api/player_prediction/LeBron James')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'prediction' in data
            assert 'PTS_PER_MIN' in data['prediction']
            assert data['prediction']['PTS_PER_MIN']['forecast'] == 0.85

def test_player_template_renders(client):
    """Test that the player.html template renders correctly"""
    response = client.get('/player/LeBron James')
    assert response.status_code == 200
    assert b'LeBron James' in response.data