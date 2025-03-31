import logging
import numpy as np
import pandas as pd
import time
import random
import re
from typing import Dict, List, Any, Optional, Tuple
from nba_api.stats.endpoints import playergamelog, playercareerstats
from nba_api.stats.static import players
from fuzzywuzzy import process, fuzz

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('analysis.player_stats')


class PlayerStatsCollector:
    """
    Collects and processes historical player statistics using the NBA API.
    Focused on retrieving time-series compatible metrics for ARIMA modeling.
    
    Implements progressive delays and backoff strategy to handle rate limiting.
    """
    
    def __init__(self):
        """Initialize the PlayerStatsCollector."""
        # Request counter to track API calls
        self.request_count = 0
        # Initial delay between requests (seconds)
        self.base_delay = 1.0
        # Maximum delay to use (seconds)
        self.max_delay = 5.0
        # Factor by which delay increases after each request
        self.delay_factor = 1.05
        # Current delay time
        self.current_delay = self.base_delay
        # Last request timestamp
        self.last_request_time = 0
        # Cache for player name to ID mapping
        self.player_id_cache = {}
    
    def _wait_between_requests(self):
        """
        Implements a progressive delay strategy to avoid rate limiting.
        - Initial small delay (self.base_delay)
        - Gradually increasing delay based on request count
        - Random jitter to prevent synchronized requests
        """
        # Calculate time since last request
        now = time.time()
        time_since_last = now - self.last_request_time
        
        # If we've already waited longer than needed, don't wait more
        if self.last_request_time > 0 and time_since_last > self.current_delay:
            pass
        else:
            # Add a small random jitter (±10%) to avoid request synchronization
            jitter = self.current_delay * random.uniform(-0.1, 0.1)
            delay = max(0, self.current_delay + jitter - time_since_last)
            
            if delay > 0:
                logger.debug(f"Waiting {delay:.2f}s before next request (total requests: {self.request_count})")
                time.sleep(delay)
        
        # Increment request counter
        self.request_count += 1
        
        # Update delay for next request (capped at max_delay)
        self.current_delay = min(
            self.max_delay, 
            self.base_delay * (self.delay_factor ** (self.request_count // 10))
        )
        
        # Update last request timestamp
        self.last_request_time = time.time()
    
    def _reset_delay(self):
        """Reset delay to initial value after a successful batch of requests."""
        if self.request_count > 20:  # Only reset if we've made a significant number of requests
            logger.info(f"Resetting delay after {self.request_count} requests")
            self.request_count = 0
            self.current_delay = self.base_delay
    
    def _is_abbreviated_name(self, name: str) -> bool:
        """
        Check if a name appears to be abbreviated (e.g., "B. Carrington").
        
        Args:
            name: Player name to check
            
        Returns:
            True if name appears to be abbreviated, False otherwise
        """
        # Common patterns for abbreviated names
        patterns = [
            r'^[A-Z]\.\s+[A-Za-z]+$',  # B. Carrington
            r'^[A-Z]\s+[A-Za-z]+$',    # B Carrington
            r'^[A-Z]\.[A-Z]\.\s+[A-Za-z]+$',  # D.J. Augustin
        ]
        
        return any(re.match(pattern, name) for pattern in patterns)
    
    def _extract_name_parts(self, name: str) -> Tuple[str, str]:
        """
        Extract first and last name parts, handling abbreviated first names.
        
        Args:
            name: Player name (e.g., "B. Carrington" or "LeBron James")
            
        Returns:
            Tuple of (first_name_or_initial, last_name)
        """
        parts = name.split()
        if len(parts) < 2:
            return "", name  # If just one name, assume it's the last name
        
        # Handle potential middle names/initials by combining all but last part
        last_name = parts[-1]
        first_parts = ' '.join(parts[:-1])
        
        return first_parts, last_name
    
    def _get_all_players(self):
        """
        Get a list of all NBA players for fuzzy matching.
        
        Returns:
            List of all NBA players
        """
        self._wait_between_requests()
        return players.get_players()
    
    def get_player_id(self, player_name: str) -> Optional[str]:
        """
        Get NBA API player ID from player name, using fuzzy matching and
        pattern recognition to handle abbreviated names.
        
        Args:
            player_name: Player's full or abbreviated name
            
        Returns:
            Player ID as string or None if not found
        """
        # First check if we've already looked up this player
        if player_name in self.player_id_cache:
            return self.player_id_cache[player_name]
        
        try:
            # Method 1: Try direct lookup first (fastest)
            self._wait_between_requests()
            direct_match = players.find_players_by_full_name(player_name)
            if direct_match and len(direct_match) > 0:
                self.player_id_cache[player_name] = direct_match[0]['id']
                logger.info(f"Direct match found for: {player_name}")
                return direct_match[0]['id']
            
            # Extract first and last names
            first_part, last_name = self._extract_name_parts(player_name)
            
            # Method 2: If we have an abbreviated name, try finding by last name + initial
            if self._is_abbreviated_name(player_name):
                # Extract initial from the abbreviated name
                initial = first_part[0].upper() if first_part else ""
                
                if initial and last_name:
                    logger.info(f"Trying to match abbreviated name: {player_name} (Initial: {initial}, Last: {last_name})")
                    
                    # Get all players with the same last name
                    self._wait_between_requests()
                    last_name_matches = players.find_players_by_last_name(last_name)
                    
                    if last_name_matches:
                        # Filter to those whose first name starts with the initial
                        initial_matches = [
                            player for player in last_name_matches 
                            if player['first_name'] and player['first_name'][0].upper() == initial
                        ]
                        
                        if initial_matches:
                            if len(initial_matches) == 1:
                                # If exactly one match, use it
                                player_id = initial_matches[0]['id']
                                self.player_id_cache[player_name] = player_id
                                logger.info(f"Matched abbreviated name {player_name} to {initial_matches[0]['full_name']}")
                                return player_id
                            else:
                                # If multiple matches, get the most recent/active player
                                logger.info(f"Multiple initial matches for {player_name}, using most recently active")
                                # Using the first result as default NBA API behavior (usually returns most relevant first)
                                player_id = initial_matches[0]['id']
                                self.player_id_cache[player_name] = player_id
                                return player_id
            
            # Method 3: Fuzzy matching as fallback
            # Get all players for fuzzy matching (only when needed)
            all_nba_players = self._get_all_players()
            player_names = [p['full_name'] for p in all_nba_players]
            
            # Get a list of best matches using fuzzywuzzy
            matches = process.extract(
                player_name,
                player_names,
                scorer=fuzz.token_sort_ratio,  # This works well for names in different orders
                limit=5
            )
            
            # Check match quality
            if matches and matches[0][1] > 75:  # 75% confidence threshold
                best_match = matches[0][0]
                logger.info(f"Fuzzy matched {player_name} to {best_match} with score {matches[0][1]}")
                
                # Get player ID for best match
                matched_player = next(p for p in all_nba_players if p['full_name'] == best_match)
                player_id = matched_player['id']
                
                # Cache this match for future lookups
                self.player_id_cache[player_name] = player_id
                return player_id
            
            # Last resort - try just the last name
            self._wait_between_requests()
            last_name_only = players.find_players_by_last_name(last_name)
            if last_name_only and len(last_name_only) > 0:
                logger.warning(f"Falling back to last name only for {player_name}, found {last_name_only[0]['full_name']}")
                self.player_id_cache[player_name] = last_name_only[0]['id']
                return last_name_only[0]['id']
            
            logger.warning(f"Could not find player ID for: {player_name}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting player ID for {player_name}: {str(e)}")
            # Implement exponential backoff for failures
            self.current_delay = min(self.max_delay, self.current_delay * 2)
            return None
    
    def get_recent_game_logs(self, player_name: str, num_games: int = 30) -> Optional[pd.DataFrame]:
        """
        Retrieve recent game logs for a player.
        
        Args:
            player_name: Player's full name
            num_games: Number of recent games to retrieve (default 30)
            
        Returns:
            DataFrame with game-by-game stats or None if retrieval fails
        """
        try:
            # Get player ID
            player_id = self.get_player_id(player_name)
            if not player_id:
                return None
            
            # Get current season in format YYYY-YY
            from datetime import datetime
            current_year = datetime.now().year
            current_month = datetime.now().month
            
            # If between January and September, use previous year
            if current_month < 10:
                season = f"{current_year-1}-{str(current_year)[-2:]}"
            else:
                season = f"{current_year}-{str(current_year+1)[-2:]}"
            
            # Wait before making request
            self._wait_between_requests()
            
            # Get game logs for most recent season
            game_logs = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season
            )
            df = game_logs.get_data_frames()[0]
            
            # If we don't have enough games from current season, get previous season too
            if len(df) < num_games:
                prev_season = f"{int(season.split('-')[0])-1}-{int(season.split('-')[0])%100:02d}"
                
                # Wait before making another request
                self._wait_between_requests()
                
                prev_logs = playergamelog.PlayerGameLog(
                    player_id=player_id,
                    season=prev_season
                )
                prev_df = prev_logs.get_data_frames()[0]
                
                # Combine current and previous season
                df = pd.concat([df, prev_df])
            
            # Take only requested number of games
            df = df.head(num_games)
            
            if len(df) == 0:
                logger.warning(f"No game logs found for player: {player_name}")
                return None
                
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving game logs for {player_name}: {str(e)}")
            # Implement exponential backoff for failures
            self.current_delay = min(self.max_delay, self.current_delay * 2)
            return None
    
    def calculate_advanced_metrics(self, game_logs: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate advanced metrics from raw game logs.
        
        Args:
            game_logs: DataFrame with raw game log stats
            
        Returns:
            DataFrame with added advanced metrics
        """
        # Make a copy to avoid modifying the original
        df = game_logs.copy()
        
        try:
            # Convert necessary columns to numeric format
            numeric_cols = ['MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 
                           'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS',
                           'PLUS_MINUS']
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                else:
                    logger.warning(f"Column {col} not found in game logs")
            
            # Fill NaN values with 0
            df = df.fillna(0)
            
            # Calculate minutes as float (convert '12:34' format to minutes)
            df['MINUTES_PLAYED'] = df['MIN'].apply(
                lambda x: float(x.split(':')[0]) + float(x.split(':')[1])/60 if isinstance(x, str) and ':' in x else float(x)
            )
            
            # 1. Field Goal Percentage (FG%)
            df['FG_PCT'] = np.where(df['FGA'] > 0, df['FGM'] / df['FGA'], 0)
            
            # 2. True Shooting Percentage (TS%)
            # TS% = PTS / (2 * (FGA + 0.44 * FTA))
            df['TS_PCT'] = np.where(
                (df['FGA'] + 0.44 * df['FTA']) > 0,
                df['PTS'] / (2 * (df['FGA'] + 0.44 * df['FTA'])),
                0
            )
            
            # 3. Points per Minute
            df['PTS_PER_MIN'] = np.where(df['MINUTES_PLAYED'] > 0, df['PTS'] / df['MINUTES_PLAYED'], 0)
            
            # 4. Plus/Minus already in the data as PLUS_MINUS
            
            # 5/6. Offensive & Defensive Ratings (simplified approximations)
            # These are complex ratings that usually require team data
            # Using simplified calculations based on individual performance
            
            # Possessions approximation = FGA - OREB + TOV + 0.44*FTA
            df['POSS'] = df['FGA'] - df['OREB'] + df['TOV'] + 0.44 * df['FTA']
            
            # Offensive Rating = Points produced per 100 possessions
            df['OFF_RATING'] = np.where(df['POSS'] > 0, 100 * (df['PTS'] + 1.5 * df['AST']) / df['POSS'], 0)
            
            # Defensive Rating = Points allowed per 100 possessions (rough approximation)
            # This is a simplified version (truly accurate defensive rating requires team data)
            df['DEF_RATING'] = np.where(
                df['MINUTES_PLAYED'] > 0,
                100 - (5 * (df['STL'] + df['BLK']) - df['PF'] - df['PLUS_MINUS']) / df['MINUTES_PLAYED'],
                0
            )
            
            # 7. Game Score (John Hollinger's formula)
            # GmSc = PTS + 0.4*FGM - 0.7*FGA - 0.4*(FTA-FTM) + 0.7*OREB + 0.3*DREB + STL + 0.7*AST + 0.7*BLK - 0.4*PF - TOV
            df['GAME_SCORE'] = (
                df['PTS'] + 0.4 * df['FGM'] - 0.7 * df['FGA'] - 0.4 * (df['FTA'] - df['FTM']) 
                + 0.7 * df['OREB'] + 0.3 * df['DREB'] + df['STL'] + 0.7 * df['AST'] 
                + 0.7 * df['BLK'] - 0.4 * df['PF'] - df['TOV']
            )
            
            # 8. Usage Rate (simplified version)
            # USG% = 100 * ((FGA + 0.44 * FTA + TOV) * (Team MP / 5)) / (MP * (Team FGA + 0.44 * Team FTA + Team TOV))
            # Since team data isn't in game logs, using approximation:
            df['USG_RATE'] = 100 * (df['FGA'] + 0.44 * df['FTA'] + df['TOV']) / df['POSS']
            
            # 9. Minutes Played already calculated as MINUTES_PLAYED
            
            # 10. Assist-to-Turnover Ratio
            df['AST_TO_RATIO'] = np.where(df['TOV'] > 0, df['AST'] / df['TOV'], df['AST'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating advanced metrics: {str(e)}")
            return game_logs  # Return original if calculation fails
    
    def get_player_stats(self, player_name: str, num_games: int = 30) -> Optional[pd.DataFrame]:
        """
        Retrieve and process player stats for time series analysis.
        
        Args:
            player_name: Player's full name
            num_games: Number of recent games to retrieve
            
        Returns:
            DataFrame with time series compatible stats or None if retrieval fails
        """
        # Log the start of player stats retrieval
        logger.info(f"Retrieving stats for player: {player_name}")
        
        # Get basic game logs
        game_logs = self.get_recent_game_logs(player_name, num_games)
        if game_logs is None:
            logger.warning(f"No game logs found for player: {player_name}")
            return None
        
        # Calculate advanced metrics
        player_stats = self.calculate_advanced_metrics(game_logs)
        
        # Select only the metrics we need for time series analysis
        if player_stats is not None:
            # Define columns to keep for time series analysis
            ts_columns = [
                'GAME_DATE', 'GAME_ID',  # Identifiers
                'FG_PCT', 'TS_PCT', 'PTS_PER_MIN',  # Efficiency metrics
                'PLUS_MINUS', 'OFF_RATING', 'DEF_RATING', 'GAME_SCORE',  # Impact metrics
                'USG_RATE', 'MINUTES_PLAYED', 'AST_TO_RATIO'  # Usage metrics
            ]
            
            # Keep only columns that exist in the dataframe
            available_columns = [col for col in ts_columns if col in player_stats.columns]
            
            # Filter and sort by date (most recent first)
            filtered_stats = player_stats[available_columns].sort_values('GAME_DATE', ascending=False)
            
            logger.info(f"Successfully retrieved stats for player: {player_name}")
            return filtered_stats
        
        return None
    
    def get_team_players_stats(self, team_name: str, lineup: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
        """
        Retrieve stats for all players in a team's lineup.
        
        Args:
            team_name: Name of the team
            lineup: List of player dictionaries
            
        Returns:
            Dictionary mapping player names to their stats DataFrames
        """
        team_stats = {}
        
        for i, player in enumerate(lineup):
            player_name = player.get('name')
            if not player_name:
                continue
                
            # Skip injured players (optional)
            status = player.get('status')
            if status and status != 'active':
                logger.info(f"Skipping injured player: {player_name} ({status})")
                continue
                
            # Get player stats
            player_stats = self.get_player_stats(player_name)
            
            if player_stats is not None and not player_stats.empty:
                team_stats[player_name] = player_stats
            
            # Reset delay periodically to adapt to changing conditions
            if (i + 1) % 5 == 0:
                self._reset_delay()
        
        logger.info(f"Retrieved stats for {len(team_stats)} players from {team_name}")
        return team_stats


# Example usage when run as script
if __name__ == "__main__":
    # Create the player stats collector
    collector = PlayerStatsCollector()
    
    # Test with a single player
    player_name = "LeBron James"
    stats = collector.get_player_stats(player_name, num_games=10)
    
    if stats is not None:
        print(f"\nStats for {player_name}:")
        print(stats[['GAME_DATE', 'FG_PCT', 'PTS_PER_MIN', 'PLUS_MINUS', 'GAME_SCORE']].head())
        
    # Test with an abbreviated name
    abbreviated_name = "L. James"
    stats = collector.get_player_stats(abbreviated_name, num_games=5)
    
    if stats is not None:
        print(f"\nSuccessfully matched abbreviated name and got stats for {abbreviated_name}:")
        print(stats[['GAME_DATE', 'FG_PCT', 'PTS_PER_MIN', 'PLUS_MINUS', 'GAME_SCORE']].head())