import logging
import numpy as np
import pandas as pd
import time
import random
import re
from typing import Dict, List, Any, Optional, Tuple
from nba_api.stats.endpoints import playergamelog, playercareerstats, playerdashboardbygeneralsplits
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
    
    Uses official NBA API advanced metrics where possible, with supplementary
    calculations only when needed.
    
    Implements progressive delays and backoff strategy to handle rate limiting.
    """
    
    # Common name suffixes to handle
    NAME_SUFFIXES = ['Jr.', 'Sr.', 'II', 'III', 'IV', 'V']
    
    # Common problematic name mappings
    NAME_CORRECTIONS = {
        'Jaren Jackson': 'Jaren Jackson Jr.',
        'Wendell Carter': 'Wendell Carter Jr.',
        'Gary Trent': 'Gary Trent Jr.',
        'Larry Nance': 'Larry Nance Jr.',
        'Tim Hardaway': 'Tim Hardaway Jr.',
        'Gary Payton': 'Gary Payton II',
        'Otto Porter': 'Otto Porter Jr.',
        'Marvin Bagley': 'Marvin Bagley III',
        'Kelly Oubre': 'Kelly Oubre Jr.',
        'Kevin Porter': 'Kevin Porter Jr.',
        'Robert Williams': 'Robert Williams III',
        'Lonnie Walker': 'Lonnie Walker IV'
    }
    
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
        # All NBA players (cached after first retrieval)
        self._all_nba_players = None
        # Season tracking for consistent retrieval
        self._current_season = self._get_current_season()
        # Default minimum games for reliable analysis
        self.min_games_for_analysis = 5
    
    def _get_current_season(self):
        """
        Get current NBA season in format YYYY-YY.
        """
        from datetime import datetime
        current_year = datetime.now().year
        current_month = datetime.now().month
        
        # If between January and September, use previous year
        if current_month < 10:
            season = f"{current_year-1}-{str(current_year)[-2:]}"
        else:
            season = f"{current_year}-{str(current_year+1)[-2:]}"
        
        return season
    
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
    
    def _normalize_name(self, name: str) -> str:
        """
        Normalize a player name for comparison.
        - Remove accents
        - Convert to lowercase
        - Remove punctuation

        Args:
            name: Player name to normalize

        Returns:
            Normalized name
        """
        # Remove periods and convert to lowercase
        normalized = name.replace('.', '').lower()
        
        # Replace special characters (like ć, ģ, ņ, etc.)
        # This is a simplified approach - for a complete solution, 
        # consider using unicodedata.normalize
        char_map = {
            'ć': 'c', 'č': 'c', 'ç': 'c',
            'ā': 'a', 'á': 'a', 'à': 'a', 'ä': 'a',
            'ē': 'e', 'é': 'e', 'è': 'e', 'ë': 'e',
            'ī': 'i', 'í': 'i', 'ì': 'i', 'ï': 'i',
            'ō': 'o', 'ó': 'o', 'ò': 'o', 'ö': 'o',
            'ū': 'u', 'ú': 'u', 'ù': 'u', 'ü': 'u',
            'ń': 'n', 'ñ': 'n', 'ņ': 'n',
            'ś': 's', 'š': 's',
            'ź': 'z', 'ž': 'z',
            'ģ': 'g', 'ķ': 'k', 'ļ': 'l',
            'ř': 'r', 'ț': 't', 'ý': 'y',
        }
        
        for char, replacement in char_map.items():
            normalized = normalized.replace(char, replacement)
        
        return normalized
    
    def _remove_suffix(self, name: str) -> str:
        """
        Remove common suffixes from a name for base comparison.
        
        Args:
            name: Player name possibly including a suffix
            
        Returns:
            Name with suffix removed
        """
        # Check predefined corrections first
        base_name = name
        
        # Iterate through known suffixes and remove them
        name_parts = name.split()
        if len(name_parts) >= 2:
            for suffix in self.NAME_SUFFIXES:
                if name.endswith(f" {suffix}"):
                    base_name = name[:-len(suffix)-1].strip()
                    break
        
        return base_name
    
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
            r'^[A-Z]\.[A-Z]\s+[A-Za-z]+$',    # D.J Augustin
            r'^[A-Z][A-Z]\s+[A-Za-z]+$',      # DJ Augustin
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
        
        # Handle hyphenated last names (e.g., "Gilgeous-Alexander")
        if len(parts) >= 3 and '-' in parts[-2]:
            last_name = f"{parts[-2]} {parts[-1]}"
            first_parts = ' '.join(parts[:-2])
            return first_parts, last_name
        
        # Standard case: last part is the last name
        last_name = parts[-1]
        first_parts = ' '.join(parts[:-1])
        
        return first_parts, last_name
    
    def _get_all_players(self):
        """
        Get a list of all NBA players for matching.
        Caches the result for future calls.
        
        Returns:
            List of all NBA players
        """
        if self._all_nba_players is None:
            self._wait_between_requests()
            self._all_nba_players = players.get_players()
        
        return self._all_nba_players
    
    def _check_for_suffixed_version(self, base_name: str, all_players: List[Dict]) -> Optional[Dict]:
        """
        Check if a base name (without suffix) has a suffixed version in the player database.
        
        Args:
            base_name: Player name without suffix
            all_players: List of all NBA players
            
        Returns:
            Player dict if a suffixed version is found, None otherwise
        """
        # Check for common corrections first
        if base_name in self.NAME_CORRECTIONS:
            corrected_name = self.NAME_CORRECTIONS[base_name]
            for player in all_players:
                if player['full_name'] == corrected_name:
                    logger.info(f"Found known correction: {base_name} -> {corrected_name}")
                    return player
        
        # Try to find a player whose name starts with the base_name and includes one of the suffixes
        normalized_base = self._normalize_name(base_name)
        
        for player in all_players:
            full_name = player['full_name']
            if full_name.startswith(base_name) and len(full_name) > len(base_name):
                # Check if the extra part matches a known suffix
                suffix_part = full_name[len(base_name):].strip()
                if suffix_part in [' ' + suffix for suffix in self.NAME_SUFFIXES]:
                    logger.info(f"Found suffixed version: {base_name} -> {full_name}")
                    return player
        
        return None
    
    def _rank_name_matches(self, candidates: List[Dict], first_initial: Optional[str] = None) -> List[Dict]:
        """
        Rank candidate player matches based on relevance.
        
        Args:
            candidates: List of player dictionaries to rank
            first_initial: Optional first initial to prioritize
            
        Returns:
            Ranked list of candidates
        """
        # If no candidates, return empty list
        if not candidates:
            return []
        
        # If only one candidate, return as is
        if len(candidates) == 1:
            return candidates
        
        # Define scoring function
        def get_score(player):
            score = 0
            
            # Prioritize active/recent players
            # NBA API usually returns more recent players first
            # Add a small score based on position in the original list
            score += len(candidates) - candidates.index(player)
            
            # Prioritize players with the right initial if provided
            if first_initial and player.get('first_name', ''):
                if player['first_name'][0].upper() == first_initial.upper():
                    score += 100  # High priority for initial match
            
            # Higher id numbers are generally more recent players
            player_id = int(player.get('id', 0))
            score += player_id / 1000000  # Small bonus for newer players
            
            return score
        
        # Sort candidates by score (descending)
        return sorted(candidates, key=get_score, reverse=True)
    
    def get_player_id(self, player_name: str) -> Optional[str]:
        """
        Get NBA API player ID from player name using a multi-strategy approach:
        1. Direct lookup (fastest)
        2. Check for known suffix variations
        3. Last name + initial matching for abbreviated names
        4. Last name search with smart ranking
        5. Fuzzy matching as final fallback
        
        Args:
            player_name: Player's full or abbreviated name
            
        Returns:
            Player ID as string or None if not found
        """
        # First check if we've already looked up this player
        if player_name in self.player_id_cache:
            return self.player_id_cache[player_name]
        
        # Check for hard-coded name corrections
        if player_name in self.NAME_CORRECTIONS:
            corrected_name = self.NAME_CORRECTIONS[player_name]
            logger.info(f"Using predefined name correction: {player_name} -> {corrected_name}")
            player_name = corrected_name
        
        try:
            # Strategy 1: Try direct lookup first (fastest)
            self._wait_between_requests()
            direct_match = players.find_players_by_full_name(player_name)
            if direct_match and len(direct_match) > 0:
                player_id = direct_match[0]['id']
                self.player_id_cache[player_name] = player_id
                logger.info(f"Direct match found for: {player_name}")
                return player_id
            
            # Get all players for advanced matching (only load once and cache)
            all_players = self._get_all_players()
            
            # Extract name parts for further processing
            first_part, last_name = self._extract_name_parts(player_name)
            
            # Strategy 2: Check for suffix variations
            # e.g., "Jaren Jackson" -> "Jaren Jackson Jr."
            if not self._is_abbreviated_name(player_name):
                suffixed_player = self._check_for_suffixed_version(player_name, all_players)
                if suffixed_player:
                    player_id = suffixed_player['id']
                    full_name = suffixed_player['full_name']
                    self.player_id_cache[player_name] = player_id
                    logger.info(f"Suffix variation matched: {player_name} -> {full_name}")
                    return player_id
            
            # Strategy 3: If we have an abbreviated name, try finding by last name + initial
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
                            # If multiple matches, rank them
                            if len(initial_matches) > 1:
                                ranked_matches = self._rank_name_matches(initial_matches, initial)
                                best_match = ranked_matches[0]
                                logger.info(f"Multiple initial matches for {player_name}, using {best_match['full_name']}")
                            else:
                                best_match = initial_matches[0]
                                logger.info(f"Matched abbreviated name {player_name} to {best_match['full_name']}")
                            
                            player_id = best_match['id']
                            self.player_id_cache[player_name] = player_id
                            return player_id
            
            # Strategy 4: Use last name search with smart ranking
            self._wait_between_requests()
            last_name_matches = players.find_players_by_last_name(last_name)
            
            if last_name_matches:
                # For abbreviated names, we may not have found a direct initial match above
                # Try a more lenient ranking approach
                if self._is_abbreviated_name(player_name) and first_part:
                    initial = first_part[0].upper()
                    ranked_matches = self._rank_name_matches(last_name_matches, initial)
                else:
                    ranked_matches = self._rank_name_matches(last_name_matches)
                
                if ranked_matches:
                    best_match = ranked_matches[0]
                    player_id = best_match['id']
                    logger.info(f"Last name match for {player_name}: found {best_match['full_name']}")
                    self.player_id_cache[player_name] = player_id
                    return player_id
            
            # Strategy 5: Fuzzy matching as final fallback
            # Get all player names for fuzzy matching
            player_names = [p['full_name'] for p in all_players]
            
            # Get a list of best matches using fuzzywuzzy
            matches = process.extract(
                player_name,
                player_names,
                scorer=fuzz.token_sort_ratio,  # Works well for names in different orders
                limit=5
            )
            
            # Check match quality
            if matches and matches[0][1] > 75:  # 75% confidence threshold
                best_match = matches[0][0]
                logger.info(f"Fuzzy matched {player_name} to {best_match} with score {matches[0][1]}")
                
                # Get player ID for best match
                matched_player = next(p for p in all_players if p['full_name'] == best_match)
                player_id = matched_player['id']
                
                # Cache this match for future lookups
                self.player_id_cache[player_name] = player_id
                return player_id
            
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
        Handles rookies and players with limited games by retrieving all available games.
        
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
            
            # Wait before making request
            self._wait_between_requests()
            
            # Get game logs for most recent season
            game_logs = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=self._current_season
            )
            df = game_logs.get_data_frames()[0]
            
            # If we don't have enough games from current season, get previous season too
            if len(df) < num_games:
                prev_season = f"{int(self._current_season.split('-')[0])-1}-{int(self._current_season.split('-')[0])%100:02d}"
                
                # Wait before making another request
                self._wait_between_requests()
                
                prev_logs = playergamelog.PlayerGameLog(
                    player_id=player_id,
                    season=prev_season
                )
                prev_df = prev_logs.get_data_frames()[0]
                
                # Combine current and previous season
                df = pd.concat([df, prev_df])
            
            # For rookies and players with limited games, use what we have
            if len(df) == 0:
                logger.warning(f"No game logs found for player: {player_name}")
                return None
            elif len(df) < num_games:
                # We have some games but fewer than requested
                logger.info(f"Found only {len(df)} games for {player_name} (requested {num_games})")
                
                # Check if we have minimum games needed for time series analysis
                if len(df) < self.min_games_for_analysis:
                    logger.warning(
                        f"Player {player_name} has only {len(df)} games, which is below the "
                        f"minimum threshold of {self.min_games_for_analysis} for reliable analysis."
                    )
                    # Still return what we have - the caller can decide whether to use it
            else:
                # Take only requested number of games if we have enough
                df = df.head(num_games)
                
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving game logs for {player_name}: {str(e)}")
            # Implement exponential backoff for failures
            self.current_delay = min(self.max_delay, self.current_delay * 2)
            return None
    
    def get_player_advanced_metrics(self, player_id: str) -> Optional[pd.DataFrame]:
        """
        Fetch official advanced metrics for a player from the NBA API.
        
        Args:
            player_id: Player's NBA API ID
            
        Returns:
            DataFrame with advanced metrics or None if retrieval fails
        """
        try:
            self._wait_between_requests()
            
            # Get player dashboard with advanced metrics
            dashboard = playerdashboardbygeneralsplits.PlayerDashboardByGeneralSplits(
                player_id=player_id,
                measure_type_detailed='Advanced',
                per_mode_detailed='PerGame',
                season=self._current_season
            )
            
            # Get the dataframe (first element is the overall stats)
            df = dashboard.get_data_frames()[0]
            
            if df.empty:
                logger.warning(f"No advanced metrics found for player ID: {player_id}")
                return None
            
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving advanced metrics for player ID {player_id}: {str(e)}")
            # Implement exponential backoff for failures
            self.current_delay = min(self.max_delay, self.current_delay * 2)
            return None
    
    def calculate_advanced_metrics(
        self,
        game_logs: pd.DataFrame,
        player_id: str | None = None
    ) -> pd.DataFrame:
        """
        Calculate advanced and fantasy-relevant metrics from raw game logs.

        Adds/minutes, raw counting stats (PTS, REB, AST, STL, BLK, FGM, FGA),
        plus existing efficiency / impact numbers.  Uses official NBA
        “Advanced” dashboard where available, falling back to local calcs.

        Args
        ----
        game_logs : DataFrame
            Raw game-log stats for a player.
        player_id : str | None
            NBA API player ID (optional—needed for dashboard pull).

        Returns
        -------
        DataFrame
            Game-by-game stats with new columns ready for time-series work.
        """
        # ------------------------------------------------------------------ #
        # Safety – copy incoming frame so caller’s version is untouched
        # ------------------------------------------------------------------ #
        df = game_logs.copy()

        try:
            # ------------------------------------------------------------------
            # 1) Ensure numeric columns
            # ------------------------------------------------------------------
            numeric_cols = [
                # core box
                "MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
                "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV",
                "PF", "PTS", "PLUS_MINUS",
                # volume we’ll add explicitly below
                "FGM", "FGA"
            ]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                else:
                    logging.getLogger(__name__).warning(
                        "Column %s not found in game logs", col
                    )

            df.fillna(0, inplace=True)

            # ------------------------------------------------------------------
            # 2) Minutes → float; we’ll keep both raw string & float
            # ------------------------------------------------------------------
            df["MINUTES_PLAYED"] = df["MIN"].apply(
                lambda x: (
                    float(x.split(":")[0]) + float(x.split(":")[1]) / 60
                )
                if isinstance(x, str) and ":" in x
                else float(x)
            )

            # ------------------------------------------------------------------
            # 3) Pull official advanced dashboard if we have a player_id
            # ------------------------------------------------------------------
            official = None
            if player_id:
                official = self.get_player_advanced_metrics(player_id)

            # ------------------------------------------------------------------
            # 4) Counting metrics – **fantasy emphasis**
            # ------------------------------------------------------------------
            # (Already numeric; keep them verbatim for later aggregation)

            # ------------------------------------------------------------------
            # 5) Derived volume/efficiency metrics
            # ------------------------------------------------------------------
            df["FG_PCT"] = np.where(df["FGA"] > 0, df["FGM"] / df["FGA"], 0)
            df["TS_PCT"] = np.where(
                (df["FGA"] + 0.44 * df["FTA"]) > 0,
                df["PTS"] / (2 * (df["FGA"] + 0.44 * df["FTA"])),
                0,
            )
            df["PTS_PER_MIN"] = np.where(
                df["MINUTES_PLAYED"] > 0,
                df["PTS"] / df["MINUTES_PLAYED"],
                0,
            )

            # possessions proxy (needed for ratings fallback)
            df["POSS"] = df["FGA"] - df["OREB"] + df["TOV"] + 0.44 * df["FTA"]

            # ------------------------------------------------------------------
            # 6) Usage & ratings (dashboard preferred)
            # ------------------------------------------------------------------
            if official is not None and not official.empty:
                base_usage = float(official["USG_PCT"].iloc[0]) * 100
                df["USG_RATE"] = base_usage  # simple: per-game variance optional

                off_rtg = float(official["OFF_RATING"].iloc[0])
                def_rtg = float(official["DEF_RATING"].iloc[0])

                df["OFF_RATING"] = off_rtg
                df["DEF_RATING"] = def_rtg
            else:
                # fallback
                df["USG_RATE"] = np.where(
                    df["MINUTES_PLAYED"] > 0,
                    df["FGA"] / df["MINUTES_PLAYED"] * 50,
                    20,
                ).clip(5, 45)

                df["OFF_RATING"] = np.where(
                    df["POSS"] > 0, 100 * df["PTS"] / df["POSS"], 100
                )
                df["DEF_RATING"] = 100  # neutral – could refine with on/off data

            # ------------------------------------------------------------------
            # 7) Hollinger Game Score – remains valuable context
            # ------------------------------------------------------------------
            df["GAME_SCORE"] = (
                df["PTS"]
                + 0.4 * df["FGM"]
                - 0.7 * df["FGA"]
                - 0.4 * (df["FTA"] - df["FTM"])
                + 0.7 * df["OREB"]
                + 0.3 * df["DREB"]
                + df["STL"]
                + 0.7 * df["AST"]
                + 0.7 * df["BLK"]
                - 0.4 * df["PF"]
                - df["TOV"]
            )

            # Assist-to-Turnover for completeness
            df["AST_TO_RATIO"] = np.where(df["TOV"] > 0, df["AST"] / df["TOV"], df["AST"])

            return df

        except Exception as exc:
            logging.getLogger(__name__).error(
                "Error calculating advanced metrics: %s", exc, exc_info=True
            )
            return game_logs  # fallback – return unmodified


    def get_player_stats(
        self,
        player_name: str,
        num_games: int = 30
    ) -> pd.DataFrame | None:
        """
        Pull recent game logs, inject advanced + fantasy-relevant stats,
        and return a trimmed DataFrame ready for time-series analysis.

        New in overhaul:
        • Includes STL, BLK, raw PTS/REB/AST, FGM/FGA, MINUTES_PLAYED
        • Orders columns so fantasy-volume metrics appear first.
        """
        logger = logging.getLogger(__name__)
        logger.info("Retrieving stats for player: %s", player_name)

        # ------------------------------------------------------------------ #
        # Identify player; abort if not found
        # ------------------------------------------------------------------ #
        player_id = self.get_player_id(player_name)
        if not player_id:
            logger.warning("Could not find player ID for %s", player_name)
            return None

        # ------------------------------------------------------------------ #
        # Raw game logs (handles rookies inside helper)
        # ------------------------------------------------------------------ #
        game_logs = self.get_recent_game_logs(player_name, num_games)
        if game_logs is None:
            logger.warning("No game logs for %s", player_name)
            return None

        # ------------------------------------------------------------------ #
        # Enrich with advanced & volume metrics
        # ------------------------------------------------------------------ #
        enriched = self.calculate_advanced_metrics(game_logs, player_id)

        if enriched is None or enriched.empty:
            logger.warning("Enrichment failed for %s", player_name)
            return None

        # ------------------------------------------------------------------ #
        # Final column whitelist – fantasy first, efficiency later
        # ------------------------------------------------------------------ #
        ts_columns = [
            # identifiers
            "GAME_DATE", "GAME_ID",
            # fantasy volume / defence
            "MINUTES_PLAYED", "PTS", "REB", "AST", "STL", "BLK",
            "FGM", "FGA",
            # useful rate/impact
            "FG_PCT", "TS_PCT",
            "PLUS_MINUS", "GAME_SCORE",
            "USG_RATE"
        ]
        available = [c for c in ts_columns if c in enriched.columns]

        return (
            enriched[available]
            .sort_values("GAME_DATE", ascending=False)
            .reset_index(drop=True)
        )


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
    
    # Test with problematic names
    test_names = [
        "L. James",           # Abbreviated first name
        "Jaren Jackson",      # Missing Jr. suffix
        "S. Gilgeous-Alexander",  # Hyphenated surname with initial
        "D. Gafford",         # Standard abbreviated name
        "J. Williams"         # Ambiguous abbreviated name
    ]
    
    for name in test_names:
        print(f"\nTesting name matching for: {name}")
        player_id = collector.get_player_id(name)
        if player_id:
            print(f"Successfully matched to player ID: {player_id}")
        else:
            print("No match found")