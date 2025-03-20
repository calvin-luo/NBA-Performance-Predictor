import re
import time
import logging
import statistics
from typing import List, Dict, Any, Tuple, Optional, Union
from datetime import datetime, timedelta

# For sentiment analysis
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from data.database import Database

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('analysis.sentiment')

# Ensure required NLTK data is downloaded
try:
    nltk.data.find('vader_lexicon')
    nltk.data.find('punkt')
    nltk.data.find('stopwords')
except LookupError:
    logger.info("Downloading required NLTK data...")
    nltk.download('vader_lexicon')
    nltk.download('punkt')
    nltk.download('stopwords')


class NBATextProcessor:
    """
    Handles text preprocessing and entity recognition for NBA data.
    """
    
    def __init__(self):
        """Initialize the NBA text processor."""
        # Load stopwords
        self.stop_words = set(stopwords.words('english'))
        
        # NBA related terms to keep, even if they're stopwords
        self.nba_keep_terms = {
            'win', 'wins', 'won', 'lose', 'loses', 'lost', 'beat', 'beats', 
            'defeating', 'defeated', 'victory', 'loss'
        }
        
        # Adjusted stop words (remove NBA terms from stop words)
        self.stop_words = self.stop_words - self.nba_keep_terms
        
        # NBA Team names and common variations (from collector.py)
        self.team_variations = {
            'Atlanta Hawks': ['hawks', 'atl', 'atlanta'],
            'Boston Celtics': ['celtics', 'boston', 'bos'],
            'Brooklyn Nets': ['nets', 'brooklyn', 'bkn'],
            'Charlotte Hornets': ['hornets', 'charlotte', 'cha'],
            'Chicago Bulls': ['bulls', 'chicago', 'chi'],
            'Cleveland Cavaliers': ['cavaliers', 'cavs', 'cleveland', 'cle'],
            'Dallas Mavericks': ['mavericks', 'mavs', 'dallas', 'dal'],
            'Denver Nuggets': ['nuggets', 'denver', 'den'],
            'Detroit Pistons': ['pistons', 'detroit', 'det'],
            'Golden State Warriors': ['warriors', 'golden state', 'gsw', 'golden'],
            'Houston Rockets': ['rockets', 'houston', 'hou'],
            'Indiana Pacers': ['pacers', 'indiana', 'ind'],
            'Los Angeles Clippers': ['clippers', 'lac', 'la clippers'],
            'Los Angeles Lakers': ['lakers', 'lal', 'la lakers'],
            'Memphis Grizzlies': ['grizzlies', 'memphis', 'mem'],
            'Miami Heat': ['heat', 'miami', 'mia'],
            'Milwaukee Bucks': ['bucks', 'milwaukee', 'mil'],
            'Minnesota Timberwolves': ['timberwolves', 'wolves', 'minnesota', 'min'],
            'New Orleans Pelicans': ['pelicans', 'new orleans', 'nop'],
            'New York Knicks': ['knicks', 'new york', 'nyk'],
            'Oklahoma City Thunder': ['thunder', 'oklahoma', 'okc', 'oklahoma city'],
            'Orlando Magic': ['magic', 'orlando', 'orl'],
            'Philadelphia 76ers': ['76ers', 'sixers', 'philadelphia', 'phi'],
            'Phoenix Suns': ['suns', 'phoenix', 'phx'],
            'Portland Trail Blazers': ['trail blazers', 'blazers', 'portland', 'por'],
            'Sacramento Kings': ['kings', 'sacramento', 'sac'],
            'San Antonio Spurs': ['spurs', 'san antonio', 'sas'],
            'Toronto Raptors': ['raptors', 'toronto', 'tor'],
            'Utah Jazz': ['jazz', 'utah', 'uta'],
            'Washington Wizards': ['wizards', 'washington', 'was']
        }
        
        # Compile regex patterns for team mentions
        self.team_patterns = {}
        for team, aliases in self.team_variations.items():
            pattern = r'\b(' + '|'.join([re.escape(alias) for alias in aliases]) + r')\b'
            self.team_patterns[team] = re.compile(pattern, re.IGNORECASE)
        
        # Create reverse lookup from variation to team name
        self.variation_to_team = {}
        for team, variations in self.team_variations.items():
            for variation in variations:
                self.variation_to_team[variation.lower()] = team
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for sentiment analysis.
        
        Args:
            text: The text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove special characters and numbers but keep apostrophes for contractions
        text = re.sub(r'[^a-zA-Z\' ]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text and remove stopwords.
        
        Args:
            text: The text to tokenize
            
        Returns:
            List of tokens
        """
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        filtered_tokens = [word for word in tokens if word.lower() not in self.stop_words]
        
        return filtered_tokens
    
    def detect_team_mentions(self, text: str) -> List[str]:
        """
        Detect mentions of NBA teams in a text.
        
        Args:
            text: Text to analyze for team mentions
            
        Returns:
            List of team names mentioned in the text
        """
        if not text:
            return []
        
        mentioned_teams = []
        
        # First pass: look for team pattern matches
        team_matches = {}
        for team, pattern in self.team_patterns.items():
            matches = pattern.findall(text)
            if matches:
                team_matches[team] = len(matches)
        
        # Sort by number of matches (most frequently mentioned first)
        sorted_teams = sorted(team_matches.items(), key=lambda x: x[1], reverse=True)
        
        # Only add teams that are likely to be actual mentions
        for team, match_count in sorted_teams:
            # For shorter team patterns, require multiple mentions or strong context
            team_aliases = self.team_variations.get(team, [])
            if any(len(alias) <= 3 for alias in team_aliases):
                # For short aliases, check for stronger context
                team_context_pattern = re.compile(
                    r'\b(' + '|'.join([re.escape(alias) for alias in team_aliases]) + r')\s+(team|game|play|win|lose|vs|against|match)\b', 
                    re.IGNORECASE
                )
                if team_context_pattern.search(text) or match_count > 1:
                    mentioned_teams.append(team)
            else:
                # For longer team names/patterns, one mention is sufficient
                mentioned_teams.append(team)
        
        return mentioned_teams
    
    def detect_player_mentions(self, text: str, team_players: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[str]]:
        """
        Detect mentions of NBA players in a text based on provided team rosters.
        
        Args:
            text: Text to analyze for player mentions
            team_players: Dictionary mapping team names to lists of player dictionaries
            
        Returns:
            Dictionary mapping team names to lists of player names mentioned
        """
        if not text or not team_players:
            return {}
        
        mentioned_players = {}
        
        # Create a regex pattern for each player's name
        # Group players by team to track which team's players are mentioned
        for team, players in team_players.items():
            player_patterns = []
            player_name_map = {}  # Map pattern matches back to full player names
            
            for player in players:
                full_name = player['name']
                
                # Handle cases where player name has suffixes or prefixes
                clean_name = re.sub(r'(Jr\.|Sr\.|III|II|IV|V)$', '', full_name).strip()
                
                # Extract first and last name
                name_parts = clean_name.split()
                if len(name_parts) >= 2:
                    first_name = name_parts[0]
                    last_name = name_parts[-1]
                    
                    # Add patterns for full name and last name only
                    full_pattern = re.escape(clean_name)
                    last_pattern = r'\b' + re.escape(last_name) + r'\b'
                    
                    player_patterns.append(full_pattern)
                    player_patterns.append(last_pattern)
                    
                    # Map both patterns back to the full player name
                    player_name_map[clean_name.lower()] = full_name
                    player_name_map[last_name.lower()] = full_name
                else:
                    # Only one name part (unusual)
                    player_patterns.append(r'\b' + re.escape(clean_name) + r'\b')
                    player_name_map[clean_name.lower()] = full_name
            
            # Create a combined pattern for all players on this team
            if player_patterns:
                combined_pattern = '|'.join(player_patterns)
                player_regex = re.compile(combined_pattern, re.IGNORECASE)
                
                # Find all matches
                matches = player_regex.findall(text)
                
                # Convert matches back to full player names and remove duplicates
                if matches:
                    team_mentions = []
                    for match in matches:
                        match_lower = match.lower()
                        if match_lower in player_name_map:
                            team_mentions.append(player_name_map[match_lower])
                    
                    if team_mentions:
                        mentioned_players[team] = list(set(team_mentions))
        
        return mentioned_players


class SentimentAnalyzer:
    """
    Analyzes sentiment in NBA-related text content.
    """
    
    def __init__(self, db: Optional[Database] = None):
        """
        Initialize the sentiment analyzer.
        
        Args:
            db: Database instance (optional, creates a new one if None)
        """
        # Initialize database (create a new one if not provided)
        self.db = db if db else Database()
        
        # Initialize VADER sentiment analyzer
        self.vader = SentimentIntensityAnalyzer()
        
        # Initialize text processor
        self.text_processor = NBATextProcessor()
        
        # NBA-specific sentiment lexicon adjustments
        self.nba_lexicon_adjustments = {
            # Positive terms in NBA context
            'win': 1.5,
            'wins': 1.5,
            'won': 1.5,
            'champion': 2.0,
            'championship': 2.0,
            'clutch': 1.8,
            'dominate': 1.5,
            'dominant': 1.5,
            'elite': 1.8,
            'great': 1.2,
            'goat': 2.0,
            'mvp': 1.7,
            'star': 1.3,
            'superstar': 1.8,
            'solid': 0.8,
            'leader': 1.2,
            'underrated': 1.0,
            
            # Negative terms in NBA context
            'lose': -1.5,
            'loses': -1.5,
            'lost': -1.5,
            'trash': -1.8,
            'weak': -1.0,
            'bust': -1.5,
            'overrated': -1.2,
            'worst': -1.5,
            'awful': -1.5,
            'horrible': -1.5,
            'terrible': -1.5,
            'disappointing': -1.2,
            'injured': -1.3,
            'injury': -1.3,
            'turnover': -0.8,
            'turnovers': -0.8,
            'foul': -0.7,
            'fouls': -0.7,
            'bench': -0.5
        }
        
        # Update the VADER lexicon with NBA-specific adjustments
        for word, score in self.nba_lexicon_adjustments.items():
            self.vader.lexicon[word] = score
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment in a text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores and entities
        """
        if not text:
            return {
                'compound': 0.0,
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 0.0,
                'teams': [],
                'team_contexts': {}
            }
        
        # Preprocess text
        processed_text = self.text_processor.preprocess_text(text)
        
        # Get sentiment scores
        sentiment_scores = self.vader.polarity_scores(processed_text)
        
        # Detect team mentions
        mentioned_teams = self.text_processor.detect_team_mentions(text)
        
        # Analyze sentiment in context of each team
        team_contexts = {}
        for team in mentioned_teams:
            # Look for sentences or segments that mention the team
            team_aliases = self.text_processor.team_variations.get(team, [team.lower()])
            
            # Split text into sentences
            sentences = re.split(r'[.!?]', text)
            
            team_sentences = []
            for sentence in sentences:
                if any(re.search(r'\b' + re.escape(alias) + r'\b', sentence, re.IGNORECASE) for alias in team_aliases):
                    team_sentences.append(sentence)
            
            # Calculate sentiment specifically for content mentioning this team
            if team_sentences:
                team_text = ' '.join(team_sentences)
                team_processed = self.text_processor.preprocess_text(team_text)
                team_sentiment = self.vader.polarity_scores(team_processed)
                
                team_contexts[team] = {
                    'sentiment': team_sentiment,
                    'context': team_text[:200] + '...' if len(team_text) > 200 else team_text
                }
        
        # Compile results
        result = {
            'compound': sentiment_scores['compound'],
            'positive': sentiment_scores['pos'],
            'negative': sentiment_scores['neg'],
            'neutral': sentiment_scores['neu'],
            'teams': mentioned_teams,
            'team_contexts': team_contexts
        }
        
        return result
    
    def analyze_reddit_content(self, content_type: str, content_id: int) -> Dict[str, Any]:
        """
        Analyze sentiment in Reddit content (post or comment).
        
        Args:
            content_type: Type of content ('post' or 'comment')
            content_id: ID of the content in the database
            
        Returns:
            Dictionary with sentiment analysis results
        """
        # Get content from database
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            if content_type == 'post':
                cursor.execute(
                    "SELECT post_id, title, content, team_mention FROM reddit_posts WHERE post_id = ?",
                    (content_id,)
                )
                data = cursor.fetchone()
                
                if not data:
                    logger.warning(f"Post with ID {content_id} not found")
                    return {}
                
                # Combine title and content for analysis
                text = f"{data['title']} {data['content']}"
                team_mention = data['team_mention']
                
            elif content_type == 'comment':
                cursor.execute(
                    "SELECT c.comment_id, c.content, p.team_mention FROM reddit_comments c JOIN reddit_posts p ON c.post_id = p.post_id WHERE c.comment_id = ?",
                    (content_id,)
                )
                data = cursor.fetchone()
                
                if not data:
                    logger.warning(f"Comment with ID {content_id} not found")
                    return {}
                
                text = data['content']
                team_mention = data['team_mention']
                
            else:
                logger.error(f"Invalid content type: {content_type}")
                return {}
        
        # Analyze sentiment
        sentiment_result = self.analyze_text(text)
        
        # Add entity information
        result = {
            'content_type': content_type,
            'content_id': content_id,
            'sentiment': sentiment_result,
            'team_mention': team_mention
        }
        
        return result
    
    def save_sentiment_analysis(self, analysis_results: Dict[str, Any]) -> List[int]:
        """
        Save sentiment analysis results to the database.
        
        Args:
            analysis_results: Dictionary with sentiment analysis results
            
        Returns:
            List of sentiment_id values for saved records
        """
        if not analysis_results or 'sentiment' not in analysis_results:
            logger.warning("No sentiment data to save")
            return []
        
        content_type = analysis_results.get('content_type')
        content_id = analysis_results.get('content_id')
        sentiment_data = analysis_results.get('sentiment', {})
        team_mention = analysis_results.get('team_mention')
        team_contexts = sentiment_data.get('team_contexts', {})
        
        saved_ids = []
        
        try:
            # Save overall sentiment for the explicitly mentioned team (if any)
            if team_mention:
                sentiment_record = {
                    'entity_type': 'team',
                    'entity_id': team_mention,
                    f"{content_type}_id": content_id,
                    'sentiment_score': sentiment_data.get('compound', 0.0),
                    'confidence': max(sentiment_data.get('positive', 0.0), sentiment_data.get('negative', 0.0))
                }
                
                sentiment_id = self.db.insert_sentiment_analysis(sentiment_record)
                saved_ids.append(sentiment_id)
            
            # Save team-specific context sentiments (for teams mentioned in the content)
            for team, context_data in team_contexts.items():
                # Skip the primary team mention (already saved above)
                if team == team_mention:
                    continue
                
                context_sentiment = context_data.get('sentiment', {})
                
                sentiment_record = {
                    'entity_type': 'team',
                    'entity_id': team,
                    f"{content_type}_id": content_id,
                    'sentiment_score': context_sentiment.get('compound', 0.0),
                    'confidence': max(
                        context_sentiment.get('pos', 0.0),
                        context_sentiment.get('neg', 0.0)
                    )
                }
                
                sentiment_id = self.db.insert_sentiment_analysis(sentiment_record)
                saved_ids.append(sentiment_id)
            
            return saved_ids
        
        except Exception as e:
            logger.error(f"Error saving sentiment analysis: {str(e)}")
            return []
    
    def analyze_and_save_reddit_content(self, content_type: str, content_id: int) -> Dict[str, Any]:
        """
        Analyze sentiment in Reddit content and save the results.
        
        Args:
            content_type: Type of content ('post' or 'comment')
            content_id: ID of the content in the database
            
        Returns:
            Dictionary with sentiment analysis results and saved IDs
        """
        # Analyze sentiment
        analysis_results = self.analyze_reddit_content(content_type, content_id)
        
        if not analysis_results:
            return {'error': f"Could not analyze {content_type} with ID {content_id}"}
        
        # Save results
        saved_ids = self.save_sentiment_analysis(analysis_results)
        
        # Add saved IDs to results
        analysis_results['saved_ids'] = saved_ids
        
        return analysis_results
    
    def analyze_recent_content(self, hours_back: int = 24) -> Tuple[int, int]:
        """
        Analyze sentiment for all recent Reddit content.
        
        Args:
            hours_back: Number of hours to look back for content
            
        Returns:
            Tuple of (posts_analyzed, comments_analyzed)
        """
        posts_analyzed = 0
        comments_analyzed = 0
        cutoff_time = (datetime.now() - timedelta(hours=hours_back)).timestamp()
        
        # Get recent posts
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Find posts without sentiment analysis
            cursor.execute(
                """
                SELECT p.post_id
                FROM reddit_posts p
                LEFT JOIN sentiment_analysis s ON s.post_id = p.post_id
                WHERE p.created_utc > ? AND s.sentiment_id IS NULL
                ORDER BY p.created_utc DESC
                """,
                (cutoff_time,)
            )
            
            post_ids = [row['post_id'] for row in cursor.fetchall()]
            
            # Find comments without sentiment analysis
            cursor.execute(
                """
                SELECT c.comment_id
                FROM reddit_comments c
                LEFT JOIN sentiment_analysis s ON s.comment_id = c.comment_id
                WHERE c.created_utc > ? AND s.sentiment_id IS NULL
                ORDER BY c.created_utc DESC
                """,
                (cutoff_time,)
            )
            
            comment_ids = [row['comment_id'] for row in cursor.fetchall()]
        
        # Analyze posts
        for post_id in post_ids:
            try:
                self.analyze_and_save_reddit_content('post', post_id)
                posts_analyzed += 1
                # Add a small delay to avoid database contention
                time.sleep(0.01)
            except Exception as e:
                logger.error(f"Error analyzing post {post_id}: {str(e)}")
        
        # Analyze comments
        for comment_id in comment_ids:
            try:
                self.analyze_and_save_reddit_content('comment', comment_id)
                comments_analyzed += 1
                # Add a small delay to avoid database contention
                time.sleep(0.01)
            except Exception as e:
                logger.error(f"Error analyzing comment {comment_id}: {str(e)}")
        
        logger.info(f"Analyzed {posts_analyzed} posts and {comments_analyzed} comments")
        return (posts_analyzed, comments_analyzed)
    
    def get_team_sentiment(self, team: str, hours_back: int = 24) -> Dict[str, Any]:
        """
        Get aggregated sentiment data for a specific team.
        
        Args:
            team: Team name
            hours_back: Number of hours to look back
            
        Returns:
            Dictionary with sentiment statistics
        """
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        cutoff_str = cutoff_time.isoformat()
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get all sentiment records for the team within the time period
            cursor.execute(
                """
                SELECT sentiment_score, confidence, created_at, post_id, comment_id
                FROM sentiment_analysis
                WHERE entity_type = 'team' AND entity_id = ? AND created_at > ?
                """,
                (team, cutoff_str)
            )
            
            records = [dict(row) for row in cursor.fetchall()]
        
        if not records:
            logger.warning(f"No sentiment data found for team {team} in the last {hours_back} hours")
            return {
                'team': team,
                'count': 0,
                'avg_sentiment': 0.0,
                'sentiment_trend': 0.0,
                'confidence': 0.0,
                'post_count': 0,
                'comment_count': 0
            }
        
        # Calculate statistics
        scores = [record['sentiment_score'] for record in records]
        confidences = [record['confidence'] for record in records]
        post_count = sum(1 for record in records if record['post_id'] is not None)
        comment_count = sum(1 for record in records if record['comment_id'] is not None)
        
        # Sort records by time for trend analysis
        time_sorted_records = sorted(records, key=lambda r: r['created_at'])
        
        # Calculate trend (change in sentiment over time)
        if len(time_sorted_records) >= 2:
            # Simple linear trend calculation
            earliest_sentiment = time_sorted_records[0]['sentiment_score']
            latest_sentiment = time_sorted_records[-1]['sentiment_score']
            sentiment_trend = latest_sentiment - earliest_sentiment
        else:
            sentiment_trend = 0.0
        
        # Compile results
        result = {
            'team': team,
            'count': len(records),
            'avg_sentiment': statistics.mean(scores) if scores else 0.0,
            'sentiment_trend': sentiment_trend,
            'confidence': statistics.mean(confidences) if confidences else 0.0,
            'post_count': post_count,
            'comment_count': comment_count,
            'min_sentiment': min(scores) if scores else 0.0,
            'max_sentiment': max(scores) if scores else 0.0
        }
        
        return result


# Example usage
if __name__ == "__main__":
    # Set up analyzer
    db = Database()
    db.initialize_database()
    
    analyzer = SentimentAnalyzer(db=db)
    
    # Example text analysis
    sample_text = """
    The Boston Celtics played incredibly well last night. Tatum hit a clutch three-pointer to seal the win.
    Even though the Lakers have been struggling lately, they still have a chance to make the playoffs.
    The Warriors looked terrible in their last game, turning the ball over way too much.
    """
    
    result = analyzer.analyze_text(sample_text)
    print("Sentiment Analysis Result:")
    print(f"Compound Score: {result['compound']}")
    print(f"Positive: {result['positive']}")
    print(f"Negative: {result['negative']}")
    print(f"Neutral: {result['neutral']}")
    print(f"Teams Mentioned: {result['teams']}")
    
    # Example team sentiment retrieval
    for team in ['Boston Celtics', 'Los Angeles Lakers', 'Golden State Warriors']:
        team_sentiment = analyzer.get_team_sentiment(team, hours_back=48)
        print(f"\n{team} Sentiment:")
        print(f"Average Score: {team_sentiment['avg_sentiment']:.2f}")
        print(f"Trend: {team_sentiment['sentiment_trend']:.2f}")
        print(f"Based on {team_sentiment['count']} data points")