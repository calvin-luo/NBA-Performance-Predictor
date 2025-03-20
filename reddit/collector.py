import re
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set, Tuple

import praw
from praw.models import Submission, Comment, Subreddit

from reddit.connector import get_connector, RedditConnector
from data.database import Database

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('reddit.collector')

# NBA Teams dictionary with common variations of team names and abbreviations
NBA_TEAMS = {
    'Atlanta Hawks': ['hawks', 'atl'],
    'Boston Celtics': ['celtics', 'boston', 'bos'],
    'Brooklyn Nets': ['nets', 'brooklyn', 'bkn'],
    'Charlotte Hornets': ['hornets', 'charlotte', 'cha'],
    'Chicago Bulls': ['bulls', 'chicago', 'chi'],
    'Cleveland Cavaliers': ['cavaliers', 'cavs', 'cleveland', 'cle'],
    'Dallas Mavericks': ['mavericks', 'mavs', 'dallas', 'dal'],
    'Denver Nuggets': ['nuggets', 'denver', 'den'],
    'Detroit Pistons': ['pistons', 'detroit', 'det'],
    'Golden State Warriors': ['warriors', 'golden state', 'gsw'],
    'Houston Rockets': ['rockets', 'houston', 'hou'],
    'Indiana Pacers': ['pacers', 'indiana', 'ind'],
    'Los Angeles Clippers': ['clippers', 'lac'],
    'Los Angeles Lakers': ['lakers', 'lal'],
    'Memphis Grizzlies': ['grizzlies', 'memphis', 'mem'],
    'Miami Heat': ['heat', 'miami', 'mia'],
    'Milwaukee Bucks': ['bucks', 'milwaukee', 'mil'],
    'Minnesota Timberwolves': ['timberwolves', 'wolves', 'minnesota', 'min'],
    'New Orleans Pelicans': ['pelicans', 'new orleans', 'nop'],
    'New York Knicks': ['knicks', 'new york', 'nyk'],
    'Oklahoma City Thunder': ['thunder', 'oklahoma', 'okc'],
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

# List of NBA-related subreddits to monitor
NBA_SUBREDDITS = ['nba', 'nbadiscussion', 'fantasybball']


class RedditCollector:
    """
    Collects NBA-related data from Reddit using the PRAW API.
    
    This class handles searching, filtering, and storing Reddit content
    related to NBA teams and players for sentiment analysis.
    """
    
    def __init__(self, connector: Optional[RedditConnector] = None, db: Optional[Database] = None):
        """
        Initialize the Reddit collector with a connector and database.
        
        Args:
            connector: RedditConnector instance (optional, creates a new one if None)
            db: Database instance (optional, creates a new one if None)
        """
        # Initialize connector (create a new one if not provided)
        self.connector = connector if connector else get_connector()
        
        # Initialize database (create a new one if not provided)
        self.db = db if db else Database()
        
        # Compile regex patterns for team mentions
        self.team_patterns = {}
        for team, aliases in NBA_TEAMS.items():
            pattern = r'\b(' + '|'.join(aliases) + r')\b'
            self.team_patterns[team] = re.compile(pattern, re.IGNORECASE)
    
    def get_subreddit(self, subreddit_name: str) -> Optional[Subreddit]:
        """
        Get a subreddit instance by name.
        
        Args:
            subreddit_name: Name of the subreddit
            
        Returns:
            Subreddit instance or None if connection fails
        """
        reddit = self.connector.get_reddit_instance()
        if not reddit:
            logger.error(f"Failed to get Reddit instance for subreddit: {subreddit_name}")
            return None
        
        try:
            return reddit.subreddit(subreddit_name)
        except Exception as e:
            logger.error(f"Error getting subreddit {subreddit_name}: {str(e)}")
            return None
    
    def search_for_team(self, team_name: str, subreddit_name: str, time_filter: str = 'day', 
                        limit: int = 100) -> List[Submission]:
        """
        Search for posts mentioning a specific team in a subreddit.
        
        Args:
            team_name: NBA team name to search for
            subreddit_name: Subreddit to search in
            time_filter: Time filter for search ('hour', 'day', 'week', 'month', 'year', 'all')
            limit: Maximum number of results to return
            
        Returns:
            List of Reddit Submission objects
        """
        subreddit = self.get_subreddit(subreddit_name)
        if not subreddit:
            return []
        
        # Get team aliases
        team_aliases = NBA_TEAMS.get(team_name, [team_name.lower()])
        
        # Create search query with team aliases
        query = ' OR '.join([f'"{alias}"' for alias in team_aliases if len(alias) > 2])
        
        try:
            logger.info(f"Searching for {team_name} in r/{subreddit_name} with query: {query}")
            search_results = list(subreddit.search(query, time_filter=time_filter, limit=limit))
            logger.info(f"Found {len(search_results)} posts for {team_name} in r/{subreddit_name}")
            return search_results
        
        except Exception as e:
            logger.error(f"Error searching for {team_name} in r/{subreddit_name}: {str(e)}")
            return []
    
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
        
        for team, pattern in self.team_patterns.items():
            if pattern.search(text):
                mentioned_teams.append(team)
        
        return mentioned_teams
    
    def extract_post_data(self, submission: Submission) -> Dict[str, Any]:
        """
        Extract relevant data from a Reddit submission.
        
        Args:
            submission: PRAW Submission object
            
        Returns:
            Dictionary containing extracted post data
        """
        # Get post content (either selftext or body text from url posts)
        content = submission.selftext if submission.is_self else submission.url
        
        # Detect team mentions in title and content
        title_mentions = self.detect_team_mentions(submission.title)
        content_mentions = self.detect_team_mentions(content)
        
        # Combine unique team mentions
        team_mentions = list(set(title_mentions + content_mentions))
        
        # Format post data
        post_data = {
            'reddit_id': submission.id,
            'subreddit': submission.subreddit.display_name,
            'title': submission.title,
            'content': content,
            'author': str(submission.author) if submission.author else '[deleted]',
            'created_utc': submission.created_utc,
            'score': submission.score,
            'team_mention': team_mentions[0] if team_mentions else None,
            'player_mention': None,  # Will be implemented in future update
            'game_id': None  # Will be linked to games later
        }
        
        return post_data
    
    def extract_comment_data(self, comment: Comment, post_id: int) -> Dict[str, Any]:
        """
        Extract relevant data from a Reddit comment.
        
        Args:
            comment: PRAW Comment object
            post_id: Database ID of the parent post
            
        Returns:
            Dictionary containing extracted comment data
        """
        comment_data = {
            'reddit_id': comment.id,
            'post_id': post_id,
            'content': comment.body,
            'author': str(comment.author) if comment.author else '[deleted]',
            'created_utc': comment.created_utc,
            'score': comment.score
        }
        
        return comment_data
    
    def collect_post_comments(self, submission: Submission, post_id: int, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Collect comments from a Reddit post.
        
        Args:
            submission: PRAW Submission object
            post_id: Database ID of the parent post
            limit: Maximum number of comments to collect
            
        Returns:
            List of dictionaries containing comment data
        """
        try:
            # Replace more comments to get a fuller comment tree
            submission.comments.replace_more(limit=5)
            
            # Get top-level comments up to the limit
            comments = list(submission.comments.list())[:limit]
            
            # Extract data from each comment
            comment_data_list = []
            for comment in comments:
                try:
                    comment_data = self.extract_comment_data(comment, post_id)
                    comment_data_list.append(comment_data)
                except Exception as e:
                    logger.error(f"Error extracting comment data: {str(e)}")
            
            logger.info(f"Collected {len(comment_data_list)} comments from post {submission.id}")
            return comment_data_list
            
        except Exception as e:
            logger.error(f"Error collecting comments from post {submission.id}: {str(e)}")
            return []
    
    def save_post_and_comments(self, submission: Submission) -> Optional[int]:
        """
        Save a Reddit post and its comments to the database.
        
        Args:
            submission: PRAW Submission object
            
        Returns:
            post_id: Database ID of the saved post or None if failed
        """
        try:
            # Extract post data
            post_data = self.extract_post_data(submission)
            
            # Save post to database
            post_id = self.db.insert_reddit_post(post_data)
            
            if post_id:
                # Collect and save comments
                comment_data_list = self.collect_post_comments(submission, post_id)
                
                for comment_data in comment_data_list:
                    self.db.insert_reddit_comment(comment_data)
                
                logger.info(f"Saved post {submission.id} with {len(comment_data_list)} comments")
                return post_id
            else:
                logger.warning(f"Failed to save post {submission.id}")
                return None
                
        except Exception as e:
            logger.error(f"Error saving post {submission.id}: {str(e)}")
            return None
    
    def collect_team_data(self, team_name: str, days_back: int = 7) -> Tuple[int, int]:
        """
        Collect Reddit data for a specific NBA team.
        
        Args:
            team_name: Name of the NBA team
            days_back: Number of days to look back for posts
            
        Returns:
            Tuple of (posts_collected, comments_collected)
        """
        if team_name not in NBA_TEAMS:
            logger.error(f"Invalid team name: {team_name}")
            return (0, 0)
        
        posts_collected = 0
        comments_collected = 0
        
        # Determine time filter based on days_back
        time_filter = 'day'
        if days_back > 1 and days_back <= 7:
            time_filter = 'week'
        elif days_back > 7 and days_back <= 30:
            time_filter = 'month'
        elif days_back > 30:
            time_filter = 'year'
        
        # Collect data from each NBA subreddit
        for subreddit_name in NBA_SUBREDDITS:
            try:
                # Search for team posts
                submissions = self.search_for_team(team_name, subreddit_name, time_filter)
                
                # Filter by date if needed
                if days_back < 7 and time_filter == 'week':
                    cutoff_time = (datetime.now() - timedelta(days=days_back)).timestamp()
                    submissions = [s for s in submissions if s.created_utc >= cutoff_time]
                
                # Save each post and its comments
                for submission in submissions:
                    post_id = self.save_post_and_comments(submission)
                    
                    if post_id:
                        posts_collected += 1
                        # Approximate comment count for return value
                        comments_collected += min(len(submission.comments.list()), 100)
                    
                    # Sleep to respect rate limits
                    time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error collecting data for {team_name} in r/{subreddit_name}: {str(e)}")
        
        logger.info(f"Collected {posts_collected} posts and ~{comments_collected} comments for {team_name}")
        return (posts_collected, comments_collected)
    
    def collect_all_teams_data(self, days_back: int = 3) -> Dict[str, Tuple[int, int]]:
        """
        Collect Reddit data for all NBA teams.
        
        Args:
            days_back: Number of days to look back for posts
            
        Returns:
            Dictionary mapping team names to (posts_collected, comments_collected) tuples
        """
        results = {}
        
        for team_name in NBA_TEAMS.keys():
            logger.info(f"Collecting data for {team_name}")
            posts, comments = self.collect_team_data(team_name, days_back)
            results[team_name] = (posts, comments)
            
            # Sleep between teams to respect rate limits
            time.sleep(2)
        
        total_posts = sum(p for p, _ in results.values())
        total_comments = sum(c for _, c in results.values())
        logger.info(f"Collection complete. Total: {total_posts} posts, ~{total_comments} comments")
        
        return results
    
    def collect_recent_data(self, days_back: int = 3, teams: Optional[List[str]] = None) -> Dict[str, Tuple[int, int]]:
        """
        Collect recent Reddit data for specified NBA teams or all teams.
        
        Args:
            days_back: Number of days to look back for posts
            teams: List of team names to collect data for (all teams if None)
            
        Returns:
            Dictionary mapping team names to (posts_collected, comments_collected) tuples
        """
        if teams is None:
            return self.collect_all_teams_data(days_back)
        
        results = {}
        for team_name in teams:
            if team_name in NBA_TEAMS:
                logger.info(f"Collecting data for {team_name}")
                posts, comments = self.collect_team_data(team_name, days_back)
                results[team_name] = (posts, comments)
                
                # Sleep between teams to respect rate limits
                time.sleep(2)
            else:
                logger.warning(f"Invalid team name: {team_name}")
                results[team_name] = (0, 0)
        
        return results


# Example usage
if __name__ == "__main__":
    # Set up database
    db = Database()
    db.initialize_database()
    
    # Create collector
    collector = RedditCollector(db=db)
    
    # Example: Collect data for a few specific teams
    teams_to_collect = ['Boston Celtics', 'Los Angeles Lakers', 'Golden State Warriors']
    results = collector.collect_recent_data(days_back=2, teams=teams_to_collect)
    
    # Print results
    for team, (posts, comments) in results.items():
        print(f"{team}: {posts} posts, ~{comments} comments")