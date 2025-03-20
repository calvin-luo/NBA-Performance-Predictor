import os
import praw
import logging
from typing import Optional
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('reddit.connector')

class RedditConnector:
    """
    Handles connection and authentication to the Reddit API using PRAW.
    
    This class manages Reddit API credentials and provides a configured
    PRAW Reddit instance for interacting with Reddit's API.
    """
    
    def __init__(self, client_id: Optional[str] = None, 
                 client_secret: Optional[str] = None, 
                 user_agent: Optional[str] = None):
        """
        Initialize the Reddit connector with API credentials.
        
        Args:
            client_id: Reddit API client ID (optional, defaults to env variable)
            client_secret: Reddit API client secret (optional, defaults to env variable)
            user_agent: Reddit API user agent (optional, defaults to env variable)
        """
        # Load environment variables if not done already
        load_dotenv()
        
        # Use provided credentials or fall back to environment variables
        self.client_id = client_id or os.getenv('REDDIT_CLIENT_ID')
        self.client_secret = client_secret or os.getenv('REDDIT_CLIENT_SECRET')
        self.user_agent = user_agent or os.getenv('REDDIT_USER_AGENT')
        
        # Validate credentials
        self._validate_credentials()
        
        # Initialize PRAW instance to None (will be created when needed)
        self.reddit = None
    
    def _validate_credentials(self) -> None:
        """
        Validate that all required credentials are present.
        
        Raises:
            ValueError: If any required credential is missing
        """
        if not self.client_id:
            logger.error("Missing Reddit client ID")
            raise ValueError("Reddit client ID is required. Set REDDIT_CLIENT_ID environment variable or pass to constructor.")
        
        if not self.client_secret:
            logger.error("Missing Reddit client secret")
            raise ValueError("Reddit client secret is required. Set REDDIT_CLIENT_SECRET environment variable or pass to constructor.")
        
        if not self.user_agent:
            logger.error("Missing Reddit user agent")
            raise ValueError("Reddit user agent is required. Set REDDIT_USER_AGENT environment variable or pass to constructor.")
    
    def connect(self) -> praw.Reddit:
        """
        Create and configure a PRAW Reddit instance.
        
        Returns:
            praw.Reddit: A configured PRAW Reddit instance
            
        Raises:
            Exception: If connection fails
        """
        try:
            # Create PRAW Reddit instance with credentials
            self.reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent
            )
            
            # Verify that the instance is working by checking rate limit
            self.reddit.auth.limits
            
            logger.info("Successfully connected to Reddit API")
            return self.reddit
        
        except Exception as e:
            logger.error(f"Failed to connect to Reddit API: {str(e)}")
            raise
    
    def get_reddit_instance(self) -> Optional[praw.Reddit]:
        """
        Get the current PRAW Reddit instance, connecting if necessary.
        
        Returns:
            praw.Reddit: A configured PRAW Reddit instance or None if connection fails
        """
        if self.reddit is None:
            try:
                self.connect()
            except Exception as e:
                logger.error(f"Could not get Reddit instance: {str(e)}")
                return None
        
        return self.reddit
    
    def check_connection(self) -> bool:
        """
        Check if the connection to Reddit API is working.
        
        Returns:
            bool: True if connection is working, False otherwise
        """
        try:
            instance = self.get_reddit_instance()
            if instance is None:
                return False
            
            # Test the connection by fetching rate limit info
            instance.auth.limits
            return True
        
        except Exception as e:
            logger.error(f"Reddit connection check failed: {str(e)}")
            return False
    
    def close(self) -> None:
        """
        Close the Reddit connection and clean up.
        
        Note: PRAW doesn't have an explicit close method, but this method
        is provided for API consistency and future-proofing.
        """
        # Set the reddit instance to None to allow for garbage collection
        if self.reddit is not None:
            self.reddit = None
            logger.info("Reddit connection closed")


def get_connector(client_id: Optional[str] = None, 
                 client_secret: Optional[str] = None, 
                 user_agent: Optional[str] = None) -> RedditConnector:
    """
    Factory function to create and return a RedditConnector instance.
    
    Args:
        client_id: Reddit API client ID (optional, defaults to env variable)
        client_secret: Reddit API client secret (optional, defaults to env variable)
        user_agent: Reddit API user agent (optional, defaults to env variable)
        
    Returns:
        RedditConnector: A configured RedditConnector instance
    """
    return RedditConnector(client_id, client_secret, user_agent)


# Example usage
if __name__ == "__main__":
    try:
        # Get a connector with environment variable credentials
        connector = get_connector()
        
        # Connect to Reddit API
        reddit = connector.connect()
        
        # Test the connection by printing some info
        print(f"Read-only: {reddit.read_only}")
        print(f"Username: {reddit.user.me() if not reddit.read_only else 'Not logged in (read-only mode)'}")
        
        # Get a subreddit and print some basic info
        nba_subreddit = reddit.subreddit("nba")
        print(f"Subreddit subscribers: {nba_subreddit.subscribers}")
        
        # Close the connection when done
        connector.close()
        
    except Exception as e:
        print(f"Error in example: {str(e)}")