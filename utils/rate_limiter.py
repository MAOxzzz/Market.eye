"""
Rate limiter utility for API calls.
Helps prevent exceeding API rate limits by throttling requests.
"""

import time
import logging
from functools import wraps
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, max_calls, period=60):
        """
        Initialize the rate limiter.
        
        Args:
            max_calls (int): Maximum number of calls allowed in the period
            period (int): Period in seconds (default: 60 seconds)
        """
        self.max_calls = max_calls
        self.period = period
        self.calls = []
    
    def __call__(self, func):
        """
        Decorator to rate limit function calls.
        
        Args:
            func: Function to rate limit
            
        Returns:
            Wrapped function with rate limiting
        """
        @wraps(func)
        def wrapped(*args, **kwargs):
            # Remove calls older than the period
            now = datetime.now()
            self.calls = [call for call in self.calls if now - call < timedelta(seconds=self.period)]
            
            # Check if we've exceeded the rate limit
            if len(self.calls) >= self.max_calls:
                # Calculate time to wait
                oldest_call = self.calls[0]
                wait_time = (oldest_call + timedelta(seconds=self.period) - now).total_seconds()
                
                if wait_time > 0:
                    logger.warning(f"Rate limit reached. Waiting {wait_time:.2f} seconds before next call.")
                    time.sleep(wait_time)
                    
                    # After waiting, clear expired calls again
                    now = datetime.now()
                    self.calls = [call for call in self.calls if now - call < timedelta(seconds=self.period)]
            
            # Add this call to the list
            self.calls.append(now)
            
            # Execute the function
            return func(*args, **kwargs)
        
        return wrapped


def retry_with_backoff(max_retries=3, initial_delay=1, backoff_factor=2):
    """
    Decorator to retry a function with exponential backoff.
    
    Args:
        max_retries (int): Maximum number of retries
        initial_delay (int): Initial delay in seconds
        backoff_factor (int): Factor to multiply delay by after each retry
        
    Returns:
        Wrapped function with retry logic
    """
    def decorator(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            retries = 0
            delay = initial_delay
            
            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    
                    if retries > max_retries:
                        logger.error(f"Max retries ({max_retries}) exceeded. Giving up.")
                        raise
                    
                    logger.warning(f"API call failed. Retrying in {delay}s. Error: {str(e)}")
                    time.sleep(delay)
                    delay *= backoff_factor
        
        return wrapped
    
    return decorator 