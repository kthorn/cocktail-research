"""Polite web scraping utilities with robots.txt compliance."""

import random
import time
from urllib import robotparser
from urllib.parse import urljoin
from typing import Optional

import requests

from .retry import retry_on_connection_error


def check_robots_allowed(base_url: str, url: str, user_agent: str = "*") -> bool:
    """Check if a URL is allowed by robots.txt.
    
    Args:
        base_url: Base URL of the site (e.g., "https://example.com")
        url: Full URL to check
        user_agent: User agent string to check against
        
    Returns:
        True if the URL is allowed, False otherwise
        
    Example:
        >>> check_robots_allowed("https://example.com", "https://example.com/page", "mybot/1.0")
        True
    """
    try:
        rp = robotparser.RobotFileParser()
        rp.set_url(urljoin(base_url, "/robots.txt"))
        rp.read()
        return rp.can_fetch(user_agent, url)
    except Exception:
        # If we can't read robots.txt, assume it's allowed
        return True


class PoliteSession:
    """A requests session that respects robots.txt and implements polite crawling delays.
    
    Attributes:
        session: The underlying requests session
        base_url: Base URL for robots.txt checking
        user_agent: User agent string
        crawl_delay: Delay between requests in seconds
        jitter_range: Random jitter range to add to delays
    """
    
    def __init__(
        self, 
        base_url: str, 
        user_agent: str = "cocktail-bot/1.0", 
        crawl_delay: Optional[float] = None,
        jitter_range: tuple = (0.5, 1.5)
    ):
        """Initialize polite session.
        
        Args:
            base_url: Base URL of the site being crawled
            user_agent: User agent string to use
            crawl_delay: Fixed crawl delay. If None, will try to read from robots.txt
            jitter_range: Tuple of (min, max) for random jitter to add to delays
        """
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self.base_url = base_url
        self.user_agent = user_agent
        self.jitter_range = jitter_range
        
        # Try to get crawl delay from robots.txt
        if crawl_delay is None:
            try:
                rp = robotparser.RobotFileParser()
                rp.set_url(urljoin(base_url, "/robots.txt"))
                rp.read()
                self.crawl_delay = rp.crawl_delay(user_agent) or 1.0
            except Exception:
                self.crawl_delay = 1.0
        else:
            self.crawl_delay = crawl_delay
    
    def _is_allowed(self, url: str) -> bool:
        """Check if URL is allowed by robots.txt."""
        return check_robots_allowed(self.base_url, url, self.user_agent)
    
    def _wait(self) -> None:
        """Wait with crawl delay plus random jitter."""
        jitter = random.uniform(*self.jitter_range)
        time.sleep(self.crawl_delay + jitter)
    
    @retry_on_connection_error()
    def get(self, url: str, **kwargs) -> requests.Response:
        """Make a polite GET request with robots.txt checking and delays.
        
        Args:
            url: URL to request
            **kwargs: Additional arguments passed to requests.get()
            
        Returns:
            Response object
            
        Raises:
            RuntimeError: If robots.txt disallows the URL
        """
        if not self._is_allowed(url):
            raise RuntimeError(f"robots.txt disallows {url}")
        
        self._wait()
        
        # Set default timeout if not provided
        if 'timeout' not in kwargs:
            kwargs['timeout'] = 30
            
        response = self.session.get(url, **kwargs)
        response.raise_for_status()
        return response