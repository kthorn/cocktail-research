"""Web scraping utilities for polite crawling."""

from .polite import PoliteSession, check_robots_allowed
from .retry import retry_on_connection_error

__all__ = ["PoliteSession", "check_robots_allowed", "retry_on_connection_error"]
