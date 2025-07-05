"""Retry decorators for handling connection errors."""

import logging
import time
from functools import wraps
from typing import Any, Callable

import requests

logger = logging.getLogger(__name__)


def retry_on_connection_error(
    max_retries: int = 3, initial_delay: float = 1.0
) -> Callable:
    """Decorator that retries a function on connection errors with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds

    Returns:
        Decorated function that retries on connection errors

    Example:
        @retry_on_connection_error(max_retries=3, initial_delay=1.0)
        def fetch_data(url):
            return requests.get(url)
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (
                    requests.exceptions.ConnectionError,
                    requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.ReadTimeout,
                    ConnectionResetError,
                ) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Connection error on attempt {attempt + 1}/{max_retries}: {e}. "
                            f"Retrying in {delay}s..."
                        )
                        time.sleep(delay)
                        delay *= 2  # Exponential backoff
                    else:
                        logger.error(f"Failed after {max_retries} attempts: {e}")
                        raise last_exception
            return None

        return wrapper

    return decorator
