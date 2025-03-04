"""Functions and classes to artificially create errors.

Examples:
- [Done] Rate limiting
- [Not done] Network errors
- [Not done] API errors
- [Not done] Data errors
"""

import time
import functools
import threading
import requests


class RateLimitError(Exception):
    """Exception raised when too many requests are made within a minute."""

    pass


class RateLimiter:
    def __init__(self, max_requests_per_minute: int):
        self.max_requests = max_requests_per_minute
        self.window_seconds = 60
        self.timestamps = []
        self.lock = threading.Lock()

    def allow_request(self):
        with self.lock:
            now = time.time()
            while self.timestamps and now - self.timestamps[0] > self.window_seconds:
                self.timestamps.pop(0)
            if len(self.timestamps) >= self.max_requests:
                raise RateLimitError(
                    f"Rate limit exceeded: {self.max_requests} requests per minute allowed."
                )
            self.timestamps.append(now)


def rate_limited(max_requests_per_minute: int):
    """Decorator to apply rate limiting to a function."""
    limiter = RateLimiter(max_requests_per_minute)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            limiter.allow_request()  # Raises error if too many requests
            return func(*args, **kwargs)

        return wrapper

    return decorator


@rate_limited(max_requests_per_minute=100)
def rl_fetch(url, **kwargs):
    """Fetches a URL using an HTTP GET request."""
    return requests.get(url, **kwargs)


test_url = "https://www.example.com"
for i in range(6):
    try:
        response = rl_fetch(test_url)
    except RateLimitError as e:
        print(f"Error at attempt {i}:", e)
