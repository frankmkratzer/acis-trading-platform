"""
Data fetching module for ACIS Trading Platform
Handles all external data acquisition from Alpha Vantage API
"""

from .base.rate_limiter import AlphaVantageRateLimiter

__all__ = [
    'AlphaVantageRateLimiter'
]