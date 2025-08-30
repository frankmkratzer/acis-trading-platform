#!/usr/bin/env python3
"""
Centralized Alpha Vantage Rate Limiter
Provides consistent rate limiting across all fetch scripts for 600 calls/min Premium API
"""

import os
import time
import math
import random
import logging
import threading
from dotenv import load_dotenv
from datetime import datetime
from collections import deque

load_dotenv()

# ─── Configuration from .env ─────────────────────────────────────
# Rate limits for Premium 600 calls/min API
MAX_CALLS_PER_MIN = int(os.getenv("AV_MAX_CALLS_PER_MIN", "580"))    # Conservative default for 600/min API
HEADROOM_PCT = float(os.getenv("AV_HEADROOM_PCT", "0.95"))          # Use 95% of capacity
EFFECTIVE_LIMIT = max(1, int(math.floor(MAX_CALLS_PER_MIN * HEADROOM_PCT)))
TOKENS_PER_SEC = EFFECTIVE_LIMIT / 60.0
BUCKET_CAPACITY = int(os.getenv("AV_BUCKET_CAPACITY", "5"))         # Small burst allowance

# Concurrency settings
MAX_WORKERS = int(os.getenv("AV_MAX_WORKERS", "8"))                 # Thread pool size
MAX_PARALLEL = int(os.getenv("AV_MAX_PARALLEL", "8"))              # Max parallel API calls
RETRY_LIMIT = int(os.getenv("AV_RETRY_LIMIT", "3"))

# Adaptive backpressure settings
SOFTLIM_BUMP_SEC = float(os.getenv("AV_SOFTLIM_BUMP_SEC", "2.0"))
SOFTLIM_MAX_SEC = float(os.getenv("AV_SOFTLIM_MAX_SEC", "20.0"))
SOFTLIM_DECAY_SEC = float(os.getenv("AV_SOFTLIM_DECAY_SEC", "0.05"))
SOFTLIM_WINDOW_SEC = float(os.getenv("AV_SOFTLIM_WINDOW_SEC", "10"))
SOFTLIM_HIT_THRESH = int(os.getenv("AV_SOFTLIM_HIT_THRESH", "3"))
AV_LIMP_DURATION_SEC = float(os.getenv("AV_LIMP_DURATION_SEC", "45"))

# Logging
VERBOSE_RATE = os.getenv("AV_VERBOSE_RATE", "0").lower() in ("1", "true", "yes")

logger = logging.getLogger(__name__)

# ─── Token Bucket Rate Limiter ──────────────────────────────────
class TokenBucket:
    """Thread-safe token bucket for rate limiting"""
    
    def __init__(self, rate_per_sec: float, capacity: int):
        self.rate = rate_per_sec
        self.capacity = capacity
        self.tokens = float(capacity)
        self.last = time.monotonic()
        self.lock = threading.Lock()
        
    def acquire(self, tokens: float = 1.0):
        """Block until 'tokens' are available (global & thread-safe)."""
        while True:
            with self.lock:
                now = time.monotonic()
                # Refill tokens based on elapsed time
                elapsed = now - self.last
                if elapsed > 0:
                    self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                    self.last = now
                
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    if VERBOSE_RATE:
                        logger.info(f"Token granted. Remaining: {self.tokens:.2f}")
                    return
                
                # Calculate wait time needed
                need = tokens - self.tokens
                wait = need / self.rate if self.rate > 0 else 0.2
                
            # Add jitter to prevent thundering herd
            jitter = random.uniform(0.005, 0.02)
            sleep_time = max(0.0, wait) + jitter
            if VERBOSE_RATE:
                logger.info(f"Rate limit: sleeping {sleep_time:.3f}s")
            time.sleep(sleep_time)

# ─── Adaptive Backpressure Manager ──────────────────────────────
class BackpressureManager:
    """Manages adaptive delays when hitting soft limits"""
    
    def __init__(self):
        self.current_delay = 0.0
        self.soft_limit_hits = deque()  # Track recent soft limit hits
        self.limp_mode_until = None
        self.lock = threading.Lock()
        
    def record_soft_limit(self):
        """Record that we hit a soft limit"""
        with self.lock:
            now = time.time()
            # Add to recent hits
            self.soft_limit_hits.append(now)
            
            # Remove old hits outside the window
            cutoff = now - SOFTLIM_WINDOW_SEC
            while self.soft_limit_hits and self.soft_limit_hits[0] < cutoff:
                self.soft_limit_hits.popleft()
            
            # Check if we should enter limp mode
            if len(self.soft_limit_hits) >= SOFTLIM_HIT_THRESH:
                self.limp_mode_until = now + AV_LIMP_DURATION_SEC
                logger.warning(f"Entering limp mode for {AV_LIMP_DURATION_SEC}s due to {len(self.soft_limit_hits)} soft limits")
            
            # Increase delay
            self.current_delay = min(self.current_delay + SOFTLIM_BUMP_SEC, SOFTLIM_MAX_SEC)
            
    def record_success(self):
        """Record a successful API call"""
        with self.lock:
            # Decay the delay on success
            self.current_delay = max(0, self.current_delay - SOFTLIM_DECAY_SEC)
            
    def get_delay(self) -> float:
        """Get current backpressure delay"""
        with self.lock:
            # Check if we're in limp mode
            if self.limp_mode_until and time.time() < self.limp_mode_until:
                return max(self.current_delay, 5.0)  # Minimum 5s delay in limp mode
            return self.current_delay
            
    def should_slow_down(self) -> bool:
        """Check if we should slow down due to backpressure"""
        return self.get_delay() > 0

# ─── Global Instances ────────────────────────────────────────────
# Create singleton instances for use across all scripts
rate_limiter = TokenBucket(TOKENS_PER_SEC, BUCKET_CAPACITY)
backpressure_manager = BackpressureManager()
request_semaphore = threading.Semaphore(MAX_PARALLEL)

# ─── Backward Compatibility Wrapper ─────────────────────────────
class AlphaVantageRateLimiter:
    """Backward compatibility wrapper for the centralized rate limiter"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self):
        self.calls_per_minute = EFFECTIVE_LIMIT
        self.total_calls = 0
        self.rate_limit_hits = 0
        self._token_bucket = rate_limiter
        self._backpressure = backpressure_manager
    
    @classmethod
    def get_instance(cls):
        """Singleton pattern for backward compatibility"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded (backward compatibility)"""
        self._token_bucket.acquire(1.0)
        
        # Apply backpressure delay if needed
        delay = self._backpressure.get_delay()
        if delay > 0:
            time.sleep(delay)
        
        # Add small jitter
        time.sleep(random.uniform(0.0, 0.01))
        
        self.total_calls += 1
    
    def report_rate_limit(self):
        """Record rate limit hit (backward compatibility)"""
        self.rate_limit_hits += 1
        self._backpressure.record_soft_limit()
        logger.warning(f"Rate limit hit #{self.rate_limit_hits}, adjusting backpressure")

# ─── Utility Functions ───────────────────────────────────────────
def rate_limited_request(func):
    """Decorator for rate-limited API requests"""
    def wrapper(*args, **kwargs):
        # Acquire semaphore for parallel limit
        with request_semaphore:
            # Acquire rate limit token
            rate_limiter.acquire(1.0)
            
            # Apply backpressure delay if needed
            delay = backpressure_manager.get_delay()
            if delay > 0:
                if VERBOSE_RATE:
                    logger.info(f"Applying backpressure delay: {delay:.2f}s")
                time.sleep(delay)
            
            # Add small jitter
            time.sleep(random.uniform(0.0, 0.01))
            
            # Execute the function
            try:
                result = func(*args, **kwargs)
                backpressure_manager.record_success()
                return result
            except Exception as e:
                # Check if it's a rate limit error
                if "Note" in str(e) or "Information" in str(e) or "rate" in str(e).lower():
                    backpressure_manager.record_soft_limit()
                raise
    
    return wrapper

def handle_alpha_vantage_response(response_json: dict, symbol: str = None) -> bool:
    """
    Check Alpha Vantage response for rate limit messages
    Returns True if response is OK, False if rate limited
    """
    if isinstance(response_json, dict):
        # Check for rate limit messages
        for key in ("Note", "Information", "Error Message"):
            if key in response_json:
                msg = response_json[key]
                logger.warning(f"Alpha Vantage message{f' for {symbol}' if symbol else ''}: {msg}")
                
                # Record soft limit if it's a rate limit message
                if any(word in msg.lower() for word in ["rate", "limit", "slow", "frequency"]):
                    backpressure_manager.record_soft_limit()
                    return False
    return True

def get_rate_limit_config():
    """Get current rate limit configuration for logging/debugging"""
    return {
        "max_calls_per_min": MAX_CALLS_PER_MIN,
        "headroom_pct": HEADROOM_PCT,
        "effective_limit": EFFECTIVE_LIMIT,
        "tokens_per_sec": TOKENS_PER_SEC,
        "bucket_capacity": BUCKET_CAPACITY,
        "max_workers": MAX_WORKERS,
        "max_parallel": MAX_PARALLEL,
        "current_delay": backpressure_manager.get_delay(),
        "in_limp_mode": backpressure_manager.limp_mode_until and time.time() < backpressure_manager.limp_mode_until
    }

def log_rate_limit_status():
    """Log current rate limiting status"""
    config = get_rate_limit_config()
    logger.info(f"Rate Limit Status: {config['effective_limit']} calls/min, "
                f"delay: {config['current_delay']:.1f}s, "
                f"limp mode: {config['in_limp_mode']}")

# Export configuration for use by other modules
__all__ = [
    # Configuration
    'MAX_CALLS_PER_MIN', 'HEADROOM_PCT', 'EFFECTIVE_LIMIT', 'TOKENS_PER_SEC',
    'BUCKET_CAPACITY', 'MAX_WORKERS', 'MAX_PARALLEL', 'RETRY_LIMIT',
    'VERBOSE_RATE',
    
    # Rate limiting
    'rate_limiter', 'backpressure_manager', 'request_semaphore',
    'rate_limited_request', 'handle_alpha_vantage_response',
    'get_rate_limit_config', 'log_rate_limit_status',
    
    # Classes
    'TokenBucket', 'BackpressureManager', 'AlphaVantageRateLimiter'
]