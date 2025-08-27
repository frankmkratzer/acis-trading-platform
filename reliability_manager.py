#!/usr/bin/env python3
"""
Reliability Manager - Makes ACIS scripts bulletproof
Handles errors, retries, validation, monitoring, and recovery
"""

import os
import sys
import time
import json
import traceback
from datetime import datetime, date, timedelta
from typing import Optional, Dict, List, Any, Callable, Tuple
from functools import wraps
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from logging_config import setup_logger

load_dotenv()
logger = setup_logger("reliability_manager")

POSTGRES_URL = os.getenv("POSTGRES_URL")
engine = create_engine(POSTGRES_URL) if POSTGRES_URL else None

class ReliabilityManager:
    """Comprehensive reliability and monitoring for ACIS scripts"""
    
    def __init__(self):
        self.today = date.today()
        self.ensure_monitoring_tables()
    
    def ensure_monitoring_tables(self):
        """Create reliability monitoring tables"""
        if not engine:
            return
            
        try:
            with engine.begin() as conn:
                # Error tracking table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS error_tracking (
                        id SERIAL PRIMARY KEY,
                        script_name TEXT NOT NULL,
                        symbol TEXT,
                        error_type TEXT NOT NULL,
                        error_message TEXT,
                        stack_trace TEXT,
                        retry_count INTEGER DEFAULT 0,
                        resolved BOOLEAN DEFAULT FALSE,
                        occurred_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        resolved_at TIMESTAMP
                    )
                """))
                
                # Data quality checks table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS data_quality_checks (
                        id SERIAL PRIMARY KEY,
                        check_type TEXT NOT NULL,
                        table_name TEXT NOT NULL,
                        symbol TEXT,
                        check_date DATE NOT NULL,
                        passed BOOLEAN NOT NULL,
                        details JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                # Script health monitoring
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS script_health_monitor (
                        id SERIAL PRIMARY KEY,
                        script_name TEXT NOT NULL,
                        execution_date DATE NOT NULL,
                        status TEXT NOT NULL, -- SUCCESS, PARTIAL_SUCCESS, FAILED
                        symbols_processed INTEGER DEFAULT 0,
                        symbols_failed INTEGER DEFAULT 0,
                        execution_time_seconds NUMERIC,
                        memory_usage_mb NUMERIC,
                        error_summary JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(script_name, execution_date)
                    )
                """))
                
        except Exception as e:
            logger.error(f"Failed to create monitoring tables: {e}")

def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0, 
                      max_delay: float = 60.0, backoff_factor: float = 2.0,
                      retry_on_exceptions: tuple = (Exception,)):
    """Decorator for retry logic with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retry_on_exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"{func.__name__} failed after {max_retries + 1} attempts: {e}")
                        raise
                    
                    delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                    logger.warning(f"{func.__name__} attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s")
                    time.sleep(delay)
                    
            raise last_exception
        return wrapper
    return decorator

def log_errors(script_name: str):
    """Decorator to log errors to database"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                symbol = kwargs.get('symbol') or (args[0] if args else None)
                log_error(script_name, str(e), symbol=symbol if isinstance(symbol, str) else None)
                raise
        return wrapper
    return decorator

def validate_price_data(df) -> Tuple[bool, List[str]]:
    """Validate price data for quality issues - uses enhanced validator"""
    # Import here to avoid circular dependency
    from data_validator import data_validator
    
    if df.empty:
        return False, ["No data provided"]
    
    # Map column names if needed (handle both schemas)
    if 'date' in df.columns and 'trade_date' not in df.columns:
        df = df.rename(columns={'date': 'trade_date'})
    if 'close_price' in df.columns and 'close' not in df.columns:
        df = df.rename(columns={
            'close_price': 'close',
            'open_price': 'open',
            'high_price': 'high',
            'low_price': 'low'
        })
    
    # Use enhanced validator
    symbol = df['symbol'].iloc[0] if 'symbol' in df.columns and len(df) > 0 else None
    is_valid, issues, metrics = data_validator.validate_price_data(df, symbol)
    
    # Log quality metrics if available
    if metrics:
        logger.info(f"Data quality metrics: {metrics}")
    
    return is_valid, issues

def validate_fundamentals_data(df) -> Tuple[bool, List[str]]:
    """Validate fundamentals data for quality issues"""
    issues = []
    
    if df.empty:
        return False, ["No fundamentals data provided"]
    
    # Check for required columns
    required_cols = ['symbol', 'fiscal_date']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")
    
    # Check for reasonable value ranges
    if 'revenue' in df.columns:
        negative_revenue = df['revenue'] < 0
        if negative_revenue.any():
            issues.append(f"Negative revenue found: {negative_revenue.sum()} records")
    
    # Check for missing fiscal dates
    if 'fiscal_date' in df.columns:
        null_dates = df['fiscal_date'].isnull()
        if null_dates.any():
            issues.append(f"Missing fiscal dates: {null_dates.sum()} records")
    
    return len(issues) == 0, issues

def log_error(script_name: str, error_message: str, symbol: str = None, 
             error_type: str = "GENERAL", retry_count: int = 0):
    """Log error to database for monitoring"""
    if not engine:
        return
        
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO error_tracking 
                (script_name, symbol, error_type, error_message, stack_trace, retry_count)
                VALUES (:script_name, :symbol, :error_type, :error_message, :stack_trace, :retry_count)
            """), {
                'script_name': script_name,
                'symbol': symbol,
                'error_type': error_type,
                'error_message': error_message[:1000],  # Limit length
                'stack_trace': traceback.format_exc()[:5000],
                'retry_count': retry_count
            })
    except Exception as e:
        logger.warning(f"Failed to log error to database: {e}")

def log_data_quality_check(check_type: str, table_name: str, passed: bool, 
                          symbol: str = None, details: dict = None):
    """Log data quality check results"""
    if not engine:
        return
        
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO data_quality_checks 
                (check_type, table_name, symbol, check_date, passed, details)
                VALUES (:check_type, :table_name, :symbol, :check_date, :passed, :details)
            """), {
                'check_type': check_type,
                'table_name': table_name,
                'symbol': symbol,
                'check_date': date.today(),
                'passed': passed,
                'details': json.dumps(details) if details else None
            })
    except Exception as e:
        logger.warning(f"Failed to log data quality check: {e}")

def log_script_health(script_name: str, status: str, symbols_processed: int = 0,
                     symbols_failed: int = 0, execution_time: float = 0,
                     memory_usage: float = 0, error_summary: dict = None):
    """Log script health metrics"""
    if not engine:
        return
        
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO script_health_monitor 
                (script_name, execution_date, status, symbols_processed, symbols_failed, 
                 execution_time_seconds, memory_usage_mb, error_summary)
                VALUES (:script_name, :execution_date, :status, :symbols_processed, :symbols_failed,
                        :execution_time, :memory_usage, :error_summary)
                ON CONFLICT (script_name, execution_date) DO UPDATE SET
                    status = EXCLUDED.status,
                    symbols_processed = EXCLUDED.symbols_processed,
                    symbols_failed = EXCLUDED.symbols_failed,
                    execution_time_seconds = EXCLUDED.execution_time_seconds,
                    memory_usage_mb = EXCLUDED.memory_usage_mb,
                    error_summary = EXCLUDED.error_summary
            """), {
                'script_name': script_name,
                'execution_date': date.today(),
                'status': status,
                'symbols_processed': symbols_processed,
                'symbols_failed': symbols_failed,
                'execution_time': execution_time,
                'memory_usage': memory_usage,
                'error_summary': json.dumps(error_summary) if error_summary else None
            })
    except Exception as e:
        logger.warning(f"Failed to log script health: {e}")

def check_database_connection() -> bool:
    """Verify database connectivity"""
    if not engine:
        return False
        
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False

def check_api_key_validity(api_key: str, api_name: str) -> bool:
    """Check if API key is valid (basic check)"""
    if not api_key or len(api_key) < 10:
        logger.error(f"{api_name} API key appears invalid")
        return False
    return True

def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0
    except Exception:
        return 0.0

def cleanup_old_logs(days_to_keep: int = 30):
    """Clean up old log entries to prevent database bloat"""
    if not engine:
        return
        
    cutoff_date = date.today() - timedelta(days=days_to_keep)
    
    try:
        with engine.begin() as conn:
            # Clean up old error tracking
            result = conn.execute(text("""
                DELETE FROM error_tracking 
                WHERE occurred_at < :cutoff_date AND resolved = TRUE
            """), {'cutoff_date': cutoff_date})
            
            deleted_errors = result.rowcount
            
            # Clean up old data quality checks
            result = conn.execute(text("""
                DELETE FROM data_quality_checks 
                WHERE created_at < :cutoff_date
            """), {'cutoff_date': cutoff_date})
            
            deleted_quality = result.rowcount
            
            # Keep script health data longer (90 days)
            health_cutoff = date.today() - timedelta(days=90)
            result = conn.execute(text("""
                DELETE FROM script_health_monitor 
                WHERE created_at < :cutoff_date
            """), {'cutoff_date': health_cutoff})
            
            deleted_health = result.rowcount
            
            logger.info(f"Cleaned up old logs: {deleted_errors} errors, {deleted_quality} quality checks, {deleted_health} health records")
            
    except Exception as e:
        logger.error(f"Failed to cleanup old logs: {e}")

class CircuitBreaker:
    """Circuit breaker pattern for external API calls"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    def can_execute(self) -> bool:
        """Check if execution is allowed"""
        if self.state == 'CLOSED':
            return True
        elif self.state == 'OPEN':
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = 'HALF_OPEN'
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def record_success(self):
        """Record successful execution"""
        self.failure_count = 0
        self.state = 'CLOSED'
    
    def record_failure(self):
        """Record failed execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

# Global circuit breakers for different APIs
api_circuit_breakers = {
    'alpha_vantage': CircuitBreaker(),
    'fmp': CircuitBreaker()
}

def with_circuit_breaker(api_name: str):
    """Decorator to apply circuit breaker pattern"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            breaker = api_circuit_breakers.get(api_name)
            if not breaker:
                return func(*args, **kwargs)
            
            if not breaker.can_execute():
                raise Exception(f"{api_name} API circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure()
                raise
        return wrapper
    return decorator

# Initialize reliability manager
reliability_manager = ReliabilityManager()