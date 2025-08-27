#!/usr/bin/env python3
"""
Smart Incremental Fetch Manager
Prevents duplicate data fetching using intelligent date tracking
Supports different update frequencies and staleness detection
"""

import os
import sys
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Tuple, Set
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from logging_config import setup_logger

load_dotenv()
logger = setup_logger("incremental_fetch_manager")

POSTGRES_URL = os.getenv("POSTGRES_URL")
if not POSTGRES_URL:
    logger.error("POSTGRES_URL not set")
    sys.exit(1)

engine = create_engine(POSTGRES_URL)

class IncrementalFetchManager:
    """Smart manager for incremental data fetching"""
    
    def __init__(self):
        self.today = date.today()
        self.ensure_tracking_tables()
    
    def ensure_tracking_tables(self):
        """Ensure fetch tracking tables exist"""
        try:
            with engine.begin() as conn:
                # Create fetch status tracking table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS fetch_status_tracking (
                        id SERIAL PRIMARY KEY,
                        data_type TEXT NOT NULL,  -- 'prices', 'fundamentals', 'dividends', etc.
                        symbol TEXT NOT NULL,
                        last_fetch_date DATE NOT NULL,
                        last_data_date DATE,  -- Latest data date we have for this symbol
                        fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        records_fetched INTEGER DEFAULT 0,
                        fetch_status TEXT DEFAULT 'SUCCESS',  -- SUCCESS, FAILED, SKIPPED
                        UNIQUE(data_type, symbol)
                    )
                """))
                
                # Create indexes
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_fetch_status_data_type_symbol 
                    ON fetch_status_tracking(data_type, symbol)
                """))
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_fetch_status_last_fetch 
                    ON fetch_status_tracking(data_type, last_fetch_date)
                """))
                
        except Exception as e:
            logger.error(f"Failed to create tracking tables: {e}")
            raise
    
    def get_fetch_status(self, data_type: str, symbol: str) -> Optional[Dict]:
        """Get fetch status for a specific data type and symbol"""
        try:
            with engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT last_fetch_date, last_data_date, records_fetched, 
                           fetch_status, fetched_at
                    FROM fetch_status_tracking
                    WHERE data_type = :data_type AND symbol = :symbol
                """), {'data_type': data_type, 'symbol': symbol})
                
                row = result.fetchone()
                if row:
                    return {
                        'last_fetch_date': row[0],
                        'last_data_date': row[1], 
                        'records_fetched': row[2],
                        'fetch_status': row[3],
                        'fetched_at': row[4]
                    }
                return None
                
        except Exception as e:
            logger.warning(f"Failed to get fetch status for {data_type}/{symbol}: {e}")
            return None
    
    def update_fetch_status(self, data_type: str, symbol: str, last_data_date: Optional[date] = None,
                          records_fetched: int = 0, status: str = 'SUCCESS'):
        """Update fetch status after processing"""
        try:
            with engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO fetch_status_tracking 
                    (data_type, symbol, last_fetch_date, last_data_date, records_fetched, fetch_status)
                    VALUES (:data_type, :symbol, :fetch_date, :data_date, :records, :status)
                    ON CONFLICT (data_type, symbol) DO UPDATE SET
                        last_fetch_date = EXCLUDED.last_fetch_date,
                        last_data_date = COALESCE(EXCLUDED.last_data_date, fetch_status_tracking.last_data_date),
                        records_fetched = EXCLUDED.records_fetched,
                        fetch_status = EXCLUDED.fetch_status,
                        fetched_at = CURRENT_TIMESTAMP
                """), {
                    'data_type': data_type,
                    'symbol': symbol,
                    'fetch_date': self.today,
                    'data_date': last_data_date,
                    'records': records_fetched,
                    'status': status
                })
                
        except Exception as e:
            logger.warning(f"Failed to update fetch status for {data_type}/{symbol}: {e}")
    
    def should_fetch_prices(self, symbol: str, force: bool = False) -> Tuple[bool, str]:
        """Determine if price data should be fetched for symbol"""
        if force:
            return True, "Forced fetch requested"
        
        # Check fetch tracking status
        status = self.get_fetch_status('prices', symbol)
        
        if status is None:
            return True, "No previous fetch record found"
        
        last_fetch = status['last_fetch_date']
        
        # Don't fetch if already fetched today successfully
        if last_fetch >= self.today and status['fetch_status'] == 'SUCCESS':
            # Double-check: Do we actually have current data in the table?
            actual_last_date = self.get_last_data_date_from_table('stock_prices', symbol, 'date')
            if actual_last_date and actual_last_date >= self.today:
                return False, f"Already fetched today with current data ({last_fetch})"
            else:
                return True, f"Fetched today but data is stale ({actual_last_date}), need retry"
        
        # Check if we have very recent data even without today's fetch
        actual_last_date = self.get_last_data_date_from_table('stock_prices', symbol, 'date')
        if actual_last_date and actual_last_date >= self.today:
            # Update tracking to reflect we have current data
            self.update_fetch_status('prices', symbol, actual_last_date, 0, 'CURRENT')
            return False, f"Already have current data ({actual_last_date})"
        
        # Need to fetch
        if actual_last_date:
            return True, f"Last data: {actual_last_date}, fetch: {last_fetch} (needs update)"
        else:
            return True, f"No price data found for {symbol}"
    
    def should_fetch_fundamentals(self, symbol: str, force: bool = False) -> Tuple[bool, str]:
        """Determine if fundamental data should be fetched for symbol"""
        if force:
            return True, "Forced fetch requested"
        
        status = self.get_fetch_status('fundamentals', symbol)
        
        if status is None:
            return True, "No previous fundamentals data found"
        
        last_fetch = status['last_fetch_date']
        
        # Fundamentals are typically updated weekly
        days_since_fetch = (self.today - last_fetch).days
        
        if days_since_fetch < 7:
            return False, f"Recently fetched ({last_fetch}, {days_since_fetch} days ago)"
        
        return True, f"Last fetched: {last_fetch} ({days_since_fetch} days ago, needs update)"
    
    def should_fetch_dividends(self, symbol: str, force: bool = False) -> Tuple[bool, str]:
        """Determine if dividend data should be fetched for symbol"""
        if force:
            return True, "Forced fetch requested"
        
        status = self.get_fetch_status('dividends', symbol)
        
        if status is None:
            return True, "No previous dividend data found"
        
        last_fetch = status['last_fetch_date']
        
        # Dividends are updated quarterly, check less frequently
        days_since_fetch = (self.today - last_fetch).days
        
        if days_since_fetch < 30:  # Monthly check is sufficient
            return False, f"Recently fetched ({last_fetch}, {days_since_fetch} days ago)"
        
        return True, f"Last fetched: {last_fetch} ({days_since_fetch} days ago, needs update)"
    
    def should_fetch_technical_indicators(self, symbol: str, force: bool = False) -> Tuple[bool, str]:
        """Determine if technical indicators should be calculated for symbol"""
        if force:
            return True, "Forced calculation requested"
        
        # Technical indicators depend on fresh price data
        # Check if we have recent price data and if indicators are stale
        
        # First check if price data is fresh
        try:
            with engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT MAX(date) FROM stock_prices WHERE symbol = :symbol
                """), {'symbol': symbol})
                
                latest_price_date = result.scalar()
                if not latest_price_date:
                    return False, "No price data available for technical indicators"
                
                # Check technical indicators freshness
                status = self.get_fetch_status('technical_indicators', symbol)
                
                if status is None:
                    return True, "No previous technical indicators found"
                
                last_fetch = status['last_fetch_date']
                
                # If indicators fetched today, skip
                if last_fetch >= self.today:
                    return False, f"Technical indicators already calculated today ({last_fetch})"
                
                # If price data is newer than indicators, recalculate
                if latest_price_date > status.get('last_data_date', date.min):
                    return True, f"Price data updated since last calculation ({latest_price_date} > {status.get('last_data_date')})"
                
                return True, f"Last calculated: {last_fetch} (stale)"
                
        except Exception as e:
            logger.warning(f"Error checking technical indicators status for {symbol}: {e}")
            return True, "Error checking status, will attempt fetch"
    
    def get_symbols_to_fetch(self, data_type: str, symbols: List[str], force: bool = False) -> List[Tuple[str, str]]:
        """Get list of symbols that need fetching for a data type"""
        should_fetch_funcs = {
            'prices': self.should_fetch_prices,
            'fundamentals': self.should_fetch_fundamentals,
            'dividends': self.should_fetch_dividends,
            'technical_indicators': self.should_fetch_technical_indicators
        }
        
        fetch_func = should_fetch_funcs.get(data_type)
        if not fetch_func:
            logger.error(f"Unknown data type: {data_type}")
            return []
        
        symbols_to_fetch = []
        skipped_count = 0
        
        for symbol in symbols:
            should_fetch, reason = fetch_func(symbol, force)
            if should_fetch:
                symbols_to_fetch.append((symbol, reason))
            else:
                skipped_count += 1
                if skipped_count <= 5:  # Log first few skips
                    logger.debug(f"Skipping {symbol} ({data_type}): {reason}")
        
        logger.info(f"{data_type.title()} fetch analysis: {len(symbols_to_fetch)} symbols to fetch, {skipped_count} skipped")
        
        return symbols_to_fetch
    
    def get_last_data_date_from_table(self, table_name: str, symbol: str, date_column: str = 'date') -> Optional[date]:
        """Get the latest data date for a symbol from a specific table"""
        try:
            with engine.connect() as conn:
                result = conn.execute(text(f"""
                    SELECT MAX({date_column}) FROM {table_name} WHERE symbol = :symbol
                """), {'symbol': symbol})
                
                return result.scalar()
                
        except Exception as e:
            logger.warning(f"Failed to get last date from {table_name} for {symbol}: {e}")
            return None
    
    def mark_symbol_skipped(self, data_type: str, symbol: str, reason: str):
        """Mark a symbol as skipped with reason"""
        self.update_fetch_status(data_type, symbol, status='SKIPPED')
        logger.debug(f"Skipped {symbol} ({data_type}): {reason}")
    
    def get_fetch_summary(self, data_type: str) -> Dict:
        """Get summary of fetch status for a data type"""
        try:
            with engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT 
                        fetch_status,
                        COUNT(*) as count,
                        MAX(last_fetch_date) as most_recent_fetch,
                        MIN(last_fetch_date) as oldest_fetch
                    FROM fetch_status_tracking 
                    WHERE data_type = :data_type
                    GROUP BY fetch_status
                    ORDER BY fetch_status
                """), {'data_type': data_type})
                
                summary = {}
                for row in result:
                    summary[row[0]] = {
                        'count': row[1],
                        'most_recent': row[2],
                        'oldest': row[3]
                    }
                
                return summary
                
        except Exception as e:
            logger.warning(f"Failed to get fetch summary for {data_type}: {e}")
            return {}

# Convenience functions for scripts
def get_manager():
    """Get singleton instance of IncrementalFetchManager"""
    if not hasattr(get_manager, '_instance'):
        get_manager._instance = IncrementalFetchManager()
    return get_manager._instance

def should_fetch_symbol(data_type: str, symbol: str, force: bool = False) -> Tuple[bool, str]:
    """Quick check if a symbol should be fetched"""
    manager = get_manager()
    
    if data_type == 'prices':
        return manager.should_fetch_prices(symbol, force)
    elif data_type == 'fundamentals':
        return manager.should_fetch_fundamentals(symbol, force)
    elif data_type == 'dividends':
        return manager.should_fetch_dividends(symbol, force)
    elif data_type == 'technical_indicators':
        return manager.should_fetch_technical_indicators(symbol, force)
    else:
        return True, f"Unknown data type {data_type}, will fetch"

def update_fetch_tracking(data_type: str, symbol: str, last_data_date: Optional[date] = None,
                         records_fetched: int = 0, status: str = 'SUCCESS'):
    """Update fetch tracking after processing"""
    manager = get_manager()
    manager.update_fetch_status(data_type, symbol, last_data_date, records_fetched, status)

def get_symbols_needing_fetch(data_type: str, all_symbols: List[str], force: bool = False) -> List[Tuple[str, str]]:
    """Get filtered list of symbols that need fetching"""
    manager = get_manager()
    return manager.get_symbols_to_fetch(data_type, all_symbols, force)