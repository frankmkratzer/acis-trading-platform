#!/usr/bin/env python3
"""
Ultra-Premium Technical Indicators Fetcher
Optimized for 300 calls/min Alpha Vantage Premium API
Uses local calculation for maximum efficiency and comprehensive coverage
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm
from reliability_manager import (
    log_errors, log_script_health, get_memory_usage,
    retry_with_backoff
)

# Load environment
load_dotenv()
POSTGRES_URL = os.getenv("POSTGRES_URL")
if not POSTGRES_URL:
    print("[ERROR] POSTGRES_URL not set")
    sys.exit(1)

engine = create_engine(POSTGRES_URL)

# Ultra-premium settings for 300 calls/min API key
MAX_WORKERS = int(os.getenv("TECH_THREADS", "8"))
BATCH_SIZE = int(os.getenv("TECH_BATCH_SIZE", "500"))
LOOKBACK_DAYS = int(os.getenv("TECH_LOOKBACK_DAYS", "500"))  # ~2 years for longest indicators

logging.basicConfig(
    filename="fetch_technical_indicators.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("fetch_technical_indicators")

class TechnicalIndicatorsCalculator:
    """Ultra-fast local technical indicators calculation"""
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period, min_periods=period).mean()
    
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """MACD (Moving Average Convergence Divergence)"""
        ema_fast = TechnicalIndicatorsCalculator.ema(data, fast)
        ema_slow = TechnicalIndicatorsCalculator.ema(data, slow)
        macd_line = ema_fast - ema_slow
        macd_signal = TechnicalIndicatorsCalculator.ema(macd_line, signal)
        macd_histogram = macd_line - macd_signal
        return macd_line, macd_signal, macd_histogram
    
    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: int = 2) -> tuple:
        """Bollinger Bands"""
        sma = TechnicalIndicatorsCalculator.sma(data, period)
        std = data.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                  k_period: int = 14, d_period: int = 3) -> tuple:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Williams %R"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        return -100 * ((highest_high - close) / (highest_high - lowest_low))
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average Directional Index (simplified version)"""
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        plus_di = 100 * (plus_dm.rolling(period).mean() / true_range.rolling(period).mean())
        minus_di = 100 * (minus_dm.rolling(period).mean() / true_range.rolling(period).mean())
        
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        return adx
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On Balance Volume"""
        obv_values = []
        obv_current = 0
        prev_close = None
        
        for i, (curr_close, vol) in enumerate(zip(close, volume)):
            if prev_close is not None:
                if curr_close > prev_close:
                    obv_current += vol
                elif curr_close < prev_close:
                    obv_current -= vol
            obv_values.append(obv_current)
            prev_close = curr_close
        
        return pd.Series(obv_values, index=close.index)

def get_symbols_to_update(limit=None):
    """Get symbols that need technical indicators"""
    query = """
    SELECT s.symbol 
    FROM symbol_universe s
    WHERE s.is_etf = FALSE
    ORDER BY s.market_cap DESC NULLS LAST, s.symbol
    """
    
    if limit:
        query += f" LIMIT {limit}"
    
    with engine.connect() as conn:
        result = conn.execute(text(query))
        symbols = [row[0] for row in result]
    
    return symbols

def get_price_data_for_symbol(symbol):
    """Get price data for technical indicator calculation"""
    cutoff_date = datetime.now().date() - timedelta(days=LOOKBACK_DAYS)
    
    query = """
    SELECT trade_date, open_price, high_price, low_price, close_price, volume
    FROM stock_prices 
    WHERE symbol = :symbol 
      AND trade_date >= :cutoff_date
      AND close_price IS NOT NULL
    ORDER BY trade_date
    """
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={
            'symbol': symbol, 
            'cutoff_date': cutoff_date
        })
    
    if df.empty:
        return None
    
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df.set_index('trade_date', inplace=True)
    return df

def calculate_all_indicators(symbol, price_data):
    """Calculate all technical indicators for a symbol"""
    if price_data is None or len(price_data) < 200:  # Need enough data for 200-day SMA
        return None
    
    close = price_data['close_price']
    high = price_data['high_price']
    low = price_data['low_price']
    volume = price_data['volume']
    
    calc = TechnicalIndicatorsCalculator()
    
    # Calculate all indicators
    indicators_df = pd.DataFrame(index=price_data.index)
    indicators_df['symbol'] = symbol
    indicators_df['date'] = indicators_df.index.date
    
    # Moving Averages
    indicators_df['sma_20'] = calc.sma(close, 20)
    indicators_df['sma_50'] = calc.sma(close, 50)
    indicators_df['sma_200'] = calc.sma(close, 200)
    indicators_df['ema_12'] = calc.ema(close, 12)
    indicators_df['ema_26'] = calc.ema(close, 26)
    
    # Oscillators
    indicators_df['rsi_14'] = calc.rsi(close, 14)
    
    # MACD
    macd, macd_signal, macd_hist = calc.macd(close)
    indicators_df['macd'] = macd
    indicators_df['macd_signal'] = macd_signal
    indicators_df['macd_histogram'] = macd_hist
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = calc.bollinger_bands(close)
    indicators_df['bb_upper'] = bb_upper
    indicators_df['bb_middle'] = bb_middle
    indicators_df['bb_lower'] = bb_lower
    
    # Stochastic
    stoch_k, stoch_d = calc.stochastic(high, low, close)
    indicators_df['stoch_k'] = stoch_k
    indicators_df['stoch_d'] = stoch_d
    
    # Other indicators
    indicators_df['williams_r'] = calc.williams_r(high, low, close)
    indicators_df['adx'] = calc.adx(high, low, close)
    indicators_df['obv'] = calc.obv(close, volume).astype('Int64')
    
    # Remove rows with insufficient data (typically first 200 days)
    indicators_df = indicators_df.dropna(subset=['sma_200'])
    
    return indicators_df

@log_errors('fetch_technical_indicators')
def process_symbol(symbol):
    """Process technical indicators for a single symbol"""
    try:
        logger.info(f"Processing {symbol}")
        
        # Get price data
        price_data = get_price_data_for_symbol(symbol)
        if price_data is None:
            logger.warning(f"No price data for {symbol}")
            return 0
        
        # Calculate indicators
        indicators_df = calculate_all_indicators(symbol, price_data)
        if indicators_df is None or indicators_df.empty:
            logger.warning(f"Could not calculate indicators for {symbol}")
            return 0
        
        # Store in database with ultra-fast upsert
        records_count = len(indicators_df)
        
        with engine.begin() as conn:
            # Create temporary table
            temp_table = f"temp_tech_indicators_{int(time.time())}"
            indicators_df.to_sql(temp_table, conn, if_exists='replace', index=False, method='multi')
            
            # Ultra-fast upsert
            result = conn.execute(text(f"""
                INSERT INTO technical_indicators (
                    symbol, date, rsi_14, sma_20, sma_50, sma_200, ema_12, ema_26,
                    macd, macd_signal, macd_histogram, bb_upper, bb_middle, bb_lower,
                    stoch_k, stoch_d, williams_r, adx, obv
                )
                SELECT 
                    symbol, date, rsi_14, sma_20, sma_50, sma_200, ema_12, ema_26,
                    macd, macd_signal, macd_histogram, bb_upper, bb_middle, bb_lower,
                    stoch_k, stoch_d, williams_r, adx, obv
                FROM {temp_table}
                ON CONFLICT (symbol, date) DO UPDATE SET
                    rsi_14 = EXCLUDED.rsi_14,
                    sma_20 = EXCLUDED.sma_20,
                    sma_50 = EXCLUDED.sma_50,
                    sma_200 = EXCLUDED.sma_200,
                    ema_12 = EXCLUDED.ema_12,
                    ema_26 = EXCLUDED.ema_26,
                    macd = EXCLUDED.macd,
                    macd_signal = EXCLUDED.macd_signal,
                    macd_histogram = EXCLUDED.macd_histogram,
                    bb_upper = EXCLUDED.bb_upper,
                    bb_middle = EXCLUDED.bb_middle,
                    bb_lower = EXCLUDED.bb_lower,
                    stoch_k = EXCLUDED.stoch_k,
                    stoch_d = EXCLUDED.stoch_d,
                    williams_r = EXCLUDED.williams_r,
                    adx = EXCLUDED.adx,
                    obv = EXCLUDED.obv
            """))
            
            # Drop temp table
            conn.execute(text(f"DROP TABLE {temp_table}"))
        
        logger.info(f"Processed {symbol}: {records_count} records")
        return records_count
        
    except Exception as e:
        logger.error(f"Failed to process {symbol}: {e}")
        return 0

def main():
    print("[TECH INDICATORS] Starting ultra-premium technical indicators calculation...")
    logger.info("Starting technical indicators calculation")
    
    start_time = time.time()
    
    # Get symbols to process
    symbols = get_symbols_to_update()
    print(f"[INFO] Processing technical indicators for {len(symbols)} symbols")
    
    if not symbols:
        print("[ERROR] No symbols found to process")
        return
    
    # Process symbols in parallel batches for maximum efficiency
    total_records = 0
    successful_symbols = 0
    failed_symbols = 0
    
    print(f"[PROCESSING] Using {MAX_WORKERS} threads for maximum performance...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Process all symbols
        future_to_symbol = {executor.submit(process_symbol, symbol): symbol for symbol in symbols}
        
        with tqdm(total=len(symbols), desc="Processing symbols") as pbar:
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    records = future.result()
                    if records > 0:
                        successful_symbols += 1
                        total_records += records
                    else:
                        failed_symbols += 1
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    failed_symbols += 1
                
                pbar.update(1)
                pbar.set_postfix({
                    'Success': successful_symbols,
                    'Failed': failed_symbols,
                    'Records': total_records
                })
    
    elapsed = time.time() - start_time
    memory_mb = get_memory_usage()
    
    # Log health metrics
    status = 'SUCCESS' if failed_symbols == 0 else 'PARTIAL_SUCCESS' if successful_symbols > 0 else 'FAILED'
    
    log_script_health(
        script_name='fetch_technical_indicators',
        status=status,
        symbols_processed=successful_symbols,
        symbols_failed=failed_symbols,
        execution_time=elapsed,
        memory_usage=memory_mb,
        error_summary={'total_records': total_records, 'records_per_second': total_records/elapsed if elapsed > 0 else 0}
    )
    
    print(f"\n[ULTRA-PREMIUM COMPLETE]")
    print(f"[SUCCESS] Symbols processed: {successful_symbols}")
    print(f"[FAILED] Symbols failed: {failed_symbols}")
    print(f"[RECORDS] Total indicator records: {total_records:,}")
    print(f"[PERFORMANCE] Execution time: {elapsed:.2f}s")
    print(f"[MEMORY] Peak usage: {memory_mb:.1f}MB")
    print(f"[EFFICIENCY] Average: {total_records/elapsed:.0f} records/second")
    
    logger.info(f"Technical indicators complete: {successful_symbols} symbols, {total_records} records in {elapsed:.2f}s")

if __name__ == "__main__":
    main()