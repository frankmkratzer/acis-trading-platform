#!/usr/bin/env python3
"""
Technical Indicators Calculator
Calculates technical indicators locally from price data
Optimized for efficiency with batch processing
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
from tqdm import tqdm

# Load environment
load_dotenv()
POSTGRES_URL = os.getenv("POSTGRES_URL")
if not POSTGRES_URL:
    print("[ERROR] POSTGRES_URL not set")
    sys.exit(1)

engine = create_engine(POSTGRES_URL)

# Configuration
MAX_WORKERS = int(os.getenv("TECH_THREADS", "8"))
BATCH_SIZE = int(os.getenv("TECH_BATCH_SIZE", "500"))

# Logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/fetch_technical_indicators.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("fetch_technical_indicators")

class TechnicalIndicatorsCalculator:
    """Local technical indicators calculation"""
    
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
        """Average Directional Index"""
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift()))
        tr3 = pd.DataFrame(abs(low - close.shift()))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = abs(100 * (minus_dm.rolling(window=period).mean() / atr))
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.rolling(window=period).mean()
        return adx
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On Balance Volume"""
        return (np.sign(close.diff()) * volume).fillna(0).cumsum()

def get_price_data(symbol: str, min_data_points: int = 200) -> pd.DataFrame:
    """Fetch price data for a symbol"""
    query = text("""
        SELECT trade_date as date, open, high, low, 
               close, adjusted_close, volume
        FROM stock_prices
        WHERE symbol = :symbol
        ORDER BY trade_date ASC
    """)
    
    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn, params={"symbol": symbol})
    
    if len(df) < min_data_points:
        logger.warning(f"Insufficient data for {symbol}: {len(df)} rows")
        return pd.DataFrame()
    
    # Use adjusted_close for close price calculations
    df['close'] = df['adjusted_close'].fillna(df['close'])
    df.set_index('date', inplace=True)
    return df

def calculate_indicators(symbol: str) -> pd.DataFrame:
    """Calculate all technical indicators for a symbol"""
    try:
        # Get price data
        df = get_price_data(symbol)
        if df.empty:
            return pd.DataFrame()
        
        calc = TechnicalIndicatorsCalculator()
        
        # Calculate indicators
        df['symbol'] = symbol
        df['rsi_14'] = calc.rsi(df['close'], 14)
        df['sma_20'] = calc.sma(df['close'], 20)
        df['sma_50'] = calc.sma(df['close'], 50)
        df['sma_200'] = calc.sma(df['close'], 200)
        df['ema_12'] = calc.ema(df['close'], 12)
        df['ema_26'] = calc.ema(df['close'], 26)
        
        # MACD
        macd, signal, histogram = calc.macd(df['close'])
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_histogram'] = histogram
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = calc.bollinger_bands(df['close'])
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        
        # Stochastic
        stoch_k, stoch_d = calc.stochastic(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d
        
        # Williams %R
        df['williams_r'] = calc.williams_r(df['high'], df['low'], df['close'])
        
        # ADX
        df['adx'] = calc.adx(df['high'], df['low'], df['close'])
        
        # OBV
        df['obv'] = calc.obv(df['close'], df['volume'])
        
        # Keep only indicator columns
        indicator_cols = [
            'symbol', 'rsi_14', 'sma_20', 'sma_50', 'sma_200', 
            'ema_12', 'ema_26', 'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_middle', 'bb_lower', 'stoch_k', 'stoch_d',
            'williams_r', 'adx', 'obv'
        ]
        
        result_df = df[indicator_cols].copy()
        result_df['created_at'] = datetime.now()
        
        # Reset index to get date column
        result_df.reset_index(inplace=True)
        
        # Remove NaN rows (from beginning due to rolling calculations)
        result_df = result_df.dropna(subset=['sma_200'])  # 200-day SMA needs most data
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error calculating indicators for {symbol}: {e}")
        return pd.DataFrame()

def upsert_indicators(df: pd.DataFrame):
    """Upsert indicators to database"""
    if df.empty:
        return 0
    
    try:
        with engine.begin() as conn:
            # Create temp table
            temp_table = f"temp_indicators_{int(time.time())}"
            df.to_sql(temp_table, conn, if_exists='replace', index=False, method='multi')
            
            # Upsert from temp table
            result = conn.execute(text(f"""
                INSERT INTO technical_indicators (
                    symbol, date, rsi_14, sma_20, sma_50, sma_200,
                    ema_12, ema_26, macd, macd_signal, macd_histogram,
                    bb_upper, bb_middle, bb_lower, stoch_k, stoch_d,
                    williams_r, adx, obv, created_at
                )
                SELECT 
                    symbol, date::date, rsi_14, sma_20, sma_50, sma_200,
                    ema_12, ema_26, macd, macd_signal, macd_histogram,
                    bb_upper, bb_middle, bb_lower, stoch_k, stoch_d,
                    williams_r, adx, obv, created_at
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
                    obv = EXCLUDED.obv,
                    created_at = EXCLUDED.created_at
            """))
            
            # Drop temp table
            conn.execute(text(f"DROP TABLE {temp_table}"))
            
            return len(df)
            
    except Exception as e:
        logger.error(f"Error upserting indicators: {e}")
        return 0

def process_symbol(symbol: str) -> tuple:
    """Process a single symbol"""
    try:
        # Calculate indicators
        df = calculate_indicators(symbol)
        if df.empty:
            return symbol, 0, 'NO_DATA'
        
        # Upsert to database
        records = upsert_indicators(df)
        
        if records > 0:
            return symbol, records, 'SUCCESS'
        else:
            return symbol, 0, 'FAILED'
            
    except Exception as e:
        logger.error(f"Error processing {symbol}: {e}")
        return symbol, 0, 'ERROR'

def process_batch(symbols: list, use_parallel: bool = True):
    """Process a batch of symbols"""
    results = {'SUCCESS': 0, 'NO_DATA': 0, 'FAILED': 0, 'ERROR': 0}
    total_records = 0
    
    with tqdm(total=len(symbols), desc="Calculating indicators", unit="symbol") as pbar:
        if use_parallel:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {executor.submit(process_symbol, s): s for s in symbols}
                
                for future in as_completed(futures):
                    symbol, records, status = future.result()
                    results[status] += 1
                    total_records += records
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'Success': results['SUCCESS'],
                        'No_Data': results['NO_DATA'],
                        'Failed': results['FAILED'],
                        'Records': total_records
                    })
        else:
            for symbol in symbols:
                symbol_result, records, status = process_symbol(symbol)
                results[status] += 1
                total_records += records
                
                pbar.update(1)
                pbar.set_postfix({
                    'Success': results['SUCCESS'],
                    'No_Data': results['NO_DATA'],
                    'Failed': results['FAILED'],
                    'Records': total_records
                })
    
    return results, total_records

def get_symbols_with_prices():
    """Get symbols that have price data"""
    query = text("""
        SELECT DISTINCT symbol 
        FROM stock_prices 
        WHERE symbol IN (
            SELECT symbol FROM symbol_universe 
            WHERE is_etf = FALSE AND country = 'USA'
        )
        ORDER BY symbol
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query)
        return [row[0] for row in result]

def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate technical indicators from price data")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to process")
    parser.add_argument("--limit", type=int, help="Limit number of symbols")
    parser.add_argument("--parallel", action="store_true", help="Use parallel processing")
    args = parser.parse_args()
    
    start_time = time.time()
    print("\n" + "=" * 60)
    print("TECHNICAL INDICATORS CALCULATOR")
    print("=" * 60)
    
    # Get symbols to process
    if args.symbols:
        symbols = args.symbols
        print(f"[INFO] Processing specific symbols: {symbols}")
    else:
        symbols = get_symbols_with_prices()
        if args.limit:
            symbols = symbols[:args.limit]
        print(f"[INFO] Found {len(symbols)} symbols with price data")
    
    if not symbols:
        print("[ERROR] No symbols to process")
        return
    
    # Process in batches
    batch_results = {'SUCCESS': 0, 'NO_DATA': 0, 'FAILED': 0, 'ERROR': 0}
    total_records = 0
    
    for i in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[i:i+BATCH_SIZE]
        print(f"\n[BATCH] Processing batch {i//BATCH_SIZE + 1} ({len(batch)} symbols)")
        
        results, records = process_batch(batch, use_parallel=args.parallel)
        
        # Aggregate results
        for status, count in results.items():
            batch_results[status] += count
        total_records += records
    
    # Summary
    duration = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total symbols: {len(symbols)}")
    print(f"Successful: {batch_results['SUCCESS']}")
    print(f"No data: {batch_results['NO_DATA']}")
    print(f"Failed: {batch_results['FAILED']}")
    print(f"Errors: {batch_results['ERROR']}")
    print(f"Total records: {total_records:,}")
    print(f"Duration: {duration:.1f}s")
    print(f"Rate: {len(symbols)/duration:.2f} symbols/sec")
    
    # Log summary
    logger.info(f"Technical indicators calculation completed: {batch_results}")

if __name__ == "__main__":
    main()