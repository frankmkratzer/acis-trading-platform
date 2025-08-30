"""
Technical Breakout Scanner - Identify momentum and breakout patterns.

This scanner identifies:
- 52-week high breakouts
- Volume surge patterns
- Base breakouts (cup & handle, flags, triangles)
- Relative strength vs market
- Moving average crossovers
- Momentum acceleration
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
from tqdm import tqdm
import time

# Setup logging first
from utils.logging_config import setup_logger, log_script_start, log_script_end
logger = setup_logger("calculate_technical_breakouts")

# Database connection
from database.db_connection_manager import DatabaseConnectionManager

def fetch_technical_data(engine, lookback_days=252):
    """Fetch price and volume data for technical analysis."""
    
    query = f"""
    WITH price_data AS (
        SELECT 
            sp.symbol,
            sp.trade_date,
            sp.open,
            sp.high,
            sp.low,
            sp.close,
            sp.volume,
            su.market_cap / 1e9 as market_cap_b,
            su.sector,
            
            -- Calculate moving averages
            AVG(sp.close) OVER (
                PARTITION BY sp.symbol 
                ORDER BY sp.trade_date 
                ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
            ) as sma_20,
            
            AVG(sp.close_price) OVER (
                PARTITION BY sp.symbol 
                ORDER BY sp.trade_date 
                ROWS BETWEEN 49 PRECEDING AND CURRENT ROW
            ) as sma_50,
            
            AVG(sp.close_price) OVER (
                PARTITION BY sp.symbol 
                ORDER BY sp.trade_date 
                ROWS BETWEEN 199 PRECEDING AND CURRENT ROW
            ) as sma_200,
            
            -- Volume averages
            AVG(sp.volume) OVER (
                PARTITION BY sp.symbol 
                ORDER BY sp.trade_date 
                ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
            ) as avg_volume_20,
            
            -- 52-week high/low
            MAX(sp.high) OVER (
                PARTITION BY sp.symbol 
                ORDER BY sp.trade_date 
                ROWS BETWEEN 251 PRECEDING AND CURRENT ROW
            ) as high_52w,
            
            MIN(sp.low) OVER (
                PARTITION BY sp.symbol 
                ORDER BY sp.trade_date 
                ROWS BETWEEN 251 PRECEDING AND CURRENT ROW
            ) as low_52w,
            
            -- Rank by date to get latest
            ROW_NUMBER() OVER (
                PARTITION BY sp.symbol 
                ORDER BY sp.trade_date DESC
            ) as rn
            
        FROM stock_prices sp
        JOIN symbol_universe su ON sp.symbol = su.symbol
        WHERE sp.trade_date >= CURRENT_DATE - INTERVAL '{lookback_days} days'
          AND su.market_cap >= 2e9
          AND su.country = 'USA'
          AND su.security_type = 'Common Stock'
    )
    SELECT *
    FROM price_data
    WHERE trade_date >= CURRENT_DATE - INTERVAL '100 days'
    ORDER BY symbol, trade_date DESC
    """
    
    logger.info("Fetching technical data...")
    df = pd.read_sql(query, engine)
    logger.info(f"Retrieved {len(df)} records for {df['symbol'].nunique()} stocks")
    
    return df

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_breakout_signals(symbol_data):
    """Calculate breakout signals for a single stock."""
    
    # Get latest data
    latest = symbol_data[symbol_data['rn'] == 1].iloc[0]
    
    # Get historical data for pattern detection
    hist_data = symbol_data[symbol_data['rn'] <= 60].copy()  # Last 60 days
    
    if len(hist_data) < 20:
        return None
    
    signals = {
        'symbol': latest['symbol'],
        'signal_date': datetime.now().date(),
        'latest_price': latest['close'],
        'latest_volume': latest['volume'],
        
        # Price position
        'price_vs_52w_high': (latest['close'] / latest['high_52w']) if latest['high_52w'] > 0 else 0,
        'price_vs_52w_low': (latest['close'] / latest['low_52w']) if latest['low_52w'] > 0 else 0,
        
        # Moving average positions
        'above_sma_20': latest['close'] > latest['sma_20'] if latest['sma_20'] else False,
        'above_sma_50': latest['close'] > latest['sma_50'] if latest['sma_50'] else False,
        'above_sma_200': latest['close'] > latest['sma_200'] if latest['sma_200'] else False,
        
        # Volume analysis
        'volume_ratio': latest['volume'] / latest['avg_volume_20'] if latest['avg_volume_20'] > 0 else 1,
        'volume_surge': False,
        
        # Breakout flags
        'new_52w_high': False,
        'near_52w_high': False,
        'base_breakout': False,
        'momentum_surge': False,
        
        # Technical scores
        'breakout_strength': 0,
        'volume_strength': 0,
        'trend_strength': 0,
        'composite_score': 0
    }
    
    # Check for 52-week high breakout
    if signals['price_vs_52w_high'] >= 0.98:  # Within 2% of 52-week high
        signals['near_52w_high'] = True
        if signals['price_vs_52w_high'] >= 1.0:
            signals['new_52w_high'] = True
    
    # Check for volume surge
    if signals['volume_ratio'] > 1.5:  # 50% above average
        signals['volume_surge'] = True
    
    # Detect base breakout (simplified - price breaking above recent consolidation)
    recent_high = hist_data['high'].iloc[1:21].max()  # 20-day high excluding today
    recent_low = hist_data['low'].iloc[1:21].min()   # 20-day low excluding today
    base_range = recent_high - recent_low
    
    if base_range > 0:
        consolidation_ratio = base_range / recent_high
        if consolidation_ratio < 0.15:  # Tight consolidation (< 15% range)
            if latest['close'] > recent_high:
                signals['base_breakout'] = True
    
    # Calculate RSI for momentum
    hist_data_sorted = hist_data.sort_values('trade_date')
    rsi = calculate_rsi(hist_data_sorted['close'], period=14)
    latest_rsi = rsi.iloc[-1] if len(rsi) > 0 else 50
    
    if latest_rsi > 70:
        signals['momentum_surge'] = True
    
    # Calculate breakout strength score (0-100)
    breakout_score = 0
    
    # Price position scoring
    if signals['new_52w_high']:
        breakout_score += 30
    elif signals['near_52w_high']:
        breakout_score += 20
    
    # Volume scoring
    if signals['volume_surge']:
        if signals['volume_ratio'] > 2.0:
            breakout_score += 25
        else:
            breakout_score += 15
    
    # Moving average scoring
    ma_score = 0
    if signals['above_sma_20']:
        ma_score += 5
    if signals['above_sma_50']:
        ma_score += 7
    if signals['above_sma_200']:
        ma_score += 8
    breakout_score += ma_score
    
    # Base breakout bonus
    if signals['base_breakout']:
        breakout_score += 15
    
    # Momentum bonus
    if signals['momentum_surge']:
        breakout_score += 10
    
    signals['breakout_strength'] = min(100, breakout_score)
    
    # Calculate volume strength (0-100)
    volume_score = min(100, signals['volume_ratio'] * 25) if signals['volume_ratio'] > 1 else signals['volume_ratio'] * 50
    signals['volume_strength'] = volume_score
    
    # Calculate trend strength (0-100)
    trend_score = 0
    if signals['above_sma_200']:
        trend_score += 40
    if signals['above_sma_50']:
        trend_score += 30
    if signals['above_sma_20']:
        trend_score += 20
    if latest['sma_20'] > latest['sma_50'] if latest['sma_50'] else False:
        trend_score += 10
    signals['trend_strength'] = min(100, trend_score)
    
    # Composite score
    signals['composite_score'] = (
        signals['breakout_strength'] * 0.4 +
        signals['volume_strength'] * 0.3 +
        signals['trend_strength'] * 0.3
    )
    
    # Determine signal strength
    if signals['composite_score'] >= 80:
        signals['signal_strength'] = 'STRONG'
    elif signals['composite_score'] >= 60:
        signals['signal_strength'] = 'MODERATE'
    elif signals['composite_score'] >= 40:
        signals['signal_strength'] = 'WEAK'
    else:
        signals['signal_strength'] = 'NONE'
    
    return signals

def calculate_all_breakouts(df):
    """Calculate breakout signals for all stocks."""
    
    results = []
    symbols = df['symbol'].unique()
    
    for symbol in tqdm(symbols, desc="Scanning for breakouts"):
        symbol_data = df[df['symbol'] == symbol].copy()
        signals = calculate_breakout_signals(symbol_data)
        
        if signals:
            results.append(signals)
    
    return pd.DataFrame(results)

def save_breakout_signals(engine, df):
    """Save breakout signals to database."""
    
    # Create table if not exists
    create_table_query = """
    CREATE TABLE IF NOT EXISTS technical_breakouts (
        symbol VARCHAR(10) NOT NULL,
        signal_date DATE NOT NULL,
        
        -- Price data
        latest_price NUMERIC(12, 2),
        latest_volume BIGINT,
        price_vs_52w_high NUMERIC(6, 4),
        price_vs_52w_low NUMERIC(6, 4),
        
        -- Moving averages
        above_sma_20 BOOLEAN,
        above_sma_50 BOOLEAN,
        above_sma_200 BOOLEAN,
        
        -- Volume analysis
        volume_ratio NUMERIC(8, 2),
        volume_surge BOOLEAN,
        
        -- Breakout signals
        new_52w_high BOOLEAN,
        near_52w_high BOOLEAN,
        base_breakout BOOLEAN,
        momentum_surge BOOLEAN,
        
        -- Strength scores
        breakout_strength NUMERIC(6, 2),
        volume_strength NUMERIC(6, 2),
        trend_strength NUMERIC(6, 2),
        composite_score NUMERIC(6, 2),
        signal_strength VARCHAR(20),
        
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, signal_date)
    );
    
    -- Create indexes
    CREATE INDEX IF NOT EXISTS idx_breakouts_composite 
        ON technical_breakouts(composite_score DESC);
    CREATE INDEX IF NOT EXISTS idx_breakouts_52w_high 
        ON technical_breakouts(new_52w_high) WHERE new_52w_high = TRUE;
    CREATE INDEX IF NOT EXISTS idx_breakouts_volume_surge 
        ON technical_breakouts(volume_surge) WHERE volume_surge = TRUE;
    CREATE INDEX IF NOT EXISTS idx_breakouts_signal_strength 
        ON technical_breakouts(signal_strength);
    """
    
    with engine.connect() as conn:
        for statement in create_table_query.split(';'):
            if statement.strip():
                try:
                    conn.execute(text(statement))
                except Exception as e:
                    if 'already exists' not in str(e):
                        logger.error(f"Error creating table: {e}")
        conn.commit()
    
    # Save signals
    if df.empty:
        logger.warning("No breakout signals to save")
        return
    
    temp_table = f"temp_breakouts_{int(time.time() * 1000)}"
    
    try:
        df.to_sql(temp_table, engine, if_exists='replace', index=False)
        
        # Build column list
        cols = df.columns.tolist()
        cols_str = ', '.join(cols)
        
        # Upsert query
        with engine.connect() as conn:
            conn.execute(text(f"""
                INSERT INTO technical_breakouts ({cols_str})
                SELECT {cols_str} FROM {temp_table}
                ON CONFLICT (symbol, signal_date) DO UPDATE SET
                    latest_price = EXCLUDED.latest_price,
                    latest_volume = EXCLUDED.latest_volume,
                    price_vs_52w_high = EXCLUDED.price_vs_52w_high,
                    volume_ratio = EXCLUDED.volume_ratio,
                    volume_surge = EXCLUDED.volume_surge,
                    new_52w_high = EXCLUDED.new_52w_high,
                    base_breakout = EXCLUDED.base_breakout,
                    breakout_strength = EXCLUDED.breakout_strength,
                    composite_score = EXCLUDED.composite_score,
                    signal_strength = EXCLUDED.signal_strength,
                    updated_at = CURRENT_TIMESTAMP
            """))
            conn.execute(text(f"DROP TABLE IF EXISTS {temp_table}"))
            conn.commit()
            
        logger.info(f"Saved breakout signals for {len(df)} stocks")
        
    except Exception as e:
        logger.error(f"Error saving breakout signals: {e}")
        with engine.connect() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {temp_table}"))
            conn.rollback()

def analyze_breakout_results(engine):
    """Analyze and display breakout results."""
    
    query = """
    SELECT 
        tb.symbol,
        su.name,
        su.sector,
        tb.latest_price,
        tb.price_vs_52w_high * 100 as pct_from_52w_high,
        tb.volume_ratio,
        tb.new_52w_high,
        tb.volume_surge,
        tb.base_breakout,
        tb.composite_score,
        tb.signal_strength
    FROM technical_breakouts tb
    JOIN symbol_universe su ON tb.symbol = su.symbol
    WHERE tb.signal_date = (SELECT MAX(signal_date) FROM technical_breakouts)
    ORDER BY tb.composite_score DESC
    """
    
    df = pd.read_sql(query, engine)
    
    print("\n" + "=" * 80)
    print("TECHNICAL BREAKOUT SCANNER RESULTS")
    print("=" * 80)
    
    # Top breakout candidates
    print("\n🚀 TOP BREAKOUT CANDIDATES (Composite Score > 70):")
    top_breakouts = df[df['composite_score'] > 70].head(15)
    
    if not top_breakouts.empty:
        for _, row in top_breakouts.iterrows():
            indicators = []
            if row['new_52w_high']:
                indicators.append("52W-HIGH")
            if row['volume_surge']:
                indicators.append(f"VOL {row['volume_ratio']:.1f}x")
            if row['base_breakout']:
                indicators.append("BASE-BREAK")
            
            print(f"  {row['symbol']:6s} | Score: {row['composite_score']:5.1f} | "
                  f"Price: ${row['latest_price']:7.2f} | "
                  f"Signal: {row['signal_strength']:8s} | "
                  f"{', '.join(indicators)}")
    else:
        print("  No strong breakout candidates found")
    
    # New 52-week highs
    print("\n📈 NEW 52-WEEK HIGHS:")
    new_highs = df[df['new_52w_high'] == True].head(10)
    if not new_highs.empty:
        for _, row in new_highs.iterrows():
            print(f"  {row['symbol']:6s} | Volume: {row['volume_ratio']:4.1f}x avg | "
                  f"{row['name'][:30] if row['name'] else 'N/A'}")
    else:
        print("  No new 52-week highs")
    
    # Volume surges
    print("\n📊 HIGHEST VOLUME SURGES:")
    volume_surges = df[df['volume_surge'] == True].nlargest(10, 'volume_ratio')
    if not volume_surges.empty:
        for _, row in volume_surges.iterrows():
            print(f"  {row['symbol']:6s} | Volume: {row['volume_ratio']:5.1f}x average | "
                  f"Score: {row['composite_score']:5.1f}")
    else:
        print("  No significant volume surges")
    
    # Signal distribution
    print("\n📊 SIGNAL DISTRIBUTION:")
    signal_dist = df['signal_strength'].value_counts()
    for strength in ['STRONG', 'MODERATE', 'WEAK', 'NONE']:
        count = signal_dist.get(strength, 0)
        pct = count * 100 / len(df) if len(df) > 0 else 0
        print(f"  {strength:8s}: {count:4d} ({pct:5.1f}%)")
    
    # Sector breakouts
    print("\n🏢 SECTORS WITH MOST BREAKOUTS:")
    sector_breakouts = df[df['signal_strength'].isin(['STRONG', 'MODERATE'])].groupby('sector').size()
    sector_breakouts = sector_breakouts.sort_values(ascending=False).head(5)
    for sector, count in sector_breakouts.items():
        print(f"  {sector:25s}: {count:3d} stocks")

def main():
    """Main execution function."""
    start_time = time.time()
    log_script_start(logger, "calculate_technical_breakouts", "Scanning for technical breakouts")
    
    print("\n" + "=" * 80)
    print("TECHNICAL BREAKOUT SCANNER")
    print("52-Week Highs | Volume Surges | Base Breakouts")
    print("=" * 80)
    
    try:
        # Setup database
        load_dotenv()
        db_manager = DatabaseConnectionManager()
        engine = db_manager.get_engine()
        
        # Fetch technical data
        df = fetch_technical_data(engine, lookback_days=252)
        
        if df.empty:
            logger.warning("No technical data found")
            return
        
        # Calculate breakout signals
        logger.info("Calculating breakout signals...")
        signals_df = calculate_all_breakouts(df)
        
        print(f"\n[INFO] Analyzed {len(signals_df)} stocks for breakout patterns")
        
        # Save to database
        save_breakout_signals(engine, signals_df)
        
        # Analyze results
        analyze_breakout_results(engine)
        
        # Trading insights
        print("\n" + "=" * 80)
        print("TRADING INSIGHTS")
        print("=" * 80)
        print("\n💡 BREAKOUT TRADING RULES:")
        print("  1. Focus on stocks with Composite Score > 70")
        print("  2. Require volume surge (>1.5x average) for confirmation")
        print("  3. Best setups: New 52W high + Volume + Above all MAs")
        print("  4. Enter on breakout, stop loss below breakout level")
        print("  5. Combine with fundamental scores for best results")
        
        # Log completion
        duration = time.time() - start_time
        log_script_end(logger, "calculate_technical_breakouts", success=True, duration=duration)
        print(f"\n[SUCCESS] Breakout scan completed in {duration:.1f} seconds")
        
    except Exception as e:
        logger.error(f"Script failed: {e}")
        log_script_end(logger, "calculate_technical_breakouts", success=False, duration=time.time()-start_time)
        raise

if __name__ == "__main__":
    main()