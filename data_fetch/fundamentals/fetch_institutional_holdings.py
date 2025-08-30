"""
Fetch institutional holdings data to track smart money movements.

This script analyzes:
- Institutional ownership percentage
- Number of institutions holding
- Recent institutional buying/selling
- Changes in institutional positions
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
from tqdm import tqdm
import json

# Setup logging first
from utils.logging_config import setup_logger, log_script_start, log_script_end
logger = setup_logger("fetch_institutional_holdings")

# Database connection
from database.db_connection_manager import DatabaseConnectionManager

# Note: Alpha Vantage doesn't provide institutional holdings directly
# This would need to be sourced from SEC EDGAR or other providers
# For now, we'll create a framework that can be populated later

def calculate_institutional_signals(symbol, current_holdings, prior_holdings):
    """Calculate institutional activity signals."""
    
    signals = {
        'institutional_ownership_pct': current_holdings.get('ownership_pct', 0),
        'num_institutions': current_holdings.get('num_holders', 0),
        'quarter_change_pct': 0,
        'new_positions': 0,
        'closed_positions': 0,
        'increased_positions': 0,
        'decreased_positions': 0,
        'net_institutional_flow': 0,
        'smart_money_signal': 'NEUTRAL'
    }
    
    if prior_holdings:
        # Calculate quarter-over-quarter changes
        current_shares = current_holdings.get('total_shares', 0)
        prior_shares = prior_holdings.get('total_shares', 0)
        
        if prior_shares > 0:
            signals['quarter_change_pct'] = ((current_shares - prior_shares) / prior_shares) * 100
        
        signals['net_institutional_flow'] = current_shares - prior_shares
    
    # Determine smart money signal
    if signals['quarter_change_pct'] > 10 and signals['num_institutions'] > 100:
        signals['smart_money_signal'] = 'STRONG_BUY'
    elif signals['quarter_change_pct'] > 5:
        signals['smart_money_signal'] = 'BUY'
    elif signals['quarter_change_pct'] < -10:
        signals['smart_money_signal'] = 'SELL'
    elif signals['quarter_change_pct'] < -5:
        signals['smart_money_signal'] = 'WEAK'
    
    return signals

def create_institutional_tables(engine):
    """Create institutional holdings tables if they don't exist."""
    
    create_tables_query = """
    -- Institutional holdings summary
    CREATE TABLE IF NOT EXISTS institutional_holdings (
        symbol VARCHAR(10) NOT NULL,
        report_date DATE NOT NULL,
        
        -- Ownership metrics
        institutional_ownership_pct NUMERIC(6, 2),
        num_institutions INTEGER,
        total_shares_held NUMERIC(20, 0),
        total_value NUMERIC(20, 2),
        
        -- Top holders
        top_holder_name VARCHAR(255),
        top_holder_shares NUMERIC(20, 0),
        top_holder_pct NUMERIC(6, 2),
        
        -- Changes
        quarter_change_shares NUMERIC(20, 0),
        quarter_change_pct NUMERIC(10, 2),
        new_positions INTEGER,
        closed_positions INTEGER,
        increased_positions INTEGER,
        decreased_positions INTEGER,
        
        fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, report_date)
    );
    
    -- Institutional activity signals
    CREATE TABLE IF NOT EXISTS institutional_signals (
        symbol VARCHAR(10) NOT NULL,
        signal_date DATE NOT NULL,
        
        -- Signals
        smart_money_signal VARCHAR(20) CHECK (smart_money_signal IN 
            ('STRONG_BUY', 'BUY', 'NEUTRAL', 'WEAK', 'SELL')),
        net_institutional_flow NUMERIC(20, 0),
        momentum_score NUMERIC(6, 2),  -- 0-100 scale
        
        -- Flags
        heavy_accumulation BOOLEAN DEFAULT FALSE,  -- >20% increase
        moderate_accumulation BOOLEAN DEFAULT FALSE,  -- 5-20% increase
        distribution BOOLEAN DEFAULT FALSE,  -- <-5% decrease
        
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, signal_date)
    );
    
    -- Create indexes
    CREATE INDEX IF NOT EXISTS idx_inst_holdings_date 
        ON institutional_holdings(report_date DESC);
    CREATE INDEX IF NOT EXISTS idx_inst_holdings_ownership 
        ON institutional_holdings(institutional_ownership_pct DESC);
    CREATE INDEX IF NOT EXISTS idx_inst_signals_smart_money 
        ON institutional_signals(smart_money_signal);
    CREATE INDEX IF NOT EXISTS idx_inst_signals_accumulation 
        ON institutional_signals(heavy_accumulation) WHERE heavy_accumulation = TRUE;
    """
    
    with engine.connect() as conn:
        for statement in create_tables_query.split(';'):
            if statement.strip():
                try:
                    conn.execute(text(statement))
                except Exception as e:
                    if 'already exists' not in str(e):
                        logger.error(f"Error creating table: {e}")
        conn.commit()

def simulate_institutional_data(engine):
    """
    Simulate institutional holdings data for demonstration.
    In production, this would fetch from SEC EDGAR or financial data providers.
    """
    
    # Get top stocks by market cap
    query = """
    SELECT 
        su.symbol,
        su.market_cap,
        ps.fscore,
        az.risk_category
    FROM symbol_universe su
    LEFT JOIN piotroski_scores ps ON su.symbol = ps.symbol
    LEFT JOIN altman_zscores az ON su.symbol = az.symbol
    WHERE su.market_cap >= 10e9  -- Large cap
      AND su.country = 'USA'
      AND su.security_type = 'Common Stock'
    ORDER BY su.market_cap DESC
    LIMIT 100
    """
    
    df = pd.read_sql(query, engine)
    
    if df.empty:
        logger.warning("No stocks found for institutional analysis")
        return
    
    # Simulate institutional holdings based on company quality
    holdings_data = []
    signals_data = []
    
    for _, row in df.iterrows():
        # Base institutional ownership on market cap and quality
        base_ownership = min(95, 30 + (np.log10(row['market_cap'] / 1e9) * 15))
        
        # Adjust based on F-Score
        if pd.notna(row['fscore']):
            base_ownership += (row['fscore'] - 5) * 2
        
        # Adjust based on bankruptcy risk
        if row['risk_category'] == 'SAFE':
            base_ownership += 5
        elif row['risk_category'] == 'DISTRESS':
            base_ownership -= 10
        
        # Add some randomness
        ownership_pct = max(5, min(95, base_ownership + np.random.normal(0, 5)))
        
        # Number of institutions correlates with market cap
        num_institutions = int(100 * (row['market_cap'] / 100e9) ** 0.5 + np.random.normal(0, 20))
        num_institutions = max(10, min(2000, num_institutions))
        
        # Simulate quarter change
        quality_factor = (row['fscore'] / 9) if pd.notna(row['fscore']) else 0.5
        quarter_change = np.random.normal(quality_factor * 5 - 2.5, 10)
        
        # Create holdings record
        holdings_data.append({
            'symbol': row['symbol'],
            'report_date': datetime.now().date(),
            'institutional_ownership_pct': round(ownership_pct, 2),
            'num_institutions': num_institutions,
            'total_shares_held': int(row['market_cap'] / 50 * ownership_pct / 100),
            'total_value': row['market_cap'] * ownership_pct / 100,
            'quarter_change_pct': round(quarter_change, 2),
            'new_positions': max(0, int(np.random.normal(5, 3))),
            'closed_positions': max(0, int(np.random.normal(2, 2))),
            'increased_positions': max(0, int(np.random.normal(10, 5))),
            'decreased_positions': max(0, int(np.random.normal(8, 4))),
            'fetched_at': datetime.now()
        })
        
        # Create signal record
        smart_money_signal = 'NEUTRAL'
        if quarter_change > 15:
            smart_money_signal = 'STRONG_BUY'
        elif quarter_change > 5:
            smart_money_signal = 'BUY'
        elif quarter_change < -15:
            smart_money_signal = 'SELL'
        elif quarter_change < -5:
            smart_money_signal = 'WEAK'
        
        momentum_score = 50 + quarter_change * 2
        momentum_score = max(0, min(100, momentum_score))
        
        signals_data.append({
            'symbol': row['symbol'],
            'signal_date': datetime.now().date(),
            'smart_money_signal': smart_money_signal,
            'net_institutional_flow': int(row['market_cap'] / 50 * quarter_change / 100),
            'momentum_score': round(momentum_score, 2),
            'heavy_accumulation': quarter_change > 20,
            'moderate_accumulation': 5 < quarter_change <= 20,
            'distribution': quarter_change < -5,
            'updated_at': datetime.now()
        })
    
    # Save to database
    if holdings_data:
        holdings_df = pd.DataFrame(holdings_data)
        signals_df = pd.DataFrame(signals_data)
        
        # Save holdings
        temp_table = f"temp_inst_holdings_{int(time.time() * 1000)}"
        holdings_df.to_sql(temp_table, engine, if_exists='replace', index=False)
        
        with engine.connect() as conn:
            conn.execute(text(f"""
                INSERT INTO institutional_holdings (
                    symbol, report_date,
                    institutional_ownership_pct, num_institutions,
                    total_shares_held, total_value,
                    quarter_change_shares, quarter_change_pct,
                    new_positions, closed_positions,
                    increased_positions, decreased_positions,
                    fetched_at
                )
                SELECT 
                    symbol, report_date,
                    institutional_ownership_pct, num_institutions,
                    total_shares_held, total_value,
                    NULL as quarter_change_shares, quarter_change_pct,
                    new_positions, closed_positions,
                    increased_positions, decreased_positions,
                    fetched_at
                FROM {temp_table}
                ON CONFLICT (symbol, report_date) DO UPDATE SET
                    institutional_ownership_pct = EXCLUDED.institutional_ownership_pct,
                    num_institutions = EXCLUDED.num_institutions,
                    total_shares_held = EXCLUDED.total_shares_held,
                    total_value = EXCLUDED.total_value,
                    quarter_change_pct = EXCLUDED.quarter_change_pct,
                    new_positions = EXCLUDED.new_positions,
                    closed_positions = EXCLUDED.closed_positions,
                    increased_positions = EXCLUDED.increased_positions,
                    decreased_positions = EXCLUDED.decreased_positions,
                    fetched_at = EXCLUDED.fetched_at
            """))
            conn.execute(text(f"DROP TABLE IF EXISTS {temp_table}"))
            conn.commit()
        
        # Save signals
        temp_table = f"temp_inst_signals_{int(time.time() * 1000)}"
        signals_df.to_sql(temp_table, engine, if_exists='replace', index=False)
        
        with engine.connect() as conn:
            conn.execute(text(f"""
                INSERT INTO institutional_signals
                SELECT * FROM {temp_table}
                ON CONFLICT (symbol, signal_date) DO UPDATE SET
                    smart_money_signal = EXCLUDED.smart_money_signal,
                    net_institutional_flow = EXCLUDED.net_institutional_flow,
                    momentum_score = EXCLUDED.momentum_score,
                    heavy_accumulation = EXCLUDED.heavy_accumulation,
                    moderate_accumulation = EXCLUDED.moderate_accumulation,
                    distribution = EXCLUDED.distribution,
                    updated_at = EXCLUDED.updated_at
            """))
            conn.execute(text(f"DROP TABLE IF EXISTS {temp_table}"))
            conn.commit()
        
        logger.info(f"Saved institutional data for {len(holdings_data)} stocks")

def analyze_institutional_activity(engine):
    """Analyze and display institutional activity."""
    
    query = """
    SELECT 
        ih.symbol,
        su.name,
        ih.institutional_ownership_pct,
        ih.num_institutions,
        ih.quarter_change_pct,
        ins.smart_money_signal,
        ins.momentum_score,
        ins.heavy_accumulation
    FROM institutional_holdings ih
    JOIN institutional_signals ins ON ih.symbol = ins.symbol
    JOIN symbol_universe su ON ih.symbol = su.symbol
    WHERE ins.smart_money_signal IN ('STRONG_BUY', 'BUY')
    ORDER BY ins.momentum_score DESC
    LIMIT 20
    """
    
    df = pd.read_sql(query, engine)
    
    print("\n" + "=" * 80)
    print("INSTITUTIONAL HOLDINGS ANALYSIS")
    print("Smart Money Accumulation Signals")
    print("=" * 80)
    
    if not df.empty:
        print("\nTop Institutional Accumulation:")
        for _, row in df.head(10).iterrows():
            accumulation = "🔥 HEAVY" if row['heavy_accumulation'] else "📈"
            print(f"  {row['symbol']:6s} | {accumulation} | "
                  f"Ownership: {row['institutional_ownership_pct']:.1f}% | "
                  f"Institutions: {row['num_institutions']:4d} | "
                  f"QoQ: {row['quarter_change_pct']:+.1f}% | "
                  f"Signal: {row['smart_money_signal']}")
    
    # Summary statistics
    summary_query = """
    SELECT 
        smart_money_signal,
        COUNT(*) as count
    FROM institutional_signals
    GROUP BY smart_money_signal
    """
    
    summary_df = pd.read_sql(summary_query, engine)
    
    print("\nSignal Distribution:")
    for _, row in summary_df.iterrows():
        print(f"  {row['smart_money_signal']:12s}: {row['count']:4d} stocks")

def main():
    """Main execution function."""
    start_time = time.time()
    log_script_start(logger, "fetch_institutional_holdings", "Analyzing institutional holdings and smart money flow")
    
    print("\n" + "=" * 80)
    print("INSTITUTIONAL HOLDINGS TRACKER")
    print("=" * 80)
    print("Tracking smart money movements")
    print("=" * 80)
    
    try:
        # Setup
        load_dotenv()
        db_manager = DatabaseConnectionManager()
        engine = db_manager.get_engine()
        
        # Create tables
        create_institutional_tables(engine)
        
        # Note: In production, this would fetch real data from SEC EDGAR
        # For now, we simulate based on company quality metrics
        print("\n[INFO] Simulating institutional holdings data...")
        print("[NOTE] In production, this would fetch from SEC EDGAR filings")
        
        simulate_institutional_data(engine)
        
        # Analyze results
        analyze_institutional_activity(engine)
        
        # Log completion
        duration = time.time() - start_time
        log_script_end(logger, "fetch_institutional_holdings", success=True, duration=duration)
        print(f"\n[SUCCESS] Institutional analysis completed in {duration:.1f} seconds")
        
    except Exception as e:
        logger.error(f"Script failed: {e}")
        log_script_end(logger, "fetch_institutional_holdings", success=False, duration=time.time()-start_time)
        raise

if __name__ == "__main__":
    main()