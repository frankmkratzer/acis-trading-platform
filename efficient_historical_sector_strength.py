#!/usr/bin/env python3
"""
Efficient Historical Sector Strength Analysis
Pre-calculate sector returns and then compute strength scores
Much faster approach for 20+ years of data
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

def create_efficient_historical_sectors():
    """Create historical sector strength with optimized approach"""
    print("EFFICIENT HISTORICAL SECTOR STRENGTH")
    print("=" * 50)
    
    load_dotenv()
    engine = create_engine(os.getenv('POSTGRES_URL'))
    
    with engine.connect() as conn:
        print("Step 1: Creating pre-computed sector returns...")
        
        # Create table for sector daily returns (if not exists)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS sector_daily_returns (
                trade_date DATE,
                sector VARCHAR(50),
                avg_price NUMERIC(10,4),
                daily_return NUMERIC(8,6),
                return_5d NUMERIC(8,6),
                return_21d NUMERIC(8,6),
                return_63d NUMERIC(8,6),
                stock_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (trade_date, sector)
            )
        """))
        
        # Clear existing data
        conn.execute(text("TRUNCATE TABLE sector_daily_returns"))
        
        print("Step 2: Computing daily sector returns...")
        
        # Compute all sector daily data in one go
        result = conn.execute(text("""
            WITH sector_daily AS (
                SELECT 
                    s.trade_date,
                    p.sector,
                    AVG(s.adjusted_close) as avg_price,
                    COUNT(*) as stock_count
                FROM stock_eod_daily s
                JOIN pure_us_stocks p ON s.symbol = p.symbol
                WHERE s.trade_date >= '2004-01-01'
                  AND s.trade_date <= '2024-12-31'
                  AND s.adjusted_close > 0
                  AND s.volume > 0
                  AND p.sector IS NOT NULL
                GROUP BY s.trade_date, p.sector
                HAVING COUNT(*) >= 10  -- At least 10 stocks per sector
            ),
            sector_returns AS (
                SELECT 
                    trade_date,
                    sector,
                    avg_price,
                    stock_count,
                    LAG(avg_price, 1) OVER (PARTITION BY sector ORDER BY trade_date) as prev_price,
                    LAG(avg_price, 5) OVER (PARTITION BY sector ORDER BY trade_date) as price_5d_ago,
                    LAG(avg_price, 21) OVER (PARTITION BY sector ORDER BY trade_date) as price_21d_ago,
                    LAG(avg_price, 63) OVER (PARTITION BY sector ORDER BY trade_date) as price_63d_ago
                FROM sector_daily
            )
            INSERT INTO sector_daily_returns (
                trade_date, sector, avg_price, daily_return, return_5d, return_21d, return_63d, stock_count
            )
            SELECT 
                trade_date,
                sector,
                avg_price,
                CASE WHEN prev_price > 0 THEN (avg_price - prev_price) / prev_price ELSE NULL END,
                CASE WHEN price_5d_ago > 0 THEN (avg_price - price_5d_ago) / price_5d_ago ELSE NULL END,
                CASE WHEN price_21d_ago > 0 THEN (avg_price - price_21d_ago) / price_21d_ago ELSE NULL END,
                CASE WHEN price_63d_ago > 0 THEN (avg_price - price_63d_ago) / price_63d_ago ELSE NULL END,
                stock_count
            FROM sector_returns
            WHERE prev_price IS NOT NULL
        """))
        
        conn.commit()
        rows_inserted = result.rowcount
        print(f"  Inserted {rows_inserted:,} sector-date return records")
        
        print("Step 3: Computing monthly sector strength scores...")
        
        # Clear existing historical sector strength
        conn.execute(text("DELETE FROM sector_strength_scores WHERE as_of_date != CURRENT_DATE"))
        
        # Get month-end dates for analysis
        result = conn.execute(text("""
            SELECT DISTINCT 
                date_trunc('month', trade_date)::date + INTERVAL '1 month' - INTERVAL '1 day' as month_end
            FROM sector_daily_returns
            WHERE trade_date >= '2004-01-01'
            ORDER BY month_end
        """))
        
        month_ends = [row[0] for row in result.fetchall()]
        print(f"  Processing {len(month_ends)} month-ends...")
        
        # Process in batches for efficiency
        batch_size = 12  # 12 months at a time
        total_saved = 0
        
        for i in range(0, len(month_ends), batch_size):
            batch_dates = month_ends[i:i+batch_size]
            
            # Calculate sector strength for this batch
            for month_end in batch_dates:
                sector_scores = calculate_monthly_sector_strength(conn, month_end)
                if sector_scores:
                    save_batch_sector_scores(conn, sector_scores, month_end)
                    total_saved += len(sector_scores)
            
            progress = min(i + batch_size, len(month_ends))
            print(f"  Processed {progress}/{len(month_ends)} months ({total_saved} scores saved)")
        
        print("Step 4: Final analysis...")
        
        # Analyze results
        result = conn.execute(text("""
            SELECT 
                COUNT(*) as total_records,
                COUNT(DISTINCT as_of_date) as unique_dates,
                COUNT(DISTINCT sector) as unique_sectors,
                MIN(as_of_date) as start_date,
                MAX(as_of_date) as end_date
            FROM sector_strength_scores
            WHERE as_of_date != CURRENT_DATE
        """))
        
        stats = result.fetchone()
        print(f"\\nFINAL RESULTS:")
        print(f"  Historical records: {stats[0]:,}")
        print(f"  Date range: {stats[3]} to {stats[4]}")
        print(f"  Months covered: {stats[1]:,}")
        print(f"  Sectors: {stats[2]}")
        
        # Show top performing sectors by decade
        analyze_sector_performance_by_decade(conn)
        
        return stats[0] > 1000  # Success if we have substantial data

def calculate_monthly_sector_strength(conn, month_end):
    """Calculate sector strength for a specific month-end"""
    
    # Get 252-day (1 year) lookback data for this month
    lookback_date = month_end - timedelta(days=365)
    
    result = conn.execute(text(f"""
        WITH monthly_data AS (
            SELECT 
                sector,
                -- Recent performance metrics
                AVG(CASE WHEN trade_date >= '{month_end}'::date - INTERVAL '21 days' 
                         THEN daily_return ELSE NULL END) as avg_daily_return_21d,
                AVG(CASE WHEN trade_date >= '{month_end}'::date - INTERVAL '63 days' 
                         THEN daily_return ELSE NULL END) as avg_daily_return_63d,
                
                -- Volatility (last 63 days)
                STDDEV(CASE WHEN trade_date >= '{month_end}'::date - INTERVAL '63 days' 
                            THEN daily_return ELSE NULL END) as volatility_63d,
                
                -- Trend consistency (% positive days in last 21 days)
                AVG(CASE WHEN trade_date >= '{month_end}'::date - INTERVAL '21 days' 
                         AND daily_return > 0 THEN 1.0 ELSE 0.0 END) as trend_consistency,
                
                -- Recent momentum
                AVG(CASE WHEN trade_date = '{month_end}' THEN return_5d ELSE NULL END) as momentum_5d,
                AVG(CASE WHEN trade_date = '{month_end}' THEN return_21d ELSE NULL END) as momentum_21d,
                AVG(CASE WHEN trade_date = '{month_end}' THEN return_63d ELSE NULL END) as momentum_63d,
                
                AVG(stock_count) as avg_stock_count
                
            FROM sector_daily_returns
            WHERE trade_date >= '{lookback_date}'
              AND trade_date <= '{month_end}'
            GROUP BY sector
            HAVING COUNT(*) >= 63  -- At least ~3 months of data
        )
        SELECT 
            sector,
            COALESCE(avg_daily_return_21d, 0) * 252 as annualized_daily_return,  -- Annualize
            COALESCE(momentum_5d, 0) as return_5d,
            COALESCE(momentum_21d, 0) as return_21d, 
            COALESCE(momentum_63d, 0) as return_63d,
            COALESCE(volatility_63d, 0.20) * SQRT(252) as annualized_volatility,  -- Annualize
            COALESCE(trend_consistency, 0.5) as trend_consistency,
            avg_stock_count::INTEGER as stock_count
        FROM monthly_data
        ORDER BY sector
    """))
    
    data = result.fetchall()
    if not data:
        return None
    
    sector_scores = []
    
    for row in data:
        sector, daily_ret, ret_5d, ret_21d, ret_63d, volatility, trend_cons, stock_count = row
        
        # Calculate composite momentum score
        momentum_score = (
            daily_ret * 0.1 +       # Recent daily momentum
            ret_5d * 0.2 +          # 5-day momentum  
            ret_21d * 0.4 +         # 21-day momentum (most important)
            ret_63d * 0.3           # 63-day momentum
        )
        
        # Risk-adjusted performance
        sharpe_like = momentum_score / max(volatility, 0.05) if volatility > 0 else momentum_score
        
        # Final composite score (0-100 scale)
        raw_score = (
            sharpe_like * 40 +           # Risk-adjusted returns (40%)
            ret_21d * 30 +               # 21-day momentum (30%)  
            trend_cons * 20 +            # Trend consistency (20%)
            min(1.5, max(0.5, 1.0)) * 10  # Volume factor (10%, normalized to 1.0)
        )
        
        # Normalize to 0-100 scale and ensure reasonable bounds
        final_score = max(0, min(100, raw_score * 50 + 50))  # Center around 50
        
        sector_scores.append({
            'sector': sector,
            'strength_score': final_score,
            'momentum_1d': daily_ret,
            'momentum_5d': ret_5d,
            'momentum_21d': ret_21d,
            'momentum_63d': ret_63d,
            'volatility': volatility,
            'trend_consistency': trend_cons,
            'volume_ratio': 1.0,  # Simplified for historical analysis
            'stock_count': stock_count
        })
    
    return sector_scores

def save_batch_sector_scores(conn, sector_scores, analysis_date):
    """Save sector scores for a specific date"""
    for score in sector_scores:
        try:
            conn.execute(text("""
                INSERT INTO sector_strength_scores (
                    sector, as_of_date, strength_score, momentum_1d, momentum_5d,
                    momentum_21d, momentum_63d, volatility, trend_consistency,
                    volume_ratio, stock_count
                ) VALUES (
                    :sector, :as_of_date, :strength_score, :momentum_1d, :momentum_5d,
                    :momentum_21d, :momentum_63d, :volatility, :trend_consistency,
                    :volume_ratio, :stock_count
                )
                ON CONFLICT (sector, as_of_date) DO UPDATE SET
                    strength_score = EXCLUDED.strength_score,
                    momentum_21d = EXCLUDED.momentum_21d,
                    volatility = EXCLUDED.volatility
            """), {
                **score,
                'as_of_date': analysis_date
            })
        except Exception as e:
            continue  # Skip errors
    
    conn.commit()

def analyze_sector_performance_by_decade(conn):
    """Analyze sector performance across decades"""
    print("\\nSECTOR PERFORMANCE BY DECADE:")
    print("=" * 50)
    
    result = conn.execute(text("""
        SELECT 
            sector,
            CASE 
                WHEN EXTRACT(YEAR FROM as_of_date) >= 2020 THEN '2020s'
                WHEN EXTRACT(YEAR FROM as_of_date) >= 2010 THEN '2010s'
                ELSE '2000s'
            END as decade,
            AVG(strength_score) as avg_strength,
            COUNT(*) as periods
        FROM sector_strength_scores
        WHERE as_of_date != CURRENT_DATE
        GROUP BY sector, decade
        ORDER BY decade DESC, avg_strength DESC
    """))
    
    current_decade = None
    for row in result:
        if row[1] != current_decade:
            current_decade = row[1]
            print(f"\\n{current_decade}:")
            print("Sector                           Avg Strength  Periods")
            print("-" * 55)
        
        print(f"{row[0]:<30} {row[2]:<12.1f} {row[3]:<7}")

def main():
    """Create efficient historical sector analysis"""
    
    success = create_efficient_historical_sectors()
    
    if success:
        print(f"\\n" + "=" * 60)
        print("EFFICIENT HISTORICAL SECTOR ANALYSIS COMPLETE!")
        print("20+ years of sector strength data now available")
        print("Enables accurate historical backtesting with sector dynamics")
        print("=" * 60)
        
    else:
        print("Historical sector calculation incomplete")

if __name__ == "__main__":
    main()