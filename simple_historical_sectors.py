#!/usr/bin/env python3
"""
Simple Historical Sector Strength
Calculate sector strength for key historical dates
Focus on quarterly rebalancing periods for backtesting
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

def create_simple_historical_sectors():
    """Create historical sector strength for quarterly periods"""
    print("SIMPLE HISTORICAL SECTOR STRENGTH")
    print("=" * 50)
    
    load_dotenv()
    engine = create_engine(os.getenv('POSTGRES_URL'))
    
    with engine.connect() as conn:
        # Get quarterly end dates from 2004-2024
        quarterly_dates = []
        for year in range(2004, 2025):
            for quarter in [3, 6, 9, 12]:  # March, June, Sept, Dec
                if quarter == 3:
                    date = f"{year}-03-31"
                elif quarter == 6:
                    date = f"{year}-06-30"
                elif quarter == 9:
                    date = f"{year}-09-30"
                else:
                    date = f"{year}-12-31"
                quarterly_dates.append(date)
        
        print(f"Calculating sector strength for {len(quarterly_dates)} quarterly periods")
        
        # Clear existing historical data except current
        conn.execute(text("DELETE FROM sector_strength_scores WHERE as_of_date < CURRENT_DATE"))
        conn.commit()
        
        successful_calculations = 0
        total_scores_saved = 0
        
        for i, quarter_date in enumerate(quarterly_dates, 1):
            print(f"Processing Q{i}/{len(quarterly_dates)}: {quarter_date}")
            
            try:
                # Calculate sector performance for this quarter
                sector_scores = calculate_quarterly_sector_strength(conn, quarter_date)
                
                if sector_scores and len(sector_scores) > 0:
                    # Save to database
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
                                ON CONFLICT (sector, as_of_date) DO NOTHING
                            """), {
                                **score,
                                'as_of_date': quarter_date
                            })
                        except Exception as e:
                            print(f"    Error saving {score['sector']}: {e}")
                            continue
                    
                    conn.commit()
                    successful_calculations += 1
                    total_scores_saved += len(sector_scores)
                    print(f"    Saved {len(sector_scores)} sector scores")
                else:
                    print(f"    No data available")
                    
            except Exception as e:
                print(f"    Error: {e}")
                continue
        
        print(f"\\nCompleted: {successful_calculations}/{len(quarterly_dates)} quarters")
        print(f"Total scores saved: {total_scores_saved}")
        
        # Analyze results
        if successful_calculations > 10:
            analyze_historical_sector_results(conn)
            return True
        else:
            return False

def calculate_quarterly_sector_strength(conn, quarter_end_date):
    """Calculate sector strength for a specific quarter end"""
    
    # Convert to datetime for calculations
    end_date = datetime.strptime(quarter_end_date, '%Y-%m-%d').date()
    start_date = end_date - timedelta(days=252)  # 1 year lookback
    
    # Get sector performance data
    result = conn.execute(text(f"""
        WITH sector_prices AS (
            SELECT 
                p.sector,
                s.trade_date,
                AVG(s.adjusted_close) as avg_price,
                COUNT(*) as stock_count
            FROM stock_eod_daily s
            JOIN pure_us_stocks p ON s.symbol = p.symbol
            WHERE s.trade_date >= '{start_date}'
              AND s.trade_date <= '{end_date}'
              AND s.adjusted_close > 0
              AND s.volume > 0
              AND p.sector IS NOT NULL
            GROUP BY p.sector, s.trade_date
            HAVING COUNT(*) >= 5  -- At least 5 stocks per sector
        ),
        sector_returns AS (
            SELECT 
                sector,
                trade_date,
                avg_price,
                LAG(avg_price, 1) OVER (PARTITION BY sector ORDER BY trade_date) as prev_day,
                LAG(avg_price, 5) OVER (PARTITION BY sector ORDER BY trade_date) as price_5d,
                LAG(avg_price, 21) OVER (PARTITION BY sector ORDER BY trade_date) as price_21d,
                LAG(avg_price, 63) OVER (PARTITION BY sector ORDER BY trade_date) as price_63d,
                stock_count
            FROM sector_prices
        ),
        latest_data AS (
            SELECT 
                sector,
                -- Get the most recent data for each sector
                MAX(CASE WHEN trade_date = '{end_date}' THEN avg_price END) as latest_price,
                MAX(CASE WHEN trade_date = '{end_date}' THEN prev_day END) as prev_price,
                MAX(CASE WHEN trade_date = '{end_date}' THEN price_5d END) as price_5d_ago,
                MAX(CASE WHEN trade_date = '{end_date}' THEN price_21d END) as price_21d_ago,
                MAX(CASE WHEN trade_date = '{end_date}' THEN price_63d END) as price_63d_ago,
                MAX(CASE WHEN trade_date = '{end_date}' THEN stock_count END) as stock_count
            FROM sector_returns
            GROUP BY sector
        )
        SELECT 
            sector,
            CASE WHEN prev_price > 0 THEN (latest_price - prev_price) / prev_price ELSE 0 END as daily_return,
            CASE WHEN price_5d_ago > 0 THEN (latest_price - price_5d_ago) / price_5d_ago ELSE 0 END as return_5d,
            CASE WHEN price_21d_ago > 0 THEN (latest_price - price_21d_ago) / price_21d_ago ELSE 0 END as return_21d,
            CASE WHEN price_63d_ago > 0 THEN (latest_price - price_63d_ago) / price_63d_ago ELSE 0 END as return_63d,
            stock_count
        FROM latest_data
        WHERE latest_price IS NOT NULL 
          AND stock_count >= 5
        ORDER BY sector
    """))
    
    sector_data = result.fetchall()
    
    if not sector_data:
        return None
    
    # Calculate volatility for each sector (simplified)
    volatility_data = {}
    for sector_info in sector_data:
        sector = sector_info[0]
        
        # Get daily returns for volatility calculation
        vol_result = conn.execute(text(f"""
            WITH sector_daily_returns AS (
                SELECT 
                    s.trade_date,
                    AVG(s.adjusted_close) as avg_price,
                    LAG(AVG(s.adjusted_close), 1) OVER (ORDER BY s.trade_date) as prev_price
                FROM stock_eod_daily s
                JOIN pure_us_stocks p ON s.symbol = p.symbol
                WHERE p.sector = '{sector}'
                  AND s.trade_date >= '{end_date}'::date - INTERVAL '63 days'
                  AND s.trade_date <= '{end_date}'
                  AND s.adjusted_close > 0
                GROUP BY s.trade_date
                HAVING COUNT(*) >= 5
                ORDER BY s.trade_date
            )
            SELECT STDDEV(
                CASE WHEN prev_price > 0 THEN (avg_price - prev_price) / prev_price ELSE NULL END
            ) * SQRT(252) as annualized_vol
            FROM sector_daily_returns
            WHERE prev_price IS NOT NULL
        """))
        
        vol_row = vol_result.fetchone()
        volatility_data[sector] = float(vol_row[0]) if vol_row and vol_row[0] else 0.20
    
    # Calculate sector strength scores
    sector_scores = []
    
    for row in sector_data:
        sector, daily_ret, ret_5d, ret_21d, ret_63d, stock_count = row
        
        # Get volatility
        volatility = volatility_data.get(sector, 0.20)
        
        # Calculate trend consistency (simplified - assume 60% for historical)
        trend_consistency = 0.60
        
        # Calculate composite momentum
        momentum_score = (
            daily_ret * 0.1 +
            ret_5d * 0.2 +
            ret_21d * 0.4 +
            ret_63d * 0.3
        )
        
        # Risk-adjusted performance
        sharpe_like = momentum_score / max(volatility, 0.05)
        
        # Final score (0-100)
        raw_score = (
            sharpe_like * 40 +
            ret_21d * 30 +
            trend_consistency * 20 +
            10  # Base volume score
        )
        
        final_score = max(0, min(100, raw_score * 50 + 50))
        
        sector_scores.append({
            'sector': sector,
            'strength_score': final_score,
            'momentum_1d': daily_ret,
            'momentum_5d': ret_5d,
            'momentum_21d': ret_21d,
            'momentum_63d': ret_63d,
            'volatility': volatility,
            'trend_consistency': trend_consistency,
            'volume_ratio': 1.0,
            'stock_count': int(stock_count)
        })
    
    return sector_scores

def analyze_historical_sector_results(conn):
    """Analyze historical sector strength results"""
    print(f"\\nHISTORICAL SECTOR ANALYSIS:")
    print("=" * 50)
    
    # Overall statistics
    result = conn.execute(text("""
        SELECT 
            COUNT(*) as total_records,
            COUNT(DISTINCT as_of_date) as quarters,
            COUNT(DISTINCT sector) as sectors,
            MIN(as_of_date) as start_date,
            MAX(as_of_date) as end_date
        FROM sector_strength_scores
        WHERE as_of_date < CURRENT_DATE
    """))
    
    stats = result.fetchone()
    print(f"Total records: {stats[0]:,}")
    print(f"Quarters: {stats[1]} (from {stats[3]} to {stats[4]})")
    print(f"Sectors tracked: {stats[2]}")
    
    # Average sector performance over time
    result = conn.execute(text("""
        SELECT 
            sector,
            AVG(strength_score) as avg_strength,
            STDDEV(strength_score) as volatility,
            MAX(strength_score) as max_strength,
            MIN(strength_score) as min_strength,
            COUNT(*) as periods
        FROM sector_strength_scores
        WHERE as_of_date < CURRENT_DATE
        GROUP BY sector
        ORDER BY avg_strength DESC
    """))
    
    print(f"\\nSECTOR PERFORMANCE (20-Year Average):")
    print("Sector                           Avg    Vol    Max    Min    Periods")
    print("-" * 70)
    
    for row in result:
        sector = row[0][:29] if row[0] else 'Unknown'
        avg_str = f"{row[1]:.1f}" if row[1] else "N/A"
        vol_str = f"{row[2]:.1f}" if row[2] else "N/A"
        max_str = f"{row[3]:.1f}" if row[3] else "N/A"
        min_str = f"{row[4]:.1f}" if row[4] else "N/A"
        periods = row[5] if row[5] else 0
        
        print(f"{sector:<30} {avg_str:<6} {vol_str:<6} {max_str:<6} {min_str:<6} {periods}")
    
    # Recent trends
    result = conn.execute(text("""
        SELECT as_of_date, sector, strength_score, momentum_21d
        FROM sector_strength_scores
        WHERE as_of_date >= (
            SELECT MAX(as_of_date) - INTERVAL '2 years' 
            FROM sector_strength_scores
            WHERE as_of_date < CURRENT_DATE
        )
        ORDER BY as_of_date DESC, strength_score DESC
        LIMIT 20
    """))
    
    print(f"\\nRECENT SECTOR LEADERS (Last 2 Years):")
    print("Date        Sector                      Strength  21d Return")
    print("-" * 60)
    for row in result:
        sector = row[1][:24] if row[1] else 'Unknown'
        momentum = f"{row[3]:.1%}" if row[3] else "N/A"
        print(f"{str(row[0]):<10} {sector:<25} {row[2]:<8.1f} {momentum}")

def main():
    """Create simple historical sector analysis"""
    
    success = create_simple_historical_sectors()
    
    if success:
        print(f"\\n" + "=" * 60)
        print("HISTORICAL SECTOR ANALYSIS COMPLETE!")
        print("Quarterly sector strength data available for backtesting")
        print("=" * 60)
        
        print(f"\\nCapabilities Enabled:")
        print(f"+ 20-year quarterly sector analysis")
        print(f"+ Sector-aware portfolio rebalancing")
        print(f"+ Historical sector rotation detection")
        print(f"+ Enhanced backtest realism")
        
    else:
        print("Historical sector calculation incomplete")

if __name__ == "__main__":
    main()