#!/usr/bin/env python3
"""
Historical Sector Strength Analysis
Calculate sector performance for all historical dates
Enables sector-aware backtesting and historical analysis
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import time

class HistoricalSectorStrength:
    def __init__(self):
        load_dotenv()
        self.engine = create_engine(os.getenv('POSTGRES_URL'))
        
    def calculate_historical_sector_strength(self, start_date='2004-01-01', end_date=None, 
                                           frequency='monthly', lookback_days=252):
        """Calculate sector strength for all historical periods"""
        
        if end_date is None:
            end_date = datetime.now().date()
        
        print("HISTORICAL SECTOR STRENGTH CALCULATION")
        print("=" * 60)
        print(f"Period: {start_date} to {end_date}")
        print(f"Frequency: {frequency}")
        print(f"Lookback: {lookback_days} days")
        print()
        
        with self.engine.connect() as conn:
            # Get available trading dates
            analysis_dates = self._get_analysis_dates(conn, start_date, end_date, frequency)
            
            if not analysis_dates:
                print("No suitable analysis dates found")
                return False
            
            print(f"Will calculate sector strength for {len(analysis_dates)} dates")
            
            # Clear existing historical data
            conn.execute(text("DELETE FROM sector_strength_scores WHERE as_of_date != CURRENT_DATE"))
            conn.commit()
            
            total_calculations = 0
            successful_dates = 0
            
            for i, analysis_date in enumerate(analysis_dates, 1):
                print(f"\\rProcessing {i}/{len(analysis_dates)}: {analysis_date}", end='', flush=True)
                
                try:
                    # Calculate sector strength for this date
                    sector_scores = self._calculate_sector_strength_for_date(
                        conn, analysis_date, lookback_days
                    )
                    
                    if sector_scores and len(sector_scores) > 0:
                        # Save to database
                        self._save_historical_sector_scores(conn, sector_scores, analysis_date)
                        successful_dates += 1
                        total_calculations += len(sector_scores)
                    
                except Exception as e:
                    print(f"\\nError processing {analysis_date}: {e}")
                    continue
            
            print(f"\\n\\nCompleted: {successful_dates}/{len(analysis_dates)} dates")
            print(f"Total sector calculations: {total_calculations}")
            
            if successful_dates > 0:
                # Analyze results
                self._analyze_historical_results(conn, start_date, end_date)
                return True
            else:
                return False
    
    def _get_analysis_dates(self, conn, start_date, end_date, frequency):
        """Get dates for sector strength analysis"""
        
        # Get available trading dates with sufficient data
        if frequency == 'monthly':
            interval = "1 month"
            min_count = 15  # At least 15 trading days per month
        elif frequency == 'quarterly':
            interval = "3 months" 
            min_count = 45  # At least 45 trading days per quarter
        elif frequency == 'weekly':
            interval = "1 week"
            min_count = 3   # At least 3 trading days per week
        else:
            interval = "1 month"
            min_count = 15
        
        result = conn.execute(text(f"""
            WITH date_periods AS (
                SELECT 
                    date_trunc('{frequency}', trade_date) as period_start,
                    MAX(trade_date) as period_end,
                    COUNT(DISTINCT trade_date) as trading_days,
                    COUNT(DISTINCT symbol) as symbols
                FROM stock_eod_daily
                WHERE trade_date >= '{start_date}'
                  AND trade_date <= '{end_date}'
                  AND adjusted_close > 0
                  AND volume > 0
                GROUP BY date_trunc('{frequency}', trade_date)
                HAVING COUNT(DISTINCT trade_date) >= {min_count}
                   AND COUNT(DISTINCT symbol) >= 500
            )
            SELECT period_end
            FROM date_periods
            ORDER BY period_end
        """))
        
        return [row[0] for row in result.fetchall()]
    
    def _calculate_sector_strength_for_date(self, conn, analysis_date, lookback_days):
        """Calculate sector strength scores for a specific date"""
        
        # Get sector performance data for this date
        query = text(f"""
            WITH sector_daily_data AS (
                SELECT 
                    p.sector,
                    s.trade_date,
                    AVG(s.adjusted_close) as avg_price,
                    AVG(s.volume) as avg_volume,
                    COUNT(*) as stock_count
                FROM stock_eod_daily s
                JOIN pure_us_stocks p ON s.symbol = p.symbol
                WHERE s.trade_date <= '{analysis_date}'
                  AND s.trade_date >= '{analysis_date}'::date - INTERVAL '{lookback_days} days'
                  AND s.adjusted_close > 0
                  AND s.volume > 0
                  AND p.sector IS NOT NULL
                GROUP BY p.sector, s.trade_date
                HAVING COUNT(*) >= 10  -- At least 10 stocks per sector per day
            ),
            sector_returns AS (
                SELECT 
                    sector,
                    trade_date,
                    avg_price,
                    LAG(avg_price, 1) OVER (PARTITION BY sector ORDER BY trade_date) as prev_price,
                    LAG(avg_price, 5) OVER (PARTITION BY sector ORDER BY trade_date) as price_5d_ago,
                    LAG(avg_price, 21) OVER (PARTITION BY sector ORDER BY trade_date) as price_21d_ago,
                    LAG(avg_price, 63) OVER (PARTITION BY sector ORDER BY trade_date) as price_63d_ago,
                    avg_volume,
                    stock_count
                FROM sector_daily_data
            )
            SELECT 
                sector,
                trade_date,
                avg_price,
                prev_price,
                price_5d_ago,
                price_21d_ago,
                price_63d_ago,
                CASE 
                    WHEN prev_price > 0 THEN (avg_price - prev_price) / prev_price
                    ELSE NULL 
                END as daily_return,
                CASE 
                    WHEN price_5d_ago > 0 THEN (avg_price - price_5d_ago) / price_5d_ago  
                    ELSE NULL
                END as return_5d,
                CASE 
                    WHEN price_21d_ago > 0 THEN (avg_price - price_21d_ago) / price_21d_ago
                    ELSE NULL 
                END as return_21d,
                CASE 
                    WHEN price_63d_ago > 0 THEN (avg_price - price_63d_ago) / price_63d_ago
                    ELSE NULL
                END as return_63d,
                avg_volume,
                stock_count
            FROM sector_returns
            WHERE prev_price IS NOT NULL
              AND trade_date = '{analysis_date}'
            ORDER BY sector
        """)
        
        df = pd.read_sql(query, conn)
        
        if len(df) == 0:
            return None
        
        # Calculate sector strength scores
        sector_scores = []
        
        for _, row in df.iterrows():
            # Calculate momentum scores
            momentum_1d = row['daily_return'] if pd.notna(row['daily_return']) else 0
            momentum_5d = row['return_5d'] if pd.notna(row['return_5d']) else 0  
            momentum_21d = row['return_21d'] if pd.notna(row['return_21d']) else 0
            momentum_63d = row['return_63d'] if pd.notna(row['return_63d']) else 0
            
            # Get historical volatility for this sector and date
            volatility = self._calculate_historical_volatility(conn, row['sector'], analysis_date, lookback_days)
            
            # Calculate trend consistency
            trend_consistency = self._calculate_trend_consistency(conn, row['sector'], analysis_date, 21)
            
            # Volume activity
            volume_ratio = 1.0  # Simplified for historical analysis
            
            # Calculate composite sector strength score
            momentum_score = (
                momentum_1d * 0.1 +      # Recent momentum
                momentum_5d * 0.2 +      # Short-term trend  
                momentum_21d * 0.4 +     # Medium-term trend
                momentum_63d * 0.3       # Longer-term trend
            )
            
            # Risk-adjusted momentum
            sharpe_like = momentum_score / volatility if volatility > 0 else momentum_score
            
            # Final composite score (0-100 scale)
            raw_score = (
                sharpe_like * 40 +                    # Risk-adjusted returns (40%)
                momentum_21d * 30 +                   # 1-month momentum (30%)  
                trend_consistency * 20 +              # Trend consistency (20%)
                min(volume_ratio, 2.0) * 10          # Volume activity (10%)
            )
            
            # Normalize to 0-100 scale
            final_score = max(0, min(100, raw_score * 100 + 50))
            
            sector_scores.append({
                'sector': row['sector'],
                'strength_score': final_score,
                'momentum_1d': momentum_1d,
                'momentum_5d': momentum_5d, 
                'momentum_21d': momentum_21d,
                'momentum_63d': momentum_63d,
                'volatility': volatility,
                'trend_consistency': trend_consistency,
                'volume_ratio': volume_ratio,
                'stock_count': int(row['stock_count']),
                'as_of_date': analysis_date
            })
        
        return sector_scores
    
    def _calculate_historical_volatility(self, conn, sector, analysis_date, lookback_days):
        """Calculate historical volatility for a sector"""
        try:
            result = conn.execute(text(f"""
                WITH sector_returns AS (
                    SELECT 
                        s.trade_date,
                        AVG(s.adjusted_close) as avg_price,
                        LAG(AVG(s.adjusted_close), 1) OVER (ORDER BY s.trade_date) as prev_price
                    FROM stock_eod_daily s
                    JOIN pure_us_stocks p ON s.symbol = p.symbol
                    WHERE p.sector = '{sector}'
                      AND s.trade_date <= '{analysis_date}'
                      AND s.trade_date >= '{analysis_date}'::date - INTERVAL '{min(lookback_days, 63)} days'
                      AND s.adjusted_close > 0
                    GROUP BY s.trade_date
                    HAVING COUNT(*) >= 5
                )
                SELECT STDDEV(
                    CASE WHEN prev_price > 0 THEN (avg_price - prev_price) / prev_price ELSE NULL END
                ) * SQRT(252) as annualized_volatility
                FROM sector_returns
                WHERE prev_price IS NOT NULL
            """))
            
            vol = result.fetchone()
            return float(vol[0]) if vol and vol[0] else 0.20  # Default 20% volatility
            
        except:
            return 0.20  # Default volatility
    
    def _calculate_trend_consistency(self, conn, sector, analysis_date, days):
        """Calculate trend consistency for a sector"""
        try:
            result = conn.execute(text(f"""
                WITH sector_returns AS (
                    SELECT 
                        s.trade_date,
                        AVG(s.adjusted_close) as avg_price,
                        LAG(AVG(s.adjusted_close), 1) OVER (ORDER BY s.trade_date) as prev_price
                    FROM stock_eod_daily s
                    JOIN pure_us_stocks p ON s.symbol = p.symbol
                    WHERE p.sector = '{sector}'
                      AND s.trade_date <= '{analysis_date}'
                      AND s.trade_date >= '{analysis_date}'::date - INTERVAL '{days} days'
                      AND s.adjusted_close > 0
                    GROUP BY s.trade_date
                    HAVING COUNT(*) >= 5
                    ORDER BY s.trade_date DESC
                    LIMIT {days}
                )
                SELECT 
                    COUNT(CASE WHEN avg_price > prev_price THEN 1 END)::FLOAT / 
                    COUNT(CASE WHEN prev_price IS NOT NULL THEN 1 END) as positive_ratio
                FROM sector_returns
                WHERE prev_price IS NOT NULL
            """))
            
            ratio = result.fetchone()
            return float(ratio[0]) if ratio and ratio[0] else 0.50  # Default 50%
            
        except:
            return 0.50  # Default trend consistency
    
    def _save_historical_sector_scores(self, conn, sector_scores, analysis_date):
        """Save historical sector scores to database"""
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
                        momentum_1d = EXCLUDED.momentum_1d,
                        momentum_5d = EXCLUDED.momentum_5d,
                        momentum_21d = EXCLUDED.momentum_21d,
                        momentum_63d = EXCLUDED.momentum_63d,
                        volatility = EXCLUDED.volatility,
                        trend_consistency = EXCLUDED.trend_consistency,
                        volume_ratio = EXCLUDED.volume_ratio,
                        stock_count = EXCLUDED.stock_count,
                        created_at = CURRENT_TIMESTAMP
                """), score)
            except Exception as e:
                print(f"Error saving {score['sector']} for {analysis_date}: {e}")
                continue
        
        conn.commit()
    
    def _analyze_historical_results(self, conn, start_date, end_date):
        """Analyze historical sector strength results"""
        print(f"\\n\\nHISTORICAL SECTOR STRENGTH ANALYSIS")
        print("=" * 60)
        
        # Overall statistics
        result = conn.execute(text(f"""
            SELECT 
                COUNT(*) as total_records,
                COUNT(DISTINCT as_of_date) as unique_dates,
                COUNT(DISTINCT sector) as unique_sectors,
                MIN(as_of_date) as earliest_date,
                MAX(as_of_date) as latest_date
            FROM sector_strength_scores
            WHERE as_of_date BETWEEN '{start_date}' AND '{end_date}'
        """))
        
        stats = result.fetchone()
        print(f"Total Records: {stats[0]:,}")
        print(f"Date Range: {stats[3]} to {stats[4]}")
        print(f"Unique Dates: {stats[1]:,}")
        print(f"Sectors Tracked: {stats[2]}")
        
        # Best performing sectors over time
        result = conn.execute(text(f"""
            WITH sector_rankings AS (
                SELECT 
                    sector,
                    AVG(strength_score) as avg_strength,
                    STDDEV(strength_score) as volatility,
                    COUNT(*) as periods,
                    MAX(strength_score) as max_strength,
                    MIN(strength_score) as min_strength
                FROM sector_strength_scores
                WHERE as_of_date BETWEEN '{start_date}' AND '{end_date}'
                GROUP BY sector
            )
            SELECT *
            FROM sector_rankings
            ORDER BY avg_strength DESC
        """))
        
        print(f"\\nHISTORICAL SECTOR PERFORMANCE (Average Strength):")
        print("Sector                           Avg    Std    Periods  Max    Min")
        print("-" * 70)
        
        for row in result:
            print(f"{row[0]:<30} {row[1]:<6.1f} {row[2]:<6.1f} {row[3]:<7} {row[4]:<6.1f} {row[5]:<6.1f}")
        
        # Sample recent data
        result = conn.execute(text(f"""
            SELECT as_of_date, sector, strength_score
            FROM sector_strength_scores
            WHERE as_of_date >= (
                SELECT MAX(as_of_date) - INTERVAL '90 days' 
                FROM sector_strength_scores
            )
            ORDER BY as_of_date DESC, strength_score DESC
            LIMIT 15
        """))
        
        print(f"\\nRECENT SECTOR STRENGTH (Last 90 Days):")
        print("Date        Sector                      Strength")
        print("-" * 50)
        for row in result:
            print(f"{str(row[0]):<10} {row[1]:<25} {row[2]:.1f}")

def main():
    """Calculate historical sector strength for backtesting"""
    analyzer = HistoricalSectorStrength()
    
    print("HISTORICAL SECTOR STRENGTH CALCULATION")
    print("This enables sector-aware backtesting across all periods")
    print()
    
    # Calculate for 20-year period with monthly frequency
    success = analyzer.calculate_historical_sector_strength(
        start_date='2004-01-01',
        end_date='2024-12-31', 
        frequency='monthly',
        lookback_days=252  # 1-year rolling analysis
    )
    
    if success:
        print(f"\\n" + "=" * 60)
        print("HISTORICAL SECTOR STRENGTH COMPLETE!")
        print("Enhanced backtesting with sector dynamics now available")
        print("=" * 60)
        
        print(f"\\nEnhanced Capabilities:")
        print(f"+ 20-year sector rotation analysis")
        print(f"+ Historical sector momentum detection")
        print(f"+ Period-specific sector strength scoring")
        print(f"+ Enhanced backtest realism with sector dynamics")
        print(f"+ Cross-cycle sector performance validation")
        
    else:
        print("Historical sector analysis incomplete")
        print("Check data availability and date ranges")

if __name__ == "__main__":
    main()