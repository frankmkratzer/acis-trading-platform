#!/usr/bin/env python3
"""
Sector Strength Analysis for Enhanced Funnel System
Analyzes sector performance to enhance stock selection
Adds sector momentum and relative strength scoring
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

class SectorStrengthAnalysis:
    def __init__(self):
        load_dotenv()
        self.engine = create_engine(os.getenv('POSTGRES_URL'))
        
    def calculate_sector_strengths(self, lookback_days=252):
        """Calculate sector strength scores based on recent performance"""
        print("SECTOR STRENGTH ANALYSIS")
        print("=" * 50)
        
        with self.engine.connect() as conn:
            # Get sector performance data
            sector_performance = self._get_sector_performance(conn, lookback_days)
            
            if sector_performance.empty:
                print("No sector performance data available")
                return pd.DataFrame()
            
            # Calculate sector strength scores
            sector_scores = self._calculate_sector_scores(sector_performance)
            
            # Display results
            self._display_sector_rankings(sector_scores)
            
            # Save to database
            self._save_sector_scores(conn, sector_scores)
            
            return sector_scores
    
    def _get_sector_performance(self, conn, lookback_days):
        """Get historical sector performance data"""
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
                WHERE s.trade_date >= CURRENT_DATE - INTERVAL '{lookback_days} days'
                  AND s.adjusted_close > 0
                  AND s.volume > 0
                  AND p.sector IS NOT NULL
                GROUP BY p.sector, s.trade_date
                HAVING COUNT(*) >= 5  -- At least 5 stocks per sector per day
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
            ORDER BY sector, trade_date DESC
        """)
        
        result = pd.read_sql(query, conn)
        print(f"Loaded {len(result)} sector-date records")
        return result
    
    def _calculate_sector_scores(self, df):
        """Calculate comprehensive sector strength scores"""
        sector_scores = []
        
        for sector in df['sector'].unique():
            sector_data = df[df['sector'] == sector].copy()
            
            if len(sector_data) < 63:  # Need at least 3 months of data
                continue
                
            # Get most recent data
            latest = sector_data.iloc[0]
            
            # Calculate momentum scores
            momentum_1d = latest['daily_return'] if pd.notna(latest['daily_return']) else 0
            momentum_5d = latest['return_5d'] if pd.notna(latest['return_5d']) else 0  
            momentum_21d = latest['return_21d'] if pd.notna(latest['return_21d']) else 0
            momentum_63d = latest['return_63d'] if pd.notna(latest['return_63d']) else 0
            
            # Calculate volatility (risk-adjusted returns)
            recent_daily_returns = sector_data['daily_return'].dropna()[:21]  # Last month
            volatility = recent_daily_returns.std() * np.sqrt(252) if len(recent_daily_returns) > 5 else 0.25
            
            # Trend consistency (what % of recent days were positive)
            recent_positive_days = (recent_daily_returns > 0).mean() if len(recent_daily_returns) > 5 else 0.5
            
            # Relative volume (activity level)
            avg_volume = sector_data['avg_volume'].mean()
            recent_volume = sector_data['avg_volume'].iloc[:5].mean()  # Last week
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
            
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
                recent_positive_days * 20 +           # Trend consistency (20%)
                min(volume_ratio, 2.0) * 10          # Volume activity (10%, capped at 2x)
            )
            
            # Normalize to 0-100 scale
            final_score = max(0, min(100, raw_score * 100 + 50))
            
            sector_scores.append({
                'sector': sector,
                'strength_score': final_score,
                'momentum_1d': momentum_1d,
                'momentum_5d': momentum_5d, 
                'momentum_21d': momentum_21d,
                'momentum_63d': momentum_63d,
                'volatility': volatility,
                'trend_consistency': recent_positive_days,
                'volume_ratio': volume_ratio,
                'stock_count': int(latest['stock_count']),
                'as_of_date': datetime.now().date()
            })
        
        return pd.DataFrame(sector_scores).sort_values('strength_score', ascending=False)
    
    def _display_sector_rankings(self, df):
        """Display sector strength rankings"""
        print(f"\nSECTOR STRENGTH RANKINGS (as of {datetime.now().strftime('%Y-%m-%d')}):")
        print("=" * 80)
        print(f"{'Rank':<4} {'Sector':<35} {'Score':<6} {'21d%':<8} {'Vol':<6} {'Trend':<6} {'Stocks':<6}")
        print("-" * 80)
        
        for idx, row in df.iterrows():
            rank = df.index.get_loc(idx) + 1
            print(f"{rank:<4} {row['sector']:<35} {row['strength_score']:<6.1f} "
                  f"{row['momentum_21d']:<8.1%} {row['volatility']:<6.1%} "
                  f"{row['trend_consistency']:<6.1%} {row['stock_count']:<6}")
        
        print("-" * 80)
        print("Score: 0-100 sector strength | 21d%: 21-day momentum | Vol: Annualized volatility")
        print("Trend: % positive days (consistency) | Stocks: Number of stocks in sector")
    
    def _save_sector_scores(self, conn, df):
        """Save sector strength scores to database"""
        # Create table if it doesn't exist
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS sector_strength_scores (
                sector VARCHAR(100),
                as_of_date DATE,
                strength_score NUMERIC(5,2),
                momentum_1d NUMERIC(8,6),
                momentum_5d NUMERIC(8,6), 
                momentum_21d NUMERIC(8,6),
                momentum_63d NUMERIC(8,6),
                volatility NUMERIC(6,4),
                trend_consistency NUMERIC(4,3),
                volume_ratio NUMERIC(6,3),
                stock_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (sector, as_of_date)
            )
        """))
        
        # Clear today's data
        conn.execute(text("DELETE FROM sector_strength_scores WHERE as_of_date = CURRENT_DATE"))
        
        # Insert new data
        for _, row in df.iterrows():
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
            """), row.to_dict())
        
        conn.commit()
        print(f"\nSaved {len(df)} sector strength scores to database")

def main():
    """Calculate and save sector strength analysis"""
    analyzer = SectorStrengthAnalysis()
    
    # Calculate sector strengths (1-year lookback)
    sector_scores = analyzer.calculate_sector_strengths(lookback_days=252)
    
    if not sector_scores.empty:
        print(f"\n" + "=" * 50)
        print("SECTOR STRENGTH ANALYSIS COMPLETE!")
        print("Enhanced funnel system can now use sector filtering")
        print("=" * 50)
        
        # Show top and bottom sectors
        top_sectors = sector_scores.head(3)['sector'].tolist()
        bottom_sectors = sector_scores.tail(3)['sector'].tolist()
        
        print(f"\nStrongest Sectors: {', '.join(top_sectors)}")
        print(f"Weakest Sectors: {', '.join(bottom_sectors)}")
        
        print(f"\nNext: Integrate sector strength into enhanced funnel scoring")
        return True
    else:
        print("No sector data available for analysis")
        return False

if __name__ == "__main__":
    main()