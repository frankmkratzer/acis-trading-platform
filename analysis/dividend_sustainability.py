#!/usr/bin/env python3
"""
Dividend Sustainability Analyzer
Core metric for ACIS dividend growth strategy

Analyzes:
1. Consecutive years of dividend payments
2. Consecutive years of dividend increases
3. Payout ratio sustainability (earnings and FCF based)
4. Dividend safety scores
5. Integration with Excess Cash Flow strength

This module implements the 5th pillar of our investment philosophy:
sustainable dividend growth from companies with long payment histories.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import text
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.logging_config import setup_logger
from database.db_connection_manager import DatabaseConnectionManager

logger = setup_logger("dividend_sustainability")
db_manager = DatabaseConnectionManager()
engine = db_manager.get_engine()


class DividendSustainabilityAnalyzer:
    """Analyze dividend sustainability and growth potential"""
    
    def __init__(self):
        self.engine = engine
        
    def calculate_dividend_streak(self, symbol):
        """
        Calculate consecutive years of dividend payments and increases
        
        Returns:
            dict: Contains payment streak, increase streak, and growth metrics
        """
        
        query = text("""
            WITH yearly_dividends AS (
                SELECT 
                    symbol,
                    EXTRACT(YEAR FROM ex_date) as year,
                    SUM(dividend) as total_dividend,
                    COUNT(*) as payment_count
                FROM dividend_history
                WHERE symbol = :symbol
                    AND ex_date >= CURRENT_DATE - INTERVAL '30 years'
                GROUP BY symbol, EXTRACT(YEAR FROM ex_date)
                ORDER BY year DESC
            ),
            dividend_changes AS (
                SELECT 
                    year,
                    total_dividend,
                    payment_count,
                    LAG(total_dividend) OVER (ORDER BY year) as prev_dividend,
                    total_dividend - LAG(total_dividend) OVER (ORDER BY year) as dividend_change,
                    (total_dividend - LAG(total_dividend) OVER (ORDER BY year)) / 
                        NULLIF(LAG(total_dividend) OVER (ORDER BY year), 0) * 100 as growth_rate
                FROM yearly_dividends
            )
            SELECT 
                year,
                total_dividend,
                payment_count,
                dividend_change,
                growth_rate,
                CASE 
                    WHEN dividend_change > 0 THEN 1
                    WHEN dividend_change = 0 THEN 0
                    ELSE -1
                END as direction
            FROM dividend_changes
            ORDER BY year DESC
        """)
        
        with self.engine.connect() as conn:
            result = conn.execute(query, {'symbol': symbol})
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
        
        if df.empty:
            return {
                'symbol': symbol,
                'payment_streak_years': 0,
                'increase_streak_years': 0,
                'total_years_paying': 0,
                'avg_growth_rate_5y': None,
                'avg_growth_rate_10y': None,
                'volatility_5y': None
            }
        
        # Calculate payment streak (consecutive years with dividends)
        payment_streak = 0
        for _, row in df.iterrows():
            if pd.notna(row['total_dividend']) and row['total_dividend'] > 0:
                payment_streak += 1
            else:
                break
        
        # Calculate increase streak (consecutive years with increases)
        increase_streak = 0
        for _, row in df.iterrows():
            if row['direction'] == 1:  # Dividend increased
                increase_streak += 1
            else:
                break
        
        # Calculate average growth rates (convert Decimal to float)
        avg_growth_5y = float(df.head(5)['growth_rate'].mean()) if len(df) >= 5 else None
        avg_growth_10y = float(df.head(10)['growth_rate'].mean()) if len(df) >= 10 else None
        # Convert to float array before calculating std to avoid Decimal type issues
        volatility_5y = df.head(5)['growth_rate'].astype(float).std() if len(df) >= 5 else None
        
        return {
            'symbol': symbol,
            'payment_streak_years': payment_streak,
            'increase_streak_years': increase_streak,
            'total_years_paying': len(df),
            'avg_growth_rate_5y': float(avg_growth_5y) if avg_growth_5y else None,
            'avg_growth_rate_10y': float(avg_growth_10y) if avg_growth_10y else None,
            'volatility_5y': float(volatility_5y) if volatility_5y else None
        }
    
    def calculate_payout_ratios(self, symbol):
        """
        Calculate dividend payout ratios based on earnings and free cash flow
        
        Returns:
            dict: Contains payout ratios and sustainability metrics
        """
        
        query = text("""
            WITH latest_fundamentals AS (
                SELECT 
                    f.symbol,
                    f.fiscal_date_ending,
                    f.diluted_eps_ttm,
                    f.free_cash_flow,
                    f.shares_outstanding,
                    -- Free cash flow per share
                    CASE 
                        WHEN f.shares_outstanding > 0 THEN
                            f.free_cash_flow::NUMERIC / f.shares_outstanding
                        ELSE NULL
                    END as fcf_per_share
                FROM fundamentals f
                WHERE f.symbol = :symbol
                    AND f.period_type = 'annual'
                    AND f.fiscal_date_ending >= CURRENT_DATE - INTERVAL '1 year'
                ORDER BY f.fiscal_date_ending DESC
                LIMIT 1
            ),
            latest_dividend AS (
                SELECT 
                    symbol,
                    EXTRACT(YEAR FROM ex_date) as year,
                    SUM(dividend) as annual_dividend
                FROM dividend_history
                WHERE symbol = :symbol
                    AND ex_date >= CURRENT_DATE - INTERVAL '1 year'
                GROUP BY symbol, EXTRACT(YEAR FROM ex_date)
                ORDER BY year DESC
                LIMIT 1
            )
            SELECT 
                lf.symbol,
                lf.diluted_eps_ttm,
                lf.fcf_per_share,
                ld.annual_dividend,
                -- Payout ratio based on earnings
                CASE 
                    WHEN lf.diluted_eps_ttm > 0 AND ld.annual_dividend > 0 THEN
                        (ld.annual_dividend / lf.diluted_eps_ttm) * 100
                    ELSE NULL
                END as payout_ratio_earnings,
                -- Payout ratio based on free cash flow
                CASE 
                    WHEN lf.fcf_per_share > 0 AND ld.annual_dividend > 0 THEN
                        (ld.annual_dividend / lf.fcf_per_share) * 100
                    ELSE NULL
                END as payout_ratio_fcf
            FROM latest_fundamentals lf
            LEFT JOIN latest_dividend ld ON lf.symbol = ld.symbol
        """)
        
        with self.engine.connect() as conn:
            result = conn.execute(query, {'symbol': symbol})
            row = result.fetchone()
        
        if not row:
            return {
                'symbol': symbol,
                'payout_ratio_earnings': None,
                'payout_ratio_fcf': None,
                'sustainability_score': None,
                'safety_rating': 'Unknown'
            }
        
        payout_earnings = row['payout_ratio_earnings']
        payout_fcf = row['payout_ratio_fcf']
        
        # Calculate sustainability score (0-100, higher is better)
        sustainability_score = self._calculate_sustainability_score(
            payout_earnings, payout_fcf
        )
        
        # Determine safety rating
        safety_rating = self._get_safety_rating(sustainability_score)
        
        return {
            'symbol': symbol,
            'payout_ratio_earnings': float(payout_earnings) if payout_earnings else None,
            'payout_ratio_fcf': float(payout_fcf) if payout_fcf else None,
            'sustainability_score': sustainability_score,
            'safety_rating': safety_rating
        }
    
    def _calculate_sustainability_score(self, payout_earnings, payout_fcf):
        """
        Calculate dividend sustainability score (0-100)
        
        Scoring criteria:
        - Payout ratio < 40%: Excellent (90-100)
        - Payout ratio 40-60%: Good (70-90)
        - Payout ratio 60-80%: Fair (50-70)
        - Payout ratio 80-100%: Poor (30-50)
        - Payout ratio > 100%: Unsustainable (0-30)
        """
        
        if payout_earnings is None and payout_fcf is None:
            return None
        
        # Use FCF payout ratio as primary, earnings as secondary
        payout = payout_fcf if payout_fcf is not None else payout_earnings
        
        if payout < 0:  # Negative earnings or FCF
            return 0
        elif payout <= 40:
            return 90 + (40 - payout) / 4  # 90-100
        elif payout <= 60:
            return 70 + (60 - payout)  # 70-90
        elif payout <= 80:
            return 50 + (80 - payout)  # 50-70
        elif payout <= 100:
            return 30 + (100 - payout) * 0.5  # 30-50
        else:
            return max(0, 30 - (payout - 100) * 0.3)  # 0-30
    
    def _get_safety_rating(self, score):
        """
        Convert sustainability score to safety rating
        """
        if score is None:
            return 'Unknown'
        elif score >= 80:
            return 'Very Safe'
        elif score >= 60:
            return 'Safe'
        elif score >= 40:
            return 'Moderate'
        elif score >= 20:
            return 'Risky'
        else:
            return 'Unsustainable'
    
    def integrate_with_excess_cash_flow(self, symbol):
        """
        Combine dividend metrics with Excess Cash Flow strength
        
        Returns:
            dict: Comprehensive dividend quality assessment
        """
        
        # Get dividend streak data
        streak_data = self.calculate_dividend_streak(symbol)
        
        # Get payout ratios
        payout_data = self.calculate_payout_ratios(symbol)
        
        # Get Excess Cash Flow metrics
        query = text("""
            SELECT 
                excess_cash_flow_pct,
                quality_rating,
                trend_5y,
                avg_excess_cf_5y
            FROM excess_cash_flow_metrics
            WHERE symbol = :symbol
        """)
        
        with self.engine.connect() as conn:
            result = conn.execute(query, {'symbol': symbol})
            ecf_row = result.fetchone()
        
        # Calculate combined dividend quality score
        quality_score = self._calculate_dividend_quality_score(
            streak_data, payout_data, ecf_row
        )
        
        return {
            'symbol': symbol,
            # Dividend history
            'payment_streak_years': streak_data['payment_streak_years'],
            'increase_streak_years': streak_data['increase_streak_years'],
            'avg_growth_rate_5y': streak_data['avg_growth_rate_5y'],
            # Sustainability
            'payout_ratio_earnings': payout_data['payout_ratio_earnings'],
            'payout_ratio_fcf': payout_data['payout_ratio_fcf'],
            'sustainability_score': payout_data['sustainability_score'],
            'safety_rating': payout_data['safety_rating'],
            # Cash flow strength
            'excess_cash_flow_pct': float(ecf_row['excess_cash_flow_pct']) if ecf_row else None,
            'cash_flow_quality': ecf_row['quality_rating'] if ecf_row else None,
            # Combined assessment
            'dividend_quality_score': quality_score,
            'dividend_quality_rating': self._get_quality_rating(quality_score)
        }
    
    def _calculate_dividend_quality_score(self, streak_data, payout_data, ecf_row):
        """
        Calculate comprehensive dividend quality score (0-100)
        
        Weights:
        - Payment streak: 25%
        - Increase streak: 20%
        - Sustainability: 30%
        - Excess cash flow: 25%
        """
        
        score = 0
        weights_used = 0
        
        # Payment streak score (25%)
        if streak_data['payment_streak_years'] > 0:
            streak_score = min(100, streak_data['payment_streak_years'] * 4)  # 25 years = 100
            score += streak_score * 0.25
            weights_used += 0.25
        
        # Increase streak score (20%)
        if streak_data['increase_streak_years'] > 0:
            increase_score = min(100, streak_data['increase_streak_years'] * 5)  # 20 years = 100
            score += increase_score * 0.20
            weights_used += 0.20
        
        # Sustainability score (30%)
        if payout_data['sustainability_score'] is not None:
            score += payout_data['sustainability_score'] * 0.30
            weights_used += 0.30
        
        # Excess cash flow score (25%)
        if ecf_row and ecf_row['excess_cash_flow_pct'] is not None:
            ecf_score = min(100, max(0, float(ecf_row['excess_cash_flow_pct'])))
            score += ecf_score * 0.25
            weights_used += 0.25
        
        # Normalize if not all components available
        if weights_used > 0:
            return score / weights_used * 100
        else:
            return None
    
    def _get_quality_rating(self, score):
        """
        Convert quality score to rating
        """
        if score is None:
            return 'Unknown'
        elif score >= 80:
            return 'Dividend Aristocrat'
        elif score >= 60:
            return 'High Quality'
        elif score >= 40:
            return 'Good Quality'
        elif score >= 20:
            return 'Fair Quality'
        else:
            return 'Poor Quality'
    
    def rank_dividend_stocks(self, min_years=5, min_market_cap=2_000_000_000):
        """
        Rank all dividend-paying stocks by quality
        
        Args:
            min_years: Minimum years of dividend payments
            min_market_cap: Minimum market cap filter
        
        Returns:
            DataFrame with stocks ranked by dividend quality
        """
        
        # Get dividend-paying stocks
        query = text("""
            SELECT DISTINCT dh.symbol
            FROM dividend_history dh
            JOIN symbol_universe su ON dh.symbol = su.symbol
            WHERE su.market_cap >= :min_cap
                AND su.is_etf = FALSE
                AND su.security_type = 'Common Stock'
            GROUP BY dh.symbol
            HAVING COUNT(DISTINCT EXTRACT(YEAR FROM dh.ex_date)) >= :min_years
        """)
        
        with self.engine.connect() as conn:
            result = conn.execute(query, {
                'min_cap': min_market_cap,
                'min_years': min_years
            })
            symbols = [row[0] for row in result.fetchall()]
        
        logger.info(f"Analyzing {len(symbols)} dividend-paying stocks...")
        
        # Calculate metrics for all symbols
        all_metrics = []
        for symbol in symbols:
            metrics = self.integrate_with_excess_cash_flow(symbol)
            if metrics['dividend_quality_score'] is not None:
                all_metrics.append(metrics)
        
        # Convert to DataFrame and rank
        df = pd.DataFrame(all_metrics)
        
        if not df.empty:
            # Sort by dividend quality score
            df = df.sort_values('dividend_quality_score', ascending=False)
            df['rank'] = range(1, len(df) + 1)
            
            # Add percentile
            df['percentile'] = 100 - (df['rank'] - 1) / len(df) * 100
        
        return df
    
    def save_to_database(self, df):
        """Save dividend metrics to database"""
        
        if df.empty:
            logger.warning("No data to save")
            return
        
        # Create table if it doesn't exist
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS dividend_sustainability_metrics (
            symbol VARCHAR(10) PRIMARY KEY,
            payment_streak_years INTEGER,
            increase_streak_years INTEGER,
            avg_growth_rate_5y NUMERIC(8, 2),
            payout_ratio_earnings NUMERIC(8, 2),
            payout_ratio_fcf NUMERIC(8, 2),
            sustainability_score NUMERIC(6, 2),
            safety_rating VARCHAR(20),
            excess_cash_flow_pct NUMERIC(6, 2),
            cash_flow_quality VARCHAR(20),
            dividend_quality_score NUMERIC(6, 2),
            dividend_quality_rating VARCHAR(30),
            rank INTEGER,
            percentile NUMERIC(5, 2),
            calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (symbol) REFERENCES symbol_universe(symbol) ON DELETE CASCADE
        )
        """
        
        with self.engine.begin() as conn:
            conn.execute(text(create_table_sql))
            
            # Save data
            df['calculated_at'] = datetime.now()
            df.to_sql('dividend_sustainability_metrics', conn, if_exists='replace', index=False)
            
        logger.info(f"Saved {len(df)} dividend sustainability metrics to database")


def main():
    """Main execution"""
    
    analyzer = DividendSustainabilityAnalyzer()
    
    # Example: Analyze a specific dividend stock
    print("\n" + "="*60)
    print("DIVIDEND SUSTAINABILITY ANALYSIS")
    print("="*60)
    
    # Analyze Johnson & Johnson as an example (dividend aristocrat)
    symbol = 'JNJ'
    metrics = analyzer.integrate_with_excess_cash_flow(symbol)
    
    if metrics:
        print(f"\n{symbol} Dividend Analysis:")
        print(f"  Payment Streak: {metrics['payment_streak_years']} years")
        print(f"  Increase Streak: {metrics['increase_streak_years']} years")
        print(f"  5Y Avg Growth: {metrics['avg_growth_rate_5y']:.1f}%" if metrics['avg_growth_rate_5y'] else "  5Y Avg Growth: N/A")
        print(f"  Payout Ratio (Earnings): {metrics['payout_ratio_earnings']:.1f}%" if metrics['payout_ratio_earnings'] else "  Payout Ratio (Earnings): N/A")
        print(f"  Payout Ratio (FCF): {metrics['payout_ratio_fcf']:.1f}%" if metrics['payout_ratio_fcf'] else "  Payout Ratio (FCF): N/A")
        print(f"  Sustainability Score: {metrics['sustainability_score']:.1f}" if metrics['sustainability_score'] else "  Sustainability Score: N/A")
        print(f"  Safety Rating: {metrics['safety_rating']}")
        print(f"  Excess Cash Flow: {metrics['excess_cash_flow_pct']:.1f}%" if metrics['excess_cash_flow_pct'] else "  Excess Cash Flow: N/A")
        print(f"  Dividend Quality Score: {metrics['dividend_quality_score']:.1f}" if metrics['dividend_quality_score'] else "  Dividend Quality Score: N/A")
        print(f"  Quality Rating: {metrics['dividend_quality_rating']}")
    
    # Rank all dividend stocks
    print("\n" + "="*60)
    print("RANKING ALL DIVIDEND STOCKS")
    print("="*60)
    
    df_ranked = analyzer.rank_dividend_stocks(min_years=5)
    
    if not df_ranked.empty:
        # Save to database
        analyzer.save_to_database(df_ranked)
        
        # Show top 10
        print("\nTop 10 Dividend Stocks by Quality:")
        for _, row in df_ranked.head(10).iterrows():
            print(f"  #{row['rank']} {row['symbol']}: Score {row['dividend_quality_score']:.1f} ({row['dividend_quality_rating']}) - {row['payment_streak_years']}yr streak")
        
        # Show statistics
        print(f"\nDividend Stock Statistics:")
        print(f"  Total analyzed: {len(df_ranked)}")
        print(f"  Dividend Aristocrats: {len(df_ranked[df_ranked['dividend_quality_rating'] == 'Dividend Aristocrat'])}")
        print(f"  High Quality: {len(df_ranked[df_ranked['dividend_quality_rating'] == 'High Quality'])}")
        print(f"  Average payment streak: {df_ranked['payment_streak_years'].mean():.1f} years")
        print(f"  Average payout ratio: {df_ranked['payout_ratio_earnings'].mean():.1f}%")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())