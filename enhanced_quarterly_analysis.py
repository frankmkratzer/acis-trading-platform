#!/usr/bin/env python3
"""
Enhanced Quarterly Analysis for Funnel System
Adds quarterly trend analysis and seasonal strength detection
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

class EnhancedQuarterlyAnalysis:
    def __init__(self):
        load_dotenv()
        self.engine = create_engine(os.getenv('POSTGRES_URL'))
    
    def analyze_quarterly_trends(self, symbols=None, min_quarters=8):
        """Analyze quarterly trends for enhanced funnel scoring"""
        print("ENHANCED QUARTERLY TREND ANALYSIS")
        print("=" * 50)
        
        with self.engine.connect() as conn:
            # Build symbol filter
            symbol_filter = ""
            if symbols:
                symbol_list = "'" + "','".join(symbols) + "'"
                symbol_filter = f"AND f.symbol IN ({symbol_list})"
            
            # Get comprehensive quarterly data with trends
            query = text(f"""
                WITH quarterly_data AS (
                    SELECT 
                        f.symbol,
                        f.fiscal_date,
                        f.totalrevenue,
                        f.netincome,
                        f.operatingcashflow,
                        f.eps,
                        f.cash_flow_per_share,
                        
                        -- Quarter-over-quarter comparisons
                        LAG(f.totalrevenue, 1) OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date) as prev_q_revenue,
                        LAG(f.netincome, 1) OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date) as prev_q_income,
                        LAG(f.eps, 1) OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date) as prev_q_eps,
                        
                        -- Year-over-year comparisons (same quarter previous year)
                        LAG(f.totalrevenue, 4) OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date) as yoy_revenue,
                        LAG(f.netincome, 4) OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date) as yoy_income,
                        LAG(f.eps, 4) OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date) as yoy_eps,
                        
                        -- Multi-quarter trends
                        LAG(f.totalrevenue, 2) OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date) as revenue_2q_ago,
                        LAG(f.totalrevenue, 3) OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date) as revenue_3q_ago,
                        
                        -- Count quarters of data
                        COUNT(*) OVER (PARTITION BY f.symbol) as quarters_available
                        
                    FROM fundamentals_quarterly f
                    WHERE f.fiscal_date >= '2022-01-01'
                      AND f.totalrevenue > 0
                      AND f.eps IS NOT NULL
                      {symbol_filter}
                ),
                trend_analysis AS (
                    SELECT 
                        symbol,
                        fiscal_date,
                        totalrevenue,
                        netincome,
                        eps,
                        cash_flow_per_share,
                        
                        -- Sequential quarterly growth rates
                        CASE 
                            WHEN prev_q_revenue > 0 THEN 
                                (totalrevenue - prev_q_revenue) / prev_q_revenue 
                            ELSE NULL 
                        END as qoq_revenue_growth,
                        
                        CASE 
                            WHEN prev_q_eps IS NOT NULL AND prev_q_eps != 0 THEN 
                                (eps - prev_q_eps) / ABS(prev_q_eps)
                            ELSE NULL 
                        END as qoq_eps_growth,
                        
                        -- Year-over-year growth rates
                        CASE 
                            WHEN yoy_revenue > 0 THEN 
                                (totalrevenue - yoy_revenue) / yoy_revenue 
                            ELSE NULL 
                        END as yoy_revenue_growth,
                        
                        CASE 
                            WHEN yoy_eps IS NOT NULL AND yoy_eps != 0 THEN 
                                (eps - yoy_eps) / ABS(yoy_eps)
                            ELSE NULL 
                        END as yoy_eps_growth,
                        
                        -- Multi-quarter revenue trend
                        CASE 
                            WHEN revenue_2q_ago > 0 AND revenue_3q_ago > 0 THEN
                                CASE 
                                    WHEN totalrevenue > prev_q_revenue AND prev_q_revenue > revenue_2q_ago THEN 'ACCELERATING'
                                    WHEN totalrevenue > prev_q_revenue AND prev_q_revenue < revenue_2q_ago THEN 'RECOVERING'
                                    WHEN totalrevenue < prev_q_revenue AND prev_q_revenue > revenue_2q_ago THEN 'DECELERATING'
                                    WHEN totalrevenue < prev_q_revenue AND prev_q_revenue < revenue_2q_ago THEN 'DECLINING'
                                    ELSE 'STABLE'
                                END
                            ELSE NULL
                        END as revenue_trend_direction,
                        
                        quarters_available
                        
                    FROM quarterly_data
                    WHERE quarters_available >= {min_quarters}
                )
                SELECT * FROM trend_analysis 
                WHERE fiscal_date >= '2023-01-01'
                ORDER BY symbol, fiscal_date DESC
            """)
            
            df = pd.read_sql(query, conn)
            
            if len(df) == 0:
                print("No quarterly data found")
                return pd.DataFrame()
            
            print(f"Loaded {len(df)} quarterly records for {df['symbol'].nunique()} stocks")
            
            # Calculate quarterly strength scores
            quarterly_scores = self._calculate_quarterly_scores(df)
            
            return quarterly_scores
    
    def _calculate_quarterly_scores(self, df):
        """Calculate quarterly strength scores for each stock"""
        stock_scores = []
        
        for symbol in df['symbol'].unique():
            stock_data = df[df['symbol'] == symbol].sort_values('fiscal_date', ascending=False)
            
            if len(stock_data) < 4:  # Need at least 4 quarters
                continue
            
            # Get most recent 8 quarters for analysis
            recent_data = stock_data.head(8)
            
            # Calculate quarterly momentum scores
            quarterly_score = self._score_quarterly_performance(recent_data)
            
            if quarterly_score:
                stock_scores.append(quarterly_score)
        
        return pd.DataFrame(stock_scores).sort_values('quarterly_score', ascending=False)
    
    def _score_quarterly_performance(self, data):
        """Score quarterly performance across multiple dimensions"""
        if len(data) == 0:
            return None
        
        symbol = data.iloc[0]['symbol']
        latest = data.iloc[0]
        
        # Component 1: Recent quarterly growth momentum (25 points)
        growth_score = 0
        
        # YoY growth consistency (last 4 quarters)
        yoy_growth_rates = data.head(4)['yoy_revenue_growth'].dropna()
        if len(yoy_growth_rates) >= 2:
            positive_yoy = (yoy_growth_rates > 0).sum()
            if positive_yoy >= 3:  # 3+ quarters of YoY growth
                growth_score += 10
            elif positive_yoy >= 2:  # 2+ quarters of YoY growth
                growth_score += 6
        
        # Sequential quarterly momentum 
        qoq_growth_rates = data.head(4)['qoq_revenue_growth'].dropna()
        if len(qoq_growth_rates) >= 2:
            positive_qoq = (qoq_growth_rates > 0).sum()
            if positive_qoq >= 3:  # 3+ quarters of QoQ growth
                growth_score += 10
            elif positive_qoq >= 2:  # 2+ quarters of QoQ growth
                growth_score += 5
        
        # Recent acceleration bonus
        if len(data) >= 2:
            recent_trend = data.iloc[0]['revenue_trend_direction']
            if recent_trend == 'ACCELERATING':
                growth_score += 5
            elif recent_trend == 'RECOVERING':
                growth_score += 3
        
        # Component 2: EPS quality and trends (20 points)
        eps_score = 0
        
        # EPS growth consistency
        eps_growth_rates = data.head(4)['yoy_eps_growth'].dropna()
        if len(eps_growth_rates) >= 2:
            positive_eps = (eps_growth_rates > 0).sum()
            if positive_eps >= 3:
                eps_score += 12
            elif positive_eps >= 2:
                eps_score += 8
        
        # Recent EPS strength
        latest_eps = latest['eps']
        if latest_eps and latest_eps > 0:
            eps_score += 8
        
        # Component 3: Cash flow per share trends (15 points)
        cfps_score = 0
        
        recent_cfps = data.head(4)['cash_flow_per_share'].dropna()
        if len(recent_cfps) >= 2:
            # Check for improving CFPS trend
            improving_cfps = 0
            for i in range(len(recent_cfps)-1):
                if recent_cfps.iloc[i] > recent_cfps.iloc[i+1]:
                    improving_cfps += 1
            
            cfps_score = min(improving_cfps * 5, 15)  # Max 15 points
        
        # Component 4: Seasonal consistency (10 points)
        seasonal_score = 0
        
        # Check for consistent seasonal performance (same quarters year-over-year)
        quarters_with_data = len(data)
        if quarters_with_data >= 8:  # 2+ years of data
            # Check consistency across seasons
            consistent_quarters = 0
            for i in range(4, min(8, quarters_with_data)):
                current_q = data.iloc[i-4]  # Current year quarter
                prev_year_q = data.iloc[i]   # Previous year same quarter
                
                if (current_q['totalrevenue'] > prev_year_q['totalrevenue'] and 
                    current_q['eps'] and prev_year_q['eps'] and 
                    current_q['eps'] > prev_year_q['eps']):
                    consistent_quarters += 1
            
            seasonal_score = min(consistent_quarters * 2, 10)
        
        # Total quarterly score (max 70 points)
        total_quarterly_score = growth_score + eps_score + cfps_score + seasonal_score
        
        return {
            'symbol': symbol,
            'quarterly_score': total_quarterly_score,
            'growth_momentum': growth_score,
            'eps_quality': eps_score,
            'cfps_trend': cfps_score,
            'seasonal_consistency': seasonal_score,
            'latest_quarter': latest['fiscal_date'],
            'latest_revenue': latest['totalrevenue'],
            'latest_eps': latest['eps'],
            'revenue_trend': latest.get('revenue_trend_direction', 'UNKNOWN'),
            'quarters_analyzed': len(data)
        }
    
    def display_quarterly_rankings(self, df, top_n=20):
        """Display top quarterly performers"""
        print(f"\nTOP {top_n} QUARTERLY PERFORMERS:")
        print("=" * 80)
        print(f"{'Rank':<4} {'Symbol':<8} {'QScore':<7} {'Growth':<6} {'EPS':<5} {'CFPS':<5} {'Season':<6} {'Trend':<12}")
        print("-" * 80)
        
        for i, (_, row) in enumerate(df.head(top_n).iterrows(), 1):
            trend = row['revenue_trend'][:11] if row['revenue_trend'] else 'N/A'
            print(f"{i:<4} {row['symbol']:<8} {row['quarterly_score']:<7.0f} "
                  f"{row['growth_momentum']:<6.0f} {row['eps_quality']:<5.0f} "
                  f"{row['cfps_trend']:<5.0f} {row['seasonal_consistency']:<6.0f} {trend:<12}")
        
        print("-" * 80)
        print("QScore=Total, Growth=Revenue Momentum, EPS=Earnings Quality")
        print("CFPS=Cash Flow Trend, Season=Seasonal Consistency")

def main():
    """Test quarterly trend analysis"""
    analyzer = EnhancedQuarterlyAnalysis()
    
    # Run comprehensive quarterly analysis
    quarterly_scores = analyzer.analyze_quarterly_trends(min_quarters=6)
    
    if not quarterly_scores.empty:
        analyzer.display_quarterly_rankings(quarterly_scores, top_n=25)
        
        print(f"\n" + "=" * 50)
        print("QUARTERLY ANALYSIS COMPLETE!")
        print(f"Analyzed {len(quarterly_scores)} stocks with quarterly trends")
        print("=" * 50)
        
        # Summary statistics
        avg_score = quarterly_scores['quarterly_score'].mean()
        top_score = quarterly_scores['quarterly_score'].max()
        
        print(f"\nQuarterly Score Statistics:")
        print(f"  Average Score: {avg_score:.1f}")
        print(f"  Top Score: {top_score:.1f}")
        print(f"  Score Range: 0-70 points")
        
        # Trend analysis
        trend_counts = quarterly_scores['revenue_trend'].value_counts()
        print(f"\nRevenue Trend Distribution:")
        for trend, count in trend_counts.head(5).items():
            pct = count / len(quarterly_scores) * 100
            print(f"  {trend:<15} {count:>3} stocks ({pct:.1f}%)")
        
        return quarterly_scores
    else:
        print("No quarterly analysis data available")
        return pd.DataFrame()

if __name__ == "__main__":
    main()