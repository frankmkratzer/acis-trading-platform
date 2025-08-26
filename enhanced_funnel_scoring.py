#!/usr/bin/env python3
"""
Enhanced Funnel-Based Scoring System
Implements professional institutional methodology:
1. Excess Cash Flow Ratio (Cash Quality)
2. 10yr/5yr/Present Trend Analysis (Advancing/Declining/Stable)
3. Historical Valuation (Price at Extremes)
4. Long-term Growth vs S&P 500 (20-year consistency)
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

class EnhancedFunnelScoring:
    def __init__(self):
        load_dotenv()
        self.engine = create_engine(os.getenv('POSTGRES_URL'))
        self.sector_weights = {}  # Will be loaded from database
        
    def calculate_funnel_scores(self, strategy_type='value', cap_type='all'):
        """Main funnel scoring methodology with sector strength integration"""
        print(f"ENHANCED FUNNEL SCORING - {strategy_type.upper()}")
        print("=" * 60)
        
        with self.engine.connect() as conn:
            # Load sector strength weights
            self._load_sector_weights(conn)
            
            # Get comprehensive multi-year data
            df = self._get_comprehensive_data(conn, cap_type)
            
            if len(df) == 0:
                print("No data found")
                return pd.DataFrame()
            
            print(f"Analyzing {len(df)} stocks with {strategy_type} funnel + sector strength...")
            
            scores = []
            for _, row in df.iterrows():
                score_data = self._calculate_stock_score(row, strategy_type)
                if score_data:
                    scores.append(score_data)
            
            if not scores:
                return pd.DataFrame()
                
            # Convert to DataFrame and rank
            scores_df = pd.DataFrame(scores)
            scores_df['percentile'] = scores_df['total_score'].rank(pct=True)
            scores_df = scores_df.sort_values('total_score', ascending=False)
            
            self._display_results(scores_df, strategy_type)
            return scores_df
    
    def _load_sector_weights(self, conn):
        """Load sector strength weights from database"""
        try:
            result = conn.execute(text("""
                SELECT sector, strength_score
                FROM sector_strength_scores 
                WHERE as_of_date = CURRENT_DATE
                ORDER BY strength_score DESC
            """))
            
            sector_data = result.fetchall()
            if sector_data:
                # Convert to normalized weights (0.8 to 1.2 range for best to worst)
                scores = [row[1] for row in sector_data]
                min_score, max_score = min(scores), max(scores)
                
                for sector, score in sector_data:
                    if max_score > min_score:
                        # Normalize: strongest gets 1.2x, weakest gets 0.8x
                        normalized = 0.8 + 0.4 * ((score - min_score) / (max_score - min_score))
                    else:
                        normalized = 1.0
                    self.sector_weights[sector] = normalized
                
                print(f"Loaded sector weights: {len(self.sector_weights)} sectors")
            else:
                print("No sector strength data found - using equal weights")
                # Default equal weights
                for sector in ['TECHNOLOGY', 'FINANCE', 'LIFE SCIENCES', 'MANUFACTURING', 
                             'TRADE & SERVICES', 'ENERGY & TRANSPORTATION', 'REAL ESTATE & CONSTRUCTION']:
                    self.sector_weights[sector] = 1.0
                    
        except Exception as e:
            print(f"Could not load sector weights: {e}")
            # Fallback to equal weights
            for sector in ['TECHNOLOGY', 'FINANCE', 'LIFE SCIENCES', 'MANUFACTURING', 
                         'TRADE & SERVICES', 'ENERGY & TRANSPORTATION', 'REAL ESTATE & CONSTRUCTION']:
                self.sector_weights[sector] = 1.0
    
    def _get_comprehensive_data(self, conn, cap_type):
        """Get multi-year fundamental and price data with sector information"""
        query = text("""
            WITH yearly_fundamentals AS (
                SELECT 
                    f.symbol,
                    EXTRACT(YEAR FROM f.fiscal_date) as year,
                    f.fiscal_date,
                    
                    -- Core Financials
                    f.totalrevenue,
                    f.netincome,
                    f.operatingcashflow,
                    f.free_cf,
                    f.totalassets,
                    f.totalshareholderequity,
                    f.totalliabilities,
                    f.dividendpayout,
                    
                    -- Critical Per-Share Metrics (now available!)
                    f.eps,
                    f.cash_flow_per_share,
                    
                    -- Growth Calculations  
                    LAG(f.totalrevenue, 1) OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date) as prev_1yr_revenue,
                    LAG(f.totalrevenue, 5) OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date) as prev_5yr_revenue,
                    LAG(f.totalrevenue, 10) OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date) as prev_10yr_revenue,
                    
                    LAG(f.netincome, 1) OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date) as prev_1yr_income,
                    LAG(f.netincome, 5) OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date) as prev_5yr_income,
                    LAG(f.netincome, 10) OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date) as prev_10yr_income,
                    
                    LAG(f.free_cf, 1) OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date) as prev_1yr_fcf,
                    LAG(f.free_cf, 5) OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date) as prev_5yr_fcf,
                    LAG(f.free_cf, 10) OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date) as prev_10yr_fcf,
                    
                    -- Count years of data
                    COUNT(*) OVER (PARTITION BY f.symbol) as years_of_data
                    
                FROM fundamentals_annual f
                WHERE f.fiscal_date >= CURRENT_DATE - INTERVAL '15 years'
                  AND f.totalrevenue > 0
                  AND f.totalshareholderequity > 0
            ),
            latest_with_history AS (
                SELECT DISTINCT ON (symbol) 
                    *,
                    -- Current price data
                    (SELECT adjusted_close 
                     FROM stock_eod_daily s 
                     WHERE s.symbol = yearly_fundamentals.symbol 
                     ORDER BY s.trade_date DESC 
                     LIMIT 1) as current_price,
                     
                    -- Historical prices for valuation analysis
                    (SELECT MIN(adjusted_close) 
                     FROM stock_eod_daily s 
                     WHERE s.symbol = yearly_fundamentals.symbol 
                     AND s.trade_date >= CURRENT_DATE - INTERVAL '5 years') as min_price_5yr,
                     
                    (SELECT MAX(adjusted_close) 
                     FROM stock_eod_daily s 
                     WHERE s.symbol = yearly_fundamentals.symbol 
                     AND s.trade_date >= CURRENT_DATE - INTERVAL '5 years') as max_price_5yr,
                     
                    -- Long-term price performance (when available)
                    (SELECT adjusted_close 
                     FROM stock_eod_daily s 
                     WHERE s.symbol = yearly_fundamentals.symbol 
                     AND s.trade_date <= CURRENT_DATE - INTERVAL '10 years'
                     ORDER BY s.trade_date DESC 
                     LIMIT 1) as price_10yr_ago
                     
                FROM yearly_fundamentals
                WHERE years_of_data >= 3  -- Minimum data requirement
                ORDER BY symbol, fiscal_date DESC
            )
            SELECT 
                l.*,
                p.sector,
                p.industry
            FROM latest_with_history l
            JOIN pure_us_stocks p ON l.symbol = p.symbol
            WHERE current_price IS NOT NULL 
              AND current_price > 5
              AND totalrevenue > 100000000  -- $100M minimum
            ORDER BY symbol
        """)
        
        return pd.read_sql(query, conn)
    
    def _calculate_stock_score(self, row, strategy_type):
        """Calculate comprehensive funnel score for a stock"""
        symbol = row['symbol']
        
        # Extract financial data
        revenue = float(row['totalrevenue']) if row['totalrevenue'] else 0
        net_income = float(row['netincome']) if row['netincome'] else 0
        operating_cf = float(row['operatingcashflow']) if row['operatingcashflow'] else 0
        free_cf = float(row['free_cf']) if row['free_cf'] else 0
        
        # Skip if insufficient data
        if revenue <= 0 or operating_cf <= 0:
            return None
        
        # FUNNEL COMPONENT 1: EXCESS CASH FLOW RATIO (25 points)
        excess_cf_score = self._calculate_excess_cash_flow_score(row)
        
        # FUNNEL COMPONENT 2: TREND ANALYSIS (30 points)  
        trend_score = self._calculate_trend_analysis_score(row)
        
        # FUNNEL COMPONENT 3: VALUATION EXTREMES (25 points)
        valuation_score = self._calculate_valuation_extreme_score(row)
        
        # FUNNEL COMPONENT 4: LONG-TERM GROWTH (20 points)
        growth_score = self._calculate_long_term_growth_score(row)
        
        # Get sector multiplier for sector strength
        sector = row.get('sector', 'UNKNOWN')
        sector_multiplier = self.sector_weights.get(sector, 1.0)
        
        # Strategy-specific weighting
        weighted_score = self._apply_strategy_weights(
            excess_cf_score, trend_score, valuation_score, growth_score, strategy_type
        )
        
        # Apply sector strength boost/penalty
        final_score = weighted_score * sector_multiplier
        
        return {
            'symbol': symbol,
            'sector': sector,
            'sector_multiplier': sector_multiplier,
            'base_score': weighted_score,
            'total_score': final_score,
            'excess_cf_score': excess_cf_score,
            'trend_score': trend_score,
            'valuation_score': valuation_score,
            'growth_score': growth_score,
            'strategy_type': strategy_type
        }
    
    def _calculate_excess_cash_flow_score(self, row):
        """Component 1: Excess Cash Flow / Total Cash Flow = % (100% = excellent)"""
        operating_cf = float(row['operatingcashflow']) if row['operatingcashflow'] else 0
        free_cf = float(row['free_cf']) if row['free_cf'] else 0
        cash_flow_per_share = float(row['cash_flow_per_share']) if row['cash_flow_per_share'] else None
        
        if operating_cf <= 0:
            return 0
        
        # Enhanced: Use CFPS quality as additional factor
        excess_ratio = free_cf / operating_cf if operating_cf > 0 else 0
        
        # Bonus for strong cash flow per share (indicates quality)
        cfps_bonus = 0
        if cash_flow_per_share and cash_flow_per_share > 5:  # $5+ CFPS is strong
            cfps_bonus = 5
        elif cash_flow_per_share and cash_flow_per_share > 2:  # $2+ CFPS is decent
            cfps_bonus = 2
        
        # Score: 0-25 points based on excess ratio + CFPS bonus
        base_score = 0
        if excess_ratio >= 0.80:  # 80%+ is excellent
            base_score = 25
        elif excess_ratio >= 0.60:  # 60-80% is good
            base_score = 20
        elif excess_ratio >= 0.40:  # 40-60% is fair  
            base_score = 15
        elif excess_ratio >= 0.20:  # 20-40% is poor
            base_score = 10
        else:  # <20% is very poor
            base_score = 5
            
        return min(base_score + cfps_bonus, 30)  # Cap at 30 with bonus
    
    def _calculate_trend_analysis_score(self, row):
        """Component 2: 10yr/5yr/Present trend analysis (Advancing/Declining/Stable)"""
        # Revenue trends
        revenue_now = float(row['totalrevenue']) if row['totalrevenue'] else 0
        revenue_5yr = float(row['prev_5yr_revenue']) if row['prev_5yr_revenue'] else 0
        revenue_10yr = float(row['prev_10yr_revenue']) if row['prev_10yr_revenue'] else 0
        
        # Income trends  
        income_now = float(row['netincome']) if row['netincome'] else 0
        income_5yr = float(row['prev_5yr_income']) if row['prev_5yr_income'] else 0
        income_10yr = float(row['prev_10yr_income']) if row['prev_10yr_income'] else 0
        
        # FCF trends
        fcf_now = float(row['free_cf']) if row['free_cf'] else 0
        fcf_5yr = float(row['prev_5yr_fcf']) if row['prev_5yr_fcf'] else 0
        fcf_10yr = float(row['prev_10yr_fcf']) if row['prev_10yr_fcf'] else 0
        
        trend_score = 0
        
        # Revenue trend (10 points)
        if revenue_5yr > 0 and revenue_now > revenue_5yr * 1.1:  # 10%+ growth over 5yr
            trend_score += 5
        if revenue_10yr > 0 and revenue_now > revenue_10yr * 1.2:  # 20%+ growth over 10yr  
            trend_score += 5
        
        # Income trend (10 points)
        if income_5yr > 0 and income_now > income_5yr * 1.1:
            trend_score += 5
        if income_10yr > 0 and income_now > income_10yr * 1.2:
            trend_score += 5
        
        # FCF trend (10 points)
        if fcf_5yr > 0 and fcf_now > fcf_5yr * 1.1:
            trend_score += 5
        if fcf_10yr > 0 and fcf_now > fcf_10yr * 1.2:
            trend_score += 5
        
        return min(trend_score, 30)  # Max 30 points
    
    def _calculate_valuation_extreme_score(self, row):
        """Component 3: Price at historical extremes (low = good for value)"""
        current_price = float(row['current_price']) if row['current_price'] else 0
        min_price_5yr = float(row['min_price_5yr']) if row['min_price_5yr'] else 0
        max_price_5yr = float(row['max_price_5yr']) if row['max_price_5yr'] else 0
        revenue = float(row['totalrevenue']) if row['totalrevenue'] else 1
        free_cf = float(row['free_cf']) if row['free_cf'] else 0
        eps = float(row['eps']) if row['eps'] else None
        cash_flow_per_share = float(row['cash_flow_per_share']) if row['cash_flow_per_share'] else None
        
        if current_price <= 0 or min_price_5yr <= 0 or max_price_5yr <= 0:
            return 0
        
        # Price position in 5-year range (lower = better for value)
        price_range = max_price_5yr - min_price_5yr
        if price_range > 0:
            price_percentile = (current_price - min_price_5yr) / price_range
        else:
            price_percentile = 0.5
        
        # Estimate market cap for ratios (rough approximation)
        shares_estimate = 1000000  # Will need actual shares outstanding
        market_cap_estimate = current_price * shares_estimate
        
        # Price/Revenue ratio analysis
        p_rev_ratio = market_cap_estimate / revenue if revenue > 0 else 999
        
        # Price/FCF ratio analysis  
        p_fcf_ratio = market_cap_estimate / free_cf if free_cf > 0 else 999
        
        valuation_score = 0
        
        # Historical position score (15 points) - lower price = higher score
        if price_percentile <= 0.2:  # Bottom 20% of 5-year range
            valuation_score += 15
        elif price_percentile <= 0.4:  # Bottom 40%
            valuation_score += 10
        elif price_percentile <= 0.6:  # Bottom 60%
            valuation_score += 5
        
        # Enhanced ratio analysis with EPS (15 points total)
        ratio_score = 0
        
        # P/E Ratio (if EPS available)
        if eps and eps > 0:
            pe_ratio = current_price / eps
            if pe_ratio < 15:  # Excellent P/E
                ratio_score += 6
            elif pe_ratio < 25:  # Good P/E
                ratio_score += 3
        
        # P/Revenue ratio 
        if p_rev_ratio < 2.0:  # P/Rev under 2x
            ratio_score += 3
        
        # P/FCF ratio
        if p_fcf_ratio < 15.0 and free_cf > 0:  # P/FCF under 15x
            ratio_score += 3
        
        # P/CFPS ratio (if CFPS available)
        if cash_flow_per_share and cash_flow_per_share > 0:
            p_cfps_ratio = current_price / cash_flow_per_share
            if p_cfps_ratio < 10:  # Excellent P/CFPS
                ratio_score += 3
        
        valuation_score += ratio_score
        return min(valuation_score, 30)  # Max 30 points with EPS enhancement
    
    def _calculate_long_term_growth_score(self, row):
        """Component 4: Long-term 20-year returns vs S&P 500"""
        current_price = float(row['current_price']) if row['current_price'] else 0
        price_10yr_ago = float(row['price_10yr_ago']) if row['price_10yr_ago'] else 0
        
        if current_price <= 0 or price_10yr_ago <= 0:
            return 10  # Neutral score for missing data
        
        # 10-year annualized return (best proxy we have for long-term)
        total_return = (current_price / price_10yr_ago) - 1
        annualized_return = ((1 + total_return) ** (1/10)) - 1
        
        # S&P 500 historical average ~10% annually
        sp500_benchmark = 0.10
        excess_return = annualized_return - sp500_benchmark
        
        # Score based on excess return vs S&P 500
        if excess_return >= 0.05:  # 5%+ annual excess return
            return 20
        elif excess_return >= 0.02:  # 2-5% excess return
            return 15
        elif excess_return >= 0:  # Positive excess return
            return 12
        elif excess_return >= -0.02:  # Within 2% of S&P
            return 8
        else:  # Significantly underperforming
            return 5
    
    def _apply_strategy_weights(self, excess_cf, trend, valuation, growth, strategy_type):
        """Apply strategy-specific weightings to funnel components"""
        
        weights = {
            'value': {
                'excess_cf': 1.2,    # Value loves cash flow
                'trend': 0.9,       # Trends less important
                'valuation': 1.4,   # Valuation most important
                'growth': 0.8       # Growth less important
            },
            'growth': {
                'excess_cf': 1.1,   # Cash flow important
                'trend': 1.3,       # Trend very important
                'valuation': 0.8,   # Valuation less important
                'growth': 1.4       # Growth most important
            },
            'momentum': {
                'excess_cf': 0.9,   # Cash flow less critical
                'trend': 1.2,       # Recent trends important
                'valuation': 0.7,   # Valuation least important
                'growth': 1.5       # Growth momentum critical
            },
            'dividend': {
                'excess_cf': 1.5,   # Cash flow critical for dividends
                'trend': 1.1,       # Stable trends important
                'valuation': 1.0,   # Fair valuation important
                'growth': 0.7       # Growth less critical
            }
        }.get(strategy_type, {
            'excess_cf': 1.0, 'trend': 1.0, 'valuation': 1.0, 'growth': 1.0
        })
        
        weighted_score = (
            excess_cf * weights['excess_cf'] +
            trend * weights['trend'] +
            valuation * weights['valuation'] +
            growth * weights['growth']
        )
        
        return weighted_score
    
    def _display_results(self, scores_df, strategy_type):
        """Display top results from funnel analysis with sector information"""
        print(f"\nTop 15 {strategy_type.upper()} Stocks - Enhanced Funnel + Sector Strength:")
        print("-" * 105)
        print(f"{'Rank':<4} {'Symbol':<8} {'Sector':<25} {'Total':<7} {'Base':<7} {'SxM':<5} {'ExCF':<5} {'Trend':<6} {'Value':<6} {'Growth':<6}")
        print("-" * 105)
        
        for i, (_, row) in enumerate(scores_df.head(15).iterrows(), 1):
            sector_short = row.get('sector', 'N/A')[:24] if row.get('sector') else 'N/A'
            base_score = row.get('base_score', row['total_score'])
            sector_mult = row.get('sector_multiplier', 1.0)
            
            print(f"{i:<4} {row['symbol']:<8} {sector_short:<25} {row['total_score']:<7.1f} "
                  f"{base_score:<7.1f} {sector_mult:<5.2f} "
                  f"{row['excess_cf_score']:<5.0f} {row['trend_score']:<6.0f} "
                  f"{row['valuation_score']:<6.0f} {row['growth_score']:<6.0f}")
                  
        print("-" * 105)
        print("Total=Final Score, Base=Pre-Sector, SxM=Sector Multiplier")
        
        # Show sector distribution
        if len(scores_df) >= 15:
            print(f"\nSector Representation in Top 15:")
            sector_counts = scores_df.head(15)['sector'].value_counts()
            for sector, count in sector_counts.items():
                print(f"  {sector[:35]:<35} {count:>2} stocks")

def main():
    """Test the enhanced funnel scoring system"""
    scorer = EnhancedFunnelScoring()
    
    # Test each strategy type
    strategies = ['value', 'growth', 'momentum', 'dividend']
    
    for strategy in strategies:
        print(f"\n{'='*70}")
        scores_df = scorer.calculate_funnel_scores(strategy_type=strategy)
        if len(scores_df) > 0:
            print(f"\nFunnel analysis complete for {strategy} - {len(scores_df)} stocks scored")
        else:
            print(f"\nNo qualifying stocks for {strategy} strategy")

if __name__ == "__main__":
    main()