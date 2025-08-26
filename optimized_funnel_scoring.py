#!/usr/bin/env python3
"""
Optimized Enhanced Funnel Scoring System
Implements key optimization recommendations:
1. Sector allocation limits (max 30% per sector, reduce Manufacturing to 25%)
2. Position size limits (max 2% per position)
3. Enhanced Mid Cap weighting (best Sharpe ratio performance)
4. Quality screening for Small Cap strategies
5. Finance sector integration (increase from 0% exposure)
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

class OptimizedFunnelScoring:
    def __init__(self):
        load_dotenv()
        self.engine = create_engine(os.getenv('POSTGRES_URL'))
        self.sector_weights = {}
        
        # Optimization parameters based on backtest analysis
        self.SECTOR_LIMITS = {
            'MANUFACTURING': 0.25,  # Reduced from 35% to 25%
            'TECHNOLOGY': 0.30,     # Max 30% any sector
            'FINANCE': 0.20,        # Increase Finance exposure
            'LIFE SCIENCES': 0.30,
            'TRADE & SERVICES': 0.30,
            'ENERGY & TRANSPORTATION': 0.30,
            'REAL ESTATE & CONSTRUCTION': 0.25,
            'INDUSTRIAL APPLICATIONS AND SERVICES': 0.30
        }
        
        # Market cap performance multipliers (based on backtest Sharpe ratios)
        self.CAP_MULTIPLIERS = {
            'small_cap': 0.93,   # 2.56 Sharpe / 2.76 avg = 0.93
            'mid_cap': 1.10,     # 3.05 Sharpe / 2.76 avg = 1.10 (BEST)
            'large_cap': 0.96    # 2.65 Sharpe / 2.76 avg = 0.96
        }
        
        self.MAX_POSITION_SIZE = 0.02  # 2% max position size
        
    def calculate_optimized_funnel_scores(self, strategy_type='value', cap_type='all'):
        """Optimized funnel scoring with sector limits and quality screening"""
        print(f"OPTIMIZED FUNNEL SCORING - {strategy_type.upper()}")
        print("=" * 60)
        print("Optimizations: Sector Limits | Position Sizing | Mid Cap Focus | Quality Screen")
        print()
        
        with self.engine.connect() as conn:
            # Load optimized sector weights
            self._load_optimized_sector_weights(conn)
            
            # Get comprehensive data with quality metrics
            df = self._get_enhanced_data_with_quality(conn, cap_type, strategy_type)
            
            if len(df) == 0:
                print("No data found")
                return pd.DataFrame()
            
            print(f"Analyzing {len(df)} stocks with optimized {strategy_type} scoring...")
            
            # Calculate enhanced scores
            scores = []
            for _, row in df.iterrows():
                score_data = self._calculate_optimized_stock_score(row, strategy_type, cap_type)
                if score_data:
                    scores.append(score_data)
            
            if not scores:
                return pd.DataFrame()
            
            # Convert to DataFrame and apply optimizations
            scores_df = pd.DataFrame(scores)
            
            # Apply sector allocation limits
            optimized_df = self._apply_sector_allocation_limits(scores_df)
            
            # Apply position size limits
            optimized_df = self._apply_position_size_limits(optimized_df)
            
            # Final ranking
            optimized_df['percentile'] = optimized_df['total_score'].rank(pct=True)
            optimized_df = optimized_df.sort_values('total_score', ascending=False)
            
            self._display_optimization_results(optimized_df, strategy_type, cap_type)
            return optimized_df
    
    def _load_optimized_sector_weights(self, conn):
        """Load sector weights with optimization adjustments"""
        try:
            result = conn.execute(text("""
                SELECT sector, strength_score
                FROM sector_strength_scores 
                WHERE as_of_date = CURRENT_DATE
                ORDER BY strength_score DESC
            """))
            
            sector_data = result.fetchall()
            if sector_data:
                scores = [row[1] for row in sector_data]
                min_score, max_score = min(scores), max(scores)
                
                for sector, score in sector_data:
                    if max_score > min_score:
                        # Base normalized weight
                        normalized = 0.85 + 0.30 * ((score - min_score) / (max_score - min_score))
                    else:
                        normalized = 1.0
                    
                    # Apply optimization adjustments
                    if sector == 'MANUFACTURING':
                        normalized *= 0.85  # Reduce Manufacturing preference
                    elif sector == 'FINANCE':
                        normalized *= 1.15  # Increase Finance preference
                    elif 'MID' in sector or sector in ['TECHNOLOGY', 'LIFE SCIENCES']:
                        normalized *= 1.05  # Slight boost to growth sectors
                    
                    self.sector_weights[sector] = normalized
                
                print(f"Loaded optimized sector weights: {len(self.sector_weights)} sectors")
                print(f"  Manufacturing weight reduced, Finance weight increased")
            else:
                # Default optimized weights
                self.sector_weights = {
                    'TECHNOLOGY': 1.10,
                    'FINANCE': 1.15,  # Increased
                    'LIFE SCIENCES': 1.05,
                    'MANUFACTURING': 0.85,  # Reduced
                    'TRADE & SERVICES': 1.00,
                    'ENERGY & TRANSPORTATION': 1.00,
                    'REAL ESTATE & CONSTRUCTION': 0.95,
                    'INDUSTRIAL APPLICATIONS AND SERVICES': 1.05
                }
                
        except Exception as e:
            print(f"Using default optimized weights: {e}")
            self.sector_weights = {
                'TECHNOLOGY': 1.10,
                'FINANCE': 1.15,
                'LIFE SCIENCES': 1.05,
                'MANUFACTURING': 0.85,
                'TRADE & SERVICES': 1.00,
                'ENERGY & TRANSPORTATION': 1.00,
                'REAL ESTATE & CONSTRUCTION': 0.95,
                'INDUSTRIAL APPLICATIONS AND SERVICES': 1.05
            }
    
    def _get_enhanced_data_with_quality(self, conn, cap_type, strategy_type):
        """Get data with additional quality metrics for screening"""
        
        # Market cap filter
        if cap_type == 'small_cap':
            cap_filter = "p.market_cap < 2000000000"
        elif cap_type == 'mid_cap':
            cap_filter = "p.market_cap BETWEEN 2000000000 AND 10000000000"
        elif cap_type == 'large_cap':
            cap_filter = "p.market_cap > 10000000000"
        else:
            cap_filter = "p.market_cap > 0"
        
        query = text(f"""
            WITH latest_fundamentals AS (
                SELECT DISTINCT ON (symbol)
                    symbol, fiscal_date, eps, totalrevenue as revenue, 
                    totalliabilities as total_debt, operatingcashflow as operating_cash_flow, 
                    totalassets as total_assets, totalshareholderequity as stockholder_equity, 
                    grossprofit as gross_profit
                FROM fundamentals_quarterly
                WHERE eps IS NOT NULL 
                  AND fiscal_date >= '2020-01-01'
                ORDER BY symbol, fiscal_date DESC
            ),
            quality_metrics AS (
                SELECT 
                    p.symbol,
                    p.market_cap,
                    p.sector,
                    s.adjusted_close as current_price,
                    f.eps,
                    f.revenue,
                    0 as total_cash,  -- Simplified for initial implementation
                    COALESCE(f.total_debt, 0) as total_debt,
                    COALESCE(f.operating_cash_flow, 0) as operating_cash_flow,
                    f.total_assets,
                    f.stockholder_equity,
                    f.gross_profit,
                    
                    -- Quality screening metrics
                    CASE WHEN f.total_debt > 0 AND f.total_assets > 0 
                         THEN f.total_debt::FLOAT / f.total_assets ELSE 0 END as debt_to_assets,
                    CASE WHEN f.revenue > 0 AND f.gross_profit > 0
                         THEN f.gross_profit::FLOAT / f.revenue ELSE 0 END as gross_margin,
                    CASE WHEN f.stockholder_equity > 0 AND f.eps > 0 
                         THEN f.eps::FLOAT / NULLIF(f.stockholder_equity::FLOAT, 0) * 1000000000 ELSE 0 END as roe_estimate,
                    CASE WHEN f.eps > 0 THEN s.adjusted_close / NULLIF(f.eps, 0) ELSE NULL END as pe_ratio,
                    
                    -- Price momentum
                    LAG(s.adjusted_close, 21) OVER (PARTITION BY p.symbol ORDER BY s.trade_date) as price_21d_ago,
                    LAG(s.adjusted_close, 63) OVER (PARTITION BY p.symbol ORDER BY s.trade_date) as price_63d_ago
                    
                FROM pure_us_stocks p
                JOIN stock_eod_daily s ON p.symbol = s.symbol
                JOIN latest_fundamentals f ON p.symbol = f.symbol
                WHERE s.trade_date = (SELECT MAX(trade_date) FROM stock_eod_daily WHERE trade_date <= CURRENT_DATE - INTERVAL '1 day')
                  AND {cap_filter}
                  AND p.sector IS NOT NULL
                  AND s.adjusted_close > 5  -- Minimum price filter
                  AND f.eps IS NOT NULL
                  AND f.revenue > 0
            )
            SELECT *,
                   CASE WHEN price_21d_ago > 0 THEN (current_price - price_21d_ago) / price_21d_ago ELSE 0 END as momentum_21d,
                   CASE WHEN price_63d_ago > 0 THEN (current_price - price_63d_ago) / price_63d_ago ELSE 0 END as momentum_63d
            FROM quality_metrics
            WHERE debt_to_assets <= 0.7  -- Quality screen: reasonable leverage
              AND gross_margin >= 0.15   -- Quality screen: minimum profitability
              AND pe_ratio BETWEEN 3 AND 100  -- Reasonable valuation range
            ORDER BY symbol
        """)
        
        df = pd.read_sql(query, conn)
        return df
    
    def _calculate_optimized_stock_score(self, row, strategy_type, cap_type):
        """Enhanced stock scoring with optimization factors"""
        try:
            symbol = row['symbol']
            sector = row['sector']
            
            # Base fundamental scores
            base_score = self._calculate_base_fundamental_score(row, strategy_type)
            if base_score <= 0:
                return None
            
            # Apply market cap performance multiplier
            cap_multiplier = self.CAP_MULTIPLIERS.get(cap_type, 1.0)
            base_score *= cap_multiplier
            
            # Apply sector optimization weights
            sector_weight = self.sector_weights.get(sector, 1.0)
            sector_adjusted_score = base_score * sector_weight
            
            # Quality screening bonus for Small Cap
            if cap_type == 'small_cap':
                quality_bonus = self._calculate_quality_bonus(row)
                sector_adjusted_score *= (1 + quality_bonus)
            
            # Momentum overlay (all strategies)
            momentum_factor = self._calculate_momentum_factor(row)
            final_score = sector_adjusted_score * momentum_factor
            
            return {
                'symbol': symbol,
                'sector': sector,
                'base_score': base_score,
                'cap_multiplier': cap_multiplier,
                'sector_weight': sector_weight,
                'momentum_factor': momentum_factor,
                'total_score': final_score,
                'current_price': row['current_price'],
                'market_cap': row['market_cap'],
                'pe_ratio': row.get('pe_ratio', 0),
                'debt_to_assets': row.get('debt_to_assets', 0),
                'gross_margin': row.get('gross_margin', 0)
            }
            
        except Exception as e:
            return None
    
    def _calculate_base_fundamental_score(self, row, strategy_type):
        """Calculate base fundamental score by strategy type"""
        try:
            eps = row.get('eps', 0)
            pe_ratio = row.get('pe_ratio', 0)
            debt_to_assets = row.get('debt_to_assets', 0)
            gross_margin = row.get('gross_margin', 0)
            operating_cash_flow = row.get('operating_cash_flow', 0)
            
            if strategy_type == 'value':
                # Value: Low PE, strong balance sheet, positive earnings
                pe_score = max(0, 20 - pe_ratio) if pe_ratio > 0 else 0
                debt_score = max(0, (1 - debt_to_assets) * 20)
                earnings_score = min(eps * 10, 30) if eps > 0 else 0
                cash_flow_score = min(operating_cash_flow / 1000000, 20) if operating_cash_flow > 0 else 0
                return pe_score + debt_score + earnings_score + cash_flow_score
                
            elif strategy_type == 'growth':
                # Growth: Strong margins, reasonable PE for growth, positive momentum
                margin_score = gross_margin * 40 if gross_margin > 0 else 0
                eps_growth_score = min(eps * 8, 30) if eps > 0 else 0
                pe_growth_score = max(0, 40 - abs(pe_ratio - 20)) if 5 < pe_ratio < 40 else 0
                return margin_score + eps_growth_score + pe_growth_score
                
            elif strategy_type == 'momentum':
                # Momentum: Recent performance, earnings acceleration
                earnings_momentum = min(eps * 12, 35) if eps > 0 else 0
                quality_factor = (1 - debt_to_assets) * 15
                margin_factor = gross_margin * 25
                return earnings_momentum + quality_factor + margin_factor
                
            elif strategy_type == 'dividend':
                # Dividend: Strong cash flow, low debt, stable earnings
                cash_flow_score = min(operating_cash_flow / 1000000, 30) if operating_cash_flow > 0 else 0
                stability_score = max(0, (1 - debt_to_assets) * 25)
                earnings_score = min(eps * 8, 20) if eps > 0 else 0
                return cash_flow_score + stability_score + earnings_score
                
            return 50  # Default score
            
        except:
            return 0
    
    def _calculate_quality_bonus(self, row):
        """Calculate quality bonus for Small Cap strategies"""
        try:
            debt_to_assets = row.get('debt_to_assets', 1.0)
            gross_margin = row.get('gross_margin', 0)
            roe_estimate = row.get('roe_estimate', 0)
            
            # Quality metrics bonus (0 to 0.15 = 15% max bonus)
            debt_quality = max(0, (0.3 - debt_to_assets) * 0.3) if debt_to_assets <= 0.5 else 0
            margin_quality = max(0, (gross_margin - 0.25) * 0.2) if gross_margin >= 0.20 else 0
            roe_quality = max(0, min(roe_estimate * 0.01, 0.05)) if roe_estimate > 0 else 0
            
            return debt_quality + margin_quality + roe_quality
            
        except:
            return 0
    
    def _calculate_momentum_factor(self, row):
        """Calculate momentum overlay factor"""
        try:
            momentum_21d = row.get('momentum_21d', 0)
            momentum_63d = row.get('momentum_63d', 0)
            
            # Momentum factor (0.90 to 1.15 range)
            momentum_score = momentum_21d * 0.6 + momentum_63d * 0.4
            
            if momentum_score > 0.10:  # Strong positive momentum
                return 1.10
            elif momentum_score > 0.05:  # Moderate positive momentum
                return 1.05
            elif momentum_score > -0.05:  # Neutral
                return 1.00
            elif momentum_score > -0.10:  # Moderate negative momentum
                return 0.97
            else:  # Strong negative momentum
                return 0.93
                
        except:
            return 1.0
    
    def _apply_sector_allocation_limits(self, scores_df):
        """Apply sector allocation limits during portfolio construction"""
        print("Applying sector allocation limits...")
        
        # Count current allocations by sector
        sector_counts = scores_df['sector'].value_counts()
        total_positions = len(scores_df)
        
        # Track positions to keep
        keep_indices = []
        sector_position_counts = {}
        
        for idx, row in scores_df.iterrows():
            sector = row['sector']
            current_sector_count = sector_position_counts.get(sector, 0)
            sector_limit = self.SECTOR_LIMITS.get(sector, 0.30)
            max_positions = int(total_positions * sector_limit)
            
            if current_sector_count < max_positions:
                keep_indices.append(idx)
                sector_position_counts[sector] = current_sector_count + 1
        
        # Filter DataFrame
        limited_df = scores_df.loc[keep_indices].copy()
        
        # Report allocation changes
        print(f"Sector allocation applied:")
        for sector, limit in self.SECTOR_LIMITS.items():
            if sector in sector_position_counts:
                actual_pct = sector_position_counts[sector] / len(limited_df) if len(limited_df) > 0 else 0
                print(f"  {sector}: {actual_pct:.1%} (limit: {limit:.1%})")
        
        return limited_df
    
    def _apply_position_size_limits(self, scores_df):
        """Apply position size limits (equal weight with max 2%)"""
        if len(scores_df) == 0:
            return scores_df
        
        # Calculate position sizes
        total_positions = len(scores_df)
        equal_weight = 1.0 / total_positions
        
        if equal_weight > self.MAX_POSITION_SIZE:
            # Reduce number of positions to meet size limit
            max_positions = int(1.0 / self.MAX_POSITION_SIZE)
            print(f"Position size limit applied: reducing from {total_positions} to {max_positions} positions")
            limited_df = scores_df.head(max_positions).copy()
            limited_df['position_size'] = self.MAX_POSITION_SIZE
        else:
            limited_df = scores_df.copy()
            limited_df['position_size'] = equal_weight
        
        return limited_df
    
    def _display_optimization_results(self, optimized_df, strategy_type, cap_type):
        """Display optimization results summary"""
        if len(optimized_df) == 0:
            print("No optimized results to display")
            return
        
        print(f"\\nOPTIMIZED RESULTS SUMMARY:")
        print(f"Strategy: {strategy_type} | Cap: {cap_type}")
        print(f"Total Positions: {len(optimized_df)}")
        print(f"Average Position Size: {optimized_df['position_size'].mean():.2%}")
        print(f"Average Score: {optimized_df['total_score'].mean():.1f}")
        print(f"Top Score: {optimized_df['total_score'].max():.1f}")
        
        # Sector distribution
        print(f"\\nSector Allocation (Optimized):")
        sector_dist = optimized_df['sector'].value_counts()
        for sector, count in sector_dist.head(8).items():
            pct = count / len(optimized_df)
            limit = self.SECTOR_LIMITS.get(sector, 0.30)
            status = "[OK]" if pct <= limit else "[OVER]"
            print(f"  {sector[:25]}: {count:2d} positions ({pct:.1%}) {status}")
        
        # Top holdings
        print(f"\\nTop 10 Holdings:")
        print("Symbol   Sector                  Score   P/E   Debt%  Margin%")
        print("-" * 60)
        for _, row in optimized_df.head(10).iterrows():
            sector_short = row['sector'][:20] if row['sector'] else 'N/A'
            pe = row.get('pe_ratio', 0)
            debt = row.get('debt_to_assets', 0) * 100
            margin = row.get('gross_margin', 0) * 100
            print(f"{row['symbol']:<8} {sector_short:<20} {row['total_score']:6.1f} {pe:5.1f} {debt:5.1f}% {margin:5.1f}%")

def main():
    """Test optimized funnel scoring"""
    print("OPTIMIZED ENHANCED FUNNEL SCORING SYSTEM")
    print("=" * 70)
    print("Implementing key optimization recommendations:")
    print("+ Sector allocation limits (Manufacturing 35% -> 25%)")
    print("+ Finance sector integration (0% -> target exposure)")
    print("+ Position size limits (max 2%)")
    print("+ Mid Cap performance weighting (+10% multiplier)")
    print("+ Small Cap quality screening")
    print()
    
    scorer = OptimizedFunnelScoring()
    
    # Test optimizations on different strategies
    strategies = [
        ('mid_cap', 'value'),     # Best performing segment
        ('mid_cap', 'growth'),    # Best performing segment
        ('small_cap', 'value'),   # Needs quality screening
        ('large_cap', 'momentum') # Add momentum overlay
    ]
    
    for cap_type, strategy_type in strategies:
        print(f"\\n{'='*70}")
        results = scorer.calculate_optimized_funnel_scores(strategy_type, cap_type)
        if len(results) > 0:
            print(f"[SUCCESS] {cap_type} {strategy_type}: {len(results)} positions generated")
        else:
            print(f"[WARNING] {cap_type} {strategy_type}: No positions generated")

if __name__ == "__main__":
    main()