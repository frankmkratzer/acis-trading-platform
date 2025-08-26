#!/usr/bin/env python3
"""
Comprehensive Funnel-Based 12-Strategy Backtesting System
Tests all 12 strategies (Small/Mid/Large Cap x Value/Growth/Momentum/Dividend)
with enhanced funnel methodology over 20-year period
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from enhanced_funnel_scoring import EnhancedFunnelScoring

class ComprehensiveFunnelBacktest:
    def __init__(self):
        load_dotenv()
        self.engine = create_engine(os.getenv('POSTGRES_URL'))
        self.scorer = EnhancedFunnelScoring()
        
        # Market cap definitions
        self.cap_definitions = {
            'small_cap': {'min': 0, 'max': 2_000_000_000, 'label': 'Small Cap'},
            'mid_cap': {'min': 2_000_000_000, 'max': 10_000_000_000, 'label': 'Mid Cap'},
            'large_cap': {'min': 10_000_000_000, 'max': float('inf'), 'label': 'Large Cap'}
        }
        
        # Strategy types
        self.strategies = ['value', 'growth', 'momentum', 'dividend']
        
    def run_comprehensive_backtest(self, start_year=2004, end_year=2024):
        """Run complete backtesting for all 12 strategies"""
        print("COMPREHENSIVE FUNNEL-BASED 12-STRATEGY BACKTEST")
        print("=" * 70)
        print(f"Period: {start_year} - {end_year}")
        print(f"Strategies: 12 (4 strategies x 3 cap sizes)")
        print()
        
        results = {}
        
        # Run backtests for all combinations
        for cap_type, cap_info in self.cap_definitions.items():
            print(f"\n{cap_info['label']} Strategies:")
            print("-" * 40)
            
            cap_results = {}
            
            for strategy in self.strategies:
                print(f"  Testing {strategy.title()}...")
                
                strategy_key = f"{strategy}_{cap_type}"
                backtest_result = self.backtest_strategy(
                    strategy_type=strategy,
                    cap_type=cap_type,
                    start_year=start_year,
                    end_year=end_year
                )
                
                cap_results[strategy] = backtest_result
                
                # Quick summary
                if backtest_result and len(backtest_result) > 0:
                    total_return = backtest_result['cumulative_return'].iloc[-1] if 'cumulative_return' in backtest_result.columns else 0
                    annual_return = ((1 + total_return) ** (1/(end_year - start_year))) - 1
                    print(f"    Annual Return: {annual_return:.1%}")
                else:
                    print(f"    No data available")
            
            results[cap_type] = cap_results
        
        # Generate comprehensive results summary
        self.generate_results_summary(results, start_year, end_year)
        
        return results
    
    def backtest_strategy(self, strategy_type, cap_type, start_year, end_year):
        """Backtest a single strategy over the specified period"""
        
        with self.engine.connect() as conn:
            # Get historical data for backtesting
            query = text(f"""
                WITH historical_fundamentals AS (
                    SELECT 
                        f.symbol,
                        f.fiscal_date,
                        EXTRACT(YEAR FROM f.fiscal_date) as year,
                        f.totalrevenue,
                        f.netincome,
                        f.operatingcashflow,
                        f.free_cf,
                        f.totalassets,
                        f.totalshareholderequity,
                        f.totalliabilities,
                        f.dividendpayout,
                        
                        -- Enhanced fields
                        f.cashandcashequivalentsatcarryingvalue,
                        f.commonstocksharesoutstanding,
                        
                        -- Lag data for trends
                        LAG(f.totalrevenue, 1) OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date) as prev_revenue,
                        LAG(f.netincome, 1) OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date) as prev_income,
                        LAG(f.free_cf, 1) OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date) as prev_fcf
                        
                    FROM fundamentals_annual f
                    WHERE EXTRACT(YEAR FROM f.fiscal_date) BETWEEN {start_year} AND {end_year}
                      AND f.totalrevenue IS NOT NULL
                      AND f.totalrevenue > 0
                      AND f.totalshareholderequity IS NOT NULL
                      AND f.totalshareholderequity > 0
                ),
                filtered_by_cap AS (
                    SELECT h.*
                    FROM historical_fundamentals h
                    WHERE ('{cap_type}' = 'small_cap' AND h.totalrevenue < {self.cap_definitions['small_cap']['max']})
                       OR ('{cap_type}' = 'mid_cap' AND h.totalrevenue >= {self.cap_definitions['mid_cap']['min']} 
                           AND h.totalrevenue < {self.cap_definitions['mid_cap']['max']})
                       OR ('{cap_type}' = 'large_cap' AND h.totalrevenue >= {self.cap_definitions['large_cap']['min']})
                ),
                with_prices AS (
                    SELECT 
                        f.*,
                        -- Current year price
                        (SELECT s.adjusted_close 
                         FROM stock_eod_daily s 
                         WHERE s.symbol = f.symbol 
                           AND EXTRACT(YEAR FROM s.trade_date) = f.year
                         ORDER BY s.trade_date DESC 
                         LIMIT 1) as current_price,
                        
                        -- Next year price for returns
                        (SELECT s.adjusted_close 
                         FROM stock_eod_daily s 
                         WHERE s.symbol = f.symbol 
                           AND EXTRACT(YEAR FROM s.trade_date) = f.year + 1
                         ORDER BY s.trade_date DESC 
                         LIMIT 1) as next_year_price,
                        
                        -- Historical prices for momentum/valuation
                        (SELECT s.adjusted_close 
                         FROM stock_eod_daily s 
                         WHERE s.symbol = f.symbol 
                           AND s.trade_date <= f.fiscal_date - INTERVAL '365 days'
                         ORDER BY s.trade_date DESC 
                         LIMIT 1) as price_1yr_ago
                         
                    FROM filtered_by_cap f
                )
                SELECT * FROM with_prices 
                WHERE current_price IS NOT NULL 
                  AND current_price > 5
                ORDER BY year, symbol
            """)
            
            df = pd.read_sql(query, conn)
            
            if len(df) == 0:
                return pd.DataFrame()
            
            print(f"    Historical data: {len(df)} records over {df['year'].nunique()} years")
            
            # Apply funnel scoring by year
            yearly_results = []
            
            for year in sorted(df['year'].unique()):
                year_data = df[df['year'] == year].copy()
                
                if len(year_data) < 10:  # Need minimum stocks
                    continue
                
                # Calculate funnel scores for this year
                year_scores = []
                for _, row in year_data.iterrows():
                    score_data = self._calculate_historical_funnel_score(row, strategy_type)
                    if score_data:
                        year_scores.append(score_data)
                
                if len(year_scores) < 10:
                    continue
                
                # Rank and select top 10
                year_scores_df = pd.DataFrame(year_scores)
                year_scores_df = year_scores_df.sort_values('total_score', ascending=False).head(10)
                
                # Calculate portfolio return
                valid_returns = []
                for _, stock in year_scores_df.iterrows():
                    current_price = stock.get('current_price', 0)
                    next_price = stock.get('next_year_price', 0)
                    
                    if current_price > 0 and next_price > 0:
                        stock_return = (next_price / current_price) - 1
                        valid_returns.append(stock_return)
                
                if valid_returns:
                    portfolio_return = np.mean(valid_returns)  # Equal weighted
                    yearly_results.append({
                        'year': int(year),
                        'portfolio_return': portfolio_return,
                        'stocks_selected': len(year_scores_df),
                        'valid_returns': len(valid_returns),
                        'avg_score': year_scores_df['total_score'].mean()
                    })
            
            # Convert to DataFrame and calculate cumulative returns
            if yearly_results:
                results_df = pd.DataFrame(yearly_results)
                results_df['cumulative_return'] = (1 + results_df['portfolio_return']).cumprod() - 1
                return results_df
            else:
                return pd.DataFrame()
    
    def _calculate_historical_funnel_score(self, row, strategy_type):
        """Calculate funnel score for historical data point"""
        # Simplified version for backtesting - similar to main funnel but adapted for historical data
        symbol = row['symbol']
        
        # Basic validation
        revenue = float(row['totalrevenue']) if row['totalrevenue'] else 0
        operating_cf = float(row['operatingcashflow']) if row['operatingcashflow'] else 0
        free_cf = float(row['free_cf']) if row['free_cf'] else 0
        
        if revenue <= 0 or operating_cf <= 0:
            return None
        
        # Component 1: Excess Cash Flow
        excess_cf_ratio = free_cf / operating_cf if operating_cf > 0 else 0
        if excess_cf_ratio >= 0.60:
            excess_cf_score = 25
        elif excess_cf_ratio >= 0.40:
            excess_cf_score = 20
        elif excess_cf_ratio >= 0.20:
            excess_cf_score = 15
        else:
            excess_cf_score = 10
        
        # Component 2: Trend (simplified - year-over-year)
        prev_revenue = float(row['prev_revenue']) if row['prev_revenue'] else 0
        trend_score = 0
        if prev_revenue > 0 and revenue > prev_revenue * 1.1:
            trend_score += 15
        if prev_revenue > 0 and revenue > prev_revenue:
            trend_score += 10
        
        # Component 3: Valuation (simplified)
        current_price = float(row['current_price']) if row['current_price'] else 0
        price_1yr_ago = float(row['price_1yr_ago']) if row['price_1yr_ago'] else 0
        
        valuation_score = 15  # Base score
        if price_1yr_ago > 0 and current_price < price_1yr_ago * 0.8:  # 20% below year ago
            valuation_score = 25
        elif price_1yr_ago > 0 and current_price < price_1yr_ago:
            valuation_score = 20
        
        # Component 4: Growth (simplified)
        growth_score = 15  # Base score
        if prev_revenue > 0:
            growth_rate = (revenue / prev_revenue) - 1
            if growth_rate > 0.15:
                growth_score = 20
            elif growth_rate > 0.05:
                growth_score = 18
        
        # Strategy-specific weighting
        weights = {
            'value': {'excess_cf': 1.2, 'trend': 0.9, 'valuation': 1.4, 'growth': 0.8},
            'growth': {'excess_cf': 1.1, 'trend': 1.3, 'valuation': 0.8, 'growth': 1.4},
            'momentum': {'excess_cf': 0.9, 'trend': 1.2, 'valuation': 0.7, 'growth': 1.5},
            'dividend': {'excess_cf': 1.5, 'trend': 1.1, 'valuation': 1.0, 'growth': 0.7}
        }.get(strategy_type, {'excess_cf': 1.0, 'trend': 1.0, 'valuation': 1.0, 'growth': 1.0})
        
        total_score = (
            excess_cf_score * weights['excess_cf'] +
            trend_score * weights['trend'] +
            valuation_score * weights['valuation'] +
            growth_score * weights['growth']
        )
        
        return {
            'symbol': symbol,
            'total_score': total_score,
            'current_price': current_price,
            'next_year_price': float(row['next_year_price']) if row['next_year_price'] else 0
        }
    
    def generate_results_summary(self, results, start_year, end_year):
        """Generate comprehensive results summary"""
        print(f"\n{'='*70}")
        print("COMPREHENSIVE BACKTEST RESULTS SUMMARY")
        print(f"{'='*70}")
        
        print(f"\nStrategy Performance ({start_year}-{end_year}):")
        print("-" * 70)
        print(f"{'Strategy':<25} {'Annual Return':<15} {'Total Return':<15} {'Years':<8}")
        print("-" * 70)
        
        strategy_summaries = []
        
        for cap_type, cap_results in results.items():
            cap_label = self.cap_definitions[cap_type]['label']
            
            for strategy, backtest_df in cap_results.items():
                if len(backtest_df) > 0:
                    total_return = backtest_df['cumulative_return'].iloc[-1]
                    years = len(backtest_df)
                    annual_return = ((1 + total_return) ** (1/years)) - 1
                    
                    strategy_name = f"{cap_label} {strategy.title()}"
                    print(f"{strategy_name:<25} {annual_return:<15.1%} {total_return:<15.1%} {years:<8}")
                    
                    strategy_summaries.append({
                        'strategy': strategy_name,
                        'annual_return': annual_return,
                        'total_return': total_return,
                        'years': years
                    })
                else:
                    strategy_name = f"{cap_label} {strategy.title()}"
                    print(f"{strategy_name:<25} {'No Data':<15} {'No Data':<15} {'0':<8}")
        
        # Best/Worst performers
        if strategy_summaries:
            best_annual = max(strategy_summaries, key=lambda x: x['annual_return'])
            worst_annual = min(strategy_summaries, key=lambda x: x['annual_return'])
            
            print(f"\nTop Performer: {best_annual['strategy']}")
            print(f"  Annual Return: {best_annual['annual_return']:.1%}")
            print(f"  Total Return: {best_annual['total_return']:.1%}")
            
            print(f"\nWorst Performer: {worst_annual['strategy']}")
            print(f"  Annual Return: {worst_annual['annual_return']:.1%}")
            print(f"  Total Return: {worst_annual['total_return']:.1%}")
            
            # Overall statistics
            avg_annual = np.mean([s['annual_return'] for s in strategy_summaries])
            print(f"\nAverage Annual Return: {avg_annual:.1%}")
            print(f"Number of Profitable Strategies: {len([s for s in strategy_summaries if s['annual_return'] > 0])}/12")
        
        print(f"\n{'='*70}")
        print("FUNNEL-BASED 12-STRATEGY BACKTEST COMPLETE!")
        print(f"{'='*70}")

def main():
    """Run the comprehensive funnel backtest"""
    backtest = ComprehensiveFunnelBacktest()
    
    # Check if we have sufficient data
    with backtest.engine.connect() as conn:
        result = conn.execute(text("""
            SELECT 
                COUNT(DISTINCT symbol) as symbols,
                MIN(EXTRACT(YEAR FROM fiscal_date)) as earliest_year,
                MAX(EXTRACT(YEAR FROM fiscal_date)) as latest_year
            FROM fundamentals_annual
        """))
        
        data_stats = result.fetchone()
        
        if not data_stats or data_stats[0] < 100:
            print("Insufficient fundamental data for backtesting.")
            print("Please run fetch_enhanced_fundamentals_20yr.py first.")
            return
        
        print(f"Available data: {data_stats[0]} symbols")
        print(f"Date range: {data_stats[1]} - {data_stats[2]}")
    
    # Run comprehensive backtest
    start_year = 2004
    end_year = 2024
    
    results = backtest.run_comprehensive_backtest(start_year, end_year)
    
    print(f"\nBacktest complete! Results cover {len(results)} cap categories")
    
    return results

if __name__ == "__main__":
    results = main()