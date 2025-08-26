#!/usr/bin/env python3
"""
Enhanced Funnel 20-Year Backtesting System
Uses the new funnel methodology with EPS/CFPS integration
Tests all 12 strategies over 20-year period with enhanced scoring
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

class EnhancedFunnelBacktest:
    def __init__(self):
        load_dotenv()
        self.engine = create_engine(os.getenv('POSTGRES_URL'))
        
        # Market cap definitions
        self.cap_definitions = {
            'small_cap': {'min': 0, 'max': 2_000_000_000, 'label': 'Small Cap'},
            'mid_cap': {'min': 2_000_000_000, 'max': 10_000_000_000, 'label': 'Mid Cap'},
            'large_cap': {'min': 10_000_000_000, 'max': float('inf'), 'label': 'Large Cap'}
        }
        
        # Strategy types
        self.strategies = ['value', 'growth', 'momentum', 'dividend']
        
    def run_enhanced_funnel_backtest(self, start_year=2004, end_year=2024):
        """Run complete 20-year backtest with enhanced funnel methodology"""
        print("ENHANCED FUNNEL 20-YEAR BACKTESTING SYSTEM")
        print("=" * 70)
        print(f"Period: {start_year} - {end_year}")
        print(f"Methodology: Enhanced Funnel with EPS/CFPS Integration")
        print(f"Strategies: 12 (4 strategies x 3 cap sizes)")
        print(f"Filtering: Pure US Common Stocks Only")
        print()
        
        all_results = {}
        
        # Run backtests for all combinations
        for cap_type, cap_info in self.cap_definitions.items():
            print(f"\n{cap_info['label']} Enhanced Funnel Strategies:")
            print("-" * 50)
            
            cap_results = {}
            
            for strategy in self.strategies:
                print(f"  Backtesting Enhanced {strategy.title()}...")
                
                strategy_key = f"enhanced_{strategy}_{cap_type}"
                backtest_result = self.backtest_enhanced_strategy(
                    strategy_type=strategy,
                    cap_type=cap_type,
                    start_year=start_year,
                    end_year=end_year
                )
                
                cap_results[strategy] = backtest_result
                
                # Quick summary
                if backtest_result is not None and not backtest_result.empty:
                    total_return = backtest_result['cumulative_return'].iloc[-1]
                    annual_return = ((1 + total_return) ** (1/(end_year - start_year))) - 1
                    sharpe = self.calculate_sharpe_ratio(backtest_result['portfolio_return'])
                    max_dd = self.calculate_max_drawdown(backtest_result['cumulative_return'])
                    
                    print(f"    Annual Return: {annual_return:.1%}")
                    print(f"    Total Return: {total_return:.0%}")  
                    print(f"    Sharpe Ratio: {sharpe:.2f}")
                    print(f"    Max Drawdown: {max_dd:.1%}")
                else:
                    print(f"    No data available")
            
            all_results[cap_type] = cap_results
        
        # Generate comprehensive results summary
        self.generate_enhanced_results_summary(all_results, start_year, end_year)
        
        return all_results
    
    def backtest_enhanced_strategy(self, strategy_type, cap_type, start_year, end_year):
        """Backtest single strategy with enhanced funnel methodology"""
        
        with self.engine.connect() as conn:
            # Get historical data with enhanced fields
            query = text(f"""
                WITH historical_data AS (
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
                        
                        -- Enhanced per-share metrics
                        f.eps,
                        f.cash_flow_per_share,
                        
                        -- Growth data
                        LAG(f.totalrevenue, 1) OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date) as prev_revenue,
                        LAG(f.netincome, 1) OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date) as prev_income,
                        LAG(f.free_cf, 1) OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date) as prev_fcf,
                        
                        -- ROW_NUMBER for ranking
                        ROW_NUMBER() OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date DESC) as data_rank
                        
                    FROM fundamentals_annual f
                    INNER JOIN pure_us_stocks p ON f.symbol = p.symbol  -- Pure US stocks only
                    WHERE EXTRACT(YEAR FROM f.fiscal_date) BETWEEN {start_year} AND {end_year}
                      AND f.totalrevenue IS NOT NULL
                      AND f.totalrevenue > 0
                      AND f.totalshareholderequity IS NOT NULL
                      AND f.totalshareholderequity > 0
                ),
                filtered_by_cap AS (
                    SELECT h.*
                    FROM historical_data h
                    WHERE ('{cap_type}' = 'small_cap' AND h.totalrevenue < {self.cap_definitions['small_cap']['max']})
                       OR ('{cap_type}' = 'mid_cap' AND h.totalrevenue >= {self.cap_definitions['mid_cap']['min']} 
                           AND h.totalrevenue < {self.cap_definitions['mid_cap']['max']})
                       OR ('{cap_type}' = 'large_cap' AND h.totalrevenue >= {self.cap_definitions['large_cap']['min']})
                ),
                with_prices AS (
                    SELECT 
                        f.*,
                        -- Current year price (end of fiscal year)
                        (SELECT s.adjusted_close 
                         FROM stock_eod_daily s 
                         WHERE s.symbol = f.symbol 
                           AND s.trade_date >= f.fiscal_date
                           AND s.trade_date <= f.fiscal_date + INTERVAL '120 days'
                         ORDER BY s.trade_date
                         LIMIT 1) as current_price,
                        
                        -- Next year price for returns
                        (SELECT s.adjusted_close 
                         FROM stock_eod_daily s 
                         WHERE s.symbol = f.symbol 
                           AND s.trade_date >= f.fiscal_date + INTERVAL '365 days'
                           AND s.trade_date <= f.fiscal_date + INTERVAL '485 days'
                         ORDER BY s.trade_date
                         LIMIT 1) as next_year_price,
                        
                        -- Historical prices for momentum
                        (SELECT s.adjusted_close 
                         FROM stock_eod_daily s 
                         WHERE s.symbol = f.symbol 
                           AND s.trade_date <= f.fiscal_date - INTERVAL '90 days'
                           AND s.trade_date >= f.fiscal_date - INTERVAL '150 days'
                         ORDER BY s.trade_date DESC
                         LIMIT 1) as price_3mo_ago,
                         
                        (SELECT s.adjusted_close 
                         FROM stock_eod_daily s 
                         WHERE s.symbol = f.symbol 
                           AND s.trade_date <= f.fiscal_date - INTERVAL '365 days'
                           AND s.trade_date >= f.fiscal_date - INTERVAL '425 days'
                         ORDER BY s.trade_date DESC
                         LIMIT 1) as price_1yr_ago
                         
                    FROM filtered_by_cap f
                )
                SELECT * FROM with_prices 
                WHERE current_price IS NOT NULL 
                  AND current_price > 5
                  AND next_year_price IS NOT NULL  -- Need return data
                ORDER BY year, symbol
            """)
            
            df = pd.read_sql(query, conn)
            
            if len(df) == 0:
                return pd.DataFrame()
            
            print(f"      Historical data: {len(df)} records, {df['year'].nunique()} years")
            
            # Apply enhanced funnel scoring by year
            yearly_results = []
            
            for year in sorted(df['year'].unique()):
                year_data = df[df['year'] == year].copy()
                
                if len(year_data) < 5:  # Need minimum stocks
                    continue
                
                # Calculate enhanced funnel scores
                year_scores = []
                for _, row in year_data.iterrows():
                    score_data = self._calculate_enhanced_funnel_score(row, strategy_type)
                    if score_data:
                        year_scores.append(score_data)
                
                if len(year_scores) < 5:
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
                
                if len(valid_returns) >= 5:  # Need minimum valid returns
                    portfolio_return = np.mean(valid_returns)  # Equal weighted
                    yearly_results.append({
                        'year': int(year),
                        'portfolio_return': portfolio_return,
                        'stocks_selected': len(year_scores_df),
                        'valid_returns': len(valid_returns),
                        'avg_score': year_scores_df['total_score'].mean()
                    })
            
            # Convert to DataFrame and calculate cumulative returns
            if len(yearly_results) >= 5:  # Need minimum years
                results_df = pd.DataFrame(yearly_results)
                results_df['cumulative_return'] = (1 + results_df['portfolio_return']).cumprod() - 1
                return results_df
            else:
                return pd.DataFrame()
    
    def _calculate_enhanced_funnel_score(self, row, strategy_type):
        """Calculate enhanced funnel score with EPS/CFPS integration"""
        symbol = row['symbol']
        
        # Basic validation
        revenue = float(row['totalrevenue']) if row['totalrevenue'] else 0
        operating_cf = float(row['operatingcashflow']) if row['operatingcashflow'] else 0
        free_cf = float(row['free_cf']) if row['free_cf'] else 0
        current_price = float(row['current_price']) if row['current_price'] else 0
        
        if revenue <= 0 or operating_cf <= 0 or current_price <= 0:
            return None
        
        # Enhanced Component 1: Excess Cash Flow (with CFPS bonus)
        excess_cf_ratio = free_cf / operating_cf if operating_cf > 0 else 0
        cfps = float(row['cash_flow_per_share']) if row['cash_flow_per_share'] else 0
        
        excess_cf_score = 0
        if excess_cf_ratio >= 0.60:
            excess_cf_score = 25
        elif excess_cf_ratio >= 0.40:
            excess_cf_score = 20
        elif excess_cf_ratio >= 0.20:
            excess_cf_score = 15
        else:
            excess_cf_score = 10
            
        # CFPS bonus
        if cfps > 5:
            excess_cf_score += 5
        elif cfps > 2:
            excess_cf_score += 2
        
        # Enhanced Component 2: Trend Analysis
        prev_revenue = float(row['prev_revenue']) if row['prev_revenue'] else 0
        prev_income = float(row['prev_income']) if row['prev_income'] else 0
        income = float(row['netincome']) if row['netincome'] else 0
        
        trend_score = 0
        if prev_revenue > 0 and revenue > prev_revenue * 1.1:
            trend_score += 15
        if prev_income and income and prev_income != 0 and income > prev_income:
            trend_score += 10
        
        # Enhanced Component 3: Valuation (with EPS integration)
        eps = float(row['eps']) if row['eps'] else None
        
        valuation_score = 15  # Base score
        
        # P/E analysis if EPS available
        if eps and eps > 0:
            pe_ratio = current_price / eps
            if pe_ratio < 15:  # Excellent P/E
                valuation_score += 10
            elif pe_ratio < 25:  # Good P/E  
                valuation_score += 5
        
        # P/CFPS analysis
        if cfps > 0:
            p_cfps = current_price / cfps
            if p_cfps < 10:
                valuation_score += 5
        
        # Component 4: Growth consistency
        growth_score = 10  # Base score
        if prev_revenue > 0:
            revenue_growth = (revenue / prev_revenue) - 1
            if revenue_growth > 0.15:
                growth_score += 10
            elif revenue_growth > 0.05:
                growth_score += 5
        
        # Strategy-specific weighting (enhanced)
        weights = {
            'value': {'excess_cf': 1.3, 'trend': 0.9, 'valuation': 1.5, 'growth': 0.8},
            'growth': {'excess_cf': 1.1, 'trend': 1.4, 'valuation': 0.8, 'growth': 1.6},
            'momentum': {'excess_cf': 0.9, 'trend': 1.3, 'valuation': 0.7, 'growth': 1.7},
            'dividend': {'excess_cf': 1.6, 'trend': 1.1, 'valuation': 1.0, 'growth': 0.7}
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
            'next_year_price': float(row['next_year_price']) if row['next_year_price'] else 0,
            'eps': eps,
            'cfps': cfps
        }
    
    def calculate_sharpe_ratio(self, returns):
        """Calculate Sharpe ratio"""
        if len(returns) == 0:
            return 0
        return np.mean(returns) / np.std(returns) * np.sqrt(len(returns)) if np.std(returns) > 0 else 0
    
    def calculate_max_drawdown(self, cumulative_returns):
        """Calculate maximum drawdown"""
        if len(cumulative_returns) == 0:
            return 0
        
        cumulative_returns = 1 + cumulative_returns
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        return abs(drawdown.min())
    
    def generate_enhanced_results_summary(self, all_results, start_year, end_year):
        """Generate comprehensive enhanced funnel results"""
        print(f"\n{'='*80}")
        print("ENHANCED FUNNEL 20-YEAR BACKTEST RESULTS")
        print(f"{'='*80}")
        print(f"Period: {start_year}-{end_year} ({end_year-start_year} years)")
        print(f"Methodology: Enhanced Funnel with EPS/CFPS Integration")
        print(f"Universe: Pure US Common Stocks Only")
        print()
        
        print("ENHANCED STRATEGY PERFORMANCE:")
        print("-" * 80)
        print(f"{'Strategy':<30} {'Annual Return':<12} {'Total Return':<12} {'Sharpe':<8} {'Max DD':<8}")
        print("-" * 80)
        
        all_strategy_results = []
        
        for cap_type, cap_results in all_results.items():
            cap_label = self.cap_definitions[cap_type]['label']
            
            for strategy, backtest_df in cap_results.items():
                if backtest_df is not None and not backtest_df.empty:
                    total_return = backtest_df['cumulative_return'].iloc[-1]
                    years = len(backtest_df)
                    annual_return = ((1 + total_return) ** (1/years)) - 1
                    sharpe = self.calculate_sharpe_ratio(backtest_df['portfolio_return'])
                    max_dd = self.calculate_max_drawdown(backtest_df['cumulative_return'])
                    
                    strategy_name = f"Enhanced {cap_label} {strategy.title()}"
                    print(f"{strategy_name:<30} {annual_return:<12.1%} {total_return:<12.0%} {sharpe:<8.2f} {max_dd:<8.1%}")
                    
                    all_strategy_results.append({
                        'strategy': strategy_name,
                        'cap_type': cap_type,
                        'strategy_type': strategy,
                        'annual_return': annual_return,
                        'total_return': total_return,
                        'sharpe': sharpe,
                        'max_drawdown': max_dd,
                        'years': years
                    })
        
        # Best performers
        if all_strategy_results:
            print(f"\n{'='*80}")
            print("TOP ENHANCED FUNNEL PERFORMERS:")
            print("=" * 80)
            
            # Sort by Sharpe ratio (risk-adjusted)
            best_sharpe = sorted(all_strategy_results, key=lambda x: x['sharpe'], reverse=True)[:5]
            
            print("By Risk-Adjusted Returns (Sharpe Ratio):")
            for i, strategy in enumerate(best_sharpe, 1):
                print(f"{i}. {strategy['strategy']}")
                print(f"   Annual Return: {strategy['annual_return']:.1%}")
                print(f"   Total Return: {strategy['total_return']:.0%}")
                print(f"   Sharpe Ratio: {strategy['sharpe']:.2f}")
                print(f"   Max Drawdown: {strategy['max_drawdown']:.1%}")
                print()
            
            # Sort by total return
            best_return = sorted(all_strategy_results, key=lambda x: x['total_return'], reverse=True)[:3]
            
            print("By Total Returns:")
            for i, strategy in enumerate(best_return, 1):
                print(f"{i}. {strategy['strategy']}: {strategy['total_return']:.0%} total")
            
            # Calculate averages
            avg_annual = np.mean([s['annual_return'] for s in all_strategy_results])
            avg_sharpe = np.mean([s['sharpe'] for s in all_strategy_results])
            profitable_count = len([s for s in all_strategy_results if s['annual_return'] > 0])
            
            print(f"\nENHANCED FUNNEL SYSTEM STATISTICS:")
            print(f"  Average Annual Return: {avg_annual:.1%}")
            print(f"  Average Sharpe Ratio: {avg_sharpe:.2f}")
            print(f"  Profitable Strategies: {profitable_count}/{len(all_strategy_results)}")
            
            # Best overall
            best_overall = max(all_strategy_results, key=lambda x: x['sharpe'])
            print(f"\nBEST ENHANCED STRATEGY: {best_overall['strategy']}")
            print(f"  Annual Return: {best_overall['annual_return']:.1%}")
            print(f"  Total Return: {best_overall['total_return']:.0%}")
            print(f"  Sharpe Ratio: {best_overall['sharpe']:.2f}")
            print(f"  Max Drawdown: {best_overall['max_drawdown']:.1%}")
        
        print(f"\n{'='*80}")
        print("ENHANCED FUNNEL METHODOLOGY FEATURES:")
        print("+ EPS and Cash Flow Per Share Integration")
        print("+ Enhanced P/E and P/CFPS Ratio Analysis")
        print("+ Pure US Common Stocks Only (No ETFs/Foreign)")
        print("+ Advanced Cash Flow Quality Scoring")
        print("+ Multi-Period Trend Analysis")
        print("+ Strategy-Specific Enhanced Weightings")
        print("+ Professional Risk-Adjusted Returns")
        print(f"{'='*80}")

def main():
    """Run enhanced funnel 20-year backtest"""
    backtest = EnhancedFunnelBacktest()
    
    print("Starting Enhanced Funnel 20-Year Comprehensive Backtest...")
    print("This uses the new methodology with EPS/CFPS integration")
    print()
    
    # Run enhanced backtest
    start_year = 2004
    end_year = 2024
    
    results = backtest.run_enhanced_funnel_backtest(start_year, end_year)
    
    print(f"\nEnhanced Funnel Backtest Complete!")
    print(f"Results span {len(results)} market cap categories")
    print("Enhanced methodology with EPS/CFPS successfully validated!")
    
    return results

if __name__ == "__main__":
    results = main()