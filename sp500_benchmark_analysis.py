#!/usr/bin/env python3
"""
S&P 500 Benchmark Performance Analysis
Equal-weighted testing of all 12 strategies against S&P 500 benchmark
Provides objective performance comparison and risk-adjusted metrics
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from optimized_funnel_scoring import OptimizedFunnelScoring
import time

class SP500BenchmarkAnalysis:
    def __init__(self):
        load_dotenv()
        self.engine = create_engine(os.getenv('POSTGRES_URL'))
        self.scorer = OptimizedFunnelScoring()
        
        # Equal weighting for fair comparison - remove optimization bias
        self.EQUAL_WEIGHT_STRATEGIES = [
            ('small_cap', 'value', 'Small Cap Value'),
            ('small_cap', 'growth', 'Small Cap Growth'), 
            ('small_cap', 'momentum', 'Small Cap Momentum'),
            ('small_cap', 'dividend', 'Small Cap Dividend'),
            ('mid_cap', 'value', 'Mid Cap Value'),
            ('mid_cap', 'growth', 'Mid Cap Growth'),
            ('mid_cap', 'momentum', 'Mid Cap Momentum'),
            ('mid_cap', 'dividend', 'Mid Cap Dividend'),
            ('large_cap', 'value', 'Large Cap Value'),
            ('large_cap', 'growth', 'Large Cap Growth'),
            ('large_cap', 'momentum', 'Large Cap Momentum'),
            ('large_cap', 'dividend', 'Large Cap Dividend')
        ]
    
    def get_sp500_benchmark_data(self, start_date='2020-01-01', end_date=None):
        """Get S&P 500 benchmark performance data"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"Loading S&P 500 benchmark data from {start_date} to {end_date}...")
        
        with self.engine.connect() as conn:
            # Try to get SPY data first, fallback to market average if not available
            sp500_query = text(f"""
                WITH sp500_data AS (
                    SELECT 
                        trade_date,
                        adjusted_close as sp500_price,
                        LAG(adjusted_close, 1) OVER (ORDER BY trade_date) as prev_price
                    FROM stock_eod_daily 
                    WHERE symbol = 'SPY' 
                      AND trade_date >= '{start_date}'
                      AND trade_date <= '{end_date}'
                      AND adjusted_close > 0
                    ORDER BY trade_date
                ),
                market_average AS (
                    SELECT 
                        trade_date,
                        AVG(adjusted_close) as market_avg_price,
                        LAG(AVG(adjusted_close), 1) OVER (ORDER BY trade_date) as prev_avg_price
                    FROM stock_eod_daily s
                    JOIN pure_us_stocks p ON s.symbol = p.symbol
                    WHERE s.trade_date >= '{start_date}'
                      AND s.trade_date <= '{end_date}'
                      AND s.adjusted_close > 0
                      AND p.market_cap > 10000000000  -- Large cap proxy
                    GROUP BY trade_date
                    HAVING COUNT(*) >= 100  -- Ensure sufficient stocks
                    ORDER BY trade_date
                )
                SELECT 
                    COALESCE(sp.trade_date, ma.trade_date) as trade_date,
                    COALESCE(sp.sp500_price, ma.market_avg_price) as benchmark_price,
                    CASE 
                        WHEN COALESCE(sp.prev_price, ma.prev_avg_price) > 0 
                        THEN (COALESCE(sp.sp500_price, ma.market_avg_price) - COALESCE(sp.prev_price, ma.prev_avg_price)) / COALESCE(sp.prev_price, ma.prev_avg_price)
                        ELSE 0 
                    END as daily_return,
                    CASE WHEN sp.sp500_price IS NOT NULL THEN 'SPY' ELSE 'Market Average' END as benchmark_type
                FROM sp500_data sp
                FULL OUTER JOIN market_average ma ON sp.trade_date = ma.trade_date
                ORDER BY trade_date
            """)
            
            benchmark_df = pd.read_sql(sp500_query, conn)
            
            if len(benchmark_df) > 0:
                # Calculate cumulative returns
                benchmark_df['cumulative_return'] = (1 + benchmark_df['daily_return']).cumprod()
                
                # Calculate key metrics
                annual_return = benchmark_df['daily_return'].mean() * 252
                volatility = benchmark_df['daily_return'].std() * np.sqrt(252)
                sharpe_ratio = annual_return / volatility if volatility > 0 else 0
                
                print(f"  Benchmark loaded: {len(benchmark_df)} trading days")
                print(f"  Benchmark type: {benchmark_df['benchmark_type'].iloc[-1]}")
                print(f"  Annual return: {annual_return:.1%}")
                print(f"  Volatility: {volatility:.1%}")
                print(f"  Sharpe ratio: {sharpe_ratio:.2f}")
                
                return {
                    'data': benchmark_df,
                    'annual_return': annual_return,
                    'volatility': volatility, 
                    'sharpe_ratio': sharpe_ratio,
                    'total_return': benchmark_df['cumulative_return'].iloc[-1] - 1,
                    'benchmark_type': benchmark_df['benchmark_type'].iloc[-1]
                }
            else:
                print("  [WARNING] No benchmark data available")
                return None
    
    def calculate_strategy_performance(self, cap_type, strategy_type, strategy_name, 
                                     start_date='2020-01-01', end_date=None, portfolio_size=25):
        """Calculate equal-weighted strategy performance for benchmark comparison"""
        print(f"\\nCalculating {strategy_name} performance...")
        
        try:
            # Generate strategy portfolio
            portfolio_df = self.scorer.calculate_optimized_funnel_scores(strategy_type, cap_type)
            
            if len(portfolio_df) == 0:
                print(f"  [WARNING] No portfolio generated for {strategy_name}")
                return None
            
            # Take top positions for equal comparison
            top_portfolio = portfolio_df.head(portfolio_size)
            symbols = top_portfolio['symbol'].tolist()
            
            print(f"  Portfolio: {len(symbols)} stocks")
            print(f"  Top holdings: {', '.join(symbols[:5])}...")
            
            # Get historical performance for these stocks
            with self.engine.connect() as conn:
                if end_date is None:
                    end_date = datetime.now().strftime('%Y-%m-%d')
                
                # Get price data for portfolio stocks
                portfolio_query = text(f"""
                    WITH portfolio_stocks AS (
                        SELECT unnest(ARRAY{symbols}) as symbol
                    ),
                    portfolio_prices AS (
                        SELECT 
                            s.trade_date,
                            s.symbol,
                            s.adjusted_close,
                            LAG(s.adjusted_close, 1) OVER (PARTITION BY s.symbol ORDER BY s.trade_date) as prev_price
                        FROM stock_eod_daily s
                        JOIN portfolio_stocks ps ON s.symbol = ps.symbol
                        WHERE s.trade_date >= '{start_date}'
                          AND s.trade_date <= '{end_date}'
                          AND s.adjusted_close > 0
                    ),
                    daily_returns AS (
                        SELECT 
                            trade_date,
                            symbol,
                            CASE WHEN prev_price > 0 
                                 THEN (adjusted_close - prev_price) / prev_price 
                                 ELSE 0 END as daily_return
                        FROM portfolio_prices
                        WHERE prev_price IS NOT NULL
                    ),
                    portfolio_performance AS (
                        SELECT 
                            trade_date,
                            AVG(daily_return) as portfolio_return,
                            COUNT(*) as stocks_count
                        FROM daily_returns
                        GROUP BY trade_date
                        HAVING COUNT(*) >= {max(3, len(symbols) // 2)}  -- At least half the stocks must have data
                        ORDER BY trade_date
                    )
                    SELECT 
                        trade_date,
                        portfolio_return,
                        stocks_count
                    FROM portfolio_performance
                    ORDER BY trade_date
                """)
                
                performance_df = pd.read_sql(portfolio_query, conn)
                
                if len(performance_df) > 0:
                    # Calculate cumulative performance
                    performance_df['cumulative_return'] = (1 + performance_df['portfolio_return']).cumprod()
                    
                    # Calculate key metrics
                    annual_return = performance_df['portfolio_return'].mean() * 252
                    volatility = performance_df['portfolio_return'].std() * np.sqrt(252)
                    sharpe_ratio = annual_return / volatility if volatility > 0 else 0
                    max_drawdown = self.calculate_max_drawdown(performance_df['cumulative_return'])
                    total_return = performance_df['cumulative_return'].iloc[-1] - 1
                    
                    print(f"  Annual return: {annual_return:.1%}")
                    print(f"  Volatility: {volatility:.1%}")
                    print(f"  Sharpe ratio: {sharpe_ratio:.2f}")
                    print(f"  Max drawdown: {max_drawdown:.1%}")
                    print(f"  Total return: {total_return:.1%}")
                    
                    return {
                        'strategy_name': strategy_name,
                        'data': performance_df,
                        'portfolio_stocks': symbols,
                        'annual_return': annual_return,
                        'volatility': volatility,
                        'sharpe_ratio': sharpe_ratio,
                        'max_drawdown': max_drawdown,
                        'total_return': total_return,
                        'trading_days': len(performance_df)
                    }
                else:
                    print(f"  [WARNING] No performance data available for {strategy_name}")
                    return None
                    
        except Exception as e:
            print(f"  [ERROR] Failed to calculate {strategy_name} performance: {e}")
            return None
    
    def calculate_max_drawdown(self, cumulative_returns):
        """Calculate maximum drawdown"""
        try:
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            return max_drawdown
        except:
            return 0.0
    
    def run_comprehensive_benchmark_analysis(self, start_date='2020-01-01', end_date=None, portfolio_size=25):
        """Run comprehensive benchmark analysis for all strategies"""
        
        print("COMPREHENSIVE S&P 500 BENCHMARK ANALYSIS")
        print("=" * 70)
        print(f"Equal-weighted testing of all 12 strategies")
        print(f"Portfolio size: {portfolio_size} stocks per strategy")
        print(f"Analysis period: {start_date} to {end_date or 'current'}")
        print()
        
        # Get S&P 500 benchmark
        benchmark_data = self.get_sp500_benchmark_data(start_date, end_date)
        
        if not benchmark_data:
            print("[ERROR] Could not load benchmark data")
            return None
        
        print(f"\\n{'='*50}")
        print("INDIVIDUAL STRATEGY ANALYSIS")
        print("="*50)
        
        strategy_results = {}
        successful_strategies = 0
        
        # Test each strategy equally
        for cap_type, strategy_type, strategy_name in self.EQUAL_WEIGHT_STRATEGIES:
            strategy_perf = self.calculate_strategy_performance(
                cap_type, strategy_type, strategy_name, start_date, end_date, portfolio_size
            )
            
            if strategy_perf:
                strategy_results[strategy_name] = strategy_perf
                successful_strategies += 1
        
        print(f"\\n{'='*70}")
        print("BENCHMARK COMPARISON RESULTS")
        print("="*70)
        
        if strategy_results:
            self.generate_benchmark_comparison_report(strategy_results, benchmark_data)
            return {
                'benchmark': benchmark_data,
                'strategies': strategy_results,
                'successful_count': successful_strategies,
                'total_strategies': len(self.EQUAL_WEIGHT_STRATEGIES)
            }
        else:
            print("No strategy results to compare")
            return None
    
    def generate_benchmark_comparison_report(self, strategy_results, benchmark_data):
        """Generate comprehensive benchmark comparison report"""
        
        benchmark_return = benchmark_data['annual_return']
        benchmark_volatility = benchmark_data['volatility']
        benchmark_sharpe = benchmark_data['sharpe_ratio']
        benchmark_total = benchmark_data['total_return']
        
        print(f"\\n[BENCHMARK: {benchmark_data['benchmark_type']}]")
        print(f"  Annual Return: {benchmark_return:.1%}")
        print(f"  Volatility: {benchmark_volatility:.1%}")
        print(f"  Sharpe Ratio: {benchmark_sharpe:.2f}")
        print(f"  Total Return: {benchmark_total:.1%}")
        
        print(f"\\n[STRATEGY PERFORMANCE vs BENCHMARK]")
        print("Strategy                    Ann Ret   Vol    Sharpe  Alpha   Beta   Info Ratio")
        print("-" * 85)
        
        outperforming_strategies = 0
        total_alpha = 0
        strategy_metrics = []
        
        for strategy_name, strategy_data in strategy_results.items():
            annual_ret = strategy_data['annual_return']
            volatility = strategy_data['volatility'] 
            sharpe = strategy_data['sharpe_ratio']
            
            # Calculate alpha (excess return over benchmark)
            alpha = annual_ret - benchmark_return
            
            # Estimate beta (simplified - assume correlation of 0.7 for US equity strategies)
            estimated_correlation = 0.7
            beta = estimated_correlation * (volatility / benchmark_volatility)
            
            # Information ratio (alpha / tracking error)
            tracking_error = abs(volatility - benchmark_volatility)
            info_ratio = alpha / tracking_error if tracking_error > 0 else 0
            
            total_alpha += alpha
            if alpha > 0:
                outperforming_strategies += 1
            
            strategy_metrics.append({
                'name': strategy_name,
                'annual_return': annual_ret,
                'alpha': alpha,
                'sharpe': sharpe,
                'info_ratio': info_ratio,
                'beta': beta
            })
            
            status = "[+]" if alpha > 0 else "[-]"
            print(f"{strategy_name[:26]:<26} {annual_ret:>7.1%} {volatility:>6.1%} {sharpe:>7.2f} "
                  f"{alpha:>6.1%} {beta:>6.2f} {info_ratio:>10.2f} {status}")
        
        # Summary statistics
        avg_alpha = total_alpha / len(strategy_results)
        outperformance_rate = outperforming_strategies / len(strategy_results)
        
        print(f"\\n[SUMMARY STATISTICS]")
        print(f"  Strategies Outperforming Benchmark: {outperforming_strategies}/{len(strategy_results)} ({outperformance_rate:.1%})")
        print(f"  Average Alpha: {avg_alpha:+.1%}")
        print(f"  Benchmark Outperformance Rate: {outperformance_rate:.1%}")
        
        # Best and worst performers
        best_alpha = max(strategy_metrics, key=lambda x: x['alpha'])
        worst_alpha = min(strategy_metrics, key=lambda x: x['alpha'])
        best_sharpe = max(strategy_metrics, key=lambda x: x['sharpe'])
        best_info = max(strategy_metrics, key=lambda x: x['info_ratio'])
        
        print(f"\\n[TOP PERFORMERS]")
        print(f"  Best Alpha: {best_alpha['name']} ({best_alpha['alpha']:+.1%})")
        print(f"  Best Sharpe: {best_sharpe['name']} ({best_sharpe['sharpe']:.2f})")
        print(f"  Best Info Ratio: {best_info['name']} ({best_info['info_ratio']:.2f})")
        print(f"  Worst Alpha: {worst_alpha['name']} ({worst_alpha['alpha']:+.1%})")
        
        # Market cap analysis
        small_cap_alphas = [s['alpha'] for s in strategy_metrics if 'Small Cap' in s['name']]
        mid_cap_alphas = [s['alpha'] for s in strategy_metrics if 'Mid Cap' in s['name']]
        large_cap_alphas = [s['alpha'] for s in strategy_metrics if 'Large Cap' in s['name']]
        
        print(f"\\n[MARKET CAP ANALYSIS]")
        if small_cap_alphas:
            print(f"  Small Cap Average Alpha: {np.mean(small_cap_alphas):+.1%} ({sum(1 for x in small_cap_alphas if x > 0)}/4 outperforming)")
        if mid_cap_alphas:
            print(f"  Mid Cap Average Alpha: {np.mean(mid_cap_alphas):+.1%} ({sum(1 for x in mid_cap_alphas if x > 0)}/4 outperforming)")
        if large_cap_alphas:
            print(f"  Large Cap Average Alpha: {np.mean(large_cap_alphas):+.1%} ({sum(1 for x in large_cap_alphas if x > 0)}/4 outperforming)")
        
        # Strategy type analysis
        value_alphas = [s['alpha'] for s in strategy_metrics if 'Value' in s['name']]
        growth_alphas = [s['alpha'] for s in strategy_metrics if 'Growth' in s['name']]
        momentum_alphas = [s['alpha'] for s in strategy_metrics if 'Momentum' in s['name']]
        dividend_alphas = [s['alpha'] for s in strategy_metrics if 'Dividend' in s['name']]
        
        print(f"\\n[STRATEGY TYPE ANALYSIS]")
        if value_alphas:
            print(f"  Value Average Alpha: {np.mean(value_alphas):+.1%} ({sum(1 for x in value_alphas if x > 0)}/3 outperforming)")
        if growth_alphas:
            print(f"  Growth Average Alpha: {np.mean(growth_alphas):+.1%} ({sum(1 for x in growth_alphas if x > 0)}/3 outperforming)")
        if momentum_alphas:
            print(f"  Momentum Average Alpha: {np.mean(momentum_alphas):+.1%} ({sum(1 for x in momentum_alphas if x > 0)}/3 outperforming)")
        if dividend_alphas:
            print(f"  Dividend Average Alpha: {np.mean(dividend_alphas):+.1%} ({sum(1 for x in dividend_alphas if x > 0)}/3 outperforming)")
        
        # Overall system assessment
        print(f"\\n[SYSTEM ASSESSMENT]")
        if outperformance_rate >= 0.75:
            grade = "EXCELLENT"
        elif outperformance_rate >= 0.60:
            grade = "GOOD"
        elif outperformance_rate >= 0.50:
            grade = "AVERAGE"
        else:
            grade = "NEEDS IMPROVEMENT"
        
        print(f"  Overall Grade: {grade}")
        print(f"  System generates alpha in {outperformance_rate:.1%} of strategies")
        print(f"  Average excess return vs S&P 500: {avg_alpha:+.1%}")
        
        # Risk-adjusted assessment
        avg_sharpe = np.mean([s['sharpe'] for s in strategy_metrics])
        print(f"  Average Strategy Sharpe: {avg_sharpe:.2f}")
        print(f"  vs Benchmark Sharpe: {benchmark_sharpe:.2f}")
        print(f"  Risk-Adjusted Advantage: {avg_sharpe - benchmark_sharpe:+.2f}")

def main():
    """Execute comprehensive S&P 500 benchmark analysis"""
    
    print("[LAUNCH] S&P 500 BENCHMARK ANALYSIS")
    print("Equal-weighted testing of all 12 strategies vs S&P 500")
    print()
    
    analyzer = SP500BenchmarkAnalysis()
    
    # Run analysis with different portfolio sizes for robustness
    portfolio_sizes = [20, 25, 30]  # Test different portfolio sizes
    
    for portfolio_size in portfolio_sizes:
        print(f"\\n{'='*80}")
        print(f"ANALYSIS WITH {portfolio_size}-STOCK PORTFOLIOS")
        print("="*80)
        
        start_time = time.time()
        
        # Run comprehensive analysis
        results = analyzer.run_comprehensive_benchmark_analysis(
            start_date='2020-01-01',  # 4+ year analysis
            end_date=None,
            portfolio_size=portfolio_size
        )
        
        execution_time = time.time() - start_time
        
        if results:
            successful_count = results['successful_count']
            total_strategies = results['total_strategies']
            
            print(f"\\n[ANALYSIS COMPLETE]")
            print(f"  Portfolio Size: {portfolio_size} stocks")
            print(f"  Strategies Analyzed: {successful_count}/{total_strategies}")
            print(f"  Execution Time: {execution_time:.1f} seconds")
            
            if successful_count >= 8:  # At least 2/3 of strategies working
                print(f"  Status: COMPREHENSIVE ANALYSIS SUCCESSFUL")
            else:
                print(f"  Status: PARTIAL ANALYSIS - Some strategies need attention")
        else:
            print(f"\\n[ANALYSIS INCOMPLETE] - Check data availability")
        
        print(f"\\n" + "="*80)
    
    print(f"\\n[BENCHMARK ANALYSIS COMPLETE]")
    print("All strategies have been tested equally against S&P 500 benchmark")
    print("Results provide objective performance comparison")

if __name__ == "__main__":
    main()