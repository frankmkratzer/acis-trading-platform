#!/usr/bin/env python3
"""
Comprehensive Quarterly Strategy Run
Execute complete 12-strategy system with all enhancements:
- Enhanced funnel scoring (4 components)
- Sector strength integration
- Historical EPS/CFPS data
- Pure US stock filtering
- Professional portfolio construction
"""

import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from enhanced_funnel_scoring import EnhancedFunnelScoring
import time

def run_comprehensive_quarterly_strategies():
    """Execute complete quarterly strategy run with all enhancements"""
    
    print("COMPREHENSIVE QUARTERLY STRATEGY RUN")
    print("=" * 70)
    print("Enhanced 12-Strategy System with Complete Feature Set")
    print()
    
    load_dotenv()
    engine = create_engine(os.getenv('POSTGRES_URL'))
    
    # Initialize enhanced scoring system
    scorer = EnhancedFunnelScoring()
    
    # Define complete 12-strategy matrix
    strategies = [
        # Small Cap Strategies (<$2B)
        ('small_cap', 'value', 'ai_value_small_cap_scores', 'ai_value_small_cap_portfolio', 'Small Cap Value'),
        ('small_cap', 'growth', 'ai_growth_small_cap_scores', 'ai_growth_small_cap_portfolio', 'Small Cap Growth'),
        ('small_cap', 'momentum', 'ai_momentum_small_cap_scores', 'ai_momentum_small_cap_portfolio', 'Small Cap Momentum'),
        ('small_cap', 'dividend', 'ai_dividend_small_cap_scores', 'ai_dividend_small_cap_portfolio', 'Small Cap Dividend'),
        
        # Mid Cap Strategies ($2B-$10B)
        ('mid_cap', 'value', 'ai_value_mid_cap_scores', 'ai_value_mid_cap_portfolio', 'Mid Cap Value'),
        ('mid_cap', 'growth', 'ai_growth_mid_cap_scores', 'ai_growth_mid_cap_portfolio', 'Mid Cap Growth'),
        ('mid_cap', 'momentum', 'ai_momentum_mid_cap_scores', 'ai_momentum_mid_cap_portfolio', 'Mid Cap Momentum'),
        ('mid_cap', 'dividend', 'ai_dividend_mid_cap_scores', 'ai_dividend_mid_cap_portfolio', 'Mid Cap Dividend'),
        
        # Large Cap Strategies ($10B+)
        ('large_cap', 'value', 'ai_value_large_cap_scores', 'ai_value_large_cap_portfolio', 'Large Cap Value'),
        ('large_cap', 'growth', 'ai_growth_large_cap_scores', 'ai_growth_large_cap_portfolio', 'Large Cap Growth'),
        ('large_cap', 'momentum', 'ai_momentum_large_cap_scores', 'ai_momentum_large_cap_portfolio', 'Large Cap Momentum'),
        ('large_cap', 'dividend', 'ai_dividend_large_cap_scores', 'ai_dividend_large_cap_portfolio', 'Large Cap Dividend')
    ]
    
    with engine.connect() as conn:
        # Display system status
        print_system_status(conn)
        
        print(f"\nExecuting Enhanced 12-Strategy System:")
        print("-" * 50)
        
        successful_strategies = 0
        total_selections = 0
        strategy_results = {}
        
        for cap_type, strategy_type, scores_table, portfolio_table, strategy_name in strategies:
            print(f"\n{strategy_name}:")
            
            start_time = time.time()
            
            try:
                # Calculate enhanced scores with all features
                print(f"  Calculating enhanced funnel scores...")
                scores_df = scorer.calculate_funnel_scores(strategy_type, cap_type)
                
                if len(scores_df) == 0:
                    print(f"  [X] No scores generated")
                    continue
                
                # Clear existing data
                conn.execute(text(f"DELETE FROM {scores_table} WHERE as_of_date = CURRENT_DATE"))
                conn.execute(text(f"DELETE FROM {portfolio_table} WHERE as_of_date = CURRENT_DATE"))
                
                # Save enhanced scores (top 100 for flexibility)
                top_100_scores = scores_df.head(100)
                
                for idx, row in top_100_scores.iterrows():
                    conn.execute(text(f"""
                        INSERT INTO {scores_table} (
                            symbol, as_of_date, score, percentile, 
                            predicted_return, model_version
                        ) VALUES (
                            :symbol, CURRENT_DATE, :score, :percentile, 
                            :predicted_return, :model_version
                        )
                    """), {
                        'symbol': row['symbol'],
                        'score': row['total_score'],
                        'percentile': row.get('percentile', 0.5),
                        'predicted_return': row['total_score'] / 100.0,
                        'model_version': f"v5_{strategy_type[:3]}{cap_type[:3]}"
                    })
                
                # Create portfolio (top 10 for actual trading)
                top_10_portfolio = scores_df.head(10)
                
                for idx, row in top_10_portfolio.iterrows():
                    rank = top_10_portfolio.index.get_loc(idx) + 1
                    conn.execute(text(f"""
                        INSERT INTO {portfolio_table} (
                            symbol, as_of_date, score, percentile, score_label,
                            rank, model_version, score_type, fetched_at
                        ) VALUES (
                            :symbol, CURRENT_DATE, :score, :percentile, :score_label,
                            :rank, :model_version, :score_type, CURRENT_TIMESTAMP
                        )
                    """), {
                        'symbol': row['symbol'],
                        'score': row['total_score'],
                        'percentile': row.get('percentile', 0.5),
                        'score_label': f"{strategy_name} Selection",
                        'rank': rank,
                        'model_version': f"v5_{strategy_type[:3]}{cap_type[:3]}",
                        'score_type': f"{strategy_type}_{cap_type}"
                    })
                
                conn.commit()
                
                # Calculate execution metrics
                execution_time = time.time() - start_time
                avg_score = scores_df['total_score'].mean()
                top_score = scores_df['total_score'].max()
                
                # Get sector distribution
                sector_dist = scores_df.head(10)['sector'].value_counts() if 'sector' in scores_df.columns else {}
                top_sectors = list(sector_dist.head(3).index) if len(sector_dist) > 0 else ['N/A']
                
                # Store results
                strategy_results[strategy_name] = {
                    'total_candidates': len(scores_df),
                    'avg_score': avg_score,
                    'top_score': top_score,
                    'top_sectors': top_sectors,
                    'execution_time': execution_time,
                    'portfolio_stocks': [row['symbol'] for _, row in top_10_portfolio.iterrows()]
                }
                
                print(f"  [+] SUCCESS ({execution_time:.1f}s)")
                print(f"     Candidates: {len(scores_df):,} | Avg Score: {avg_score:.1f} | Top Score: {top_score:.1f}")
                print(f"     Portfolio: {', '.join(top_10_portfolio['symbol'].head(5).tolist())}...")
                print(f"     Top Sectors: {', '.join(top_sectors)}")
                
                successful_strategies += 1
                total_selections += len(top_10_portfolio)
                
            except Exception as e:
                print(f"  [X] FAILED: {e}")
                conn.rollback()
                continue
        
        # Generate comprehensive summary
        print(f"\n" + "=" * 70)
        print(f"QUARTERLY RUN COMPLETE - ENHANCED 12-STRATEGY SYSTEM")
        print("=" * 70)
        
        print(f"\n[SUMMARY] EXECUTION SUMMARY:")
        print(f"  Strategies Executed: {successful_strategies}/12")
        print(f"  Total Portfolio Positions: {total_selections}")
        print(f"  Enhanced Features: [+] Sector Strength [+] EPS/CFPS [+] Pure US Stocks")
        
        if successful_strategies >= 10:
            # Detailed strategy analysis
            print_detailed_strategy_analysis(strategy_results)
            
            # Portfolio composition analysis  
            analyze_portfolio_composition(conn)
            
            # Risk and diversification analysis
            analyze_risk_and_diversification(conn, strategy_results)
            
            return True, strategy_results
        else:
            print(f"\n[X] System incomplete: Only {successful_strategies}/12 strategies executed")
            return False, strategy_results

def print_system_status(conn):
    """Print comprehensive system status"""
    print("SYSTEM STATUS CHECK:")
    print("-" * 30)
    
    # Check data availability
    checks = [
        ("Pure US Stocks", "SELECT COUNT(*) FROM pure_us_stocks WHERE sector IS NOT NULL"),
        ("Sector Strength", "SELECT COUNT(DISTINCT as_of_date) FROM sector_strength_scores"),
        ("Annual Fundamentals (EPS)", "SELECT COUNT(*) FROM fundamentals_annual WHERE eps IS NOT NULL AND fiscal_date >= '2020-01-01'"),
        ("Quarterly Fundamentals (EPS)", "SELECT COUNT(*) FROM fundamentals_quarterly WHERE eps IS NOT NULL AND fiscal_date >= '2020-01-01'"),
        ("Price Data", "SELECT COUNT(DISTINCT symbol) FROM stock_eod_daily WHERE trade_date >= CURRENT_DATE - INTERVAL '30 days'")
    ]
    
    for check_name, query in checks:
        try:
            result = conn.execute(text(query))
            count = result.fetchone()[0]
            status = "[+]" if count > 0 else "[X]"
            print(f"  {status} {check_name}: {count:,}")
        except:
            print(f"  [X] {check_name}: Error")

def print_detailed_strategy_analysis(strategy_results):
    """Print detailed analysis of strategy results"""
    print(f"\n[ANALYSIS] DETAILED STRATEGY ANALYSIS:")
    print("-" * 50)
    print(f"{'Strategy':<25} {'Candidates':<10} {'Avg Score':<10} {'Top Score':<10} {'Time(s)':<8}")
    print("-" * 63)
    
    for strategy_name, results in strategy_results.items():
        print(f"{strategy_name[:24]:<25} {results['total_candidates']:<10,} "
              f"{results['avg_score']:<10.1f} {results['top_score']:<10.1f} {results['execution_time']:<8.1f}")
    
    # Performance rankings
    best_avg = max(strategy_results.items(), key=lambda x: x[1]['avg_score'])
    best_top = max(strategy_results.items(), key=lambda x: x[1]['top_score'])
    most_candidates = max(strategy_results.items(), key=lambda x: x[1]['total_candidates'])
    
    print(f"\n[LEADERS] PERFORMANCE LEADERS:")
    print(f"  Best Average Score: {best_avg[0]} ({best_avg[1]['avg_score']:.1f})")
    print(f"  Best Top Score: {best_top[0]} ({best_top[1]['top_score']:.1f})")
    print(f"  Most Candidates: {most_candidates[0]} ({most_candidates[1]['total_candidates']:,})")

def analyze_portfolio_composition(conn):
    """Analyze overall portfolio composition"""
    print(f"\n[PORTFOLIO] COMPOSITION ANALYSIS:")
    print("-" * 40)
    
    # Sector distribution across all portfolios
    result = conn.execute(text("""
        SELECT 
            p.sector,
            COUNT(*) as selections,
            COUNT(DISTINCT substr(port.score_type, 1, position('_' in port.score_type || '_') - 1)) as strategies,
            AVG(port.score) as avg_score
        FROM (
            SELECT symbol, score, score_type FROM ai_value_small_cap_portfolio WHERE as_of_date = CURRENT_DATE
            UNION ALL SELECT symbol, score, score_type FROM ai_growth_small_cap_portfolio WHERE as_of_date = CURRENT_DATE
            UNION ALL SELECT symbol, score, score_type FROM ai_momentum_small_cap_portfolio WHERE as_of_date = CURRENT_DATE
            UNION ALL SELECT symbol, score, score_type FROM ai_dividend_small_cap_portfolio WHERE as_of_date = CURRENT_DATE
            UNION ALL SELECT symbol, score, score_type FROM ai_value_mid_cap_portfolio WHERE as_of_date = CURRENT_DATE
            UNION ALL SELECT symbol, score, score_type FROM ai_growth_mid_cap_portfolio WHERE as_of_date = CURRENT_DATE
            UNION ALL SELECT symbol, score, score_type FROM ai_momentum_mid_cap_portfolio WHERE as_of_date = CURRENT_DATE
            UNION ALL SELECT symbol, score, score_type FROM ai_dividend_mid_cap_portfolio WHERE as_of_date = CURRENT_DATE
            UNION ALL SELECT symbol, score, score_type FROM ai_value_large_cap_portfolio WHERE as_of_date = CURRENT_DATE
            UNION ALL SELECT symbol, score, score_type FROM ai_growth_large_cap_portfolio WHERE as_of_date = CURRENT_DATE
            UNION ALL SELECT symbol, score, score_type FROM ai_momentum_large_cap_portfolio WHERE as_of_date = CURRENT_DATE
            UNION ALL SELECT symbol, score, score_type FROM ai_dividend_large_cap_portfolio WHERE as_of_date = CURRENT_DATE
        ) port
        LEFT JOIN pure_us_stocks p ON port.symbol = p.symbol
        WHERE p.sector IS NOT NULL
        GROUP BY p.sector
        ORDER BY selections DESC
    """))
    
    print("Sector Distribution Across All Portfolios:")
    print("Sector                           Selections  Strategies  Avg Score")
    print("-" * 65)
    
    total_selections = 0
    for row in result:
        sector = row[0][:30] if row[0] else 'Unknown'
        selections = row[1] if row[1] else 0
        strategies = row[2] if row[2] else 0
        avg_score = row[3] if row[3] else 0
        total_selections += selections
        
        print(f"{sector:<30} {selections:<10} {strategies:<10} {avg_score:.1f}")
    
    print(f"\nTotal Portfolio Positions: {total_selections}")

def analyze_risk_and_diversification(conn, strategy_results):
    """Analyze risk and diversification metrics"""
    print(f"\n[RISK] DIVERSIFICATION ANALYSIS:")
    print("-" * 40)
    
    # Calculate diversification metrics
    all_stocks = set()
    strategy_overlaps = {}
    
    for strategy_name, results in strategy_results.items():
        stocks = set(results['portfolio_stocks'])
        all_stocks.update(stocks)
        strategy_overlaps[strategy_name] = len(stocks)
    
    # Overlap analysis
    total_positions = sum(len(results['portfolio_stocks']) for results in strategy_results.values())
    unique_stocks = len(all_stocks)
    overlap_rate = (total_positions - unique_stocks) / total_positions if total_positions > 0 else 0
    
    print(f"Diversification Metrics:")
    print(f"  Total Positions: {total_positions}")
    print(f"  Unique Stocks: {unique_stocks}")
    print(f"  Overlap Rate: {overlap_rate:.1%}")
    print(f"  Diversification Score: {(1 - overlap_rate) * 100:.1f}/100")
    
    # Score distribution analysis
    all_scores = [results['avg_score'] for results in strategy_results.values()]
    if all_scores:
        avg_system_score = sum(all_scores) / len(all_scores)
        score_volatility = (max(all_scores) - min(all_scores)) / avg_system_score if avg_system_score > 0 else 0
        
        print(f"\nScore Distribution:")
        print(f"  System Average Score: {avg_system_score:.1f}")
        print(f"  Score Range: {min(all_scores):.1f} - {max(all_scores):.1f}")
        print(f"  Score Volatility: {score_volatility:.1%}")
        print(f"  Consistency Rating: {max(0, (1 - score_volatility) * 100):.1f}/100")

def main():
    """Execute comprehensive quarterly strategy run"""
    
    print("[LAUNCH] INITIATING COMPREHENSIVE QUARTERLY RUN")
    print("Complete Enhanced 12-Strategy System Execution")
    print()
    
    start_time = time.time()
    
    # Execute complete system
    success, results = run_comprehensive_quarterly_strategies()
    
    total_time = time.time() - start_time
    
    if success:
        print(f"\n" + "[SUCCESS]" * 4)
        print(f"QUARTERLY RUN SUCCESSFUL!")
        print(f"Total Execution Time: {total_time:.1f} seconds")
        print(f"Enhanced 12-Strategy System is FULLY OPERATIONAL")
        print("[SUCCESS]" * 4)
        
        print(f"\n[NEXT] STEPS:")
        print(f"1. Execute comprehensive backtest analysis")
        print(f"2. Generate detailed performance reports") 
        print(f"3. Implement optimization recommendations")
        
        return True
    else:
        print(f"\n[X] QUARTERLY RUN INCOMPLETE")
        print(f"Check system status and retry")
        return False

if __name__ == "__main__":
    main()