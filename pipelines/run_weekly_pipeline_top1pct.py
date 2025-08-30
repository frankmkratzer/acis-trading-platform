#!/usr/bin/env python3
"""
ACIS Weekly Pipeline - TOP 1% Enhanced Version
Includes backtesting, optimization, and performance attribution
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.logging_config import setup_logger

logger = setup_logger("weekly_pipeline_top1pct")


def run_script(script_path: str, description: str, timeout: int = 3600) -> bool:
    """Run a script and return success status"""
    print(f"\n[INFO] Running: {description}")
    logger.info(f"Running: {description}")
    
    script_full_path = Path(__file__).parent.parent / script_path
    
    if not script_full_path.exists():
        print(f"[WARNING] Script not found: {script_path} - Skipping")
        logger.warning(f"Script not found: {script_path}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_full_path)],
            capture_output=False,  # Show output directly
            text=True,
            timeout=timeout
        )
        
        if result.returncode == 0:
            print(f"[SUCCESS] Completed: {description}")
            logger.info(f"Completed: {description}")
            return True
        else:
            print(f"[FAILED] Error in: {description}")
            logger.error(f"Failed: {description}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {description} exceeded {timeout/60:.0f} minutes")
        logger.error(f"Timeout: {description}")
        return False
    except Exception as e:
        print(f"[ERROR] {description} failed: {e}")
        logger.error(f"Error in {description}: {e}")
        return False


def main():
    """Run TOP 1% weekly pipeline"""
    start_time = time.time()
    
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║          ACIS WEEKLY PIPELINE - TOP 1% STRATEGY           ║
    ║         Deep Analysis, Backtesting & Optimization          ║
    ╔════════════════════════════════════════════════════════════╗
    """)
    
    logger.info("="*60)
    logger.info("Starting ACIS Weekly Pipeline - TOP 1% Enhanced")
    logger.info("="*60)
    
    results = {}
    
    # Phase 1: Update Stock Universe
    print("\n" + "="*60)
    print("PHASE 1: UPDATE STOCK UNIVERSE")
    print("="*60)
    
    # Refresh stock universe
    results['stocks'] = run_script(
        "data_fetch/market_data/fetch_quality_stocks.py",
        "Updating stock universe ($2B+ market cap)",
        timeout=600
    )
    
    # Phase 2: Update Fundamentals
    print("\n" + "="*60)
    print("PHASE 2: UPDATE FUNDAMENTALS")
    print("="*60)
    
    # Update company overviews
    results['overview'] = run_script(
        "data_fetch/fundamentals/fetch_company_overview.py",
        "Updating company overviews",
        timeout=3600
    )
    
    # Update financial statements
    results['fundamentals'] = run_script(
        "data_fetch/fundamentals/fetch_fundamentals.py",
        "Updating financial statements",
        timeout=10800  # 3 hours
    )
    
    # Update dividend history
    results['dividends'] = run_script(
        "data_fetch/fundamentals/fetch_dividend_history.py",
        "Updating dividend history",
        timeout=3600
    )
    
    # Update earnings estimates
    results['earnings'] = run_script(
        "data_fetch/fundamentals/fetch_earnings_estimates.py",
        "Updating earnings estimates",
        timeout=3600
    )
    
    # Phase 3: Recalculate All Metrics (TOP 1%)
    print("\n" + "="*60)
    print("PHASE 3: RECALCULATE ALL METRICS")
    print("="*60)
    
    # Core metrics
    results['excess_cf'] = run_script(
        "analysis/excess_cash_flow.py",
        "Recalculating excess cash flow metrics",
        timeout=1800
    )
    
    results['div_sustain'] = run_script(
        "analysis/dividend_sustainability.py",
        "Recalculating dividend sustainability",
        timeout=1800
    )
    
    # Fundamental scores
    results['piotroski'] = run_script(
        "analysis/calculate_piotroski_fscore.py",
        "Recalculating Piotroski F-Scores",
        timeout=1800
    )
    
    results['altman'] = run_script(
        "analysis/calculate_altman_zscore.py",
        "Recalculating Altman Z-Scores",
        timeout=1800
    )
    
    results['beneish'] = run_script(
        "analysis/calculate_beneish_mscore.py",
        "Recalculating Beneish M-Scores",
        timeout=1800
    )
    
    # Risk metrics
    results['risk'] = run_script(
        "analysis/calculate_risk_metrics.py",
        "Recalculating risk metrics",
        timeout=2400
    )
    
    # Technical analysis
    results['breakouts'] = run_script(
        "analysis/calculate_technical_breakouts.py",
        "Recalculating technical breakouts",
        timeout=1800
    )
    
    # Phase 4: Portfolio Optimization (TOP 1%)
    print("\n" + "="*60)
    print("PHASE 4: PORTFOLIO OPTIMIZATION")
    print("="*60)
    
    # Kelly Criterion
    results['kelly'] = run_script(
        "analysis/calculate_kelly_criterion.py",
        "Recalculating Kelly Criterion",
        timeout=1800
    )
    
    # Sector rotation
    results['sectors'] = run_script(
        "analysis/sector_rotation_matrix.py",
        "Updating sector rotation matrix",
        timeout=1800
    )
    
    # Master scores
    results['master_scores'] = run_script(
        "analysis/calculate_master_scores.py",
        "Recalculating master composite scores",
        timeout=2400
    )
    
    # Phase 5: Backtesting & Validation (TOP 1%)
    print("\n" + "="*60)
    print("PHASE 5: BACKTESTING & VALIDATION")
    print("="*60)
    
    # Run backtests
    results['backtest'] = run_script(
        "analysis/backtesting_framework.py",
        "Running strategy backtests",
        timeout=3600
    )
    
    # Walk-forward optimization
    if os.getenv("ENABLE_OPTIMIZATION", "false").lower() == "true":
        results['optimization'] = run_script(
            "analysis/walk_forward_optimization.py",
            "Running walk-forward optimization",
            timeout=7200  # 2 hours
        )
    else:
        print("[INFO] Walk-forward optimization disabled (set ENABLE_OPTIMIZATION=true)")
    
    # Performance attribution
    results['attribution'] = run_script(
        "analysis/performance_attribution.py",
        "Calculating performance attribution",
        timeout=2400
    )
    
    # Phase 6: Machine Learning (TOP 1%)
    print("\n" + "="*60)
    print("PHASE 6: MACHINE LEARNING")
    print("="*60)
    
    # Train ML models
    if os.getenv("ENABLE_ML", "false").lower() == "true":
        results['ml_train'] = run_script(
            "ml_analysis/strategies/xgboost_return_predictor.py",
            "Training XGBoost model",
            timeout=5400  # 1.5 hours
        )
    else:
        print("[INFO] ML training disabled (set ENABLE_ML=true to enable)")
    
    # Phase 7: Update Portfolios
    print("\n" + "="*60)
    print("PHASE 7: UPDATE PORTFOLIOS")
    print("="*60)
    
    # Final portfolio update with all new data
    results['portfolios'] = run_script(
        "portfolios/portfolio_manager.py",
        "Finalizing portfolio selections",
        timeout=1800
    )
    
    # Summary
    duration = time.time() - start_time
    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    
    print("\n" + "="*60)
    print("TOP 1% WEEKLY PIPELINE SUMMARY")
    print("="*60)
    print(f"Duration: {duration/60:.1f} minutes")
    print(f"Success Rate: {success_count}/{total_count} tasks")
    
    # Detailed results
    print("\n📊 Data Updates:")
    for task in ['stocks', 'overview', 'fundamentals', 'dividends', 'earnings']:
        if task in results:
            status = "✓" if results[task] else "✗"
            print(f"  {status} {task}")
    
    print("\n📈 Scoring Systems:")
    for task in ['piotroski', 'altman', 'beneish', 'risk', 'master_scores']:
        if task in results:
            status = "✓" if results[task] else "✗"
            print(f"  {status} {task}")
    
    print("\n🎯 Optimization:")
    for task in ['kelly', 'sectors', 'backtest', 'optimization', 'attribution']:
        if task in results:
            status = "✓" if results[task] else "✗"
            print(f"  {status} {task}")
    
    print("\n🤖 Machine Learning:")
    for task in ['ml_train']:
        if task in results:
            status = "✓" if results[task] else "✗"
            print(f"  {status} {task}")
    
    logger.info("="*60)
    logger.info(f"TOP 1% Weekly pipeline completed in {duration/60:.1f} minutes")
    logger.info(f"Success rate: {success_count}/{total_count}")
    logger.info("="*60)
    
    # Return 0 if critical tasks succeeded
    critical_tasks = ['fundamentals', 'master_scores', 'portfolios']
    critical_success = all(results.get(task, False) for task in critical_tasks)
    
    if critical_success:
        print("\n🎉 [SUCCESS] TOP 1% weekly pipeline completed successfully!")
        print("\n💡 Next Steps:")
        print("  1. Review backtest results for strategy performance")
        print("  2. Check performance attribution for alpha sources")
        print("  3. Monitor Kelly Criterion for position sizing")
        print("  4. Review sector rotation recommendations")
        return 0
    else:
        print("\n⚠️  [WARNING] Weekly pipeline completed with some errors")
        return 1


if __name__ == "__main__":
    sys.exit(main())