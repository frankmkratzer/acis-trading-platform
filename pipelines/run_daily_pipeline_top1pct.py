#!/usr/bin/env python3
"""
ACIS Daily Pipeline - TOP 1% Enhanced Version
Includes all advanced analytics and scoring systems
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

logger = setup_logger("daily_pipeline_top1pct")


def run_script(script_path: str, description: str, timeout: int = 1800) -> bool:
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
    """Run TOP 1% daily pipeline"""
    start_time = time.time()
    
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║           ACIS DAILY PIPELINE - TOP 1% STRATEGY           ║
    ║                  Institutional-Grade Analytics             ║
    ╔════════════════════════════════════════════════════════════╗
    """)
    
    logger.info("="*60)
    logger.info("Starting ACIS Daily Pipeline - TOP 1% Enhanced")
    logger.info("="*60)
    
    results = {}
    
    # Phase 1: Update Market Data
    print("\n" + "="*60)
    print("PHASE 1: UPDATE MARKET DATA")
    print("="*60)
    
    # Update S&P 500 benchmark
    results['sp500'] = run_script(
        "data_fetch/market_data/fetch_sp500_history.py",
        "Updating S&P 500 benchmark data",
        timeout=300
    )
    
    # Update stock prices
    results['prices'] = run_script(
        "data_fetch/market_data/fetch_prices.py",
        "Updating stock prices",
        timeout=7200  # 2 hours
    )
    
    # Phase 2: Smart Money Tracking (TOP 1%)
    print("\n" + "="*60)
    print("PHASE 2: SMART MONEY TRACKING")
    print("="*60)
    
    # Fetch insider transactions
    results['insiders'] = run_script(
        "data_fetch/fundamentals/fetch_insider_transactions.py",
        "Fetching insider transactions",
        timeout=3600
    )
    
    # Update institutional holdings (if available)
    results['institutions'] = run_script(
        "data_fetch/fundamentals/fetch_institutional_holdings.py",
        "Updating institutional holdings",
        timeout=1800
    )
    
    # Phase 3: Fundamental Analysis (TOP 1%)
    print("\n" + "="*60)
    print("PHASE 3: FUNDAMENTAL SCORING SYSTEMS")
    print("="*60)
    
    # Calculate Piotroski F-Score
    results['piotroski'] = run_script(
        "analysis/calculate_piotroski_fscore.py",
        "Calculating Piotroski F-Scores",
        timeout=1800
    )
    
    # Calculate Altman Z-Score
    results['altman'] = run_script(
        "analysis/calculate_altman_zscore.py",
        "Calculating Altman Z-Scores",
        timeout=1800
    )
    
    # Calculate Beneish M-Score
    results['beneish'] = run_script(
        "analysis/calculate_beneish_mscore.py",
        "Calculating Beneish M-Scores",
        timeout=1800
    )
    
    # Phase 4: Risk & Technical Analysis (TOP 1%)
    print("\n" + "="*60)
    print("PHASE 4: RISK & TECHNICAL ANALYSIS")
    print("="*60)
    
    # Calculate risk metrics
    results['risk'] = run_script(
        "analysis/calculate_risk_metrics.py",
        "Calculating risk metrics (Sharpe, Sortino, VaR)",
        timeout=2400
    )
    
    # Detect technical breakouts
    results['breakouts'] = run_script(
        "analysis/calculate_technical_breakouts.py",
        "Detecting technical breakouts",
        timeout=1800
    )
    
    # Update sector rotation
    results['sectors'] = run_script(
        "analysis/sector_rotation_matrix.py",
        "Updating sector rotation matrix",
        timeout=1800
    )
    
    # Phase 5: Portfolio Optimization (TOP 1%)
    print("\n" + "="*60)
    print("PHASE 5: PORTFOLIO OPTIMIZATION")
    print("="*60)
    
    # Calculate Kelly Criterion
    results['kelly'] = run_script(
        "analysis/calculate_kelly_criterion.py",
        "Calculating Kelly Criterion position sizes",
        timeout=1800
    )
    
    # Phase 6: Master Scoring & Portfolio Management
    print("\n" + "="*60)
    print("PHASE 6: MASTER SCORING SYSTEM")
    print("="*60)
    
    # Calculate master composite scores
    results['master_scores'] = run_script(
        "analysis/calculate_master_scores.py",
        "Calculating master composite scores",
        timeout=2400
    )
    
    # Update portfolio selections
    results['portfolios'] = run_script(
        "portfolios/portfolio_manager.py",
        "Updating portfolio selections",
        timeout=1800
    )
    
    # Phase 7: Machine Learning (Optional - Heavy)
    print("\n" + "="*60)
    print("PHASE 7: MACHINE LEARNING (OPTIONAL)")
    print("="*60)
    
    # Run ML predictions if enabled
    if os.getenv("ENABLE_ML", "false").lower() == "true":
        results['ml'] = run_script(
            "ml_analysis/strategies/xgboost_return_predictor.py",
            "Running XGBoost predictions",
            timeout=3600
        )
    else:
        print("[INFO] ML predictions disabled (set ENABLE_ML=true to enable)")
    
    # Summary
    duration = time.time() - start_time
    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    
    print("\n" + "="*60)
    print("TOP 1% PIPELINE SUMMARY")
    print("="*60)
    print(f"Duration: {duration/60:.1f} minutes")
    print(f"Success Rate: {success_count}/{total_count} tasks")
    
    # Group results by category
    print("\n📊 Market Data:")
    for task in ['sp500', 'prices']:
        if task in results:
            status = "✓" if results[task] else "✗"
            print(f"  {status} {task}")
    
    print("\n🔍 Smart Money:")
    for task in ['insiders', 'institutions']:
        if task in results:
            status = "✓" if results[task] else "✗"
            print(f"  {status} {task}")
    
    print("\n📈 Fundamental Scores:")
    for task in ['piotroski', 'altman', 'beneish']:
        if task in results:
            status = "✓" if results[task] else "✗"
            print(f"  {status} {task}")
    
    print("\n⚖️ Risk & Technical:")
    for task in ['risk', 'breakouts', 'sectors']:
        if task in results:
            status = "✓" if results[task] else "✗"
            print(f"  {status} {task}")
    
    print("\n🎯 Portfolio Management:")
    for task in ['kelly', 'master_scores', 'portfolios']:
        if task in results:
            status = "✓" if results[task] else "✗"
            print(f"  {status} {task}")
    
    logger.info("="*60)
    logger.info(f"TOP 1% Pipeline completed in {duration/60:.1f} minutes")
    logger.info(f"Success rate: {success_count}/{total_count}")
    logger.info("="*60)
    
    # Return 0 if critical tasks succeeded
    critical_tasks = ['prices', 'master_scores', 'portfolios']
    critical_success = all(results.get(task, False) for task in critical_tasks)
    
    if critical_success:
        print("\n🎉 [SUCCESS] TOP 1% daily pipeline completed successfully!")
        return 0
    else:
        print("\n⚠️  [WARNING] Daily pipeline completed with some errors")
        return 1


if __name__ == "__main__":
    sys.exit(main())