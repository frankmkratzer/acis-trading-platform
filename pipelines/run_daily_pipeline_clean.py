#!/usr/bin/env python3
"""
ACIS Daily Pipeline - Clean Version
Simplified daily pipeline for three-portfolio strategy
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

logger = setup_logger("daily_pipeline_clean")


def run_script(script_path: str, description: str, timeout: int = 1800) -> bool:
    """Run a script and return success status"""
    print(f"\n[INFO] Running: {description}")
    logger.info(f"Running: {description}")
    
    script_full_path = Path(__file__).parent.parent / script_path
    
    if not script_full_path.exists():
        print(f"[ERROR] Script not found: {script_path}")
        logger.error(f"Script not found: {script_path}")
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
    """Run daily pipeline"""
    start_time = time.time()
    
    print("""
    ================================================================
                    ACIS DAILY PIPELINE - CLEAN
    ================================================================
    """)
    
    logger.info("="*60)
    logger.info("Starting ACIS Daily Pipeline - Clean Version")
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
    
    # Update stock prices (this is the longest step)
    results['prices'] = run_script(
        "data_fetch/market_data/fetch_prices.py",
        "Updating stock prices",
        timeout=7200  # 2 hours
    )
    
    # Phase 2: Portfolio Analysis
    print("\n" + "="*60)
    print("PHASE 2: PORTFOLIO ANALYSIS")
    print("="*60)
    
    # Calculate excess cash flow
    results['excess_cf'] = run_script(
        "analysis/excess_cash_flow.py",
        "Calculating excess cash flow metrics",
        timeout=1800
    )
    
    # Analyze dividend sustainability
    results['dividends'] = run_script(
        "analysis/dividend_sustainability.py",
        "Analyzing dividend sustainability",
        timeout=1800
    )
    
    # Detect breakout signals
    results['breakouts'] = run_script(
        "analysis/breakout_detector.py",
        "Detecting breakout signals",
        timeout=1800
    )
    
    # Phase 3: Portfolio Management
    print("\n" + "="*60)
    print("PHASE 3: PORTFOLIO MANAGEMENT")
    print("="*60)
    
    # Update portfolio scores and selections
    results['portfolios'] = run_script(
        "portfolios/portfolio_manager.py",
        "Updating portfolio selections",
        timeout=1800
    )
    
    # Summary
    duration = time.time() - start_time
    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    print(f"Duration: {duration/60:.1f} minutes")
    print(f"Success Rate: {success_count}/{total_count} tasks")
    print("\nTask Results:")
    for task, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {task}")
    
    logger.info("="*60)
    logger.info(f"Pipeline completed in {duration/60:.1f} minutes")
    logger.info(f"Success rate: {success_count}/{total_count}")
    logger.info("="*60)
    
    # Return 0 if all critical tasks succeeded
    critical_tasks = ['sp500', 'prices', 'portfolios']
    critical_success = all(results.get(task, False) for task in critical_tasks)
    
    if critical_success:
        print("\n[SUCCESS] Daily pipeline completed successfully!")
        return 0
    else:
        print("\n[WARNING] Daily pipeline completed with some errors")
        return 1


if __name__ == "__main__":
    sys.exit(main())