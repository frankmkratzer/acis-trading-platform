#!/usr/bin/env python3
"""
ACIS Weekly Pipeline - Clean Version
Updates fundamentals and company data weekly
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

logger = setup_logger("weekly_pipeline_clean")


def run_script(script_path: str, description: str, timeout: int = 3600) -> bool:
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
    """Run weekly pipeline"""
    start_time = time.time()
    
    print("""
    ================================================================
                    ACIS WEEKLY PIPELINE - CLEAN
    ================================================================
    """)
    
    logger.info("="*60)
    logger.info("Starting ACIS Weekly Pipeline - Clean Version")
    logger.info("="*60)
    
    results = {}
    
    # Phase 1: Update Stock Universe
    print("\n" + "="*60)
    print("PHASE 1: UPDATE STOCK UNIVERSE")
    print("="*60)
    
    # Refresh stock universe (new listings, delistings)
    results['stocks'] = run_script(
        "data_fetch/market_data/fetch_quality_stocks.py",
        "Updating stock universe ($2B+ market cap)",
        timeout=600
    )
    
    # Phase 2: Update Fundamentals
    print("\n" + "="*60)
    print("PHASE 2: UPDATE FUNDAMENTALS")
    print("="*60)
    
    # Update company overviews (sector, industry, market cap)
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
    
    # Phase 3: Recalculate All Metrics
    print("\n" + "="*60)
    print("PHASE 3: RECALCULATE METRICS")
    print("="*60)
    
    # Recalculate excess cash flow with new fundamentals
    results['excess_cf'] = run_script(
        "analysis/excess_cash_flow.py",
        "Recalculating excess cash flow metrics",
        timeout=1800
    )
    
    # Recalculate dividend sustainability
    results['div_sustain'] = run_script(
        "analysis/dividend_sustainability.py",
        "Recalculating dividend sustainability",
        timeout=1800
    )
    
    # Phase 4: Update Portfolio Scores
    print("\n" + "="*60)
    print("PHASE 4: UPDATE PORTFOLIOS")
    print("="*60)
    
    # Recalculate all portfolio scores with fresh data
    results['portfolios'] = run_script(
        "portfolios/portfolio_manager.py",
        "Recalculating portfolio scores",
        timeout=1800
    )
    
    # Summary
    duration = time.time() - start_time
    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    
    print("\n" + "="*60)
    print("WEEKLY PIPELINE SUMMARY")
    print("="*60)
    print(f"Duration: {duration/60:.1f} minutes")
    print(f"Success Rate: {success_count}/{total_count} tasks")
    print("\nTask Results:")
    for task, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {task}")
    
    logger.info("="*60)
    logger.info(f"Weekly pipeline completed in {duration/60:.1f} minutes")
    logger.info(f"Success rate: {success_count}/{total_count}")
    logger.info("="*60)
    
    # Return 0 if all critical tasks succeeded
    critical_tasks = ['fundamentals', 'portfolios']
    critical_success = all(results.get(task, False) for task in critical_tasks)
    
    if critical_success:
        print("\n[SUCCESS] Weekly pipeline completed successfully!")
        return 0
    else:
        print("\n[WARNING] Weekly pipeline completed with some errors")
        return 1


if __name__ == "__main__":
    sys.exit(main())