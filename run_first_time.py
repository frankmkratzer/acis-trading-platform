#!/usr/bin/env python3
"""
ACIS First Time Setup Script
Run this once to populate your database with all necessary data
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

def run_command(cmd, description):
    """Run a command and track execution"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    start = time.time()
    result = subprocess.run(cmd, shell=True, cwd=Path(__file__).parent)
    duration = time.time() - start
    
    if result.returncode == 0:
        print(f"✓ SUCCESS: {description} ({duration/60:.1f} minutes)")
    else:
        print(f"✗ FAILED: {description}")
        print("Continuing with next step...")
    
    return result.returncode == 0

def main():
    """Run complete first-time setup"""
    
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║            ACIS TRADING PLATFORM - FIRST TIME SETUP          ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  This will:                                                   ║
    ║  1. Create database tables                                   ║
    ║  2. Fetch stock universe (mid/large caps)                    ║
    ║  3. Download historical prices                               ║
    ║  4. Calculate rankings and metrics                           ║
    ║  5. Generate portfolio recommendations                       ║
    ║                                                              ║
    ║  Expected time: 3-5 hours                                    ║
    ║  Ensure you have set POSTGRES_URL and ALPHA_VANTAGE_API_KEY ║
    ╚═══════════════════════════════════════════════════════════════╝
    
    Starting automated setup...
    """)
    
    start_time = time.time()
    steps_completed = []
    
    # Step 1: Database Setup
    if run_command(
        "python database/setup_schema_clean.py",
        "Setting up database schema (15 tables)"
    ):
        steps_completed.append("Database setup")
    
    # Step 2: Verify tables
    run_command(
        "python database/verify_tables.py",
        "Verifying database tables"
    )
    
    # Step 3: Fetch stock universe
    if run_command(
        "python data_fetch/market_data/fetch_quality_stocks.py",
        "Fetching mid/large cap stocks ($2B+)"
    ):
        steps_completed.append("Stock universe")
    
    # Step 4: Fetch S&P 500 history
    if run_command(
        "python data_fetch/market_data/fetch_sp500_history.py",
        "Fetching S&P 500 constituents"
    ):
        steps_completed.append("S&P 500 history")
    
    # Step 5: Fetch price data
    print("\n⚠️  Fetching price data - this takes 2-3 hours")
    if run_command(
        "python data_fetch/market_data/fetch_prices.py",
        "Fetching historical price data"
    ):
        steps_completed.append("Price data")
    
    # Step 6: Fetch fundamentals (important for portfolio scoring)
    print("\n📊 Fetching fundamental data - this takes 2-3 hours")
    if run_command(
        "python data_fetch/fundamentals/fetch_fundamentals.py",
        "Fetching company fundamentals"
    ):
        steps_completed.append("Fundamentals")
    
    # Step 7: Calculate portfolio scores
    print("\n🎯 Calculating portfolio scores")
    if run_command(
        "python strategies/calculate_portfolio_scores.py",
        "Calculating portfolio scores"
    ):
        steps_completed.append("Portfolio scores")
    
    # Final summary
    total_time = time.time() - start_time
    print(f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                    SETUP COMPLETE!                           ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  Total time: {total_time/3600:.1f} hours                              ║
    ║  Steps completed: {len(steps_completed)}/6                              ║
    ╚═══════════════════════════════════════════════════════════════╝
    
    Completed steps:
    """)
    for step in steps_completed:
        print(f"  ✓ {step}")
    
    print("""
    Next steps:
    1. Run daily updates: python main.py --daily
    2. View portfolios: python portfolios/view_portfolios.py
    3. Check data: python database/verify_tables.py
    
    For complete guide, see RUN_GUIDE.md
    """)

if __name__ == "__main__":
    main()