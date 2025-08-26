#!/usr/bin/env python3
"""
Fetch Enhanced Fundamentals for 20-Year Analysis
Fetches comprehensive Alpha Vantage fundamental data including all funnel analysis fields
"""

import os
import subprocess
import time
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

def main():
    print("FETCHING ENHANCED FUNDAMENTALS FOR 20-YEAR ANALYSIS")
    print("=" * 60)
    
    load_dotenv()
    engine = create_engine(os.getenv('POSTGRES_URL'))
    
    # Set environment variable for aggressive data fetching
    os.environ["FUND_FRESHNESS_DAYS"] = "0"  # Force fetch all data regardless of age
    
    print("Starting enhanced fundamental data fetch...")
    print("This will take 30-60 minutes for complete dataset")
    print()
    
    # Get symbol count for progress tracking
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(DISTINCT symbol) FROM symbol_universe"))
        total_symbols = result.fetchone()[0]
        print(f"Fetching data for {total_symbols} symbols")
        
        # Check current data status
        result = conn.execute(text("""
            SELECT 
                COUNT(DISTINCT symbol) as symbols_with_data,
                MAX(fiscal_date) as latest_date,
                MIN(fiscal_date) as earliest_date
            FROM fundamentals_annual
        """))
        
        current_stats = result.fetchone()
        if current_stats and current_stats[0]:
            print(f"Current data: {current_stats[0]} symbols")
            print(f"Date range: {current_stats[2]} to {current_stats[1]}")
        else:
            print("No existing fundamental data found")
    
    print()
    print("Starting fetch process...")
    
    # Run the enhanced fetch_fundamentals.py
    try:
        start_time = time.time()
        
        result = subprocess.run([
            "python", "fetch_fundamentals.py"
        ], 
        cwd="C:\\Users\\frank\\PycharmProjects\\PythonProject\\acis-trading-platform",
        capture_output=True, 
        text=True, 
        timeout=3600  # 1 hour timeout
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✓ Fetch completed successfully in {elapsed/60:.1f} minutes")
            print("STDOUT:", result.stdout[-1000:] if result.stdout else "No output")  # Last 1000 chars
        else:
            print(f"✗ Fetch failed with return code {result.returncode}")
            print("STDERR:", result.stderr[-1000:] if result.stderr else "No error output")
            print("STDOUT:", result.stdout[-1000:] if result.stdout else "No output")
            
    except subprocess.TimeoutExpired:
        print("✗ Fetch timed out after 1 hour")
        return False
    except Exception as e:
        print(f"✗ Error running fetch: {e}")
        return False
    
    # Verify the enhanced data
    print("\nVerifying enhanced fundamental data...")
    
    with engine.connect() as conn:
        # Check updated data counts
        result = conn.execute(text("""
            SELECT 
                COUNT(DISTINCT symbol) as symbols_with_data,
                COUNT(*) as total_records,
                MAX(fiscal_date) as latest_date,
                MIN(fiscal_date) as earliest_date,
                COUNT(DISTINCT EXTRACT(YEAR FROM fiscal_date)) as years_covered
            FROM fundamentals_annual
        """))
        
        final_stats = result.fetchone()
        print(f"Final data: {final_stats[0]} symbols, {final_stats[1]} records")
        print(f"Date range: {final_stats[3]} to {final_stats[2]}")
        print(f"Years covered: {final_stats[4]}")
        
        # Check enhanced fields availability
        result = conn.execute(text("""
            SELECT 
                COUNT(*) as total_records,
                COUNT(cashandcashequivalentsatcarryingvalue) as cash_records,
                COUNT(commonstocksharesoutstanding) as shares_records,
                COUNT(longtermdebt) as debt_records
            FROM fundamentals_annual
            WHERE fiscal_date >= CURRENT_DATE - INTERVAL '5 years'
        """))
        
        enhanced_stats = result.fetchone()
        if enhanced_stats:
            print(f"\nEnhanced Fields (last 5 years):")
            print(f"  Cash data: {enhanced_stats[1]}/{enhanced_stats[0]} records ({enhanced_stats[1]/enhanced_stats[0]*100:.1f}%)")
            print(f"  Shares data: {enhanced_stats[2]}/{enhanced_stats[0]} records ({enhanced_stats[2]/enhanced_stats[0]*100:.1f}%)")
            print(f"  Debt data: {enhanced_stats[3]}/{enhanced_stats[0]} records ({enhanced_stats[3]/enhanced_stats[0]*100:.1f}%)")
        
        # Show sample of enhanced data
        result = conn.execute(text("""
            SELECT symbol, fiscal_date, totalrevenue, cashandcashequivalentsatcarryingvalue, 
                   commonstocksharesoutstanding, longtermdebt
            FROM fundamentals_annual 
            WHERE cashandcashequivalentsatcarryingvalue IS NOT NULL
              AND commonstocksharesoutstanding IS NOT NULL
            ORDER BY fiscal_date DESC 
            LIMIT 5
        """))
        
        sample_data = result.fetchall()
        if sample_data:
            print(f"\nSample Enhanced Data:")
            print(f"{'Symbol':<8} {'Date':<12} {'Revenue':<15} {'Cash':<15} {'Shares':<15} {'Debt':<15}")
            print("-" * 90)
            for row in sample_data:
                revenue = f"${row[2]/1e9:.1f}B" if row[2] else "N/A"
                cash = f"${row[3]/1e9:.1f}B" if row[3] else "N/A"
                shares = f"{row[4]/1e6:.0f}M" if row[4] else "N/A"
                debt = f"${row[5]/1e9:.1f}B" if row[5] else "N/A"
                print(f"{row[0]:<8} {str(row[1]):<12} {revenue:<15} {cash:<15} {shares:<15} {debt:<15}")
    
    print(f"\n{'='*60}")
    print("ENHANCED FUNDAMENTAL DATA FETCH COMPLETE!")
    print("Ready for comprehensive funnel analysis and 20-year backtesting")
    print(f"{'='*60}")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\nNext steps:")
        print("1. Run enhanced funnel scoring")
        print("2. Execute 12-strategy system")
        print("3. Perform comprehensive backtesting")
    else:
        print("\nFetch failed. Check logs and retry.")