#!/usr/bin/env python3
"""
AI Analysis Phase Runner
Run the AI/ML analysis on the loaded market data to generate trading signals
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

def run_analysis_phase():
    """Run the AI analysis phase of the pipeline"""
    
    print("="*60)
    print("ACIS TRADING PLATFORM - AI ANALYSIS PHASE")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    print("This phase will:")
    print("1. Compute forward returns for all stocks")
    print("2. Calculate value, momentum, and growth scores")  
    print("3. Train AI models on the massive dataset")
    print("4. Generate stock rankings and portfolios")
    print("5. Create trading signals")
    print()
    
    # Check if ingestion is still running
    print("Checking if data ingestion is complete...")
    
    try:
        # Quick data check
        import os
        from sqlalchemy import create_engine, text
        from dotenv import load_dotenv
        
        load_dotenv()
        engine = create_engine(os.getenv('POSTGRES_URL'))
        
        with engine.connect() as conn:
            result = conn.execute(text('SELECT COUNT(*) FROM stock_eod_daily'))
            price_count = result.fetchone()[0]
            
            result = conn.execute(text('SELECT COUNT(DISTINCT symbol) FROM stock_eod_daily'))
            symbol_count = result.fetchone()[0]
        
        print(f"Current dataset: {price_count:,} price records, {symbol_count:,} symbols")
        
        if symbol_count >= 500 and price_count >= 100000:
            print("Dataset is sufficient for AI analysis!")
        else:
            print("Warning: Dataset may be small for optimal AI training")
        
    except Exception as e:
        print(f"Could not check dataset: {e}")
        print("Proceeding with analysis anyway...")
    
    print()
    print("Starting AI analysis phase...")
    print("This may take 30-60 minutes depending on data size.")
    print()
    
    # Run the analysis phase
    cmd = [
        sys.executable, 
        "run_eod_full_pipeline.py", 
        "--only-phase", "analysis",
        "--verbose",
        "--continue-on-error"
    ]
    
    try:
        print("Executing: python run_eod_full_pipeline.py --only-phase analysis")
        print("-" * 60)
        
        # Start the analysis
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Stream output
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        # Get return code
        return_code = process.poll()
        
        print()
        print("=" * 60)
        if return_code == 0:
            print("AI ANALYSIS COMPLETED SUCCESSFULLY!")
            print()
            print("Your system now has:")
            print("- Stock rankings and scores")
            print("- AI model predictions") 
            print("- Portfolio recommendations")
            print("- Trading signals")
            print()
            print("Ready for:")
            print("- Paper trading: python live_trading_engine.py --paper")
            print("- Advanced backtests: python backtest_engine.py")
            print("- Portfolio optimization: python multi_factor_optimizer.py")
        else:
            print(f"AI analysis completed with warnings (exit code: {return_code})")
            print("Check logs for details. System may still be usable.")
        
        print("=" * 60)
        return return_code == 0
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        return False
    except Exception as e:
        print(f"Error running analysis: {e}")
        return False

if __name__ == "__main__":
    success = run_analysis_phase()
    sys.exit(0 if success else 1)