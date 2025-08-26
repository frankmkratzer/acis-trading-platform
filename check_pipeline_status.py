#!/usr/bin/env python3
"""
Pipeline Status Checker - Quick status without emojis for Windows compatibility
"""
import os
from pathlib import Path
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

def check_status():
    load_dotenv()
    
    print("ACIS Trading Platform - Pipeline Status")
    print("=" * 50)
    
    # Database check
    try:
        engine = create_engine(os.getenv('POSTGRES_URL'))
        
        with engine.connect() as conn:
            # Get table counts
            tables_data = {}
            tables = [
                ('symbol_universe', 'Companies'),
                ('stock_eod_daily', 'Price Records'), 
                ('sp500_price_history', 'S&P 500 History'),
                ('fundamentals_annual', 'Fundamentals'),
                ('dividend_history', 'Dividends')
            ]
            
            total_records = 0
            for table, label in tables:
                try:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    count = result.fetchone()[0]
                    tables_data[table] = count
                    total_records += count
                    print(f"{label:.<20} {count:,}")
                except Exception as e:
                    tables_data[table] = 0
                    print(f"{label:.<20} ERROR: {e}")
            
            print("-" * 50)
            print(f"{'Total Records':.<20} {total_records:,}")
            
            # Get unique symbols and latest date
            try:
                result = conn.execute(text("""
                    SELECT COUNT(DISTINCT symbol) as symbols,
                           MAX(trade_date) as latest_date
                    FROM stock_eod_daily
                """))
                row = result.fetchone()
                print(f"{'Active Symbols':.<20} {row[0]:,}")
                print(f"{'Latest Data':.<20} {row[1] or 'None'}")
            except:
                print("Active Symbols....... Unable to determine")
                print("Latest Data.......... Unable to determine")
    
    except Exception as e:
        print(f"Database connection failed: {e}")
    
    # Check log file
    print("\n" + "=" * 50)
    print("Recent Pipeline Activity:")
    print("-" * 30)
    
    log_file = Path("logs/eod_pipeline.log")
    if log_file.exists():
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Get last 5 lines that contain useful info
            useful_lines = []
            for line in reversed(lines):
                if any(keyword in line for keyword in ['SUCCESS:', 'Executing', 'Wave', 'ERROR:', 'FAILED']):
                    useful_lines.append(line.strip())
                if len(useful_lines) >= 5:
                    break
            
            for line in reversed(useful_lines):
                # Extract timestamp and message
                if '] ' in line:
                    timestamp_part, message = line.split('] ', 1)
                    timestamp = timestamp_part.split(' ')[-1] if ' ' in timestamp_part else timestamp_part
                    print(f"{timestamp} - {message}")
                else:
                    print(line)
                    
        except Exception as e:
            print(f"Could not read log file: {e}")
    else:
        print("No log file found")
    
    # Assessment
    print("\n" + "=" * 50)
    print("Assessment:")
    
    if total_records > 10_000_000:
        print("Status: EXCELLENT - Massive institutional dataset loaded")
        print("Ready for: Advanced AI analysis and live trading")
    elif total_records > 1_000_000:
        print("Status: VERY GOOD - Large dataset available")  
        print("Ready for: Full backtesting and paper trading")
    elif total_records > 100_000:
        print("Status: GOOD - Substantial data loaded")
        print("Ready for: Basic backtesting and strategy testing")
    elif total_records > 1000:
        print("Status: FAIR - Some data available")
        print("Ready for: Limited testing")
    else:
        print("Status: MINIMAL - Pipeline just starting")
        print("Ready for: Wait for more data")
    
    print("=" * 50)

if __name__ == "__main__":
    check_status()