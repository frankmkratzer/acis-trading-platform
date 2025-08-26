#!/usr/bin/env python3
"""
Real-time Pipeline Monitoring Dashboard
Monitor the data ingestion progress and database population
"""

import os
import time
import json
from datetime import datetime
from pathlib import Path
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

def get_database_stats():
    """Get current database statistics"""
    try:
        engine = create_engine(os.getenv("POSTGRES_URL"))
        
        with engine.connect() as conn:
            # Check table row counts
            tables = [
                'symbol_universe',
                'stock_eod_daily', 
                'sp500_price_history',
                'fundamentals_annual',
                'dividend_history'
            ]
            
            stats = {}
            for table in tables:
                try:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    count = result.fetchone()[0]
                    stats[table] = count
                except:
                    stats[table] = 0
            
            # Get latest data dates
            try:
                result = conn.execute(text("""
                    SELECT MAX(trade_date) as latest_price_date,
                           COUNT(DISTINCT symbol) as unique_symbols
                    FROM stock_eod_daily
                """))
                row = result.fetchone()
                stats['latest_price_date'] = str(row[0]) if row[0] else 'None'
                stats['unique_symbols'] = row[1] or 0
            except:
                stats['latest_price_date'] = 'None'
                stats['unique_symbols'] = 0
            
            return stats
            
    except Exception as e:
        return {'error': str(e)}

def get_pipeline_logs():
    """Get recent pipeline logs"""
    log_file = Path("logs/eod_pipeline.log")
    if not log_file.exists():
        return []
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Get last 10 lines
        recent_lines = lines[-10:] if len(lines) > 10 else lines
        return [line.strip() for line in recent_lines if line.strip()]
        
    except Exception as e:
        return [f"Error reading logs: {e}"]

def display_dashboard():
    """Display the monitoring dashboard"""
    while True:
        # Clear screen
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("="*70)
        print("🚀 ACIS TRADING PLATFORM - PIPELINE MONITOR")
        print("="*70)
        print(f"⏰ Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Database Statistics
        print("📊 DATABASE STATISTICS")
        print("-" * 30)
        
        stats = get_database_stats()
        
        if 'error' in stats:
            print(f"❌ Database Error: {stats['error']}")
        else:
            print(f"🏢 Companies:        {stats.get('symbol_universe', 0):,}")
            print(f"📈 Price Records:    {stats.get('stock_eod_daily', 0):,}")
            print(f"📊 S&P 500 History:  {stats.get('sp500_price_history', 0):,}")
            print(f"💼 Fundamentals:     {stats.get('fundamentals_annual', 0):,}")
            print(f"💰 Dividends:        {stats.get('dividend_history', 0):,}")
            print(f"🎯 Active Symbols:   {stats.get('unique_symbols', 0):,}")
            print(f"📅 Latest Data:      {stats.get('latest_price_date', 'None')}")
        
        print()
        
        # Data Quality Assessment
        total_records = sum(v for k, v in stats.items() if isinstance(v, int) and k != 'unique_symbols')
        
        if total_records > 1000:
            quality = "🟢 EXCELLENT"
        elif total_records > 100:
            quality = "🟡 GOOD"
        elif total_records > 0:
            quality = "🟠 MINIMAL"
        else:
            quality = "🔴 EMPTY"
        
        print(f"📋 Data Quality: {quality} ({total_records:,} total records)")
        print()
        
        # Recent Pipeline Activity
        print("📝 RECENT PIPELINE ACTIVITY")
        print("-" * 40)
        
        logs = get_pipeline_logs()
        if logs:
            for log_line in logs[-5:]:  # Show last 5 log entries
                # Clean up log line for display
                if '[INFO]' in log_line:
                    emoji = "ℹ️ "
                elif '[SUCCESS]' in log_line or 'SUCCESS:' in log_line:
                    emoji = "✅ "
                elif '[ERROR]' in log_line or 'ERROR:' in log_line:
                    emoji = "❌ "
                elif '[WARN]' in log_line:
                    emoji = "⚠️ "
                else:
                    emoji = "📄 "
                
                # Extract just the message part
                if '] ' in log_line:
                    message = log_line.split('] ', 1)[-1]
                else:
                    message = log_line
                
                print(f"{emoji}{message[:60]}...")
        else:
            print("📄 No recent logs available")
        
        print()
        
        # Progress Estimation
        print("🎯 PIPELINE PROGRESS ESTIMATION")
        print("-" * 35)
        
        if stats.get('symbol_universe', 0) > 400:
            print("✅ Phase 1: Symbol Universe Complete (~500 companies)")
        elif stats.get('symbol_universe', 0) > 0:
            print("🔄 Phase 1: Symbol Universe In Progress...")
        else:
            print("⏳ Phase 1: Symbol Universe Pending")
        
        if stats.get('stock_eod_daily', 0) > 50000:
            print("✅ Phase 2: Price Data Complete (5+ years)")
        elif stats.get('stock_eod_daily', 0) > 1000:
            print("🔄 Phase 2: Price Data In Progress...")
        else:
            print("⏳ Phase 2: Price Data Pending")
        
        if stats.get('fundamentals_annual', 0) > 1000:
            print("✅ Phase 3: Fundamentals Complete")
        elif stats.get('fundamentals_annual', 0) > 0:
            print("🔄 Phase 3: Fundamentals In Progress...")
        else:
            print("⏳ Phase 3: Fundamentals Pending")
        
        print()
        
        # Trading Readiness
        total_symbols = stats.get('unique_symbols', 0)
        price_records = stats.get('stock_eod_daily', 0)
        
        if total_symbols >= 100 and price_records >= 10000:
            readiness = "🟢 READY FOR TRADING"
            recommendation = "System has sufficient data for live trading"
        elif total_symbols >= 50 and price_records >= 1000:
            readiness = "🟡 READY FOR TESTING"
            recommendation = "Good for paper trading and backtesting"
        elif total_symbols > 0 and price_records > 0:
            readiness = "🟠 LIMITED DATA"
            recommendation = "Wait for more data before trading"
        else:
            readiness = "🔴 NOT READY"
            recommendation = "Pipeline still starting up"
        
        print(f"🚦 Trading Readiness: {readiness}")
        print(f"💡 Recommendation: {recommendation}")
        
        print()
        print("Press Ctrl+C to exit monitoring...")
        print("="*70)
        
        # Wait 30 seconds before refresh
        try:
            time.sleep(30)
        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped. Pipeline continues running in background.")
            break

if __name__ == "__main__":
    try:
        display_dashboard()
    except Exception as e:
        print(f"Monitor error: {e}")
        print("Make sure the database is accessible and pipeline is running.")