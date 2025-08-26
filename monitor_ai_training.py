#!/usr/bin/env python3
"""
AI Training Progress Monitor
Track the progress of both data ingestion and AI model training
"""

import os
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

def get_training_progress():
    """Check AI training progress by looking at output tables"""
    load_dotenv()
    
    try:
        engine = create_engine(os.getenv('POSTGRES_URL'))
        
        with engine.connect() as conn:
            # Check for AI training outputs
            ai_tables = [
                ('forward_returns', 'Forward Returns'),
                ('ai_value_scores', 'Value Model Scores'),
                ('ai_growth_scores', 'Growth Model Scores'), 
                ('ai_momentum_scores', 'Momentum Scores'),
                ('ai_dividend_scores', 'Dividend Scores'),
                ('ai_value_portfolio', 'Value Portfolio'),
                ('ai_growth_portfolio', 'Growth Portfolio'),
                ('ai_momentum_portfolio', 'Momentum Portfolio'),
                ('ai_dividend_portfolio', 'Dividend Portfolio')
            ]
            
            progress = {}
            total_ai_records = 0
            
            for table, label in ai_tables:
                try:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    count = result.fetchone()[0]
                    progress[table] = {'count': count, 'label': label}
                    total_ai_records += count
                except:
                    progress[table] = {'count': 0, 'label': label}
            
            # Check for model training logs in ai_model_run_log
            try:
                result = conn.execute(text("""
                    SELECT COUNT(*) as runs,
                           MAX(created_at) as latest_run
                    FROM ai_model_run_log 
                    WHERE created_at > NOW() - INTERVAL '2 hours'
                """))
                row = result.fetchone()
                progress['model_runs'] = {
                    'count': row[0] or 0,
                    'latest': row[1] or 'None'
                }
            except:
                progress['model_runs'] = {'count': 0, 'latest': 'None'}
            
            progress['total_ai_records'] = total_ai_records
            return progress
            
    except Exception as e:
        return {'error': str(e)}

def display_training_monitor():
    """Display real-time training progress"""
    start_time = datetime.now()
    
    while True:
        # Clear screen (Windows compatible)
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("="*70)
        print("🧠 ACIS AI TRAINING MONITOR")
        print("="*70)
        print(f"⏰ Monitoring Started: {start_time.strftime('%H:%M:%S')}")
        print(f"🕐 Current Time: {datetime.now().strftime('%H:%M:%S')}")
        print(f"⏱️  Elapsed: {str(datetime.now() - start_time).split('.')[0]}")
        print()
        
        # Base data status
        print("📊 BASE DATA STATUS")
        print("-" * 30)
        
        try:
            load_dotenv()
            engine = create_engine(os.getenv('POSTGRES_URL'))
            
            with engine.connect() as conn:
                # Core data counts
                result = conn.execute(text("""
                    SELECT 
                        (SELECT COUNT(*) FROM symbol_universe) as companies,
                        (SELECT COUNT(*) FROM stock_eod_daily) as prices,
                        (SELECT COUNT(*) FROM fundamentals_annual) as fundamentals,
                        (SELECT COUNT(DISTINCT symbol) FROM stock_eod_daily) as symbols,
                        (SELECT MAX(trade_date) FROM stock_eod_daily) as latest_date
                """))
                
                row = result.fetchone()
                
                print(f"Companies.......... {row[0]:,}")
                print(f"Price Records...... {row[1]:,}")
                print(f"Fundamentals....... {row[2]:,}")
                print(f"Active Symbols..... {row[3]:,}")
                print(f"Latest Data........ {row[4] or 'None'}")
                
                total_base = (row[0] or 0) + (row[1] or 0) + (row[2] or 0)
                print(f"Total Base Data.... {total_base:,}")
        
        except Exception as e:
            print(f"❌ Database Error: {str(e)[:50]}")
        
        print()
        
        # AI Training Progress
        print("🧠 AI TRAINING PROGRESS")
        print("-" * 35)
        
        progress = get_training_progress()
        
        if 'error' in progress:
            print(f"❌ Training Status Error: {progress['error'][:50]}")
        else:
            # Show progress for each component
            training_stages = [
                ('forward_returns', '1. Forward Returns'),
                ('ai_value_scores', '2. Value Model'),
                ('ai_growth_scores', '3. Growth Model'),
                ('ai_momentum_scores', '4. Momentum Model'),
                ('ai_dividend_scores', '5. Dividend Model'),
            ]
            
            for table, stage in training_stages:
                count = progress.get(table, {}).get('count', 0)
                if count > 1000:
                    status = "✅ COMPLETE"
                elif count > 0:
                    status = "🔄 IN PROGRESS"
                else:
                    status = "⏳ PENDING"
                
                print(f"{stage:.<25} {status} ({count:,})")
            
            print()
            
            # Portfolio Generation
            portfolio_stages = [
                ('ai_value_portfolio', 'Value Portfolio'),
                ('ai_growth_portfolio', 'Growth Portfolio'),
                ('ai_momentum_portfolio', 'Momentum Portfolio'),
                ('ai_dividend_portfolio', 'Dividend Portfolio'),
            ]
            
            portfolios_ready = 0
            for table, name in portfolio_stages:
                count = progress.get(table, {}).get('count', 0)
                if count > 0:
                    portfolios_ready += 1
                    print(f"📈 {name}: {count:,} selections")
                else:
                    print(f"⏳ {name}: Pending")
            
            print()
            
            # Model Training Status
            model_runs = progress.get('model_runs', {})
            print(f"🤖 Model Training Runs: {model_runs.get('count', 0)}")
            print(f"🕐 Latest Training: {model_runs.get('latest', 'None')}")
            
            total_ai = progress.get('total_ai_records', 0)
            print(f"📊 Total AI Records: {total_ai:,}")
        
        print()
        
        # Overall Status Assessment
        print("🎯 OVERALL STATUS")
        print("-" * 20)
        
        if not ('error' in progress):
            ai_records = progress.get('total_ai_records', 0)
            
            if ai_records > 10000:
                status = "🟢 TRAINING COMPLETE!"
                next_step = "Ready for live trading!"
            elif ai_records > 1000:
                status = "🟡 TRAINING IN PROGRESS"
                next_step = "Models are being trained..."
            elif ai_records > 0:
                status = "🟠 TRAINING STARTED"
                next_step = "Initial processing underway..."
            else:
                status = "🔴 TRAINING NOT STARTED"
                next_step = "Waiting for data ingestion..."
            
            print(f"Status: {status}")
            print(f"Next: {next_step}")
            
            # Estimated completion
            if 0 < ai_records < 10000:
                print("⏱️  Estimated completion: 15-45 minutes")
            elif ai_records >= 10000:
                print("🎉 Training appears complete!")
        
        print()
        print("="*70)
        print("Press Ctrl+C to exit monitoring (processes continue running)")
        print("Refreshing in 30 seconds...")
        
        try:
            time.sleep(30)
        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped. AI training continues in background.")
            print("\nTo check completion status:")
            print("  python check_pipeline_status.py")
            print("\nTo resume monitoring:")
            print("  python monitor_ai_training.py")
            break

if __name__ == "__main__":
    try:
        display_training_monitor()
    except Exception as e:
        print(f"Monitor error: {e}")
        print("Check that database is accessible and training is running.")