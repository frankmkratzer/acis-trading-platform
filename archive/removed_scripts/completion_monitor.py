#!/usr/bin/env python3
"""
AI Training Completion Monitor
Real-time monitoring of the final stages of AI training
"""

import os
import time
from datetime import datetime
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

def check_completion_status():
    """Check the completion status of all AI components"""
    load_dotenv()
    
    try:
        engine = create_engine(os.getenv('POSTGRES_URL'))
        
        with engine.connect() as conn:
            # Core AI model outputs
            core_models = [
                ('forward_returns', 'Forward Returns'),
                ('ai_value_scores', 'Value Model'),
                ('ai_growth_scores', 'Growth Model'), 
                ('ai_momentum_scores', 'Momentum Model'),
                ('ai_dividend_scores', 'Dividend Model')
            ]
            
            # Portfolio outputs
            portfolios = [
                ('ai_value_portfolio', 'Value Portfolio'),
                ('ai_growth_portfolio', 'Growth Portfolio'),
                ('ai_momentum_portfolio', 'Momentum Portfolio'),
                ('ai_dividend_portfolio', 'Dividend Portfolio')
            ]
            
            status = {
                'models': {},
                'portfolios': {},
                'total_records': 0,
                'completion_percentage': 0
            }
            
            # Check core models
            completed_models = 0
            for table, name in core_models:
                try:
                    result = conn.execute(text(f'SELECT COUNT(*) FROM {table}'))
                    count = result.fetchone()[0]
                    status['models'][name] = count
                    status['total_records'] += count
                    if count > 0:
                        completed_models += 1
                except:
                    status['models'][name] = 0
            
            # Check portfolios
            completed_portfolios = 0
            for table, name in portfolios:
                try:
                    result = conn.execute(text(f'SELECT COUNT(*) FROM {table}'))
                    count = result.fetchone()[0]
                    status['portfolios'][name] = count
                    if count > 0:
                        completed_portfolios += 1
                except:
                    status['portfolios'][name] = 0
            
            # Calculate completion percentage
            total_components = len(core_models) + len(portfolios)
            completed_components = completed_models + completed_portfolios
            status['completion_percentage'] = (completed_components / total_components) * 100
            status['completed_models'] = completed_models
            status['completed_portfolios'] = completed_portfolios
            
            # Check for recent model training activity
            try:
                result = conn.execute(text("""
                    SELECT COUNT(*) FROM ai_model_run_log 
                    WHERE created_at > NOW() - INTERVAL '1 hour'
                """))
                status['recent_training_runs'] = result.fetchone()[0] or 0
            except:
                status['recent_training_runs'] = 0
            
            return status
            
    except Exception as e:
        return {'error': str(e)}

def display_completion_status():
    """Display completion status in a clean format"""
    
    print("ACIS AI TRAINING - COMPLETION MONITOR")
    print("=" * 50)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    status = check_completion_status()
    
    if 'error' in status:
        print(f"Error checking status: {status['error']}")
        return False
    
    # Display model completion
    print("AI MODEL COMPLETION STATUS:")
    print("-" * 30)
    
    for name, count in status['models'].items():
        if count > 1000:
            status_icon = "✅"
            status_text = "COMPLETE"
        elif count > 0:
            status_icon = "🔄"
            status_text = "PARTIAL"
        else:
            status_icon = "⏳"
            status_text = "PENDING"
        
        print(f"{status_icon} {name:.<20} {status_text} ({count:,})")
    
    print()
    
    # Display portfolio status
    print("PORTFOLIO GENERATION STATUS:")
    print("-" * 30)
    
    for name, count in status['portfolios'].items():
        if count > 0:
            status_icon = "✅"
            status_text = f"READY ({count:,} picks)"
        else:
            status_icon = "⏳"
            status_text = "PENDING"
        
        print(f"{status_icon} {name:.<20} {status_text}")
    
    print()
    
    # Overall status
    completion = status['completion_percentage']
    total_records = status['total_records']
    
    print("OVERALL PROGRESS:")
    print("-" * 20)
    print(f"Completion: {completion:.1f}%")
    print(f"Total AI Records: {total_records:,}")
    print(f"Models Complete: {status['completed_models']}/5")
    print(f"Portfolios Ready: {status['completed_portfolios']}/4")
    
    if status['recent_training_runs'] > 0:
        print(f"Recent Training Activity: {status['recent_training_runs']} runs")
    
    print()
    
    # Status assessment
    if completion >= 100:
        print("🎉 STATUS: FULLY COMPLETE!")
        print("🚀 READY FOR: Live trading, backtesting, portfolio management")
        return True
    elif completion >= 80:
        print("🟡 STATUS: NEARLY COMPLETE")
        print("⏱️  ESTIMATED: 5-15 minutes remaining")
        return False
    elif completion >= 60:
        print("🟠 STATUS: SUBSTANTIAL PROGRESS")
        print("⏱️  ESTIMATED: 15-30 minutes remaining") 
        return False
    else:
        print("🔄 STATUS: IN PROGRESS")
        print("⏱️  ESTIMATED: 30+ minutes remaining")
        return False

def monitor_until_complete():
    """Monitor continuously until training is complete"""
    print("Starting continuous monitoring...")
    print("Press Ctrl+C to stop monitoring (training continues)")
    print()
    
    start_time = datetime.now()
    
    try:
        while True:
            # Clear screen
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print(f"Monitoring Duration: {str(datetime.now() - start_time).split('.')[0]}")
            print()
            
            is_complete = display_completion_status()
            
            if is_complete:
                print()
                print("🎊 AI TRAINING COMPLETED SUCCESSFULLY! 🎊")
                print()
                print("Your trading system is now FULLY OPERATIONAL!")
                print()
                print("Next steps:")
                print("1. python live_trading_engine.py --paper  # Start paper trading")
                print("2. python demo_backtest.py               # Run advanced backtests")
                print("3. python multi_factor_optimizer.py      # Optimize portfolios")
                break
            
            print()
            print("Refreshing in 30 seconds...")
            print("=" * 50)
            
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped. Training continues in background.")
        print("\nTo check completion:")
        print("  python completion_monitor.py")

if __name__ == "__main__":
    try:
        monitor_until_complete()
    except Exception as e:
        print(f"Monitor error: {e}")
        print("Running single status check...")
        display_completion_status()