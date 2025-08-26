#!/usr/bin/env python3
"""
Smart Scheduler for ACIS Trading Platform
Frequency-based selective script execution with intelligent resource management
"""

import os
import sys
import subprocess
import logging
import time
from datetime import datetime, timedelta
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
import pytz

class SmartTradingScheduler:
    def __init__(self):
        self.scheduler = BlockingScheduler(timezone=pytz.timezone('US/Eastern'))
        self.setup_logging()
        
        # Script execution definitions by frequency
        self.execution_scripts = {
            'market_hours': [
                'sector_strength_update.py',
                'price_data_refresh.py'
            ],
            'daily': [
                'daily_data_update.py', 
                'sector_strength_calculation.py',
                'quick_portfolio_check.py'
            ],
            'weekly': [
                'fundamental_data_update.py',
                'optimized_quarterly_run.py',
                'portfolio_rebalancing_check.py',
                'performance_monitoring.py'
            ],
            'monthly': [
                'enhanced_benchmark_analysis.py',
                'comprehensive_backtest_analysis.py',
                'historical_sector_strength.py',
                'system_optimization_review.py'
            ],
            'quarterly': [
                'comprehensive_quarterly_run.py',
                'complete_fundamental_refresh.py', 
                'strategy_performance_audit.py',
                'optimization_recommendations.py'
            ]
        }
        
        # Resource limits and timeouts by frequency
        self.execution_limits = {
            'market_hours': {'timeout': 300, 'max_retries': 2},    # 5 minutes
            'daily': {'timeout': 600, 'max_retries': 2},           # 10 minutes
            'weekly': {'timeout': 1800, 'max_retries': 1},         # 30 minutes
            'monthly': {'timeout': 3600, 'max_retries': 1},        # 60 minutes
            'quarterly': {'timeout': 7200, 'max_retries': 1}       # 120 minutes
        }
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('/var/log/acis-trading/scheduler.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger('SmartScheduler')
    
    def execute_scripts(self, frequency_type, scripts=None):
        """Execute scripts for a specific frequency with error handling"""
        
        if scripts is None:
            scripts = self.execution_scripts.get(frequency_type, [])
        
        self.logger.info(f"Starting {frequency_type} execution: {len(scripts)} scripts")
        
        execution_start = time.time()
        successful_executions = 0
        failed_executions = 0
        
        for script in scripts:
            script_start = time.time()
            
            try:
                self.logger.info(f"Executing {script}...")
                
                # Check if script exists
                script_path = f"/opt/acis-trading/{script}"
                if not os.path.exists(script_path):
                    self.logger.error(f"Script not found: {script_path}")
                    failed_executions += 1
                    continue
                
                # Execute script with timeout
                limits = self.execution_limits.get(frequency_type, {'timeout': 600, 'max_retries': 1})
                
                result = subprocess.run(
                    [sys.executable, script_path],
                    timeout=limits['timeout'],
                    capture_output=True,
                    text=True,
                    cwd="/opt/acis-trading/"
                )
                
                script_time = time.time() - script_start
                
                if result.returncode == 0:
                    self.logger.info(f"✓ {script} completed successfully ({script_time:.1f}s)")
                    successful_executions += 1
                else:
                    self.logger.error(f"✗ {script} failed with code {result.returncode}")
                    self.logger.error(f"Error output: {result.stderr}")
                    failed_executions += 1
                    
                    # Retry logic for critical scripts
                    if limits['max_retries'] > 0 and frequency_type in ['daily', 'weekly']:
                        self.logger.info(f"Retrying {script}...")
                        # Implement retry logic here
                        
            except subprocess.TimeoutExpired:
                self.logger.error(f"✗ {script} timed out after {limits['timeout']}s")
                failed_executions += 1
                
            except Exception as e:
                self.logger.error(f"✗ {script} execution error: {e}")
                failed_executions += 1
        
        total_time = time.time() - execution_start
        
        # Execution summary
        self.logger.info(f"{frequency_type.upper()} EXECUTION COMPLETE")
        self.logger.info(f"  Successful: {successful_executions}/{len(scripts)}")
        self.logger.info(f"  Failed: {failed_executions}/{len(scripts)}")
        self.logger.info(f"  Total Time: {total_time:.1f}s")
        
        # Send alerts for failures
        if failed_executions > 0:
            self.send_alert(frequency_type, successful_executions, failed_executions)
        
        return successful_executions, failed_executions
    
    def send_alert(self, frequency_type, successful, failed):
        """Send alert for failed executions"""
        alert_message = f"ACIS Trading Alert: {frequency_type} execution had {failed} failures out of {successful + failed} scripts"
        
        # Log alert (can be extended to email/SMS/Slack)
        self.logger.warning(f"ALERT: {alert_message}")
        
        # TODO: Add email/SMS/Slack notification here
        # self.send_email_alert(alert_message)
        # self.send_slack_notification(alert_message)
    
    def is_market_hours(self):
        """Check if current time is during market hours (9:30 AM - 4:00 PM ET)"""
        now = datetime.now(pytz.timezone('US/Eastern'))
        
        # Skip weekends
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
            
        # Check if between 9:30 AM and 4:00 PM
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    def setup_schedule(self):
        """Setup all scheduled jobs"""
        
        self.logger.info("Setting up ACIS Trading Platform schedule...")
        
        # Market Hours Updates (every 30 minutes during market hours)
        self.scheduler.add_job(
            func=self.market_hours_execution,
            trigger=IntervalTrigger(minutes=30, timezone='US/Eastern'),
            id='market_hours_updates',
            name='Market Hours Updates',
            replace_existing=True
        )
        
        # Daily Updates (6:00 PM ET after market close)
        self.scheduler.add_job(
            func=self.daily_execution,
            trigger=CronTrigger(hour=18, minute=0, timezone='US/Eastern'),
            id='daily_updates',
            name='Daily Updates',
            replace_existing=True
        )
        
        # Weekly Updates (Sunday 8:00 PM ET)
        self.scheduler.add_job(
            func=self.weekly_execution,
            trigger=CronTrigger(day_of_week='sun', hour=20, minute=0, timezone='US/Eastern'),
            id='weekly_updates', 
            name='Weekly Portfolio Updates',
            replace_existing=True
        )
        
        # Monthly Analysis (First Sunday of month, 10:00 PM ET)
        self.scheduler.add_job(
            func=self.monthly_execution,
            trigger=CronTrigger(day='1-7', day_of_week='sun', hour=22, minute=0, timezone='US/Eastern'),
            id='monthly_analysis',
            name='Monthly Comprehensive Analysis',
            replace_existing=True
        )
        
        # Quarterly Refresh (Last Sunday of Mar/Jun/Sep/Dec, 11:00 PM ET)
        for month in [3, 6, 9, 12]:  # March, June, September, December
            self.scheduler.add_job(
                func=self.quarterly_execution,
                trigger=CronTrigger(month=month, day='22-31', day_of_week='sun', hour=23, minute=0, timezone='US/Eastern'),
                id=f'quarterly_q{month//3}',
                name=f'Quarterly Refresh Q{month//3}',
                replace_existing=True
            )
        
        self.logger.info("All scheduled jobs configured successfully")
    
    def market_hours_execution(self):
        """Execute market hours updates"""
        if self.is_market_hours():
            self.execute_scripts('market_hours')
        else:
            self.logger.info("Outside market hours, skipping market hours execution")
    
    def daily_execution(self):
        """Execute daily updates"""
        self.execute_scripts('daily')
    
    def weekly_execution(self):
        """Execute weekly portfolio updates"""
        self.execute_scripts('weekly')
    
    def monthly_execution(self):
        """Execute monthly comprehensive analysis"""
        self.execute_scripts('monthly')
    
    def quarterly_execution(self):
        """Execute quarterly system refresh"""
        self.execute_scripts('quarterly')
    
    def run_manual_execution(self, frequency_type):
        """Run manual execution for testing"""
        self.logger.info(f"Manual execution requested: {frequency_type}")
        
        if frequency_type in self.execution_scripts:
            return self.execute_scripts(frequency_type)
        else:
            self.logger.error(f"Invalid frequency type: {frequency_type}")
            return 0, 1
    
    def start_scheduler(self):
        """Start the scheduler"""
        try:
            self.setup_schedule()
            self.logger.info("ACIS Trading Platform Scheduler starting...")
            self.logger.info("Scheduled jobs:")
            
            for job in self.scheduler.get_jobs():
                self.logger.info(f"  - {job.name}: {job.trigger}")
            
            self.scheduler.start()
            
        except KeyboardInterrupt:
            self.logger.info("Scheduler stopped by user")
            
        except Exception as e:
            self.logger.error(f"Scheduler error: {e}")
            raise

def main():
    """Main scheduler execution"""
    
    # Handle command line arguments for manual execution
    if len(sys.argv) > 1:
        frequency_type = sys.argv[1].lower()
        
        scheduler = SmartTradingScheduler()
        
        if frequency_type == 'test':
            print("Testing all frequency types...")
            for freq in ['market_hours', 'daily', 'weekly', 'monthly', 'quarterly']:
                print(f"\n=== Testing {freq} execution ===")
                successful, failed = scheduler.run_manual_execution(freq)
                print(f"Result: {successful} successful, {failed} failed")
        else:
            successful, failed = scheduler.run_manual_execution(frequency_type)
            sys.exit(0 if failed == 0 else 1)
    else:
        # Start the full scheduler
        scheduler = SmartTradingScheduler()
        scheduler.start_scheduler()

if __name__ == "__main__":
    main()