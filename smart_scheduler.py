#!/usr/bin/env python3
"""
Smart Pipeline Scheduler
Determines when daily vs weekly pipelines should run based on:
- Day of week (Friday for weekly)
- Last run dates in database
- Market calendar (trading days)
- Force flags for manual execution
"""

import os
import sys
import time
import argparse
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from logging_config import setup_logger, log_script_start, log_script_end

load_dotenv()
logger = setup_logger("smart_scheduler")

POSTGRES_URL = os.getenv("POSTGRES_URL")
if not POSTGRES_URL:
    logger.error("POSTGRES_URL not set")
    sys.exit(1)

engine = create_engine(POSTGRES_URL)

class SmartScheduler:
    """Smart scheduler for ACIS pipelines"""
    
    def __init__(self):
        self.today = datetime.now().date()
        self.weekday = self.today.weekday()  # 0=Monday, 6=Sunday
        self.script_dir = Path(__file__).parent
    
    def is_trading_day(self, date=None):
        """Check if date is a trading day (weekday, excluding major holidays)"""
        if date is None:
            date = self.today
        
        # Weekends are not trading days
        weekday = date.weekday()
        if weekday >= 5:  # Saturday=5, Sunday=6
            return False
        
        # TODO: Add major holiday checks (NYSE calendar)
        # For now, just check weekdays
        return True
    
    def get_last_run_date(self, pipeline_type):
        """Get the last successful run date for a pipeline type"""
        try:
            with engine.connect() as conn:
                # Create scheduler log table if it doesn't exist
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS pipeline_scheduler_log (
                        id SERIAL PRIMARY KEY,
                        pipeline_type TEXT NOT NULL,
                        run_date DATE NOT NULL,
                        status TEXT NOT NULL,
                        duration_seconds NUMERIC,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                # Get last successful run
                result = conn.execute(text("""
                    SELECT MAX(run_date) 
                    FROM pipeline_scheduler_log 
                    WHERE pipeline_type = :pipeline_type 
                      AND status = 'SUCCESS'
                """), {'pipeline_type': pipeline_type})
                
                last_run = result.scalar()
                return last_run
                
        except Exception as e:
            logger.warning(f"Could not get last run date for {pipeline_type}: {e}")
            return None
    
    def log_pipeline_run(self, pipeline_type, status, duration=None):
        """Log pipeline execution to scheduler log"""
        try:
            with engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO pipeline_scheduler_log 
                    (pipeline_type, run_date, status, duration_seconds)
                    VALUES (:pipeline_type, :run_date, :status, :duration)
                """), {
                    'pipeline_type': pipeline_type,
                    'run_date': self.today,
                    'status': status,
                    'duration': duration
                })
                conn.commit()
        except Exception as e:
            logger.warning(f"Could not log pipeline run: {e}")
    
    def should_run_daily(self, force=False):
        """Determine if daily pipeline should run"""
        if force:
            logger.info("Daily pipeline: FORCED RUN requested")
            return True, "Forced execution"
        
        # Always run on trading days
        if not self.is_trading_day():
            return False, f"Non-trading day ({self.today.strftime('%A')})"
        
        # Check last run
        last_run = self.get_last_run_date('daily')
        if last_run is None:
            return True, "No previous daily run found"
        
        if last_run < self.today:
            return True, f"Last run: {last_run} (needs update)"
        
        return False, f"Already ran today ({last_run})"
    
    def should_run_weekly(self, force=False):
        """Determine if weekly pipeline should run"""
        if force:
            logger.info("Weekly pipeline: FORCED RUN requested")
            return True, "Forced execution"
        
        # Weekly runs on Fridays (4 = Friday)
        if self.weekday != 4:
            return False, f"Not Friday (today is {self.today.strftime('%A')})"
        
        # Must be a trading Friday
        if not self.is_trading_day():
            return False, f"Non-trading Friday ({self.today})"
        
        # Check last run
        last_run = self.get_last_run_date('weekly')
        if last_run is None:
            return True, "No previous weekly run found"
        
        # Don't run if we already ran this week
        days_since_last = (self.today - last_run).days
        if days_since_last < 7:
            return False, f"Already ran this week ({last_run}, {days_since_last} days ago)"
        
        return True, f"Last run: {last_run} ({days_since_last} days ago)"
    
    def run_pipeline(self, pipeline_name, args=None):
        """Run a pipeline and return results"""
        script_path = self.script_dir / f"run_{pipeline_name}_pipeline.py"
        
        if not script_path.exists():
            logger.error(f"Pipeline script not found: {script_path}")
            return False, 0
        
        cmd = [sys.executable, str(script_path)]
        if args:
            cmd.extend(args)
        
        logger.info(f"Executing {pipeline_name} pipeline...")
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.script_dir),
                capture_output=True,
                text=True,
                timeout=7200,  # 2 hour timeout
                check=False
            )
            
            duration = time.time() - start_time
            success = result.returncode == 0
            
            if success:
                logger.info(f"✅ {pipeline_name.title()} pipeline completed successfully in {duration:.1f}s")
            else:
                logger.error(f"❌ {pipeline_name.title()} pipeline failed with return code {result.returncode}")
                if result.stderr:
                    logger.error(f"STDERR: {result.stderr}")
            
            # Log the run
            self.log_pipeline_run(pipeline_name, 'SUCCESS' if success else 'FAILED', duration)
            
            return success, duration
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            logger.error(f"⏰ {pipeline_name.title()} pipeline timed out after 2 hours")
            self.log_pipeline_run(pipeline_name, 'TIMEOUT', duration)
            return False, duration
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"💥 {pipeline_name.title()} pipeline failed with exception: {e}")
            self.log_pipeline_run(pipeline_name, 'ERROR', duration)
            return False, duration
    
    def execute_scheduled_pipelines(self, force_daily=False, force_weekly=False, 
                                  skip_non_critical=False, dry_run=False):
        """Execute pipelines based on schedule"""
        
        # Check what should run
        run_daily, daily_reason = self.should_run_daily(force_daily)
        run_weekly, weekly_reason = self.should_run_weekly(force_weekly)
        
        logger.info("=" * 80)
        logger.info("📅 ACIS SMART SCHEDULER ANALYSIS")
        logger.info("=" * 80)
        logger.info(f"Date: {self.today} ({self.today.strftime('%A')})")
        logger.info(f"Trading Day: {'Yes' if self.is_trading_day() else 'No'}")
        logger.info("")
        logger.info(f"Daily Pipeline: {'🟢 RUN' if run_daily else '🔴 SKIP'} - {daily_reason}")
        logger.info(f"Weekly Pipeline: {'🟢 RUN' if run_weekly else '🔴 SKIP'} - {weekly_reason}")
        logger.info("=" * 80)
        
        if dry_run:
            logger.info("DRY RUN MODE - No pipelines will be executed")
            return True
        
        if not run_daily and not run_weekly:
            logger.info("📋 No pipelines scheduled to run today")
            return True
        
        # Prepare arguments
        args = []
        if skip_non_critical:
            args.append("--skip-non-critical")
        
        # Execute pipelines
        results = {}
        
        if run_daily:
            logger.info("🌅 Executing scheduled daily pipeline...")
            success, duration = self.run_pipeline("daily", args)
            results['daily'] = {'success': success, 'duration': duration}
            
            # If daily fails and it's also a weekly day, skip weekly
            if not success and run_weekly:
                logger.warning("❌ Daily pipeline failed - skipping weekly pipeline")
                run_weekly = False
        
        if run_weekly:
            logger.info("📅 Executing scheduled weekly pipeline...")  
            success, duration = self.run_pipeline("weekly", args)
            results['weekly'] = {'success': success, 'duration': duration}
        
        # Summary
        logger.info("=" * 80)
        logger.info("🎯 SCHEDULED EXECUTION SUMMARY")
        logger.info("=" * 80)
        
        all_success = True
        for pipeline, result in results.items():
            status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
            logger.info(f"{pipeline.title()} Pipeline: {status} ({result['duration']:.1f}s)")
            if not result['success']:
                all_success = False
        
        if all_success:
            logger.info("🎉 ALL SCHEDULED PIPELINES COMPLETED SUCCESSFULLY")
        else:
            logger.info("💥 ONE OR MORE SCHEDULED PIPELINES FAILED")
        
        return all_success

def main():
    """Main execution function"""
    start_time = time.time()
    log_script_start(logger, "smart_scheduler", "Smart pipeline scheduler for ACIS data collection")
    
    parser = argparse.ArgumentParser(description="ACIS Smart Pipeline Scheduler")
    parser.add_argument("--force-daily", action="store_true",
                       help="Force daily pipeline to run regardless of schedule")
    parser.add_argument("--force-weekly", action="store_true", 
                       help="Force weekly pipeline to run regardless of schedule")
    parser.add_argument("--skip-non-critical", action="store_true",
                       help="Skip non-critical scripts in pipelines")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be executed without running")
    
    args = parser.parse_args()
    
    try:
        scheduler = SmartScheduler()
        
        success = scheduler.execute_scheduled_pipelines(
            force_daily=args.force_daily,
            force_weekly=args.force_weekly,
            skip_non_critical=args.skip_non_critical,
            dry_run=args.dry_run
        )
        
        duration = time.time() - start_time
        if success:
            log_script_end(logger, "smart_scheduler", True, duration, {
                "Scheduler status": "SUCCESS",
                "Execution mode": "DRY RUN" if args.dry_run else "LIVE"
            })
        else:
            log_script_end(logger, "smart_scheduler", False, duration)
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Smart scheduler execution failed: {e}")
        log_script_end(logger, "smart_scheduler", False, time.time() - start_time)
        sys.exit(1)

if __name__ == "__main__":
    main()