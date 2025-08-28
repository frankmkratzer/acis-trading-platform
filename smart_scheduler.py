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
import psutil
import signal
import json
from datetime import datetime, timedelta, date
from pathlib import Path
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from logging_config import setup_logger, log_script_start, log_script_end
from typing import Dict, List, Tuple, Optional
import pandas as pd
import threading
from dataclasses import dataclass
from enum import Enum

load_dotenv()
logger = setup_logger("smart_scheduler")

POSTGRES_URL = os.getenv("POSTGRES_URL")
if not POSTGRES_URL:
    logger.error("POSTGRES_URL not set")
    sys.exit(1)

engine = create_engine(POSTGRES_URL)

# NYSE Market Holidays for 2025-2026
NYSE_HOLIDAYS = [
    # 2025 holidays
    date(2025, 1, 1),   # New Year's Day
    date(2025, 1, 20),  # Martin Luther King Jr. Day
    date(2025, 2, 17),  # Presidents' Day
    date(2025, 4, 18),  # Good Friday
    date(2025, 5, 26),  # Memorial Day
    date(2025, 6, 19),  # Juneteenth
    date(2025, 7, 4),   # Independence Day
    date(2025, 9, 1),   # Labor Day
    date(2025, 11, 27), # Thanksgiving
    date(2025, 12, 25), # Christmas
    # 2026 holidays (add more as needed)
    date(2026, 1, 1),   # New Year's Day
    date(2026, 1, 19),  # Martin Luther King Jr. Day
    date(2026, 2, 16),  # Presidents' Day
    date(2026, 4, 3),   # Good Friday
    date(2026, 5, 25),  # Memorial Day
    date(2026, 7, 3),   # Independence Day (observed)
    date(2026, 9, 7),   # Labor Day
    date(2026, 11, 26), # Thanksgiving
    date(2026, 12, 25), # Christmas
]

# Early close days (market closes at 1:00 PM ET)
EARLY_CLOSE_DAYS = [
    date(2025, 7, 3),   # Day before Independence Day
    date(2025, 11, 28), # Day after Thanksgiving
    date(2025, 12, 24), # Christmas Eve
]

class PipelineStatus(Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"
    RUNNING = "RUNNING"
    SKIPPED = "SKIPPED"

@dataclass
class PipelineConfig:
    """Configuration for a pipeline"""
    name: str
    script: str
    timeout: int = 7200  # 2 hours default
    max_retries: int = 2
    critical: bool = True
    dependencies: List[str] = None

class ProcessManager:
    """Manage subprocess lifecycle and cleanup"""
    
    def __init__(self):
        self.processes: Dict[int, subprocess.Popen] = {}
        self.lock = threading.Lock()
    
    def register(self, process: subprocess.Popen):
        """Register a subprocess for tracking"""
        with self.lock:
            self.processes[process.pid] = process
            logger.debug(f"Registered process {process.pid}")
    
    def unregister(self, pid: int):
        """Unregister a subprocess"""
        with self.lock:
            if pid in self.processes:
                del self.processes[pid]
                logger.debug(f"Unregistered process {pid}")
    
    def cleanup_zombie_processes(self):
        """Clean up any zombie processes"""
        with self.lock:
            for pid, proc in list(self.processes.items()):
                try:
                    if proc.poll() is not None:  # Process has terminated
                        logger.info(f"Cleaning up terminated process {pid}")
                        self.unregister(pid)
                except Exception as e:
                    logger.warning(f"Error checking process {pid}: {e}")
    
    def terminate_all(self):
        """Terminate all managed processes"""
        with self.lock:
            for pid, proc in list(self.processes.items()):
                try:
                    logger.warning(f"Terminating process {pid}")
                    proc.terminate()
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Force killing process {pid}")
                    proc.kill()
                except Exception as e:
                    logger.error(f"Error terminating process {pid}: {e}")
                finally:
                    self.unregister(pid)

class SmartScheduler:
    """World-class smart scheduler for ACIS pipelines"""
    
    def __init__(self):
        self.today = datetime.now().date()
        self.weekday = self.today.weekday()  # 0=Monday, 6=Sunday
        self.script_dir = Path(__file__).parent
        self.process_manager = ProcessManager()
        self.execution_lock = threading.Lock()
        self.running_pipelines: Dict[str, datetime] = {}
    
    def is_trading_day(self, check_date=None):
        """Check if date is a NYSE trading day"""
        if check_date is None:
            check_date = self.today
        
        # Weekends are not trading days
        if check_date.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        
        # Check if it's a holiday
        if check_date in NYSE_HOLIDAYS:
            return False
        
        return True
    
    def is_early_close_day(self, check_date=None):
        """Check if it's an early close day"""
        if check_date is None:
            check_date = self.today
        return check_date in EARLY_CLOSE_DAYS
    
    def get_next_trading_day(self, from_date=None):
        """Get the next trading day from the given date"""
        if from_date is None:
            from_date = self.today
        
        next_day = from_date + timedelta(days=1)
        while not self.is_trading_day(next_day):
            next_day += timedelta(days=1)
        
        return next_day
    
    def get_previous_trading_day(self, from_date=None):
        """Get the previous trading day from the given date"""
        if from_date is None:
            from_date = self.today
        
        prev_day = from_date - timedelta(days=1)
        while not self.is_trading_day(prev_day):
            prev_day -= timedelta(days=1)
        
        return prev_day
    
    def get_last_run_date(self, pipeline_type):
        """Get the last successful run date for a pipeline type"""
        try:
            with engine.begin() as conn:
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
            with engine.begin() as conn:
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
    
    def check_pipeline_conflicts(self, pipeline_name: str) -> bool:
        """Check if pipeline can run without conflicts"""
        with self.execution_lock:
            # Check if already running
            if pipeline_name in self.running_pipelines:
                runtime = (datetime.now() - self.running_pipelines[pipeline_name]).total_seconds()
                logger.warning(f"Pipeline {pipeline_name} already running for {runtime:.0f}s")
                return False
            
            # Check for dependency conflicts
            if pipeline_name == 'weekly' and 'daily' in self.running_pipelines:
                logger.warning("Cannot run weekly while daily is running")
                return False
            
            return True
    
    def run_pipeline_with_retry(self, pipeline_config: PipelineConfig, args=None) -> Tuple[bool, float]:
        """Run pipeline with retry logic and proper process management"""
        for attempt in range(pipeline_config.max_retries + 1):
            if attempt > 0:
                logger.info(f"Retry attempt {attempt} for {pipeline_config.name}")
            
            success, duration = self.run_pipeline(pipeline_config.name, args, pipeline_config.timeout)
            
            if success:
                return True, duration
            
            if attempt < pipeline_config.max_retries:
                wait_time = min(60 * (2 ** attempt), 300)  # Max 5 min wait
                logger.info(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
        
        return False, 0
    
    def run_pipeline(self, pipeline_name: str, args=None, timeout: int = 7200) -> Tuple[bool, float]:
        """Run a pipeline with proper process management and monitoring"""
        script_path = self.script_dir / f"run_{pipeline_name}_pipeline.py"
        
        if not script_path.exists():
            logger.error(f"Pipeline script not found: {script_path}")
            return False, 0
        
        # Check for conflicts
        if not self.check_pipeline_conflicts(pipeline_name):
            return False, 0
        
        # Mark as running
        with self.execution_lock:
            self.running_pipelines[pipeline_name] = datetime.now()
        
        cmd = [sys.executable, str(script_path)]
        if args:
            cmd.extend(args)
        
        logger.info(f"Executing {pipeline_name} pipeline...")
        start_time = time.time()
        
        try:
            # Clean up any zombie processes first
            self.process_manager.cleanup_zombie_processes()
            
            # Start process with proper management
            process = subprocess.Popen(
                cmd,
                cwd=str(self.script_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Register process for management
            self.process_manager.register(process)
            
            # Monitor process with timeout
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                return_code = process.returncode
                
            except subprocess.TimeoutExpired:
                logger.error(f"Pipeline {pipeline_name} timed out after {timeout}s")
                
                # Try graceful termination first
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Force killing {pipeline_name} process")
                    process.kill()
                    process.wait()
                
                duration = time.time() - start_time
                self.log_pipeline_run(pipeline_name, PipelineStatus.TIMEOUT.value, duration)
                return False, duration
            
            finally:
                self.process_manager.unregister(process.pid)
            
            duration = time.time() - start_time
            success = return_code == 0
            
            if success:
                logger.info(f"{pipeline_name.title()} pipeline completed successfully in {duration:.1f}s")
            else:
                logger.error(f"{pipeline_name.title()} pipeline failed with return code {return_code}")
                if stderr:
                    logger.error(f"STDERR: {stderr[:1000]}...")  # Limit error output
            
            # Log the run
            self.log_pipeline_run(
                pipeline_name, 
                PipelineStatus.SUCCESS.value if success else PipelineStatus.FAILED.value,
                duration
            )
            
            return success, duration
            
        except Exception as e:
            duration = time.time() - start_time
            logger.exception(f"{pipeline_name.title()} pipeline failed with exception: {e}")
            self.log_pipeline_run(pipeline_name, PipelineStatus.ERROR.value, duration)
            return False, duration
        
        finally:
            # Remove from running pipelines
            with self.execution_lock:
                if pipeline_name in self.running_pipelines:
                    del self.running_pipelines[pipeline_name]
    
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