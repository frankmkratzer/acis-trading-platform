#!/usr/bin/env python3
"""
ACIS Full Pipeline Wrapper
Runs both daily and weekly pipelines in sequence
For production scheduling, use the individual pipelines directly
"""

import os
import sys
import time
import subprocess
import argparse
from pathlib import Path
from logging_config import setup_logger, log_script_start, log_script_end

logger = setup_logger("full_pipeline_wrapper")

def run_pipeline_script(script_name: str, script_dir: Path, args: list = None) -> dict:
    """Run a pipeline script and return results"""
    script_path = script_dir / script_name
    
    if not script_path.exists():
        logger.error(f"Pipeline script not found: {script_path}")
        return {'success': False, 'error': f"Script not found: {script_name}"}
    
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
    
    logger.info(f"Running {script_name}...")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(script_dir),
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout for full pipelines
            check=False
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            logger.info(f"✅ {script_name} completed successfully in {duration:.1f}s")
            return {
                'success': True,
                'duration': duration,
                'returncode': result.returncode
            }
        else:
            logger.error(f"❌ {script_name} failed with return code {result.returncode}")
            return {
                'success': False,
                'duration': duration,
                'returncode': result.returncode,
                'stderr': result.stderr,
                'error': f"Pipeline failed with return code {result.returncode}"
            }
    
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        logger.error(f"⏰ {script_name} timed out after 2 hours")
        return {
            'success': False,
            'duration': duration,
            'error': "Pipeline timed out after 2 hours"
        }
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"💥 {script_name} failed with exception: {e}")
        return {
            'success': False,
            'duration': duration,
            'error': str(e)
        }

def main():
    """Main execution function"""
    start_time = time.time()
    log_script_start(logger, "full_pipeline_wrapper", "Run both daily and weekly ACIS pipelines")
    
    parser = argparse.ArgumentParser(description="ACIS Full Pipeline (Daily + Weekly)")
    parser.add_argument("--daily-only", action="store_true",
                       help="Run only the daily pipeline")
    parser.add_argument("--weekly-only", action="store_true", 
                       help="Run only the weekly pipeline")
    parser.add_argument("--skip-non-critical", action="store_true",
                       help="Skip non-critical scripts in both pipelines")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be executed without running")
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    pipeline_args = []
    
    if args.skip_non_critical:
        pipeline_args.append("--skip-non-critical")
    if args.dry_run:
        pipeline_args.append("--dry-run")
    
    try:
        results = {}
        
        # Determine which pipelines to run
        run_daily = not args.weekly_only
        run_weekly = not args.daily_only
        
        if args.dry_run:
            logger.info("DRY RUN MODE - Full pipeline execution plan:")
            if run_daily:
                logger.info("1. Daily Pipeline (run_daily_pipeline.py)")
                logger.info("   - fetch_symbol_universe.py")
                logger.info("   - fetch_sp500_history.py")
                logger.info("   - fetch_prices.py")
                logger.info("   - fetch_technical_indicators.py")
            if run_weekly:
                logger.info("2. Weekly Pipeline (run_weekly_pipeline.py)")
                logger.info("   - fetch_fundamentals.py")
                logger.info("   - fetch_dividend_history.py")
                logger.info("   - compute_forward_returns.py")
            return
        
        # Run daily pipeline
        if run_daily:
            logger.info("🌅 Starting Daily Pipeline...")
            results['daily'] = run_pipeline_script("run_daily_pipeline.py", script_dir, pipeline_args)
            
            if not results['daily']['success']:
                logger.error("❌ Daily pipeline failed - stopping execution")
                log_script_end(logger, "full_pipeline_wrapper", False, time.time() - start_time, {
                    "Daily pipeline": "FAILED",
                    "Weekly pipeline": "NOT STARTED"
                })
                sys.exit(1)
        
        # Run weekly pipeline
        if run_weekly:
            logger.info("📅 Starting Weekly Pipeline...")
            results['weekly'] = run_pipeline_script("run_weekly_pipeline.py", script_dir, pipeline_args)
        
        # Summary
        duration = time.time() - start_time
        all_success = all(result['success'] for result in results.values())
        
        logger.info("=" * 80)
        logger.info("🚀 FULL PIPELINE SUMMARY")
        logger.info("=" * 80)
        
        total_duration = sum(result['duration'] for result in results.values())
        
        if 'daily' in results:
            status = "✅ SUCCESS" if results['daily']['success'] else "❌ FAILED"
            logger.info(f"Daily Pipeline: {status} ({results['daily']['duration']:.1f}s)")
        
        if 'weekly' in results:
            status = "✅ SUCCESS" if results['weekly']['success'] else "❌ FAILED"
            logger.info(f"Weekly Pipeline: {status} ({results['weekly']['duration']:.1f}s)")
        
        logger.info(f"Total Pipeline Duration: {total_duration:.1f}s")
        
        if all_success:
            logger.info("🎉 ALL PIPELINES COMPLETED SUCCESSFULLY")
            log_script_end(logger, "full_pipeline_wrapper", True, duration, {
                "Daily pipeline": "SUCCESS" if 'daily' in results else "SKIPPED",
                "Weekly pipeline": "SUCCESS" if 'weekly' in results else "SKIPPED",
                "Total duration": f"{total_duration:.1f}s"
            })
        else:
            logger.info("💥 ONE OR MORE PIPELINES FAILED")
            log_script_end(logger, "full_pipeline_wrapper", False, duration, {
                "Daily pipeline": "SUCCESS" if results.get('daily', {}).get('success') else "FAILED",
                "Weekly pipeline": "SUCCESS" if results.get('weekly', {}).get('success') else "FAILED"
            })
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Full pipeline execution failed: {e}")
        log_script_end(logger, "full_pipeline_wrapper", False, time.time() - start_time)
        sys.exit(1)

if __name__ == "__main__":
    main()