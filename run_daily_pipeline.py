#!/usr/bin/env python3
"""
ACIS Daily Pipeline - High Frequency Data
Runs daily market data collection scripts
"""

import os
import sys
import time
import subprocess
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
from logging_config import setup_logger, log_script_start, log_script_end

logger = setup_logger("daily_pipeline")

@dataclass
class ScriptConfig:
    """Configuration for pipeline script"""
    name: str
    timeout: int
    critical: bool
    dependencies: tuple = ()
    description: str = ""

class DailyPipelineRunner:
    """Daily pipeline runner for high-frequency data scripts"""
    
    def __init__(self, script_directory: Path):
        self.script_dir = script_directory
        self.results = {}
        
        # Define daily scripts - data that changes every trading day
        self.scripts = [
            ScriptConfig(
                "fetch_symbol_universe.py",
                600,
                True, 
                (), 
                "Update US stock symbol universe (NYSE, NASDAQ, AMEX)"
            ),
            ScriptConfig(
                "fetch_sp500_history.py", 
                300, 
                False, 
                ("fetch_symbol_universe.py",), 
                "Fetch S&P 500 price history for benchmarking"
            ),
            ScriptConfig(
                "fetch_prices.py", 
                2400, 
                True, 
                ("fetch_symbol_universe.py",), 
                "Ultra-premium price fetcher (Alpha Vantage 600 calls/min)"
            ),
            ScriptConfig(
                "fetch_options.py",
                1800,
                False,  # Not critical - options are supplemental data
                ("fetch_symbol_universe.py",),
                "Fetch real-time and historical options data with Greeks"
            ),
            ScriptConfig(
                "fetch_technical_indicators.py", 
                3600, 
                True, 
                ("fetch_prices.py",), 
                "Calculate technical indicators locally from price data"
            ),
            ScriptConfig(
                "calculate_quality_rankings.py",
                1800,
                False,  # Not critical - rankings are derived data
                ("fetch_prices.py", "fetch_sp500_history.py"),  # Needs prices and SP500 data
                "Calculate stock quality rankings (SP500 outperformance, FCF, fundamentals)"
            ),
        ]
        
        logger.info(f"Initialized daily pipeline with {len(self.scripts)} scripts")
    
    def check_dependencies(self, script: ScriptConfig) -> bool:
        """Check if all dependencies completed successfully"""
        if not script.dependencies:
            return True
            
        for dep in script.dependencies:
            dep_result = self.results.get(dep)
            if not dep_result or not dep_result['success']:
                logger.warning(f"{script.name} dependency {dep} failed or not completed")
                return False
        
        return True
    
    def run_script(self, script: ScriptConfig) -> dict:
        """Run a single script and return results"""
        script_path = self.script_dir / script.name
        
        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            return {
                'success': False,
                'duration': 0,
                'error': f"Script file not found: {script.name}"
            }
        
        # Check dependencies
        if not self.check_dependencies(script):
            return {
                'success': False,
                'duration': 0,
                'error': "Dependencies not satisfied"
            }
        
        logger.info(f"Starting {script.name}: {script.description}")
        start_time = time.time()
        
        try:
            # Run the script
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(self.script_dir),
                capture_output=True,
                text=True,
                timeout=script.timeout,
                check=False
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                logger.info(f"[SUCCESS] {script.name} completed successfully in {duration:.1f}s")
                return {
                    'success': True,
                    'duration': duration,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'returncode': result.returncode
                }
            else:
                logger.error(f"[FAILED] {script.name} failed with return code {result.returncode}")
                if result.stderr:
                    logger.error(f"STDERR: {result.stderr}")
                return {
                    'success': False,
                    'duration': duration,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'returncode': result.returncode,
                    'error': f"Script failed with return code {result.returncode}"
                }
                
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            logger.error(f"[TIMEOUT] {script.name} timed out after {script.timeout}s")
            return {
                'success': False,
                'duration': duration,
                'error': f"Script timed out after {script.timeout} seconds"
            }
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"[EXCEPTION] {script.name} failed with exception: {e}")
            return {
                'success': False,
                'duration': duration,
                'error': str(e)
            }
    
    def run_pipeline(self, skip_non_critical: bool = False) -> dict:
        """Run the daily pipeline"""
        pipeline_start = time.time()
        successful_scripts = 0
        failed_scripts = 0
        skipped_scripts = 0
        
        logger.info("[PIPELINE] Starting ACIS Daily Data Pipeline")
        logger.info(f"Pipeline mode: {'Skip non-critical on failure' if skip_non_critical else 'Run all scripts'}")
        
        for script in self.scripts:
            # Skip non-critical scripts if we have failures and skip mode is enabled
            if skip_non_critical and not script.critical and failed_scripts > 0:
                logger.info(f"[SKIP] Skipping non-critical script {script.name} due to previous failures")
                skipped_scripts += 1
                continue
            
            # Run the script
            result = self.run_script(script)
            self.results[script.name] = result
            
            if result['success']:
                successful_scripts += 1
            else:
                failed_scripts += 1
                
                # Stop pipeline if critical script fails
                if script.critical:
                    logger.error(f"[CRITICAL] Script {script.name} failed - stopping daily pipeline")
                    break
        
        pipeline_duration = time.time() - pipeline_start
        
        # Generate summary
        summary = {
            'success': failed_scripts == 0,
            'duration': pipeline_duration,
            'successful_scripts': successful_scripts,
            'failed_scripts': failed_scripts,
            'skipped_scripts': skipped_scripts,
            'total_scripts': len(self.scripts),
            'results': self.results
        }
        
        return summary
    
    def print_summary(self, summary: dict):
        """Print pipeline execution summary"""
        logger.info("=" * 80)
        logger.info("[SUMMARY] ACIS DAILY PIPELINE SUMMARY")
        logger.info("=" * 80)
        
        logger.info(f"Total Duration: {summary['duration']:.1f} seconds")
        logger.info(f"Scripts Successful: {summary['successful_scripts']}/{summary['total_scripts']}")
        logger.info(f"Scripts Failed: {summary['failed_scripts']}")
        logger.info(f"Scripts Skipped: {summary['skipped_scripts']}")
        
        logger.info("\nDaily Script Results:")
        for script_name, result in summary['results'].items():
            status = "[SUCCESS]" if result['success'] else "[FAILED]"
            duration = result['duration']
            logger.info(f"  {script_name}: {status} ({duration:.1f}s)")
            
            if not result['success'] and 'error' in result:
                logger.info(f"    Error: {result['error']}")
        
        overall_status = "[COMPLETE] DAILY PIPELINE COMPLETED" if summary['success'] else "[FAILED] DAILY PIPELINE FAILED"
        logger.info(f"\n{overall_status}")
        logger.info("=" * 80)

def main():
    """Main execution function"""
    start_time = time.time()
    log_script_start(logger, "daily_pipeline", "ACIS daily market data collection pipeline")
    
    parser = argparse.ArgumentParser(description="ACIS Daily Market Data Pipeline")
    parser.add_argument("--skip-non-critical", action="store_true", 
                       help="Skip non-critical scripts if any script fails")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be executed without running")
    
    args = parser.parse_args()
    
    # Get script directory
    script_dir = Path(__file__).parent
    
    try:
        runner = DailyPipelineRunner(script_dir)
        
        if args.dry_run:
            logger.info("DRY RUN MODE - Daily pipeline execution order:")
            for i, script in enumerate(runner.scripts, 1):
                deps = f" (depends on: {', '.join(script.dependencies)})" if script.dependencies else ""
                critical = " [CRITICAL]" if script.critical else ""
                logger.info(f"  {i}. {script.name}{critical} - {script.description}{deps}")
            return
        
        # Run the daily pipeline
        summary = runner.run_pipeline(skip_non_critical=args.skip_non_critical)
        runner.print_summary(summary)
        
        # Log final results
        duration = time.time() - start_time
        if summary['success']:
            log_script_end(logger, "daily_pipeline", True, duration, {
                "Scripts completed": f"{summary['successful_scripts']}/{summary['total_scripts']}",
                "Failed scripts": summary['failed_scripts'],
                "Pipeline status": "SUCCESS",
                "Frequency": "Daily"
            })
        else:
            log_script_end(logger, "daily_pipeline", False, duration, {
                "Scripts completed": f"{summary['successful_scripts']}/{summary['total_scripts']}",
                "Failed scripts": summary['failed_scripts'],
                "Pipeline status": "FAILED",
                "Frequency": "Daily"
            })
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Daily pipeline execution failed: {e}")
        log_script_end(logger, "daily_pipeline", False, time.time() - start_time)
        sys.exit(1)

if __name__ == "__main__":
    main()