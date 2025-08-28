#!/usr/bin/env python3
"""
ACIS Weekly/Monthly Pipeline - Low Frequency Data
Runs weekly/monthly data collection scripts for less frequent data
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

logger = setup_logger("weekly_pipeline")

@dataclass
class ScriptConfig:
    """Configuration for pipeline script"""
    name: str
    timeout: int
    critical: bool
    dependencies: tuple = ()
    description: str = ""
    frequency: str = "weekly"

class WeeklyPipelineRunner:
    """Weekly/Monthly pipeline runner for low-frequency data scripts"""
    
    def __init__(self, script_directory: Path):
        self.script_dir = script_directory
        self.results = {}
        
        # Define weekly/monthly scripts - data that changes infrequently
        # Order is critical: Fundamentals -> News -> Derived Analysis -> Rankings
        self.scripts = [
            # STEP 1: Fetch company overview (sector, industry, market cap)
            ScriptConfig(
                "fetch_company_overview.py",
                3600,
                False,  # Not critical but very valuable for analysis
                (),  # No dependencies - uses symbol_universe table
                "Fetch sector, industry, market cap and fundamental metrics",
                "weekly"
            ),
            
            # STEP 2: Fetch fundamental data (quarterly/annual reports)
            ScriptConfig(
                "fetch_fundamentals.py", 
                3600, 
                True,  # Critical - quality rankings depend on this
                ("fetch_company_overview.py",),  # Can use sector/industry from overview
                "Fetch company fundamentals (quarterly/annual data)",
                "weekly"
            ),
            
            # STEP 3: Fetch news sentiment (refreshed weekly)
            ScriptConfig(
                "fetch_news_sentiment.py",
                2400,  # 40 minutes - lots of API calls
                False,  # Not critical but valuable signal
                (),  # No dependencies - uses symbol_universe table
                "Fetch news sentiment and market narrative analysis",
                "weekly"
            ),
            
            # STEP 4: Extract dividend history from price data
            ScriptConfig(
                "fetch_dividend_history.py", 
                600, 
                False,  # Not critical but useful for analysis
                (),  # Uses stock_prices table which should exist from daily runs
                "Extract dividend data from accumulated price data",
                "weekly"
            ),
            
            # STEP 5: Compute forward returns for ML features
            ScriptConfig(
                "compute_forward_returns.py", 
                1200,  # Increased timeout for large dataset
                True,  # Critical for ML features and analysis
                (),  # Uses stock_prices table which should exist from daily runs
                "Calculate forward returns from accumulated price history",
                "weekly"
            ),
            
            # STEP 6: Calculate comprehensive quality rankings (all 7 components)
            ScriptConfig(
                "calculate_quality_rankings.py",
                1800,
                True,  # Critical - main output of the system
                ("fetch_fundamentals.py", "fetch_news_sentiment.py"),  # Depends on fundamentals and sentiment
                "Calculate 7-factor rankings: SP500, Cash Flow, Fundamentals, Sentiment, Value, Breakout, Growth",
                "weekly"
            ),
            
            # STEP 7: Generate historical rankings for ML training (if needed)
            ScriptConfig(
                "calculate_historical_rankings.py",
                7200,  # 2 hours - can be long for initial run
                False,  # Not critical for weekly operations
                ("calculate_quality_rankings.py",),  # Depends on current rankings being calculated
                "Generate historical rankings for ML/DL model training (skips existing dates)",
                "monthly"  # Run less frequently
            ),
            
            # STEP 8: Calculate forward returns for ML targets
            ScriptConfig(
                "calculate_forward_returns.py",
                1200,
                False,  # Not critical but needed for ML
                ("calculate_quality_rankings.py",),
                "Calculate forward returns for ML target variables",
                "weekly"
            ),
            
            # STEP 8: Train/update ML models
            ScriptConfig(
                "ml_strategy_framework.py",
                3600,  # 1 hour for model training
                False,  # Not critical for weekly operations
                ("calculate_forward_returns.py",),
                "Train ML models for return prediction and strategy generation",
                "monthly"  # Retrain monthly or on demand
            ),
            
            # STEP 9: Generate trading signals
            ScriptConfig(
                "generate_trading_signals.py",
                600,
                False,  # Not critical but valuable
                ("ml_strategy_framework.py",),
                "Generate ML-based trading signals and position recommendations",
                "weekly"
            ),
        ]
        
        logger.info(f"Initialized weekly pipeline with {len(self.scripts)} scripts")
    
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
        
        logger.info(f"Starting {script.name} [{script.frequency.upper()}]: {script.description}")
        start_time = time.time()
        
        try:
            # Run the script with UTF-8 encoding to handle Unicode properly
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(self.script_dir),
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',  # Replace undecodable chars instead of failing
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
        """Run the weekly pipeline"""
        pipeline_start = time.time()
        successful_scripts = 0
        failed_scripts = 0
        skipped_scripts = 0
        
        logger.info("[PIPELINE] Starting ACIS Weekly/Monthly Data Pipeline")
        logger.info(f"Pipeline mode: {'Skip non-critical on failure' if skip_non_critical else 'Run all scripts'}")
        logger.info("Processing low-frequency data: fundamentals, dividends, forward returns")
        
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
                    logger.error(f"[CRITICAL] Script {script.name} failed - stopping weekly pipeline")
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
        logger.info("[SUMMARY] ACIS WEEKLY/MONTHLY PIPELINE SUMMARY")
        logger.info("=" * 80)
        
        logger.info(f"Total Duration: {summary['duration']:.1f} seconds")
        logger.info(f"Scripts Successful: {summary['successful_scripts']}/{summary['total_scripts']}")
        logger.info(f"Scripts Failed: {summary['failed_scripts']}")
        logger.info(f"Scripts Skipped: {summary['skipped_scripts']}")
        
        logger.info("\nWeekly/Monthly Script Results:")
        for script_name, result in summary['results'].items():
            status = "[SUCCESS]" if result['success'] else "[FAILED]"
            duration = result['duration']
            logger.info(f"  {script_name}: {status} ({duration:.1f}s)")
            
            if not result['success'] and 'error' in result:
                logger.info(f"    Error: {result['error']}")
        
        overall_status = "[COMPLETE] WEEKLY PIPELINE COMPLETED" if summary['success'] else "[FAILED] WEEKLY PIPELINE FAILED"
        logger.info(f"\n{overall_status}")
        logger.info("=" * 80)

def main():
    """Main execution function"""
    start_time = time.time()
    log_script_start(logger, "weekly_pipeline", "ACIS weekly/monthly low-frequency data collection pipeline")
    
    parser = argparse.ArgumentParser(description="ACIS Weekly/Monthly Data Pipeline")
    parser.add_argument("--skip-non-critical", action="store_true", 
                       help="Skip non-critical scripts if any script fails")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be executed without running")
    
    args = parser.parse_args()
    
    # Get script directory
    script_dir = Path(__file__).parent
    
    try:
        runner = WeeklyPipelineRunner(script_dir)
        
        if args.dry_run:
            logger.info("DRY RUN MODE - Weekly/Monthly pipeline execution order:")
            for i, script in enumerate(runner.scripts, 1):
                deps = f" (depends on: {', '.join(script.dependencies)})" if script.dependencies else ""
                critical = " [CRITICAL]" if script.critical else ""
                freq = f" [{script.frequency.upper()}]"
                logger.info(f"  {i}. {script.name}{critical}{freq} - {script.description}{deps}")
            return
        
        # Run the weekly pipeline
        summary = runner.run_pipeline(skip_non_critical=args.skip_non_critical)
        runner.print_summary(summary)
        
        # Log final results
        duration = time.time() - start_time
        if summary['success']:
            log_script_end(logger, "weekly_pipeline", True, duration, {
                "Scripts completed": f"{summary['successful_scripts']}/{summary['total_scripts']}",
                "Failed scripts": summary['failed_scripts'],
                "Pipeline status": "SUCCESS",
                "Frequency": "Weekly/Monthly"
            })
        else:
            log_script_end(logger, "weekly_pipeline", False, duration, {
                "Scripts completed": f"{summary['successful_scripts']}/{summary['total_scripts']}",
                "Failed scripts": summary['failed_scripts'],
                "Pipeline status": "FAILED",
                "Frequency": "Weekly/Monthly"
            })
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Weekly pipeline execution failed: {e}")
        log_script_end(logger, "weekly_pipeline", False, time.time() - start_time)
        sys.exit(1)

if __name__ == "__main__":
    main()