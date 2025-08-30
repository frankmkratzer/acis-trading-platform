#!/usr/bin/env python3
"""
ACIS Trading Platform - Main Entry Point
Simplified orchestrator for three-portfolio strategy
"""

import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from utils.logging_config import setup_logger

logger = setup_logger("acis_main")


def print_banner():
    """Print ACIS banner"""
    banner = """
    ================================================================
                      ACIS TRADING PLATFORM                      
         Three-Portfolio Strategy (VALUE/GROWTH/DIVIDEND)       
    ================================================================
      Clean Architecture Edition - v3.0                         
    ================================================================
    """
    print(banner)


def run_command(script_path, description):
    """Run a Python script and return its exit code"""
    print(f"\n[INFO] Running: {description}")
    logger.info(f"Running: {description}")
    cmd = [sys.executable, script_path]
    # Show output directly to console
    result = subprocess.run(cmd, cwd=Path(__file__).parent, capture_output=False, text=True)
    if result.returncode == 0:
        print(f"[SUCCESS] Completed: {description}")
        logger.info(f"[SUCCESS] Completed: {description}")
    else:
        print(f"[FAILED] Error in: {description}")
        logger.error(f"[FAILED] Error in: {description}")
    return result.returncode


def run_data_fetch(args):
    """Run data fetching operations"""
    logger.info("Starting data fetch operations...")
    
    if args.fetch_type == "stocks":
        return run_command("data_fetch/market_data/fetch_quality_stocks.py", 
                          "Fetching quality stocks ($2B+ market cap)")
    elif args.fetch_type == "prices":
        return run_command("data_fetch/market_data/fetch_prices.py",
                          "Fetching stock prices")
    elif args.fetch_type == "sp500":
        return run_command("data_fetch/market_data/fetch_sp500_history.py",
                          "Fetching S&P 500 history")
    elif args.fetch_type == "overview":
        return run_command("data_fetch/fundamentals/fetch_company_overview.py",
                          "Fetching company overviews")
    elif args.fetch_type == "fundamentals":
        return run_command("data_fetch/fundamentals/fetch_fundamentals.py",
                          "Fetching financial statements")
    elif args.fetch_type == "dividends":
        return run_command("data_fetch/fundamentals/fetch_dividend_history.py",
                          "Fetching dividend history")
    elif args.fetch_type == "all":
        # Run all in sequence
        scripts = [
            ("data_fetch/market_data/fetch_quality_stocks.py", "quality stocks"),
            ("data_fetch/market_data/fetch_sp500_history.py", "S&P 500 history"),
            ("data_fetch/market_data/fetch_prices.py", "stock prices"),
            ("data_fetch/fundamentals/fetch_company_overview.py", "company overviews"),
            ("data_fetch/fundamentals/fetch_fundamentals.py", "fundamentals"),
            ("data_fetch/fundamentals/fetch_dividend_history.py", "dividends")
        ]
        for script, desc in scripts:
            if run_command(script, f"Fetching {desc}") != 0:
                logger.warning(f"Failed to fetch {desc}, continuing...")
        return 0
    else:
        logger.error(f"Unknown fetch type: {args.fetch_type}")
        return 1


def run_analysis(args):
    """Run analysis operations"""
    logger.info("Starting analysis operations...")
    
    if args.analysis_type == "portfolio":
        return run_command("portfolios/portfolio_manager.py",
                          "Managing portfolios")
    elif args.analysis_type == "breakout":
        return run_command("analysis/breakout_detector.py",
                          "Detecting breakout signals")
    elif args.analysis_type == "excess_cf":
        return run_command("analysis/excess_cash_flow.py",
                          "Calculating excess cash flow")
    elif args.analysis_type == "dividend":
        return run_command("analysis/dividend_sustainability.py",
                          "Analyzing dividend sustainability")
    else:
        logger.error(f"Unknown analysis type: {args.analysis_type}")
        return 1


def run_pipeline(args):
    """Run complete pipeline"""
    logger.info(f"Starting {args.pipeline_type} pipeline...")
    
    if args.pipeline_type == "daily":
        return run_command("pipelines/run_daily_pipeline_clean.py",
                          "Daily pipeline")
    elif args.pipeline_type == "weekly":
        return run_command("pipelines/run_weekly_pipeline_clean.py",
                          "Weekly pipeline")
    else:
        logger.error(f"Unknown pipeline type: {args.pipeline_type}")
        return 1


def run_database(args):
    """Run database operations"""
    logger.info(f"Running database operation: {args.db_operation}")
    
    if args.db_operation == "setup":
        return run_command("database/setup_schema_clean.py",
                          "Setting up database schema")
    elif args.db_operation == "verify":
        return run_command("database/verify_tables.py",
                          "Verifying database tables")
    else:
        logger.error(f"Unknown database operation: {args.db_operation}")
        return 1


def run_first_time():
    """Run first-time setup"""
    logger.info("Running first-time setup...")
    return run_command("run_first_time.py", "First-time setup")


def main():
    """Main entry point"""
    print_banner()
    
    parser = argparse.ArgumentParser(
        description="ACIS Trading Platform - Main Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # First-time setup
  python main.py setup
  
  # Run daily pipeline
  python main.py pipeline --type daily
  
  # Fetch all data
  python main.py fetch --type all
  
  # Fetch specific data
  python main.py fetch --type prices
  python main.py fetch --type fundamentals
  
  # Calculate portfolio scores
  python main.py analysis --type portfolio
  
  # Verify database
  python main.py database --operation verify
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Setup Command (first-time)
    setup_parser = subparsers.add_parser("setup", help="Run first-time setup")
    
    # Data Fetch Command
    fetch_parser = subparsers.add_parser("fetch", help="Run data fetching operations")
    fetch_parser.add_argument("--type", 
                            choices=["stocks", "prices", "sp500", "overview", 
                                   "fundamentals", "dividends", "all"],
                            default="all", 
                            dest="fetch_type",
                            help="Type of data to fetch")
    
    # Analysis Command
    analysis_parser = subparsers.add_parser("analysis", help="Run analysis")
    analysis_parser.add_argument("--type", 
                                choices=["portfolio", "breakout", "excess_cf", "dividend"],
                                default="portfolio",
                                dest="analysis_type", 
                                help="Analysis type")
    
    # Pipeline Command
    pipe_parser = subparsers.add_parser("pipeline", help="Run complete pipelines")
    pipe_parser.add_argument("--type", 
                           choices=["daily", "weekly"],
                           default="daily", 
                           dest="pipeline_type",
                           help="Pipeline type")
    
    # Database Command
    db_parser = subparsers.add_parser("database", help="Database operations")
    db_parser.add_argument("--operation", 
                         choices=["setup", "verify"],
                         default="verify",
                         dest="db_operation", 
                         help="Database operation")
    
    # Quick Commands
    parser.add_argument("--daily", action="store_true",
                       help="Shortcut for daily pipeline")
    parser.add_argument("--weekly", action="store_true",
                       help="Shortcut for weekly pipeline")
    
    args = parser.parse_args()
    
    # Handle shortcuts
    if args.daily:
        args.command = "pipeline"
        args.pipeline_type = "daily"
    elif args.weekly:
        args.command = "pipeline"
        args.pipeline_type = "weekly"
    
    # If no command specified, show help
    if not args.command:
        parser.print_help()
        return 0
    
    logger.info(f"Starting ACIS Trading Platform - {datetime.now()}")
    logger.info(f"Command: {args.command}")
    
    try:
        # Route to appropriate handler
        if args.command == "setup":
            return run_first_time()
        elif args.command == "fetch":
            return run_data_fetch(args)
        elif args.command == "analysis":
            return run_analysis(args)
        elif args.command == "pipeline":
            return run_pipeline(args)
        elif args.command == "database":
            return run_database(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        logger.error("\nOperation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())