#!/usr/bin/env python3
"""
ACIS Master Control Center
Central management for the TOP 1% Trading Platform
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add current directory to path
sys.path.append(str(Path(__file__).parent))
from utils.logging_config import setup_logger

logger = setup_logger("master_control")
load_dotenv()


class ACISMasterControl:
    """Master control for ACIS platform operations"""
    
    def __init__(self):
        self.base_path = Path(__file__).parent
        
    def run_command(self, command: str, description: str) -> bool:
        """Execute a command and return success status"""
        print(f"\n[INFO] {description}")
        logger.info(f"Executing: {description}")
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=False,
                text=True,
                cwd=self.base_path
            )
            
            if result.returncode == 0:
                print(f"[SUCCESS] {description} completed")
                return True
            else:
                print(f"[FAILED] {description} failed")
                return False
                
        except Exception as e:
            print(f"[ERROR] {description}: {e}")
            logger.error(f"Error in {description}: {e}")
            return False
    
    def verify_database(self):
        """Verify all database tables are present"""
        print("\n" + "="*60)
        print("DATABASE VERIFICATION")
        print("="*60)
        
        return self.run_command(
            f"{sys.executable} database/verify_tables_enhanced.py",
            "Verifying database tables"
        )
    
    def run_daily_pipeline(self, enhanced=True):
        """Run daily pipeline"""
        print("\n" + "="*60)
        print("DAILY PIPELINE")
        print("="*60)
        
        if enhanced:
            script = "pipelines/run_daily_pipeline_top1pct.py"
            desc = "Running TOP 1% daily pipeline"
        else:
            script = "pipelines/run_daily_pipeline_clean.py"
            desc = "Running basic daily pipeline"
        
        return self.run_command(f"{sys.executable} {script}", desc)
    
    def run_weekly_pipeline(self, enhanced=True):
        """Run weekly pipeline"""
        print("\n" + "="*60)
        print("WEEKLY PIPELINE")
        print("="*60)
        
        if enhanced:
            script = "pipelines/run_weekly_pipeline_top1pct.py"
            desc = "Running TOP 1% weekly pipeline"
        else:
            script = "pipelines/run_weekly_pipeline_clean.py"
            desc = "Running basic weekly pipeline"
        
        return self.run_command(f"{sys.executable} {script}", desc)
    
    def run_specific_analysis(self, analysis_type):
        """Run specific analysis module"""
        print("\n" + "="*60)
        print(f"RUNNING: {analysis_type.upper()}")
        print("="*60)
        
        analysis_scripts = {
            'piotroski': 'analysis/calculate_piotroski_fscore.py',
            'altman': 'analysis/calculate_altman_zscore.py',
            'beneish': 'analysis/calculate_beneish_mscore.py',
            'insider': 'data_fetch/fundamentals/fetch_insider_transactions.py',
            'institutional': 'data_fetch/fundamentals/fetch_institutional_holdings.py',
            'risk': 'analysis/calculate_risk_metrics.py',
            'breakout': 'analysis/calculate_technical_breakouts.py',
            'kelly': 'analysis/calculate_kelly_criterion.py',
            'sector': 'analysis/sector_rotation_matrix.py',
            'backtest': 'analysis/backtesting_framework.py',
            'optimize': 'analysis/walk_forward_optimization.py',
            'attribution': 'analysis/performance_attribution.py',
            'ml': 'ml_analysis/strategies/xgboost_return_predictor.py',
            'master': 'analysis/calculate_master_scores.py'
        }
        
        if analysis_type in analysis_scripts:
            script = analysis_scripts[analysis_type]
            return self.run_command(
                f"{sys.executable} {script}",
                f"Running {analysis_type} analysis"
            )
        else:
            print(f"[ERROR] Unknown analysis type: {analysis_type}")
            print(f"Available: {', '.join(analysis_scripts.keys())}")
            return False
    
    def show_portfolio_status(self):
        """Display current portfolio status"""
        print("\n" + "="*60)
        print("PORTFOLIO STATUS")
        print("="*60)
        
        try:
            from sqlalchemy import create_engine, text
            
            postgres_url = os.getenv("POSTGRES_URL")
            if not postgres_url:
                print("[ERROR] POSTGRES_URL not set")
                return False
            
            engine = create_engine(postgres_url)
            
            with engine.connect() as conn:
                # Get portfolio summary
                result = conn.execute(text("""
                    SELECT 
                        portfolio_type,
                        COUNT(*) as num_stocks,
                        MAX(last_updated) as last_update
                    FROM portfolio_holdings
                    WHERE is_current = true
                    GROUP BY portfolio_type
                    ORDER BY portfolio_type
                """))
                
                portfolios = result.fetchall()
                
                if portfolios:
                    print("\nCurrent Portfolios:")
                    for portfolio_type, num_stocks, last_update in portfolios:
                        print(f"  {portfolio_type:10s}: {num_stocks:2d} stocks (Updated: {last_update})")
                else:
                    print("\n[WARNING] No portfolio holdings found")
                
                # Get top scores
                result = conn.execute(text("""
                    SELECT 
                        symbol,
                        composite_score,
                        value_score,
                        growth_score,
                        dividend_score
                    FROM master_scores
                    ORDER BY composite_score DESC
                    LIMIT 5
                """))
                
                top_stocks = result.fetchall()
                
                if top_stocks:
                    print("\nTop 5 Stocks by Composite Score:")
                    for symbol, comp, val, growth, div in top_stocks:
                        print(f"  {symbol:6s} | Composite: {comp:.1f} | "
                              f"V: {val:.1f} | G: {growth:.1f} | D: {div:.1f}")
                
                return True
                
        except Exception as e:
            print(f"[ERROR] Could not get portfolio status: {e}")
            return False
    
    def setup_database(self):
        """Initialize or reset database schema"""
        print("\n" + "="*60)
        print("DATABASE SETUP")
        print("="*60)
        
        return self.run_command(
            f"{sys.executable} database/setup_schema_chunked.py",
            "Setting up database schema"
        )


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description='ACIS Master Control - TOP 1% Trading Platform',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python master_control.py --daily           # Run TOP 1% daily pipeline
  python master_control.py --weekly          # Run TOP 1% weekly pipeline
  python master_control.py --basic-daily     # Run basic daily pipeline
  python master_control.py --verify          # Verify database tables
  python master_control.py --status          # Show portfolio status
  python master_control.py --analyze kelly   # Run Kelly Criterion analysis
  python master_control.py --setup           # Setup database schema

Available analysis types:
  piotroski     - Piotroski F-Score
  altman        - Altman Z-Score
  beneish       - Beneish M-Score
  insider       - Insider transactions
  institutional - Institutional holdings
  risk          - Risk metrics (Sharpe, Sortino)
  breakout      - Technical breakouts
  kelly         - Kelly Criterion
  sector        - Sector rotation
  backtest      - Backtesting framework
  optimize      - Walk-forward optimization
  attribution   - Performance attribution
  ml            - Machine learning predictions
  master        - Master composite scores
        """
    )
    
    # Pipeline options
    parser.add_argument('--daily', action='store_true',
                       help='Run TOP 1% daily pipeline')
    parser.add_argument('--weekly', action='store_true',
                       help='Run TOP 1% weekly pipeline')
    parser.add_argument('--basic-daily', action='store_true',
                       help='Run basic daily pipeline')
    parser.add_argument('--basic-weekly', action='store_true',
                       help='Run basic weekly pipeline')
    
    # Analysis options
    parser.add_argument('--analyze', type=str, metavar='TYPE',
                       help='Run specific analysis')
    
    # Management options
    parser.add_argument('--verify', action='store_true',
                       help='Verify database tables')
    parser.add_argument('--status', action='store_true',
                       help='Show portfolio status')
    parser.add_argument('--setup', action='store_true',
                       help='Setup database schema')
    
    args = parser.parse_args()
    
    # Initialize control
    control = ACISMasterControl()
    
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║             ACIS MASTER CONTROL CENTER                     ║
    ║           TOP 1% Institutional-Grade Platform              ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # Execute requested operation
    if args.daily:
        return 0 if control.run_daily_pipeline(enhanced=True) else 1
    elif args.weekly:
        return 0 if control.run_weekly_pipeline(enhanced=True) else 1
    elif args.basic_daily:
        return 0 if control.run_daily_pipeline(enhanced=False) else 1
    elif args.basic_weekly:
        return 0 if control.run_weekly_pipeline(enhanced=False) else 1
    elif args.analyze:
        return 0 if control.run_specific_analysis(args.analyze) else 1
    elif args.verify:
        return 0 if control.verify_database() else 1
    elif args.status:
        return 0 if control.show_portfolio_status() else 1
    elif args.setup:
        return 0 if control.setup_database() else 1
    else:
        # Default: show status
        control.show_portfolio_status()
        print("\nUse --help for all options")
        return 0


if __name__ == "__main__":
    sys.exit(main())