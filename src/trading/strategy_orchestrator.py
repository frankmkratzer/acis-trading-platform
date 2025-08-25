# =====================================
# 7. Main Strategy Orchestrator
# =====================================
"""
#!/usr/bin/env python3
# File: strategy_orchestrator.py
# Purpose: Orchestrate all strategies and manage portfolio lifecycle
"""

import schedule
import time
from datetime import datetime
import logging


class StrategyOrchestrator:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.risk_manager = RiskManager()
        self.dashboard = PerformanceDashboard()

        logging.basicConfig(
            filename='orchestrator.log',
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def run_daily_pipeline(self):
        """Run complete daily pipeline"""
        self.logger.info("Starting daily pipeline")

        try:
            # 1. Update data
            self._update_data()

            # 2. Run each strategy
            for strategy_config in self.config_manager.get_enabled_strategies():
                self._run_strategy(strategy_config)

            # 3. Check risk limits
            self._check_risk_limits()

            # 4. Generate reports
            self.dashboard.generate_report()

            self.logger.info("Daily pipeline completed successfully")

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise

    def _update_data(self):
        """Update market data"""
        import subprocess

        scripts = [
            'fetch_prices.py',
            'fetch_fundamentals.py',
            'compute_forward_returns.py'
        ]

        for script in scripts:
            self.logger.info(f"Running {script}")
            result = subprocess.run(['python', script])
            if result.returncode != 0:
                raise RuntimeError(f"{script} failed")

    def _run_strategy(self, config: StrategyConfig):
        """Run a single strategy"""
        self.logger.info(f"Running {config.name} strategy")

        # Map strategy names to scripts
        script_map = {
            'value': ['train_ai_value_model.py', 'score_ai_value_model.py'],
            'growth': ['train_ai_growth_model.py', 'score_ai_growth_model.py'],
            'momentum': ['compute_value_momentum_and_growth_scores.py'],
            'dividend': ['compute_dividend_growth_scores.py']
        }

        scripts = script_map.get(config.name, [])

        for script in scripts:
            if os.path.exists(script):
                import subprocess
                subprocess.run(['python', script])

    def _check_risk_limits(self):
        """Check portfolio risk limits"""
        # Load current portfolios
        query = text("""
                     SELECT *
                     FROM mv_current_ai_portfolios
                     """)

        with engine.connect() as conn:
            portfolios = pd.read_sql(query, conn)

        for strategy in portfolios['strategy'].unique():
            strategy_holdings = portfolios[portfolios['strategy'] == strategy]

            # Check constraints
            checks = self.risk_manager.check_portfolio_constraints(strategy_holdings)

            if not all(checks.values()):
                self.logger.warning(f"Risk limits violated for {strategy}: {checks}")

    def schedule_jobs(self):
        """Schedule recurring jobs"""
        # Daily at market close
        schedule.every().day.at("16:30").do(self.run_daily_pipeline)

        # Weekly performance report
        schedule.every().friday.at("17:00").do(self.dashboard.generate_report)

        self.logger.info("Jobs scheduled")

    def run(self):
        """Run the orchestrator"""
        self.schedule_jobs()

        self.logger.info("Orchestrator started")

        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute


if __name__ == "__main__":
    orchestrator = StrategyOrchestrator()
    orchestrator.run()