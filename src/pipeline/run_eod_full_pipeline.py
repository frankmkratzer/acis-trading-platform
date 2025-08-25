# =====================================
# Enhanced run_eod_full_pipeline.py
# =====================================
"""
#!/usr/bin/env python3
# File: run_eod_full_pipeline.py (ENHANCED)
# Purpose: Enhanced pipeline with better error handling and monitoring
"""

import os
import sys
import time
import argparse
import logging
import subprocess
import json
from datetime import datetime
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
except:
    pass

# Logging setup
LOG_PATH = "logs/eod_full_pipeline.log"
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("eod_pipeline")

# Pipeline configuration
PIPELINE_CONFIG = {
    "max_retries": 3,
    "retry_delay": 30,
    "timeout_default": 600,
    "critical_scripts": [
        "setup_schema.py",
        "fetch_symbol_metadata.py",
        "fetch_prices.py"
    ],
    "optional_scripts": [
        "populate_stock_metadata.py",
        "fetch_dividend_history.py"
    ]
}

# Script execution order
INGEST_SCRIPTS = [
    ("setup_schema.py", 60),
    ("fetch_symbol_metadata.py", 600),
    ("populate_stock_metadata.py", 120),
    ("fetch_sp500_history.py", 120),
    ("fetch_prices.py", 1800),
    ("fetch_dividend_history.py", 300),
    ("fetch_fundamentals.py", 2400),
]

ANALYSIS_SCRIPTS = [
    ("compute_forward_returns.py", 300),
    ("compute_dividend_growth_scores.py", 300),
    ("compute_value_momentum_and_growth_scores.py", 600),
    ("compute_ai_dividend_scores.py", 300),
    ("compute_sp500_outperformance_scores.py", 300),
    ("train_ai_value_model.py", 600),
    ("score_ai_value_model.py", 300),
    ("train_ai_growth_model.py", 600),
    ("score_ai_growth_model.py", 300),
    ("run_rank_value_stocks.py", 120),
    ("run_rank_growth_stocks.py", 120),
    ("run_rank_dividend_stocks.py", 120),
    ("run_rank_momentum_stocks.py", 120),
]

MATERIALIZED_VIEWS = [
    "mv_latest_annual_fundamentals",
    "mv_symbol_with_metadata",
    "mv_latest_forward_returns",
    "mv_current_ai_portfolios",
]


class PipelineRunner:
    def __init__(self, args):
        self.args = args
        self.start_time = time.time()
        self.failures = []
        self.warnings = []
        self.success_count = 0

    def run_script_with_retry(self, script: str, timeout: int) -> int:
        """Run a script with retry logic"""
        script_path = Path(script)

        if not script_path.exists():
            if script in PIPELINE_CONFIG["optional_scripts"]:
                logger.warning(f"Optional script {script} not found, skipping")
                self.warnings.append((script, "Not found"))
                return 0
            else:
                logger.error(f"Critical script {script} not found")
                return 127

        for attempt in range(1, PIPELINE_CONFIG["max_retries"] + 1):
            logger.info(f"Running {script} (attempt {attempt})")
            print(f"\n▶️ Running {script} (attempt {attempt}/{PIPELINE_CONFIG['max_retries']})...")

            try:
                started = time.time()
                proc = subprocess.run(
                    [sys.executable, script_path],
                    timeout=timeout,
                    capture_output=True,
                    text=True
                )
                elapsed = time.time() - started

                if proc.returncode == 0:
                    logger.info(f"✅ {script} succeeded in {elapsed:.2f}s")
                    print(f"✅ {script} completed ({elapsed:.2f}s)")
                    self.success_count += 1
                    return 0
                else:
                    logger.warning(f"Script {script} failed with code {proc.returncode}")
                    if proc.stderr:
                        logger.error(f"Error output: {proc.stderr[-1000:]}")  # Last 1000 chars

                    if attempt < PIPELINE_CONFIG["max_retries"]:
                        time.sleep(PIPELINE_CONFIG["retry_delay"])

            except subprocess.TimeoutExpired:
                logger.error(f"Script {script} timed out after {timeout}s")
                if attempt < PIPELINE_CONFIG["max_retries"]:
                    time.sleep(PIPELINE_CONFIG["retry_delay"])
            except Exception as e:
                logger.exception(f"Unexpected error running {script}: {e}")
                if attempt < PIPELINE_CONFIG["max_retries"]:
                    time.sleep(PIPELINE_CONFIG["retry_delay"])

        return 1  # Failed after all retries

    def refresh_materialized_views(self):
        """Refresh materialized views with better error handling"""
        try:
            from sqlalchemy import create_engine, text

            pg_url = os.getenv("POSTGRES_URL")
            if not pg_url:
                logger.warning("POSTGRES_URL not set, skipping MV refresh")
                return

            engine = create_engine(pg_url)
            refreshed = []
            failed = []

            with engine.begin() as conn:
                for mv in MATERIALIZED_VIEWS:
                    try:
                        # Try concurrent refresh first
                        conn.execute(text(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {mv};"))
                        refreshed.append(mv)
                        logger.info(f"Refreshed {mv}")
                        print(f"🔄 Refreshed {mv}")
                    except Exception as e1:
                        try:
                            # Fallback to non-concurrent
                            conn.execute(text(f"REFRESH MATERIALIZED VIEW {mv};"))
                            refreshed.append(mv)
                            logger.info(f"Refreshed {mv} (non-concurrent)")
                            print(f"🔄 Refreshed {mv} (non-concurrent)")
                        except Exception as e2:
                            failed.append(mv)
                            logger.error(f"Failed to refresh {mv}: {e2}")
                            self.warnings.append((mv, f"Refresh failed: {e2}"))

            if refreshed:
                print(f"✅ Refreshed {len(refreshed)} materialized views")
            if failed:
                print(f"⚠️ Failed to refresh {len(failed)} views: {', '.join(failed)}")

        except Exception as e:
            logger.exception(f"MV refresh error: {e}")
            self.warnings.append(("MV Refresh", str(e)))

    def send_notification(self, status: str, details: dict):
        """Send email notification about pipeline status"""
        if not os.getenv("SMTP_HOST"):
            return

        try:
            msg = MIMEMultipart()
            msg["From"] = os.getenv("SMTP_FROM", "pipeline@localhost")
            msg["To"] = os.getenv("SMTP_TO", "admin@localhost")
            msg["Subject"] = f"Pipeline {status}: {datetime.now().strftime('%Y-%m-%d')}"

            body = f"""
            Pipeline Status: {status}
            Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            Duration: {details.get('duration', 'N/A')}

            Success: {details.get('success_count', 0)} scripts
            Failures: {len(details.get('failures', []))}
            Warnings: {len(details.get('warnings', []))}
            """

            if details.get('failures'):
                body += "\n\nFailures:\n"
                for script, code in details['failures']:
                    body += f"  - {script}: exit code {code}\n"

            if details.get('warnings'):
                body += "\n\nWarnings:\n"
                for item, msg in details['warnings']:
                    body += f"  - {item}: {msg}\n"

            msg.attach(MIMEText(body, 'plain'))

            with smtplib.SMTP(os.getenv("SMTP_HOST"), int(os.getenv("SMTP_PORT", 587))) as server:
                if os.getenv("SMTP_USER"):
                    server.starttls()
                    server.login(os.getenv("SMTP_USER"), os.getenv("SMTP_PASS"))
                server.send_message(msg)

            logger.info(f"Notification sent: {status}")

        except Exception as e:
            logger.error(f"Failed to send notification: {e}")

    def generate_report(self):
        """Generate execution report"""
        duration = time.time() - self.start_time

        report = {
            "date": datetime.now().isoformat(),
            "duration_seconds": duration,
            "success_count": self.success_count,
            "failures": self.failures,
            "warnings": self.warnings,
            "status": "SUCCESS" if not self.failures else "FAILED"
        }

        # Save report
        report_path = f"logs/pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n📊 Report saved to {report_path}")

        return report

    def run(self):
        """Execute the pipeline"""
        print("🚀 Starting enhanced EOD pipeline...")
        logger.info("Pipeline started")

        # Check environment
        self.check_environment()

        # Determine which scripts to run
        scripts_to_run = self.get_scripts_to_run()

        # Execute scripts
        for script, timeout in scripts_to_run:
            if self.args.dry_run:
                print(f"[DRY RUN] Would execute: {script}")
                continue

            rc = self.run_script_with_retry(script, timeout or PIPELINE_CONFIG["timeout_default"])

            if rc != 0:
                self.failures.append((script, rc))
                if not self.args.continue_on_error and script in PIPELINE_CONFIG["critical_scripts"]:
                    logger.error(f"Critical script {script} failed, aborting")
                    break

        # Refresh materialized views
        if not self.args.skip_mv_refresh and not self.failures:
            self.refresh_materialized_views()

        # Generate report
        report = self.generate_report()

        # Send notification
        if self.args.notify:
            self.send_notification(report["status"], report)

        # Print summary
        self.print_summary(report)

        # Exit with appropriate code
        sys.exit(0 if not self.failures else 1)

    def check_environment(self):
        """Check required environment variables"""
        required = ["POSTGRES_URL"]
        optional = ["ALPHA_VANTAGE_API_KEY", "YAHOO_API_KEY"]

        missing_required = []
        missing_optional = []

        for var in required:
            if not os.getenv(var):
                missing_required.append(var)

        for var in optional:
            if not os.getenv(var):
                missing_optional.append(var)

        if missing_required:
            print(f"❌ Missing required environment variables: {', '.join(missing_required)}")
            logger.error(f"Missing required vars: {missing_required}")
            sys.exit(1)

        if missing_optional:
            print(f"⚠️ Missing optional environment variables: {', '.join(missing_optional)}")
            self.warnings.extend([(var, "Not set") for var in missing_optional])

    def get_scripts_to_run(self):
        """Determine which scripts to run based on arguments"""
        scripts = []

        if self.args.only:
            # Run only specified scripts
            for script_name in self.args.only:
                # Find timeout for script
                timeout = PIPELINE_CONFIG["timeout_default"]
                for s, t in INGEST_SCRIPTS + ANALYSIS_SCRIPTS:
                    if s == script_name:
                        timeout = t
                        break
                scripts.append((script_name, timeout))
        else:
            if self.args.only_phase == "ingest":
                scripts = INGEST_SCRIPTS
            elif self.args.only_phase == "analysis":
                scripts = ANALYSIS_SCRIPTS
            else:
                if not self.args.skip_ingest:
                    scripts.extend(INGEST_SCRIPTS)
                if not self.args.skip_analysis:
                    scripts.extend(ANALYSIS_SCRIPTS)

        return scripts

    def print_summary(self, report):
        """Print execution summary"""
        print("\n" + "=" * 70)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 70)
        print(f"Status: {report['status']}")
        print(f"Duration: {report['duration_seconds']:.2f} seconds")
        print(f"Successful: {report['success_count']} scripts")
        print(f"Failed: {len(report['failures'])}")
        print(f"Warnings: {len(report['warnings'])}")

        if report['failures']:
            print("\nFailed Scripts:")
            for script, code in report['failures']:
                print(f"  ❌ {script} (exit code: {code})")

        if report['warnings']:
            print("\nWarnings:")
            for item, msg in report['warnings'][:10]:  # Show first 10
                print(f"  ⚠️ {item}: {msg}")
            if len(report['warnings']) > 10:
                print(f"  ... and {len(report['warnings']) - 10} more")

        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Enhanced EOD Pipeline Runner")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Continue execution even if scripts fail")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Skip data ingestion phase")
    parser.add_argument("--skip-analysis", action="store_true",
                        help="Skip analysis phase")
    parser.add_argument("--skip-mv-refresh", action="store_true",
                        help="Skip materialized view refresh")
    parser.add_argument("--only", nargs="*",
                        help="Run only specified scripts")
    parser.add_argument("--only-phase", choices=["ingest", "analysis"],
                        help="Run only specified phase")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be executed without running")
    parser.add_argument("--notify", action="store_true",
                        help="Send email notification on completion")

    args = parser.parse_args()

    runner = PipelineRunner(args)
    runner.run()


if __name__ == "__main__":
    main()