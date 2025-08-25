#!/usr/bin/env python3
"""
Enhanced run_eod_full_pipeline.py - OPTIMIZED VERSION
Purpose: Optimized pipeline with improved performance, reliability, and maintainability
"""

import os
import sys
import time
import argparse
import logging
import subprocess
import json
import asyncio
import concurrent.futures
from datetime import datetime, timezone
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any
from contextlib import asynccontextmanager
import psutil
import signal
from enum import Enum

# Load environment variables with better error handling
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Using system environment variables.")
except Exception as e:
    print(f"Warning: Could not load .env file: {e}")

# Enhanced logging setup with rotation
from logging.handlers import RotatingFileHandler


def setup_logging():
    """Setup logging with rotation and better formatting"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)8s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_dir / "eod_pipeline.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return logging.getLogger("eod_pipeline")


logger = setup_logging()


class ScriptStatus(Enum):
    """Script execution status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


@dataclass
class ScriptConfig:
    """Configuration for a single script"""
    name: str
    timeout: int
    is_critical: bool = True
    dependencies: List[str] = None
    max_retries: int = 3
    retry_delay: int = 30

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class ExecutionResult:
    """Result of script execution"""
    script: str
    status: ScriptStatus
    return_code: int
    duration: float
    attempt: int
    stdout: str = ""
    stderr: str = ""
    error_message: str = ""


class PipelineConfig:
    """Enhanced pipeline configuration"""

    # Resource limits
    MAX_CONCURRENT_SCRIPTS = min(4, os.cpu_count())
    MEMORY_THRESHOLD_GB = 8.0  # Stop if less than 8GB available

    # Script configurations with dependencies
    INGEST_SCRIPTS = [
        ScriptConfig("setup_schema.py", 60, True, []),
        ScriptConfig("fetch_symbol_metadata.py", 600, True, ["setup_schema.py"]),
        ScriptConfig("populate_stock_metadata.py", 120, False, ["fetch_symbol_metadata.py"]),
        ScriptConfig("fetch_sp500_history.py", 120, False, ["setup_schema.py"]),
        ScriptConfig("fetch_prices.py", 1800, True, ["fetch_symbol_metadata.py"]),
        ScriptConfig("fetch_dividend_history.py", 300, False, ["fetch_prices.py"]),
        ScriptConfig("fetch_fundamentals.py", 2400, True, ["fetch_prices.py"]),
    ]

    ANALYSIS_SCRIPTS = [
        ScriptConfig("compute_forward_returns.py", 300, True, ["fetch_prices.py"]),
        ScriptConfig("compute_dividend_growth_scores.py", 300, False, ["fetch_dividend_history.py"]),
        ScriptConfig("compute_value_momentum_and_growth_scores.py", 600, True, ["fetch_fundamentals.py"]),
        ScriptConfig("compute_ai_dividend_scores.py", 300, False, ["compute_dividend_growth_scores.py"]),
        ScriptConfig("compute_sp500_outperformance_scores.py", 300, True, ["compute_forward_returns.py"]),
        ScriptConfig("train_ai_value_model.py", 600, True, ["compute_value_momentum_and_growth_scores.py"]),
        ScriptConfig("score_ai_value_model.py", 300, True, ["train_ai_value_model.py"]),
        ScriptConfig("train_ai_growth_model.py", 600, True, ["compute_value_momentum_and_growth_scores.py"]),
        ScriptConfig("score_ai_growth_model.py", 300, True, ["train_ai_growth_model.py"]),
        ScriptConfig("run_rank_value_stocks.py", 120, True, ["score_ai_value_model.py"]),
        ScriptConfig("run_rank_growth_stocks.py", 120, True, ["score_ai_growth_model.py"]),
        ScriptConfig("run_rank_dividend_stocks.py", 120, False, ["compute_ai_dividend_scores.py"]),
        ScriptConfig("run_rank_momentum_stocks.py", 120, True, ["compute_sp500_outperformance_scores.py"]),
    ]

    MATERIALIZED_VIEWS = [
        "mv_latest_annual_fundamentals",
        "mv_symbol_with_metadata",
        "mv_latest_forward_returns",
        "mv_current_ai_portfolios",
    ]


class ResourceMonitor:
    """Monitor system resources during execution"""

    @staticmethod
    def check_memory() -> Tuple[bool, float]:
        """Check available memory"""
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024 ** 3)
        return available_gb > PipelineConfig.MEMORY_THRESHOLD_GB, available_gb

    @staticmethod
    def check_disk_space(path: str = ".") -> Tuple[bool, float]:
        """Check available disk space"""
        disk = psutil.disk_usage(path)
        free_gb = disk.free / (1024 ** 3)
        return free_gb > 2.0, free_gb  # At least 2GB free

    @staticmethod
    def get_system_stats() -> Dict[str, Any]:
        """Get current system statistics"""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage(".").percent,
            "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None,
        }


class DatabaseManager:
    """Enhanced database operations"""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self._engine = None

    @property
    def engine(self):
        """Lazy initialization of database engine"""
        if self._engine is None:
            try:
                from sqlalchemy import create_engine
                self._engine = create_engine(
                    self.connection_string,
                    pool_size=5,
                    max_overflow=10,
                    pool_pre_ping=True,
                    connect_args={
                        "connect_timeout": 10,
                        "application_name": "eod_pipeline"
                    }
                )
            except ImportError:
                logger.error("SQLAlchemy not installed. Database operations will be skipped.")
                return None
        return self._engine

    async def refresh_materialized_views(self) -> Tuple[List[str], List[Tuple[str, str]]]:
        """Refresh materialized views asynchronously"""
        if not self.engine:
            return [], [("Database", "SQLAlchemy not available")]

        refreshed = []
        failed = []

        try:
            from sqlalchemy import text

            with self.engine.begin() as conn:
                for mv in PipelineConfig.MATERIALIZED_VIEWS:
                    try:
                        logger.info(f"Refreshing materialized view: {mv}")
                        # Try concurrent refresh first
                        conn.execute(text(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {mv}"))
                        refreshed.append(mv)
                        logger.info(f"✅ Refreshed {mv}")
                    except Exception as e1:
                        logger.warning(f"Concurrent refresh failed for {mv}, trying non-concurrent: {e1}")
                        try:
                            # Fallback to non-concurrent
                            conn.execute(text(f"REFRESH MATERIALIZED VIEW {mv}"))
                            refreshed.append(mv)
                            logger.info(f"✅ Refreshed {mv} (non-concurrent)")
                        except Exception as e2:
                            failed.append((mv, str(e2)))
                            logger.error(f"❌ Failed to refresh {mv}: {e2}")

        except Exception as e:
            logger.exception(f"Database connection error: {e}")
            failed.append(("Database Connection", str(e)))

        return refreshed, failed


class NotificationManager:
    """Enhanced notification system"""

    @staticmethod
    def send_email(subject: str, body: str, is_html: bool = False) -> bool:
        """Send email notification"""
        smtp_config = {
            'host': os.getenv("SMTP_HOST"),
            'port': int(os.getenv("SMTP_PORT", 587)),
            'user': os.getenv("SMTP_USER"),
            'password': os.getenv("SMTP_PASS"),
            'from_email': os.getenv("SMTP_FROM", "pipeline@localhost"),
            'to_email': os.getenv("SMTP_TO", "admin@localhost"),
        }

        if not smtp_config['host']:
            logger.warning("SMTP not configured, skipping email notification")
            return False

        try:
            msg = MIMEMultipart('alternative')
            msg["From"] = smtp_config['from_email']
            msg["To"] = smtp_config['to_email']
            msg["Subject"] = subject

            content_type = 'html' if is_html else 'plain'
            msg.attach(MIMEText(body, content_type))

            with smtplib.SMTP(smtp_config['host'], smtp_config['port']) as server:
                if smtp_config['user']:
                    server.starttls()
                    server.login(smtp_config['user'], smtp_config['password'])
                server.send_message(msg)

            logger.info("📧 Email notification sent successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False


class EnhancedPipelineRunner:
    """Enhanced pipeline runner with async execution and better resource management"""

    def __init__(self, args):
        self.args = args
        self.start_time = time.time()
        self.results: Dict[str, ExecutionResult] = {}
        self.resource_monitor = ResourceMonitor()
        self.db_manager = DatabaseManager(os.getenv("POSTGRES_URL", ""))
        self.notification_manager = NotificationManager()
        self._shutdown_requested = False

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self._shutdown_requested = True

    async def check_prerequisites(self) -> bool:
        """Check system prerequisites before execution"""
        logger.info("🔍 Checking system prerequisites...")

        # Check memory
        memory_ok, memory_gb = self.resource_monitor.check_memory()
        if not memory_ok:
            logger.error(
                f"Insufficient memory: {memory_gb:.1f}GB available, need {PipelineConfig.MEMORY_THRESHOLD_GB}GB")
            return False

        # Check disk space
        disk_ok, disk_gb = self.resource_monitor.check_disk_space()
        if not disk_ok:
            logger.error(f"Insufficient disk space: {disk_gb:.1f}GB available")
            return False

        # Check required environment variables
        required_vars = ["POSTGRES_URL"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]

        if missing_vars:
            logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
            return False

        # Check database connectivity
        if self.db_manager.engine:
            try:
                with self.db_manager.engine.connect() as conn:
                    conn.execute("SELECT 1")
                logger.info("✅ Database connection verified")
            except Exception as e:
                logger.error(f"Database connection failed: {e}")
                return False

        logger.info("✅ All prerequisites met")
        return True

    def build_execution_plan(self) -> List[List[ScriptConfig]]:
        """Build execution plan with dependency resolution"""
        all_scripts = []

        if self.args.only:
            # Run only specified scripts
            script_map = {s.name: s for s in PipelineConfig.INGEST_SCRIPTS + PipelineConfig.ANALYSIS_SCRIPTS}
            all_scripts = [script_map[name] for name in self.args.only if name in script_map]
        else:
            if self.args.only_phase == "ingest":
                all_scripts = PipelineConfig.INGEST_SCRIPTS
            elif self.args.only_phase == "analysis":
                all_scripts = PipelineConfig.ANALYSIS_SCRIPTS
            else:
                if not self.args.skip_ingest:
                    all_scripts.extend(PipelineConfig.INGEST_SCRIPTS)
                if not self.args.skip_analysis:
                    all_scripts.extend(PipelineConfig.ANALYSIS_SCRIPTS)

        # Resolve dependencies and create execution waves
        execution_plan = []
        remaining = set(all_scripts)
        completed = set()

        while remaining:
            current_wave = []
            for script in list(remaining):
                # Check if all dependencies are completed
                deps_completed = all(
                    any(comp.name == dep for comp in completed)
                    for dep in script.dependencies
                )

                if deps_completed:
                    current_wave.append(script)
                    remaining.remove(script)

            if not current_wave:
                # Circular dependency or missing dependency
                logger.error("Circular dependency detected or missing dependencies")
                break

            execution_plan.append(current_wave)
            completed.update(current_wave)

        return execution_plan

    async def execute_script(self, script: ScriptConfig) -> ExecutionResult:
        """Execute a single script with enhanced error handling"""
        script_path = Path(script.name)

        if not script_path.exists():
            status = ScriptStatus.SKIPPED if not script.is_critical else ScriptStatus.FAILED
            return ExecutionResult(
                script=script.name,
                status=status,
                return_code=127,
                duration=0.0,
                attempt=0,
                error_message="Script not found"
            )

        for attempt in range(1, script.max_retries + 1):
            if self._shutdown_requested:
                return ExecutionResult(
                    script=script.name,
                    status=ScriptStatus.FAILED,
                    return_code=130,  # SIGINT
                    duration=0.0,
                    attempt=attempt,
                    error_message="Shutdown requested"
                )

            logger.info(f"🚀 Executing {script.name} (attempt {attempt}/{script.max_retries})")

            start_time = time.time()

            try:
                process = await asyncio.create_subprocess_exec(
                    sys.executable, str(script_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=script.timeout
                    )

                    duration = time.time() - start_time
                    stdout_str = stdout.decode('utf-8', errors='replace')
                    stderr_str = stderr.decode('utf-8', errors='replace')

                    if process.returncode == 0:
                        logger.info(f"✅ {script.name} completed successfully ({duration:.2f}s)")
                        return ExecutionResult(
                            script=script.name,
                            status=ScriptStatus.SUCCESS,
                            return_code=0,
                            duration=duration,
                            attempt=attempt,
                            stdout=stdout_str[-1000:],  # Last 1000 chars
                            stderr=stderr_str[-1000:]
                        )
                    else:
                        logger.warning(f"⚠️ {script.name} failed with return code {process.returncode}")
                        if attempt < script.max_retries:
                            await asyncio.sleep(script.retry_delay)
                            continue

                        return ExecutionResult(
                            script=script.name,
                            status=ScriptStatus.FAILED,
                            return_code=process.returncode,
                            duration=duration,
                            attempt=attempt,
                            stdout=stdout_str[-1000:],
                            stderr=stderr_str[-1000:],
                            error_message=f"Process exited with code {process.returncode}"
                        )

                except asyncio.TimeoutError:
                    logger.error(f"⏱️ {script.name} timed out after {script.timeout}s")
                    process.kill()
                    await process.wait()

                    if attempt < script.max_retries:
                        await asyncio.sleep(script.retry_delay)
                        continue

                    return ExecutionResult(
                        script=script.name,
                        status=ScriptStatus.TIMEOUT,
                        return_code=124,  # timeout exit code
                        duration=script.timeout,
                        attempt=attempt,
                        error_message=f"Timeout after {script.timeout}s"
                    )

            except Exception as e:
                logger.exception(f"💥 Unexpected error executing {script.name}: {e}")
                if attempt < script.max_retries:
                    await asyncio.sleep(script.retry_delay)
                    continue

                return ExecutionResult(
                    script=script.name,
                    status=ScriptStatus.FAILED,
                    return_code=1,
                    duration=time.time() - start_time,
                    attempt=attempt,
                    error_message=str(e)
                )

        # Should never reach here
        return ExecutionResult(
            script=script.name,
            status=ScriptStatus.FAILED,
            return_code=1,
            duration=0.0,
            attempt=script.max_retries,
            error_message="Max retries exceeded"
        )

    async def execute_wave(self, wave: List[ScriptConfig]) -> List[ExecutionResult]:
        """Execute a wave of scripts concurrently"""
        if not wave:
            return []

        logger.info(f"📦 Executing wave with {len(wave)} scripts: {[s.name for s in wave]}")

        # Limit concurrent execution
        semaphore = asyncio.Semaphore(min(len(wave), PipelineConfig.MAX_CONCURRENT_SCRIPTS))

        async def execute_with_semaphore(script):
            async with semaphore:
                return await self.execute_script(script)

        if self.args.dry_run:
            return [
                ExecutionResult(
                    script=script.name,
                    status=ScriptStatus.SKIPPED,
                    return_code=0,
                    duration=0.0,
                    attempt=0,
                    error_message="Dry run mode"
                )
                for script in wave
            ]

        tasks = [execute_with_semaphore(script) for script in wave]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task exception for {wave[i].name}: {result}")
                processed_results.append(ExecutionResult(
                    script=wave[i].name,
                    status=ScriptStatus.FAILED,
                    return_code=1,
                    duration=0.0,
                    attempt=0,
                    error_message=str(result)
                ))
            else:
                processed_results.append(result)

        return processed_results

    def should_continue_execution(self, wave_results: List[ExecutionResult]) -> bool:
        """Determine if execution should continue after a wave"""
        if self.args.continue_on_error:
            return True

        # Check for critical script failures
        for result in wave_results:
            if result.status in [ScriptStatus.FAILED, ScriptStatus.TIMEOUT]:
                script_config = next(
                    (s for s in PipelineConfig.INGEST_SCRIPTS + PipelineConfig.ANALYSIS_SCRIPTS
                     if s.name == result.script),
                    None
                )
                if script_config and script_config.is_critical:
                    logger.error(f"Critical script {result.script} failed, stopping execution")
                    return False

        return True

    def generate_execution_report(self) -> Dict[str, Any]:
        """Generate comprehensive execution report"""
        total_duration = time.time() - self.start_time

        # Categorize results
        successful = [r for r in self.results.values() if r.status == ScriptStatus.SUCCESS]
        failed = [r for r in self.results.values() if r.status in [ScriptStatus.FAILED, ScriptStatus.TIMEOUT]]
        skipped = [r for r in self.results.values() if r.status == ScriptStatus.SKIPPED]

        # Calculate statistics
        avg_duration = sum(r.duration for r in successful) / len(successful) if successful else 0

        report = {
            "pipeline": {
                "start_time": datetime.fromtimestamp(self.start_time, timezone.utc).isoformat(),
                "end_time": datetime.now(timezone.utc).isoformat(),
                "total_duration": total_duration,
                "status": "SUCCESS" if not failed else "FAILED"
            },
            "execution": {
                "total_scripts": len(self.results),
                "successful": len(successful),
                "failed": len(failed),
                "skipped": len(skipped),
                "average_duration": avg_duration,
                "total_attempts": sum(r.attempt for r in self.results.values())
            },
            "system": self.resource_monitor.get_system_stats(),
            "results": {name: asdict(result) for name, result in self.results.items()},
            "failures": [
                {
                    "script": r.script,
                    "return_code": r.return_code,
                    "duration": r.duration,
                    "attempts": r.attempt,
                    "error": r.error_message,
                    "stderr": r.stderr[-500:] if r.stderr else ""  # Last 500 chars
                }
                for r in failed
            ]
        }

        return report

    def save_report(self, report: Dict[str, Any]) -> Path:
        """Save execution report to file"""
        report_dir = Path("logs/reports")
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"pipeline_report_{timestamp}.json"

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"📊 Report saved to {report_path}")
        return report_path

    def print_summary(self, report: Dict[str, Any]):
        """Print execution summary"""
        pipeline = report["pipeline"]
        execution = report["execution"]

        print("\n" + "=" * 80)
        print("🏁 ENHANCED PIPELINE EXECUTION SUMMARY")
        print("=" * 80)
        print(f"Status: {'✅ SUCCESS' if pipeline['status'] == 'SUCCESS' else '❌ FAILED'}")
        print(f"Duration: {pipeline['total_duration']:.2f} seconds")
        print(f"Scripts executed: {execution['total_scripts']}")
        print(f"  ✅ Successful: {execution['successful']}")
        print(f"  ❌ Failed: {execution['failed']}")
        print(f"  ⏭️  Skipped: {execution['skipped']}")
        print(f"Average script duration: {execution['average_duration']:.2f}s")

        if execution['failed'] > 0:
            print(f"\n❌ Failed Scripts:")
            for failure in report['failures']:
                print(f"  • {failure['script']}: {failure['error']}")

        # System stats
        system = report['system']
        print(f"\n📊 System Resources:")
        print(f"  CPU: {system.get('cpu_percent', 'N/A')}%")
        print(f"  Memory: {system.get('memory_percent', 'N/A')}%")
        print(f"  Disk: {system.get('disk_percent', 'N/A')}%")

        print("=" * 80)

    async def run(self):
        """Main execution method"""
        logger.info("🚀 Starting Enhanced EOD Pipeline...")
        print("🚀 Enhanced ACIS EOD Trading Pipeline")
        print("=" * 60)

        # Check prerequisites
        if not await self.check_prerequisites():
            logger.error("Prerequisites not met, aborting")
            sys.exit(1)

        # Build execution plan
        execution_plan = self.build_execution_plan()
        total_scripts = sum(len(wave) for wave in execution_plan)

        logger.info(f"📋 Execution plan: {len(execution_plan)} waves, {total_scripts} scripts total")
        print(f"📋 Execution plan: {len(execution_plan)} waves, {total_scripts} scripts")

        # Execute waves
        for wave_num, wave in enumerate(execution_plan, 1):
            if self._shutdown_requested:
                logger.info("Shutdown requested, stopping execution")
                break

            print(f"\n🌊 Wave {wave_num}/{len(execution_plan)}: {[s.name for s in wave]}")

            wave_results = await self.execute_wave(wave)

            # Store results
            for result in wave_results:
                self.results[result.script] = result

            # Check if we should continue
            if not self.should_continue_execution(wave_results):
                break

        # Refresh materialized views
        if not self.args.skip_mv_refresh and not any(
                r.status in [ScriptStatus.FAILED, ScriptStatus.TIMEOUT]
                for r in self.results.values()
        ):
            print("\n🔄 Refreshing materialized views...")
            refreshed, failed_mvs = await self.db_manager.refresh_materialized_views()
            if refreshed:
                print(f"✅ Refreshed {len(refreshed)} materialized views")
            if failed_mvs:
                print(f"⚠️ Failed to refresh {len(failed_mvs)} views")

        # Generate and save report
        report = self.generate_execution_report()
        report_path = self.save_report(report)

        # Send notification
        if self.args.notify:
            subject = f"Pipeline {report['pipeline']['status']}: {datetime.now().strftime('%Y-%m-%d')}"
            body = f"""
            Pipeline execution completed.

            Status: {report['pipeline']['status']}
            Duration: {report['pipeline']['total_duration']:.2f}s
            Success: {report['execution']['successful']}/{report['execution']['total_scripts']} scripts

            Full report: {report_path}
            """
            self.notification_manager.send_email(subject, body)

        # Print summary
        self.print_summary(report)

        # Exit with appropriate code
        failed_count = report['execution']['failed']
        sys.exit(0 if failed_count == 0 else 1)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Enhanced ACIS EOD Trading Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Run full pipeline
  %(prog)s --only-phase ingest       # Run only ingestion phase
  %(prog)s --only fetch_prices.py    # Run only specific script
  %(prog)s --dry-run                 # Show what would be executed
  %(prog)s --continue-on-error       # Continue even if scripts fail
        """
    )

    parser.add_argument("--continue-on-error", action="store_true",
                        help="Continue execution even if critical scripts fail")
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
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        runner = EnhancedPipelineRunner(args)
        asyncio.run(runner.run())
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Pipeline failed with unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()