
# =====================================
# Optimized Database Utilities
# =====================================
"""
Common database optimization utilities for ACIS trading platform
"""

import os
import logging
from contextlib import contextmanager
from sqlalchemy import create_engine, text
import time
import psutil
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class DatabaseOptimizer:
    """Database optimization utilities for trading platform scripts"""

    def __init__(self, postgres_url: str):
        self.postgres_url = postgres_url
        self._engine = None

    @property
    def engine(self):
        """Lazy initialization of optimized database engine"""
        if self._engine is None:
            self._engine = create_engine(
                self.postgres_url,
                # Connection pool optimizations
                pool_size=20,
                max_overflow=40,
                pool_pre_ping=True,
                pool_recycle=3600,  # 1 hour

                # Connection-level optimizations
                connect_args={
                    "options": "-c work_mem=512MB -c maintenance_work_mem=2GB -c effective_cache_size=8GB -c random_page_cost=1.1",
                    "connect_timeout": 10,
                    "application_name": "acis_trading_platform"
                }
            )
        return self._engine

    def check_system_resources(self) -> Dict[str, Any]:
        """Check system resources before heavy operations"""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage(".")

        return {
            "memory_available_gb": memory.available / (1024 ** 3),
            "memory_percent": memory.percent,
            "disk_free_gb": disk.free / (1024 ** 3),
            "cpu_percent": psutil.cpu_percent(interval=1)
        }

    @contextmanager
    def bulk_operation_context(self, table_name: Optional[str] = None):
        """Context manager for bulk database operations with optimizations"""

        resources = self.check_system_resources()
        if resources["memory_available_gb"] < 2.0:
            logger.warning(f"Low memory: {resources['memory_available_gb']:.1f}GB available")

        with self.engine.begin() as conn:
            # Store original settings
            original_settings = {}

            # Optimizations for bulk operations
            bulk_optimizations = {
                "synchronous_commit": "OFF",
                "work_mem": "1GB",
                "maintenance_work_mem": "4GB",
                "checkpoint_completion_target": "0.9",
                "wal_buffers": "64MB",
                "effective_io_concurrency": "200",
                "max_parallel_workers_per_gather": "4",
            }

            try:
                # Apply optimizations
                for setting, value in bulk_optimizations.items():
                    try:
                        # Get current value
                        result = conn.execute(text(f"SHOW {setting}")).fetchone()
                        if result:
                            original_settings[setting] = result[0]

                        # Set new value
                        conn.execute(text(f"SET {setting} = '{value}'"))
                        logger.debug(f"Set {setting} = {value}")

                    except Exception as e:
                        logger.warning(f"Could not optimize {setting}: {e}")

                # Table-specific optimizations
                if table_name:
                    try:
                        # Temporarily disable autovacuum for bulk operations
                        conn.execute(text(f"ALTER TABLE {table_name} SET (autovacuum_enabled = false)"))
                        logger.debug(f"Disabled autovacuum for {table_name}")
                    except Exception as e:
                        logger.warning(f"Could not disable autovacuum for {table_name}: {e}")

                yield conn

            finally:
                # Restore original settings
                for setting, original_value in original_settings.items():
                    try:
                        conn.execute(text(f"SET {setting} = '{original_value}'"))
                        logger.debug(f"Restored {setting} = {original_value}")
                    except Exception as e:
                        logger.warning(f"Could not restore {setting}: {e}")

                # Re-enable autovacuum and analyze
                if table_name:
                    try:
                        conn.execute(text(f"ALTER TABLE {table_name} SET (autovacuum_enabled = true)"))
                        conn.execute(text(f"ANALYZE {table_name}"))
                        logger.debug(f"Re-enabled autovacuum and analyzed {table_name}")
                    except Exception as e:
                        logger.warning(f"Could not cleanup {table_name}: {e}")