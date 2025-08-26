#!/usr/bin/env python3
"""
Enhanced run_eod_full_pipeline.py - OPTIMIZED VERSION 2.0
Purpose: Production-ready pipeline with all suggested improvements
Single file implementation maintaining flat project structure
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
from datetime import datetime, timezone, timedelta
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional, Any, Callable, Set
from contextlib import asynccontextmanager
import psutil
import signal
from enum import Enum
import hashlib
import pickle
import threading
from collections import defaultdict
import heapq
import yaml
import re
from functools import lru_cache, wraps
from logging.handlers import RotatingFileHandler

# Optional imports with fallback
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Using system environment variables.")

try:
    import redis
    from redis import asyncio as aioredis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("Warning: Redis not installed. Caching disabled.")

try:
    from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("Warning: prometheus_client not installed. Metrics export disabled.")

# Initialize logger early
logger = logging.getLogger("eod_pipeline")


# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

class ConfigManager:
    """Centralized configuration management with YAML support"""

    DEFAULT_CONFIG = {
        'pipeline': {
            'name': 'ACIS EOD Trading Pipeline',
            'version': '2.0.0',
            'environment': 'development'
        },
        'execution': {
            'max_concurrent_scripts': 4,
            'memory_threshold_gb': 8.0,
            'disk_threshold_gb': 2.0,
            'default_timeout': 300,
            'retry_policy': {
                'max_attempts': 3,
                'delay_seconds': 30,
                'backoff_multiplier': 2.0
            }
        },
        'circuit_breaker': {
            'failure_threshold': 5,
            'recovery_timeout': 60,
            'half_open_requests': 2
        },
        'cache': {
            'redis_url': 'redis://localhost:6379',
            'default_ttl': 3600,
            'max_memory': '2gb'
        },
        'database': {
            'pool_size': 5,
            'max_overflow': 10,
            'pool_timeout': 30,
            'pool_recycle': 3600
        },
        'monitoring': {
            'prometheus_port': 9090,
            'health_check_interval': 30,
            'metrics_retention_days': 30
        },
        'market_hours': {
            'timezone': 'America/New_York',
            'pre_market_start': '04:00',
            'market_open': '09:30',
            'market_close': '16:00',
            'after_hours_end': '20:00',
            'trading_days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        }
    }

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else Path("pipeline_config.yaml")
        self._config = None
        self._env_pattern = re.compile(r'\${([^}:]+)(?::([^}]+))?}')

    @property
    def config(self) -> Dict[str, Any]:
        """Lazy load configuration"""
        if self._config is None:
            self._config = self.load_config()
        return self._config

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                # Merge with defaults
                config = self._deep_merge(self.DEFAULT_CONFIG.copy(), file_config)
            except Exception as e:
                print(f"Warning: Failed to load config file: {e}. Using defaults.")
                config = self.DEFAULT_CONFIG.copy()
        else:
            config = self.DEFAULT_CONFIG.copy()

        # Process environment variables
        return self._process_env_vars(config)

    def _deep_merge(self, base: dict, override: dict) -> dict:
        """Deep merge two dictionaries"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = self._deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    def _process_env_vars(self, obj: Any) -> Any:
        """Recursively process environment variables in config"""
        if isinstance(obj, str):
            return self._replace_env_vars(obj)
        elif isinstance(obj, dict):
            return {k: self._process_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._process_env_vars(item) for item in obj]
        return obj

    def _replace_env_vars(self, value: str) -> Any:
        """Replace ${VAR:default} with environment variable"""

        def replacer(match):
            var_name = match.group(1)
            default_value = match.group(2)
            env_value = os.getenv(var_name, default_value)

            if env_value and env_value.lower() in ('true', 'false'):
                return env_value.lower() == 'true'
            try:
                return int(env_value)
            except (ValueError, TypeError):
                try:
                    return float(env_value)
                except (ValueError, TypeError):
                    return env_value

        result = self._env_pattern.sub(replacer, value)
        return result


# ============================================================================
# ENHANCED LOGGING
# ============================================================================

def setup_logging(verbose: bool = False):
    """Setup logging with rotation and better formatting"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)8s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler = RotatingFileHandler(
        log_dir / "eod_pipeline.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'  # Add UTF-8 encoding for file handler
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Console handler with UTF-8 encoding for Windows
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Set encoding for Windows console
    if sys.platform == 'win32':
        import locale
        if sys.stdout.encoding != 'utf-8':
            try:
                sys.stdout.reconfigure(encoding='utf-8')
                sys.stderr.reconfigure(encoding='utf-8')
            except AttributeError:
                # Python < 3.7
                import codecs
                sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
                sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return logging.getLogger("eod_pipeline")


# Initialize global logger
logger = setup_logging()


# ============================================================================
# DATA CLASSES AND ENUMS
# ============================================================================

class ScriptStatus(Enum):
    """Script execution status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"
    CIRCUIT_OPEN = "circuit_open"


@dataclass(frozen=True)  # Make it hashable by freezing
class ScriptConfig:
    """Enhanced configuration for a single script"""
    name: str
    timeout: int
    is_critical: bool = True
    dependencies: Tuple[str, ...] = field(default_factory=tuple)  # Use tuple instead of list
    max_retries: int = 3
    retry_delay: int = 30
    resource_profile: str = "medium"  # low, medium, high, gpu
    sla_seconds: Optional[int] = None
    cache_key: Optional[str] = None
    incremental: bool = False
    priority: str = "normal"  # low, normal, high, critical

    def __post_init__(self):
        # Convert list to tuple if needed
        if isinstance(self.dependencies, list):
            object.__setattr__(self, 'dependencies', tuple(self.dependencies))
        if self.sla_seconds is None:
            object.__setattr__(self, 'sla_seconds', int(self.timeout * 0.8))


@dataclass
class ExecutionResult:
    """Enhanced result of script execution"""
    script: str
    status: ScriptStatus
    return_code: int
    duration: float
    attempt: int
    stdout: str = ""
    stderr: str = ""
    error_message: str = ""
    memory_peak: float = 0
    cpu_time: float = 0
    cache_hits: int = 0
    cache_misses: int = 0


@dataclass
class ExecutionMetrics:
    """Comprehensive execution metrics"""
    script: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    memory_start: float = 0
    memory_peak: float = 0
    memory_end: float = 0
    cpu_time_user: float = 0
    cpu_time_system: float = 0
    io_read_bytes: int = 0
    io_write_bytes: int = 0
    database_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ProcessingState:
    """State for incremental processing"""
    script: str
    last_processed_timestamp: datetime
    last_processed_id: Optional[int]
    records_processed: int
    last_run_duration: float
    last_run_status: str


# ============================================================================
# CIRCUIT BREAKER
# ============================================================================

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker implementation"""

    def __init__(self, name: str, failure_threshold: int = 5,
                 recovery_timeout: int = 60, success_threshold: int = 2):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitState.CLOSED
        self._lock = threading.Lock()

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker"""
        with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    logger.info(f"Circuit breaker '{self.name}' entering HALF_OPEN state")
                else:
                    raise Exception(f"Circuit breaker '{self.name}' is OPEN")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    async def async_call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function through circuit breaker"""
        with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                else:
                    raise Exception(f"Circuit breaker '{self.name}' is OPEN")

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit"""
        return (
                self.last_failure_time and
                datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout)
        )

    def _on_success(self):
        """Handle successful execution"""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    logger.info(f"Circuit breaker '{self.name}' is now CLOSED")
            else:
                self.failure_count = 0

    def _on_failure(self):
        """Handle failed execution"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()

            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker '{self.name}' is now OPEN (half-open test failed)")
            elif self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker '{self.name}' is now OPEN (threshold reached)")


# ============================================================================
# CACHE MANAGER
# ============================================================================

class CacheManager:
    """Redis-based caching system with fallback to memory cache"""

    def __init__(self, redis_url: Optional[str] = None, default_ttl: int = 3600):
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self._memory_cache: Dict[str, Tuple[Any, datetime]] = {}
        self._cache_lock = threading.Lock()
        self.stats = {'hits': 0, 'misses': 0, 'errors': 0}

        # Initialize Redis if available
        self._redis_client = None
        if REDIS_AVAILABLE and redis_url:
            try:
                self._redis_client = redis.from_url(redis_url, decode_responses=False)
                self._redis_client.ping()
                logger.info("Redis cache initialized")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}. Using memory cache.")
                self._redis_client = None

    def _generate_key(self, namespace: str, key: str) -> str:
        """Generate cache key with namespace"""
        return f"pipeline:{namespace}:{key}"

    def get(self, namespace: str, key: str) -> Optional[Any]:
        """Get value from cache"""
        cache_key = self._generate_key(namespace, key)

        # Try Redis first
        if self._redis_client:
            try:
                data = self._redis_client.get(cache_key)
                if data:
                    self.stats['hits'] += 1
                    return pickle.loads(data)
            except Exception as e:
                logger.debug(f"Redis get error: {e}")

        # Fallback to memory cache
        with self._cache_lock:
            if cache_key in self._memory_cache:
                value, expiry = self._memory_cache[cache_key]
                if datetime.now() < expiry:
                    self.stats['hits'] += 1
                    return value
                else:
                    del self._memory_cache[cache_key]

        self.stats['misses'] += 1
        return None

    def set(self, namespace: str, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        cache_key = self._generate_key(namespace, key)
        ttl = ttl or self.default_ttl

        # Try Redis first
        if self._redis_client:
            try:
                data = pickle.dumps(value)
                self._redis_client.setex(cache_key, ttl, data)
                return True
            except Exception as e:
                logger.debug(f"Redis set error: {e}")

        # Fallback to memory cache
        with self._cache_lock:
            expiry = datetime.now() + timedelta(seconds=ttl)
            self._memory_cache[cache_key] = (value, expiry)

            # Cleanup old entries
            if len(self._memory_cache) > 1000:  # Limit memory cache size
                now = datetime.now()
                expired = [k for k, (_, exp) in self._memory_cache.items() if exp < now]
                for k in expired[:100]:  # Remove up to 100 expired entries
                    del self._memory_cache[k]

        return True

    def get_stats(self) -> dict:
        """Get cache statistics"""
        total = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total * 100) if total > 0 else 0

        return {
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'errors': self.stats['errors'],
            'hit_rate': hit_rate,
            'memory_cache_size': len(self._memory_cache)
        }


# ============================================================================
# METRICS COLLECTOR
# ============================================================================

class MetricsCollector:
    """Comprehensive metrics collection with Prometheus support"""

    def __init__(self, prometheus_enabled: bool = True):
        self.metrics: Dict[str, ExecutionMetrics] = {}
        self.historical_metrics: List[ExecutionMetrics] = []
        self._lock = threading.Lock()

        # Initialize Prometheus metrics if available
        if prometheus_enabled and PROMETHEUS_AVAILABLE:
            self.prom_script_duration = Histogram(
                'pipeline_script_duration_seconds',
                'Script execution duration in seconds',
                ['script', 'status']
            )
            self.prom_script_total = Counter(
                'pipeline_script_executions_total',
                'Total number of script executions',
                ['script', 'status']
            )
            self.prom_active_scripts = Gauge(
                'pipeline_active_scripts',
                'Number of currently running scripts'
            )
            self.prom_system_cpu = Gauge(
                'pipeline_system_cpu_percent',
                'System CPU usage percentage'
            )
            self.prom_system_memory = Gauge(
                'pipeline_system_memory_percent',
                'System memory usage percentage'
            )
            self.prometheus_enabled = True
        else:
            self.prometheus_enabled = False

    def start_execution(self, script: str) -> ExecutionMetrics:
        """Start tracking execution metrics"""
        process = psutil.Process()

        metrics = ExecutionMetrics(
            script=script,
            start_time=datetime.now(),
            memory_start=process.memory_info().rss,
            cpu_time_user=process.cpu_times().user,
            cpu_time_system=process.cpu_times().system
        )

        with self._lock:
            self.metrics[script] = metrics

        if self.prometheus_enabled:
            self.prom_active_scripts.inc()

        return metrics

    def end_execution(self, script: str, status: str = "success"):
        """End tracking execution metrics"""
        with self._lock:
            if script not in self.metrics:
                return None

            metrics = self.metrics[script]
            process = psutil.Process()

            metrics.end_time = datetime.now()
            metrics.duration = (metrics.end_time - metrics.start_time).total_seconds()
            metrics.memory_end = process.memory_info().rss
            metrics.memory_peak = max(metrics.memory_start, metrics.memory_end)

            # Calculate CPU time
            cpu_times = process.cpu_times()
            metrics.cpu_time_user = cpu_times.user - metrics.cpu_time_user
            metrics.cpu_time_system = cpu_times.system - metrics.cpu_time_system

            # Store in history
            self.historical_metrics.append(metrics)

            # Update Prometheus metrics
            if self.prometheus_enabled:
                self.prom_script_duration.labels(
                    script=script, status=status
                ).observe(metrics.duration)
                self.prom_script_total.labels(
                    script=script, status=status
                ).inc()
                self.prom_active_scripts.dec()

            # Remove from active metrics
            del self.metrics[script]
            return metrics

    def update_system_metrics(self):
        """Update system-wide metrics"""
        if self.prometheus_enabled:
            self.prom_system_cpu.set(psutil.cpu_percent(interval=1))
            self.prom_system_memory.set(psutil.virtual_memory().percent)


# ============================================================================
# SMART SCHEDULER
# ============================================================================

class SmartScheduler:
    """Intelligent script scheduling based on historical data"""

    def __init__(self, history_file: str = "data/execution_history.json"):
        self.history_file = Path(history_file)
        self.history: Dict[str, Dict] = {}
        self.load_history()

    def load_history(self):
        """Load execution history from file"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    self.history = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load execution history: {e}")

    def save_history(self):
        """Save execution history to file"""
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save execution history: {e}")

    def update_history(self, script: str, duration: float, memory: float,
                       cpu: float, success: bool):
        """Update execution history for a script"""
        if script not in self.history:
            self.history[script] = {
                'avg_duration': duration,
                'avg_memory': memory,
                'avg_cpu': cpu,
                'failure_rate': 0 if success else 1,
                'last_execution': datetime.now().isoformat(),
                'execution_count': 1
            }
        else:
            hist = self.history[script]
            count = hist['execution_count']

            # Update moving averages
            hist['avg_duration'] = (hist['avg_duration'] * count + duration) / (count + 1)
            hist['avg_memory'] = (hist['avg_memory'] * count + memory) / (count + 1)
            hist['avg_cpu'] = (hist['avg_cpu'] * count + cpu) / (count + 1)

            # Update failure rate
            failures = hist['failure_rate'] * count
            if not success:
                failures += 1
            hist['failure_rate'] = failures / (count + 1)

            hist['last_execution'] = datetime.now().isoformat()
            hist['execution_count'] = count + 1

        self.save_history()

    def optimize_execution_order(self, scripts: List[ScriptConfig],
                                 max_parallel: int = 4) -> List[List[ScriptConfig]]:
        """Optimize execution order based on historical data and dependencies"""
        # First, resolve dependencies
        dependency_waves = self._resolve_dependencies(scripts)

        # Then optimize within each wave
        optimized_waves = []
        for wave in dependency_waves:
            optimized_wave = self._optimize_wave(wave, max_parallel)
            optimized_waves.extend(optimized_wave)

        return optimized_waves

    def _resolve_dependencies(self, scripts: List[ScriptConfig]) -> List[List[ScriptConfig]]:
        """Resolve dependencies and create execution waves"""
        waves = []
        remaining = list(scripts)  # Work with a list copy
        completed_names = set()  # Track completed script names

        while remaining:
            current_wave = []
            scripts_to_remove = []

            for script in remaining:
                # Check if all dependencies are completed
                deps_met = all(
                    dep in completed_names or dep not in [s.name for s in scripts]
                    for dep in script.dependencies
                )

                if deps_met:
                    current_wave.append(script)
                    scripts_to_remove.append(script)

            # Remove scripts that will be executed in this wave
            for script in scripts_to_remove:
                remaining.remove(script)
                completed_names.add(script.name)

            if not current_wave and remaining:
                # Circular dependency or missing dependency
                logger.error(f"Circular or missing dependencies detected for: {[s.name for s in remaining]}")
                current_wave = remaining.copy()  # Force execution of remaining
                remaining.clear()

            if current_wave:
                waves.append(current_wave)

        return waves

    def _optimize_wave(self, scripts: List[ScriptConfig],
                       max_parallel: int) -> List[List[ScriptConfig]]:
        """Optimize execution within a wave based on resource profiles"""
        # Score and sort scripts
        scored = []
        for script in scripts:
            score = self._calculate_priority_score(script)
            scored.append((score, script))

        scored.sort(reverse=True, key=lambda x: x[0])

        # Create sub-waves based on resource constraints
        sub_waves = []
        current_wave = []
        current_resources = {'cpu': 0, 'memory': 0}

        for _, script in scored:
            resources = self._get_resource_profile(script)

            if self._can_add_to_wave(current_resources, resources, max_parallel, len(current_wave)):
                current_wave.append(script)
                current_resources['cpu'] += resources['cpu']
                current_resources['memory'] += resources['memory']
            else:
                if current_wave:
                    sub_waves.append(current_wave)
                current_wave = [script]
                current_resources = resources.copy()

        if current_wave:
            sub_waves.append(current_wave)

        return sub_waves if sub_waves else [scripts]

    def _calculate_priority_score(self, script: ScriptConfig) -> float:
        """Calculate priority score for a script"""
        score = 0.0

        # Priority from config
        priority_weights = {'critical': 100, 'high': 50, 'normal': 10, 'low': 1}
        score += priority_weights.get(script.priority, 10)

        # Critical flag
        if script.is_critical:
            score += 30

        # Historical performance
        if script.name in self.history:
            hist = self.history[script.name]
            score += (1 - hist['failure_rate']) * 20

            # Prefer faster scripts
            if hist['avg_duration'] < 60:
                score += 10
            elif hist['avg_duration'] > 600:
                score -= 10

        return score

    def _get_resource_profile(self, script: ScriptConfig) -> Dict[str, float]:
        """Get estimated resource profile for a script"""
        profiles = {
            'low': {'cpu': 0.5, 'memory': 0.5},
            'medium': {'cpu': 1.0, 'memory': 1.0},
            'high': {'cpu': 2.0, 'memory': 2.0},
            'gpu': {'cpu': 1.0, 'memory': 3.0}
        }

        base_profile = profiles.get(script.resource_profile, profiles['medium'])

        # Adjust based on history
        if script.name in self.history:
            hist = self.history[script.name]
            if hist['avg_memory'] > 0:
                base_profile['memory'] = hist['avg_memory'] / (2 * 1024 ** 3)  # Normalize to GB

        return base_profile

    def _can_add_to_wave(self, current: Dict, new: Dict,
                         max_parallel: int, current_count: int) -> bool:
        """Check if a script can be added to current wave"""
        if current_count >= max_parallel:
            return False

        # Check resource constraints
        if current['cpu'] + new['cpu'] > 4.0:  # Max 400% CPU
            return False
        if current['memory'] + new['memory'] > 6.0:  # Max 6GB
            return False

        return True


# ============================================================================
# INCREMENTAL PROCESSOR
# ============================================================================

class IncrementalProcessor:
    """Handle incremental data processing"""

    def __init__(self, state_file: str = "data/processing_state.json"):
        self.state_file = Path(state_file)
        self.states: Dict[str, ProcessingState] = {}
        self.load_state()

    def load_state(self):
        """Load processing state from file"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    for script, state_data in data.items():
                        self.states[script] = ProcessingState(
                            script=script,
                            last_processed_timestamp=datetime.fromisoformat(
                                state_data['last_processed_timestamp']
                            ),
                            last_processed_id=state_data.get('last_processed_id'),
                            records_processed=state_data['records_processed'],
                            last_run_duration=state_data['last_run_duration'],
                            last_run_status=state_data['last_run_status']
                        )
            except Exception as e:
                logger.warning(f"Failed to load processing state: {e}")

    def save_state(self):
        """Save processing state to file"""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        for script, state in self.states.items():
            data[script] = {
                'last_processed_timestamp': state.last_processed_timestamp.isoformat(),
                'last_processed_id': state.last_processed_id,
                'records_processed': state.records_processed,
                'last_run_duration': state.last_run_duration,
                'last_run_status': state.last_run_status
            }

        try:
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save processing state: {e}")

    def should_process(self, script: str, force: bool = False) -> bool:
        """Determine if a script should run"""
        if force:
            return True

        state = self.states.get(script)
        if not state:
            return True

        # Check based on script type
        if "prices" in script:
            # Run if new market day
            return self._is_new_market_day(state.last_processed_timestamp)
        elif "fundamentals" in script:
            # Run weekly
            return datetime.now() - state.last_processed_timestamp > timedelta(days=7)
        else:
            # Default: run daily
            return datetime.now() - state.last_processed_timestamp > timedelta(hours=24)

    def _is_new_market_day(self, last_timestamp: datetime) -> bool:
        """Check if it's a new market day"""
        # Simple check - could be enhanced with market calendar
        today = datetime.now().date()
        last_date = last_timestamp.date()

        # Skip weekends
        if today.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False

        return today > last_date

    def update_state(self, script: str, timestamp: datetime,
                     records: int, duration: float, status: str = "success"):
        """Update processing state"""
        self.states[script] = ProcessingState(
            script=script,
            last_processed_timestamp=timestamp,
            last_processed_id=None,  # Could be enhanced
            records_processed=records,
            last_run_duration=duration,
            last_run_status=status
        )
        self.save_state()


# ============================================================================
# MAIN PIPELINE CONFIGURATION
# ============================================================================

class PipelineConfig:
    """Enhanced pipeline configuration with smart defaults"""

    # Resource limits
    MAX_CONCURRENT_SCRIPTS = min(4, os.cpu_count() or 4)
    MEMORY_THRESHOLD_GB = 8.0

    # Script configurations with enhanced metadata - INCREASED TIMEOUTS
    INGEST_SCRIPTS = [
        ScriptConfig("setup_schema.py", 300, True, (), priority="critical"),  # Increased to 5 min
        ScriptConfig("fetch_symbol_metadata.py", 900, True, ("setup_schema.py",),  # 15 min
                     cache_key="symbols", priority="high"),
        ScriptConfig("fetch_sp500_history.py", 300, False, ("setup_schema.py",),  # 5 min
                     cache_key="sp500"),
        ScriptConfig("fetch_prices.py", 2400, True, ("fetch_symbol_metadata.py",),  # 40 min
                     incremental=True, priority="critical", resource_profile="high"),
        ScriptConfig("fetch_dividend_history.py", 600, False, ("fetch_prices.py",),  # 10 min
                     incremental=True),
        ScriptConfig("fetch_fundamentals.py", 3600, True, ("fetch_prices.py",),  # 60 min
                     incremental=True, resource_profile="high"),
    ]

    ANALYSIS_SCRIPTS = [
        ScriptConfig("compute_forward_returns.py", 600, True, ("fetch_prices.py",),  # 10 min
                     resource_profile="high"),
        ScriptConfig("compute_dividend_growth_scores.py", 600, False, ("fetch_dividend_history.py",)),
        ScriptConfig("compute_value_momentum_and_growth_scores.py", 900, True,  # 15 min
                     ("fetch_fundamentals.py",), resource_profile="high"),
        ScriptConfig("compute_ai_dividend_scores.py", 600, False,
                     ("compute_dividend_growth_scores.py",)),
        ScriptConfig("compute_sp500_outperformance_scores.py", 600, True,
                     ("compute_forward_returns.py",)),
        ScriptConfig("train_ai_value_model.py", 1200, True,  # 20 min
                     ("compute_value_momentum_and_growth_scores.py",), resource_profile="gpu"),
        ScriptConfig("score_ai_value_model.py", 600, True, ("train_ai_value_model.py",)),
        ScriptConfig("train_ai_growth_model.py", 1200, True,  # 20 min
                     ("compute_value_momentum_and_growth_scores.py",), resource_profile="gpu"),
        ScriptConfig("score_ai_growth_model.py", 600, True, ("train_ai_growth_model.py",)),
        ScriptConfig("run_rank_value_stocks.py", 300, True, ("score_ai_value_model.py",)),
        ScriptConfig("run_rank_growth_stocks.py", 300, True, ("score_ai_growth_model.py",)),
        ScriptConfig("run_rank_dividend_stocks.py", 300, False, ("compute_ai_dividend_scores.py",)),
        ScriptConfig("run_rank_momentum_stocks.py", 300, True,
                     ("compute_sp500_outperformance_scores.py",)),
    ]

    MATERIALIZED_VIEWS = [
        "mv_latest_annual_fundamentals",
        "mv_symbol_with_metadata",
        "mv_latest_forward_returns",
        "mv_current_ai_portfolios",
    ]


# ============================================================================
# ENHANCED PIPELINE RUNNER
# ============================================================================

class EnhancedPipelineRunner:
    """Production-ready pipeline runner with all enhancements"""

    def __init__(self, args):
        self.args = args
        self.start_time = time.time()
        self.results: Dict[str, ExecutionResult] = {}
        self._shutdown_requested = False

        # Initialize configuration
        self.config_manager = ConfigManager(args.config if hasattr(args, 'config') else None)
        self.config = self.config_manager.config

        # Initialize components
        self.cache_manager = CacheManager(
            redis_url=self.config['cache'].get('redis_url'),
            default_ttl=self.config['cache'].get('default_ttl', 3600)
        )

        self.metrics_collector = MetricsCollector(
            prometheus_enabled=not args.no_metrics if hasattr(args, 'no_metrics') else True
        )

        self.smart_scheduler = SmartScheduler()
        self.incremental_processor = IncrementalProcessor()

        # Circuit breakers for each script
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}

        # Database manager
        self.db_manager = DatabaseManager(os.getenv("POSTGRES_URL", ""))

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Start Prometheus metrics server if available
        if PROMETHEUS_AVAILABLE and not getattr(args, 'no_metrics', False):
            try:
                start_http_server(self.config['monitoring'].get('prometheus_port', 9090))
                logger.info(f"Prometheus metrics server started on port {self.config['monitoring']['prometheus_port']}")
            except Exception as e:
                logger.warning(f"Failed to start Prometheus server: {e}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self._shutdown_requested = True

    def get_circuit_breaker(self, script: str) -> CircuitBreaker:
        """Get or create circuit breaker for a script"""
        if script not in self.circuit_breakers:
            cb_config = self.config['circuit_breaker']
            self.circuit_breakers[script] = CircuitBreaker(
                name=script,
                failure_threshold=cb_config['failure_threshold'],
                recovery_timeout=cb_config['recovery_timeout']
            )
        return self.circuit_breakers[script]

    async def check_prerequisites(self) -> bool:
        """Check system prerequisites before execution"""
        logger.info("Checking system prerequisites...")

        # Check memory
        memory = psutil.virtual_memory()
        memory_gb = memory.available / (1024 ** 3)
        if memory_gb < self.config['execution']['memory_threshold_gb']:
            logger.error(f"Insufficient memory: {memory_gb:.1f}GB available")
            return False

        # Check disk space
        disk = psutil.disk_usage(".")
        disk_gb = disk.free / (1024 ** 3)
        if disk_gb < self.config['execution']['disk_threshold_gb']:
            logger.error(f"Insufficient disk space: {disk_gb:.1f}GB available")
            return False

        # Check database connectivity
        if self.db_manager.engine:
            try:
                from sqlalchemy import text
                with self.db_manager.engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                logger.info("Database connection verified")
            except Exception as e:
                logger.error(f"Database connection failed: {e}")
                if not self.args.continue_on_error:
                    return False

        logger.info("All prerequisites met")
        return True

    async def execute_script(self, script: ScriptConfig) -> ExecutionResult:
        """Execute a single script with all enhancements"""
        script_path = Path(script.name)

        # Check if script exists
        if not script_path.exists():
            return ExecutionResult(
                script=script.name,
                status=ScriptStatus.SKIPPED,
                return_code=127,
                duration=0.0,
                attempt=0,
                error_message="Script not found"
            )

        # Check incremental processing
        if script.incremental and not self.args.force:
            if not self.incremental_processor.should_process(script.name):
                logger.info(f"Skipping {script.name} (no new data to process)")
                return ExecutionResult(
                    script=script.name,
                    status=ScriptStatus.SKIPPED,
                    return_code=0,
                    duration=0.0,
                    attempt=0,
                    error_message="No new data to process"
                )

        # Get circuit breaker
        circuit_breaker = self.get_circuit_breaker(script.name)

        # Try cache first if applicable
        if script.cache_key and not self.args.no_cache:
            cached_result = self.cache_manager.get("scripts", script.cache_key)
            if cached_result:
                logger.info(f"Using cached result for {script.name}")
                self.metrics_collector.start_execution(script.name)
                self.metrics_collector.end_execution(script.name, "cached")
                return ExecutionResult(
                    script=script.name,
                    status=ScriptStatus.SUCCESS,
                    return_code=0,
                    duration=0.0,
                    attempt=0,
                    cache_hits=1
                )

        # Execute with retries
        for attempt in range(1, script.max_retries + 1):
            if self._shutdown_requested:
                return ExecutionResult(
                    script=script.name,
                    status=ScriptStatus.FAILED,
                    return_code=130,
                    duration=0.0,
                    attempt=attempt,
                    error_message="Shutdown requested"
                )

            try:
                # Check circuit breaker
                if circuit_breaker.state == CircuitState.OPEN:
                    logger.warning(f"Circuit breaker open for {script.name}")
                    return ExecutionResult(
                        script=script.name,
                        status=ScriptStatus.CIRCUIT_OPEN,
                        return_code=1,
                        duration=0.0,
                        attempt=attempt,
                        error_message="Circuit breaker open"
                    )

                logger.info(f"Executing {script.name} (attempt {attempt}/{script.max_retries})")

                # Start metrics collection
                metrics = self.metrics_collector.start_execution(script.name)

                start_time = time.time()

                # Execute through circuit breaker
                process = await circuit_breaker.async_call(
                    asyncio.create_subprocess_exec,
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

                    # End metrics collection
                    execution_metrics = self.metrics_collector.end_execution(
                        script.name,
                        "success" if process.returncode == 0 else "failed"
                    )

                    # Update smart scheduler
                    if execution_metrics:
                        self.smart_scheduler.update_history(
                            script.name,
                            duration,
                            execution_metrics.memory_peak,
                            execution_metrics.cpu_time_user + execution_metrics.cpu_time_system,
                            process.returncode == 0
                        )

                    # Update incremental processor
                    if script.incremental:
                        self.incremental_processor.update_state(
                            script.name,
                            datetime.now(),
                            0,  # Would need to parse from output
                            duration,
                            "success" if process.returncode == 0 else "failed"
                        )

                    if process.returncode == 0:
                        logger.info(f"SUCCESS: {script.name} completed successfully ({duration:.2f}s)")

                        # Cache result if applicable
                        if script.cache_key:
                            self.cache_manager.set(
                                "scripts",
                                script.cache_key,
                                {"status": "success", "timestamp": datetime.now()},
                                ttl=3600
                            )

                        return ExecutionResult(
                            script=script.name,
                            status=ScriptStatus.SUCCESS,
                            return_code=0,
                            duration=duration,
                            attempt=attempt,
                            stdout=stdout.decode('utf-8', errors='replace')[-1000:],
                            stderr=stderr.decode('utf-8', errors='replace')[-1000:],
                            memory_peak=execution_metrics.memory_peak if execution_metrics else 0
                        )
                    else:
                        logger.warning(f"WARNING: {script.name} failed with return code {process.returncode}")
                        if attempt < script.max_retries:
                            await asyncio.sleep(script.retry_delay * (2 ** (attempt - 1)))  # Exponential backoff
                            continue

                        return ExecutionResult(
                            script=script.name,
                            status=ScriptStatus.FAILED,
                            return_code=process.returncode,
                            duration=duration,
                            attempt=attempt,
                            error_message=f"Process exited with code {process.returncode}"
                        )

                except asyncio.TimeoutError:
                    logger.error(f"TIMEOUT: {script.name} timed out after {script.timeout}s")
                    process.kill()
                    await process.wait()

                    if attempt < script.max_retries:
                        await asyncio.sleep(script.retry_delay * (2 ** (attempt - 1)))
                        continue

                    return ExecutionResult(
                        script=script.name,
                        status=ScriptStatus.TIMEOUT,
                        return_code=124,
                        duration=script.timeout,
                        attempt=attempt,
                        error_message=f"Timeout after {script.timeout}s"
                    )

            except Exception as e:
                logger.exception(f"ERROR: Unexpected error executing {script.name}: {e}")
                if attempt < script.max_retries:
                    await asyncio.sleep(script.retry_delay * (2 ** (attempt - 1)))
                    continue

                return ExecutionResult(
                    script=script.name,
                    status=ScriptStatus.FAILED,
                    return_code=1,
                    duration=time.time() - start_time if 'start_time' in locals() else 0,
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

        logger.info(f"Executing wave with {len(wave)} scripts: {[s.name for s in wave]}")

        # Update system metrics
        self.metrics_collector.update_system_metrics()

        # Limit concurrent execution
        max_concurrent = min(len(wave), self.config['execution']['max_concurrent_scripts'])
        semaphore = asyncio.Semaphore(max_concurrent)

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

    async def run(self):
        """Main execution method with all enhancements"""
        logger.info("Starting Enhanced EOD Pipeline v2.0...")
        print("\nEnhanced ACIS EOD Trading Pipeline v2.0")
        print("=" * 60)

        # Check prerequisites
        if not await self.check_prerequisites():
            logger.error("Prerequisites not met, aborting")
            sys.exit(1)

        # Build script list
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

        # Use smart scheduler to optimize execution
        execution_waves = self.smart_scheduler.optimize_execution_order(
            all_scripts,
            self.config['execution']['max_concurrent_scripts']
        )

        total_scripts = sum(len(wave) for wave in execution_waves)
        logger.info(f"Execution plan: {len(execution_waves)} waves, {total_scripts} scripts total")

        # Execute waves
        for wave_num, wave in enumerate(execution_waves, 1):
            if self._shutdown_requested:
                logger.info("Shutdown requested, stopping execution")
                break

            print(f"\nWave {wave_num}/{len(execution_waves)}: {[s.name for s in wave]}")

            wave_results = await self.execute_wave(wave)

            # Store results
            for result in wave_results:
                self.results[result.script] = result

            # Check if we should continue
            if not self.args.continue_on_error:
                critical_failures = [
                    r for r in wave_results
                    if r.status in [ScriptStatus.FAILED, ScriptStatus.TIMEOUT]
                       and any(s.name == r.script and s.is_critical for s in wave)
                ]
                if critical_failures:
                    logger.error(f"Critical script failures detected, stopping execution")
                    break

        # Refresh materialized views
        if not self.args.skip_mv_refresh:
            await self._refresh_materialized_views()

        # Generate and save report
        report = self._generate_report()
        self._save_report(report)

        # Print summary
        self._print_summary(report)

        # Print cache statistics
        cache_stats = self.cache_manager.get_stats()
        print(f"\nCache Statistics:")
        print(f"  Hits: {cache_stats['hits']}")
        print(f"  Misses: {cache_stats['misses']}")
        print(f"  Hit Rate: {cache_stats['hit_rate']:.1f}%")

        # Exit with appropriate code
        failed_count = len([r for r in self.results.values()
                            if r.status in [ScriptStatus.FAILED, ScriptStatus.TIMEOUT]])
        sys.exit(0 if failed_count == 0 else 1)

    async def _refresh_materialized_views(self):
        """Refresh materialized views"""
        print("\nRefreshing materialized views...")
        refreshed = []
        failed = []

        if not self.db_manager.engine:
            logger.warning("Database engine not available, skipping materialized view refresh")
            return

        from sqlalchemy import text

        for mv in PipelineConfig.MATERIALIZED_VIEWS:
            try:
                logger.info(f"Refreshing {mv}...")
                with self.db_manager.engine.begin() as conn:
                    # Try concurrent refresh first
                    try:
                        conn.execute(text(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {mv}"))
                        refreshed.append(mv)
                        logger.info(f"Refreshed {mv} (concurrent)")
                    except Exception as e1:
                        # Fallback to non-concurrent
                        logger.debug(f"Concurrent refresh failed for {mv}: {e1}")
                        conn.execute(text(f"REFRESH MATERIALIZED VIEW {mv}"))
                        refreshed.append(mv)
                        logger.info(f"Refreshed {mv} (non-concurrent)")
            except Exception as e:
                logger.error(f"Failed to refresh {mv}: {e}")
                failed.append(mv)

        if refreshed:
            print(f"[OK] Refreshed {len(refreshed)} materialized views")
        if failed:
            print(f"[WARNING] Failed to refresh {len(failed)} views")

    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive execution report"""
        total_duration = time.time() - self.start_time

        successful = [r for r in self.results.values() if r.status == ScriptStatus.SUCCESS]
        failed = [r for r in self.results.values()
                  if r.status in [ScriptStatus.FAILED, ScriptStatus.TIMEOUT]]
        skipped = [r for r in self.results.values() if r.status == ScriptStatus.SKIPPED]

        return {
            "pipeline": {
                "start_time": datetime.fromtimestamp(self.start_time, timezone.utc).isoformat(),
                "end_time": datetime.now(timezone.utc).isoformat(),
                "total_duration": total_duration,
                "status": "SUCCESS" if not failed else "FAILED",
                "version": "2.0.0"
            },
            "execution": {
                "total_scripts": len(self.results),
                "successful": len(successful),
                "failed": len(failed),
                "skipped": len(skipped),
                "cache_stats": self.cache_manager.get_stats()
            },
            "results": {name: asdict(result) for name, result in self.results.items()}
        }

    def _save_report(self, report: Dict[str, Any]):
        """Save execution report to file"""
        report_dir = Path("logs/reports")
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"pipeline_report_{timestamp}.json"

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Report saved to {report_path}")

    def _print_summary(self, report: Dict[str, Any]):
        """Print execution summary"""
        pipeline = report["pipeline"]
        execution = report["execution"]

        print("\n" + "=" * 80)
        print("ENHANCED PIPELINE EXECUTION SUMMARY")
        print("=" * 80)
        print(f"Status: {'SUCCESS' if pipeline['status'] == 'SUCCESS' else 'FAILED'}")
        print(f"Duration: {pipeline['total_duration']:.2f} seconds")
        print(f"Scripts executed: {execution['total_scripts']}")
        print(f"  Successful: {execution['successful']}")
        print(f"  Failed: {execution['failed']}")
        print(f"  Skipped: {execution['skipped']}")
        print("=" * 80)


# ============================================================================
# DATABASE MANAGER (keeping from original)
# ============================================================================

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


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Enhanced ACIS EOD Trading Pipeline v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Run full pipeline
  %(prog)s --only-phase ingest       # Run only ingestion phase
  %(prog)s --only fetch_prices.py    # Run only specific script
  %(prog)s --dry-run                 # Show what would be executed
  %(prog)s --continue-on-error       # Continue even if scripts fail
  %(prog)s --no-cache                # Disable caching
  %(prog)s --force                   # Force run all scripts
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
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable caching")
    parser.add_argument("--force", action="store_true",
                        help="Force run all scripts (ignore incremental)")
    parser.add_argument("--no-metrics", action="store_true",
                        help="Disable Prometheus metrics")
    parser.add_argument("--config", help="Path to configuration file")

    args = parser.parse_args()

    # Setup or reconfigure logging based on verbose flag
    global logger
    logger = setup_logging(args.verbose)

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