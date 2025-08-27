#!/usr/bin/env python3
"""
Centralized logging configuration for ACIS Trading Platform
Provides consistent logging setup across all scripts
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logger(script_name: str, log_level: str = "INFO", verbose: bool = False) -> logging.Logger:
    """
    Setup standardized logger for ACIS scripts
    
    Args:
        script_name: Name of the script (e.g., "fetch_prices") 
        log_level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
        verbose: Enable verbose console output
    
    Returns:
        Configured logger instance
    """
    
    # Ensure logs directory exists
    logs_dir = Path(__file__).parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(script_name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler with rotation (10MB max, keep 5 backups)
    log_file = logs_dir / f"{script_name}.log"
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Standard formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Prevent duplicate logs
    logger.propagate = False
    
    return logger

def log_script_start(logger: logging.Logger, script_name: str, description: str = ""):
    """Log script startup with standard format"""
    logger.info("=" * 80)
    logger.info(f"Starting {script_name}")
    if description:
        logger.info(f"Description: {description}")
    logger.info("=" * 80)

def log_script_end(logger: logging.Logger, script_name: str, success: bool = True, 
                   duration: float = None, stats: dict = None):
    """Log script completion with standard format"""
    logger.info("=" * 80)
    status = "COMPLETED SUCCESSFULLY" if success else "FAILED"
    logger.info(f"{script_name} {status}")
    
    if duration:
        logger.info(f"Execution time: {duration:.2f} seconds")
    
    if stats:
        for key, value in stats.items():
            logger.info(f"{key}: {value}")
    
    logger.info("=" * 80)