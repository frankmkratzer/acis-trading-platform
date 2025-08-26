#!/usr/bin/env python3
"""
Basic system tests for ACIS Trading Platform
Tests core functionality without database dependencies
"""

import os
import sys
import importlib.util
import time
import json
from pathlib import Path
from datetime import datetime
import unittest
from typing import Dict, List

# Basic test framework
class TestRunner:
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.failures = []
    
    def test(self, name: str, condition: bool, error_msg: str = ""):
        """Run a test and track results"""
        self.tests_run += 1
        if condition:
            self.tests_passed += 1
            print(f"[PASS] {name}")
            return True
        else:
            self.failures.append(f"{name}: {error_msg}")
            print(f"[FAIL] {name}: {error_msg}")
            return False
    
    def summary(self):
        """Print test summary"""
        print(f"\nTest Results: {self.tests_passed}/{self.tests_run} passed")
        if self.failures:
            print("\nFailures:")
            for failure in self.failures:
                print(f"  - {failure}")
        return self.tests_passed == self.tests_run

def test_python_environment(runner: TestRunner):
    """Test Python environment and basic dependencies"""
    print("\n=== Testing Python Environment ===")
    
    # Test Python version
    runner.test(
        "Python version >= 3.8",
        sys.version_info >= (3, 8),
        f"Found Python {sys.version_info.major}.{sys.version_info.minor}"
    )
    
    # Test core dependencies
    required_modules = [
        'pandas', 'numpy', 'datetime', 'json', 'pathlib', 
        'os', 'sys', 'time', 'logging', 'argparse'
    ]
    
    for module in required_modules:
        try:
            __import__(module)
            runner.test(f"Module '{module}' import", True)
        except ImportError as e:
            runner.test(f"Module '{module}' import", False, str(e))

def test_file_structure(runner: TestRunner):
    """Test that required files exist"""
    print("\n=== Testing File Structure ===")
    
    base_path = Path(".")
    
    # Core files that should exist
    required_files = [
        "pipeline_config.yaml",
        "run_eod_full_pipeline.py",
        "risk_management.py",
        "backtest_engine.py",
        "live_trading_engine.py",
        "train_ai_value_model.py"
    ]
    
    for file in required_files:
        file_path = base_path / file
        runner.test(
            f"File exists: {file}",
            file_path.exists(),
            f"Missing file: {file_path}"
        )

def test_configuration_loading(runner: TestRunner):
    """Test configuration system"""
    print("\n=== Testing Configuration ===")
    
    try:
        # Test YAML loading
        config_file = Path("pipeline_config.yaml")
        if config_file.exists():
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            runner.test(
                "Config YAML loads",
                isinstance(config, dict),
                "Config is not a dictionary"
            )
            
            # Test required config sections
            required_sections = ['pipeline', 'execution', 'scripts']
            for section in required_sections:
                runner.test(
                    f"Config section '{section}' exists",
                    section in config,
                    f"Missing config section: {section}"
                )
        else:
            runner.test("Config file exists", False, "pipeline_config.yaml not found")
    
    except ImportError:
        runner.test("PyYAML available", False, "PyYAML not installed")
    except Exception as e:
        runner.test("Config loading", False, str(e))

def test_risk_management(runner: TestRunner):
    """Test risk management module"""
    print("\n=== Testing Risk Management ===")
    
    try:
        # Import and test basic functionality
        spec = importlib.util.spec_from_file_location("risk_management", "risk_management.py")
        risk_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(risk_module)
        
        # Test RiskManager class exists
        runner.test(
            "RiskManager class exists",
            hasattr(risk_module, 'RiskManager'),
            "RiskManager class not found"
        )
        
        if hasattr(risk_module, 'RiskManager'):
            # Create instance
            risk_manager = risk_module.RiskManager()
            
            # Test basic methods exist
            methods = ['calculate_portfolio_metrics', 'optimize_portfolio', 'calculate_var']
            for method in methods:
                runner.test(
                    f"RiskManager.{method} exists",
                    hasattr(risk_manager, method),
                    f"Method {method} not found"
                )
    
    except Exception as e:
        runner.test("Risk management import", False, str(e))

def test_backtest_engine(runner: TestRunner):
    """Test backtesting functionality"""
    print("\n=== Testing Backtest Engine ===")
    
    try:
        spec = importlib.util.spec_from_file_location("backtest_engine", "backtest_engine.py")
        backtest_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(backtest_module)
        
        runner.test(
            "BacktestEngine class exists",
            hasattr(backtest_module, 'BacktestEngine'),
            "BacktestEngine class not found"
        )
        
        runner.test(
            "BacktestConfig class exists",
            hasattr(backtest_module, 'BacktestConfig'),
            "BacktestConfig class not found"
        )
    
    except Exception as e:
        runner.test("Backtest engine import", False, str(e))

def test_model_components(runner: TestRunner):
    """Test AI model components"""
    print("\n=== Testing AI Model Components ===")
    
    try:
        # Test value model
        spec = importlib.util.spec_from_file_location("train_ai_value_model", "train_ai_value_model.py")
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        
        runner.test(
            "ValueModelTrainer class exists",
            hasattr(model_module, 'ValueModelTrainer'),
            "ValueModelTrainer class not found"
        )
        
        # Test if we can create an instance
        if hasattr(model_module, 'ValueModelTrainer'):
            trainer = model_module.ValueModelTrainer()
            runner.test(
                "ValueModelTrainer instantiation",
                trainer is not None,
                "Failed to create ValueModelTrainer instance"
            )
    
    except Exception as e:
        runner.test("AI model import", False, str(e))

def create_sample_data(runner: TestRunner):
    """Create sample data for testing"""
    print("\n=== Creating Sample Data ===")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create data directory
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        # Create sample price data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        price_data = []
        for symbol in symbols:
            base_price = np.random.uniform(50, 300)
            prices = base_price * (1 + np.cumsum(np.random.normal(0, 0.02, len(dates))))
            
            for i, date in enumerate(dates):
                price_data.append({
                    'symbol': symbol,
                    'date': date,
                    'close': prices[i],
                    'volume': np.random.randint(1000000, 10000000)
                })
        
        df = pd.DataFrame(price_data)
        
        # Save to CSV
        sample_file = data_dir / "sample_prices.csv"
        df.to_csv(sample_file, index=False)
        
        runner.test(
            "Sample data creation",
            sample_file.exists(),
            "Failed to create sample data file"
        )
        
        runner.test(
            "Sample data size",
            len(df) == len(symbols) * len(dates),
            f"Expected {len(symbols) * len(dates)} rows, got {len(df)}"
        )
        
    except ImportError as e:
        runner.test("Pandas/numpy for sample data", False, f"Missing dependency: {e}")
    except Exception as e:
        runner.test("Sample data creation", False, str(e))

def test_pipeline_components(runner: TestRunner):
    """Test pipeline components"""
    print("\n=== Testing Pipeline Components ===")
    
    # Test if pipeline script is executable
    pipeline_script = Path("run_eod_full_pipeline.py")
    if pipeline_script.exists():
        runner.test(
            "Pipeline script exists",
            True
        )
        
        # Test if it has main execution
        with open(pipeline_script, 'r', encoding='utf-8') as f:
            content = f.read()
            runner.test(
                "Pipeline has main execution",
                'if __name__ == "__main__"' in content,
                "No main execution block found"
            )
    else:
        runner.test("Pipeline script exists", False, "run_eod_full_pipeline.py not found")

def main():
    """Run all tests"""
    print("ACIS Trading Platform - System Tests")
    print("=" * 60)
    
    runner = TestRunner()
    
    # Run test suites
    test_python_environment(runner)
    test_file_structure(runner)
    test_configuration_loading(runner) 
    test_risk_management(runner)
    test_backtest_engine(runner)
    test_model_components(runner)
    create_sample_data(runner)
    test_pipeline_components(runner)
    
    # Print final summary
    print("\n" + "=" * 60)
    success = runner.summary()
    
    if success:
        print("\nAll tests passed! The system is ready for testing.")
        print("\nNext steps:")
        print("1. Set up your .env file with database credentials")
        print("2. Run: python test_db_connection.py")
        print("3. Run: python run_eod_full_pipeline.py --dry-run")
    else:
        print("\nSome tests failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()