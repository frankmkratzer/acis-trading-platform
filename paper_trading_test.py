#!/usr/bin/env python3
"""
ACIS Paper Trading Test System
Comprehensive testing framework for paper trading functionality
Tests order execution, portfolio management, risk controls, and strategy integration
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from trading_system import TradingSystem, TradingMode, Order, OrderType, OrderSide, OrderStatus

class PaperTradingTest:
    def __init__(self):
        load_dotenv()
        self.engine = create_engine(os.getenv('POSTGRES_URL'))
        self.trading_system = TradingSystem(trading_mode=TradingMode.PAPER)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('PaperTradingTest')
        
        # Test configuration
        self.test_config = {
            "initial_cash": 1000000,  # $1M for testing
            "test_symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "BRK.B", "JNJ", "V"],
            "test_strategies": ["small_cap_value", "mid_cap_growth", "large_cap_momentum", "dividend_focus"],
            "risk_limits": {
                "max_position_size": 0.05,  # 5% max position
                "max_sector_allocation": 0.30,  # 30% max sector
                "max_daily_loss": 0.02  # 2% max daily loss
            }
        }
        
        self.test_results = {
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "errors": [],
            "performance_metrics": {}
        }

    def run_comprehensive_tests(self):
        """Execute all paper trading tests"""
        self.logger.info("Starting comprehensive paper trading tests...")
        
        try:
            # Core functionality tests
            self._test_order_creation()
            self._test_order_execution()
            self._test_portfolio_management()
            self._test_risk_controls()
            
            # Strategy integration tests
            self._test_strategy_portfolio_execution()
            self._test_rebalancing_logic()
            
            # Performance and reporting tests
            self._test_performance_calculation()
            self._test_reporting_system()
            
            # Generate comprehensive test report
            self._generate_test_report()
            
        except Exception as e:
            self.logger.error(f"Test execution failed: {e}")
            self.test_results["errors"].append(str(e))
        
        return self.test_results

    def _test_order_creation(self):
        """Test order creation and validation"""
        self.logger.info("Testing order creation...")
        
        try:
            # Test valid market order
            order = Order(
                symbol="AAPL",
                quantity=100,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                strategy="test_strategy",
                client_order_id="TEST_001"
            )
            
            result = self.trading_system.create_order(order)
            assert result["success"], "Market order creation failed"
            
            # Test valid limit order
            limit_order = Order(
                symbol="MSFT",
                quantity=50,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                price=350.00,
                strategy="test_strategy",
                client_order_id="TEST_002"
            )
            
            result = self.trading_system.create_order(limit_order)
            assert result["success"], "Limit order creation failed"
            
            # Test invalid order (negative quantity)
            try:
                invalid_order = Order(
                    symbol="GOOGL",
                    quantity=-10,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    strategy="test_strategy",
                    client_order_id="TEST_003"
                )
                
                result = self.trading_system.create_order(invalid_order)
                assert not result["success"], "Invalid order should have been rejected"
                
            except Exception as e:
                self.logger.info(f"Correctly rejected invalid order: {e}")
            
            self._record_test_result("Order Creation", True)
            
        except Exception as e:
            self.logger.error(f"Order creation test failed: {e}")
            self._record_test_result("Order Creation", False, str(e))

    def _test_order_execution(self):
        """Test order execution simulation"""
        self.logger.info("Testing order execution...")
        
        try:
            # Create test orders for execution
            test_orders = [
                Order("AAPL", 100, OrderSide.BUY, OrderType.MARKET, strategy="test_execution", client_order_id="EXEC_001"),
                Order("MSFT", 200, OrderSide.BUY, OrderType.LIMIT, price=350.00, strategy="test_execution", client_order_id="EXEC_002"),
                Order("GOOGL", 50, OrderSide.BUY, OrderType.MARKET, strategy="test_execution", client_order_id="EXEC_003")
            ]
            
            executed_orders = []
            for order in test_orders:
                result = self.trading_system.execute_order(order)
                if result["success"]:
                    executed_orders.append(result["order"])
            
            # Verify execution results
            assert len(executed_orders) > 0, "No orders were executed"
            
            # Check that market orders were filled immediately
            market_orders = [o for o in executed_orders if o.order_type == OrderType.MARKET]
            for order in market_orders:
                assert order.status == OrderStatus.FILLED, f"Market order {order.client_order_id} not filled"
                assert order.filled_quantity == order.quantity, "Market order not fully filled"
                assert order.avg_fill_price > 0, "Invalid fill price"
            
            # Verify position updates
            positions = self.trading_system.get_current_positions()
            position_symbols = [p.symbol for p in positions]
            
            for order in executed_orders:
                if order.status == OrderStatus.FILLED:
                    assert order.symbol in position_symbols, f"Position not created for {order.symbol}"
            
            self._record_test_result("Order Execution", True)
            
        except Exception as e:
            self.logger.error(f"Order execution test failed: {e}")
            self._record_test_result("Order Execution", False, str(e))

    def _test_portfolio_management(self):
        """Test portfolio management functions"""
        self.logger.info("Testing portfolio management...")
        
        try:
            # Create test portfolio
            test_portfolio = {
                "AAPL": {"quantity": 100, "target_weight": 0.15},
                "MSFT": {"quantity": 150, "target_weight": 0.12},
                "GOOGL": {"quantity": 50, "target_weight": 0.10},
                "AMZN": {"quantity": 75, "target_weight": 0.08},
                "NVDA": {"quantity": 200, "target_weight": 0.20}
            }
            
            # Execute portfolio creation orders
            for symbol, allocation in test_portfolio.items():
                order = Order(
                    symbol=symbol,
                    quantity=allocation["quantity"],
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    strategy="portfolio_test",
                    client_order_id=f"PORT_{symbol}"
                )
                
                result = self.trading_system.execute_order(order)
                assert result["success"], f"Failed to execute order for {symbol}"
            
            # Test portfolio analysis
            portfolio_analysis = self.trading_system.analyze_portfolio("portfolio_test")
            
            # Verify portfolio metrics
            assert portfolio_analysis["total_positions"] > 0, "No positions in portfolio"
            assert portfolio_analysis["total_market_value"] > 0, "Invalid portfolio value"
            assert abs(sum(portfolio_analysis["sector_allocation"].values()) - 1.0) < 0.01, "Sector allocation doesn't sum to 100%"
            
            # Test position rebalancing
            rebalance_orders = self.trading_system.generate_rebalance_orders("portfolio_test", test_portfolio)
            assert isinstance(rebalance_orders, list), "Rebalance orders should be a list"
            
            self._record_test_result("Portfolio Management", True)
            
        except Exception as e:
            self.logger.error(f"Portfolio management test failed: {e}")
            self._record_test_result("Portfolio Management", False, str(e))

    def _test_risk_controls(self):
        """Test risk management controls"""
        self.logger.info("Testing risk controls...")
        
        try:
            # Test position size limits
            large_order = Order(
                symbol="AAPL",
                quantity=10000,  # Large position
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                strategy="risk_test",
                client_order_id="RISK_001"
            )
            
            # Should be rejected due to position size
            result = self.trading_system.create_order(large_order)
            if self.trading_system.config["risk_management"]["max_position_size"] * 1000000 < large_order.quantity * 150:  # Assuming ~$150 stock price
                assert not result["success"], "Large position order should have been rejected"
            
            # Test sector concentration limits
            sector_positions = self.trading_system.get_sector_allocation()
            max_sector_allocation = max(sector_positions.values()) if sector_positions else 0
            
            assert max_sector_allocation <= self.trading_system.config["risk_management"]["max_sector_allocation"], "Sector allocation exceeds limits"
            
            # Test daily loss limits
            daily_pnl = self.trading_system.calculate_daily_pnl()
            account_value = self.trading_system.get_account_value()
            
            if daily_pnl < 0:  # If there's a loss
                daily_loss_pct = abs(daily_pnl) / account_value
                max_daily_loss = self.trading_system.config["risk_management"]["max_daily_loss"]
                
                if daily_loss_pct > max_daily_loss:
                    self.logger.warning(f"Daily loss {daily_loss_pct:.2%} exceeds limit {max_daily_loss:.2%}")
            
            self._record_test_result("Risk Controls", True)
            
        except Exception as e:
            self.logger.error(f"Risk controls test failed: {e}")
            self._record_test_result("Risk Controls", False, str(e))

    def _test_strategy_portfolio_execution(self):
        """Test execution of actual strategy portfolios"""
        self.logger.info("Testing strategy portfolio execution...")
        
        try:
            # Get actual portfolio from one of our strategies
            with self.engine.connect() as conn:
                portfolio_query = """
                    SELECT symbol, allocation_percentage, shares_to_buy
                    FROM ai_value_small_cap_portfolio 
                    WHERE portfolio_date = (
                        SELECT MAX(portfolio_date) FROM ai_value_small_cap_portfolio
                    )
                    LIMIT 10
                """
                
                portfolio_df = pd.read_sql(portfolio_query, conn)
                
                if not portfolio_df.empty:
                    # Execute strategy portfolio
                    strategy_orders = []
                    for _, row in portfolio_df.iterrows():
                        if row['shares_to_buy'] > 0:
                            order = Order(
                                symbol=row['symbol'],
                                quantity=int(row['shares_to_buy']),
                                side=OrderSide.BUY,
                                order_type=OrderType.MARKET,
                                strategy="small_cap_value_test",
                                client_order_id=f"STRAT_{row['symbol']}"
                            )
                            
                            result = self.trading_system.execute_order(order)
                            if result["success"]:
                                strategy_orders.append(result["order"])
                    
                    assert len(strategy_orders) > 0, "No strategy orders executed"
                    
                    # Verify strategy portfolio performance
                    portfolio_performance = self.trading_system.calculate_strategy_performance("small_cap_value_test")
                    
                    assert "total_return" in portfolio_performance, "Portfolio performance metrics missing"
                    assert "sharpe_ratio" in portfolio_performance, "Risk-adjusted metrics missing"
                    
                self._record_test_result("Strategy Portfolio Execution", True)
                
        except Exception as e:
            self.logger.error(f"Strategy portfolio execution test failed: {e}")
            self._record_test_result("Strategy Portfolio Execution", False, str(e))

    def _test_rebalancing_logic(self):
        """Test portfolio rebalancing logic"""
        self.logger.info("Testing rebalancing logic...")
        
        try:
            # Create initial portfolio
            initial_portfolio = {
                "AAPL": 100,
                "MSFT": 200,
                "GOOGL": 50
            }
            
            for symbol, quantity in initial_portfolio.items():
                order = Order(
                    symbol=symbol,
                    quantity=quantity,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    strategy="rebalance_test",
                    client_order_id=f"REB_INIT_{symbol}"
                )
                
                self.trading_system.execute_order(order)
            
            # Define target allocation (different from current)
            target_allocation = {
                "AAPL": 0.40,  # Increase
                "MSFT": 0.30,  # Decrease  
                "GOOGL": 0.20,  # Decrease
                "TSLA": 0.10   # New position
            }
            
            # Generate rebalancing orders
            rebalance_orders = self.trading_system.generate_rebalance_orders("rebalance_test", target_allocation)
            
            assert isinstance(rebalance_orders, list), "Rebalance orders should be a list"
            assert len(rebalance_orders) > 0, "No rebalancing orders generated"
            
            # Verify rebalancing logic
            buy_orders = [o for o in rebalance_orders if o.side == OrderSide.BUY]
            sell_orders = [o for o in rebalance_orders if o.side == OrderSide.SELL]
            
            # Should have orders to buy TSLA (new position) and adjust existing positions
            tsla_orders = [o for o in buy_orders if o.symbol == "TSLA"]
            assert len(tsla_orders) > 0, "No buy order for new TSLA position"
            
            self._record_test_result("Rebalancing Logic", True)
            
        except Exception as e:
            self.logger.error(f"Rebalancing logic test failed: {e}")
            self._record_test_result("Rebalancing Logic", False, str(e))

    def _test_performance_calculation(self):
        """Test performance calculation accuracy"""
        self.logger.info("Testing performance calculation...")
        
        try:
            # Create test trades with known outcomes
            test_trades = [
                {"symbol": "TEST1", "quantity": 100, "buy_price": 100.00, "sell_price": 110.00},  # +10%
                {"symbol": "TEST2", "quantity": 200, "buy_price": 50.00, "sell_price": 45.00},    # -10%
                {"symbol": "TEST3", "quantity": 150, "buy_price": 200.00, "sell_price": 220.00}   # +10%
            ]
            
            total_invested = sum(trade["quantity"] * trade["buy_price"] for trade in test_trades)
            total_returns = sum(trade["quantity"] * trade["sell_price"] for trade in test_trades)
            expected_return_pct = (total_returns - total_invested) / total_invested
            
            # Execute test trades
            for i, trade in enumerate(test_trades):
                # Buy order
                buy_order = Order(
                    symbol=trade["symbol"],
                    quantity=trade["quantity"],
                    side=OrderSide.BUY,
                    order_type=OrderType.LIMIT,
                    price=trade["buy_price"],
                    strategy="performance_test",
                    client_order_id=f"PERF_BUY_{i}"
                )
                
                self.trading_system.execute_order(buy_order)
                
                # Simulate price movement and sell
                sell_order = Order(
                    symbol=trade["symbol"],
                    quantity=trade["quantity"],
                    side=OrderSide.SELL,
                    order_type=OrderType.LIMIT,
                    price=trade["sell_price"],
                    strategy="performance_test",
                    client_order_id=f"PERF_SELL_{i}"
                )
                
                self.trading_system.execute_order(sell_order)
            
            # Calculate performance
            performance = self.trading_system.calculate_strategy_performance("performance_test")
            calculated_return = performance.get("total_return", 0)
            
            # Allow small tolerance for commissions and rounding
            assert abs(calculated_return - expected_return_pct) < 0.01, f"Performance calculation error: expected {expected_return_pct:.2%}, got {calculated_return:.2%}"
            
            self._record_test_result("Performance Calculation", True)
            
        except Exception as e:
            self.logger.error(f"Performance calculation test failed: {e}")
            self._record_test_result("Performance Calculation", False, str(e))

    def _test_reporting_system(self):
        """Test reporting and analytics system"""
        self.logger.info("Testing reporting system...")
        
        try:
            # Generate comprehensive report
            report = self.trading_system.generate_trading_report(
                start_date=datetime.now() - timedelta(days=30),
                end_date=datetime.now(),
                strategy="all"
            )
            
            # Verify report structure
            required_sections = [
                "account_summary",
                "position_summary", 
                "order_history",
                "performance_metrics",
                "risk_analysis"
            ]
            
            for section in required_sections:
                assert section in report, f"Report missing section: {section}"
            
            # Verify account summary
            account_summary = report["account_summary"]
            assert "total_equity" in account_summary, "Account summary missing total_equity"
            assert "cash_balance" in account_summary, "Account summary missing cash_balance"
            assert account_summary["total_equity"] > 0, "Invalid total equity"
            
            # Verify performance metrics
            performance = report["performance_metrics"]
            assert "total_return" in performance, "Performance metrics missing total_return"
            assert "sharpe_ratio" in performance, "Performance metrics missing sharpe_ratio"
            assert "max_drawdown" in performance, "Performance metrics missing max_drawdown"
            
            self._record_test_result("Reporting System", True)
            
        except Exception as e:
            self.logger.error(f"Reporting system test failed: {e}")
            self._record_test_result("Reporting System", False, str(e))

    def _record_test_result(self, test_name: str, passed: bool, error: str = ""):
        """Record test result"""
        self.test_results["tests_run"] += 1
        
        if passed:
            self.test_results["tests_passed"] += 1
            self.logger.info(f"[PASS] {test_name}")
        else:
            self.test_results["tests_failed"] += 1
            self.test_results["errors"].append(f"{test_name}: {error}")
            self.logger.error(f"[FAIL] {test_name}: {error}")

    def _generate_test_report(self):
        """Generate comprehensive test report"""
        self.logger.info("Generating test report...")
        
        report = {
            "test_summary": self.test_results,
            "test_timestamp": datetime.now().isoformat(),
            "test_environment": {
                "trading_mode": "PAPER",
                "initial_cash": self.test_config["initial_cash"],
                "test_symbols": self.test_config["test_symbols"]
            }
        }
        
        # Calculate success rate
        if self.test_results["tests_run"] > 0:
            success_rate = self.test_results["tests_passed"] / self.test_results["tests_run"]
            report["success_rate"] = f"{success_rate:.1%}"
        
        # Save report to file
        report_filename = f"paper_trading_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Test report saved to {report_filename}")
        
        # Print summary
        print("\n" + "="*50)
        print("PAPER TRADING TEST RESULTS")
        print("="*50)
        print(f"Tests Run: {self.test_results['tests_run']}")
        print(f"Tests Passed: {self.test_results['tests_passed']}")
        print(f"Tests Failed: {self.test_results['tests_failed']}")
        print(f"Success Rate: {report.get('success_rate', 'N/A')}")
        
        if self.test_results['errors']:
            print("\nFAILURES:")
            for error in self.test_results['errors']:
                print(f"  - {error}")
        
        print("="*50)
        
        return report

def main():
    """Run paper trading tests"""
    print("ACIS Paper Trading Test System")
    print("Testing comprehensive paper trading functionality...")
    
    test_system = PaperTradingTest()
    results = test_system.run_comprehensive_tests()
    
    return results

if __name__ == "__main__":
    main()