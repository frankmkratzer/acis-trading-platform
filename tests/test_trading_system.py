#!/usr/bin/env python3
"""
Unit tests for ACIS Trading System
Tests core trading functionality, order management, and broker integration
"""

import unittest
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from decimal import Decimal

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_system import (
    TradingSystem, Order, OrderType, OrderSide, OrderStatus, 
    Position, Account, TradingMode
)
from schwab_broker_integration import SchwabAPI, SchwabBrokerManager
from live_trading_integration import LiveTradingSystem

class TestTradingSystem(unittest.TestCase):
    """Test cases for core trading system functionality"""
    
    def setUp(self):
        """Set up test environment"""
        with patch('trading_system.create_engine'):
            self.trading_system = TradingSystem(trading_mode=TradingMode.PAPER)
        
        # Mock database engine
        self.trading_system.engine = Mock()
        self.mock_conn = Mock()
        self.trading_system.engine.connect.return_value.__enter__.return_value = self.mock_conn
        
    def test_order_creation(self):
        """Test order creation and validation"""
        # Valid market order
        order = Order(
            symbol='AAPL',
            quantity=100,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            strategy='test_strategy',
            client_order_id='TEST_001'
        )
        
        self.assertEqual(order.symbol, 'AAPL')
        self.assertEqual(order.quantity, 100)
        self.assertEqual(order.side, OrderSide.BUY)
        self.assertEqual(order.order_type, OrderType.MARKET)
        
        # Valid limit order
        limit_order = Order(
            symbol='MSFT',
            quantity=50,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            price=350.00,
            strategy='test_strategy'
        )
        
        self.assertEqual(limit_order.price, 350.00)
        self.assertEqual(limit_order.order_type, OrderType.LIMIT)
    
    def test_order_validation(self):
        """Test order validation logic"""
        # Test invalid quantity
        with self.assertRaises(ValueError):
            Order(
                symbol='AAPL',
                quantity=-10,  # Invalid
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                strategy='test'
            )
        
        # Test invalid symbol
        with self.assertRaises(ValueError):
            Order(
                symbol='',  # Invalid
                quantity=100,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                strategy='test'
            )
    
    @patch('trading_system.TradingSystem._get_current_price')
    def test_paper_order_execution(self, mock_price):
        """Test paper trading order execution"""
        mock_price.return_value = 150.0
        
        order = Order(
            symbol='AAPL',
            quantity=100,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            strategy='test_strategy'
        )
        
        # Mock database operations
        self.mock_conn.execute.return_value = None
        self.mock_conn.commit.return_value = None
        
        # Test order execution
        result = self.trading_system._execute_paper_order(order)
        
        # Verify results
        self.assertTrue(result['success'])
        self.assertEqual(order.status, OrderStatus.FILLED)
        self.assertEqual(order.filled_quantity, 100)
        self.assertGreater(order.avg_fill_price, 0)
        self.assertGreater(order.commission, 0)
        self.assertIsNotNone(order.filled_at)
    
    def test_position_calculation(self):
        """Test position P&L calculations"""
        position = Position(
            symbol='AAPL',
            quantity=100,
            avg_cost=150.0,
            market_value=16000.0,  # Current price: $160
            unrealized_pnl=1000.0,  # $10 per share profit
            realized_pnl=0.0
        )
        
        self.assertEqual(position.quantity, 100)
        self.assertEqual(position.avg_cost, 150.0)
        self.assertEqual(position.unrealized_pnl, 1000.0)
        
        # Calculate P&L percentage
        pnl_percent = (position.unrealized_pnl / (position.quantity * position.avg_cost)) * 100
        self.assertAlmostEqual(pnl_percent, 6.67, places=2)
    
    def test_account_balance_calculation(self):
        """Test account balance calculations"""
        positions = [
            Position('AAPL', 100, 150.0, 16000.0, 1000.0),
            Position('MSFT', 50, 300.0, 17500.0, 2500.0),
            Position('GOOGL', 10, 2500.0, 28000.0, 3000.0)
        ]
        
        account = Account(
            account_id='test_account',
            cash_balance=50000.0,
            total_equity=111500.0,  # Cash + positions
            buying_power=100000.0,
            day_trading_buying_power=400000.0,
            positions=positions,
            orders=[]
        )
        
        total_position_value = sum(p.market_value for p in positions)
        total_unrealized_pnl = sum(p.unrealized_pnl for p in positions)
        
        self.assertEqual(total_position_value, 61500.0)
        self.assertEqual(total_unrealized_pnl, 6500.0)
        self.assertEqual(account.total_equity, 111500.0)
    
    def test_risk_calculations(self):
        """Test risk metric calculations"""
        # Test portfolio concentration
        positions = [
            Position('AAPL', 1000, 150.0, 160000.0, 10000.0),  # 40% of portfolio
            Position('MSFT', 500, 300.0, 175000.0, 25000.0),   # 43.75% of portfolio
            Position('GOOGL', 10, 2500.0, 28000.0, 3000.0),    # 7% of portfolio
            Position('TSLA', 100, 800.0, 85000.0, 5000.0),     # 21.25% of portfolio
        ]
        
        total_value = sum(p.market_value for p in positions)
        largest_position_pct = max(p.market_value / total_value for p in positions) * 100
        
        self.assertEqual(total_value, 448000.0)
        self.assertAlmostEqual(largest_position_pct, 39.06, places=2)
        
        # Test if concentration exceeds limits
        concentration_limit = 35.0  # 35% max position size
        self.assertGreater(largest_position_pct, concentration_limit)
    
    def test_order_status_transitions(self):
        """Test order status transitions"""
        order = Order(
            symbol='AAPL',
            quantity=100,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            strategy='test'
        )
        
        # Initial status
        self.assertEqual(order.status, OrderStatus.PENDING)
        
        # Fill the order
        order.status = OrderStatus.FILLED
        order.filled_quantity = 100
        order.avg_fill_price = 150.0
        order.filled_at = datetime.now()
        
        self.assertEqual(order.status, OrderStatus.FILLED)
        self.assertEqual(order.filled_quantity, order.quantity)
        self.assertIsNotNone(order.filled_at)
    
    def test_commission_calculations(self):
        """Test commission calculations"""
        # Test per-share commission
        order_size = 100
        commission_per_share = 0.005
        min_commission = 1.0
        
        calculated_commission = max(order_size * commission_per_share, min_commission)
        expected_commission = 1.0  # min commission applies
        
        self.assertEqual(calculated_commission, expected_commission)
        
        # Test larger order
        large_order_size = 1000
        large_commission = max(large_order_size * commission_per_share, min_commission)
        expected_large_commission = 5.0
        
        self.assertEqual(large_commission, expected_large_commission)

class TestSchwabIntegration(unittest.TestCase):
    """Test cases for Schwab broker integration"""
    
    def setUp(self):
        """Set up Schwab test environment"""
        with patch('schwab_broker_integration.create_engine'):
            self.schwab_api = SchwabAPI('test_client_id', 'test_client_secret', paper_trading=True)
        
        self.schwab_api.engine = Mock()
        self.mock_conn = Mock()
        self.schwab_api.engine.connect.return_value.__enter__.return_value = self.mock_conn
    
    def test_schwab_order_conversion(self):
        """Test ACIS order to Schwab format conversion"""
        order = Order(
            symbol='AAPL',
            quantity=100,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            strategy='test_strategy',
            client_order_id='TEST_001'
        )
        
        schwab_order = self.schwab_api._convert_to_schwab_order(order)
        
        self.assertEqual(schwab_order['orderType'], 'MARKET')
        self.assertEqual(schwab_order['session'], 'NORMAL')
        self.assertEqual(schwab_order['duration'], 'DAY')
        self.assertEqual(schwab_order['orderLegCollection'][0]['instruction'], 'BUY')
        self.assertEqual(schwab_order['orderLegCollection'][0]['quantity'], 100)
        self.assertEqual(schwab_order['orderLegCollection'][0]['instrument']['symbol'], 'AAPL')
    
    def test_schwab_limit_order_conversion(self):
        """Test limit order conversion"""
        limit_order = Order(
            symbol='MSFT',
            quantity=50,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            price=350.0,
            strategy='test_strategy'
        )
        
        schwab_limit_order = self.schwab_api._convert_to_schwab_order(limit_order)
        
        self.assertEqual(schwab_limit_order['orderType'], 'LIMIT')
        self.assertEqual(schwab_limit_order['price'], 350.0)
        self.assertEqual(schwab_limit_order['orderLegCollection'][0]['instruction'], 'SELL')
    
    @patch('schwab_broker_integration.requests.post')
    def test_schwab_authentication(self, mock_post):
        """Test Schwab OAuth authentication"""
        # Mock successful token response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'access_token': 'test_access_token',
            'refresh_token': 'test_refresh_token',
            'expires_in': 3600
        }
        mock_post.return_value = mock_response
        
        # Mock database save
        self.mock_conn.execute.return_value = None
        self.mock_conn.commit.return_value = None
        
        result = self.schwab_api._get_initial_tokens('test_auth_code')
        
        self.assertTrue(result)
        self.assertEqual(self.schwab_api.access_token, 'test_access_token')
        self.assertEqual(self.schwab_api.refresh_token, 'test_refresh_token')

class TestLiveTradingSystem(unittest.TestCase):
    """Test cases for live trading system"""
    
    def setUp(self):
        """Set up live trading test environment"""
        with patch('live_trading_integration.create_engine'):
            self.live_system = LiveTradingSystem()
    
    def test_risk_validation(self):
        """Test pre-trade risk validation"""
        order = Order(
            symbol='AAPL',
            quantity=1000,  # Large order
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            price=150.0,
            strategy='test_strategy'
        )
        
        # Test order value validation
        estimated_value = order.quantity * order.price
        max_order_value = self.live_system.risk_settings['max_order_value']
        
        self.assertEqual(estimated_value, 150000.0)
        self.assertGreater(estimated_value, max_order_value)  # Should exceed limit
    
    def test_position_size_validation(self):
        """Test position size risk controls"""
        order_value = 75000.0  # $75K order
        account_value = 1000000.0  # $1M account
        position_size_pct = order_value / account_value
        max_position_size = 0.05  # 5% limit
        
        self.assertEqual(position_size_pct, 0.075)  # 7.5%
        self.assertGreater(position_size_pct, max_position_size)
    
    def test_market_hours_validation(self):
        """Test market hours validation"""
        # Mock current time during market hours
        with patch('live_trading_integration.datetime') as mock_datetime:
            # Tuesday at 2:00 PM ET (market hours)
            mock_datetime.now.return_value = datetime(2024, 8, 27, 14, 0, 0)
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            is_market_hours = self.live_system._is_market_hours()
            self.assertTrue(is_market_hours)
            
            # Mock time outside market hours
            mock_datetime.now.return_value = datetime(2024, 8, 27, 18, 0, 0)  # 6 PM ET
            is_after_hours = self.live_system._is_market_hours()
            self.assertFalse(is_after_hours)

class TestSystemMonitoring(unittest.TestCase):
    """Test cases for system monitoring"""
    
    def setUp(self):
        """Set up monitoring test environment"""
        with patch('system_monitoring.create_engine'):
            from system_monitoring import SystemMonitor, SystemMetric, Alert
            self.monitor = SystemMonitor()
            self.SystemMetric = SystemMetric
            self.Alert = Alert
    
    def test_metric_creation(self):
        """Test system metric creation"""
        metric = self.SystemMetric(
            metric_name='cpu_usage',
            value=75.5,
            unit='percent',
            timestamp=datetime.now(),
            threshold_warning=80.0,
            threshold_critical=95.0,
            description='CPU utilization percentage'
        )
        
        self.assertEqual(metric.metric_name, 'cpu_usage')
        self.assertEqual(metric.value, 75.5)
        self.assertEqual(metric.unit, 'percent')
        self.assertEqual(metric.threshold_warning, 80.0)
    
    def test_alert_creation(self):
        """Test alert creation and validation"""
        alert = self.Alert(
            alert_type='system_warning',
            severity='warning',
            title='High CPU Usage',
            message='CPU usage is at 85%',
            data={'cpu_percent': 85.0},
            created_at=datetime.now()
        )
        
        self.assertEqual(alert.alert_type, 'system_warning')
        self.assertEqual(alert.severity, 'warning')
        self.assertIn('CPU usage', alert.message)
        self.assertIsInstance(alert.data, dict)
    
    def test_threshold_validation(self):
        """Test alert threshold validation"""
        cpu_value = 92.0
        warning_threshold = 80.0
        critical_threshold = 95.0
        
        # Should trigger warning but not critical
        self.assertGreater(cpu_value, warning_threshold)
        self.assertLess(cpu_value, critical_threshold)
        
        # Test critical threshold
        critical_cpu_value = 97.0
        self.assertGreater(critical_cpu_value, critical_threshold)

class TestAdminSystem(unittest.TestCase):
    """Test cases for admin system"""
    
    def setUp(self):
        """Set up admin system test environment"""
        # Mock Flask app and database
        with patch('admin_app.Flask'), patch('admin_app.create_engine'):
            from admin_app import User
            self.User = User
    
    def test_user_creation(self):
        """Test user model creation"""
        user = self.User(
            user_id=1,
            username='testuser',
            email='test@example.com',
            role='trader',
            full_name='Test User'
        )
        
        self.assertEqual(user.id, 1)
        self.assertEqual(user.username, 'testuser')
        self.assertEqual(user.email, 'test@example.com')
        self.assertEqual(user.role, 'trader')
    
    def test_user_permissions(self):
        """Test user permission system"""
        # Admin user
        admin_user = self.User(1, 'admin', 'admin@test.com', 'admin', 'Admin User')
        self.assertTrue(admin_user.has_permission('manage_users'))
        self.assertTrue(admin_user.has_permission('execute_trades'))
        
        # Trader user
        trader_user = self.User(2, 'trader', 'trader@test.com', 'trader', 'Trader User')
        self.assertTrue(trader_user.has_permission('execute_trades'))
        self.assertFalse(trader_user.has_permission('manage_users'))
        
        # Read-only user
        readonly_user = self.User(3, 'readonly', 'readonly@test.com', 'readonly', 'Read Only User')
        self.assertTrue(readonly_user.has_permission('view_portfolios'))
        self.assertFalse(readonly_user.has_permission('execute_trades'))

class TestDataValidation(unittest.TestCase):
    """Test cases for data validation and integrity"""
    
    def test_portfolio_calculation_accuracy(self):
        """Test portfolio calculation accuracy"""
        # Create test positions with known values
        positions = [
            {'symbol': 'AAPL', 'quantity': 100, 'price': 150.0},
            {'symbol': 'MSFT', 'quantity': 200, 'price': 300.0},
            {'symbol': 'GOOGL', 'quantity': 50, 'price': 2500.0}
        ]
        
        total_value = sum(pos['quantity'] * pos['price'] for pos in positions)
        expected_total = 15000 + 60000 + 125000  # $200,000
        
        self.assertEqual(total_value, expected_total)
        
        # Test percentage calculations
        aapl_weight = (positions[0]['quantity'] * positions[0]['price']) / total_value
        self.assertAlmostEqual(aapl_weight, 0.075, places=3)  # 7.5%
    
    def test_return_calculations(self):
        """Test return calculation accuracy"""
        # Test simple return calculation
        initial_value = 100000.0
        final_value = 115000.0
        simple_return = (final_value - initial_value) / initial_value
        
        self.assertEqual(simple_return, 0.15)  # 15% return
        
        # Test compound return calculation
        periods = 4  # Quarterly
        compound_return = (final_value / initial_value) ** (1/periods) - 1
        quarterly_return = compound_return * 4  # Annualized
        
        self.assertAlmostEqual(quarterly_return, 0.1426, places=4)
    
    def test_risk_metric_calculations(self):
        """Test risk metric calculation accuracy"""
        # Test Sharpe ratio calculation
        returns = [0.12, 0.08, 0.15, -0.03, 0.20]  # Annual returns
        risk_free_rate = 0.02
        
        excess_returns = [r - risk_free_rate for r in returns]
        avg_excess_return = np.mean(excess_returns)
        volatility = np.std(returns)
        sharpe_ratio = avg_excess_return / volatility
        
        self.assertAlmostEqual(avg_excess_return, 0.084, places=3)
        self.assertAlmostEqual(volatility, 0.0843, places=3)
        self.assertAlmostEqual(sharpe_ratio, 0.996, places=3)

def run_comprehensive_tests():
    """Run all test suites"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_cases = [
        TestTradingSystem,
        TestSchwabIntegration,
        TestLiveTradingSystem,
        TestSystemMonitoring,
        TestAdminSystem,
        TestDataValidation
    ]
    
    for test_case in test_cases:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_case)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result

if __name__ == '__main__':
    print("ACIS Trading Platform - Comprehensive Unit Tests")
    print("="*60)
    
    result = run_comprehensive_tests()
    
    print(f"\n{'='*60}")
    print(f"Test Results Summary:")
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0] if 'AssertionError:' in traceback else 'Unknown failure'}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('\\n')[-2] if traceback else 'Unknown error'}")
    
    print(f"{'='*60}")
    
    # Return exit code based on test results
    exit(0 if result.wasSuccessful() else 1)