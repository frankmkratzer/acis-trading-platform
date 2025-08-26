#!/usr/bin/env python3
"""
Unit tests for ACIS Strategy Performance
Tests individual strategy logic, portfolio construction, and performance calculations
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

class TestStrategyLogic(unittest.TestCase):
    """Test cases for individual strategy logic"""
    
    def setUp(self):
        """Set up test environment"""
        # Create sample stock data for testing
        self.sample_data = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'] * 10,
            'market_cap': [3000000, 2800000, 1800000, 1600000, 800000] * 10,
            'price': [150.0, 350.0, 2800.0, 3200.0, 800.0] * 10,
            'pe_ratio': [25.0, 28.0, 22.0, 45.0, 80.0] * 10,
            'pb_ratio': [7.5, 12.0, 5.5, 8.0, 15.0] * 10,
            'dividend_yield': [0.5, 0.7, 0.0, 0.0, 0.0] * 10,
            'roe': [0.28, 0.45, 0.18, 0.23, 0.12] * 10,
            'revenue_growth': [0.08, 0.12, 0.15, 0.20, 0.45] * 10,
            'earnings_growth': [0.05, 0.18, 0.12, 0.35, -0.05] * 10,
            'debt_to_equity': [1.5, 0.4, 0.1, 0.8, 0.6] * 10,
            'current_ratio': [1.2, 2.5, 3.0, 1.8, 1.4] * 10,
            'volume': [50000000, 25000000, 1500000, 3000000, 40000000] * 10,
            'price_momentum_3m': [0.08, 0.12, -0.02, 0.15, 0.25] * 10,
            'price_momentum_6m': [0.15, 0.20, 0.05, 0.25, 0.45] * 10,
            'earnings_surprise': [0.02, 0.08, 0.15, 0.03, -0.12] * 10,
            'analyst_rating': [4.2, 4.5, 4.0, 4.3, 3.8] * 10,
            'sector': ['Technology', 'Technology', 'Technology', 'Consumer Discretionary', 'Consumer Discretionary'] * 10
        })
    
    def test_value_strategy_scoring(self):
        """Test value strategy stock scoring logic"""
        # Value strategy should favor low PE, low PB, high dividend yield
        value_scores = []
        
        for _, stock in self.sample_data.head(5).iterrows():
            # Simple value scoring (lower is better for PE/PB, higher for dividend)
            pe_score = max(0, 50 - stock['pe_ratio']) / 50  # Normalize PE
            pb_score = max(0, 20 - stock['pb_ratio']) / 20   # Normalize PB
            div_score = min(stock['dividend_yield'] / 5, 1)   # Normalize dividend yield
            roe_score = min(stock['roe'] / 0.5, 1)           # Normalize ROE
            
            total_score = (pe_score * 0.3 + pb_score * 0.3 + div_score * 0.2 + roe_score * 0.2) * 100
            value_scores.append(total_score)
        
        # AAPL should score well (reasonable PE, good ROE, dividend)
        # TSLA should score poorly (high PE, no dividend)
        aapl_score = value_scores[0]
        tsla_score = value_scores[4]
        
        self.assertGreater(aapl_score, tsla_score)
        self.assertGreater(aapl_score, 30)  # Should be a decent value score
        self.assertLess(tsla_score, 30)     # Should be a poor value score
    
    def test_growth_strategy_scoring(self):
        """Test growth strategy stock scoring logic"""
        growth_scores = []
        
        for _, stock in self.sample_data.head(5).iterrows():
            # Growth strategy should favor high revenue growth, earnings growth
            revenue_score = min(stock['revenue_growth'] / 0.5, 1)     # Normalize revenue growth
            earnings_score = max(0, min(stock['earnings_growth'] / 0.4, 1))  # Normalize earnings growth
            momentum_score = max(0, min(stock['price_momentum_6m'] / 0.5, 1))  # Price momentum
            analyst_score = stock['analyst_rating'] / 5               # Analyst rating
            
            total_score = (revenue_score * 0.3 + earnings_score * 0.3 + momentum_score * 0.2 + analyst_score * 0.2) * 100
            growth_scores.append(total_score)
        
        # TSLA should score well (high growth rates)
        # AAPL should score moderately (steady growth)
        tsla_score = growth_scores[4]
        aapl_score = growth_scores[0]
        
        self.assertGreater(tsla_score, aapl_score)
        self.assertGreater(tsla_score, 60)  # Should be a good growth score
    
    def test_momentum_strategy_scoring(self):
        """Test momentum strategy stock scoring logic"""
        momentum_scores = []
        
        for _, stock in self.sample_data.head(5).iterrows():
            # Momentum strategy should favor positive price momentum, earnings surprise
            momentum_3m = max(0, min(stock['price_momentum_3m'] / 0.3, 1))
            momentum_6m = max(0, min(stock['price_momentum_6m'] / 0.5, 1))
            earnings_surprise = max(0, min(stock['earnings_surprise'] / 0.2, 1))
            volume_score = min(stock['volume'] / 100000000, 1)  # Liquidity
            
            total_score = (momentum_3m * 0.3 + momentum_6m * 0.3 + earnings_surprise * 0.2 + volume_score * 0.2) * 100
            momentum_scores.append(total_score)
        
        # Stocks with positive momentum should score higher
        positive_momentum_stocks = [i for i, score in enumerate(momentum_scores) if 
                                   self.sample_data.iloc[i]['price_momentum_6m'] > 0]
        
        self.assertGreater(len(positive_momentum_stocks), 0)
        
        # Check that positive momentum stocks generally score higher
        avg_positive_score = np.mean([momentum_scores[i] for i in positive_momentum_stocks])
        self.assertGreater(avg_positive_score, 30)
    
    def test_dividend_strategy_scoring(self):
        """Test dividend strategy stock scoring logic"""
        dividend_scores = []
        
        for _, stock in self.sample_data.head(5).iterrows():
            # Dividend strategy should favor high dividend yield, stable earnings, low debt
            div_yield_score = min(stock['dividend_yield'] / 5, 1)
            stability_score = max(0, min(stock['roe'] / 0.3, 1))  # ROE for stability
            debt_score = max(0, (3 - stock['debt_to_equity']) / 3)  # Lower debt is better
            current_ratio_score = min(stock['current_ratio'] / 3, 1)  # Liquidity
            
            # Only consider stocks with dividends
            if stock['dividend_yield'] > 0:
                total_score = (div_yield_score * 0.4 + stability_score * 0.25 + 
                              debt_score * 0.2 + current_ratio_score * 0.15) * 100
            else:
                total_score = 0  # No dividend = no score
            
            dividend_scores.append(total_score)
        
        # AAPL and MSFT should score (they have dividends)
        # GOOGL, AMZN, TSLA should not score (no dividends)
        self.assertGreater(dividend_scores[0], 0)  # AAPL
        self.assertGreater(dividend_scores[1], 0)  # MSFT
        self.assertEqual(dividend_scores[2], 0)     # GOOGL
        self.assertEqual(dividend_scores[3], 0)     # AMZN
        self.assertEqual(dividend_scores[4], 0)     # TSLA

class TestPortfolioConstruction(unittest.TestCase):
    """Test portfolio construction and optimization"""
    
    def setUp(self):
        """Set up portfolio test environment"""
        # Create scored stocks for portfolio construction
        self.scored_stocks = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'CRM', 'ADBE'],
            'score': [85.5, 82.3, 79.8, 77.2, 74.6, 72.1, 69.8, 67.5, 65.2, 62.9],
            'market_cap': [3000000, 2800000, 1800000, 1600000, 800000, 700000, 1100000, 200000, 250000, 280000],
            'sector': ['Technology', 'Technology', 'Technology', 'Consumer Discretionary', 
                      'Consumer Discretionary', 'Technology', 'Technology', 'Technology', 
                      'Technology', 'Technology'],
            'price': [150.0, 350.0, 2800.0, 3200.0, 800.0, 300.0, 450.0, 400.0, 220.0, 580.0]
        })
        
        self.portfolio_value = 1000000  # $1M portfolio
    
    def test_basic_portfolio_construction(self):
        """Test basic portfolio construction logic"""
        # Select top N stocks
        top_stocks = self.scored_stocks.head(5)
        
        # Equal weight allocation
        equal_weight = 1.0 / len(top_stocks)
        allocations = [equal_weight] * len(top_stocks)
        
        # Calculate dollar amounts
        dollar_amounts = [allocation * self.portfolio_value for allocation in allocations]
        
        # Verify allocations sum to 100%
        self.assertAlmostEqual(sum(allocations), 1.0, places=6)
        
        # Verify dollar amounts sum to portfolio value
        self.assertAlmostEqual(sum(dollar_amounts), self.portfolio_value, places=2)
        
        # Each position should be 20% of portfolio
        for amount in dollar_amounts:
            self.assertAlmostEqual(amount / self.portfolio_value, 0.2, places=6)
    
    def test_sector_concentration_limits(self):
        """Test sector concentration limits in portfolio construction"""
        max_sector_allocation = 0.30  # 30% max per sector
        
        # Count technology stocks in top selections
        top_10 = self.scored_stocks.head(10)
        tech_stocks = top_10[top_10['sector'] == 'Technology']
        tech_count = len(tech_stocks)
        
        # If we select equal weights, would tech exceed limit?
        equal_weight = 1.0 / 10
        tech_weight = tech_count * equal_weight
        
        if tech_weight > max_sector_allocation:
            # Need sector limits
            max_tech_positions = int(max_sector_allocation / equal_weight)
            self.assertLessEqual(max_tech_positions * equal_weight, max_sector_allocation)
    
    def test_position_size_limits(self):
        """Test individual position size limits"""
        max_position_size = 0.05  # 5% max per position
        num_positions = 30
        
        # With 30 positions, equal weight would be 3.33%
        equal_weight = 1.0 / num_positions
        self.assertLess(equal_weight, max_position_size)
        
        # With 15 positions, equal weight would be 6.67% (exceeds limit)
        large_position_weight = 1.0 / 15
        self.assertGreater(large_position_weight, max_position_size)
    
    def test_market_cap_filtering(self):
        """Test market cap filtering for different strategies"""
        # Small cap: < $2B
        small_cap_threshold = 2000000  # $2B in thousands
        small_cap_stocks = self.scored_stocks[self.scored_stocks['market_cap'] < small_cap_threshold]
        
        # Mid cap: $2B - $10B
        mid_cap_stocks = self.scored_stocks[
            (self.scored_stocks['market_cap'] >= 2000000) & 
            (self.scored_stocks['market_cap'] < 10000000)
        ]
        
        # Large cap: > $10B
        large_cap_stocks = self.scored_stocks[self.scored_stocks['market_cap'] >= 10000000]
        
        # Verify filtering works
        self.assertGreater(len(small_cap_stocks), 0)
        self.assertGreater(len(mid_cap_stocks), 0)
        
        # Verify no overlap
        total_stocks = len(small_cap_stocks) + len(mid_cap_stocks) + len(large_cap_stocks)
        self.assertEqual(total_stocks, len(self.scored_stocks))
    
    def test_share_calculation(self):
        """Test share quantity calculations"""
        allocation_amount = 50000  # $50K allocation
        stock_price = 150.0
        
        # Calculate shares (round down to avoid over-allocation)
        shares = int(allocation_amount / stock_price)
        actual_amount = shares * stock_price
        
        self.assertEqual(shares, 333)  # $49,950 worth
        self.assertLess(actual_amount, allocation_amount)
        self.assertGreaterEqual(allocation_amount - actual_amount, 0)  # Positive cash remainder
    
    def test_portfolio_rebalancing(self):
        """Test portfolio rebalancing logic"""
        # Current portfolio (drifted from targets)
        current_positions = {
            'AAPL': {'shares': 400, 'price': 160.0, 'value': 64000, 'target_weight': 0.20},
            'MSFT': {'shares': 120, 'price': 380.0, 'value': 45600, 'target_weight': 0.20},
            'GOOGL': {'shares': 18, 'price': 2900.0, 'value': 52200, 'target_weight': 0.20}
        }
        
        total_value = sum(pos['value'] for pos in current_positions.values())
        target_value_per_position = total_value / 3  # Equal weight targets
        
        # Calculate rebalancing needs
        rebalance_actions = {}
        for symbol, position in current_positions.items():
            current_value = position['value']
            target_value = target_value_per_position
            difference = target_value - current_value
            
            if abs(difference) > 1000:  # Only rebalance if difference > $1K
                rebalance_actions[symbol] = {
                    'action': 'buy' if difference > 0 else 'sell',
                    'amount': abs(difference),
                    'shares': int(abs(difference) / position['price'])
                }
        
        # AAPL is overweight, MSFT is underweight
        self.assertEqual(rebalance_actions['AAPL']['action'], 'sell')
        self.assertEqual(rebalance_actions['MSFT']['action'], 'buy')

class TestPerformanceCalculations(unittest.TestCase):
    """Test performance calculation accuracy"""
    
    def setUp(self):
        """Set up performance test data"""
        # Create sample price history
        dates = pd.date_range(start='2020-01-01', end='2024-08-26', freq='D')
        np.random.seed(42)  # For reproducible results
        
        # Generate realistic return series
        daily_returns = np.random.normal(0.0008, 0.015, len(dates))  # ~20% annual return, 15% volatility
        price_series = 100 * (1 + pd.Series(daily_returns, index=dates)).cumprod()
        
        self.price_data = pd.DataFrame({
            'date': dates,
            'price': price_series,
            'returns': daily_returns
        })
    
    def test_total_return_calculation(self):
        """Test total return calculation"""
        initial_price = self.price_data['price'].iloc[0]
        final_price = self.price_data['price'].iloc[-1]
        
        total_return = (final_price - initial_price) / initial_price
        expected_return = (final_price / initial_price) - 1
        
        self.assertAlmostEqual(total_return, expected_return, places=6)
        
        # Verify using cumulative returns
        cumulative_return = (1 + self.price_data['returns']).prod() - 1
        self.assertAlmostEqual(total_return, cumulative_return, places=3)
    
    def test_annualized_return_calculation(self):
        """Test annualized return calculation"""
        initial_price = self.price_data['price'].iloc[0]
        final_price = self.price_data['price'].iloc[-1]
        days_held = len(self.price_data)
        years_held = days_held / 252  # Trading days per year
        
        total_return = (final_price / initial_price) - 1
        annualized_return = (1 + total_return) ** (1 / years_held) - 1
        
        # Should be positive (upward trending test data)
        self.assertGreater(annualized_return, 0)
        
        # Should be reasonable (not too high)
        self.assertLess(annualized_return, 1.0)  # Less than 100% annual return
    
    def test_volatility_calculation(self):
        """Test volatility calculation"""
        daily_returns = self.price_data['returns'].dropna()
        daily_volatility = daily_returns.std()
        annualized_volatility = daily_volatility * np.sqrt(252)
        
        # Should be positive
        self.assertGreater(annualized_volatility, 0)
        
        # Should be reasonable for stock returns
        self.assertGreater(annualized_volatility, 0.05)  # > 5%
        self.assertLess(annualized_volatility, 0.50)     # < 50%
    
    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation"""
        daily_returns = self.price_data['returns'].dropna()
        annual_return = daily_returns.mean() * 252
        annual_volatility = daily_returns.std() * np.sqrt(252)
        risk_free_rate = 0.02  # 2%
        
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
        
        # Should be a reasonable value
        self.assertGreater(sharpe_ratio, -2.0)
        self.assertLess(sharpe_ratio, 5.0)
    
    def test_maximum_drawdown_calculation(self):
        """Test maximum drawdown calculation"""
        prices = self.price_data['price']
        
        # Calculate running maximum
        running_max = prices.expanding().max()
        
        # Calculate drawdown
        drawdown = (prices - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Max drawdown should be negative or zero
        self.assertLessEqual(max_drawdown, 0)
        
        # Should be a reasonable value (not too extreme)
        self.assertGreater(max_drawdown, -0.50)  # Not more than 50% drawdown
    
    def test_benchmark_comparison(self):
        """Test benchmark comparison calculations"""
        # Create benchmark data (slightly lower returns)
        np.random.seed(43)
        benchmark_returns = np.random.normal(0.0006, 0.012, len(self.price_data))
        benchmark_prices = 100 * (1 + pd.Series(benchmark_returns)).cumprod()
        
        # Calculate alpha (excess return)
        portfolio_annual_return = self.price_data['returns'].mean() * 252
        benchmark_annual_return = pd.Series(benchmark_returns).mean() * 252
        alpha = portfolio_annual_return - benchmark_annual_return
        
        # Calculate beta (sensitivity to benchmark)
        covariance = np.cov(self.price_data['returns'].dropna(), benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        beta = covariance / benchmark_variance
        
        # Beta should be positive (both should move in same direction generally)
        self.assertGreater(beta, 0)
        
        # Beta should be reasonable
        self.assertLess(beta, 3.0)
        self.assertGreater(beta, 0.1)

class TestRiskManagement(unittest.TestCase):
    """Test risk management calculations and controls"""
    
    def setUp(self):
        """Set up risk management test data"""
        self.portfolio_positions = [
            {'symbol': 'AAPL', 'value': 80000, 'beta': 1.2, 'sector': 'Technology'},
            {'symbol': 'MSFT', 'value': 70000, 'beta': 0.9, 'sector': 'Technology'},
            {'symbol': 'GOOGL', 'value': 60000, 'beta': 1.1, 'sector': 'Technology'},
            {'symbol': 'JNJ', 'value': 50000, 'beta': 0.7, 'sector': 'Healthcare'},
            {'symbol': 'PG', 'value': 40000, 'beta': 0.6, 'sector': 'Consumer Staples'}
        ]
        self.total_portfolio_value = sum(pos['value'] for pos in self.portfolio_positions)
    
    def test_position_concentration_risk(self):
        """Test position concentration risk calculations"""
        # Calculate position weights
        position_weights = [pos['value'] / self.total_portfolio_value for pos in self.portfolio_positions]
        max_position_weight = max(position_weights)
        
        # Largest position should be AAPL at ~26.7%
        self.assertAlmostEqual(max_position_weight, 80000 / 300000, places=3)
        
        # Check concentration limits
        concentration_limit = 0.30  # 30%
        self.assertLess(max_position_weight, concentration_limit)
    
    def test_sector_concentration_risk(self):
        """Test sector concentration risk"""
        # Calculate sector exposures
        sector_values = {}
        for pos in self.portfolio_positions:
            sector = pos['sector']
            if sector not in sector_values:
                sector_values[sector] = 0
            sector_values[sector] += pos['value']
        
        sector_weights = {sector: value / self.total_portfolio_value 
                         for sector, value in sector_values.items()}
        
        max_sector_weight = max(sector_weights.values())
        
        # Technology should be the largest sector
        tech_weight = sector_weights['Technology']
        self.assertEqual(max_sector_weight, tech_weight)
        
        # Check if it exceeds limits
        sector_limit = 0.40  # 40%
        if tech_weight > sector_limit:
            self.fail(f"Technology sector weight {tech_weight:.1%} exceeds limit {sector_limit:.1%}")
    
    def test_portfolio_beta_calculation(self):
        """Test portfolio beta calculation"""
        # Calculate weighted beta
        total_weight = 0
        weighted_beta = 0
        
        for pos in self.portfolio_positions:
            weight = pos['value'] / self.total_portfolio_value
            weighted_beta += weight * pos['beta']
            total_weight += weight
        
        portfolio_beta = weighted_beta / total_weight
        
        # Portfolio beta should be reasonable
        self.assertGreater(portfolio_beta, 0.5)
        self.assertLess(portfolio_beta, 1.5)
        
        # Should be weighted average of individual betas
        self.assertAlmostEqual(total_weight, 1.0, places=6)
    
    def test_var_calculation(self):
        """Test Value at Risk (VaR) calculation"""
        # Assume portfolio has 15% annual volatility
        annual_volatility = 0.15
        daily_volatility = annual_volatility / np.sqrt(252)
        
        # 95% VaR (1.645 standard deviations)
        var_95 = 1.645 * daily_volatility * self.total_portfolio_value
        
        # 99% VaR (2.33 standard deviations)
        var_99 = 2.33 * daily_volatility * self.total_portfolio_value
        
        # VaR should be positive and reasonable
        self.assertGreater(var_95, 0)
        self.assertGreater(var_99, var_95)
        
        # Should be reasonable percentage of portfolio
        var_95_pct = var_95 / self.total_portfolio_value
        self.assertLess(var_95_pct, 0.05)  # Less than 5% daily VaR
    
    def test_correlation_risk(self):
        """Test correlation risk between positions"""
        # Simulate correlation matrix (Technology stocks highly correlated)
        correlation_matrix = {
            ('AAPL', 'MSFT'): 0.75,
            ('AAPL', 'GOOGL'): 0.70,
            ('MSFT', 'GOOGL'): 0.80,
            ('JNJ', 'PG'): 0.45,  # Healthcare and consumer staples less correlated
            ('AAPL', 'JNJ'): 0.35,
            ('MSFT', 'PG'): 0.30
        }
        
        # Check for high correlations within sectors
        tech_stocks = ['AAPL', 'MSFT', 'GOOGL']
        high_correlations = 0
        
        for i, stock1 in enumerate(tech_stocks):
            for stock2 in tech_stocks[i+1:]:
                correlation = correlation_matrix.get((stock1, stock2), 0)
                if correlation > 0.70:
                    high_correlations += 1
        
        # Should have some high correlations within tech
        self.assertGreater(high_correlations, 0)
        
        # But not too many (would indicate over-concentration)
        total_pairs = len(tech_stocks) * (len(tech_stocks) - 1) // 2
        correlation_ratio = high_correlations / total_pairs
        self.assertLess(correlation_ratio, 1.0)

def run_strategy_tests():
    """Run all strategy performance tests"""
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_cases = [
        TestStrategyLogic,
        TestPortfolioConstruction,
        TestPerformanceCalculations,
        TestRiskManagement
    ]
    
    for test_case in test_cases:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_case)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result

if __name__ == '__main__':
    print("ACIS Trading Platform - Strategy Performance Tests")
    print("="*60)
    
    result = run_strategy_tests()
    
    print(f"\n{'='*60}")
    print(f"Strategy Test Results:")
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.wasSuccessful():
        print("✅ All strategy tests passed!")
    else:
        print("❌ Some strategy tests failed - review before deployment")
    
    exit(0 if result.wasSuccessful() else 1)