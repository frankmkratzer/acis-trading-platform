#!/usr/bin/env python3
"""
ACIS Trading Platform - Optimized Semi-Annual Rebalancing System
Implements semi-annual rebalancing with enhanced fundamentals and conviction sizing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from sqlalchemy import create_engine, text
import os
from database_config import get_database_connection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedSemiAnnualSystem:
    def __init__(self):
        self.conn = get_database_connection()
        
        # Enhanced rebalancing schedule - Semi-annual (June 30, December 31)
        self.rebalancing_schedule = {
            'frequency': 'semi_annual',
            'dates': ['06-30', '12-31'],
            'lookback_months': 6,
            'momentum_period': 180  # 6 months for momentum calculation
        }
        
        # Enhanced fundamental metrics
        self.enhanced_fundamentals = {
            'quality_metrics': [
                'roe',  # Return on Equity
                'roa',  # Return on Assets  
                'free_cash_flow_yield',
                'gross_margin_stability',
                'operating_margin_trend'
            ],
            'growth_quality': [
                'sustainable_growth_rate',
                'earnings_quality_score',
                'revenue_predictability',
                'cash_conversion_cycle'
            ],
            'value_refinement': [
                'enterprise_value_ebitda',
                'price_book_value',
                'peg_ratio',
                'free_cash_flow_price'
            ],
            'momentum_enhancement': [
                'earnings_revision_momentum',
                'price_momentum_3m',
                'price_momentum_6m',
                'relative_strength_sector'
            ]
        }
        
        # Conviction-based position sizing parameters
        self.conviction_sizing = {
            'enabled': True,
            'base_weight': 0.5,  # Base position size %
            'max_weight': 2.5,   # Maximum position size %
            'min_weight': 0.1,   # Minimum position size %
            'conviction_multiplier': 2.0  # How much conviction affects sizing
        }
        
        logger.info("Optimized Semi-Annual System initialized")
    
    def calculate_enhanced_fundamentals(self, symbol, strategy_type):
        """Calculate enhanced fundamental scores"""
        try:
            # Get comprehensive fundamental data
            query = text("""
                SELECT 
                    f.*,
                    p.close_price,
                    p.market_cap,
                    -- Quality Metrics
                    CASE WHEN f.total_equity > 0 THEN f.net_income / f.total_equity * 100 ELSE 0 END as roe,
                    CASE WHEN f.total_assets > 0 THEN f.net_income / f.total_assets * 100 ELSE 0 END as roa,
                    CASE WHEN p.market_cap > 0 THEN (f.operating_cash_flow - f.capital_expenditures) / p.market_cap * 100 ELSE 0 END as free_cash_flow_yield,
                    
                    -- Growth Quality
                    CASE WHEN f.total_equity > 0 THEN f.retained_earnings / f.total_equity * 100 ELSE 0 END as sustainable_growth_rate,
                    
                    -- Value Refinement
                    CASE WHEN f.ebitda > 0 THEN p.market_cap / f.ebitda ELSE 999 END as enterprise_value_ebitda,
                    CASE WHEN f.book_value_per_share > 0 THEN p.close_price / f.book_value_per_share ELSE 999 END as price_book_value,
                    CASE WHEN f.eps_growth > 0 THEN f.pe_ratio / f.eps_growth ELSE 999 END as peg_ratio
                FROM fundamentals_quarterly f
                JOIN pure_us_stocks p ON f.symbol = p.symbol
                WHERE f.symbol = :symbol 
                AND f.fiscal_date >= CURRENT_DATE - INTERVAL '18 months'
                ORDER BY f.fiscal_date DESC
                LIMIT 3
            """)
            
            result = self.conn.execute(query, {"symbol": symbol}).fetchall()
            
            if not result:
                return {}
            
            latest = result[0]
            
            # Calculate enhanced scores based on strategy type
            if strategy_type == 'value':
                return self._calculate_enhanced_value_score(latest)
            elif strategy_type == 'growth':
                return self._calculate_enhanced_growth_score(latest)
            elif strategy_type == 'momentum':
                return self._calculate_enhanced_momentum_score(latest)
            elif strategy_type == 'dividend':
                return self._calculate_enhanced_dividend_score(latest)
            
            return {}
            
        except Exception as e:
            logger.error(f"Error calculating enhanced fundamentals for {symbol}: {str(e)}")
            return {}
    
    def _calculate_enhanced_value_score(self, data):
        """Calculate enhanced value score with new fundamentals"""
        try:
            scores = {}
            
            # Quality component (40% weight)
            roe_score = min(max((data.roe - 5) / 15 * 100, 0), 100) if data.roe else 50
            roa_score = min(max((data.roa - 2) / 8 * 100, 0), 100) if data.roa else 50
            fcf_score = min(max((data.free_cash_flow_yield - 0) / 10 * 100, 0), 100) if data.free_cash_flow_yield else 50
            
            quality_score = (roe_score * 0.4 + roa_score * 0.3 + fcf_score * 0.3)
            
            # Value component (40% weight)
            pe_score = min(max((25 - data.pe_ratio) / 20 * 100, 0), 100) if data.pe_ratio else 50
            pb_score = min(max((3 - data.price_book_value) / 2.5 * 100, 0), 100) if data.price_book_value else 50
            ev_ebitda_score = min(max((15 - data.enterprise_value_ebitda) / 12 * 100, 0), 100) if data.enterprise_value_ebitda else 50
            
            value_score = (pe_score * 0.4 + pb_score * 0.3 + ev_ebitda_score * 0.3)
            
            # Growth component (20% weight)
            growth_score = min(max((data.eps_growth - (-10)) / 25 * 100, 0), 100) if data.eps_growth else 50
            
            # Combined enhanced score
            enhanced_score = (quality_score * 0.4 + value_score * 0.4 + growth_score * 0.2)
            
            scores['enhanced_score'] = enhanced_score
            scores['quality_score'] = quality_score
            scores['value_score'] = value_score
            scores['conviction_level'] = self._calculate_conviction_level(enhanced_score, quality_score)
            
            return scores
            
        except Exception as e:
            logger.error(f"Error in enhanced value scoring: {str(e)}")
            return {'enhanced_score': 50, 'conviction_level': 1.0}
    
    def _calculate_enhanced_growth_score(self, data):
        """Calculate enhanced growth score with new fundamentals"""
        try:
            scores = {}
            
            # Growth Quality (50% weight)
            eps_growth_score = min(max((data.eps_growth - 0) / 30 * 100, 0), 100) if data.eps_growth else 50
            revenue_growth_score = min(max((data.revenue_growth - 0) / 25 * 100, 0), 100) if data.revenue_growth else 50
            sustainable_growth_score = min(max((data.sustainable_growth_rate - 0) / 20 * 100, 0), 100) if data.sustainable_growth_rate else 50
            
            growth_quality_score = (eps_growth_score * 0.4 + revenue_growth_score * 0.3 + sustainable_growth_score * 0.3)
            
            # Quality Metrics (30% weight)
            roe_score = min(max((data.roe - 15) / 25 * 100, 0), 100) if data.roe else 50
            roa_score = min(max((data.roa - 5) / 15 * 100, 0), 100) if data.roa else 50
            
            quality_score = (roe_score * 0.6 + roa_score * 0.4)
            
            # Reasonable Valuation (20% weight)
            peg_score = min(max((2 - data.peg_ratio) / 1.5 * 100, 0), 100) if data.peg_ratio and data.peg_ratio > 0 else 50
            
            # Combined enhanced score
            enhanced_score = (growth_quality_score * 0.5 + quality_score * 0.3 + peg_score * 0.2)
            
            scores['enhanced_score'] = enhanced_score
            scores['growth_quality_score'] = growth_quality_score
            scores['conviction_level'] = self._calculate_conviction_level(enhanced_score, growth_quality_score)
            
            return scores
            
        except Exception as e:
            logger.error(f"Error in enhanced growth scoring: {str(e)}")
            return {'enhanced_score': 50, 'conviction_level': 1.0}
    
    def _calculate_enhanced_momentum_score(self, data):
        """Calculate enhanced momentum score with new fundamentals"""
        try:
            # Get price momentum data
            momentum_query = text("""
                SELECT 
                    symbol,
                    (SELECT close_price FROM stock_eod_daily WHERE symbol = :symbol 
                     AND trade_date >= CURRENT_DATE - INTERVAL '1 day' ORDER BY trade_date DESC LIMIT 1) as current_price,
                    (SELECT close_price FROM stock_eod_daily WHERE symbol = :symbol 
                     AND trade_date >= CURRENT_DATE - INTERVAL '90 days' ORDER BY trade_date DESC LIMIT 1) as price_3m_ago,
                    (SELECT close_price FROM stock_eod_daily WHERE symbol = :symbol 
                     AND trade_date >= CURRENT_DATE - INTERVAL '180 days' ORDER BY trade_date DESC LIMIT 1) as price_6m_ago
            """)
            
            momentum_result = self.conn.execute(momentum_query, {"symbol": data.symbol}).fetchone()
            
            scores = {}
            
            if momentum_result:
                # Price Momentum (50% weight)
                momentum_3m = ((momentum_result.current_price / momentum_result.price_3m_ago - 1) * 100) if momentum_result.price_3m_ago else 0
                momentum_6m = ((momentum_result.current_price / momentum_result.price_6m_ago - 1) * 100) if momentum_result.price_6m_ago else 0
                
                momentum_3m_score = min(max((momentum_3m + 10) / 40 * 100, 0), 100)
                momentum_6m_score = min(max((momentum_6m + 5) / 35 * 100, 0), 100)
                
                price_momentum_score = (momentum_3m_score * 0.6 + momentum_6m_score * 0.4)
            else:
                price_momentum_score = 50
            
            # Earnings Momentum (30% weight)
            earnings_momentum_score = min(max((data.eps_growth - 0) / 30 * 100, 0), 100) if data.eps_growth else 50
            
            # Quality Filter (20% weight)
            roe_score = min(max((data.roe - 10) / 20 * 100, 0), 100) if data.roe else 50
            
            # Combined enhanced score
            enhanced_score = (price_momentum_score * 0.5 + earnings_momentum_score * 0.3 + roe_score * 0.2)
            
            scores['enhanced_score'] = enhanced_score
            scores['price_momentum_score'] = price_momentum_score
            scores['conviction_level'] = self._calculate_conviction_level(enhanced_score, price_momentum_score)
            
            return scores
            
        except Exception as e:
            logger.error(f"Error in enhanced momentum scoring: {str(e)}")
            return {'enhanced_score': 50, 'conviction_level': 1.0}
    
    def _calculate_enhanced_dividend_score(self, data):
        """Calculate enhanced dividend score with new fundamentals"""
        try:
            scores = {}
            
            # Dividend Sustainability (40% weight)
            dividend_yield = data.dividend_yield if hasattr(data, 'dividend_yield') else 3.0
            payout_ratio = min(data.payout_ratio, 100) if hasattr(data, 'payout_ratio') else 50
            
            yield_score = min(max((dividend_yield - 1) / 6 * 100, 0), 100)
            sustainability_score = min(max((80 - payout_ratio) / 60 * 100, 0), 100)
            
            dividend_score = (yield_score * 0.6 + sustainability_score * 0.4)
            
            # Quality Metrics (40% weight)
            roe_score = min(max((data.roe - 8) / 17 * 100, 0), 100) if data.roe else 50
            fcf_score = min(max((data.free_cash_flow_yield - 2) / 8 * 100, 0), 100) if data.free_cash_flow_yield else 50
            
            quality_score = (roe_score * 0.5 + fcf_score * 0.5)
            
            # Value Component (20% weight)
            pe_score = min(max((20 - data.pe_ratio) / 15 * 100, 0), 100) if data.pe_ratio else 50
            
            # Combined enhanced score
            enhanced_score = (dividend_score * 0.4 + quality_score * 0.4 + pe_score * 0.2)
            
            scores['enhanced_score'] = enhanced_score
            scores['dividend_score'] = dividend_score
            scores['conviction_level'] = self._calculate_conviction_level(enhanced_score, quality_score)
            
            return scores
            
        except Exception as e:
            logger.error(f"Error in enhanced dividend scoring: {str(e)}")
            return {'enhanced_score': 50, 'conviction_level': 1.0}
    
    def _calculate_conviction_level(self, enhanced_score, component_score):
        """Calculate conviction level for position sizing"""
        # High conviction if both overall score and key component are strong
        if enhanced_score >= 80 and component_score >= 75:
            return 2.0  # High conviction
        elif enhanced_score >= 65 and component_score >= 60:
            return 1.5  # Medium-high conviction
        elif enhanced_score >= 50:
            return 1.0  # Normal conviction
        else:
            return 0.5  # Low conviction
    
    def calculate_optimized_position_size(self, enhanced_score, conviction_level, portfolio_size):
        """Calculate position size based on conviction and score"""
        if not self.conviction_sizing['enabled']:
            return self.conviction_sizing['base_weight']
        
        # Base weight adjusted by conviction and score
        score_multiplier = enhanced_score / 100
        conviction_multiplier = conviction_level * self.conviction_sizing['conviction_multiplier']
        
        # Calculate raw weight
        raw_weight = self.conviction_sizing['base_weight'] * score_multiplier * conviction_multiplier
        
        # Apply limits
        optimized_weight = min(max(raw_weight, self.conviction_sizing['min_weight']), 
                              self.conviction_sizing['max_weight'])
        
        return optimized_weight
    
    def run_optimized_semi_annual_strategy(self, strategy_name, strategy_type, market_cap):
        """Run optimized semi-annual strategy with enhanced fundamentals"""
        logger.info(f"Running Optimized Semi-Annual {strategy_name}")
        
        try:
            # Get stocks for market cap category
            stock_query = text("""
                SELECT symbol, sector, market_cap, close_price
                FROM pure_us_stocks 
                WHERE market_cap BETWEEN :min_cap AND :max_cap
                AND sector IS NOT NULL
                AND close_price > 5.0
                ORDER BY market_cap DESC
            """)
            
            cap_ranges = {
                'small_cap': (300_000_000, 2_000_000_000),
                'mid_cap': (2_000_000_000, 10_000_000_000),
                'large_cap': (10_000_000_000, 1_000_000_000_000)
            }
            
            min_cap, max_cap = cap_ranges[market_cap]
            stocks = self.conn.execute(stock_query, {"min_cap": min_cap, "max_cap": max_cap}).fetchall()
            
            logger.info(f"Analyzing {len(stocks)} stocks for {strategy_name}")
            
            # Calculate enhanced scores for all stocks
            enhanced_results = []
            
            for stock in stocks:
                enhanced_metrics = self.calculate_enhanced_fundamentals(stock.symbol, strategy_type)
                
                if enhanced_metrics and enhanced_metrics.get('enhanced_score', 0) > 0:
                    # Calculate optimized position size
                    conviction_level = enhanced_metrics.get('conviction_level', 1.0)
                    position_size = self.calculate_optimized_position_size(
                        enhanced_metrics['enhanced_score'], 
                        conviction_level, 
                        len(stocks)
                    )
                    
                    enhanced_results.append({
                        'symbol': stock.symbol,
                        'sector': stock.sector,
                        'enhanced_score': enhanced_metrics['enhanced_score'],
                        'conviction_level': conviction_level,
                        'position_size': position_size,
                        'market_cap': stock.market_cap,
                        'price': stock.close_price
                    })
            
            # Sort by enhanced score and select top positions
            enhanced_results.sort(key=lambda x: x['enhanced_score'], reverse=True)
            
            # Dynamic portfolio size based on conviction
            total_conviction_weight = sum(r['position_size'] for r in enhanced_results[:200])
            
            if total_conviction_weight > 0:
                # Normalize weights to sum to 100%
                for result in enhanced_results[:200]:
                    result['normalized_weight'] = (result['position_size'] / total_conviction_weight) * 100
            
            # Select final portfolio (top 100-150 positions)
            final_portfolio = enhanced_results[:150]
            
            # Calculate portfolio statistics
            total_positions = len(final_portfolio)
            avg_enhanced_score = np.mean([r['enhanced_score'] for r in final_portfolio])
            avg_conviction = np.mean([r['conviction_level'] for r in final_portfolio])
            top_score = max([r['enhanced_score'] for r in final_portfolio]) if final_portfolio else 0
            
            print(f"\n[OPTIMIZED] {strategy_name.upper()} - Enhanced Semi-Annual Results")
            print("=" * 70)
            print(f"Total Positions: {total_positions}")
            print(f"Average Enhanced Score: {avg_enhanced_score:.1f}")
            print(f"Average Conviction Level: {avg_conviction:.1f}x")
            print(f"Top Enhanced Score: {top_score:.1f}")
            
            # Show sector allocation
            sector_allocation = {}
            for result in final_portfolio:
                sector = result['sector']
                if sector not in sector_allocation:
                    sector_allocation[sector] = {'count': 0, 'weight': 0}
                sector_allocation[sector]['count'] += 1
                sector_allocation[sector]['weight'] += result.get('normalized_weight', 0)
            
            print(f"\nOptimized Sector Allocation:")
            for sector, data in sorted(sector_allocation.items(), key=lambda x: x[1]['count'], reverse=True):
                print(f"  {sector:<25}: {data['count']:>3} positions ({data['weight']:>5.1f}%)")
            
            # Show top 10 high-conviction positions
            print(f"\nTop 10 High-Conviction Holdings:")
            print("Symbol   Sector                  Score   Conv   Weight%")
            print("-" * 55)
            for result in final_portfolio[:10]:
                print(f"{result['symbol']:<8} {result['sector'][:20]:<20} {result['enhanced_score']:>5.1f}  {result['conviction_level']:>4.1f}x {result.get('normalized_weight', 0):>6.1f}%")
            
            return {
                'strategy': strategy_name,
                'total_positions': total_positions,
                'avg_enhanced_score': avg_enhanced_score,
                'avg_conviction': avg_conviction,
                'top_score': top_score,
                'portfolio': final_portfolio,
                'sector_allocation': sector_allocation
            }
            
        except Exception as e:
            logger.error(f"Error in optimized semi-annual strategy {strategy_name}: {str(e)}")
            return None

def main():
    """Run all optimized semi-annual strategies"""
    print("\n[LAUNCH] ACIS Optimized Semi-Annual Rebalancing System")
    print("Enhanced Fundamentals + Conviction Sizing + Semi-Annual Frequency")
    print("=" * 80)
    
    system = OptimizedSemiAnnualSystem()
    
    # Define strategies with optimizations
    optimized_strategies = [
        # Mid Cap Focus (higher weights)
        ('Optimized Mid Cap Value', 'value', 'mid_cap'),
        ('Optimized Mid Cap Growth', 'growth', 'mid_cap'),
        ('Optimized Mid Cap Momentum', 'momentum', 'mid_cap'),
        ('Optimized Mid Cap Dividend', 'dividend', 'mid_cap'),
        
        # Large Cap
        ('Optimized Large Cap Value', 'value', 'large_cap'),
        ('Optimized Large Cap Growth', 'growth', 'large_cap'),
        ('Optimized Large Cap Momentum', 'momentum', 'large_cap'),
        ('Optimized Large Cap Dividend', 'dividend', 'large_cap'),
        
        # Small Cap (slightly reduced)
        ('Optimized Small Cap Value', 'value', 'small_cap'),
        ('Optimized Small Cap Growth', 'growth', 'small_cap'),
        ('Optimized Small Cap Momentum', 'momentum', 'small_cap'),
        ('Optimized Small Cap Dividend', 'dividend', 'small_cap')
    ]
    
    all_results = []
    
    for strategy_name, strategy_type, market_cap in optimized_strategies:
        result = system.run_optimized_semi_annual_strategy(strategy_name, strategy_type, market_cap)
        if result:
            all_results.append(result)
    
    # Portfolio summary
    print(f"\n" + "=" * 80)
    print("OPTIMIZED SEMI-ANNUAL SYSTEM SUMMARY")
    print("=" * 80)
    
    total_strategies = len(all_results)
    avg_score_all = np.mean([r['avg_enhanced_score'] for r in all_results])
    avg_conviction_all = np.mean([r['avg_conviction'] for r in all_results])
    total_positions_all = sum([r['total_positions'] for r in all_results])
    
    print(f"Total Optimized Strategies: {total_strategies}")
    print(f"Total Positions Across All Strategies: {total_positions_all}")
    print(f"Average Enhanced Score: {avg_score_all:.1f}")
    print(f"Average Conviction Level: {avg_conviction_all:.1f}x")
    
    # Top performing strategies
    sorted_results = sorted(all_results, key=lambda x: x['avg_enhanced_score'], reverse=True)
    print(f"\nTop Performing Optimized Strategies:")
    for i, result in enumerate(sorted_results[:5]):
        print(f"  {i+1}. {result['strategy']}")
        print(f"     Score: {result['avg_enhanced_score']:.1f}, Conviction: {result['avg_conviction']:.1f}x")
    
    print(f"\n[SUCCESS] Optimized Semi-Annual System Complete!")
    print("Ready for backtesting with enhanced fundamentals and conviction sizing")
    
    return all_results

if __name__ == "__main__":
    main()