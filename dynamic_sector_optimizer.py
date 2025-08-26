#!/usr/bin/env python3
"""
ACIS Trading Platform - Dynamic Sector Rotation Optimizer
Implements dynamic sector allocation based on fundamental strength and economic indicators
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from sqlalchemy import create_engine, text
from database_config import get_database_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DynamicSectorOptimizer:
    def __init__(self):
        self.conn = get_database_connection()
        
        # Economic cycle indicators
        self.economic_indicators = {
            'interest_rate_environment': 'rising',  # rising, falling, stable
            'gdp_growth_trend': 'moderate',        # strong, moderate, weak
            'inflation_environment': 'moderate',    # high, moderate, low
            'market_cycle_phase': 'expansion'       # expansion, peak, contraction, trough
        }
        
        # Sector rotation matrix based on economic conditions
        self.sector_rotation_matrix = {
            'expansion': {
                'TECHNOLOGY': 1.3,
                'LIFE SCIENCES': 1.2,
                'FINANCE': 1.2,
                'MANUFACTURING': 1.1,
                'TRADE & SERVICES': 1.1,
                'ENERGY & TRANSPORTATION': 1.0,
                'REAL ESTATE & CONSTRUCTION': 0.9
            },
            'peak': {
                'FINANCE': 1.3,
                'ENERGY & TRANSPORTATION': 1.2,
                'MANUFACTURING': 1.1,
                'TECHNOLOGY': 1.0,
                'LIFE SCIENCES': 1.0,
                'TRADE & SERVICES': 0.9,
                'REAL ESTATE & CONSTRUCTION': 0.8
            },
            'contraction': {
                'LIFE SCIENCES': 1.3,
                'TRADE & SERVICES': 1.2,
                'TECHNOLOGY': 1.1,
                'FINANCE': 0.8,
                'MANUFACTURING': 0.8,
                'ENERGY & TRANSPORTATION': 0.7,
                'REAL ESTATE & CONSTRUCTION': 0.7
            }
        }
        
        # Strategy-specific sector preferences
        self.strategy_sector_preferences = {
            'value': {
                'FINANCE': 1.2,
                'ENERGY & TRANSPORTATION': 1.1,
                'MANUFACTURING': 1.1,
                'REAL ESTATE & CONSTRUCTION': 1.0
            },
            'growth': {
                'TECHNOLOGY': 1.3,
                'LIFE SCIENCES': 1.2,
                'TRADE & SERVICES': 1.1,
                'FINANCE': 1.0
            },
            'momentum': {
                'TECHNOLOGY': 1.2,
                'LIFE SCIENCES': 1.1,
                'FINANCE': 1.1,
                'MANUFACTURING': 1.0
            },
            'dividend': {
                'FINANCE': 1.2,
                'TRADE & SERVICES': 1.1,
                'MANUFACTURING': 1.1,
                'ENERGY & TRANSPORTATION': 1.0
            }
        }
        
        logger.info("Dynamic Sector Optimizer initialized")
    
    def calculate_sector_fundamental_strength(self):
        """Calculate fundamental strength scores for each sector"""
        try:
            sector_strength_query = text("""
                SELECT 
                    p.sector,
                    COUNT(*) as stock_count,
                    AVG(f.roe) as avg_roe,
                    AVG(f.roa) as avg_roa,
                    AVG(f.eps_growth) as avg_eps_growth,
                    AVG(f.revenue_growth) as avg_revenue_growth,
                    AVG(f.pe_ratio) as avg_pe_ratio,
                    AVG(f.debt_to_equity) as avg_debt_to_equity,
                    AVG(f.profit_margin) as avg_profit_margin,
                    -- Calculate momentum
                    AVG(CASE 
                        WHEN current_prices.current_price > 0 AND past_prices.past_price > 0 
                        THEN (current_prices.current_price / past_prices.past_price - 1) * 100 
                        ELSE 0 
                    END) as price_momentum_6m
                FROM pure_us_stocks p
                JOIN fundamentals_quarterly f ON p.symbol = f.symbol
                LEFT JOIN (
                    SELECT symbol, close_price as current_price
                    FROM stock_eod_daily 
                    WHERE trade_date >= CURRENT_DATE - INTERVAL '7 days'
                    AND trade_date <= CURRENT_DATE
                ) current_prices ON p.symbol = current_prices.symbol
                LEFT JOIN (
                    SELECT symbol, close_price as past_price
                    FROM stock_eod_daily 
                    WHERE trade_date >= CURRENT_DATE - INTERVAL '180 days'
                    AND trade_date <= CURRENT_DATE - INTERVAL '173 days'
                ) past_prices ON p.symbol = past_prices.symbol
                WHERE f.fiscal_date >= CURRENT_DATE - INTERVAL '12 months'
                AND p.sector IS NOT NULL
                GROUP BY p.sector
                HAVING COUNT(*) >= 5
                ORDER BY p.sector
            """)
            
            sector_data = self.conn.execute(sector_strength_query).fetchall()
            
            if not sector_data:
                logger.warning("No sector data found, using default weights")
                return {}
            
            # Calculate composite strength scores
            sector_strength = {}
            
            for sector_row in sector_data:
                sector = sector_row.sector
                
                # Quality score (40% weight)
                roe_score = min(max((sector_row.avg_roe or 10) / 20 * 100, 0), 100)
                roa_score = min(max((sector_row.avg_roa or 5) / 15 * 100, 0), 100)
                margin_score = min(max((sector_row.avg_profit_margin or 5) / 20 * 100, 0), 100)
                
                quality_score = (roe_score * 0.4 + roa_score * 0.3 + margin_score * 0.3)
                
                # Growth score (30% weight)
                eps_growth_score = min(max((sector_row.avg_eps_growth or 5) / 25 * 100, 0), 100)
                revenue_growth_score = min(max((sector_row.avg_revenue_growth or 3) / 20 * 100, 0), 100)
                
                growth_score = (eps_growth_score * 0.6 + revenue_growth_score * 0.4)
                
                # Value score (20% weight)
                pe_score = min(max((25 - (sector_row.avg_pe_ratio or 20)) / 15 * 100, 0), 100)
                debt_score = min(max((100 - (sector_row.avg_debt_to_equity or 50)) / 80 * 100, 0), 100)
                
                value_score = (pe_score * 0.7 + debt_score * 0.3)
                
                # Momentum score (10% weight)
                momentum_score = min(max((sector_row.price_momentum_6m or 0) + 10, 0), 100) * 2
                momentum_score = min(momentum_score, 100)
                
                # Composite strength
                composite_strength = (
                    quality_score * 0.4 + 
                    growth_score * 0.3 + 
                    value_score * 0.2 + 
                    momentum_score * 0.1
                )
                
                sector_strength[sector] = {
                    'composite_strength': composite_strength,
                    'quality_score': quality_score,
                    'growth_score': growth_score,
                    'value_score': value_score,
                    'momentum_score': momentum_score,
                    'stock_count': sector_row.stock_count
                }
            
            logger.info(f"Calculated strength scores for {len(sector_strength)} sectors")
            return sector_strength
            
        except Exception as e:
            logger.error(f"Error calculating sector fundamental strength: {str(e)}")
            return {}
    
    def get_dynamic_sector_weights(self, strategy_type='value'):
        """Get optimized sector weights based on current conditions"""
        try:
            # Get fundamental strength scores
            sector_strength = self.calculate_sector_fundamental_strength()
            
            if not sector_strength:
                logger.warning("Using default sector weights")
                return self._get_default_sector_weights()
            
            # Get economic cycle weights
            cycle_phase = self.economic_indicators['market_cycle_phase']
            cycle_weights = self.sector_rotation_matrix.get(cycle_phase, {})
            
            # Get strategy-specific preferences
            strategy_preferences = self.strategy_sector_preferences.get(strategy_type, {})
            
            # Calculate dynamic weights
            dynamic_weights = {}
            base_weight = 1.0
            
            for sector, strength_data in sector_strength.items():
                # Start with fundamental strength (normalized to 0.5-1.5 range)
                strength_multiplier = 0.5 + (strength_data['composite_strength'] / 100)
                
                # Apply economic cycle adjustment
                cycle_multiplier = cycle_weights.get(sector, 1.0)
                
                # Apply strategy preference
                strategy_multiplier = strategy_preferences.get(sector, 1.0)
                
                # Combine all factors
                combined_weight = base_weight * strength_multiplier * cycle_multiplier * strategy_multiplier
                
                # Apply reasonable bounds (0.5x to 2.0x normal allocation)
                dynamic_weights[sector] = min(max(combined_weight, 0.5), 2.0)
            
            # Normalize weights so total doesn't exceed reasonable bounds
            total_weight = sum(dynamic_weights.values())
            if total_weight > len(dynamic_weights) * 1.2:  # If average weight > 1.2x
                normalization_factor = (len(dynamic_weights) * 1.1) / total_weight
                for sector in dynamic_weights:
                    dynamic_weights[sector] *= normalization_factor
            
            return dynamic_weights
            
        except Exception as e:
            logger.error(f"Error calculating dynamic sector weights: {str(e)}")
            return self._get_default_sector_weights()
    
    def _get_default_sector_weights(self):
        """Get default sector weights as fallback"""
        return {
            'TECHNOLOGY': 1.0,
            'LIFE SCIENCES': 1.0,
            'FINANCE': 1.0,
            'MANUFACTURING': 1.0,
            'TRADE & SERVICES': 1.0,
            'ENERGY & TRANSPORTATION': 1.0,
            'REAL ESTATE & CONSTRUCTION': 1.0
        }
    
    def apply_dynamic_sector_allocation(self, portfolio_results, strategy_type='value'):
        """Apply dynamic sector weights to portfolio results"""
        try:
            if not portfolio_results:
                return portfolio_results
            
            # Get dynamic weights
            dynamic_weights = self.get_dynamic_sector_weights(strategy_type)
            
            # Apply weights to portfolio
            for result in portfolio_results:
                sector = result.get('sector', 'UNKNOWN')
                sector_weight = dynamic_weights.get(sector, 1.0)
                
                # Adjust position size based on sector weight
                current_weight = result.get('normalized_weight', result.get('position_size', 1.0))
                result['sector_adjusted_weight'] = current_weight * sector_weight
                result['sector_multiplier'] = sector_weight
            
            # Renormalize after sector adjustment
            total_adjusted_weight = sum(r.get('sector_adjusted_weight', 1.0) for r in portfolio_results)
            if total_adjusted_weight > 0:
                for result in portfolio_results:
                    result['final_weight'] = (result.get('sector_adjusted_weight', 1.0) / total_adjusted_weight) * 100
            
            # Sort by final weight
            portfolio_results.sort(key=lambda x: x.get('final_weight', 0), reverse=True)
            
            logger.info(f"Applied dynamic sector allocation with {len(dynamic_weights)} sector adjustments")
            return portfolio_results
            
        except Exception as e:
            logger.error(f"Error applying dynamic sector allocation: {str(e)}")
            return portfolio_results
    
    def print_sector_analysis(self, strategy_type='value'):
        """Print detailed sector analysis"""
        try:
            print(f"\n[SECTOR ANALYSIS] Dynamic Sector Rotation - {strategy_type.upper()} Strategy")
            print("=" * 70)
            
            # Get sector strength
            sector_strength = self.calculate_sector_fundamental_strength()
            dynamic_weights = self.get_dynamic_sector_weights(strategy_type)
            
            if not sector_strength or not dynamic_weights:
                print("Insufficient data for sector analysis")
                return
            
            print(f"Economic Environment: {self.economic_indicators['market_cycle_phase'].title()} Phase")
            print(f"Strategy Type: {strategy_type.title()}")
            
            print(f"\nSector Strength Analysis:")
            print("Sector                    Composite  Quality  Growth  Value  Weight")
            print("-" * 70)
            
            # Sort sectors by dynamic weight
            sorted_sectors = sorted(dynamic_weights.items(), key=lambda x: x[1], reverse=True)
            
            for sector, weight in sorted_sectors:
                if sector in sector_strength:
                    strength = sector_strength[sector]
                    print(f"{sector:<25} {strength['composite_strength']:>7.1f}  {strength['quality_score']:>7.1f} {strength['growth_score']:>7.1f} {strength['value_score']:>6.1f} {weight:>6.1f}x")
                else:
                    print(f"{sector:<25} {'N/A':<7}  {'N/A':<7} {'N/A':<7} {'N/A':<6} {weight:>6.1f}x")
            
            # Overweight/underweight summary
            overweight_sectors = [s for s, w in dynamic_weights.items() if w > 1.1]
            underweight_sectors = [s for s, w in dynamic_weights.items() if w < 0.9]
            
            print(f"\nSector Positioning Summary:")
            if overweight_sectors:
                print(f"  Overweight: {', '.join(overweight_sectors)}")
            if underweight_sectors:
                print(f"  Underweight: {', '.join(underweight_sectors)}")
            
            print(f"  Neutral: {len(dynamic_weights) - len(overweight_sectors) - len(underweight_sectors)} sectors")
            
        except Exception as e:
            logger.error(f"Error in sector analysis: {str(e)}")

def main():
    """Test dynamic sector optimization"""
    print("\n[LAUNCH] Dynamic Sector Rotation Optimizer")
    print("Testing sector strength calculation and dynamic allocation")
    
    optimizer = DynamicSectorOptimizer()
    
    # Test all strategy types
    strategy_types = ['value', 'growth', 'momentum', 'dividend']
    
    for strategy in strategy_types:
        optimizer.print_sector_analysis(strategy)
        print()
    
    print("[SUCCESS] Dynamic Sector Optimization Analysis Complete")

if __name__ == "__main__":
    main()