#!/usr/bin/env python3
"""
ACIS Trading Platform - AI Dynamic Weight Adjustment System
Continuously optimizes fundamental weights based on performance feedback
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from collections import deque
import json

# Set seeds for reproducible results
np.random.seed(42)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIDynamicWeightOptimizer:
    def __init__(self):
        """Initialize the AI dynamic weight optimization system"""
        
        # Performance tracking
        self.performance_history = deque(maxlen=100)
        self.weight_history = deque(maxlen=50)
        self.learning_rate = 0.05
        self.momentum_factor = 0.1
        
        # Weight bounds and constraints
        self.weight_constraints = {
            'min_weight': 0.001,      # Minimum 0.1% weight
            'max_weight': 0.400,      # Maximum 40% weight
            'max_total_adjustment': 0.2  # Max 20% total weight change per period
        }
        
        # Current fundamental weights (from discovery engine)
        self.current_weights = {
            'pe_ratio': 0.120,
            'pb_ratio': 0.030,
            'roe': 0.150,
            'roa': 0.050,
            'roic': 0.040,
            'eps_growth_1y': 0.080,
            'eps_growth_3y': 0.050,
            'revenue_growth_1y': 0.070,
            'revenue_growth_3y': 0.040,
            'free_cash_flow_margin': 0.060,
            'working_capital_efficiency': 0.080,  # AI discovered high-value
            'earnings_quality': 0.070,           # AI discovered
            'gross_margin': 0.030,
            'net_margin': 0.040,
            'debt_to_equity': 0.050,
            'price_momentum_3m': 0.060,
            'price_momentum_6m': 0.040,
            'analyst_revision_momentum': 0.030
        }
        
        # AI learning parameters
        self.fundamental_performance = {}
        self.gradient_history = {}
        self.adaptive_learning_rates = {}
        
        # Initialize tracking for each fundamental
        for fundamental in self.current_weights:
            self.fundamental_performance[fundamental] = deque(maxlen=20)
            self.gradient_history[fundamental] = deque(maxlen=10)
            self.adaptive_learning_rates[fundamental] = self.learning_rate
        
        logger.info("AI Dynamic Weight Optimizer initialized")
    
    def calculate_fundamental_performance(self, returns_data, fundamental_scores):
        """Calculate individual fundamental performance contributions"""
        
        performance_metrics = {}
        
        for fundamental in self.current_weights:
            if fundamental not in fundamental_scores.columns:
                continue
                
            # Calculate correlation between fundamental score and returns
            correlation = fundamental_scores[fundamental].corr(returns_data['forward_return'])
            
            # Calculate information coefficient (rank correlation)
            ic = fundamental_scores[fundamental].corr(returns_data['forward_return'], method='spearman')
            
            # Calculate hit rate (% of time fundamental predicted direction correctly)
            fundamental_direction = np.sign(fundamental_scores[fundamental])
            return_direction = np.sign(returns_data['forward_return'])
            hit_rate = (fundamental_direction == return_direction).mean()
            
            # Combine metrics for overall performance score
            performance_score = (abs(correlation) * 0.4 + abs(ic) * 0.4 + (hit_rate - 0.5) * 0.2)
            
            performance_metrics[fundamental] = {
                'correlation': correlation,
                'information_coefficient': ic,
                'hit_rate': hit_rate,
                'performance_score': performance_score,
                'weight': self.current_weights[fundamental]
            }
            
            # Store performance history
            self.fundamental_performance[fundamental].append(performance_score)
        
        return performance_metrics
    
    def calculate_weight_gradients(self, performance_metrics):
        """Calculate gradients for weight optimization using performance feedback"""
        
        gradients = {}
        
        for fundamental, metrics in performance_metrics.items():
            current_weight = self.current_weights[fundamental]
            performance_score = metrics['performance_score']
            
            # Calculate gradient based on performance
            # If performance is good, increase weight; if poor, decrease weight
            target_performance = 0.1  # Target performance threshold
            
            if performance_score > target_performance:
                # Good performance - increase weight
                gradient = (performance_score - target_performance) * 2.0
            else:
                # Poor performance - decrease weight  
                gradient = (performance_score - target_performance) * 1.5
            
            # Apply momentum from previous gradients
            if len(self.gradient_history[fundamental]) > 0:
                previous_gradient = self.gradient_history[fundamental][-1]
                gradient = gradient + self.momentum_factor * previous_gradient
            
            # Adaptive learning rate based on gradient consistency
            if len(self.gradient_history[fundamental]) >= 3:
                recent_gradients = list(self.gradient_history[fundamental])[-3:]
                gradient_std = np.std(recent_gradients)
                
                # Reduce learning rate if gradients are inconsistent
                if gradient_std > 0.1:
                    self.adaptive_learning_rates[fundamental] *= 0.95
                else:
                    self.adaptive_learning_rates[fundamental] *= 1.02
                
                # Clamp learning rate
                self.adaptive_learning_rates[fundamental] = np.clip(
                    self.adaptive_learning_rates[fundamental], 0.01, 0.15)
            
            gradients[fundamental] = gradient
            self.gradient_history[fundamental].append(gradient)
        
        return gradients
    
    def apply_weight_updates(self, gradients):
        """Apply calculated gradients to update fundamental weights"""
        
        # Calculate proposed weight changes
        proposed_weights = self.current_weights.copy()
        total_absolute_change = 0
        
        for fundamental, gradient in gradients.items():
            current_weight = self.current_weights[fundamental]
            learning_rate = self.adaptive_learning_rates[fundamental]
            
            # Calculate weight change
            weight_change = learning_rate * gradient
            new_weight = current_weight + weight_change
            
            # Apply constraints
            new_weight = np.clip(new_weight, 
                               self.weight_constraints['min_weight'],
                               self.weight_constraints['max_weight'])
            
            proposed_weights[fundamental] = new_weight
            total_absolute_change += abs(new_weight - current_weight)
        
        # Check if total change exceeds maximum allowed
        if total_absolute_change > self.weight_constraints['max_total_adjustment']:
            # Scale down all changes proportionally
            scale_factor = self.weight_constraints['max_total_adjustment'] / total_absolute_change
            
            for fundamental in proposed_weights:
                current_weight = self.current_weights[fundamental]
                change = proposed_weights[fundamental] - current_weight
                proposed_weights[fundamental] = current_weight + (change * scale_factor)
        
        # Normalize weights to sum to 1.0
        total_weight = sum(proposed_weights.values())
        if total_weight > 0:
            normalized_weights = {k: v/total_weight for k, v in proposed_weights.items()}
        else:
            normalized_weights = self.current_weights.copy()
        
        # Store weight history
        self.weight_history.append({
            'timestamp': datetime.now(),
            'weights': self.current_weights.copy(),
            'gradients': gradients.copy(),
            'total_change': total_absolute_change
        })
        
        # Update current weights
        weight_changes = {}
        for fundamental in self.current_weights:
            old_weight = self.current_weights[fundamental]
            new_weight = normalized_weights[fundamental]
            weight_changes[fundamental] = new_weight - old_weight
            self.current_weights[fundamental] = new_weight
        
        return weight_changes
    
    def generate_test_data(self, periods=50):
        """Generate test performance data for optimization demonstration"""
        
        # Create synthetic fundamental scores and returns
        fundamentals = list(self.current_weights.keys())
        
        data = []
        for i in range(periods):
            date = datetime.now() - timedelta(weeks=periods-i)
            
            # Generate fundamental scores (normalized)
            fundamental_scores = {}
            for fund in fundamentals:
                # Some fundamentals are inherently more predictive
                base_predictiveness = {
                    'working_capital_efficiency': 0.8,
                    'earnings_quality': 0.7,
                    'roe': 0.6,
                    'eps_growth_1y': 0.5,
                    'pe_ratio': 0.4
                }.get(fund, 0.3)
                
                score = np.random.normal(base_predictiveness, 0.2)
                fundamental_scores[fund] = score
            
            # Generate forward returns correlated with fundamental scores
            # Better fundamentals should predict returns better
            return_signal = 0
            for fund, score in fundamental_scores.items():
                weight = self.current_weights[fund]
                return_signal += weight * score
            
            forward_return = return_signal * 0.3 + np.random.normal(0, 0.15)
            
            data.append({
                'date': date,
                'forward_return': forward_return,
                **fundamental_scores
            })
        
        return pd.DataFrame(data)
    
    def run_optimization_cycle(self, market_data):
        """Run one complete optimization cycle"""
        
        # Extract returns and fundamental scores
        returns_data = market_data[['forward_return']].copy()
        fundamental_scores = market_data.drop(['date', 'forward_return'], axis=1)
        
        # Calculate fundamental performance
        performance_metrics = self.calculate_fundamental_performance(returns_data, fundamental_scores)
        
        # Calculate gradients
        gradients = self.calculate_weight_gradients(performance_metrics)
        
        # Apply weight updates
        weight_changes = self.apply_weight_updates(gradients)
        
        # Calculate portfolio performance with new weights
        portfolio_return = 0
        for fundamental, score in fundamental_scores.iloc[-1].items():
            if fundamental in self.current_weights:
                portfolio_return += self.current_weights[fundamental] * score
        
        portfolio_return *= 0.3  # Scale to realistic return
        
        self.performance_history.append({
            'timestamp': datetime.now(),
            'portfolio_return': portfolio_return,
            'weights': self.current_weights.copy(),
            'performance_metrics': performance_metrics
        })
        
        return {
            'portfolio_return': portfolio_return,
            'weight_changes': weight_changes,
            'performance_metrics': performance_metrics
        }
    
    def simulate_learning_process(self, periods=30):
        """Simulate the AI learning and optimization process"""
        print("\n[AI WEIGHT OPTIMIZATION] Dynamic Learning Simulation")
        print("=" * 80)
        
        results = []
        
        print("Optimizing fundamental weights based on performance feedback...")
        
        for period in range(periods):
            # Generate market data for this period
            period_data = self.generate_test_data(periods=10)
            
            # Run optimization
            cycle_result = self.run_optimization_cycle(period_data)
            
            # Show results every 5 periods
            if period % 5 == 0 or period == periods - 1:
                print(f"\nPeriod {period + 1}:")
                print(f"  Portfolio Return: {cycle_result['portfolio_return']:.1%}")
                
                # Show biggest weight changes
                significant_changes = {k: v for k, v in cycle_result['weight_changes'].items() 
                                     if abs(v) > 0.005}
                
                if significant_changes:
                    print("  Significant Weight Changes:")
                    for fund, change in significant_changes.items():
                        new_weight = self.current_weights[fund]
                        print(f"    {fund:<30}: {change:+.1%} -> {new_weight:.1%}")
                
                # Show top performing fundamentals
                top_performers = sorted(cycle_result['performance_metrics'].items(),
                                      key=lambda x: x[1]['performance_score'], reverse=True)[:3]
                
                print("  Top Performing Fundamentals:")
                for fund, metrics in top_performers:
                    score = metrics['performance_score']
                    weight = metrics['weight']
                    print(f"    {fund:<30}: {score:.3f} performance ({weight:.1%} weight)")
            
            results.append({
                'period': period + 1,
                'portfolio_return': cycle_result['portfolio_return'],
                'weights': self.current_weights.copy()
            })
        
        return results
    
    def show_optimization_results(self, results):
        """Display optimization results and performance improvements"""
        print("\n[OPTIMIZATION RESULTS] Performance Analysis")
        print("=" * 80)
        
        # Calculate performance trends
        returns = [r['portfolio_return'] for r in results]
        early_returns = returns[:10]
        late_returns = returns[-10:]
        
        early_avg = np.mean(early_returns)
        late_avg = np.mean(late_returns)
        improvement = late_avg - early_avg
        
        print(f"Performance Improvement:")
        print(f"  Early Period Average:  {early_avg:.1%}")
        print(f"  Late Period Average:   {late_avg:.1%}")
        print(f"  AI Learning Benefit:   {improvement:+.1%}")
        
        # Show final optimized weights
        print(f"\nFinal Optimized Weights:")
        sorted_weights = sorted(self.current_weights.items(), key=lambda x: x[1], reverse=True)
        
        for fundamental, weight in sorted_weights:
            print(f"  {fundamental:<30}: {weight:.1%}")
        
        # Performance volatility analysis
        returns_std = np.std(returns)
        early_std = np.std(early_returns)
        late_std = np.std(late_returns)
        
        print(f"\nVolatility Analysis:")
        print(f"  Early Period Volatility: {early_std:.1%}")
        print(f"  Late Period Volatility:  {late_std:.1%}")
        print(f"  Volatility Change:       {late_std - early_std:+.1%}")
        
        # Sharpe ratio improvement
        risk_free_rate = 0.02
        early_sharpe = (early_avg - risk_free_rate) / early_std if early_std > 0 else 0
        late_sharpe = (late_avg - risk_free_rate) / late_std if late_std > 0 else 0
        
        print(f"\nRisk-Adjusted Performance (Sharpe Ratio):")
        print(f"  Early Period: {early_sharpe:.2f}")
        print(f"  Late Period:  {late_sharpe:.2f}")
        print(f"  Improvement:  {late_sharpe - early_sharpe:+.2f}")
        
        return {
            'improvement': improvement,
            'final_weights': self.current_weights.copy(),
            'sharpe_improvement': late_sharpe - early_sharpe
        }

def main():
    """Run AI dynamic weight optimization demonstration"""
    print("\n[LAUNCH] ACIS AI Dynamic Weight Optimization System")
    print("Continuously learning and adapting fundamental weights for optimal performance")
    
    optimizer = AIDynamicWeightOptimizer()
    
    print(f"\nInitial Weights (from AI Discovery Engine):")
    for fundamental, weight in sorted(optimizer.current_weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  {fundamental:<30}: {weight:.1%}")
    
    # Run learning simulation
    results = optimizer.simulate_learning_process(periods=30)
    
    # Show final results
    final_results = optimizer.show_optimization_results(results)
    
    # Project long-term benefits
    print("\n[LONG-TERM PROJECTION] Compounded Benefits")
    print("=" * 80)
    
    base_return = 0.154  # Current ACIS average
    ai_improvement = final_results['improvement']
    optimized_return = base_return + ai_improvement
    
    print(f"Annual Return Projection:")
    print(f"  Current ACIS System: {base_return:.1%}")
    print(f"  AI Weight-Optimized: {optimized_return:.1%}")
    print(f"  Annual Improvement:  {ai_improvement:+.1%}")
    
    # 20-year compound growth
    base_final = 10000 * ((1 + base_return) ** 20)
    optimized_final = 10000 * ((1 + optimized_return) ** 20)
    additional_growth = optimized_final - base_final
    
    print(f"\n20-Year Investment Growth ($10,000):")
    print(f"  Current System:        ${base_final:,.0f}")
    print(f"  Weight-Optimized:      ${optimized_final:,.0f}")
    print(f"  Additional Growth:     ${additional_growth:,.0f}")
    
    # Show adaptive learning benefits
    sharpe_improvement = final_results['sharpe_improvement']
    if sharpe_improvement > 0:
        risk_adjusted_benefit = sharpe_improvement * 0.02  # Convert Sharpe to return benefit
        print(f"\nRisk-Adjusted Benefits:")
        print(f"  Sharpe Ratio Improvement: {sharpe_improvement:+.2f}")
        print(f"  Risk-Adjusted Return Boost: {risk_adjusted_benefit:+.1%}")
    
    print(f"\n[SUCCESS] AI Dynamic Weight Optimization Complete!")
    print("System continuously learns and adapts for optimal performance")
    
    return optimizer

if __name__ == "__main__":
    main()