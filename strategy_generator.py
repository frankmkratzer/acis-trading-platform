#!/usr/bin/env python3
# File: strategy_generator.py
# Purpose: Automatically generate and test trading strategies

import logging
import random
import copy
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)


class AutomatedStrategyGenerator:
    """Generate trading strategies using genetic programming and ML"""

    def __init__(self):
        self.population = []
        self.best_strategy = None
        self.generation = 0

    def generate_random_strategy(self) -> Dict:
        """Generate a random strategy configuration"""

        strategy = {
            'name': f'strategy_{random.randint(1000, 9999)}',

            # Entry conditions
            'entry_indicators': random.sample([
                'rsi', 'macd', 'bb_position', 'sma_cross',
                'volume_spike', 'momentum', 'mean_reversion'
            ], k=random.randint(2, 4)),

            'entry_thresholds': {
                'rsi': random.uniform(20, 40) if random.random() > 0.5 else random.uniform(60, 80),
                'macd': random.choice(['bullish_cross', 'bearish_cross']),
                'bb_position': random.uniform(0, 0.3) if random.random() > 0.5 else random.uniform(0.7, 1),
                'sma_cross': random.choice(['golden_cross', 'death_cross']),
                'volume_spike': random.uniform(1.5, 3.0),
                'momentum': random.uniform(0.02, 0.10),
                'mean_reversion': random.uniform(-2, -1) if random.random() > 0.5 else random.uniform(1, 2)
            },

            # Exit conditions
            'stop_loss': random.uniform(0.02, 0.10),
            'take_profit': random.uniform(0.05, 0.20),
            'trailing_stop': random.choice([None, random.uniform(0.01, 0.05)]),
            'time_stop': random.choice([None, random.randint(5, 30)]),

            # Position sizing
            'position_sizing': random.choice(['fixed', 'kelly', 'volatility', 'risk_parity']),
            'max_position_size': random.uniform(0.05, 0.20),

            # Filters
            'min_volume': random.uniform(100000, 1000000),
            'min_price': random.uniform(5, 20),
            'max_spread': random.uniform(0.001, 0.01),

            # Timing
            'trade_frequency': random.choice(['daily', 'weekly', 'monthly']),
            'holding_period': random.randint(1, 30),

            # Risk management
            'max_positions': random.randint(5, 20),
            'correlation_limit': random.uniform(0.5, 0.9),
            'sector_limit': random.uniform(0.2, 0.5)
        }

        return strategy

    def evaluate_strategy(self, strategy: Dict, data: pd.DataFrame) -> Dict:
        """Evaluate a strategy on historical data"""

        # Simple backtest
        signals = self.generate_signals(strategy, data)
        returns = self.calculate_returns(signals, data)

        # Calculate metrics
        metrics = {
            'total_return': (1 + returns).prod() - 1,
            'annual_return': (1 + returns.mean()) ** 252 - 1,
            'sharpe_ratio': np.sqrt(252) * returns.mean() / returns.std(),
            'max_drawdown': self.calculate_max_drawdown(returns),
            'win_rate': (returns > 0).mean(),
            'profit_factor': returns[returns > 0].sum() / abs(returns[returns < 0].sum()),
            'trades': len(returns),
            'fitness': 0  # Will be calculated
        }

        # Calculate fitness score
        metrics['fitness'] = (
                metrics['sharpe_ratio'] * 0.3 +
                metrics['profit_factor'] * 0.2 +
                (1 - metrics['max_drawdown']) * 0.2 +
                metrics['win_rate'] * 0.15 +
                min(metrics['trades'] / 100, 1) * 0.15  # Prefer active strategies
        )

        return metrics

    def genetic_optimization(self,
                             data: pd.DataFrame,
                             population_size: int = 100,
                             generations: int = 50,
                             mutation_rate: float = 0.1) -> Dict:
        """Optimize strategies using genetic algorithm"""

        # Initialize population
        self.population = [
            self.generate_random_strategy()
            for _ in range(population_size)
        ]

        for generation in range(generations):
            logger.info(f"Generation {generation + 1}/{generations}")

            # Evaluate fitness
            fitness_scores = []
            for strategy in self.population:
                metrics = self.evaluate_strategy(strategy, data)
                fitness_scores.append(metrics['fitness'])

            # Select best strategies
            sorted_indices = np.argsort(fitness_scores)[::-1]
            elite_size = population_size // 4
            elite = [self.population[i] for i in sorted_indices[:elite_size]]

            # Create new population
            new_population = elite.copy()

            while len(new_population) < population_size:
                # Crossover
                parent1 = random.choice(elite)
                parent2 = random.choice(elite)
                child = self.crossover(parent1, parent2)

                # Mutation
                if random.random() < mutation_rate:
                    child = self.mutate(child)

                new_population.append(child)

            self.population = new_population
            self.generation = generation + 1

            # Track best
            best_idx = sorted_indices[0]
            self.best_strategy = self.population[best_idx]

            logger.info(f"Best fitness: {fitness_scores[best_idx]:.4f}")

        return self.best_strategy

    def crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Crossover two strategies to create offspring"""

        child = {}

        for key in parent1.keys():
            if random.random() > 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]

        return child

    def mutate(self, strategy: Dict) -> Dict:
        """Mutate a strategy"""

        mutated = copy.deepcopy(strategy)

        # Randomly mutate one aspect
        mutation_type = random.choice(['indicators', 'thresholds', 'exits', 'sizing'])

        if mutation_type == 'indicators':
            # Change indicators
            available = ['rsi', 'macd', 'bb_position', 'sma_cross', 'volume_spike']
            mutated['entry_indicators'] = random.sample(available, k=random.randint(2, 4))

        elif mutation_type == 'thresholds':
            # Adjust thresholds
            for key in mutated['entry_thresholds']:
                if isinstance(mutated['entry_thresholds'][key], (int, float)):
                    mutated['entry_thresholds'][key] *= random.uniform(0.8, 1.2)

        elif mutation_type == 'exits':
            # Adjust exit parameters
            mutated['stop_loss'] *= random.uniform(0.8, 1.2)
            mutated['take_profit'] *= random.uniform(0.8, 1.2)

        elif mutation_type == 'sizing':
            # Change position sizing
            mutated['max_position_size'] *= random.uniform(0.8, 1.2)
            mutated['max_position_size'] = min(0.25, max(0.05, mutated['max_position_size']))

        return mutated

    def generate_signals(self, strategy: Dict, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals from strategy"""

        # Simplified signal generation
        signals = pd.Series(0, index=data.index)

        # Apply entry conditions
        for indicator in strategy['entry_indicators']:
            if indicator == 'rsi' and 'rsi_14' in data.columns:
                threshold = strategy['entry_thresholds'].get('rsi', 30)
                if threshold < 50:
                    signals[data['rsi_14'] < threshold] = 1
                else:
                    signals[data['rsi_14'] > threshold] = 1

        return signals

    def calculate_returns(self, signals: pd.Series, data: pd.DataFrame) -> pd.Series:
        """Calculate returns from signals"""

        # Simple return calculation
        positions = signals.shift(1)  # Enter next day
        returns = positions * data['returns']

        return returns

    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""

        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        return abs(drawdown.min())