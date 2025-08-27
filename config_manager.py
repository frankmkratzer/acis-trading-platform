# =====================================
# 5. Enhanced Configuration System
# =====================================
"""
#!/usr/bin/env python3
# File: config_manager.py
# Purpose: Centralized configuration management
"""

import yaml
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from logging_config import setup_logger

logger = setup_logger("config_manager")


@dataclass
class StrategyConfig:
    name: str
    enabled: bool = True
    top_k: int = 20
    min_market_cap: float = 1e9
    max_market_cap: Optional[float] = None
    rebalance_frequency: str = 'monthly'

    # Factors and weights
    factors: Dict[str, float] = None

    # Filters
    filters: Dict[str, any] = None

    # Risk limits
    max_position_size: float = 0.10
    max_sector_weight: float = 0.30
    min_positions: int = 15
    max_positions: int = 30

    def __post_init__(self):
        if self.factors is None:
            self.factors = {}
        if self.filters is None:
            self.filters = {}


class ConfigManager:
    def __init__(self, config_path='config/strategies.yaml'):
        self.config_path = config_path
        self.strategies = {}
        self.load_config()

    def load_config(self):
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            logger.info(f"Config file not found, creating default: {self.config_path}")
            self.create_default_config()

        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)

            for name, settings in config.get('strategies', {}).items():
                self.strategies[name] = StrategyConfig(name=name, **settings)
            
            logger.info(f"Loaded {len(self.strategies)} strategies from {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            raise

    def create_default_config(self):
        """Create default configuration file"""
        default_config = {
            'strategies': {
                'value': {
                    'enabled': True,
                    'top_k': 20,
                    'min_market_cap': 1e9,
                    'rebalance_frequency': 'monthly',
                    'factors': {
                        'earnings_yield': 0.25,
                        'fcf_yield': 0.35,
                        'book_to_market': 0.20,
                        'dividend_yield': 0.20
                    },
                    'filters': {
                        'min_roe': 0.10,
                        'max_debt_to_equity': 2.0,
                        'positive_fcf': True
                    }
                },
                'growth': {
                    'enabled': True,
                    'top_k': 20,
                    'min_market_cap': 1e9,
                    'rebalance_frequency': 'quarterly',
                    'factors': {
                        'revenue_growth': 0.30,
                        'earnings_growth': 0.40,
                        'fcf_growth': 0.30
                    },
                    'filters': {
                        'min_revenue_growth': 0.10,
                        'min_gross_margin': 0.30
                    }
                },
                'momentum': {
                    'enabled': True,
                    'top_k': 20,
                    'min_market_cap': 1e9,
                    'rebalance_frequency': 'monthly',
                    'factors': {
                        'return_1m': 0.20,
                        'return_3m': 0.30,
                        'return_6m': 0.30,
                        'return_12m': 0.20
                    },
                    'filters': {
                        'min_volume': 1000000,
                        'positive_6m_return': True
                    }
                },
                'dividend': {
                    'enabled': True,
                    'top_k': 20,
                    'min_market_cap': 5e9,
                    'rebalance_frequency': 'quarterly',
                    'factors': {
                        'dividend_yield': 0.30,
                        'dividend_growth': 0.40,
                        'payout_ratio': 0.30
                    },
                    'filters': {
                        'min_dividend_yield': 0.02,
                        'max_payout_ratio': 0.80,
                        'consecutive_years': 5
                    }
                }
            }
        }

        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)

    def get_strategy(self, name: str) -> Optional[StrategyConfig]:
        """Get configuration for a specific strategy"""
        return self.strategies.get(name)

    def get_enabled_strategies(self) -> List[StrategyConfig]:
        """Get all enabled strategies"""
        return [s for s in self.strategies.values() if s.enabled]

    def save_config(self):
        """Save current configuration to file"""
        config = {
            'strategies': {
                name: asdict(strategy)
                for name, strategy in self.strategies.items()
            }
        }

        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)