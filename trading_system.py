#!/usr/bin/env python3
"""
ACIS Trading System - Paper Trading & Live Trading Integration
Comprehensive trading infrastructure with risk management and execution
Supports both simulation and live trading with multiple brokers
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class TradingMode(Enum):
    PAPER = "paper"
    LIVE = "live"

@dataclass
class Order:
    """Order data structure"""
    symbol: str
    quantity: int
    side: OrderSide
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    strategy: str = ""
    client_order_id: str = ""
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    created_at: datetime = None
    filled_at: Optional[datetime] = None

@dataclass 
class Position:
    """Position data structure"""
    symbol: str
    quantity: int
    avg_cost: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float = 0.0
    strategy: str = ""
    last_updated: datetime = None

@dataclass
class Account:
    """Account data structure"""
    account_id: str
    cash_balance: float
    total_equity: float
    buying_power: float
    day_trading_buying_power: float
    positions: List[Position]
    orders: List[Order]
    last_updated: datetime = None

class TradingSystem:
    def __init__(self, trading_mode: TradingMode = TradingMode.PAPER):
        load_dotenv()
        self.engine = create_engine(os.getenv('POSTGRES_URL'))
        self.trading_mode = trading_mode
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('TradingSystem')
        
        # Trading configuration
        self.config = {
            "paper_trading": {
                "initial_cash": 1000000,  # $1M for paper trading
                "commission_per_share": 0.005,  # $0.005 per share
                "min_commission": 1.0,
                "slippage_bps": 5,  # 5 basis points slippage simulation
                "market_impact_threshold": 10000  # Market impact above $10K orders
            },
            
            "risk_management": {
                "max_position_size": 0.05,  # 5% max position size
                "max_sector_allocation": 0.30,  # 30% max sector allocation
                "max_daily_loss": 0.02,  # 2% max daily loss
                "max_drawdown": 0.15,  # 15% max drawdown
                "position_concentration_limit": 0.20  # 20% max single position
            },
            
            "execution_settings": {
                "market_hours_only": True,
                "price_improvement_enabled": True,
                "smart_routing": True,
                "dark_pool_enabled": False
            }
        }
        
        # Broker integrations (for live trading)
        self.broker_configs = {
            "alpaca": {
                "base_url": "https://paper-api.alpaca.markets",  # Paper trading URL
                "api_key": os.getenv('ALPACA_API_KEY'),
                "secret_key": os.getenv('ALPACA_SECRET_KEY'),
                "supported_orders": ["market", "limit", "stop", "stop_limit"],
                "commission_free": True
            },
            
            "interactive_brokers": {
                "gateway_host": "127.0.0.1",
                "gateway_port": 7497,  # Paper trading port
                "client_id": 1,
                "supported_orders": ["market", "limit", "stop", "stop_limit", "trailing_stop"],
                "commission_structure": "tiered"
            },
            
            "td_ameritrade": {
                "base_url": "https://api.tdameritrade.com",
                "client_id": os.getenv('TD_CLIENT_ID'),
                "refresh_token": os.getenv('TD_REFRESH_TOKEN'),
                "commission_free": True
            }
        }
        
        # Initialize database tables
        self._init_trading_tables()
    
    def _init_trading_tables(self):
        """Initialize trading database tables"""
        with self.engine.connect() as conn:
            # Orders table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS trading_orders (
                    id SERIAL PRIMARY KEY,
                    client_order_id VARCHAR(50) UNIQUE,
                    broker_order_id VARCHAR(50),
                    symbol VARCHAR(10),
                    quantity INTEGER,
                    side VARCHAR(10),
                    order_type VARCHAR(20),
                    price DECIMAL(10,4),
                    stop_price DECIMAL(10,4),
                    status VARCHAR(20),
                    filled_quantity INTEGER DEFAULT 0,
                    avg_fill_price DECIMAL(10,4),
                    commission DECIMAL(10,4),
                    strategy VARCHAR(50),
                    trading_mode VARCHAR(10),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    filled_at TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            # Positions table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS trading_positions (
                    id SERIAL PRIMARY KEY,
                    account_id VARCHAR(50),
                    symbol VARCHAR(10),
                    quantity INTEGER,
                    avg_cost DECIMAL(10,4),
                    market_value DECIMAL(12,2),
                    unrealized_pnl DECIMAL(12,2),
                    realized_pnl DECIMAL(12,2) DEFAULT 0,
                    strategy VARCHAR(50),
                    trading_mode VARCHAR(10),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(account_id, symbol, trading_mode)
                )
            """))
            
            # Account balances table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS trading_accounts (
                    id SERIAL PRIMARY KEY,
                    account_id VARCHAR(50) UNIQUE,
                    cash_balance DECIMAL(12,2),
                    total_equity DECIMAL(12,2),
                    buying_power DECIMAL(12,2),
                    day_trading_buying_power DECIMAL(12,2),
                    trading_mode VARCHAR(10),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            # Trades/executions table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS trading_executions (
                    id SERIAL PRIMARY KEY,
                    client_order_id VARCHAR(50),
                    symbol VARCHAR(10),
                    quantity INTEGER,
                    price DECIMAL(10,4),
                    side VARCHAR(10),
                    commission DECIMAL(10,4),
                    strategy VARCHAR(50),
                    execution_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            conn.commit()
            self.logger.info("Trading database tables initialized")
    
    def create_paper_trading_account(self, account_id: str, initial_cash: float = None) -> Account:
        """Create a new paper trading account"""
        
        if initial_cash is None:
            initial_cash = self.config["paper_trading"]["initial_cash"]
        
        account = Account(
            account_id=account_id,
            cash_balance=initial_cash,
            total_equity=initial_cash,
            buying_power=initial_cash,
            day_trading_buying_power=initial_cash * 4,  # 4:1 day trading leverage
            positions=[],
            orders=[],
            last_updated=datetime.now()
        )
        
        # Save to database
        with self.engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO trading_accounts (
                    account_id, cash_balance, total_equity, buying_power, 
                    day_trading_buying_power, trading_mode
                ) VALUES (
                    :account_id, :cash_balance, :total_equity, :buying_power,
                    :day_trading_buying_power, :trading_mode
                )
                ON CONFLICT (account_id) DO UPDATE SET
                    cash_balance = EXCLUDED.cash_balance,
                    total_equity = EXCLUDED.total_equity,
                    buying_power = EXCLUDED.buying_power,
                    day_trading_buying_power = EXCLUDED.day_trading_buying_power,
                    updated_at = CURRENT_TIMESTAMP
            """), {
                'account_id': account_id,
                'cash_balance': initial_cash,
                'total_equity': initial_cash,
                'buying_power': initial_cash,
                'day_trading_buying_power': initial_cash * 4,
                'trading_mode': self.trading_mode.value
            })
            conn.commit()
        
        self.logger.info(f"Created paper trading account {account_id} with ${initial_cash:,.2f}")
        return account
    
    def submit_order(self, order: Order) -> str:
        """Submit an order (paper or live trading)"""
        
        # Generate client order ID if not provided
        if not order.client_order_id:
            order.client_order_id = f"{order.strategy}_{order.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Risk management checks
        if not self._validate_order(order):
            raise ValueError("Order failed risk management validation")
        
        # Set creation time
        if order.created_at is None:
            order.created_at = datetime.now()
        
        # Execute based on trading mode
        if self.trading_mode == TradingMode.PAPER:
            return self._execute_paper_order(order)
        else:
            return self._execute_live_order(order)
    
    def create_order(self, order: Order) -> dict:
        """Create and validate order"""
        try:
            # Validate order parameters
            if order.quantity <= 0:
                return {"success": False, "error": "Invalid quantity"}
            
            if order.order_type == OrderType.LIMIT and order.price <= 0:
                return {"success": False, "error": "Invalid limit price"}
                
            # Generate client order ID if not provided
            if not order.client_order_id:
                order.client_order_id = f"{order.strategy}_{order.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            return {"success": True, "order": order}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def execute_order(self, order: Order) -> dict:
        """Execute order (paper or live trading)"""
        try:
            if self.trading_mode == TradingMode.PAPER:
                return self._execute_paper_order(order)
            else:
                return self._execute_live_order(order)
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _validate_order(self, order: Order) -> bool:
        """Validate order against risk management rules"""
        
        try:
            # Get current account and positions
            account = self.get_account("default")
            
            # Check position size limit
            order_value = order.quantity * (order.price or self._get_current_price(order.symbol))
            position_size_pct = order_value / account.total_equity
            
            if position_size_pct > self.config["risk_management"]["max_position_size"]:
                self.logger.warning(f"Order exceeds max position size: {position_size_pct:.2%}")
                return False
            
            # Check buying power
            if order.side == OrderSide.BUY:
                required_buying_power = order_value * 1.05  # 5% buffer
                if required_buying_power > account.buying_power:
                    self.logger.warning(f"Insufficient buying power: need ${required_buying_power:,.2f}, have ${account.buying_power:,.2f}")
                    return False
            
            # Check if we have the position to sell
            if order.side == OrderSide.SELL:
                current_position = next((p for p in account.positions if p.symbol == order.symbol), None)
                if not current_position or current_position.quantity < order.quantity:
                    self.logger.warning(f"Insufficient position to sell: need {order.quantity}, have {current_position.quantity if current_position else 0}")
                    return False
            
            self.logger.info(f"Order validation passed for {order.symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Order validation error: {e}")
            return False
    
    def _execute_paper_order(self, order: Order) -> str:
        """Execute order in paper trading simulation"""
        
        # Save order to database
        order_id = self._save_order_to_db(order)
        
        # Simulate market execution
        if order.order_type == OrderType.MARKET:
            # Immediate fill for market orders
            fill_price = self._get_current_price(order.symbol)
            
            # Apply slippage
            slippage = self.config["paper_trading"]["slippage_bps"] / 10000
            if order.side == OrderSide.BUY:
                fill_price *= (1 + slippage)
            else:
                fill_price *= (1 - slippage)
            
            # Calculate commission
            commission = max(
                order.quantity * self.config["paper_trading"]["commission_per_share"],
                self.config["paper_trading"]["min_commission"]
            )
            
            # Fill the order
            self._fill_order(order, order.quantity, fill_price, commission)
            
            self.logger.info(f"Paper order filled: {order.symbol} {order.quantity} shares at ${fill_price:.4f}")
            
        else:
            # Limit orders would need price monitoring - simplified for now
            self.logger.info(f"Paper limit order submitted: {order.symbol} {order.quantity} shares at ${order.price:.4f}")
        
        return order.client_order_id
    
    def _execute_live_order(self, order: Order) -> str:
        """Execute order via live broker integration"""
        
        # This would integrate with actual broker APIs
        # For now, we'll log the order and return
        
        broker = "alpaca"  # Default broker
        
        if broker == "alpaca":
            return self._execute_alpaca_order(order)
        elif broker == "interactive_brokers":
            return self._execute_ib_order(order)
        elif broker == "td_ameritrade":
            return self._execute_td_order(order)
        else:
            raise ValueError(f"Unsupported broker: {broker}")
    
    def _execute_alpaca_order(self, order: Order) -> str:
        """Execute order via Alpaca API"""
        
        # Placeholder for Alpaca integration
        # In real implementation, this would use alpaca-trade-api
        
        self.logger.info(f"Alpaca order submitted: {order.symbol} {order.quantity} shares")
        return order.client_order_id
    
    def _fill_order(self, order: Order, filled_qty: int, fill_price: float, commission: float):
        """Process order fill"""
        
        order.status = OrderStatus.FILLED
        order.filled_quantity = filled_qty
        order.avg_fill_price = fill_price
        order.commission = commission
        order.filled_at = datetime.now()
        
        # Update positions
        self._update_position(order)
        
        # Update account balance
        self._update_account_balance(order)
        
        # Save execution to database
        self._save_execution_to_db(order)
        
        # Update order in database
        self._update_order_in_db(order)
    
    def _update_position(self, order: Order):
        """Update position after order fill"""
        
        with self.engine.connect() as conn:
            if order.side == OrderSide.BUY:
                # Add to position
                conn.execute(text("""
                    INSERT INTO trading_positions (
                        account_id, symbol, quantity, avg_cost, market_value, 
                        unrealized_pnl, strategy, trading_mode
                    ) VALUES (
                        'default', :symbol, :quantity, :avg_cost, :market_value,
                        0, :strategy, :trading_mode
                    )
                    ON CONFLICT (account_id, symbol, trading_mode) DO UPDATE SET
                        quantity = trading_positions.quantity + EXCLUDED.quantity,
                        avg_cost = (trading_positions.avg_cost * trading_positions.quantity + EXCLUDED.avg_cost * EXCLUDED.quantity) 
                                  / (trading_positions.quantity + EXCLUDED.quantity),
                        market_value = EXCLUDED.market_value,
                        updated_at = CURRENT_TIMESTAMP
                """), {
                    'symbol': order.symbol,
                    'quantity': order.filled_quantity,
                    'avg_cost': order.avg_fill_price,
                    'market_value': order.filled_quantity * order.avg_fill_price,
                    'strategy': order.strategy,
                    'trading_mode': self.trading_mode.value
                })
            else:
                # Reduce position
                conn.execute(text("""
                    UPDATE trading_positions 
                    SET quantity = quantity - :quantity,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE account_id = 'default' 
                      AND symbol = :symbol 
                      AND trading_mode = :trading_mode
                """), {
                    'quantity': order.filled_quantity,
                    'symbol': order.symbol,
                    'trading_mode': self.trading_mode.value
                })
            
            conn.commit()
    
    def _update_account_balance(self, order: Order):
        """Update account balance after order fill"""
        
        trade_value = order.filled_quantity * order.avg_fill_price
        
        with self.engine.connect() as conn:
            if order.side == OrderSide.BUY:
                # Decrease cash balance
                conn.execute(text("""
                    UPDATE trading_accounts
                    SET cash_balance = cash_balance - :trade_value - :commission,
                        buying_power = buying_power - :trade_value - :commission,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE account_id = 'default'
                """), {
                    'trade_value': trade_value,
                    'commission': order.commission
                })
            else:
                # Increase cash balance
                conn.execute(text("""
                    UPDATE trading_accounts
                    SET cash_balance = cash_balance + :trade_value - :commission,
                        buying_power = buying_power + :trade_value - :commission,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE account_id = 'default'
                """), {
                    'trade_value': trade_value,
                    'commission': order.commission
                })
            
            conn.commit()
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current market price for symbol"""
        
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT adjusted_close
                FROM stock_eod_daily
                WHERE symbol = :symbol
                  AND trade_date = (SELECT MAX(trade_date) FROM stock_eod_daily WHERE symbol = :symbol)
            """), {'symbol': symbol})
            
            row = result.fetchone()
            return float(row[0]) if row else 100.0  # Default price if not found
    
    def _save_order_to_db(self, order: Order) -> int:
        """Save order to database"""
        
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                INSERT INTO trading_orders (
                    client_order_id, symbol, quantity, side, order_type,
                    price, stop_price, status, strategy, trading_mode
                ) VALUES (
                    :client_order_id, :symbol, :quantity, :side, :order_type,
                    :price, :stop_price, :status, :strategy, :trading_mode
                )
                RETURNING id
            """), {
                'client_order_id': order.client_order_id,
                'symbol': order.symbol,
                'quantity': order.quantity,
                'side': order.side.value,
                'order_type': order.order_type.value,
                'price': order.price,
                'stop_price': order.stop_price,
                'status': order.status.value,
                'strategy': order.strategy,
                'trading_mode': self.trading_mode.value
            })
            
            order_id = result.fetchone()[0]
            conn.commit()
            return order_id
    
    def _update_order_in_db(self, order: Order):
        """Update order status in database"""
        
        with self.engine.connect() as conn:
            conn.execute(text("""
                UPDATE trading_orders
                SET status = :status,
                    filled_quantity = :filled_quantity,
                    avg_fill_price = :avg_fill_price,
                    commission = :commission,
                    filled_at = :filled_at,
                    updated_at = CURRENT_TIMESTAMP
                WHERE client_order_id = :client_order_id
            """), {
                'status': order.status.value,
                'filled_quantity': order.filled_quantity,
                'avg_fill_price': order.avg_fill_price,
                'commission': order.commission,
                'filled_at': order.filled_at,
                'client_order_id': order.client_order_id
            })
            conn.commit()
    
    def _save_execution_to_db(self, order: Order):
        """Save execution details to database"""
        
        with self.engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO trading_executions (
                    order_id, symbol, quantity, price, side, commission, strategy
                ) 
                SELECT id, :symbol, :quantity, :price, :side, :commission, :strategy
                FROM trading_orders 
                WHERE client_order_id = :client_order_id
            """), {
                'symbol': order.symbol,
                'quantity': order.filled_quantity,
                'price': order.avg_fill_price,
                'side': order.side.value,
                'commission': order.commission,
                'strategy': order.strategy,
                'client_order_id': order.client_order_id
            })
            conn.commit()
    
    def get_account(self, account_id: str = "default") -> Account:
        """Get account details with current positions"""
        
        with self.engine.connect() as conn:
            # Get account info
            account_result = conn.execute(text("""
                SELECT cash_balance, total_equity, buying_power, day_trading_buying_power
                FROM trading_accounts
                WHERE account_id = :account_id AND trading_mode = :trading_mode
            """), {
                'account_id': account_id,
                'trading_mode': self.trading_mode.value
            })
            
            account_row = account_result.fetchone()
            if not account_row:
                # Create default paper trading account
                return self.create_paper_trading_account(account_id)
            
            # Get positions
            positions_result = conn.execute(text("""
                SELECT symbol, quantity, avg_cost, market_value, unrealized_pnl, realized_pnl, strategy
                FROM trading_positions
                WHERE account_id = :account_id AND trading_mode = :trading_mode AND quantity > 0
            """), {
                'account_id': account_id,
                'trading_mode': self.trading_mode.value
            })
            
            positions = []
            for pos_row in positions_result:
                position = Position(
                    symbol=pos_row[0],
                    quantity=pos_row[1],
                    avg_cost=float(pos_row[2]),
                    market_value=float(pos_row[3]),
                    unrealized_pnl=float(pos_row[4]),
                    realized_pnl=float(pos_row[5]),
                    strategy=pos_row[6],
                    last_updated=datetime.now()
                )
                positions.append(position)
            
            # Get open orders
            orders_result = conn.execute(text("""
                SELECT client_order_id, symbol, quantity, side, order_type, price, 
                       stop_price, status, filled_quantity, avg_fill_price, commission, strategy
                FROM trading_orders
                WHERE status IN ('pending', 'partially_filled') AND trading_mode = :trading_mode
                ORDER BY created_at DESC
            """), {'trading_mode': self.trading_mode.value})
            
            orders = []
            for order_row in orders_result:
                order = Order(
                    client_order_id=order_row[0],
                    symbol=order_row[1],
                    quantity=order_row[2],
                    side=OrderSide(order_row[3]),
                    order_type=OrderType(order_row[4]),
                    price=float(order_row[5]) if order_row[5] else None,
                    stop_price=float(order_row[6]) if order_row[6] else None,
                    status=OrderStatus(order_row[7]),
                    filled_quantity=order_row[8],
                    avg_fill_price=float(order_row[9]) if order_row[9] else 0.0,
                    commission=float(order_row[10]) if order_row[10] else 0.0,
                    strategy=order_row[11]
                )
                orders.append(order)
            
            return Account(
                account_id=account_id,
                cash_balance=float(account_row[0]),
                total_equity=float(account_row[1]),
                buying_power=float(account_row[2]),
                day_trading_buying_power=float(account_row[3]),
                positions=positions,
                orders=orders,
                last_updated=datetime.now()
            )
    
    def execute_strategy_rebalancing(self, strategy_name: str, target_positions: List[Dict]):
        """Execute rebalancing for a strategy"""
        
        self.logger.info(f"Starting rebalancing for strategy: {strategy_name}")
        
        # Get current positions for this strategy
        current_positions = {pos.symbol: pos for pos in self.get_account().positions 
                           if pos.strategy == strategy_name}
        
        # Generate orders for rebalancing
        orders_to_execute = []
        
        for target in target_positions:
            symbol = target['symbol']
            target_quantity = target['quantity']
            current_quantity = current_positions.get(symbol, Position(symbol, 0, 0, 0, 0)).quantity
            
            quantity_diff = target_quantity - current_quantity
            
            if quantity_diff > 0:
                # Need to buy
                order = Order(
                    symbol=symbol,
                    quantity=quantity_diff,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    strategy=strategy_name
                )
                orders_to_execute.append(order)
                
            elif quantity_diff < 0:
                # Need to sell
                order = Order(
                    symbol=symbol,
                    quantity=abs(quantity_diff),
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    strategy=strategy_name
                )
                orders_to_execute.append(order)
        
        # Execute orders
        executed_orders = []
        for order in orders_to_execute:
            try:
                order_id = self.submit_order(order)
                executed_orders.append(order_id)
                self.logger.info(f"Executed rebalancing order: {order.symbol} {order.side.value} {order.quantity}")
            except Exception as e:
                self.logger.error(f"Failed to execute order for {order.symbol}: {e}")
        
        self.logger.info(f"Rebalancing complete: {len(executed_orders)} orders executed")
        return executed_orders
    
    def get_performance_summary(self, account_id: str = "default") -> Dict:
        """Get performance summary for account"""
        
        account = self.get_account(account_id)
        
        with self.engine.connect() as conn:
            # Get trade history
            result = conn.execute(text("""
                SELECT 
                    SUM(CASE WHEN side = 'buy' THEN -(quantity * price + commission)
                             ELSE quantity * price - commission END) as realized_pnl,
                    COUNT(*) as total_trades,
                    MIN(execution_time) as first_trade_date
                FROM trading_executions
                WHERE strategy IS NOT NULL
            """))
            
            trade_stats = result.fetchone()
            realized_pnl = float(trade_stats[0]) if trade_stats[0] else 0
            total_trades = trade_stats[1] if trade_stats[1] else 0
            
            # Calculate unrealized P&L
            unrealized_pnl = sum(pos.unrealized_pnl for pos in account.positions)
            
            # Calculate total return
            initial_equity = self.config["paper_trading"]["initial_cash"]
            total_return_pct = ((account.total_equity - initial_equity) / initial_equity) * 100
        
        return {
            "account_id": account_id,
            "trading_mode": self.trading_mode.value,
            "current_equity": account.total_equity,
            "cash_balance": account.cash_balance,
            "positions_count": len(account.positions),
            "total_return_pct": round(total_return_pct, 2),
            "realized_pnl": round(realized_pnl, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "total_pnl": round(realized_pnl + unrealized_pnl, 2),
            "total_trades": total_trades,
            "last_updated": datetime.now().isoformat()
        }

def main():
    """Test the trading system"""
    
    # Initialize paper trading system
    trading_system = TradingSystem(TradingMode.PAPER)
    
    # Create paper trading account
    account = trading_system.create_paper_trading_account("test_account", 100000)
    print(f"Created account with ${account.cash_balance:,.2f}")
    
    # Test order execution
    test_order = Order(
        symbol="AAPL",
        quantity=100,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        strategy="test_strategy"
    )
    
    try:
        order_id = trading_system.submit_order(test_order)
        print(f"Order submitted: {order_id}")
        
        # Get updated account
        updated_account = trading_system.get_account("test_account")
        print(f"Updated cash balance: ${updated_account.cash_balance:,.2f}")
        print(f"Positions: {len(updated_account.positions)}")
        
        # Performance summary
        performance = trading_system.get_performance_summary("test_account")
        print(f"Performance: {json.dumps(performance, indent=2)}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()