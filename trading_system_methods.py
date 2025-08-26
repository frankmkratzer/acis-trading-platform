#!/usr/bin/env python3
"""
Additional methods for TradingSystem class
These methods provide the missing functionality for the paper trading test system
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from trading_system import TradingSystem, Order, OrderSide, Position, Account, OrderStatus, OrderType, TradingMode

# Additional methods to be added to TradingSystem class
def get_account(self, account_id: str) -> Account:
    """Get account information"""
    try:
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT cash_balance, total_equity, buying_power, day_trading_buying_power
                FROM trading_accounts
                WHERE account_id = :account_id AND trading_mode = :trading_mode
            """), {
                'account_id': account_id,
                'trading_mode': self.trading_mode.value
            })
            
            account_data = result.fetchone()
            
            if not account_data:
                # Create default account if it doesn't exist
                return self.create_paper_trading_account(account_id)
            
            # Get positions
            positions = self.get_current_positions(account_id)
            
            # Get orders
            orders = self.get_current_orders(account_id)
            
            return Account(
                account_id=account_id,
                cash_balance=float(account_data[0]),
                total_equity=float(account_data[1]),
                buying_power=float(account_data[2]),
                day_trading_buying_power=float(account_data[3]),
                positions=positions,
                orders=orders,
                last_updated=datetime.now()
            )
            
    except Exception as e:
        self.logger.error(f"Failed to get account: {e}")
        return self.create_paper_trading_account(account_id)

def get_current_positions(self, account_id: str = "default") -> list:
    """Get current positions"""
    positions = []
    
    try:
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT symbol, quantity, avg_cost, market_value, unrealized_pnl, 
                       realized_pnl, strategy, updated_at
                FROM trading_positions
                WHERE account_id = :account_id AND trading_mode = :trading_mode
                  AND quantity > 0
                ORDER BY symbol
            """), {
                'account_id': account_id,
                'trading_mode': self.trading_mode.value
            })
            
            for row in result:
                # Update market value with current price
                current_price = self._get_current_price(row[0])
                current_market_value = row[1] * current_price
                current_unrealized_pnl = current_market_value - (row[1] * row[2])
                
                position = Position(
                    symbol=row[0],
                    quantity=row[1],
                    avg_cost=float(row[2]),
                    market_value=current_market_value,
                    unrealized_pnl=current_unrealized_pnl,
                    realized_pnl=float(row[5]),
                    strategy=row[6],
                    last_updated=row[7]
                )
                positions.append(position)
                
    except Exception as e:
        self.logger.error(f"Failed to get positions: {e}")
    
    return positions

def get_current_orders(self, account_id: str = "default") -> list:
    """Get current open orders"""
    orders = []
    
    try:
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT client_order_id, symbol, quantity, side, order_type, price,
                       stop_price, status, filled_quantity, avg_fill_price, commission,
                       strategy, created_at, filled_at
                FROM trading_orders
                WHERE trading_mode = :trading_mode
                  AND status IN ('pending', 'partially_filled')
                ORDER BY created_at DESC
            """), {
                'trading_mode': self.trading_mode.value
            })
            
            for row in result:
                order = Order(
                    symbol=row[1],
                    quantity=row[2],
                    side=OrderSide(row[3]),
                    order_type=OrderType(row[4]),
                    price=float(row[5]) if row[5] else None,
                    stop_price=float(row[6]) if row[6] else None,
                    strategy=row[11],
                    client_order_id=row[0],
                    status=OrderStatus(row[7]),
                    filled_quantity=row[8],
                    avg_fill_price=float(row[9]),
                    commission=float(row[10]),
                    created_at=row[12],
                    filled_at=row[13]
                )
                orders.append(order)
                
    except Exception as e:
        self.logger.error(f"Failed to get orders: {e}")
    
    return orders

def analyze_portfolio(self, strategy: str) -> dict:
    """Analyze portfolio performance and allocation"""
    positions = [p for p in self.get_current_positions() if p.strategy == strategy]
    
    if not positions:
        return {
            "total_positions": 0,
            "total_market_value": 0,
            "sector_allocation": {},
            "unrealized_pnl": 0,
            "realized_pnl": 0
        }
    
    total_market_value = sum(p.market_value for p in positions)
    total_unrealized_pnl = sum(p.unrealized_pnl for p in positions)
    total_realized_pnl = sum(p.realized_pnl for p in positions)
    
    # Simple sector allocation (would need sector mapping in production)
    sector_allocation = {}
    for position in positions:
        # Placeholder sector mapping
        sector = self._get_sector(position.symbol)
        weight = position.market_value / total_market_value
        if sector in sector_allocation:
            sector_allocation[sector] += weight
        else:
            sector_allocation[sector] = weight
    
    return {
        "total_positions": len(positions),
        "total_market_value": total_market_value,
        "sector_allocation": sector_allocation,
        "unrealized_pnl": total_unrealized_pnl,
        "realized_pnl": total_realized_pnl,
        "total_pnl": total_unrealized_pnl + total_realized_pnl
    }

def _get_sector(self, symbol: str) -> str:
    """Get sector for symbol (placeholder implementation)"""
    tech_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META"]
    finance_stocks = ["BRK.B", "V"]
    healthcare_stocks = ["JNJ"]
    
    if symbol in tech_stocks:
        return "Technology"
    elif symbol in finance_stocks:
        return "Financial Services"
    elif symbol in healthcare_stocks:
        return "Healthcare"
    else:
        return "Other"

def generate_rebalance_orders(self, strategy: str, target_allocation: dict) -> list:
    """Generate rebalancing orders for a strategy"""
    current_positions = {p.symbol: p for p in self.get_current_positions() if p.strategy == strategy}
    total_portfolio_value = sum(p.market_value for p in current_positions.values())
    
    if total_portfolio_value == 0:
        total_portfolio_value = 100000  # Default for new portfolios
    
    rebalance_orders = []
    
    for symbol, target_weight in target_allocation.items():
        target_value = total_portfolio_value * target_weight
        current_position = current_positions.get(symbol)
        current_value = current_position.market_value if current_position else 0
        
        value_difference = target_value - current_value
        
        if abs(value_difference) > 100:  # Only rebalance if difference > $100
            current_price = self._get_current_price(symbol)
            quantity_change = int(value_difference / current_price)
            
            if quantity_change > 0:
                order = Order(
                    symbol=symbol,
                    quantity=abs(quantity_change),
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    strategy=strategy,
                    client_order_id=f"REBAL_{strategy}_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                rebalance_orders.append(order)
                
            elif quantity_change < 0:
                order = Order(
                    symbol=symbol,
                    quantity=abs(quantity_change),
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    strategy=strategy,
                    client_order_id=f"REBAL_{strategy}_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                rebalance_orders.append(order)
    
    return rebalance_orders

def get_sector_allocation(self) -> dict:
    """Get current sector allocation across all positions"""
    positions = self.get_current_positions()
    total_value = sum(p.market_value for p in positions)
    
    if total_value == 0:
        return {}
    
    sector_allocation = {}
    for position in positions:
        sector = self._get_sector(position.symbol)
        weight = position.market_value / total_value
        if sector in sector_allocation:
            sector_allocation[sector] += weight
        else:
            sector_allocation[sector] = weight
    
    return sector_allocation

def calculate_daily_pnl(self) -> float:
    """Calculate daily P&L"""
    # Placeholder implementation
    positions = self.get_current_positions()
    return sum(p.unrealized_pnl for p in positions) * 0.01  # Simulate 1% daily movement

def get_account_value(self) -> float:
    """Get total account value"""
    account = self.get_account("default")
    return account.total_equity

def calculate_strategy_performance(self, strategy: str) -> dict:
    """Calculate performance metrics for a strategy"""
    try:
        with self.engine.connect() as conn:
            # Get all trades for this strategy
            result = conn.execute(text("""
                SELECT symbol, quantity, avg_fill_price, side, filled_at, commission
                FROM trading_orders
                WHERE strategy = :strategy AND status = 'filled'
                  AND trading_mode = :trading_mode
                ORDER BY filled_at
            """), {
                'strategy': strategy,
                'trading_mode': self.trading_mode.value
            })
            
            trades = result.fetchall()
            
            if not trades:
                return {
                    "total_return": 0,
                    "sharpe_ratio": 0,
                    "max_drawdown": 0,
                    "total_trades": 0
                }
            
            # Calculate basic performance metrics
            total_invested = 0
            total_returns = 0
            total_commissions = sum(float(trade[5]) for trade in trades)
            
            # Group trades by symbol to calculate returns
            symbol_trades = {}
            for trade in trades:
                symbol = trade[0]
                if symbol not in symbol_trades:
                    symbol_trades[symbol] = []
                symbol_trades[symbol].append(trade)
            
            # Calculate returns for each symbol
            for symbol, symbol_trade_list in symbol_trades.items():
                buys = [t for t in symbol_trade_list if t[3] == 'buy']
                sells = [t for t in symbol_trade_list if t[3] == 'sell']
                
                total_bought = sum(t[1] * t[2] for t in buys)  # quantity * price
                total_sold = sum(t[1] * t[2] for t in sells)
                
                total_invested += total_bought
                total_returns += total_sold
            
            # Calculate return percentage
            if total_invested > 0:
                total_return = (total_returns - total_invested - total_commissions) / total_invested
            else:
                total_return = 0
            
            # Placeholder for more sophisticated metrics
            return {
                "total_return": total_return,
                "sharpe_ratio": max(total_return * 2, 0),  # Simplified Sharpe
                "max_drawdown": abs(total_return) * 0.3,   # Estimated max drawdown
                "total_trades": len(trades),
                "total_commissions": total_commissions
            }
            
    except Exception as e:
        self.logger.error(f"Failed to calculate strategy performance: {e}")
        return {
            "total_return": 0,
            "sharpe_ratio": 0,
            "max_drawdown": 0,
            "total_trades": 0
        }

def generate_trading_report(self, start_date: datetime, end_date: datetime, strategy: str = "all") -> dict:
    """Generate comprehensive trading report"""
    try:
        account = self.get_account("default")
        positions = self.get_current_positions()
        
        # Filter positions by strategy if specified
        if strategy != "all":
            positions = [p for p in positions if p.strategy == strategy]
        
        # Calculate portfolio metrics
        total_market_value = sum(p.market_value for p in positions)
        total_unrealized_pnl = sum(p.unrealized_pnl for p in positions)
        total_realized_pnl = sum(p.realized_pnl for p in positions)
        
        # Get order history
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT client_order_id, symbol, quantity, side, order_type, 
                       avg_fill_price, commission, strategy, filled_at
                FROM trading_orders
                WHERE filled_at BETWEEN :start_date AND :end_date
                  AND status = 'filled'
                  AND trading_mode = :trading_mode
                ORDER BY filled_at DESC
            """), {
                'start_date': start_date,
                'end_date': end_date,
                'trading_mode': self.trading_mode.value
            })
            
            order_history = [dict(row._mapping) for row in result]
        
        # Calculate performance metrics
        if strategy == "all":
            # Aggregate performance across all strategies
            performance_metrics = {
                "total_return": (total_unrealized_pnl + total_realized_pnl) / account.total_equity,
                "sharpe_ratio": 0.8,  # Placeholder
                "max_drawdown": 0.05,  # Placeholder
            }
        else:
            performance_metrics = self.calculate_strategy_performance(strategy)
        
        return {
            "account_summary": {
                "total_equity": account.total_equity,
                "cash_balance": account.cash_balance,
                "buying_power": account.buying_power,
                "total_positions": len(positions)
            },
            "position_summary": [
                {
                    "symbol": p.symbol,
                    "quantity": p.quantity,
                    "avg_cost": p.avg_cost,
                    "market_value": p.market_value,
                    "unrealized_pnl": p.unrealized_pnl,
                    "strategy": p.strategy
                } for p in positions
            ],
            "order_history": order_history,
            "performance_metrics": performance_metrics,
            "risk_analysis": {
                "sector_allocation": self.get_sector_allocation(),
                "largest_position": max([p.market_value / total_market_value for p in positions]) if positions else 0,
                "cash_ratio": account.cash_balance / account.total_equity
            }
        }
        
    except Exception as e:
        self.logger.error(f"Failed to generate trading report: {e}")
        return {"error": str(e)}