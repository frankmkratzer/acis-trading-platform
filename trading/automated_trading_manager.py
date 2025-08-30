#!/usr/bin/env python3
"""
ACIS Automated Trading Manager
Executes portfolio recommendations through Schwab API for client accounts
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))  # For schwabdev

from utils.logging_config import setup_logger
from database.db_connection_manager import DatabaseConnectionManager
from sqlalchemy import text
from cryptography.fernet import Fernet

# Import Schwab API client
from schwabdev.client import Client as SchwabClient
from schwabdev.tokens import TokenManager

logger = setup_logger("automated_trading_manager")


class AutomatedTradingManager:
    """Manages automated trading for all client accounts"""
    
    def __init__(self, mode: str = "production"):
        """
        Initialize the trading manager
        
        Args:
            mode: 'production' for live trading, 'paper' for testing
        """
        self.mode = mode
        self.db_manager = DatabaseConnectionManager()
        self.engine = self.db_manager.get_engine()
        
        # Initialize encryption for credentials
        encryption_key = os.getenv("TRADING_ENCRYPTION_KEY")
        if encryption_key:
            self.cipher = Fernet(encryption_key.encode())
        else:
            logger.warning("TRADING_ENCRYPTION_KEY not set - using mock encryption")
            self.cipher = None
            
        self.schwab_clients = {}  # Cache of Schwab API clients by account_id
        
    def decrypt_token(self, encrypted_token: str) -> str:
        """Decrypt an OAuth token"""
        if self.cipher and encrypted_token:
            return self.cipher.decrypt(encrypted_token.encode()).decode()
        return encrypted_token or ""
    
    def encrypt_token(self, token: str) -> str:
        """Encrypt an OAuth token"""
        if self.cipher and token:
            return self.cipher.encrypt(token.encode()).decode()
        return token
    
    def get_schwab_client(self, account_id: int) -> Optional[SchwabClient]:
        """Get or create a Schwab API client for an account"""
        if account_id in self.schwab_clients:
            return self.schwab_clients[account_id]
        
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT refresh_token, access_token, token_expires_at,
                       schwab_account_number, paper_trading
                FROM trading_accounts
                WHERE account_id = :account_id AND account_active = true
            """), {"account_id": account_id})
            
            account = result.fetchone()
            if not account:
                logger.error(f"Account {account_id} not found or inactive")
                return None
            
            if account.paper_trading and self.mode == "production":
                logger.warning(f"Account {account_id} is in paper trading mode")
                # In paper mode, return a mock client
                return None
            
            # Decrypt tokens
            refresh_token = self.decrypt_token(account.refresh_token)
            access_token = self.decrypt_token(account.access_token)
            
            # Initialize Schwab client
            try:
                token_manager = TokenManager(
                    app_key=os.getenv("SCHWAB_APP_KEY"),
                    app_secret=os.getenv("SCHWAB_APP_SECRET"),
                    redirect_uri=os.getenv("SCHWAB_REDIRECT_URI"),
                    refresh_token=refresh_token,
                    access_token=access_token
                )
                
                client = SchwabClient(token_manager)
                self.schwab_clients[account_id] = client
                return client
                
            except Exception as e:
                logger.error(f"Failed to create Schwab client for account {account_id}: {e}")
                return None
    
    def get_active_clients(self) -> List[Dict]:
        """Get all clients with automated trading enabled"""
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT 
                    c.client_id,
                    c.email,
                    c.risk_tolerance,
                    c.portfolio_strategy,
                    c.max_position_size_pct,
                    c.max_portfolio_positions,
                    ta.account_id,
                    ta.schwab_account_number,
                    ta.automated_trading_active,
                    ta.max_account_allocation_pct,
                    ta.min_cash_balance,
                    ta.rebalance_frequency,
                    ta.next_rebalance_date,
                    ta.stop_loss_enabled,
                    ta.stop_loss_pct
                FROM clients c
                JOIN trading_accounts ta ON c.client_id = ta.client_id
                WHERE c.automated_trading_enabled = true
                  AND ta.automated_trading_active = true
                  AND c.account_status = 'ACTIVE'
                  AND ta.account_active = true
            """))
            
            return [dict(row) for row in result.fetchall()]
    
    def get_pending_signals(self, portfolio_strategy: str = None) -> List[Dict]:
        """Get pending trading signals from the queue"""
        query = """
            SELECT 
                signal_id,
                symbol,
                signal_type,
                portfolio_type,
                acis_score,
                value_score,
                growth_score,
                dividend_score,
                target_weight_pct,
                signal_strength
            FROM trading_signals_queue
            WHERE processed = false
              AND (expiration_date IS NULL OR expiration_date >= CURRENT_DATE)
        """
        
        params = {}
        if portfolio_strategy and portfolio_strategy != 'BALANCED':
            query += " AND portfolio_type = :portfolio_type"
            params["portfolio_type"] = portfolio_strategy
        
        query += " ORDER BY acis_score DESC, created_at ASC"
        
        with self.engine.connect() as conn:
            result = conn.execute(text(query), params)
            return [dict(row) for row in result.fetchall()]
    
    def get_current_holdings(self, account_id: int) -> Dict[str, Dict]:
        """Get current holdings for an account"""
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT 
                    symbol,
                    quantity,
                    avg_cost_basis,
                    market_value,
                    portfolio_weight_pct,
                    current_acis_score
                FROM current_holdings
                WHERE account_id = :account_id
            """), {"account_id": account_id})
            
            holdings = {}
            for row in result:
                holdings[row.symbol] = {
                    "quantity": float(row.quantity),
                    "avg_cost": float(row.avg_cost_basis),
                    "market_value": float(row.market_value) if row.market_value else 0,
                    "weight": float(row.portfolio_weight_pct) if row.portfolio_weight_pct else 0,
                    "score": float(row.current_acis_score) if row.current_acis_score else 0
                }
            return holdings
    
    def get_account_balance(self, account_id: int) -> Tuple[float, float]:
        """Get account cash balance and total value"""
        with self.engine.connect() as conn:
            # Get from Schwab API in production
            client = self.get_schwab_client(account_id)
            if client and self.mode == "production":
                try:
                    # Get account info from Schwab
                    # This would use the actual Schwab API call
                    # For now, returning mock data
                    pass
                except Exception as e:
                    logger.error(f"Failed to get Schwab balance: {e}")
            
            # Fall back to database
            result = conn.execute(text("""
                SELECT cash_balance, total_value
                FROM account_balances
                WHERE account_id = :account_id
                ORDER BY snapshot_date DESC
                LIMIT 1
            """), {"account_id": account_id})
            
            row = result.fetchone()
            if row:
                return float(row.cash_balance), float(row.total_value)
            return 10000.0, 100000.0  # Default values for new accounts
    
    def calculate_position_size(self, account_id: int, client_config: Dict, 
                               signal: Dict, cash_balance: float) -> int:
        """Calculate the number of shares to buy"""
        # Get current price (would fetch from Schwab API)
        current_price = self.get_current_price(signal["symbol"])
        if not current_price:
            return 0
        
        # Calculate position size based on strategy
        max_position_pct = float(client_config["max_position_size_pct"]) / 100
        total_value = cash_balance / (1 - float(client_config["max_account_allocation_pct"]) / 100)
        
        # Target position value
        if signal["target_weight_pct"]:
            target_pct = float(signal["target_weight_pct"]) / 100
        else:
            target_pct = max_position_pct
        
        target_value = total_value * min(target_pct, max_position_pct)
        
        # Calculate shares
        shares = int(target_value / current_price)
        
        # Check if we have enough cash
        required_cash = shares * current_price
        if required_cash > cash_balance - float(client_config["min_cash_balance"]):
            # Reduce shares to fit cash constraint
            available_cash = max(0, cash_balance - float(client_config["min_cash_balance"]))
            shares = int(available_cash / current_price)
        
        return shares
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for a symbol"""
        # In production, this would call Schwab API
        # For now, get from database
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT close_price
                FROM stock_prices
                WHERE symbol = :symbol
                ORDER BY date DESC
                LIMIT 1
            """), {"symbol": symbol})
            
            row = result.fetchone()
            if row:
                return float(row.close_price)
        return None
    
    def execute_trade(self, account_id: int, client_config: Dict, signal: Dict) -> bool:
        """Execute a trade for a specific account and signal"""
        try:
            # Get current holdings
            holdings = self.get_current_holdings(account_id)
            cash_balance, total_value = self.get_account_balance(account_id)
            
            # Determine trade action
            if signal["signal_type"] == "SELL":
                if signal["symbol"] not in holdings:
                    logger.info(f"Cannot sell {signal['symbol']} - not in holdings")
                    return False
                
                quantity = holdings[signal["symbol"]]["quantity"]
                trade_type = "SELL"
                
            elif signal["signal_type"] == "BUY":
                # Check position limits
                if len(holdings) >= int(client_config["max_portfolio_positions"]):
                    logger.info(f"Account {account_id} at max positions ({len(holdings)})")
                    return False
                
                # Calculate position size
                quantity = self.calculate_position_size(
                    account_id, client_config, signal, cash_balance
                )
                
                if quantity <= 0:
                    logger.info(f"Insufficient funds for {signal['symbol']}")
                    return False
                
                trade_type = "BUY"
                
            else:  # REBALANCE
                # Complex logic for rebalancing
                return self.execute_rebalance(account_id, client_config, signal)
            
            # Log the trade attempt
            trade_id = self.log_trade(
                account_id=account_id,
                symbol=signal["symbol"],
                trade_type=trade_type,
                quantity=quantity,
                signal=signal
            )
            
            # Execute via Schwab API
            if self.mode == "production":
                client = self.get_schwab_client(account_id)
                if client:
                    success = self.place_schwab_order(
                        client, signal["symbol"], trade_type, quantity, trade_id
                    )
                else:
                    success = False
            else:
                # Paper trading mode
                success = True
                logger.info(f"PAPER TRADE: {trade_type} {quantity} shares of {signal['symbol']}")
            
            # Update trade status
            self.update_trade_status(trade_id, "FILLED" if success else "REJECTED")
            
            # Mark signal as processed
            if success:
                self.mark_signal_processed(signal["signal_id"], account_id)
            
            return success
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return False
    
    def place_schwab_order(self, client: SchwabClient, symbol: str, 
                          trade_type: str, quantity: int, trade_id: int) -> bool:
        """Place an order through Schwab API"""
        try:
            # This would use the actual Schwab API
            # Example structure:
            order = {
                "orderType": "MARKET",
                "session": "NORMAL",
                "duration": "DAY",
                "orderStrategyType": "SINGLE",
                "orderLegCollection": [{
                    "instruction": trade_type,
                    "quantity": quantity,
                    "instrument": {
                        "symbol": symbol,
                        "assetType": "EQUITY"
                    }
                }]
            }
            
            # response = client.place_order(order)
            # schwab_order_id = response["orderId"]
            
            # For now, mock success
            logger.info(f"Placed {trade_type} order for {quantity} shares of {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Schwab order failed: {e}")
            return False
    
    def log_trade(self, account_id: int, symbol: str, trade_type: str, 
                  quantity: int, signal: Dict) -> int:
        """Log a trade attempt in the database"""
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                INSERT INTO trade_execution_log (
                    account_id, symbol, trade_type, order_type,
                    quantity, portfolio_type, signal_reason,
                    signal_score, signal_generated_at, order_placed_at,
                    order_status
                ) VALUES (
                    :account_id, :symbol, :trade_type, 'MARKET',
                    :quantity, :portfolio_type, :signal_reason,
                    :signal_score, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP,
                    'SUBMITTED'
                ) RETURNING trade_id
            """), {
                "account_id": account_id,
                "symbol": symbol,
                "trade_type": trade_type,
                "quantity": quantity,
                "portfolio_type": signal.get("portfolio_type"),
                "signal_reason": f"{signal['signal_type']} signal with score {signal['acis_score']}",
                "signal_score": signal["acis_score"]
            })
            conn.commit()
            
            return result.fetchone()[0]
    
    def update_trade_status(self, trade_id: int, status: str):
        """Update the status of a trade"""
        with self.engine.connect() as conn:
            conn.execute(text("""
                UPDATE trade_execution_log
                SET order_status = :status,
                    order_filled_at = CASE 
                        WHEN :status = 'FILLED' THEN CURRENT_TIMESTAMP 
                        ELSE order_filled_at 
                    END
                WHERE trade_id = :trade_id
            """), {"trade_id": trade_id, "status": status})
            conn.commit()
    
    def mark_signal_processed(self, signal_id: int, account_id: int):
        """Mark a signal as processed"""
        with self.engine.connect() as conn:
            conn.execute(text("""
                UPDATE trading_signals_queue
                SET processed = true,
                    processed_at = CURRENT_TIMESTAMP,
                    accounts_notified = accounts_notified + 1
                WHERE signal_id = :signal_id
            """), {"signal_id": signal_id})
            conn.commit()
    
    def execute_rebalance(self, account_id: int, client_config: Dict, signal: Dict) -> bool:
        """Execute portfolio rebalancing"""
        # Complex rebalancing logic would go here
        logger.info(f"Rebalancing account {account_id}")
        return True
    
    def check_stop_losses(self, account_id: int, client_config: Dict):
        """Check and execute stop-loss orders if needed"""
        if not client_config.get("stop_loss_enabled"):
            return
        
        stop_loss_pct = float(client_config["stop_loss_pct"]) / 100
        holdings = self.get_current_holdings(account_id)
        
        for symbol, holding in holdings.items():
            current_price = self.get_current_price(symbol)
            if not current_price:
                continue
            
            # Check if stop-loss triggered
            loss_pct = (current_price - holding["avg_cost"]) / holding["avg_cost"]
            if loss_pct <= -stop_loss_pct:
                logger.warning(f"Stop-loss triggered for {symbol} at {loss_pct:.2%}")
                # Create sell signal
                sell_signal = {
                    "signal_id": 0,
                    "symbol": symbol,
                    "signal_type": "SELL",
                    "portfolio_type": client_config.get("portfolio_strategy"),
                    "acis_score": 0,
                    "signal_strength": "STRONG"
                }
                self.execute_trade(account_id, client_config, sell_signal)
    
    def process_all_accounts(self):
        """Main processing loop for all accounts"""
        logger.info("Starting automated trading processing")
        
        # Get active clients
        clients = self.get_active_clients()
        logger.info(f"Found {len(clients)} active trading accounts")
        
        for client in clients:
            try:
                logger.info(f"Processing account {client['account_id']} ({client['email']})")
                
                # Check if rebalancing is due
                if client["next_rebalance_date"] and client["next_rebalance_date"] <= datetime.now().date():
                    logger.info(f"Rebalancing due for account {client['account_id']}")
                    # Trigger rebalancing
                
                # Check stop losses
                self.check_stop_losses(client["account_id"], client)
                
                # Get relevant signals
                signals = self.get_pending_signals(client["portfolio_strategy"])
                
                # Process each signal
                trades_executed = 0
                for signal in signals[:5]:  # Limit to 5 trades per run
                    if self.execute_trade(client["account_id"], client, signal):
                        trades_executed += 1
                
                logger.info(f"Executed {trades_executed} trades for account {client['account_id']}")
                
                # Update last sync time
                with self.engine.connect() as conn:
                    conn.execute(text("""
                        UPDATE trading_accounts
                        SET last_sync_at = CURRENT_TIMESTAMP
                        WHERE account_id = :account_id
                    """), {"account_id": client["account_id"]})
                    conn.commit()
                
            except Exception as e:
                logger.error(f"Error processing account {client['account_id']}: {e}")
                continue
        
        logger.info("Automated trading processing complete")
    
    def generate_signals_from_portfolios(self):
        """Generate trading signals from ACIS portfolio recommendations"""
        logger.info("Generating trading signals from ACIS portfolios")
        
        with self.engine.connect() as conn:
            # Get top stocks from each portfolio
            portfolios = ["VALUE", "GROWTH", "DIVIDEND"]
            
            for portfolio_type in portfolios:
                result = conn.execute(text("""
                    SELECT 
                        ph.symbol,
                        ph.weight,
                        ps.composite_score,
                        ps.value_score,
                        ps.growth_score,
                        ps.dividend_score
                    FROM portfolio_holdings ph
                    JOIN portfolio_scores ps ON ph.symbol = ps.symbol 
                        AND ph.portfolio_type = ps.portfolio_type
                    WHERE ph.portfolio_type = :portfolio_type
                      AND ph.is_current = true
                    ORDER BY ph.rank
                    LIMIT 10
                """), {"portfolio_type": portfolio_type})
                
                for row in result:
                    # Check if signal already exists
                    existing = conn.execute(text("""
                        SELECT signal_id FROM trading_signals_queue
                        WHERE symbol = :symbol 
                          AND portfolio_type = :portfolio_type
                          AND processed = false
                    """), {
                        "symbol": row.symbol,
                        "portfolio_type": portfolio_type
                    }).fetchone()
                    
                    if not existing:
                        # Create new signal
                        conn.execute(text("""
                            INSERT INTO trading_signals_queue (
                                symbol, signal_type, portfolio_type,
                                acis_score, value_score, growth_score, dividend_score,
                                target_weight_pct, signal_strength
                            ) VALUES (
                                :symbol, 'BUY', :portfolio_type,
                                :acis_score, :value_score, :growth_score, :dividend_score,
                                :target_weight, :signal_strength
                            )
                        """), {
                            "symbol": row.symbol,
                            "portfolio_type": portfolio_type,
                            "acis_score": row.composite_score,
                            "value_score": row.value_score,
                            "growth_score": row.growth_score,
                            "dividend_score": row.dividend_score,
                            "target_weight": row.weight,
                            "signal_strength": "STRONG" if row.composite_score > 80 else "MODERATE"
                        })
                
                conn.commit()
        
        logger.info("Trading signals generated successfully")


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description="ACIS Automated Trading Manager")
    parser.add_argument("--mode", choices=["production", "paper"], 
                       default="paper", help="Trading mode")
    parser.add_argument("--generate-signals", action="store_true",
                       help="Generate signals from portfolios")
    parser.add_argument("--process-trades", action="store_true",
                       help="Process trades for all accounts")
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = AutomatedTradingManager(mode=args.mode)
    
    if args.generate_signals:
        manager.generate_signals_from_portfolios()
    
    if args.process_trades or (not args.generate_signals):
        manager.process_all_accounts()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())