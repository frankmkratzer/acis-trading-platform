# !/usr/bin/env python3
"""
Live Trading Engine with Schwab Integration
Production-ready live trading system with multiple broker support including Charles Schwab
"""

import os
import time
import threading
import queue
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import logging
import json
import asyncio
import websocket
import requests
from urllib.parse import urlencode

# Import broker APIs
try:
    import alpaca_trade_api as tradeapi  # Alpaca
    from ib_insync import IB, Stock, MarketOrder, LimitOrder  # Interactive Brokers
    import robin_stocks as rs  # Robinhood
except ImportError:
    pass

load_dotenv()
engine = create_engine(os.getenv("POSTGRES_URL"))

logging.basicConfig(
    filename="live_trading.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ===== Order Types and States =====
class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    MARKET_ON_CLOSE = "market_on_close"
    LIMIT_ON_CLOSE = "limit_on_close"


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    ACCEPTED = "accepted"
    WORKING = "working"
    PENDING_CANCEL = "pending_cancel"
    PENDING_REPLACE = "pending_replace"
    REPLACED = "replaced"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"
    BUY_TO_COVER = "buy_to_cover"
    SELL_SHORT = "sell_short"


@dataclass
class Order:
    """Order representation"""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "day"  # day, gtc, ioc, fok
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    commission: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class Position:
    """Position tracking"""
    symbol: str
    quantity: int
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    last_updated: datetime
    maintenance_requirement: float = 0.0


# ===== Broker Interface =====
class BrokerInterface:
    """Abstract broker interface"""

    def connect(self) -> bool:
        raise NotImplementedError

    def disconnect(self):
        raise NotImplementedError

    def submit_order(self, order: Order) -> str:
        raise NotImplementedError

    def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError

    def get_order_status(self, order_id: str) -> OrderStatus:
        raise NotImplementedError

    def get_positions(self) -> List[Position]:
        raise NotImplementedError

    def get_account_info(self) -> Dict:
        raise NotImplementedError

    def get_market_data(self, symbol: str) -> Dict:
        raise NotImplementedError

    def replace_order(self, order_id: str, new_order: Order) -> str:
        """Replace an existing order (optional)"""
        # Default: cancel and submit new
        if self.cancel_order(order_id):
            return self.submit_order(new_order)
        return None


# ===== Schwab Authentication Manager =====
class SchwabAuthManager:
    """Manage Schwab OAuth 2.0 authentication"""

    def __init__(self):
        self.client_id = os.getenv("SCHWAB_CLIENT_ID")
        self.client_secret = os.getenv("SCHWAB_CLIENT_SECRET")
        self.redirect_uri = os.getenv("SCHWAB_REDIRECT_URI", "http://localhost:8080/callback")
        self.account_id = os.getenv("SCHWAB_ACCOUNT_ID")

        self.access_token = None
        self.refresh_token = None
        self.token_expiry = None

        # Schwab API endpoints
        self.base_url = 'https://api.schwabapi.com/trader/v1'
        self.auth_url = 'https://api.schwabapi.com/v1/oauth/authorize'
        self.token_url = 'https://api.schwabapi.com/v1/oauth/token'

        # Load saved tokens if they exist
        self.load_tokens()

    def get_authorization_url(self) -> str:
        """Generate authorization URL for initial OAuth flow"""
        params = {
            'response_type': 'code',
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
            'scope': 'read write trading'
        }
        return f"{self.auth_url}?{urlencode(params)}"

    def exchange_code_for_tokens(self, authorization_code: str) -> bool:
        """Exchange authorization code for access and refresh tokens"""
        data = {
            'grant_type': 'authorization_code',
            'code': authorization_code,
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'redirect_uri': self.redirect_uri
        }

        try:
            response = requests.post(self.token_url, data=data)
            response.raise_for_status()

            token_data = response.json()
            self.access_token = token_data['access_token']
            self.refresh_token = token_data['refresh_token']
            self.token_expiry = datetime.now() + timedelta(seconds=token_data['expires_in'])

            self.save_tokens()
            logger.info("Successfully obtained Schwab tokens")
            return True

        except Exception as e:
            logger.error(f"Failed to exchange code for tokens: {e}")
            return False

    def refresh_access_token(self) -> bool:
        """Refresh the access token using refresh token"""
        if not self.refresh_token:
            logger.error("No refresh token available")
            return False

        data = {
            'grant_type': 'refresh_token',
            'refresh_token': self.refresh_token,
            'client_id': self.client_id,
            'client_secret': self.client_secret
        }

        try:
            response = requests.post(self.token_url, data=data)
            response.raise_for_status()

            token_data = response.json()
            self.access_token = token_data['access_token']
            if 'refresh_token' in token_data:
                self.refresh_token = token_data['refresh_token']
            self.token_expiry = datetime.now() + timedelta(seconds=token_data['expires_in'])

            self.save_tokens()
            logger.info("Successfully refreshed Schwab access token")
            return True

        except Exception as e:
            logger.error(f"Failed to refresh token: {e}")
            return False

    def get_valid_token(self) -> Optional[str]:
        """Get a valid access token, refreshing if necessary"""
        if self.access_token and self.token_expiry:
            # Check if token is still valid (with 5 minute buffer)
            if datetime.now() < (self.token_expiry - timedelta(minutes=5)):
                return self.access_token

        # Try to refresh
        if self.refresh_access_token():
            return self.access_token

        return None

    def save_tokens(self):
        """Save tokens to secure storage"""
        token_data = {
            'access_token': self.access_token,
            'refresh_token': self.refresh_token,
            'token_expiry': self.token_expiry.isoformat() if self.token_expiry else None
        }

        # In production, use secure storage (e.g., AWS Secrets Manager)
        token_file = os.path.expanduser("~/.schwab_tokens.json")
        with open(token_file, 'w') as f:
            json.dump(token_data, f)
        os.chmod(token_file, 0o600)  # Restrict permissions

    def load_tokens(self):
        """Load saved tokens"""
        token_file = os.path.expanduser("~/.schwab_tokens.json")

        if os.path.exists(token_file):
            try:
                with open(token_file, 'r') as f:
                    token_data = json.load(f)

                self.access_token = token_data.get('access_token')
                self.refresh_token = token_data.get('refresh_token')

                if token_data.get('token_expiry'):
                    self.token_expiry = datetime.fromisoformat(token_data['token_expiry'])

                logger.info("Loaded saved Schwab tokens")
            except Exception as e:
                logger.error(f"Failed to load tokens: {e}")


# ===== Schwab Broker Implementation =====
class SchwabBroker(BrokerInterface):
    """Charles Schwab broker implementation"""

    def __init__(self):
        self.auth_manager = SchwabAuthManager()
        self.account_id = os.getenv("SCHWAB_ACCOUNT_ID")
        self.session = requests.Session()
        self.positions_cache = {}
        self.orders_cache = {}
        self.last_cache_update = None
        self.cache_ttl = 5  # seconds

    def _get_headers(self) -> Dict:
        """Get request headers with authentication"""
        token = self.auth_manager.get_valid_token()
        if not token:
            raise Exception("Failed to get valid Schwab token")

        return {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

    def _make_request(self, method: str, endpoint: str, data: Dict = None) -> Dict:
        """Make authenticated request to Schwab API"""
        url = f"{self.auth_manager.base_url}{endpoint}"
        headers = self._get_headers()

        try:
            if method == 'GET':
                response = self.session.get(url, headers=headers, params=data)
            elif method == 'POST':
                response = self.session.post(url, headers=headers, json=data)
            elif method == 'PUT':
                response = self.session.put(url, headers=headers, json=data)
            elif method == 'DELETE':
                response = self.session.delete(url, headers=headers)
            else:
                raise ValueError(f"Unsupported method: {method}")

            response.raise_for_status()

            if response.text:
                return response.json()
            return {}

        except requests.exceptions.HTTPError as e:
            logger.error(f"Schwab API error: {e}")
            if e.response:
                logger.error(f"Response: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise

    def connect(self) -> bool:
        """Connect to Schwab and verify authentication"""
        try:
            # Test connection by getting account info
            accounts = self._make_request('GET', '/accounts')
            if accounts:
                if not self.account_id and isinstance(accounts, list) and len(accounts) > 0:
                    # Auto-detect account ID
                    self.account_id = accounts[0].get('securitiesAccount', {}).get('accountId')
                logger.info(f"Connected to Schwab. Account ID: {self.account_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Schwab: {e}")
            return False

    def disconnect(self):
        """Disconnect from Schwab"""
        self.session.close()
        logger.info("Disconnected from Schwab")

    def submit_order(self, order: Order) -> str:
        """Submit order to Schwab"""
        # Map generic order to Schwab format
        schwab_order = {
            "orderType": self._map_order_type(order.order_type),
            "session": "NORMAL",
            "duration": self._map_time_in_force(order.time_in_force),
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [
                {
                    "instruction": self._map_order_side(order.side),
                    "quantity": order.quantity,
                    "instrument": {
                        "symbol": order.symbol,
                        "assetType": "EQUITY"
                    }
                }
            ]
        }

        # Add price for limit orders
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            schwab_order["price"] = order.limit_price

        # Add stop price for stop orders
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            schwab_order["stopPrice"] = order.stop_price

        endpoint = f"/accounts/{self.account_id}/orders"

        try:
            response = self.session.post(
                f"{self.auth_manager.base_url}{endpoint}",
                headers=self._get_headers(),
                json=schwab_order
            )
            response.raise_for_status()

            # Extract order ID from Location header
            if 'Location' in response.headers:
                order_id = response.headers['Location'].split('/')[-1]
            else:
                order_id = str(datetime.now().timestamp())

            order.order_id = order_id
            order.status = OrderStatus.SUBMITTED
            order.submitted_at = datetime.now()

            logger.info(f"Order submitted to Schwab: {order_id}")
            return order_id

        except Exception as e:
            logger.error(f"Failed to submit order to Schwab: {e}")
            order.status = OrderStatus.REJECTED
            raise

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        endpoint = f"/accounts/{self.account_id}/orders/{order_id}"

        try:
            self._make_request('DELETE', endpoint)
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def replace_order(self, order_id: str, new_order: Order) -> str:
        """Replace an existing order"""
        # Create Schwab order format
        schwab_order = {
            "orderType": self._map_order_type(new_order.order_type),
            "session": "NORMAL",
            "duration": self._map_time_in_force(new_order.time_in_force),
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [
                {
                    "instruction": self._map_order_side(new_order.side),
                    "quantity": new_order.quantity,
                    "instrument": {
                        "symbol": new_order.symbol,
                        "assetType": "EQUITY"
                    }
                }
            ]
        }

        if new_order.limit_price:
            schwab_order["price"] = new_order.limit_price
        if new_order.stop_price:
            schwab_order["stopPrice"] = new_order.stop_price

        endpoint = f"/accounts/{self.account_id}/orders/{order_id}"

        try:
            response = self.session.put(
                f"{self.auth_manager.base_url}{endpoint}",
                headers=self._get_headers(),
                json=schwab_order
            )
            response.raise_for_status()

            # Extract new order ID from Location header
            if 'Location' in response.headers:
                new_order_id = response.headers['Location'].split('/')[-1]
                logger.info(f"Order replaced: {order_id} -> {new_order_id}")
                return new_order_id

            return order_id

        except Exception as e:
            logger.error(f"Failed to replace order {order_id}: {e}")
            raise

    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status"""
        endpoint = f"/accounts/{self.account_id}/orders/{order_id}"

        try:
            order = self._make_request('GET', endpoint)
            schwab_status = order.get('status', 'PENDING')

            # Map Schwab status to generic status
            status_map = {
                'AWAITING_PARENT_ORDER': OrderStatus.PENDING,
                'AWAITING_CONDITION': OrderStatus.PENDING,
                'AWAITING_MANUAL_REVIEW': OrderStatus.PENDING,
                'ACCEPTED': OrderStatus.ACCEPTED,
                'PENDING_ACTIVATION': OrderStatus.PENDING,
                'QUEUED': OrderStatus.PENDING,
                'WORKING': OrderStatus.WORKING,
                'REJECTED': OrderStatus.REJECTED,
                'PENDING_CANCEL': OrderStatus.PENDING_CANCEL,
                'CANCELED': OrderStatus.CANCELLED,
                'PENDING_REPLACE': OrderStatus.PENDING_REPLACE,
                'REPLACED': OrderStatus.REPLACED,
                'FILLED': OrderStatus.FILLED,
                'EXPIRED': OrderStatus.EXPIRED
            }

            return status_map.get(schwab_status, OrderStatus.PENDING)

        except Exception as e:
            logger.error(f"Failed to get order status: {e}")
            return OrderStatus.PENDING

    def get_positions(self) -> List[Position]:
        """Get current positions"""
        # Check cache
        if self._is_cache_valid():
            return list(self.positions_cache.values())

        endpoint = f"/accounts/{self.account_id}"
        params = {'fields': 'positions'}

        try:
            account_data = self._make_request('GET', endpoint, params)
            positions = []

            if 'securitiesAccount' in account_data:
                for pos in account_data['securitiesAccount'].get('positions', []):
                    instrument = pos.get('instrument', {})

                    position = Position(
                        symbol=instrument.get('symbol'),
                        quantity=int(pos.get('longQuantity', 0) - pos.get('shortQuantity', 0)),
                        avg_cost=float(pos.get('averagePrice', 0)),
                        current_price=float(pos.get('marketValue', 0)) / max(float(pos.get('longQuantity', 1)), 1),
                        market_value=float(pos.get('marketValue', 0)),
                        unrealized_pnl=float(pos.get('unrealizedPNL', 0)),
                        realized_pnl=float(pos.get('realizedPNL', 0)),
                        maintenance_requirement=float(pos.get('maintenanceRequirement', 0)),
                        last_updated=datetime.now()
                    )

                    positions.append(position)
                    self.positions_cache[position.symbol] = position

            self.last_cache_update = datetime.now()
            return positions

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    def get_account_info(self) -> Dict:
        """Get account information"""
        endpoint = f"/accounts/{self.account_id}"
        params = {'fields': 'positions,orders'}

        try:
            account_data = self._make_request('GET', endpoint, params)

            if 'securitiesAccount' in account_data:
                account = account_data['securitiesAccount']
                balances = account.get('currentBalances', {})

                return {
                    'account_id': account.get('accountId'),
                    'buying_power': float(balances.get('buyingPower', 0)),
                    'cash': float(balances.get('cashBalance', 0)),
                    'portfolio_value': float(balances.get('liquidationValue', 0)),
                    'day_trading_buying_power': float(balances.get('dayTradingBuyingPower', 0)),
                    'maintenance_requirement': float(balances.get('maintenanceRequirement', 0)),
                    'available_funds': float(balances.get('availableFunds', 0)),
                    'pattern_day_trader': account.get('isDayTrader', False),
                    'trading_blocked': False,  # Schwab doesn't provide this directly
                    'transfers_blocked': False,
                    'account_blocked': False
                }

            return {}

        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return {}

    def get_market_data(self, symbol: str) -> Dict:
        """Get real-time market data"""
        endpoint = "/marketdata/v1/quotes"
        params = {'symbols': symbol}

        try:
            response = self._make_request('GET', endpoint, params)

            if symbol in response:
                quote = response[symbol]
                return {
                    'bid': float(quote.get('bidPrice', 0)),
                    'ask': float(quote.get('askPrice', 0)),
                    'last': float(quote.get('lastPrice', 0)),
                    'volume': int(quote.get('totalVolume', 0)),
                    'high': float(quote.get('highPrice', 0)),
                    'low': float(quote.get('lowPrice', 0)),
                    'close': float(quote.get('closePrice', 0)),
                    'timestamp': datetime.now()
                }

            return {}

        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            return {}

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid"""
        if not self.last_cache_update:
            return False

        age = (datetime.now() - self.last_cache_update).total_seconds()
        return age < self.cache_ttl

    def _map_order_type(self, order_type: OrderType) -> str:
        """Map generic order type to Schwab order type"""
        mapping = {
            OrderType.MARKET: "MARKET",
            OrderType.LIMIT: "LIMIT",
            OrderType.STOP: "STOP",
            OrderType.STOP_LIMIT: "STOP_LIMIT",
            OrderType.TRAILING_STOP: "TRAILING_STOP",
            OrderType.MARKET_ON_CLOSE: "MARKET_ON_CLOSE",
            OrderType.LIMIT_ON_CLOSE: "LIMIT_ON_CLOSE"
        }
        return mapping.get(order_type, "MARKET")

    def _map_order_side(self, side: OrderSide) -> str:
        """Map generic order side to Schwab instruction"""
        mapping = {
            OrderSide.BUY: "BUY",
            OrderSide.SELL: "SELL",
            OrderSide.BUY_TO_COVER: "BUY_TO_COVER",
            OrderSide.SELL_SHORT: "SELL_SHORT"
        }
        return mapping.get(side, "BUY")

    def _map_time_in_force(self, tif: str) -> str:
        """Map time in force to Schwab duration"""
        mapping = {
            'day': 'DAY',
            'gtc': 'GOOD_TILL_CANCEL',
            'ioc': 'IMMEDIATE_OR_CANCEL',
            'fok': 'FILL_OR_KILL'
        }
        return mapping.get(tif.lower(), 'DAY')


# ===== Alpaca Broker Implementation (unchanged) =====
class AlpacaBroker(BrokerInterface):
    """Alpaca broker implementation"""

    def __init__(self):
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")
        self.base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        self.api = None

    def connect(self) -> bool:
        """Connect to Alpaca"""
        try:
            self.api = tradeapi.REST(
                self.api_key,
                self.secret_key,
                self.base_url,
                api_version='v2'
            )
            account = self.api.get_account()
            logger.info(f"Connected to Alpaca. Account status: {account.status}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            return False

    def submit_order(self, order: Order) -> str:
        """Submit order to Alpaca"""
        try:
            alpaca_order = self.api.submit_order(
                symbol=order.symbol,
                qty=order.quantity,
                side=order.side.value,
                type=order.order_type.value,
                time_in_force=order.time_in_force,
                limit_price=order.limit_price,
                stop_price=order.stop_price
            )

            order.order_id = alpaca_order.id
            order.status = OrderStatus.SUBMITTED
            order.submitted_at = datetime.now()

            logger.info(f"Order submitted to Alpaca: {order.order_id}")
            return alpaca_order.id

        except Exception as e:
            logger.error(f"Failed to submit order to Alpaca: {e}")
            order.status = OrderStatus.REJECTED
            raise

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            self.api.cancel_order(order_id)
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status"""
        try:
            order = self.api.get_order(order_id)
            status_map = {
                'pending_new': OrderStatus.PENDING,
                'accepted': OrderStatus.SUBMITTED,
                'filled': OrderStatus.FILLED,
                'partially_filled': OrderStatus.PARTIAL,
                'cancelled': OrderStatus.CANCELLED,
                'rejected': OrderStatus.REJECTED,
                'expired': OrderStatus.EXPIRED
            }
            return status_map.get(order.status, OrderStatus.PENDING)
        except Exception as e:
            logger.error(f"Failed to get order status: {e}")
            return OrderStatus.PENDING

    def get_positions(self) -> List[Position]:
        """Get current positions"""
        try:
            positions = []
            for p in self.api.list_positions():
                positions.append(Position(
                    symbol=p.symbol,
                    quantity=int(p.qty),
                    avg_cost=float(p.avg_entry_price),
                    current_price=float(p.current_price),
                    market_value=float(p.market_value),
                    unrealized_pnl=float(p.unrealized_pl),
                    realized_pnl=float(p.realized_pl) if hasattr(p, 'realized_pl') else 0,
                    last_updated=datetime.now()
                ))
            return positions
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    def get_account_info(self) -> Dict:
        """Get account information"""
        try:
            account = self.api.get_account()
            return {
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'pattern_day_trader': account.pattern_day_trader,
                'trading_blocked': account.trading_blocked,
                'transfers_blocked': account.transfers_blocked,
                'account_blocked': account.account_blocked
            }
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return {}

    def get_market_data(self, symbol: str) -> Dict:
        """Get real-time market data"""
        try:
            quote = self.api.get_latest_quote(symbol)
            trade = self.api.get_latest_trade(symbol)
            return {
                'bid': float(quote.bid_price),
                'ask': float(quote.ask_price),
                'last': float(trade.price),
                'volume': int(trade.size),
                'timestamp': trade.timestamp
            }
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            return {}

    def disconnect(self):
        """Disconnect from Alpaca"""
        logger.info("Disconnected from Alpaca")


# ===== Enhanced Order Management System =====
class OrderManagementSystem:
    """Complete order management system with multi-broker support"""

    def __init__(self, broker: BrokerInterface = None, broker_name: str = "schwab"):
        """Initialize OMS with specified broker"""

        # Select broker based on configuration
        if broker:
            self.broker = broker
        else:
            broker_name = broker_name.lower()
            if broker_name == "schwab":
                self.broker = SchwabBroker()
            elif broker_name == "alpaca":
                self.broker = AlpacaBroker()
            elif broker_name == "ib":
                self.broker = IBBroker()
            else:
                raise ValueError(f"Unsupported broker: {broker_name}")

        self.broker_name = broker_name
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.order_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.monitor_thread = None

        # Risk limits
        self.max_position_size = float(os.getenv("MAX_POSITION_SIZE", "0.10"))
        self.max_daily_loss = float(os.getenv("MAX_DAILY_LOSS", "0.02"))
        self.max_orders_per_minute = int(os.getenv("MAX_ORDERS_PER_MINUTE", "10"))

        # Tracking
        self.daily_pnl = 0.0
        self.order_count = 0
        self.last_order_time = datetime.now()

        logger.info(f"Order Management System initialized with {broker_name} broker")

    def start(self) -> bool:
        """Start the OMS"""
        if self.broker.connect():
            self.monitor_thread = threading.Thread(target=self._monitor_orders)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            logger.info(f"Order Management System started with {self.broker_name}")
            return True
        return False

    def stop(self):
        """Stop the OMS"""
        self.stop_event.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.broker.disconnect()
        logger.info("Order Management System stopped")

    def submit_order(self, order: Order) -> bool:
        """Submit an order with risk checks"""

        # Risk checks
        if not self._validate_order(order):
            logger.warning(f"Order failed validation: {order}")
            return False

        # Rate limiting
        if not self._check_rate_limit():
            logger.warning("Rate limit exceeded")
            return False

        # Submit to broker
        try:
            order_id = self.broker.submit_order(order)
            self.orders[order_id] = order
            self.order_count += 1
            self.last_order_time = datetime.now()

            # Log to database
            self._log_order_to_db(order)

            logger.info(f"Order {order_id} submitted successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to submit order: {e}")
            return False

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        if order_id in self.orders:
            if self.broker.cancel_order(order_id):
                self.orders[order_id].status = OrderStatus.CANCELLED
                self._log_order_to_db(self.orders[order_id])
                return True
        return False

    def replace_order(self, order_id: str, new_order: Order) -> Optional[str]:
        """Replace an existing order"""
        if order_id not in self.orders:
            logger.warning(f"Order {order_id} not found")
            return None

        try:
            # Use broker's replace functionality if available
            if hasattr(self.broker, 'replace_order'):
                new_order_id = self.broker.replace_order(order_id, new_order)

                # Update tracking
                if new_order_id:
                    del self.orders[order_id]
                    new_order.order_id = new_order_id
                    self.orders[new_order_id] = new_order
                    self._log_order_to_db(new_order)

                return new_order_id
            else:
                # Fallback: cancel and submit new
                if self.cancel_order(order_id):
                    return self.broker.submit_order(new_order)

        except Exception as e:
            logger.error(f"Failed to replace order {order_id}: {e}")

        return None

    def _validate_order(self, order: Order) -> bool:
        """Validate order against risk limits"""

        # Check position size
        account_info = self.broker.get_account_info()
        portfolio_value = account_info.get('portfolio_value', 0)

        if portfolio_value > 0:
            market_data = self.broker.get_market_data(order.symbol)
            order_value = order.quantity * market_data.get('last', 0)
            position_pct = order_value / portfolio_value

            if position_pct > self.max_position_size:
                logger.warning(f"Order exceeds max position size: {position_pct:.2%}")
                return False

        # Check daily loss limit
        if self.daily_pnl < -abs(self.max_daily_loss * portfolio_value):
            logger.warning(f"Daily loss limit reached: {self.daily_pnl}")
            return False

        # Check for existing positions
        if order.symbol in self.positions:
            current_pos = self.positions[order.symbol]
            if order.side == OrderSide.BUY and current_pos.quantity < 0:
                logger.warning("Cannot buy when short position exists")
                return False
            elif order.side == OrderSide.SELL and current_pos.quantity > 0:
                if order.quantity > current_pos.quantity:
                    logger.warning("Cannot sell more than current position")
                    return False

        return True

    def _check_rate_limit(self) -> bool:
        """Check order rate limits"""
        time_since_last = (datetime.now() - self.last_order_time).total_seconds()
        min_interval = 60.0 / self.max_orders_per_minute

        if time_since_last < min_interval:
            time.sleep(min_interval - time_since_last)

        return True

    def _monitor_orders(self):
        """Monitor order status and update positions"""
        while not self.stop_event.is_set():
            try:
                # Update order statuses
                for order_id, order in list(self.orders.items()):
                    if order.status not in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                        new_status = self.broker.get_order_status(order_id)

                        if new_status != order.status:
                            order.status = new_status
                            self._log_order_to_db(order)

                            if new_status == OrderStatus.FILLED:
                                self._update_position(order)

                # Update positions
                broker_positions = self.broker.get_positions()
                self.positions = {p.symbol: p for p in broker_positions}

                # Calculate daily P&L
                self._calculate_daily_pnl()

                time.sleep(1)  # Check every second

            except Exception as e:
                logger.error(f"Error in order monitor: {e}")
                time.sleep(5)

    def _update_position(self, order: Order):
        """Update position after fill"""
        try:
            if order.symbol not in self.positions:
                self.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=0,
                    avg_cost=0,
                    current_price=0,
                    market_value=0,
                    unrealized_pnl=0,
                    realized_pnl=0,
                    last_updated=datetime.now()
                )

            pos = self.positions[order.symbol]

            if order.side in [OrderSide.BUY, OrderSide.BUY_TO_COVER]:
                # Update average cost
                total_cost = (pos.quantity * pos.avg_cost) + (order.filled_quantity * order.avg_fill_price)
                pos.quantity += order.filled_quantity
                pos.avg_cost = total_cost / pos.quantity if pos.quantity > 0 else 0
            else:  # SELL or SELL_SHORT
                # Calculate realized P&L
                if pos.quantity > 0:  # Closing long position
                    realized = order.filled_quantity * (order.avg_fill_price - pos.avg_cost)
                    pos.realized_pnl += realized
                pos.quantity -= order.filled_quantity

            pos.last_updated = datetime.now()

        except Exception as e:
            logger.error(f"Failed to update position: {e}")

    def _calculate_daily_pnl(self):
        """Calculate daily P&L"""
        total_pnl = 0
        for pos in self.positions.values():
            total_pnl += pos.unrealized_pnl + pos.realized_pnl
        self.daily_pnl = total_pnl

    def _log_order_to_db(self, order: Order):
        """Log order to database"""
        try:
            with engine.begin() as conn:
                conn.execute(text("""
                                  INSERT INTO trading_orders (order_id, symbol, side, quantity, order_type,
                                                              limit_price, stop_price, status, filled_quantity,
                                                              avg_fill_price, submitted_at, filled_at, commission)
                                  VALUES (:order_id, :symbol, :side, :quantity, :order_type,
                                          :limit_price, :stop_price, :status, :filled_quantity,
                                          :avg_fill_price, :submitted_at, :filled_at,
                                          :commission) ON CONFLICT (order_id) DO
                                  UPDATE SET
                                      status = EXCLUDED.status,
                                      filled_quantity = EXCLUDED.filled_quantity,
                                      avg_fill_price = EXCLUDED.avg_fill_price,
                                      filled_at = EXCLUDED.filled_at,
                                      commission = EXCLUDED.commission
                                  """), {
                                 'order_id': order.order_id,
                                 'symbol': order.symbol,
                                 'side': order.side.value,
                                 'quantity': order.quantity,
                                 'order_type': order.order_type.value,
                                 'limit_price': order.limit_price,
                                 'stop_price': order.stop_price,
                                 'status': order.status.value,
                                 'filled_quantity': order.filled_quantity,
                                 'avg_fill_price': order.avg_fill_price,
                                 'submitted_at': order.submitted_at,
                                 'filled_at': order.filled_at,
                                 'commission': order.commission
                             })
        except Exception as e:
            logger.error(f"Failed to log order to database: {e}")

    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary"""
        account_info = self.broker.get_account_info()

        return {
            'broker': self.broker_name,
            'portfolio_value': account_info.get('portfolio_value', 0),
            'cash': account_info.get('cash', 0),
            'buying_power': account_info.get('buying_power', 0),
            'daily_pnl': self.daily_pnl,
            'num_positions': len(self.positions),
            'positions': list(self.positions.values()),
            'pending_orders': sum(1 for o in self.orders.values()
                                  if o.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.WORKING]),
            'filled_orders_today': self.order_count
        }


# ===== Main Execution Example =====
def main():
    """Example usage of the enhanced OMS with Schwab"""

    # Initialize OMS with Schwab broker
    oms = OrderManagementSystem(broker_name="schwab")

    # Start the OMS
    if not oms.start():
        logger.error("Failed to start OMS")
        return

    try:
        # Get portfolio summary
        summary = oms.get_portfolio_summary()
        print(f"Portfolio Value: ${summary['portfolio_value']:,.2f}")
        print(f"Cash: ${summary['cash']:,.2f}")
        print(f"Positions: {summary['num_positions']}")

        # Example: Submit a limit order
        order = Order(
            order_id="",  # Will be assigned by broker
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.LIMIT,
            limit_price=150.00,
            time_in_force="day"
        )

        if oms.submit_order(order):
            print(f"Order submitted: {order.order_id}")

        # Monitor for a while
        print("\nMonitoring orders... Press Ctrl+C to stop")
        while True:
            time.sleep(10)
            summary = oms.get_portfolio_summary()
            print(f"Daily P&L: ${summary['daily_pnl']:,.2f}")

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        oms.stop()


if __name__ == "__main__":
    main()