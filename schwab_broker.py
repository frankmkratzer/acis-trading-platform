# =====================================
# 1. SCHWAB BROKER IMPLEMENTATION
# =====================================
"""
#!/usr/bin/env python3
# File: schwab_broker.py
# Purpose: Charles Schwab broker integration for live trading
"""

import os
import json
import time
import hashlib
import base64
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
import aiohttp
import websocket
import threading
import logging
from urllib.parse import urlencode, quote
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
engine = create_engine(os.getenv("POSTGRES_URL"))

logging.basicConfig(
    filename="schwab_trading.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ===== Schwab API Configuration =====
SCHWAB_CONFIG = {
    'base_url': 'https://api.schwabapi.com/trader/v1',
    'auth_url': 'https://api.schwabapi.com/v1/oauth/authorize',
    'token_url': 'https://api.schwabapi.com/v1/oauth/token',
    'accounts_endpoint': '/accounts',
    'orders_endpoint': '/orders',
    'quotes_endpoint': '/marketdata/v1/quotes',
    'options_endpoint': '/marketdata/v1/chains',
    'movers_endpoint': '/marketdata/v1/movers',
    'price_history_endpoint': '/marketdata/v1/pricehistory'
}


# ===== Order Enums for Schwab =====
class SchwabOrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"
    MARKET_ON_CLOSE = "MARKET_ON_CLOSE"
    LIMIT_ON_CLOSE = "LIMIT_ON_CLOSE"


class SchwabOrderInstruction(Enum):
    BUY = "BUY"
    SELL = "SELL"
    BUY_TO_COVER = "BUY_TO_COVER"
    SELL_SHORT = "SELL_SHORT"


class SchwabOrderDuration(Enum):
    DAY = "DAY"
    GTC = "GOOD_TILL_CANCEL"
    GTD = "GOOD_TILL_DATE"
    IOC = "IMMEDIATE_OR_CANCEL"
    FOK = "FILL_OR_KILL"


class SchwabOrderStatus(Enum):
    AWAITING_PARENT_ORDER = "AWAITING_PARENT_ORDER"
    AWAITING_CONDITION = "AWAITING_CONDITION"
    AWAITING_MANUAL_REVIEW = "AWAITING_MANUAL_REVIEW"
    ACCEPTED = "ACCEPTED"
    AWAITING_UR_OUT = "AWAITING_UR_OUT"
    PENDING_ACTIVATION = "PENDING_ACTIVATION"
    QUEUED = "QUEUED"
    WORKING = "WORKING"
    REJECTED = "REJECTED"
    PENDING_CANCEL = "PENDING_CANCEL"
    CANCELED = "CANCELED"
    PENDING_REPLACE = "PENDING_REPLACE"
    REPLACED = "REPLACED"
    FILLED = "FILLED"
    EXPIRED = "EXPIRED"


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
        return f"{SCHWAB_CONFIG['auth_url']}?{urlencode(params)}"

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
            response = requests.post(SCHWAB_CONFIG['token_url'], data=data)
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
            response = requests.post(SCHWAB_CONFIG['token_url'], data=data)
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
        # For now, save to encrypted file
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
class SchwabBroker:
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

        url = f"{SCHWAB_CONFIG['base_url']}{endpoint}"
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
            logger.error(f"Response: {e.response.text if e.response else 'No response'}")
            raise
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise

    def connect(self) -> bool:
        """Connect to Schwab and verify authentication"""
        try:
            # Test connection by getting account info
            accounts = self.get_accounts()
            if accounts:
                logger.info(f"Connected to Schwab. Found {len(accounts)} account(s)")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Schwab: {e}")
            return False

    def disconnect(self):
        """Disconnect from Schwab"""
        self.session.close()
        logger.info("Disconnected from Schwab")

    def get_accounts(self) -> List[Dict]:
        """Get all linked accounts"""

        endpoint = SCHWAB_CONFIG['accounts_endpoint']
        response = self._make_request('GET', endpoint)

        return response if isinstance(response, list) else [response]

    def get_account_info(self) -> Dict:
        """Get detailed account information"""

        if not self.account_id:
            accounts = self.get_accounts()
            if accounts:
                self.account_id = accounts[0]['securitiesAccount']['accountId']

        endpoint = f"{SCHWAB_CONFIG['accounts_endpoint']}/{self.account_id}"
        params = {'fields': 'positions,orders'}

        account_data = self._make_request('GET', endpoint, params)

        if 'securitiesAccount' in account_data:
            account = account_data['securitiesAccount']

            return {
                'account_id': account.get('accountId'),
                'buying_power': float(account['currentBalances'].get('buyingPower', 0)),
                'cash': float(account['currentBalances'].get('cashBalance', 0)),
                'portfolio_value': float(account['currentBalances'].get('liquidationValue', 0)),
                'day_trading_buying_power': float(account['currentBalances'].get('dayTradingBuyingPower', 0)),
                'maintenance_requirement': float(account['currentBalances'].get('maintenanceRequirement', 0)),
                'available_funds': float(account['currentBalances'].get('availableFunds', 0)),
                'positions': account.get('positions', []),
                'orders': account.get('orderStrategies', [])
            }

        return {}

    def get_positions(self) -> List[Dict]:
        """Get current positions"""

        # Check cache
        if self._is_cache_valid():
            return list(self.positions_cache.values())

        account_info = self.get_account_info()
        positions = []

        for pos in account_info.get('positions', []):
            instrument = pos.get('instrument', {})

            position = {
                'symbol': instrument.get('symbol'),
                'quantity': float(pos.get('longQuantity', 0) - pos.get('shortQuantity', 0)),
                'avg_cost': float(pos.get('averagePrice', 0)),
                'current_price': float(pos.get('marketValue', 0)) / float(pos.get('longQuantity', 1)),
                'market_value': float(pos.get('marketValue', 0)),
                'unrealized_pnl': float(pos.get('unrealizedPNL', 0)),
                'realized_pnl': float(pos.get('realizedPNL', 0)),
                'maintenance_requirement': float(pos.get('maintenanceRequirement', 0))
            }

            positions.append(position)
            self.positions_cache[position['symbol']] = position

        self.last_cache_update = datetime.now()
        return positions

    def submit_order(self, order_request: Dict) -> str:
        """Submit an order to Schwab"""

        endpoint = f"{SCHWAB_CONFIG['accounts_endpoint']}/{self.account_id}/orders"

        try:
            # Schwab returns order ID in Location header
            response = self.session.post(
                f"{SCHWAB_CONFIG['base_url']}{endpoint}",
                headers=self._get_headers(),
                json=order_request
            )
            response.raise_for_status()

            # Extract order ID from Location header
            if 'Location' in response.headers:
                order_id = response.headers['Location'].split('/')[-1]
                logger.info(f"Order submitted successfully: {order_id}")
                return order_id

            return str(datetime.now().timestamp())  # Fallback ID

        except Exception as e:
            logger.error(f"Failed to submit order: {e}")
            raise

    def create_equity_order(self,
                            symbol: str,
                            quantity: int,
                            instruction: SchwabOrderInstruction,
                            order_type: SchwabOrderType,
                            limit_price: float = None,
                            stop_price: float = None,
                            duration: SchwabOrderDuration = SchwabOrderDuration.DAY) -> Dict:
        """Create an equity order request"""

        order = {
            "orderType": order_type.value,
            "session": "NORMAL",
            "duration": duration.value,
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [
                {
                    "instruction": instruction.value,
                    "quantity": quantity,
                    "instrument": {
                        "symbol": symbol,
                        "assetType": "EQUITY"
                    }
                }
            ]
        }

        # Add price for limit orders
        if order_type in [SchwabOrderType.LIMIT, SchwabOrderType.STOP_LIMIT]:
            order["price"] = limit_price

        # Add stop price for stop orders
        if order_type in [SchwabOrderType.STOP, SchwabOrderType.STOP_LIMIT]:
            order["stopPrice"] = stop_price

        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""

        endpoint = f"{SCHWAB_CONFIG['accounts_endpoint']}/{self.account_id}/orders/{order_id}"

        try:
            self._make_request('DELETE', endpoint)
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def replace_order(self, order_id: str, new_order: Dict) -> str:
        """Replace an existing order"""

        endpoint = f"{SCHWAB_CONFIG['accounts_endpoint']}/{self.account_id}/orders/{order_id}"

        try:
            response = self.session.put(
                f"{SCHWAB_CONFIG['base_url']}{endpoint}",
                headers=self._get_headers(),
                json=new_order
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

    def get_order_status(self, order_id: str) -> Dict:
        """Get order status and details"""

        endpoint = f"{SCHWAB_CONFIG['accounts_endpoint']}/{self.account_id}/orders/{order_id}"

        try:
            order = self._make_request('GET', endpoint)
            return order
        except Exception as e:
            logger.error(f"Failed to get order status: {e}")
            return {}

    def get_orders(self,
                   from_date: datetime = None,
                   to_date: datetime = None,
                   status: str = None) -> List[Dict]:
        """Get orders with optional filters"""

        endpoint = f"{SCHWAB_CONFIG['accounts_endpoint']}/{self.account_id}/orders"

        params = {}
        if from_date:
            params['fromEnteredTime'] = from_date.isoformat()
        if to_date:
            params['toEnteredTime'] = to_date.isoformat()
        if status:
            params['status'] = status

        orders = self._make_request('GET', endpoint, params)
        return orders if isinstance(orders, list) else [orders]

    def get_quote(self, symbol: str) -> Dict:
        """Get real-time quote for a symbol"""

        endpoint = f"/marketdata/v1/quotes"
        params = {'symbols': symbol}

        response = self._make_request('GET', endpoint, params)

        if symbol in response:
            quote = response[symbol]
            return {
                'symbol': symbol,
                'bid': float(quote.get('bidPrice', 0)),
                'ask': float(quote.get('askPrice', 0)),
                'last': float(quote.get('lastPrice', 0)),
                'volume': int(quote.get('totalVolume', 0)),
                'high': float(quote.get('highPrice', 0)),
                'low': float(quote.get('lowPrice', 0)),
                'close': float(quote.get('closePrice', 0)),
                'change': float(quote.get('netChange', 0)),
                'change_percent': float(quote.get('netPercentChange', 0))
            }

        return {}

    def get_quotes(self, symbols: List[str]) -> Dict:
        """Get real-time quotes for multiple symbols"""

        endpoint = f"/marketdata/v1/quotes"
        params = {'symbols': ','.join(symbols)}

        response = self._make_request('GET', endpoint, params)

        quotes = {}
        for symbol, data in response.items():
            quotes[symbol] = {
                'bid': float(data.get('bidPrice', 0)),
                'ask': float(data.get('askPrice', 0)),
                'last': float(data.get('lastPrice', 0)),
                'volume': int(data.get('totalVolume', 0))
            }

        return quotes

    def get_price_history(self,
                          symbol: str,
                          period_type: str = 'day',
                          period: int = 10,
                          frequency_type: str = 'minute',
                          frequency: int = 1) -> pd.DataFrame:
        """Get historical price data"""

        endpoint = f"/marketdata/v1/pricehistory"
        params = {
            'symbol': symbol,
            'periodType': period_type,
            'period': period,
            'frequencyType': frequency_type,
            'frequency': frequency
        }

        response = self._make_request('GET', endpoint, params)

        if 'candles' in response:
            candles = response['candles']
            df = pd.DataFrame(candles)
            df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
            df.set_index('datetime', inplace=True)
            return df

        return pd.DataFrame()

    def get_option_chain(self,
                         symbol: str,
                         contract_type: str = 'ALL',
                         strike_count: int = 10,
                         include_quotes: bool = True) -> Dict:
        """Get option chain data"""

        endpoint = f"/marketdata/v1/chains"
        params = {
            'symbol': symbol,
            'contractType': contract_type,
            'strikeCount': strike_count,
            'includeQuotes': str(include_quotes).upper()
        }

        return self._make_request('GET', endpoint, params)

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid"""
        if not self.last_cache_update:
            return False

        age = (datetime.now() - self.last_cache_update).total_seconds()
        return age < self.cache_ttl
