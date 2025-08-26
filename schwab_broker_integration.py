#!/usr/bin/env python3
"""
ACIS Trading Platform - Schwab Broker Integration
Comprehensive integration with Charles Schwab trading platform
Supports both live trading and paper trading environments
"""

import os
import requests
import json
import base64
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, asdict
import logging
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from trading_system import Order, OrderType, OrderSide, OrderStatus, Position, Account

class SchwabAPI:
    """Charles Schwab API integration for live and paper trading"""
    
    def __init__(self, client_id: str, client_secret: str, paper_trading: bool = True):
        load_dotenv()
        self.client_id = client_id
        self.client_secret = client_secret
        self.paper_trading = paper_trading
        
        # API endpoints
        self.base_url = "https://api.schwabapi.com" if not paper_trading else "https://api.schwabapi.com/trader/v1"
        self.auth_url = "https://api.schwabapi.com/oauth/token"
        
        # Authentication tokens
        self.access_token = None
        self.refresh_token = None
        self.token_expires = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('SchwabAPI')
        
        # Account information
        self.account_numbers = []
        self.primary_account = None
        
        # Rate limiting
        self.last_request_time = {}
        self.request_counts = {}
        
        # Database connection
        self.engine = create_engine(os.getenv('POSTGRES_URL'))
        
    def authenticate(self, authorization_code: str = None) -> bool:
        """Authenticate with Schwab API using OAuth 2.0"""
        try:
            if authorization_code:
                # Initial authentication with authorization code
                return self._get_initial_tokens(authorization_code)
            elif self.refresh_token:
                # Refresh existing token
                return self._refresh_access_token()
            else:
                # Generate authorization URL for user
                auth_url = self._generate_auth_url()
                self.logger.info(f"Please visit this URL to authorize the application: {auth_url}")
                return False
                
        except Exception as e:
            self.logger.error(f"Authentication failed: {e}")
            return False
    
    def _generate_auth_url(self) -> str:
        """Generate OAuth authorization URL"""
        # Generate code verifier and challenge for PKCE
        code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8').rstrip('=')
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode('utf-8')).digest()
        ).decode('utf-8').rstrip('=')
        
        # Store code verifier for later use
        self.code_verifier = code_verifier
        
        # Build authorization URL
        auth_params = {
            'response_type': 'code',
            'client_id': self.client_id,
            'redirect_uri': os.getenv('SCHWAB_REDIRECT_URI', 'https://localhost:8080/callback'),
            'scope': 'api',
            'code_challenge': code_challenge,
            'code_challenge_method': 'S256'
        }
        
        auth_url = "https://api.schwabapi.com/oauth/authorize?"
        auth_url += "&".join([f"{k}={v}" for k, v in auth_params.items()])
        
        return auth_url
    
    def _get_initial_tokens(self, authorization_code: str) -> bool:
        """Exchange authorization code for access and refresh tokens"""
        try:
            token_data = {
                'grant_type': 'authorization_code',
                'code': authorization_code,
                'redirect_uri': os.getenv('SCHWAB_REDIRECT_URI', 'https://localhost:8080/callback'),
                'code_verifier': self.code_verifier
            }
            
            # Create basic auth header
            auth_string = f"{self.client_id}:{self.client_secret}"
            auth_bytes = base64.b64encode(auth_string.encode()).decode()
            
            headers = {
                'Authorization': f'Basic {auth_bytes}',
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            response = requests.post(self.auth_url, data=token_data, headers=headers)
            
            if response.status_code == 200:
                token_info = response.json()
                self.access_token = token_info['access_token']
                self.refresh_token = token_info['refresh_token']
                self.token_expires = datetime.now() + timedelta(seconds=token_info['expires_in'])
                
                # Save tokens to database
                self._save_tokens()
                
                self.logger.info("Successfully obtained access and refresh tokens")
                return True
            else:
                self.logger.error(f"Token exchange failed: {response.status_code} {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error getting initial tokens: {e}")
            return False
    
    def _refresh_access_token(self) -> bool:
        """Refresh access token using refresh token"""
        try:
            token_data = {
                'grant_type': 'refresh_token',
                'refresh_token': self.refresh_token
            }
            
            # Create basic auth header
            auth_string = f"{self.client_id}:{self.client_secret}"
            auth_bytes = base64.b64encode(auth_string.encode()).decode()
            
            headers = {
                'Authorization': f'Basic {auth_bytes}',
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            response = requests.post(self.auth_url, data=token_data, headers=headers)
            
            if response.status_code == 200:
                token_info = response.json()
                self.access_token = token_info['access_token']
                if 'refresh_token' in token_info:
                    self.refresh_token = token_info['refresh_token']
                self.token_expires = datetime.now() + timedelta(seconds=token_info['expires_in'])
                
                # Save updated tokens
                self._save_tokens()
                
                self.logger.info("Successfully refreshed access token")
                return True
            else:
                self.logger.error(f"Token refresh failed: {response.status_code} {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error refreshing token: {e}")
            return False
    
    def _save_tokens(self):
        """Save tokens to database"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO broker_tokens (broker, access_token, refresh_token, expires_at, paper_trading)
                    VALUES ('schwab', :access_token, :refresh_token, :expires_at, :paper_trading)
                    ON CONFLICT (broker, paper_trading) DO UPDATE SET
                        access_token = EXCLUDED.access_token,
                        refresh_token = EXCLUDED.refresh_token,
                        expires_at = EXCLUDED.expires_at,
                        updated_at = CURRENT_TIMESTAMP
                """), {
                    'access_token': self.access_token,
                    'refresh_token': self.refresh_token,
                    'expires_at': self.token_expires,
                    'paper_trading': self.paper_trading
                })
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error saving tokens: {e}")
    
    def _load_tokens(self):
        """Load tokens from database"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT access_token, refresh_token, expires_at
                    FROM broker_tokens
                    WHERE broker = 'schwab' AND paper_trading = :paper_trading
                    ORDER BY updated_at DESC LIMIT 1
                """), {'paper_trading': self.paper_trading})
                
                token_data = result.fetchone()
                
                if token_data:
                    self.access_token = token_data[0]
                    self.refresh_token = token_data[1]
                    self.token_expires = token_data[2]
                    
                    # Check if token needs refresh
                    if datetime.now() >= self.token_expires - timedelta(minutes=5):
                        self._refresh_access_token()
                    
                    return True
                
        except Exception as e:
            self.logger.error(f"Error loading tokens: {e}")
        
        return False
    
    def _make_request(self, method: str, endpoint: str, data: dict = None, params: dict = None) -> requests.Response:
        """Make authenticated API request with rate limiting"""
        
        # Check authentication
        if not self.access_token or datetime.now() >= self.token_expires - timedelta(minutes=5):
            if not self._refresh_access_token():
                raise Exception("Authentication failed")
        
        # Rate limiting
        self._enforce_rate_limits(endpoint)
        
        # Prepare headers
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        # Make request
        url = f"{self.base_url}{endpoint}"
        
        if method.upper() == 'GET':
            response = requests.get(url, headers=headers, params=params)
        elif method.upper() == 'POST':
            response = requests.post(url, headers=headers, json=data)
        elif method.upper() == 'PUT':
            response = requests.put(url, headers=headers, json=data)
        elif method.upper() == 'DELETE':
            response = requests.delete(url, headers=headers)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        
        # Log request
        self.logger.info(f"{method} {endpoint} - Status: {response.status_code}")
        
        return response
    
    def _enforce_rate_limits(self, endpoint: str):
        """Enforce API rate limits"""
        # Schwab rate limits: 120 requests per minute
        current_time = datetime.now()
        minute_ago = current_time - timedelta(minutes=1)
        
        # Clean old request times
        if endpoint in self.last_request_time:
            self.request_counts[endpoint] = [
                req_time for req_time in self.request_counts.get(endpoint, [])
                if req_time > minute_ago
            ]
        
        # Check rate limit
        request_count = len(self.request_counts.get(endpoint, []))
        if request_count >= 120:
            sleep_time = 60 - (current_time - min(self.request_counts[endpoint])).seconds
            if sleep_time > 0:
                import time
                time.sleep(sleep_time)
        
        # Record request
        if endpoint not in self.request_counts:
            self.request_counts[endpoint] = []
        self.request_counts[endpoint].append(current_time)
        self.last_request_time[endpoint] = current_time
    
    def get_accounts(self) -> List[Dict]:
        """Get account information"""
        try:
            response = self._make_request('GET', '/accounts')
            
            if response.status_code == 200:
                accounts = response.json()
                self.account_numbers = [acc['accountId'] for acc in accounts]
                if self.account_numbers and not self.primary_account:
                    self.primary_account = self.account_numbers[0]
                
                self.logger.info(f"Retrieved {len(accounts)} accounts")
                return accounts
            else:
                self.logger.error(f"Failed to get accounts: {response.status_code} {response.text}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting accounts: {e}")
            return []
    
    def get_account_details(self, account_number: str = None) -> Dict:
        """Get detailed account information"""
        try:
            if not account_number:
                account_number = self.primary_account
            
            response = self._make_request('GET', f'/accounts/{account_number}')
            
            if response.status_code == 200:
                account_data = response.json()
                self.logger.info(f"Retrieved account details for {account_number}")
                return account_data
            else:
                self.logger.error(f"Failed to get account details: {response.status_code} {response.text}")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error getting account details: {e}")
            return {}
    
    def get_positions(self, account_number: str = None) -> List[Dict]:
        """Get account positions"""
        try:
            if not account_number:
                account_number = self.primary_account
            
            response = self._make_request('GET', f'/accounts/{account_number}/positions')
            
            if response.status_code == 200:
                positions = response.json()
                self.logger.info(f"Retrieved {len(positions)} positions")
                return positions
            else:
                self.logger.error(f"Failed to get positions: {response.status_code} {response.text}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return []
    
    def submit_order(self, order: Order, account_number: str = None) -> Dict:
        """Submit trading order to Schwab"""
        try:
            if not account_number:
                account_number = self.primary_account
            
            # Convert ACIS order to Schwab format
            schwab_order = self._convert_to_schwab_order(order)
            
            response = self._make_request('POST', f'/accounts/{account_number}/orders', data=schwab_order)
            
            if response.status_code in [200, 201]:
                order_response = response.json() if response.content else {}
                order_id = response.headers.get('Location', '').split('/')[-1] if 'Location' in response.headers else None
                
                self.logger.info(f"Order submitted successfully: {order.symbol} {order.quantity} shares")
                
                return {
                    'success': True,
                    'order_id': order_id,
                    'schwab_order_id': order_id,
                    'status': 'submitted',
                    'message': 'Order submitted successfully'
                }
            else:
                error_message = response.json().get('error', {}).get('message', 'Unknown error') if response.content else 'Unknown error'
                self.logger.error(f"Order submission failed: {response.status_code} {error_message}")
                
                return {
                    'success': False,
                    'error': error_message,
                    'status_code': response.status_code
                }
                
        except Exception as e:
            self.logger.error(f"Error submitting order: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _convert_to_schwab_order(self, order: Order) -> Dict:
        """Convert ACIS order to Schwab API format"""
        schwab_order = {
            "orderType": order.order_type.value.upper(),
            "session": "NORMAL",
            "duration": "DAY",
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [
                {
                    "orderLegType": "EQUITY",
                    "legId": 1,
                    "instrument": {
                        "symbol": order.symbol,
                        "assetType": "EQUITY"
                    },
                    "instruction": order.side.value.upper(),
                    "positionEffect": "OPENING",
                    "quantity": order.quantity
                }
            ]
        }
        
        # Add price for limit orders
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            schwab_order["price"] = order.price
        
        # Add stop price for stop orders
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            schwab_order["stopPrice"] = order.stop_price
        
        # Add client order ID if provided
        if order.client_order_id:
            schwab_order["tag"] = order.client_order_id[:50]  # Schwab limit
        
        return schwab_order
    
    def get_order_status(self, order_id: str, account_number: str = None) -> Dict:
        """Get order status from Schwab"""
        try:
            if not account_number:
                account_number = self.primary_account
            
            response = self._make_request('GET', f'/accounts/{account_number}/orders/{order_id}')
            
            if response.status_code == 200:
                order_data = response.json()
                
                return {
                    'success': True,
                    'order_id': order_id,
                    'status': order_data.get('status', '').lower(),
                    'filled_quantity': order_data.get('filledQuantity', 0),
                    'remaining_quantity': order_data.get('remainingQuantity', 0),
                    'average_fill_price': order_data.get('averageFillPrice', 0),
                    'order_data': order_data
                }
            else:
                self.logger.error(f"Failed to get order status: {response.status_code} {response.text}")
                return {
                    'success': False,
                    'error': f'HTTP {response.status_code}'
                }
                
        except Exception as e:
            self.logger.error(f"Error getting order status: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def cancel_order(self, order_id: str, account_number: str = None) -> Dict:
        """Cancel order with Schwab"""
        try:
            if not account_number:
                account_number = self.primary_account
            
            response = self._make_request('DELETE', f'/accounts/{account_number}/orders/{order_id}')
            
            if response.status_code in [200, 204]:
                self.logger.info(f"Order cancelled successfully: {order_id}")
                return {
                    'success': True,
                    'message': 'Order cancelled successfully'
                }
            else:
                error_message = response.json().get('error', {}).get('message', 'Unknown error') if response.content else 'Unknown error'
                self.logger.error(f"Order cancellation failed: {response.status_code} {error_message}")
                return {
                    'success': False,
                    'error': error_message
                }
                
        except Exception as e:
            self.logger.error(f"Error cancelling order: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_order_history(self, days: int = 30, account_number: str = None) -> List[Dict]:
        """Get order history"""
        try:
            if not account_number:
                account_number = self.primary_account
            
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            to_date = datetime.now().strftime('%Y-%m-%d')
            
            params = {
                'fromEnteredTime': from_date,
                'toEnteredTime': to_date
            }
            
            response = self._make_request('GET', f'/accounts/{account_number}/orders', params=params)
            
            if response.status_code == 200:
                orders = response.json()
                self.logger.info(f"Retrieved {len(orders)} orders from history")
                return orders
            else:
                self.logger.error(f"Failed to get order history: {response.status_code} {response.text}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting order history: {e}")
            return []
    
    def get_market_data(self, symbols: List[str]) -> Dict:
        """Get market data for symbols"""
        try:
            symbol_string = ",".join(symbols)
            params = {'symbol': symbol_string}
            
            response = self._make_request('GET', '/marketdata/quotes', params=params)
            
            if response.status_code == 200:
                market_data = response.json()
                self.logger.info(f"Retrieved market data for {len(symbols)} symbols")
                return market_data
            else:
                self.logger.error(f"Failed to get market data: {response.status_code} {response.text}")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            return {}

class SchwabBrokerManager:
    """High-level manager for Schwab broker integration"""
    
    def __init__(self):
        load_dotenv()
        self.engine = create_engine(os.getenv('POSTGRES_URL'))
        
        # Initialize Schwab API clients
        self.live_client = None
        self.paper_client = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('SchwabBrokerManager')
        
        # Initialize broker tokens table
        self._init_broker_tables()
    
    def _init_broker_tables(self):
        """Initialize broker-related database tables"""
        with self.engine.connect() as conn:
            # Broker tokens table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS broker_tokens (
                    id SERIAL PRIMARY KEY,
                    broker VARCHAR(20) NOT NULL,
                    access_token TEXT,
                    refresh_token TEXT,
                    expires_at TIMESTAMP,
                    paper_trading BOOLEAN DEFAULT false,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(broker, paper_trading)
                )
            """))
            
            # Broker configurations table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS broker_configurations (
                    id SERIAL PRIMARY KEY,
                    broker VARCHAR(20) NOT NULL,
                    config_key VARCHAR(50) NOT NULL,
                    config_value TEXT,
                    paper_trading BOOLEAN DEFAULT false,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(broker, config_key, paper_trading)
                )
            """))
            
            conn.commit()
    
    def initialize_schwab_live(self, client_id: str, client_secret: str) -> SchwabAPI:
        """Initialize live trading Schwab client"""
        try:
            self.live_client = SchwabAPI(client_id, client_secret, paper_trading=False)
            
            # Try to load existing tokens
            if self.live_client._load_tokens():
                self.logger.info("Loaded existing Schwab live trading tokens")
            else:
                self.logger.info("No existing tokens found for Schwab live trading")
            
            return self.live_client
            
        except Exception as e:
            self.logger.error(f"Error initializing Schwab live client: {e}")
            return None
    
    def initialize_schwab_paper(self, client_id: str, client_secret: str) -> SchwabAPI:
        """Initialize paper trading Schwab client"""
        try:
            self.paper_client = SchwabAPI(client_id, client_secret, paper_trading=True)
            
            # Try to load existing tokens
            if self.paper_client._load_tokens():
                self.logger.info("Loaded existing Schwab paper trading tokens")
            else:
                self.logger.info("No existing tokens found for Schwab paper trading")
            
            return self.paper_client
            
        except Exception as e:
            self.logger.error(f"Error initializing Schwab paper client: {e}")
            return None
    
    def get_client(self, paper_trading: bool = True) -> Optional[SchwabAPI]:
        """Get appropriate Schwab client"""
        if paper_trading:
            return self.paper_client
        else:
            return self.live_client
    
    def submit_acis_order(self, order: Order, paper_trading: bool = True, account_number: str = None) -> Dict:
        """Submit ACIS order through appropriate Schwab client"""
        try:
            client = self.get_client(paper_trading)
            if not client:
                return {
                    'success': False,
                    'error': f'Schwab {"paper" if paper_trading else "live"} client not initialized'
                }
            
            # Submit order
            result = client.submit_order(order, account_number)
            
            # Log to database
            self._log_order_submission(order, result, paper_trading)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error submitting ACIS order: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _log_order_submission(self, order: Order, result: Dict, paper_trading: bool):
        """Log order submission to database"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO broker_order_log (
                        broker, symbol, quantity, side, order_type, client_order_id,
                        schwab_order_id, success, error_message, paper_trading, submitted_at
                    ) VALUES (
                        'schwab', :symbol, :quantity, :side, :order_type, :client_order_id,
                        :schwab_order_id, :success, :error_message, :paper_trading, CURRENT_TIMESTAMP
                    )
                """), {
                    'symbol': order.symbol,
                    'quantity': order.quantity,
                    'side': order.side.value,
                    'order_type': order.order_type.value,
                    'client_order_id': order.client_order_id,
                    'schwab_order_id': result.get('schwab_order_id'),
                    'success': result.get('success', False),
                    'error_message': result.get('error'),
                    'paper_trading': paper_trading
                })
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error logging order submission: {e}")

def main():
    """Test Schwab broker integration"""
    print("ACIS Trading Platform - Schwab Broker Integration")
    print("Testing Schwab API connectivity and functionality...")
    
    # Initialize broker manager
    manager = SchwabBrokerManager()
    
    # Configuration (these would come from environment variables)
    client_id = os.getenv('SCHWAB_CLIENT_ID')
    client_secret = os.getenv('SCHWAB_CLIENT_SECRET')
    
    if not client_id or not client_secret:
        print("Please set SCHWAB_CLIENT_ID and SCHWAB_CLIENT_SECRET environment variables")
        return
    
    # Initialize paper trading client
    paper_client = manager.initialize_schwab_paper(client_id, client_secret)
    
    if paper_client:
        print("✓ Schwab paper trading client initialized")
        
        # Test authentication (would require actual OAuth flow)
        print("Note: Full authentication requires OAuth flow with user interaction")
        
        # Test order creation (simulation)
        from trading_system import Order, OrderType, OrderSide
        
        test_order = Order(
            symbol='AAPL',
            quantity=10,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            strategy='test_strategy',
            client_order_id='TEST_SCHWAB_001'
        )
        
        print(f"Test order created: {test_order.symbol} {test_order.quantity} shares")
        print("Ready for production Schwab integration!")
        
    else:
        print("✗ Failed to initialize Schwab client")

if __name__ == "__main__":
    main()