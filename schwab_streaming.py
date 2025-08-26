#!/usr/bin/env python3
# File: schwab_streaming.py
# Purpose: Real-time streaming data from Schwab

from schwab_broker import SchwabBroker, SchwabOrderType, SchwabOrderInstruction, SchwabOrderDuration, SchwabAuthManager
import websocket
import logging
import json
import threading
import time
from typing import List, Dict, Optional

# Set up logging
logger = logging.getLogger(__name__)


class SchwabStreamingClient:
    """WebSocket streaming client for Schwab real-time data"""

    def __init__(self, auth_manager: SchwabAuthManager):
        self.auth_manager = auth_manager
        self.ws = None
        self.connected = False
        self.subscriptions = {}
        self.callbacks = {}
        self.heartbeat_thread = None
        self.receiver_thread = None

    def connect(self):
        """Connect to Schwab streaming service"""

        # Get streaming endpoint and credentials
        streaming_url = "wss://streamer.schwab.com/ws"
        token = self.auth_manager.get_valid_token()

        if not token:
            logger.error("No valid token for streaming")
            return False

        try:
            # Create WebSocket connection
            self.ws = websocket.WebSocketApp(
                streaming_url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                header={'Authorization': f'Bearer {token}'}
            )

            # Start connection in separate thread
            self.receiver_thread = threading.Thread(target=self.ws.run_forever)
            self.receiver_thread.daemon = True
            self.receiver_thread.start()

            # Wait for connection
            time.sleep(2)

            if self.connected:
                self._start_heartbeat()
                logger.info("Connected to Schwab streaming service")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to connect to streaming service: {e}")
            return False

    def disconnect(self):
        """Disconnect from streaming service"""

        self.connected = False

        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=1)

        if self.ws:
            self.ws.close()

        if self.receiver_thread:
            self.receiver_thread.join(timeout=2)

        logger.info("Disconnected from Schwab streaming service")

    def subscribe_quotes(self, symbols: List[str], callback=None):
        """Subscribe to real-time quotes"""

        request = {
            "requests": [
                {
                    "service": "QUOTE",
                    "requestid": "1",
                    "command": "SUBS",
                    "account": self.auth_manager.account_id,
                    "source": self.auth_manager.client_id,
                    "parameters": {
                        "keys": ",".join(symbols),
                        "fields": "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
                    }
                }
            ]
        }

        self.subscriptions['quotes'] = symbols
        if callback:
            self.callbacks['quotes'] = callback

        self._send_request(request)
        logger.info(f"Subscribed to quotes for {len(symbols)} symbols")

    def subscribe_level2(self, symbols: List[str], callback=None):
        """Subscribe to Level 2 market depth"""

        request = {
            "requests": [
                {
                    "service": "LISTED_BOOK",
                    "requestid": "2",
                    "command": "SUBS",
                    "account": self.auth_manager.account_id,
                    "source": self.auth_manager.client_id,
                    "parameters": {
                        "keys": ",".join(symbols)
                    }
                }
            ]
        }

        self.subscriptions['level2'] = symbols
        if callback:
            self.callbacks['level2'] = callback

        self._send_request(request)
        logger.info(f"Subscribed to Level 2 for {len(symbols)} symbols")

    def subscribe_trades(self, symbols: List[str], callback=None):
        """Subscribe to time and sales"""

        request = {
            "requests": [
                {
                    "service": "TIMESALE_EQUITY",
                    "requestid": "3",
                    "command": "SUBS",
                    "account": self.auth_manager.account_id,
                    "source": self.auth_manager.client_id,
                    "parameters": {
                        "keys": ",".join(symbols)
                    }
                }
            ]
        }

        self.subscriptions['trades'] = symbols
        if callback:
            self.callbacks['trades'] = callback

        self._send_request(request)
        logger.info(f"Subscribed to trades for {len(symbols)} symbols")

    def _on_open(self, ws):
        """Handle connection open"""
        self.connected = True
        logger.info("Streaming connection opened")

    def _on_message(self, ws, message):
        """Handle incoming messages"""

        try:
            data = json.loads(message)

            # Handle different message types
            if 'data' in data:
                for item in data['data']:
                    service = item.get('service')
                    content = item.get('content', [])

                    # Route to appropriate callback
                    if service == 'QUOTE' and 'quotes' in self.callbacks:
                        self.callbacks['quotes'](content)
                    elif service == 'LISTED_BOOK' and 'level2' in self.callbacks:
                        self.callbacks['level2'](content)
                    elif service == 'TIMESALE_EQUITY' and 'trades' in self.callbacks:
                        self.callbacks['trades'](content)

            # Handle responses
            elif 'response' in data:
                for response in data['response']:
                    if response.get('content', {}).get('code') == 0:
                        logger.debug(f"Request successful: {response.get('service')}")
                    else:
                        logger.warning(f"Request failed: {response}")

        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def _on_error(self, ws, error):
        """Handle errors"""
        logger.error(f"Streaming error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        """Handle connection close"""
        self.connected = False
        logger.info(f"Streaming connection closed: {close_msg}")

    def _send_request(self, request: Dict):
        """Send request to streaming service"""

        if self.ws and self.connected:
            self.ws.send(json.dumps(request))
        else:
            logger.warning("Cannot send request: not connected")

    def _start_heartbeat(self):
        """Start heartbeat to keep connection alive"""

        def heartbeat():
            while self.connected:
                self._send_request({"heartbeat": {}})
                time.sleep(30)

        self.heartbeat_thread = threading.Thread(target=heartbeat)
        self.heartbeat_thread.daemon = True
        self.heartbeat_thread.start()