# =====================================
# 5. MAIN SCHWAB TRADING INTERFACE
# =====================================
"""
#!/usr/bin/env python3
# File: schwab_trading_interface.py
# Purpose: Main interface for Schwab trading integration
"""


class SchwabTradingInterface:
    """Complete Schwab trading interface"""

    def __init__(self):
        self.broker = SchwabBroker()
        self.streaming = None
        self.order_manager = SchwabOrderManager(self.broker)
        self.analytics = SchwabPortfolioAnalytics(self.broker)
        self.running = False

    def initialize(self) -> bool:
        """Initialize Schwab connection"""

        # Connect to broker
        if not self.broker.connect():
            logger.error("Failed to connect to Schwab")
            return False

        # Initialize streaming
        self.streaming = SchwabStreamingClient(self.broker.auth_manager)
        if not self.streaming.connect():
            logger.warning("Streaming connection failed - will use polling")

        self.running = True
        logger.info("Schwab trading interface initialized")
        return True

    def shutdown(self):
        """Shutdown connections"""

        self.running = False

        if self.streaming:
            self.streaming.disconnect()

        self.broker.disconnect()

        logger.info("Schwab trading interface shutdown")

    def execute_strategy(self, signals: pd.DataFrame):
        """Execute trading signals"""

        account_info = self.broker.get_account_info()
        buying_power = account_info['buying_power']

        for _, signal in signals.iterrows():
            symbol = signal['symbol']
            action = signal['action']  # 'BUY' or 'SELL'
            quantity = signal['quantity']

            try:
                if action == 'BUY':
                    # Check buying power
                    quote = self.broker.get_quote(symbol)
                    required = quantity * quote['ask']

                    if required > buying_power:
                        logger.warning(f"Insufficient buying power for {symbol}")
                        continue

                    # Place buy order
                    order = self.broker.create_equity_order(
                        symbol=symbol,
                        quantity=quantity,
                        instruction=SchwabOrderInstruction.BUY,
                        order_type=SchwabOrderType.LIMIT,
                        limit_price=quote['ask'] * 1.001,  # Small buffer
                        duration=SchwabOrderDuration.DAY
                    )

                elif action == 'SELL':
                    # Check position
                    positions = self.broker.get_positions()
                    position = next((p for p in positions if p['symbol'] == symbol), None)

                    if not position or position['quantity'] < quantity:
                        logger.warning(f"Insufficient position for {symbol}")
                        continue

                    # Place sell order
                    order = self.broker.create_equity_order(
                        symbol=symbol,
                        quantity=quantity,
                        instruction=SchwabOrderInstruction.SELL,
                        order_type=SchwabOrderType.LIMIT,
                        limit_price=quote['bid'] * 0.999,  # Small buffer
                        duration=SchwabOrderDuration.DAY
                    )

                # Submit order
                order_id = self.broker.submit_order(order)
                logger.info(f"Order submitted: {order_id} for {symbol}")

            except Exception as e:
                logger.error(f"Failed to execute signal for {symbol}: {e}")

    def monitor_positions(self):
        """Monitor and manage positions"""

        while self.running:
            try:
                positions = self.broker.get_positions()

                for position in positions:
                    # Check each position for risk management
                    self.order_manager.manage_position_risk(position['symbol'])

                # Get portfolio summary
                summary = self.analytics.get_portfolio_summary()

                # Log portfolio status
                logger.info(f"Portfolio value: ${summary['total_value']:,.2f}, "
                            f"P&L: ${summary['total_pnl']:,.2f} ({summary['total_pnl_pct']:.2f}%)")

                # Sleep before next check
                time.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error monitoring positions: {e}")
                time.sleep(60)


# =====================================
# USAGE EXAMPLE
# =====================================

def main():
    """Example usage of Schwab trading integration"""

    # Initialize interface
    interface = SchwabTradingInterface()

    if not interface.initialize():
        print("Failed to initialize Schwab interface")
        return

    try:
        # Get account info
        account = interface.broker.get_account_info()
        print(f"Account ID: {account['account_id']}")
        print(f"Buying Power: ${account['buying_power']:,.2f}")
        print(f"Portfolio Value: ${account['portfolio_value']:,.2f}")

        # Get positions
        positions = interface.broker.get_positions()
        print(f"\nPositions: {len(positions)}")
        for pos in positions:
            print(f"  {pos['symbol']}: {pos['quantity']} shares @ ${pos['avg_cost']:.2f}")

        # Get quote
        quote = interface.broker.get_quote("AAPL")
        print(f"\nAAPL Quote: ${quote['last']:.2f} ({quote['change_percent']:.2f}%)")

        # Place a test order (commented out for safety)
        # order = interface.broker.create_equity_order(
        #     symbol="AAPL",
        #     quantity=1,
        #     instruction=SchwabOrderInstruction.BUY,
        #     order_type=SchwabOrderType.LIMIT,
        #     limit_price=150.00
        # )
        # order_id = interface.broker.submit_order(order)
        # print(f"Order placed: {order_id}")

        # Start monitoring (runs in background)
        monitor_thread = threading.Thread(target=interface.monitor_positions)
        monitor_thread.daemon = True
        monitor_thread.start()

        # Keep running
        print("\nPress Ctrl+C to stop...")
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        interface.shutdown()


if __name__ == "__main__":
    main()