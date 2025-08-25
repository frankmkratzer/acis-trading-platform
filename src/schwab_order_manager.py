# =====================================
# 3. SCHWAB ORDER MANAGER
# =====================================
"""
#!/usr/bin/env python3
# File: schwab_order_manager.py
# Purpose: Advanced order management for Schwab
"""


class SchwabOrderManager:
    """Advanced order management with Schwab-specific features"""

    def __init__(self, broker: SchwabBroker):
        self.broker = broker
        self.pending_orders = {}
        self.filled_orders = {}
        self.order_history = []

    def place_bracket_order(self,
                            symbol: str,
                            quantity: int,
                            entry_price: float,
                            stop_loss: float,
                            take_profit: float) -> List[str]:
        """Place a bracket order (entry + stop loss + take profit)"""

        # Create the main order
        main_order = self.broker.create_equity_order(
            symbol=symbol,
            quantity=quantity,
            instruction=SchwabOrderInstruction.BUY,
            order_type=SchwabOrderType.LIMIT,
            limit_price=entry_price,
            duration=SchwabOrderDuration.DAY
        )

        # Add One-Cancels-Other (OCO) bracket
        main_order["orderStrategyType"] = "TRIGGER"
        main_order["childOrderStrategies"] = [
            {
                "orderStrategyType": "OCO",
                "childOrderStrategies": [
                    # Stop loss order
                    {
                        "orderType": "STOP",
                        "session": "NORMAL",
                        "duration": "GTC",
                        "stopPrice": stop_loss,
                        "orderLegCollection": [
                            {
                                "instruction": "SELL",
                                "quantity": quantity,
                                "instrument": {
                                    "symbol": symbol,
                                    "assetType": "EQUITY"
                                }
                            }
                        ]
                    },
                    # Take profit order
                    {
                        "orderType": "LIMIT",
                        "session": "NORMAL",
                        "duration": "GTC",
                        "price": take_profit,
                        "orderLegCollection": [
                            {
                                "instruction": "SELL",
                                "quantity": quantity,
                                "instrument": {
                                    "symbol": symbol,
                                    "assetType": "EQUITY"
                                }
                            }
                        ]
                    }
                ]
            }
        ]

        # Submit the bracket order
        order_id = self.broker.submit_order(main_order)

        logger.info(f"Bracket order placed for {symbol}: Entry={entry_price}, SL={stop_loss}, TP={take_profit}")

        return [order_id]

    def place_trailing_stop_order(self,
                                  symbol: str,
                                  quantity: int,
                                  trail_amount: float = None,
                                  trail_percent: float = None) -> str:
        """Place a trailing stop order"""

        order = {
            "orderType": "TRAILING_STOP",
            "session": "NORMAL",
            "duration": "GTC",
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [
                {
                    "instruction": "SELL",
                    "quantity": quantity,
                    "instrument": {
                        "symbol": symbol,
                        "assetType": "EQUITY"
                    }
                }
            ]
        }

        if trail_amount:
            order["stopPriceLinkBasis"] = "MANUAL"
            order["stopPriceLinkType"] = "VALUE"
            order["stopPriceOffset"] = trail_amount
        elif trail_percent:
            order["stopPriceLinkBasis"] = "MANUAL"
            order["stopPriceLinkType"] = "PERCENT"
            order["stopPriceOffset"] = trail_percent

        order_id = self.broker.submit_order(order)

        logger.info(f"Trailing stop order placed for {symbol}")

        return order_id

    def place_conditional_order(self,
                                symbol: str,
                                quantity: int,
                                condition_type: str,
                                condition_symbol: str,
                                condition_price: float,
                                order_type: SchwabOrderType,
                                limit_price: float = None) -> str:
        """Place a conditional order"""

        order = self.broker.create_equity_order(
            symbol=symbol,
            quantity=quantity,
            instruction=SchwabOrderInstruction.BUY,
            order_type=order_type,
            limit_price=limit_price
        )

        # Add condition
        order["releaseTime"] = None  # Will be triggered by condition
        order["specialInstruction"] = "ALL_OR_NONE"
        order["orderStrategyType"] = "CONDITIONAL"
        order["condition"] = {
            "conditionType": condition_type,  # e.g., "PRICE", "TIME", "MARGIN"
            "symbol": condition_symbol,
            "price": condition_price,
            "operator": "GTE"  # Greater than or equal
        }

        order_id = self.broker.submit_order(order)

        logger.info(f"Conditional order placed for {symbol}")

        return order_id

    def scale_in_position(self,
                          symbol: str,
                          total_quantity: int,
                          num_orders: int,
                          price_range: Tuple[float, float]) -> List[str]:
        """Scale into a position with multiple orders"""

        order_ids = []
        quantity_per_order = total_quantity // num_orders
        price_increment = (price_range[1] - price_range[0]) / (num_orders - 1)

        for i in range(num_orders):
            price = price_range[0] + (i * price_increment)

            order = self.broker.create_equity_order(
                symbol=symbol,
                quantity=quantity_per_order,
                instruction=SchwabOrderInstruction.BUY,
                order_type=SchwabOrderType.LIMIT,
                limit_price=price,
                duration=SchwabOrderDuration.GTC
            )

            order_id = self.broker.submit_order(order)
            order_ids.append(order_id)

            logger.info(f"Scale-in order {i + 1}/{num_orders} for {symbol} at {price}")

        return order_ids

    def manage_position_risk(self, symbol: str):
        """Actively manage position risk"""

        # Get current position
        positions = self.broker.get_positions()
        position = next((p for p in positions if p['symbol'] == symbol), None)

        if not position:
            logger.warning(f"No position found for {symbol}")
            return

        # Get current quote
        quote = self.broker.get_quote(symbol)
        current_price = quote['last']

        # Calculate risk metrics
        position_value = position['quantity'] * current_price
        unrealized_pnl_pct = position['unrealized_pnl'] / position['market_value']

        # Risk management rules
        if unrealized_pnl_pct < -0.08:  # 8% loss
            # Place stop loss
            stop_price = current_price * 0.98
            self.place_trailing_stop_order(symbol, position['quantity'], trail_percent=2)
            logger.warning(f"Stop loss triggered for {symbol} at {unrealized_pnl_pct:.2%} loss")

        elif unrealized_pnl_pct > 0.15:  # 15% gain
            # Take partial profits
            sell_quantity = int(position['quantity'] * 0.5)
            if sell_quantity > 0:
                order = self.broker.create_equity_order(
                    symbol=symbol,
                    quantity=sell_quantity,
                    instruction=SchwabOrderInstruction.SELL,
                    order_type=SchwabOrderType.MARKET
                )
                self.broker.submit_order(order)
                logger.info(f"Taking partial profits for {symbol} at {unrealized_pnl_pct:.2%} gain")
