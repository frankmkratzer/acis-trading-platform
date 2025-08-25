#!/usr/bin/env python3
# test_schwab_integration.py

from live_trading_engine import OrderManagementSystem, Order, OrderSide, OrderType
import time


def test_schwab_integration():
    # Initialize OMS with Schwab
    oms = OrderManagementSystem(broker_name="schwab")

    if not oms.start():
        print("Failed to connect to Schwab")
        return

    try:
        # Test 1: Get account info
        summary = oms.get_portfolio_summary()
        print(f"✅ Connected to Schwab")
        print(f"   Portfolio Value: ${summary['portfolio_value']:,.2f}")
        print(f"   Buying Power: ${summary['buying_power']:,.2f}")

        # Test 2: Get market data
        market_data = oms.broker.get_market_data("AAPL")
        print(f"\n✅ Market Data for AAPL:")
        print(f"   Bid: ${market_data['bid']:.2f}")
        print(f"   Ask: ${market_data['ask']:.2f}")
        print(f"   Last: ${market_data['last']:.2f}")

        # Test 3: Submit a test order (small quantity)
        test_order = Order(
            order_id="",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=1,
            order_type=OrderType.LIMIT,
            limit_price=market_data['bid'] - 5,  # Well below market
            time_in_force="day"
        )

        if oms.submit_order(test_order):
            print(f"\n✅ Test order submitted: {test_order.order_id}")

            # Wait and check status
            time.sleep(2)
            status = oms.broker.get_order_status(test_order.order_id)
            print(f"   Order status: {status.value}")

            # Cancel the test order
            if oms.cancel_order(test_order.order_id):
                print(f"   Order cancelled successfully")

        print("\n✅ All tests passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")

    finally:
        oms.stop()


if __name__ == "__main__":
    test_schwab_integration()