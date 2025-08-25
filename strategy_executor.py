# =====================================
# 4. LIVE TRADING STRATEGY EXECUTOR
# =====================================
"""
#!/usr/bin/env python3
# File: strategy_executor.py
# Purpose: Execute strategies in live trading with real-time monitoring
"""


class StrategyExecutor:
    """Execute trading strategies in production"""

    def __init__(self, oms: OrderManagementSystem, risk_manager: AdvancedRiskManager):
        self.oms = oms
        self.risk_manager = risk_manager
        self.strategies = {}
        self.active = False
        self.execution_thread = None

    def add_strategy(self, name: str, strategy):
        """Add a strategy to execute"""
        self.strategies[name] = strategy
        logger.info(f"Added strategy: {name}")

    def start(self):
        """Start strategy execution"""
        self.active = True
        self.execution_thread = threading.Thread(target=self._execution_loop)
        self.execution_thread.start()
        logger.info("Strategy executor started")

    def stop(self):
        """Stop strategy execution"""
        self.active = False
        if self.execution_thread:
            self.execution_thread.join()
        logger.info("Strategy executor stopped")

    def _execution_loop(self):
        """Main execution loop"""

        while self.active:
            try:
                current_time = datetime.now()

                # Check if market is open
                if not self._is_market_open():
                    time.sleep(60)
                    continue

                # Get current positions
                positions = self.oms.broker.get_positions()
                account_info = self.oms.broker.get_account_info()

                # Run each strategy
                for name, strategy in self.strategies.items():
                    try:
                        # Generate signals
                        signals = strategy.generate_signals(current_time)

                        # Apply risk management
                        adjusted_signals = self._apply_risk_management(
                            signals,
                            positions,
                            account_info
                        )

                        # Generate orders
                        orders = self._generate_orders(
                            adjusted_signals,
                            positions,
                            account_info
                        )

                        # Submit orders
                        for order in orders:
                            self.oms.submit_order(order)

                        logger.info(f"Executed strategy {name}: {len(orders)} orders")

                    except Exception as e:
                        logger.error(f"Error executing strategy {name}: {e}")

                # Sleep until next execution
                time.sleep(60)  # Execute every minute

            except Exception as e:
                logger.error(f"Error in execution loop: {e}")
                time.sleep(60)

    def _is_market_open(self) -> bool:
        """Check if market is open"""
        now = datetime.now()

        # Simple check for US market hours (9:30 AM - 4:00 PM ET)
        if now.weekday() >= 5:  # Weekend
            return False

        market_open = now.replace(hour=9, minute=30, second=0)
        market_close = now.replace(hour=16, minute=0, second=0)

        return market_open <= now <= market_close

    def _apply_risk_management(self,
                               signals: pd.DataFrame,
                               positions: List[Position],
                               account_info: Dict) -> pd.DataFrame:
        """Apply risk management to signals"""

        adjusted = signals.copy()

        # Calculate current exposure
        total_exposure = sum(p.market_value for p in positions)
        portfolio_value = account_info.get('portfolio_value', 0)

        # Apply position limits
        for symbol in adjusted.index:
            # Check existing position
            existing = next((p for p in positions if p.symbol == symbol), None)

            if existing:
                current_weight = existing.market_value / portfolio_value if portfolio_value > 0 else 0

                # Limit additional exposure
                max_additional = 0.10 - current_weight  # Max 10% per position
                adjusted.loc[symbol, 'weight'] = min(
                    adjusted.loc[symbol, 'weight'],
                    max_additional
                )

        # Apply portfolio-level constraints
        total_target = adjusted['weight'].sum()
        if total_target > 1.0:
            adjusted['weight'] = adjusted['weight'] / total_target

        return adjusted

    def _generate_orders(self,
                         signals: pd.DataFrame,
                         positions: List[Position],
                         account_info: Dict) -> List[Order]:
        """Generate orders from signals"""

        orders = []
        portfolio_value = account_info.get('portfolio_value', 0)

        for symbol, row in signals.iterrows():
            target_weight = row['weight']
            target_value = portfolio_value * target_weight

            # Get current position
            current_pos = next((p for p in positions if p.symbol == symbol), None)
            current_value = current_pos.market_value if current_pos else 0

            # Calculate trade value
            trade_value = target_value - current_value

            # Skip small trades
            if abs(trade_value) < 100:  # Min trade size $100
                continue

            # Get current price
            market_data = self.oms.broker.get_market_data(symbol)
            current_price = market_data.get('last', 0)

            if current_price <= 0:
                continue

            # Calculate shares
            shares = int(trade_value / current_price)

            if shares == 0:
                continue

            # Create order
            order = Order(
                order_id=f"{symbol}_{datetime.now().timestamp()}",
                symbol=symbol,
                side=OrderSide.BUY if shares > 0 else OrderSide.SELL,
                quantity=abs(shares),
                order_type=OrderType.LIMIT,
                limit_price=current_price * (1.001 if shares > 0 else 0.999),  # Slight buffer
                time_in_force='day'
            )

            orders.append(order)

        return orders


# =====================================
# 5. REAL-TIME MONITORING DASHBOARD
# =====================================
"""
#!/usr/bin/env python3
# File: realtime_dashboard.py
# Purpose: Real-time monitoring dashboard for live trading
"""

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.express as px


class TradingDashboard:
    """Real-time trading dashboard"""

    def __init__(self, oms: OrderManagementSystem):
        self.oms = oms
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self):
        """Setup dashboard layout"""

        self.app.layout = html.Div([
            html.H1("Live Trading Dashboard"),

            # Account summary
            html.Div(id='account-summary', className='summary-section'),

            # Risk metrics
            html.Div(id='risk-metrics', className='risk-section'),

            # Positions table
            html.H2("Current Positions"),
            html.Div(id='positions-table'),

            # Orders table
            html.H2("Recent Orders"),
            html.Div(id='orders-table'),

            # P&L chart
            html.H2("P&L Performance"),
            dcc.Graph(id='pnl-chart'),

            # Position allocation
            html.H2("Position Allocation"),
            dcc.Graph(id='allocation-chart'),

            # Auto-refresh
            dcc.Interval(id='interval-component', interval=5000)  # 5 seconds
        ])

    def setup_callbacks(self):
        """Setup dashboard callbacks"""

        @self.app.callback(
            [Output('account-summary', 'children'),
             Output('risk-metrics', 'children'),
             Output('positions-table', 'children'),
             Output('orders-table', 'children'),
             Output('pnl-chart', 'figure'),
             Output('allocation-chart', 'figure')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            account = self._get_account_summary()
            risk = self._get_risk_metrics()
            positions = self._get_positions_table()
            orders = self._get_orders_table()
            pnl_fig = self._get_pnl_chart()
            alloc_fig = self._get_allocation_chart()

            return account, risk, positions, orders, pnl_fig, alloc_fig

    def _get_account_summary(self):
        """Get account summary"""
        account = self.oms.broker.get_account_info()

        return html.Div([
            html.H3("Account Summary"),
            html.P(f"Portfolio Value: ${account.get('portfolio_value', 0):,.2f}"),
            html.P(f"Cash: ${account.get('cash', 0):,.2f}"),
            html.P(f"Buying Power: ${account.get('buying_power', 0):,.2f}"),
            html.P(f"Daily P&L: ${self.oms.daily_pnl:,.2f}",
                   className='positive' if self.oms.daily_pnl >= 0 else 'negative')
        ])

    def _get_risk_metrics(self):
        """Get risk metrics"""
        # Calculate current risk metrics
        positions = self.oms.broker.get_positions()

        total_exposure = sum(p.market_value for p in positions)
        max_position = max((p.market_value for p in positions), default=0)

        return html.Div([
            html.H3("Risk Metrics"),
            html.P(f"Total Exposure: ${total_exposure:,.2f}"),
            html.P(f"Largest Position: ${max_position:,.2f}"),
            html.P(f"Number of Positions: {len(positions)}"),
            html.P(f"Orders Today: {self.oms.order_count}")
        ])

    def _get_positions_table(self):
        """Get positions table"""
        positions = self.oms.broker.get_positions()

        data = []
        for p in positions:
            data.append({
                'Symbol': p.symbol,
                'Quantity': p.quantity,
                'Avg Cost': f"${p.avg_cost:.2f}",
                'Current': f"${p.current_price:.2f}",
                'Market Value': f"${p.market_value:,.2f}",
                'Unrealized P&L': f"${p.unrealized_pnl:,.2f}",
                'Realized P&L': f"${p.realized_pnl:,.2f}"
            })

        return dash_table.DataTable(
            data=data,
            columns=[{'name': c, 'id': c} for c in data[0].keys()] if data else [],
            style_cell={'textAlign': 'right'},
            style_data_conditional=[
                {
                    'if': {'column_id': 'Unrealized P&L'},
                    'color': 'green'
                }
            ]
        )

    def _get_orders_table(self):
        """Get orders table"""
        # Get recent orders
        orders = list(self.oms.orders.values())[-20:]  # Last 20 orders

        data = []
        for o in orders:
            data.append({
                'Time': o.submitted_at.strftime('%H:%M:%S') if o.submitted_at else '',
                'Symbol': o.symbol,
                'Side': o.side.value,
                'Quantity': o.quantity,
                'Type': o.order_type.value,
                'Price': f"${o.limit_price:.2f}" if o.limit_price else 'Market',
                'Status': o.status.value,
                'Filled': o.filled_quantity
            })

        return dash_table.DataTable(
            data=data,
            columns=[{'name': c, 'id': c} for c in data[0].keys()] if data else [],
            style_cell={'textAlign': 'center'}
        )

    def _get_pnl_chart(self):
        """Get P&L chart"""
        # This would typically load from database
        # Placeholder data
        dates = pd.date_range(end=datetime.now(), periods=30)
        pnl = np.cumsum(np.random.randn(30) * 1000)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=pnl,
            mode='lines',
            name='Cumulative P&L',
            line=dict(color='green' if pnl[-1] >= 0 else 'red')
        ))

        fig.update_layout(
            title='Cumulative P&L',
            xaxis_title='Date',
            yaxis_title='P&L ($)',
            hovermode='x'
        )

        return fig

    def _get_allocation_chart(self):
        """Get allocation pie chart"""
        positions = self.oms.broker.get_positions()

        if positions:
            symbols = [p.symbol for p in positions]
            values = [p.market_value for p in positions]

            fig = px.pie(
                values=values,
                names=symbols,
                title='Portfolio Allocation'
            )
        else:
            fig = go.Figure()
            fig.add_annotation(
                text="No positions",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False
            )

        return fig

    def run(self, port=8050):
        """Run the dashboard"""
        self.app.run_server(host='0.0.0.0', port=port, debug=False)


# =====================================
# MAIN EXECUTION
# =====================================

def main():
    """Main execution for live trading"""

    # Initialize components
    broker = AlpacaBroker()  # or IBBroker()
    oms = OrderManagementSystem(broker)
    risk_manager = AdvancedRiskManager()

    # Start OMS
    if not oms.start():
        logger.error("Failed to start OMS")
        return

    # Initialize strategy executor
    executor = StrategyExecutor(oms, risk_manager)

    # Add strategies
    # executor.add_strategy('value', ValueStrategy())
    # executor.add_strategy('momentum', MomentumStrategy())

    # Start executor
    executor.start()

    # Start dashboard
    dashboard = TradingDashboard(oms)

    try:
        # Run dashboard (blocks)
        dashboard.run()
    finally:
        # Cleanup
        executor.stop()
        oms.stop()


if __name__ == "__main__":
    main()