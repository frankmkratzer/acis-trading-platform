# =====================================
# 6. Performance Dashboard
# =====================================
"""
#!/usr/bin/env python3
# File: performance_dashboard.py
# Purpose: Generate performance reports and analytics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import warnings

warnings.filterwarnings('ignore')

load_dotenv()
engine = create_engine(os.getenv("POSTGRES_URL"))


class PerformanceDashboard:
    def __init__(self):
        self.strategies = ['value', 'growth', 'momentum', 'dividend']

    def calculate_strategy_performance(self, strategy, start_date=None, end_date=None):
        """Calculate performance metrics for a strategy"""

        # Get portfolio holdings over time
        query = text("""
                     WITH portfolio AS (SELECT p.symbol,
                                               p.as_of_date,
                                               p.score,
                                               p.rank,
                                               s.trade_date,
                                               s.adjusted_close as    price,
                                               LEAD(s.adjusted_close) OVER (
                        PARTITION BY p.symbol 
                        ORDER BY s.trade_date
                    ) as next_price
                                        FROM ai_{strategy} _portfolio p
                         JOIN stock_eod_daily s
                     ON p.symbol = s.symbol
                         AND s.trade_date >= p.as_of_date
                         AND s.trade_date < p.as_of_date + INTERVAL '30 days'
                     WHERE p.rank <= 20
                       AND (:start_date IS NULL
                        OR p.as_of_date >= :start_date)
                       AND (:end_date IS NULL
                        OR p.as_of_date <= :end_date)
                         )
                     SELECT symbol,
                            as_of_date,
                            trade_date,
                            price,
                            next_price,
                            (next_price / price - 1) as daily_return
                     FROM portfolio
                     WHERE next_price IS NOT NULL
                     ORDER BY trade_date
                     """.replace('{strategy}', strategy))

        df = pd.read_sql(
            query,
            engine,
            params={'start_date': start_date, 'end_date': end_date}
        )

        if df.empty:
            return None

        # Calculate portfolio returns (equal weight)
        daily_returns = df.groupby('trade_date')['daily_return'].mean()

        # Calculate metrics
        metrics = self._calculate_metrics(daily_returns)
        metrics['strategy'] = strategy

        return metrics

    def _calculate_metrics(self, returns):
        """Calculate performance metrics from returns series"""

        # Basic stats
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        annual_vol = returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0

        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Other metrics
        win_rate = (returns > 0).mean()
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = returns[returns < 0].mean() if (returns < 0).any() else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }

    def generate_report(self, output_path='reports/performance_report.html'):
        """Generate comprehensive performance report"""

        # Collect performance for all strategies
        results = []
        for strategy in self.strategies:
            metrics = self.calculate_strategy_performance(strategy)
            if metrics:
                results.append(metrics)

        if not results:
            print("No performance data available")
            return

        df = pd.DataFrame(results)

        # Create HTML report
        html = f"""
        <html>
        <head>
            <title>Strategy Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: right; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .positive {{ color: green; font-weight: bold; }}
                .negative {{ color: red; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>Strategy Performance Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

            <h2>Performance Summary</h2>
            <table>
                <tr>
                    <th>Strategy</th>
                    <th>Annual Return</th>
                    <th>Volatility</th>
                    <th>Sharpe Ratio</th>
                    <th>Max Drawdown</th>
                    <th>Win Rate</th>
                </tr>
        """

        for _, row in df.iterrows():
            return_class = 'positive' if row['annual_return'] > 0 else 'negative'
            dd_class = 'negative' if row['max_drawdown'] < -0.20 else ''

            html += f"""
                <tr>
                    <td style="text-align: left; font-weight: bold;">{row['strategy'].title()}</td>
                    <td class="{return_class}">{row['annual_return']:.2%}</td>
                    <td>{row['annual_volatility']:.2%}</td>
                    <td>{row['sharpe_ratio']:.2f}</td>
                    <td class="{dd_class}">{row['max_drawdown']:.2%}</td>
                    <td>{row['win_rate']:.2%}</td>
                </tr>
            """

        html += """
            </table>

            <h2>Risk-Adjusted Performance</h2>
            <p>Strategies are ranked by Sharpe Ratio (return per unit of risk)</p>

            <h2>Recommendations</h2>
            <ul>
        """

        # Add recommendations based on performance
        best_sharpe = df.loc[df['sharpe_ratio'].idxmax()]
        best_return = df.loc[df['annual_return'].idxmax()]
        lowest_risk = df.loc[df['annual_volatility'].idxmin()]

        html += f"""
                <li><strong>Best Risk-Adjusted:</strong> {best_sharpe['strategy'].title()} (Sharpe: {best_sharpe['sharpe_ratio']:.2f})</li>
                <li><strong>Highest Return:</strong> {best_return['strategy'].title()} ({best_return['annual_return']:.2%} annually)</li>
                <li><strong>Lowest Risk:</strong> {lowest_risk['strategy'].title()} ({lowest_risk['annual_volatility']:.2%} volatility)</li>
            </ul>
        </body>
        </html>
        """

        # Save report
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(html)

        print(f"✅ Performance report saved to {output_path}")

        # Also save metrics to CSV
        csv_path = output_path.replace('.html', '.csv')
        df.to_csv(csv_path, index=False)

        return df
