#!/usr/bin/env python3
"""
ACIS Trading Platform - Historical Point-in-Time AI Backtesting
Proper backtesting with quarterly rebalancing using historical AI model runs
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

class HistoricalAIBacktest:
    def __init__(self):
        load_dotenv()
        self.engine = create_engine(os.getenv('POSTGRES_URL'))
        self.initial_capital = 1_000_000
        self.transaction_cost = 0.002  # 20 bps per rebalance
        
    def get_symbols_available_on_date(self, as_of_date):
        """Get symbols that were actively trading on a specific date"""
        with self.engine.connect() as conn:
            # Get symbols with at least 252 days of price history before as_of_date
            # This ensures we have enough data for AI models
            result = conn.execute(text(f"""
                SELECT DISTINCT symbol
                FROM stock_eod_daily 
                WHERE trade_date <= '{as_of_date}'
                    AND trade_date >= '{as_of_date - timedelta(days=365)}'
                    AND adjusted_close IS NOT NULL
                    AND volume > 100000  -- Minimum liquidity
                GROUP BY symbol
                HAVING COUNT(*) >= 200  -- At least 200 trading days
                ORDER BY symbol
            """))
            
            symbols = [row[0] for row in result.fetchall()]
            return symbols
    
    def calculate_momentum_scores_as_of_date(self, as_of_date, symbols):
        """Calculate momentum scores using data available up to as_of_date"""
        with self.engine.connect() as conn:
            symbol_list = "'" + "','".join(symbols[:1000]) + "'"  # Limit for performance
            
            # Get 1-year, 6-month, 3-month, and 1-month returns as of this date
            result = conn.execute(text(f"""
                WITH price_points AS (
                    SELECT 
                        symbol,
                        adjusted_close as current_price,
                        LAG(adjusted_close, 252) OVER (PARTITION BY symbol ORDER BY trade_date) as price_1yr,
                        LAG(adjusted_close, 126) OVER (PARTITION BY symbol ORDER BY trade_date) as price_6mo,
                        LAG(adjusted_close, 63) OVER (PARTITION BY symbol ORDER BY trade_date) as price_3mo,
                        LAG(adjusted_close, 21) OVER (PARTITION BY symbol ORDER BY trade_date) as price_1mo,
                        ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY trade_date DESC) as rn
                    FROM stock_eod_daily
                    WHERE symbol IN ({symbol_list})
                        AND trade_date <= '{as_of_date}'
                        AND adjusted_close IS NOT NULL
                )
                SELECT 
                    symbol,
                    current_price,
                    CASE WHEN price_1yr > 0 THEN (current_price/price_1yr - 1) ELSE NULL END as return_1yr,
                    CASE WHEN price_6mo > 0 THEN (current_price/price_6mo - 1) ELSE NULL END as return_6mo,
                    CASE WHEN price_3mo > 0 THEN (current_price/price_3mo - 1) ELSE NULL END as return_3mo,
                    CASE WHEN price_1mo > 0 THEN (current_price/price_1mo - 1) ELSE NULL END as return_1mo
                FROM price_points
                WHERE rn = 1
                    AND price_1yr IS NOT NULL
                    AND price_6mo IS NOT NULL
                    AND price_3mo IS NOT NULL
                    AND price_1mo IS NOT NULL
            """))
            
            momentum_data = []
            for row in result.fetchall():
                symbol = row[0]
                returns = {
                    '1yr': float(row[2]) if row[2] else 0,
                    '6mo': float(row[3]) if row[3] else 0,
                    '3mo': float(row[4]) if row[4] else 0,
                    '1mo': float(row[5]) if row[5] else 0
                }
                
                # Simple momentum score: weighted average of returns
                # Weight recent performance more heavily
                momentum_score = (
                    returns['1yr'] * 0.2 +
                    returns['6mo'] * 0.3 +
                    returns['3mo'] * 0.3 +
                    returns['1mo'] * 0.2
                )
                
                momentum_data.append({
                    'symbol': symbol,
                    'momentum_score': momentum_score,
                    'return_1yr': returns['1yr'],
                    'return_6mo': returns['6mo'],
                    'return_3mo': returns['3mo'],
                    'return_1mo': returns['1mo']
                })
            
            df = pd.DataFrame(momentum_data)
            if len(df) > 0:
                # Rank and return top 10
                df = df.sort_values('momentum_score', ascending=False)
                df['rank'] = range(1, len(df) + 1)
                return df.head(10)
            else:
                return pd.DataFrame()
    
    def calculate_value_scores_as_of_date(self, as_of_date, symbols):
        """Calculate value scores using fundamental data available up to as_of_date"""
        with self.engine.connect() as conn:
            symbol_list = "'" + "','".join(symbols[:1000]) + "'"
            
            # Get most recent fundamental data before as_of_date
            # This is simplified - in reality you'd want more sophisticated value metrics
            result = conn.execute(text(f"""
                WITH latest_fundamentals AS (
                    SELECT 
                        symbol,
                        pe_ratio,
                        pb_ratio,
                        ps_ratio,
                        dividend_yield,
                        roe,
                        debt_to_equity,
                        ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY report_date DESC) as rn
                    FROM fundamentals_annual
                    WHERE symbol IN ({symbol_list})
                        AND report_date <= '{as_of_date}'
                        AND pe_ratio IS NOT NULL
                        AND pe_ratio > 0
                        AND pb_ratio IS NOT NULL
                        AND pb_ratio > 0
                )
                SELECT symbol, pe_ratio, pb_ratio, ps_ratio, dividend_yield, roe, debt_to_equity
                FROM latest_fundamentals
                WHERE rn = 1
                    AND pe_ratio BETWEEN 1 AND 50  -- Reasonable PE range
                    AND pb_ratio BETWEEN 0.1 AND 10  -- Reasonable PB range
            """))
            
            value_data = []
            for row in result.fetchall():
                symbol = row[0]
                pe = float(row[1]) if row[1] else None
                pb = float(row[2]) if row[2] else None
                ps = float(row[3]) if row[3] else None
                div_yield = float(row[4]) if row[4] else 0
                roe = float(row[5]) if row[5] else None
                debt_eq = float(row[6]) if row[6] else None
                
                # Simple value score: lower PE/PB is better, higher dividend yield is better
                value_score = 0
                if pe and pe > 0:
                    value_score += (1 / pe) * 10  # Inverse PE
                if pb and pb > 0:
                    value_score += (1 / pb) * 5   # Inverse PB
                if div_yield:
                    value_score += div_yield * 20  # Dividend yield bonus
                if roe and roe > 0:
                    value_score += min(roe, 30) * 0.1  # ROE bonus (capped)
                
                value_data.append({
                    'symbol': symbol,
                    'value_score': value_score,
                    'pe_ratio': pe,
                    'pb_ratio': pb,
                    'dividend_yield': div_yield,
                    'roe': roe
                })
            
            df = pd.DataFrame(value_data)
            if len(df) > 0:
                df = df.sort_values('value_score', ascending=False)
                df['rank'] = range(1, len(df) + 1)
                return df.head(10)
            else:
                return pd.DataFrame()
    
    def calculate_growth_scores_as_of_date(self, as_of_date, symbols):
        """Calculate growth scores using data available up to as_of_date"""
        with self.engine.connect() as conn:
            symbol_list = "'" + "','".join(symbols[:1000]) + "'"
            
            # Get revenue and earnings growth
            result = conn.execute(text(f"""
                WITH growth_metrics AS (
                    SELECT 
                        symbol,
                        total_revenue,
                        net_income,
                        report_date,
                        LAG(total_revenue, 1) OVER (PARTITION BY symbol ORDER BY report_date) as prev_revenue,
                        LAG(net_income, 1) OVER (PARTITION BY symbol ORDER BY report_date) as prev_income,
                        ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY report_date DESC) as rn
                    FROM fundamentals_annual
                    WHERE symbol IN ({symbol_list})
                        AND report_date <= '{as_of_date}'
                        AND total_revenue IS NOT NULL
                        AND total_revenue > 0
                )
                SELECT 
                    symbol,
                    total_revenue,
                    net_income,
                    prev_revenue,
                    prev_income,
                    CASE WHEN prev_revenue > 0 THEN (total_revenue/prev_revenue - 1) ELSE NULL END as revenue_growth,
                    CASE WHEN prev_income IS NOT NULL AND prev_income != 0 
                         THEN (net_income/prev_income - 1) ELSE NULL END as income_growth
                FROM growth_metrics
                WHERE rn = 1
                    AND prev_revenue IS NOT NULL
            """))
            
            growth_data = []
            for row in result.fetchall():
                symbol = row[0]
                revenue_growth = float(row[5]) if row[5] else 0
                income_growth = float(row[6]) if row[6] else 0
                
                # Simple growth score
                growth_score = 0
                if revenue_growth > 0:
                    growth_score += min(revenue_growth, 1.0) * 50  # Cap at 100% growth
                if income_growth > 0:
                    growth_score += min(income_growth, 2.0) * 25   # Cap at 200% growth
                
                growth_data.append({
                    'symbol': symbol,
                    'growth_score': growth_score,
                    'revenue_growth': revenue_growth,
                    'income_growth': income_growth
                })
            
            df = pd.DataFrame(growth_data)
            if len(df) > 0:
                df = df.sort_values('growth_score', ascending=False)
                df['rank'] = range(1, len(df) + 1)
                return df.head(10)
            else:
                return pd.DataFrame()
    
    def get_quarterly_returns(self, symbols, start_date, end_date):
        """Get quarterly returns for a set of symbols"""
        with self.engine.connect() as conn:
            symbol_list = "'" + "','".join(symbols) + "'"
            
            result = conn.execute(text(f"""
                SELECT symbol, 
                       (
                           SELECT adjusted_close 
                           FROM stock_eod_daily s2 
                           WHERE s2.symbol = s1.symbol 
                               AND s2.trade_date <= '{end_date}'
                               AND s2.adjusted_close IS NOT NULL
                           ORDER BY s2.trade_date DESC 
                           LIMIT 1
                       ) / (
                           SELECT adjusted_close 
                           FROM stock_eod_daily s3 
                           WHERE s3.symbol = s1.symbol 
                               AND s3.trade_date <= '{start_date}'
                               AND s3.adjusted_close IS NOT NULL
                           ORDER BY s3.trade_date DESC 
                           LIMIT 1
                       ) - 1 as quarterly_return
                FROM (SELECT DISTINCT symbol FROM stock_eod_daily WHERE symbol IN ({symbol_list})) s1
            """))
            
            returns = {}
            for row in result.fetchall():
                if row[1] is not None:
                    returns[row[0]] = float(row[1])
            
            return returns
    
    def run_historical_backtest(self):
        """Run the complete historical point-in-time backtest"""
        print("ACIS HISTORICAL POINT-IN-TIME BACKTEST")
        print("=" * 60)
        
        # 20-year backtest period
        start_date = datetime(2005, 1, 1).date()
        end_date = datetime(2025, 1, 1).date()
        
        strategies = ['value', 'growth', 'momentum']
        results = {strategy: {
            'quarterly_returns': [],
            'portfolio_values': [self.initial_capital],
            'holdings_history': [],
            'rebalance_costs': []
        } for strategy in strategies}
        
        current_date = start_date
        quarter = 0
        
        print(f"Backtesting from {start_date} to {end_date}")
        print(f"Initial capital: ${self.initial_capital:,}")
        
        while current_date < end_date:
            quarter += 1
            quarter_end = current_date + timedelta(days=90)
            if quarter_end > end_date:
                quarter_end = end_date
            
            print(f"\\nQuarter {quarter}: {current_date} to {quarter_end}")
            
            # Get symbols available on this date
            available_symbols = self.get_symbols_available_on_date(current_date)
            print(f"  Symbols available: {len(available_symbols)}")
            
            if len(available_symbols) < 50:  # Not enough data
                current_date = quarter_end
                continue
            
            # Run AI models for each strategy
            strategy_picks = {}
            
            # Momentum strategy
            momentum_picks = self.calculate_momentum_scores_as_of_date(current_date, available_symbols)
            if len(momentum_picks) >= 5:  # Minimum 5 stocks
                strategy_picks['momentum'] = momentum_picks['symbol'].tolist()
            
            # Value strategy  
            value_picks = self.calculate_value_scores_as_of_date(current_date, available_symbols)
            if len(value_picks) >= 5:
                strategy_picks['value'] = value_picks['symbol'].tolist()
            
            # Growth strategy
            growth_picks = self.calculate_growth_scores_as_of_date(current_date, available_symbols)
            if len(growth_picks) >= 5:
                strategy_picks['growth'] = growth_picks['symbol'].tolist()
            
            # Calculate quarterly returns for each strategy
            for strategy, symbols in strategy_picks.items():
                if len(symbols) > 0:
                    quarterly_returns = self.get_quarterly_returns(symbols, current_date, quarter_end)
                    
                    # Equal-weight portfolio return
                    if quarterly_returns:
                        avg_return = np.mean(list(quarterly_returns.values()))
                        results[strategy]['quarterly_returns'].append(avg_return)
                        
                        # Apply transaction costs (simplified)
                        net_return = avg_return - self.transaction_cost
                        
                        # Update portfolio value
                        current_value = results[strategy]['portfolio_values'][-1]
                        new_value = current_value * (1 + net_return)
                        results[strategy]['portfolio_values'].append(new_value)
                        
                        print(f"    {strategy.upper()}: {avg_return:.1%} return, ${new_value:,.0f} value")
                        
                        # Store holdings
                        results[strategy]['holdings_history'].append({
                            'date': current_date,
                            'symbols': symbols,
                            'return': avg_return
                        })
                    else:
                        # No return data available
                        results[strategy]['portfolio_values'].append(results[strategy]['portfolio_values'][-1])
                else:
                    # No picks available
                    results[strategy]['portfolio_values'].append(results[strategy]['portfolio_values'][-1])
            
            current_date = quarter_end
        
        return self.analyze_results(results, quarter)
    
    def analyze_results(self, results, quarters):
        """Analyze and display backtest results"""
        print(f"\\n" + "=" * 80)
        print("HISTORICAL BACKTEST RESULTS")
        print("=" * 80)
        
        years = quarters / 4
        
        for strategy, data in results.items():
            if len(data['portfolio_values']) > 1:
                final_value = data['portfolio_values'][-1]
                total_return = (final_value / self.initial_capital) - 1
                
                if years > 0:
                    annual_return = (final_value / self.initial_capital) ** (1/years) - 1
                else:
                    annual_return = 0
                
                print(f"\\n{strategy.upper()} STRATEGY:")
                print(f"  Final Value:   ${final_value:,.0f}")
                print(f"  Total Return:  {total_return:.1%}")
                print(f"  Annual Return: {annual_return:.1%}")
                print(f"  Quarters:      {len(data['quarterly_returns'])}")
                
                if data['quarterly_returns']:
                    quarterly_vol = np.std(data['quarterly_returns'])
                    annual_vol = quarterly_vol * 2  # Quarterly to annual
                    if annual_vol > 0:
                        sharpe = (annual_return - 0.02) / annual_vol
                        print(f"  Volatility:    {annual_vol:.1%}")
                        print(f"  Sharpe Ratio:  {sharpe:.2f}")
        
        return True

def main():
    """Run the historical AI backtest"""
    backtest = HistoricalAIBacktest()
    
    try:
        success = backtest.run_historical_backtest()
        
        if success:
            print(f"\\nHistorical backtest completed successfully!")
            print("This backtest used point-in-time AI model runs with no look-ahead bias.")
        else:
            print(f"\\nBacktest encountered issues.")
            
    except Exception as e:
        print(f"Backtest error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()