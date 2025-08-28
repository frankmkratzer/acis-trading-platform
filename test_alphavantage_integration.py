#!/usr/bin/env python3
"""
Test Alpha Vantage Integration
Shows how to fetch and use Alpha Vantage data in your trading platform
"""

import os
import json
from datetime import datetime
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import requests

load_dotenv()

# Your Alpha Vantage API key should be in .env file
API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', 'demo')
BASE_URL = 'https://www.alphavantage.co/query'

def test_quote(symbol='AAPL'):
    """Get real-time quote for a symbol"""
    params = {
        'function': 'GLOBAL_QUOTE',
        'symbol': symbol,
        'apikey': API_KEY
    }
    
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    
    if 'Global Quote' in data:
        quote = data['Global Quote']
        print(f"\n{symbol} Quote:")
        print(f"  Price: ${quote['05. price']}")
        print(f"  Change: {quote['10. change percent']}")
        print(f"  Volume: {int(quote['06. volume']):,}")
        print(f"  Date: {quote['07. latest trading day']}")
        return quote
    else:
        print(f"Error fetching quote: {data}")
        return None

def test_daily_prices(symbol='AAPL'):
    """Get daily adjusted prices"""
    params = {
        'function': 'TIME_SERIES_DAILY_ADJUSTED',
        'symbol': symbol,
        'outputsize': 'compact',  # Last 100 days
        'apikey': API_KEY
    }
    
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    
    if 'Time Series (Daily)' in data:
        time_series = data['Time Series (Daily)']
        latest_date = list(time_series.keys())[0]
        latest_data = time_series[latest_date]
        
        print(f"\n{symbol} Daily Data ({latest_date}):")
        print(f"  Open: ${latest_data['1. open']}")
        print(f"  High: ${latest_data['2. high']}")
        print(f"  Low: ${latest_data['3. low']}")
        print(f"  Close: ${latest_data['4. close']}")
        print(f"  Adjusted Close: ${latest_data['5. adjusted close']}")
        print(f"  Volume: {int(latest_data['6. volume']):,}")
        
        return time_series
    else:
        print(f"Error fetching daily data: {data}")
        return None

def test_technical_indicator(symbol='AAPL', indicator='RSI'):
    """Get technical indicator (RSI example)"""
    params = {
        'function': indicator,
        'symbol': symbol,
        'interval': 'daily',
        'time_period': 14,
        'series_type': 'close',
        'apikey': API_KEY
    }
    
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    
    if f'Technical Analysis: {indicator}' in data:
        ta_data = data[f'Technical Analysis: {indicator}']
        latest_date = list(ta_data.keys())[0]
        latest_value = ta_data[latest_date][indicator]
        
        print(f"\n{symbol} {indicator}:")
        print(f"  Date: {latest_date}")
        print(f"  {indicator}: {latest_value}")
        
        # Show last 5 values
        print(f"\n  Last 5 {indicator} values:")
        for i, (date, values) in enumerate(list(ta_data.items())[:5]):
            print(f"    {date}: {values[indicator]}")
        
        return ta_data
    else:
        print(f"Error fetching {indicator}: {data}")
        return None

def test_earnings(symbol='AAPL'):
    """Get earnings data"""
    params = {
        'function': 'EARNINGS',
        'symbol': symbol,
        'apikey': API_KEY
    }
    
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    
    if 'quarterlyEarnings' in data:
        quarterly = data['quarterlyEarnings']
        
        print(f"\n{symbol} Recent Earnings:")
        for q in quarterly[:4]:  # Last 4 quarters
            print(f"  {q['fiscalDateEnding']}:")
            print(f"    EPS: ${q.get('reportedEPS', 'N/A')}")
            print(f"    Estimated: ${q.get('estimatedEPS', 'N/A')}")
            print(f"    Surprise: {q.get('surprise', 'N/A')}")
        
        return quarterly
    else:
        print(f"Error fetching earnings: {data}")
        return None

def test_company_overview(symbol='AAPL'):
    """Get company fundamentals"""
    params = {
        'function': 'OVERVIEW',
        'symbol': symbol,
        'apikey': API_KEY
    }
    
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    
    if 'Symbol' in data:
        print(f"\n{symbol} Company Overview:")
        print(f"  Name: {data.get('Name', 'N/A')}")
        print(f"  Sector: {data.get('Sector', 'N/A')}")
        print(f"  Industry: {data.get('Industry', 'N/A')}")
        print(f"  Market Cap: ${int(data.get('MarketCapitalization', 0)):,}")
        print(f"  P/E Ratio: {data.get('PERatio', 'N/A')}")
        print(f"  Dividend Yield: {data.get('DividendYield', 'N/A')}%")
        print(f"  52 Week High: ${data.get('52WeekHigh', 'N/A')}")
        print(f"  52 Week Low: ${data.get('52WeekLow', 'N/A')}")
        
        return data
    else:
        print(f"Error fetching overview: {data}")
        return None

def save_to_database(symbol, data_type, data):
    """Save Alpha Vantage data to your database"""
    engine = create_engine(os.getenv("POSTGRES_URL"))
    
    try:
        with engine.begin() as conn:
            if data_type == 'quote':
                # Save to stock_prices table
                sql = text("""
                    INSERT INTO stock_prices (
                        symbol, trade_date, open, high, low, close, volume
                    ) VALUES (
                        :symbol, :trade_date, :open, :high, :low, :close, :volume
                    )
                    ON CONFLICT (symbol, trade_date) DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume
                """)
                
                conn.execute(sql, {
                    'symbol': symbol,
                    'trade_date': data['07. latest trading day'],
                    'open': float(data['02. open']),
                    'high': float(data['03. high']),
                    'low': float(data['04. low']),
                    'close': float(data['05. price']),
                    'volume': int(data['06. volume'])
                })
                
                print(f"Saved {symbol} quote to database")
                
    except Exception as e:
        print(f"Database error: {e}")

def main():
    """Run all tests"""
    print("=" * 60)
    print("Alpha Vantage Integration Test")
    print("=" * 60)
    
    symbol = 'AAPL'  # Test with Apple
    
    # Test different API functions
    quote = test_quote(symbol)
    daily = test_daily_prices(symbol)
    rsi = test_technical_indicator(symbol, 'RSI')
    earnings = test_earnings(symbol)
    overview = test_company_overview(symbol)
    
    # Optionally save to database
    # if quote:
    #     save_to_database(symbol, 'quote', quote)
    
    print("\n" + "=" * 60)
    print("Available Alpha Vantage Functions:")
    print("=" * 60)
    print("""
    Core Stock APIs:
    - TIME_SERIES_INTRADAY: Intraday (1min, 5min, 15min, etc.)
    - TIME_SERIES_DAILY: Daily prices
    - TIME_SERIES_DAILY_ADJUSTED: Daily adjusted prices
    - TIME_SERIES_WEEKLY: Weekly prices
    - TIME_SERIES_MONTHLY: Monthly prices
    - GLOBAL_QUOTE: Latest price
    - SYMBOL_SEARCH: Search for symbols
    
    Fundamental Data:
    - OVERVIEW: Company information
    - EARNINGS: Earnings history
    - INCOME_STATEMENT: Income statements
    - BALANCE_SHEET: Balance sheets
    - CASH_FLOW: Cash flow statements
    
    Technical Indicators:
    - SMA, EMA, WMA, DEMA, TEMA: Moving averages
    - RSI: Relative Strength Index
    - MACD: Moving Average Convergence Divergence
    - STOCH: Stochastic Oscillator
    - ADX: Average Directional Index
    - CCI: Commodity Channel Index
    - BBANDS: Bollinger Bands
    - And many more...
    
    Economic Indicators:
    - REAL_GDP: Real GDP
    - TREASURY_YIELD: Treasury yields
    - FEDERAL_FUNDS_RATE: Federal funds rate
    - CPI: Consumer Price Index
    - INFLATION: Inflation rates
    - UNEMPLOYMENT: Unemployment rate
    """)

if __name__ == "__main__":
    main()