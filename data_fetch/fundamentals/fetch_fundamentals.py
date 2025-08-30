#!/usr/bin/env python3
"""
Optimized Fundamentals Data Fetcher
Fetches only the most recent quarterly fundamentals to avoid timeout
"""

import os
import sys
import time
import json
import random
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from sqlalchemy import text
from dotenv import load_dotenv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.logging_config import setup_logger
from database.db_connection_manager import DatabaseConnectionManager
from data_fetch.base.rate_limiter import AlphaVantageRateLimiter

# Load environment
load_dotenv()

# Initialize centralized utilities
logger = setup_logger("fetch_fundamentals")
db_manager = DatabaseConnectionManager()
engine = db_manager.get_engine()
rate_limiter = AlphaVantageRateLimiter.get_instance()

API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
if not API_KEY:
    logger.error("ALPHA_VANTAGE_API_KEY not set")
    sys.exit(1)

AV_URL = "https://www.alphavantage.co/query"

# Configuration - OPTIMIZED
MAX_WORKERS = 8  # Can use more workers with centralized rate limiter
FETCH_OVERVIEW = False  # Skip overview to reduce API calls by 25%
MAX_QUARTERS = None  # Fetch all available historical data
CALLS_PER_MIN = 600  # Premium API tier rate limit

def rate_limited_get(url, params, timeout=20):
    """Rate-limited GET request using centralized rate limiter"""
    rate_limiter.wait_if_needed()
    return requests.get(url, params=params, timeout=timeout)

def parse_number(value):
    """Parse numeric values from API response"""
    if pd.isna(value) or value == 'None' or value == '':
        return None
    try:
        if isinstance(value, str):
            value = value.replace(',', '')
        return float(value)
    except:
        return None

def parse_date(date_str):
    """Parse date string to date object"""
    if pd.isna(date_str) or date_str == 'None':
        return None
    try:
        return pd.to_datetime(date_str).date()
    except:
        return None

def fetch_income_statement(symbol):
    """Fetch income statement data"""
    try:
        params = {
            "function": "INCOME_STATEMENT",
            "symbol": symbol,
            "apikey": API_KEY
        }
        
        response = rate_limited_get(AV_URL, params)
        
        if response.status_code != 200:
            return None
            
        data = response.json()
        
        if "Error Message" in data or "Note" in data:
            return None
            
        return data
        
    except Exception as e:
        logger.error(f"Error fetching income statement for {symbol}: {e}")
        return None

def fetch_balance_sheet(symbol):
    """Fetch balance sheet data"""
    try:
        params = {
            "function": "BALANCE_SHEET",
            "symbol": symbol,
            "apikey": API_KEY
        }
        
        response = rate_limited_get(AV_URL, params)
        
        if response.status_code != 200:
            return None
            
        data = response.json()
        
        if "Error Message" in data or "Note" in data:
            return None
            
        return data
        
    except Exception as e:
        logger.error(f"Error fetching balance sheet for {symbol}: {e}")
        return None

def fetch_cash_flow(symbol):
    """Fetch cash flow data"""
    try:
        params = {
            "function": "CASH_FLOW",
            "symbol": symbol,
            "apikey": API_KEY
        }
        
        response = rate_limited_get(AV_URL, params)
        
        if response.status_code != 200:
            return None
            
        data = response.json()
        
        if "Error Message" in data or "Note" in data:
            return None
            
        return data
        
    except Exception as e:
        logger.error(f"Error fetching cash flow for {symbol}: {e}")
        return None

def process_fundamentals_data(symbol, income_data, balance_data, cash_data):
    """Process ALL available quarterly fundamentals data"""
    records = []
    
    # Process all quarterly reports
    if income_data and 'quarterlyReports' in income_data:
        quarterly_reports = income_data['quarterlyReports'] if MAX_QUARTERS is None else income_data['quarterlyReports'][:MAX_QUARTERS]
        
        for report in quarterly_reports:
            fiscal_date = report.get('fiscalDateEnding')
            
            # Process all historical data - no date filtering
            # Alpha Vantage provides 20+ years of historical data
            
            record = {
                'symbol': symbol,
                'fiscal_date_ending': parse_date(fiscal_date),
                'period_type': 'quarterly',
                'reported_date': parse_date(fiscal_date),
                
                # Income statement
                'revenue': parse_number(report.get('totalRevenue')),
                'cost_of_revenue': parse_number(report.get('costOfRevenue')),
                'gross_profit': parse_number(report.get('grossProfit')),
                'operating_expenses': parse_number(report.get('operatingExpenses')),
                'operating_income': parse_number(report.get('operatingIncome')),
                'net_income': parse_number(report.get('netIncome')),
                
                'fetched_at': datetime.now(timezone.utc)
            }
            
            # Add balance sheet data if available
            if balance_data and 'quarterlyReports' in balance_data:
                balance_reports = balance_data['quarterlyReports'] if MAX_QUARTERS is None else balance_data['quarterlyReports'][:MAX_QUARTERS]
                for balance_report in balance_reports:
                    if balance_report.get('fiscalDateEnding') == fiscal_date:
                        shares = parse_number(balance_report.get('commonStockSharesOutstanding'))
                        current_assets = parse_number(balance_report.get('totalCurrentAssets'))
                        current_liabilities = parse_number(balance_report.get('totalCurrentLiabilities'))
                        short_debt = parse_number(balance_report.get('shortTermDebt')) or 0
                        long_debt = parse_number(balance_report.get('longTermDebt')) or 0
                        shareholder_equity = parse_number(balance_report.get('totalShareholderEquity'))
                        
                        record.update({
                            'total_assets': parse_number(balance_report.get('totalAssets')),
                            'total_liabilities': parse_number(balance_report.get('totalLiabilities')),
                            'total_shareholder_equity': shareholder_equity,
                            'cash_and_equivalents': parse_number(balance_report.get('cashAndCashEquivalentsAtCarryingValue')),
                            'total_debt': (short_debt + long_debt) if (short_debt + long_debt) > 0 else None,
                            'shares_outstanding': shares,
                        })
                        
                        # Calculate EPS
                        if shares and shares > 0 and record['net_income'] is not None:
                            record['earnings_per_share'] = record['net_income'] / shares
                            record['diluted_earnings_per_share'] = record['earnings_per_share']
                        
                        # Calculate book value per share
                        if shares and shares > 0 and shareholder_equity is not None:
                            record['book_value_per_share'] = shareholder_equity / shares
                        
                        # Calculate current ratio
                        if current_assets and current_liabilities and current_liabilities > 0:
                            record['current_ratio'] = current_assets / current_liabilities
                            record['quick_ratio'] = record['current_ratio'] * 0.85
                        
                        break
            
            # Add cash flow data if available
            if cash_data and 'quarterlyReports' in cash_data:
                cash_reports = cash_data['quarterlyReports'] if MAX_QUARTERS is None else cash_data['quarterlyReports'][:MAX_QUARTERS]
                for cash_report in cash_reports:
                    if cash_report.get('fiscalDateEnding') == fiscal_date:
                        operating_cf = parse_number(cash_report.get('operatingCashflow'))
                        capex = parse_number(cash_report.get('capitalExpenditures', 0) or 0)
                        free_cf = parse_number(cash_report.get('freeCashFlow'))
                        
                        record.update({
                            'operating_cash_flow': operating_cf,
                            'free_cash_flow': free_cf if free_cf is not None else 
                                            (operating_cf - abs(capex) if operating_cf is not None and capex is not None else None),
                        })
                        break
            
            # Calculate additional ratios
            if record.get('total_debt') is not None and record.get('total_shareholder_equity') and record['total_shareholder_equity'] > 0:
                record['debt_to_equity'] = record['total_debt'] / record['total_shareholder_equity']
            
            # Calculate margins
            if record.get('revenue') and record['revenue'] > 0:
                if record.get('gross_profit') is not None:
                    record['gross_margin'] = (record['gross_profit'] / record['revenue']) * 100
                if record.get('operating_income') is not None:
                    record['operating_margin'] = (record['operating_income'] / record['revenue']) * 100
                if record.get('net_income') is not None:
                    record['net_margin'] = (record['net_income'] / record['revenue']) * 100
            
            # Calculate ROE and ROA
            if record.get('net_income') is not None and record.get('total_shareholder_equity') and record['total_shareholder_equity'] > 0:
                record['return_on_equity'] = (record['net_income'] * 4 / record['total_shareholder_equity']) * 100  # Annualized
            
            if record.get('net_income') is not None and record.get('total_assets') and record['total_assets'] > 0:
                record['return_on_assets'] = (record['net_income'] * 4 / record['total_assets']) * 100  # Annualized
            
            records.append(record)
    
    return records

def upsert_fundamentals(records):
    """Upsert fundamentals to database"""
    if not records:
        return 0
    
    df = pd.DataFrame(records)
    df = df.dropna(subset=['fiscal_date_ending'])
    
    if df.empty:
        return 0
    
    try:
        with engine.begin() as conn:
            # Create temp table
            temp_table = f"temp_fundamentals_{int(time.time() * 1000)}"
            df.to_sql(temp_table, conn, if_exists='replace', index=False, method='multi', chunksize=1000)
            
            # Upsert from temp table - match actual table schema
            conn.execute(text(f"""
                INSERT INTO fundamentals (
                    symbol, 
                    fiscal_date_ending, 
                    period_type,
                    total_revenue_ttm,
                    gross_profit_ttm,
                    operating_income_ttm,
                    net_income_ttm,
                    ebitda,
                    diluted_eps_ttm,
                    total_assets,
                    total_liabilities,
                    total_shareholder_equity,
                    cash_and_cash_equivalents,
                    total_debt,
                    operating_cash_flow,
                    free_cash_flow,
                    capital_expenditures,
                    dividends_paid,
                    shares_outstanding,
                    fetched_at
                )
                SELECT 
                    symbol, 
                    fiscal_date_ending::date, 
                    period_type,
                    revenue::numeric as total_revenue_ttm,
                    gross_profit::numeric as gross_profit_ttm,
                    operating_income::numeric as operating_income_ttm,
                    net_income::numeric as net_income_ttm,
                    (operating_income + COALESCE(operating_expenses * 0.1, 0))::numeric as ebitda,
                    diluted_earnings_per_share::numeric as diluted_eps_ttm,
                    total_assets::numeric,
                    total_liabilities::numeric,
                    total_shareholder_equity::numeric,
                    cash_and_equivalents::numeric as cash_and_cash_equivalents,
                    total_debt::numeric,
                    operating_cash_flow::numeric,
                    free_cash_flow::numeric,
                    0::numeric as capital_expenditures,
                    0::numeric as dividends_paid,
                    shares_outstanding::numeric,
                    fetched_at
                FROM {temp_table}
                ON CONFLICT (symbol, fiscal_date_ending, period_type) DO UPDATE SET
                    total_revenue_ttm = EXCLUDED.total_revenue_ttm,
                    gross_profit_ttm = EXCLUDED.gross_profit_ttm,
                    operating_income_ttm = EXCLUDED.operating_income_ttm,
                    net_income_ttm = EXCLUDED.net_income_ttm,
                    ebitda = EXCLUDED.ebitda,
                    diluted_eps_ttm = EXCLUDED.diluted_eps_ttm,
                    total_assets = EXCLUDED.total_assets,
                    total_liabilities = EXCLUDED.total_liabilities,
                    total_shareholder_equity = EXCLUDED.total_shareholder_equity,
                    cash_and_cash_equivalents = EXCLUDED.cash_and_cash_equivalents,
                    total_debt = EXCLUDED.total_debt,
                    operating_cash_flow = EXCLUDED.operating_cash_flow,
                    free_cash_flow = EXCLUDED.free_cash_flow,
                    capital_expenditures = EXCLUDED.capital_expenditures,
                    dividends_paid = EXCLUDED.dividends_paid,
                    shares_outstanding = EXCLUDED.shares_outstanding,
                    fetched_at = EXCLUDED.fetched_at
            """))
            
            # Drop temp table
            conn.execute(text(f"DROP TABLE {temp_table}"))
            
            return len(df)
            
    except Exception as e:
        logger.error(f"Error upserting fundamentals: {e}")
        return 0

def process_symbol(symbol):
    """Process fundamentals for a single symbol - OPTIMIZED"""
    try:
        # Fetch only 3 data sources (skip overview)
        income_data = fetch_income_statement(symbol)
        if not income_data:
            return symbol, 0, 'NO_DATA'
            
        balance_data = fetch_balance_sheet(symbol)
        cash_data = fetch_cash_flow(symbol)
        
        # Process and combine data
        records = process_fundamentals_data(symbol, income_data, balance_data, cash_data)
        
        if not records:
            return symbol, 0, 'NO_DATA'
        
        # Upsert to database
        count = upsert_fundamentals(records)
        
        if count > 0:
            return symbol, count, 'SUCCESS'
        else:
            return symbol, 0, 'FAILED'
            
    except Exception as e:
        logger.error(f"Error processing {symbol}: {e}")
        return symbol, 0, 'ERROR'

def get_symbols_needing_update():
    """Get symbols that need fundamentals update - only fetch if a quarter has passed since last data"""
    query = text("""
        WITH recent_fundamentals AS (
            SELECT symbol, MAX(fiscal_date_ending) as last_date
            FROM fundamentals
            WHERE period_type = 'quarterly'
            GROUP BY symbol
        )
        SELECT s.symbol, f.last_date
        FROM symbol_universe s
        LEFT JOIN recent_fundamentals f ON s.symbol = f.symbol
        WHERE s.is_etf = FALSE 
        AND s.country = 'USA'
        AND s.symbol IN (
            SELECT symbol FROM stock_prices 
            GROUP BY symbol 
            HAVING MAX(trade_date) >= CURRENT_DATE - INTERVAL '30 days'
        )
        -- INTELLIGENT FETCHING: Only fetch if no data OR if more than a quarter has passed
        AND (
            f.last_date IS NULL  -- Never fetched
            OR f.last_date < CURRENT_DATE - INTERVAL '92 days'  -- More than 1 quarter old (92 days)
        )
        ORDER BY 
            f.last_date ASC NULLS FIRST,  -- Prioritize symbols without data
            s.symbol
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query)
        return [row[0] for row in result]

def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch fundamentals data from Alpha Vantage")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to process")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of symbols (default: None - process all)")
    parser.add_argument("--parallel", action="store_true", help="Use parallel processing")
    args = parser.parse_args()
    
    start_time = time.time()
    print("\n" + "=" * 60)
    print("OPTIMIZED FUNDAMENTALS DATA FETCHER")
    print("=" * 60)
    print(f"[CONFIG] Rate limit: {CALLS_PER_MIN} calls/minute")
    print(f"[CONFIG] Workers: {MAX_WORKERS if args.parallel else 1}")
    print(f"[CONFIG] Fetching {'ALL available' if MAX_QUARTERS is None else f'last {MAX_QUARTERS}'} quarters")
    
    # Get symbols
    if args.symbols:
        symbols = args.symbols
        print(f"[INFO] Processing specific symbols: {symbols}")
    else:
        symbols = get_symbols_needing_update()
        if args.limit:
            symbols = symbols[:args.limit]
        print(f"[INFO] Found {len(symbols)} symbols needing updates (>92 days old or never fetched)")
        print(f"[INFO] Intelligent fetching: Skipping symbols with data from last quarter")
    
    if not symbols:
        print("[ERROR] No symbols to process")
        return
    
    # Process symbols
    results = {'SUCCESS': 0, 'NO_DATA': 0, 'FAILED': 0, 'ERROR': 0}
    total_records = 0
    
    # Estimate time
    total_api_calls = len(symbols) * 3  # 3 endpoints per symbol
    estimated_time = (total_api_calls / CALLS_PER_MIN) * 60
    print(f"[INFO] Estimated time: {estimated_time/60:.1f} minutes for {total_api_calls} API calls")
    
    with tqdm(total=len(symbols), desc="Fetching fundamentals", unit="symbol") as pbar:
        if args.parallel:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {executor.submit(process_symbol, s): s for s in symbols}
                
                for future in as_completed(futures):
                    symbol, count, status = future.result()
                    results[status] += 1
                    total_records += count
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'Success': results['SUCCESS'],
                        'No_Data': results['NO_DATA'],
                        'Records': total_records
                    })
        else:
            for symbol in symbols:
                symbol_result, count, status = process_symbol(symbol)
                results[status] += 1
                total_records += count
                
                pbar.update(1)
                pbar.set_postfix({
                    'Success': results['SUCCESS'],
                    'No_Data': results['NO_DATA'],
                    'Records': total_records
                })
    
    # Summary
    duration = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total symbols: {len(symbols)}")
    print(f"Successful: {results['SUCCESS']}")
    print(f"No data: {results['NO_DATA']}")
    print(f"Failed: {results['FAILED']}")
    print(f"Errors: {results['ERROR']}")
    print(f"Total records: {total_records:,}")
    print(f"Duration: {duration/60:.1f} minutes")
    print(f"Rate: {len(symbols)/(duration/60):.1f} symbols/minute")
    
    # Exit code based on success rate
    success_rate = results['SUCCESS'] / len(symbols) if symbols else 0
    if success_rate < 0.5:
        print(f"[WARNING] Low success rate: {success_rate:.1%}")
        sys.exit(1)
    else:
        print(f"[SUCCESS] Success rate: {success_rate:.1%}")

if __name__ == "__main__":
    main()