#!/usr/bin/env python3
"""
Fixed Fundamentals Data Fetcher
Fetches quarterly fundamentals from Alpha Vantage with proper field mappings
"""

import os
import sys
import time
import json
import random
import logging
import threading
import requests
import pandas as pd
from datetime import datetime, timezone
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Load environment
load_dotenv()
API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
POSTGRES_URL = os.getenv("POSTGRES_URL")

if not API_KEY:
    print("[ERROR] ALPHA_VANTAGE_API_KEY not set")
    sys.exit(1)

if not POSTGRES_URL:
    print("[ERROR] POSTGRES_URL not set")
    sys.exit(1)

engine = create_engine(POSTGRES_URL)
AV_URL = "https://www.alphavantage.co/query"

# Configuration
MAX_WORKERS = int(os.getenv("AV_FUND_THREADS", "8"))
CALLS_PER_MIN = 580  # Slightly below 600 for safety
MIN_INTERVAL = 60.0 / CALLS_PER_MIN

# Logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/fetch_fundamentals.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("fetch_fundamentals")

# Simple rate limiter
last_call_time = 0
rate_limit_lock = threading.Lock()

def rate_limited_get(url, params, timeout=20):
    """Rate-limited GET request"""
    global last_call_time
    
    with rate_limit_lock:
        current_time = time.time()
        time_since_last = current_time - last_call_time
        if time_since_last < MIN_INTERVAL:
            sleep_time = MIN_INTERVAL - time_since_last
            time.sleep(sleep_time)
        last_call_time = time.time()
    
    return requests.get(url, params=params, timeout=timeout)

def parse_number(value):
    """Parse numeric values from API response"""
    if pd.isna(value) or value == 'None' or value == '':
        return None
    try:
        # Remove commas and convert
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
            logger.warning(f"HTTP {response.status_code} for {symbol} income statement")
            return None
            
        data = response.json()
        
        # Check for errors
        if "Error Message" in data:
            logger.warning(f"API error for {symbol}: {data['Error Message']}")
            return None
        
        if "Note" in data:
            logger.warning(f"Rate limit for {symbol}: {data['Note']}")
            time.sleep(60)
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
            logger.warning(f"HTTP {response.status_code} for {symbol} balance sheet")
            return None
            
        data = response.json()
        
        # Check for errors
        if "Error Message" in data:
            logger.warning(f"API error for {symbol}: {data['Error Message']}")
            return None
        
        if "Note" in data:
            logger.warning(f"Rate limit for {symbol}: {data['Note']}")
            time.sleep(60)
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
            logger.warning(f"HTTP {response.status_code} for {symbol} cash flow")
            return None
            
        data = response.json()
        
        # Check for errors
        if "Error Message" in data:
            logger.warning(f"API error for {symbol}: {data['Error Message']}")
            return None
        
        if "Note" in data:
            logger.warning(f"Rate limit for {symbol}: {data['Note']}")
            time.sleep(60)
            return None
            
        return data
        
    except Exception as e:
        logger.error(f"Error fetching cash flow for {symbol}: {e}")
        return None

def fetch_overview(symbol):
    """Fetch company overview with key ratios"""
    try:
        params = {
            "function": "OVERVIEW",
            "symbol": symbol,
            "apikey": API_KEY
        }
        
        response = rate_limited_get(AV_URL, params)
        
        if response.status_code != 200:
            logger.warning(f"HTTP {response.status_code} for {symbol} overview")
            return None
            
        data = response.json()
        
        # Check for errors
        if "Error Message" in data:
            logger.warning(f"API error for {symbol}: {data['Error Message']}")
            return None
        
        if "Note" in data:
            logger.warning(f"Rate limit for {symbol}: {data['Note']}")
            time.sleep(60)
            return None
            
        return data
        
    except Exception as e:
        logger.error(f"Error fetching overview for {symbol}: {e}")
        return None

def process_fundamentals_data(symbol, income_data, balance_data, cash_data, overview_data):
    """Combine and process ALL historical fundamentals data"""
    records = []
    
    # Process ALL quarterly reports (no limit - get full history)
    if income_data and 'quarterlyReports' in income_data:
        quarterly_reports = income_data['quarterlyReports']  # Get ALL quarterly data
        
        for report in quarterly_reports:
            fiscal_date = report.get('fiscalDateEnding')
            
            record = {
                'symbol': symbol,
                'fiscal_date_ending': parse_date(fiscal_date),
                'period_type': 'quarterly',
                'reported_date': parse_date(fiscal_date),  # Alpha Vantage doesn't provide separate reported date
                
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
                for balance_report in balance_data['quarterlyReports']:
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
                        
                        # Calculate EPS if we have the data
                        if shares and shares > 0 and record['net_income'] is not None:
                            record['earnings_per_share'] = record['net_income'] / shares
                            record['diluted_earnings_per_share'] = record['earnings_per_share']
                        else:
                            record['earnings_per_share'] = None
                            record['diluted_earnings_per_share'] = None
                        
                        # Calculate book value per share
                        if shares and shares > 0 and shareholder_equity is not None:
                            record['book_value_per_share'] = shareholder_equity / shares
                        else:
                            record['book_value_per_share'] = None
                        
                        # Calculate current ratio
                        if current_assets and current_liabilities and current_liabilities > 0:
                            record['current_ratio'] = current_assets / current_liabilities
                            # Quick ratio approximation (no inventory data from Alpha Vantage)
                            record['quick_ratio'] = record['current_ratio'] * 0.85
                        else:
                            record['current_ratio'] = None
                            record['quick_ratio'] = None
                        
                        break
            
            # Add cash flow data if available
            if cash_data and 'quarterlyReports' in cash_data:
                for cash_report in cash_data['quarterlyReports']:
                    if cash_report.get('fiscalDateEnding') == fiscal_date:
                        operating_cf = parse_number(cash_report.get('operatingCashflow'))
                        capex = parse_number(cash_report.get('capitalExpenditures', 0) or 0)
                        free_cf = parse_number(cash_report.get('freeCashFlow'))
                        
                        record.update({
                            'operating_cash_flow': operating_cf,
                            # Calculate free cash flow if not provided
                            'free_cash_flow': free_cf if free_cf is not None else 
                                            (operating_cf - abs(capex) if operating_cf is not None and capex is not None else None),
                        })
                        break
            
            # Add overview metrics (these are company-wide, not period-specific)
            if overview_data:
                record.update({
                    'pe_ratio': parse_number(overview_data.get('PERatio')),
                    'peg_ratio': parse_number(overview_data.get('PEGRatio')),
                    'pb_ratio': parse_number(overview_data.get('PriceToBookRatio')),
                    'ps_ratio': parse_number(overview_data.get('PriceToSalesRatioTTM')),
                    'ev_to_revenue': parse_number(overview_data.get('EVToRevenue')),
                    'ev_to_ebitda': parse_number(overview_data.get('EVToEBITDA')),
                    'return_on_equity': parse_number(overview_data.get('ReturnOnEquityTTM')),
                    'return_on_assets': parse_number(overview_data.get('ReturnOnAssetsTTM')),
                    'gross_margin': parse_number(overview_data.get('GrossProfitTTM')) / parse_number(overview_data.get('RevenueTTM')) * 100 if overview_data.get('RevenueTTM') and overview_data.get('GrossProfitTTM') else None,
                    'operating_margin': parse_number(overview_data.get('OperatingMarginTTM')),
                    'net_margin': parse_number(overview_data.get('ProfitMargin')),
                })
            
            # Calculate additional ratios
            if record.get('total_debt') is not None and record.get('total_shareholder_equity') and record['total_shareholder_equity'] > 0:
                record['debt_to_equity'] = record['total_debt'] / record['total_shareholder_equity']
            else:
                record['debt_to_equity'] = None
            
            # Calculate ROIC (approximation)
            if record.get('operating_income') and record.get('total_assets') and record.get('cash_and_equivalents'):
                invested_capital = record['total_assets'] - record.get('cash_and_equivalents', 0)
                if invested_capital > 0:
                    # Annualize quarterly operating income
                    record['return_on_invested_capital'] = (record['operating_income'] * 4) / invested_capital * 100
                else:
                    record['return_on_invested_capital'] = None
            else:
                record['return_on_invested_capital'] = None
            
            records.append(record)
    
    # ALSO process ALL annual reports for complete historical coverage
    if income_data and 'annualReports' in income_data:
        annual_reports = income_data['annualReports']  # Get ALL annual data
        
        for report in annual_reports:
            fiscal_date = report.get('fiscalDateEnding')
            
            record = {
                'symbol': symbol,
                'fiscal_date_ending': parse_date(fiscal_date),
                'period_type': 'annual',
                'reported_date': parse_date(fiscal_date),  # Alpha Vantage doesn't provide separate reported date
                
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
            if balance_data and 'annualReports' in balance_data:
                for balance_report in balance_data['annualReports']:
                    if balance_report.get('fiscalDateEnding') == fiscal_date:
                        shares = parse_number(balance_report.get('commonStockSharesOutstanding'))
                        shareholder_equity = parse_number(balance_report.get('totalShareholderEquity'))
                        total_assets = parse_number(balance_report.get('totalAssets'))
                        total_liabilities = parse_number(balance_report.get('totalLiabilities'))
                        
                        record.update({
                            'total_assets': total_assets,
                            'total_liabilities': total_liabilities,
                            'total_shareholder_equity': shareholder_equity,
                            'cash_and_equivalents': parse_number(balance_report.get('cashAndCashEquivalentsAtCarryingValue')),
                            'total_debt': parse_number(balance_report.get('totalDebt')) or 
                                        ((parse_number(balance_report.get('shortTermDebt')) or 0) + 
                                         (parse_number(balance_report.get('longTermDebt')) or 0)) or None,
                            'shares_outstanding': shares,
                        })
                        
                        # Calculate EPS if we have the data
                        if shares and shares > 0 and record['net_income'] is not None:
                            record['earnings_per_share'] = record['net_income'] / shares
                            record['diluted_earnings_per_share'] = record['earnings_per_share']
                        else:
                            record['earnings_per_share'] = None
                            record['diluted_earnings_per_share'] = None
                        
                        # Calculate book value per share
                        if shares and shares > 0 and shareholder_equity is not None:
                            record['book_value_per_share'] = shareholder_equity / shares
                        else:
                            record['book_value_per_share'] = None
                        
                        # Current ratio for annual (less precise)
                        if total_assets and total_liabilities and total_liabilities > 0:
                            record['current_ratio'] = total_assets / total_liabilities
                            record['quick_ratio'] = record['current_ratio'] * 0.7  # More conservative for annual
                        else:
                            record['current_ratio'] = None
                            record['quick_ratio'] = None
                        
                        break
            
            # Add cash flow data if available
            if cash_data and 'annualReports' in cash_data:
                for cash_report in cash_data['annualReports']:
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
            
            # Add overview metrics 
            if overview_data:
                record.update({
                    'pe_ratio': parse_number(overview_data.get('PERatio')),
                    'peg_ratio': parse_number(overview_data.get('PEGRatio')),
                    'pb_ratio': parse_number(overview_data.get('PriceToBookRatio')),
                    'ps_ratio': parse_number(overview_data.get('PriceToSalesRatioTTM')),
                    'ev_to_revenue': parse_number(overview_data.get('EVToRevenue')),
                    'ev_to_ebitda': parse_number(overview_data.get('EVToEBITDA')),
                    'return_on_equity': parse_number(overview_data.get('ReturnOnEquityTTM')),
                    'return_on_assets': parse_number(overview_data.get('ReturnOnAssetsTTM')),
                    'gross_margin': parse_number(overview_data.get('GrossProfitTTM')) / parse_number(overview_data.get('RevenueTTM')) * 100 if overview_data.get('RevenueTTM') and overview_data.get('GrossProfitTTM') else None,
                    'operating_margin': parse_number(overview_data.get('OperatingMarginTTM')),
                    'net_margin': parse_number(overview_data.get('ProfitMargin')),
                })
            
            # Calculate additional ratios
            if record.get('total_debt') is not None and record.get('total_shareholder_equity') and record['total_shareholder_equity'] > 0:
                record['debt_to_equity'] = record['total_debt'] / record['total_shareholder_equity']
            else:
                record['debt_to_equity'] = None
            
            # Calculate ROIC 
            if record.get('operating_income') and record.get('total_assets') and record.get('cash_and_equivalents'):
                invested_capital = record['total_assets'] - record.get('cash_and_equivalents', 0)
                if invested_capital > 0:
                    record['return_on_invested_capital'] = record['operating_income'] / invested_capital * 100
                else:
                    record['return_on_invested_capital'] = None
            else:
                record['return_on_invested_capital'] = None
            
            records.append(record)
    
    return records

def upsert_fundamentals(records):
    """Upsert fundamentals to database"""
    if not records:
        return 0
    
    df = pd.DataFrame(records)
    
    # Remove records with null fiscal_date_ending
    df = df.dropna(subset=['fiscal_date_ending'])
    
    if df.empty:
        return 0
    
    try:
        with engine.begin() as conn:
            # Create temp table
            temp_table = f"temp_fundamentals_{int(time.time())}"
            df.to_sql(temp_table, conn, if_exists='replace', index=False, method='multi')
            
            # Upsert from temp table with proper type casting
            conn.execute(text(f"""
                INSERT INTO fundamentals (
                    symbol, fiscal_date_ending, period_type, reported_date,
                    revenue, cost_of_revenue, gross_profit, operating_expenses,
                    operating_income, net_income, earnings_per_share, diluted_earnings_per_share,
                    total_assets, total_liabilities, total_shareholder_equity,
                    cash_and_equivalents, total_debt, free_cash_flow, operating_cash_flow,
                    shares_outstanding, pe_ratio, peg_ratio, pb_ratio, ps_ratio,
                    ev_to_revenue, ev_to_ebitda, debt_to_equity, current_ratio,
                    gross_margin, operating_margin, net_margin,
                    return_on_equity, return_on_assets, 
                    quick_ratio, book_value_per_share, return_on_invested_capital,
                    fetched_at
                )
                SELECT 
                    symbol, 
                    fiscal_date_ending::date, 
                    period_type, 
                    reported_date::date,
                    revenue::numeric, 
                    cost_of_revenue::numeric, 
                    gross_profit::numeric, 
                    operating_expenses::numeric,
                    operating_income::numeric, 
                    net_income::numeric, 
                    earnings_per_share::numeric,
                    diluted_earnings_per_share::numeric,
                    total_assets::numeric, 
                    total_liabilities::numeric, 
                    total_shareholder_equity::numeric,
                    cash_and_equivalents::numeric, 
                    total_debt::numeric, 
                    free_cash_flow::numeric, 
                    operating_cash_flow::numeric,
                    shares_outstanding::numeric, 
                    pe_ratio::numeric, 
                    peg_ratio::numeric, 
                    pb_ratio::numeric, 
                    ps_ratio::numeric,
                    ev_to_revenue::numeric, 
                    ev_to_ebitda::numeric, 
                    debt_to_equity::numeric, 
                    current_ratio::numeric,
                    gross_margin::numeric, 
                    operating_margin::numeric, 
                    net_margin::numeric,
                    return_on_equity::numeric, 
                    return_on_assets::numeric,
                    quick_ratio::numeric,
                    book_value_per_share::numeric,
                    return_on_invested_capital::numeric,
                    fetched_at
                FROM {temp_table}
                ON CONFLICT (symbol, fiscal_date_ending, period_type) DO UPDATE SET
                    reported_date = EXCLUDED.reported_date,
                    revenue = EXCLUDED.revenue,
                    cost_of_revenue = EXCLUDED.cost_of_revenue,
                    gross_profit = EXCLUDED.gross_profit,
                    operating_expenses = EXCLUDED.operating_expenses,
                    operating_income = EXCLUDED.operating_income,
                    net_income = EXCLUDED.net_income,
                    earnings_per_share = EXCLUDED.earnings_per_share,
                    diluted_earnings_per_share = EXCLUDED.diluted_earnings_per_share,
                    total_assets = EXCLUDED.total_assets,
                    total_liabilities = EXCLUDED.total_liabilities,
                    total_shareholder_equity = EXCLUDED.total_shareholder_equity,
                    cash_and_equivalents = EXCLUDED.cash_and_equivalents,
                    total_debt = EXCLUDED.total_debt,
                    free_cash_flow = EXCLUDED.free_cash_flow,
                    operating_cash_flow = EXCLUDED.operating_cash_flow,
                    shares_outstanding = EXCLUDED.shares_outstanding,
                    pe_ratio = EXCLUDED.pe_ratio,
                    peg_ratio = EXCLUDED.peg_ratio,
                    pb_ratio = EXCLUDED.pb_ratio,
                    ps_ratio = EXCLUDED.ps_ratio,
                    ev_to_revenue = EXCLUDED.ev_to_revenue,
                    ev_to_ebitda = EXCLUDED.ev_to_ebitda,
                    debt_to_equity = EXCLUDED.debt_to_equity,
                    current_ratio = EXCLUDED.current_ratio,
                    gross_margin = EXCLUDED.gross_margin,
                    operating_margin = EXCLUDED.operating_margin,
                    net_margin = EXCLUDED.net_margin,
                    return_on_equity = EXCLUDED.return_on_equity,
                    return_on_assets = EXCLUDED.return_on_assets,
                    quick_ratio = EXCLUDED.quick_ratio,
                    book_value_per_share = EXCLUDED.book_value_per_share,
                    return_on_invested_capital = EXCLUDED.return_on_invested_capital,
                    fetched_at = EXCLUDED.fetched_at
            """))
            
            # Drop temp table
            conn.execute(text(f"DROP TABLE {temp_table}"))
            
            return len(df)
            
    except Exception as e:
        logger.error(f"Error upserting fundamentals: {e}")
        return 0

def process_symbol(symbol):
    """Process fundamentals for a single symbol"""
    try:
        # Fetch all data
        income_data = fetch_income_statement(symbol)
        balance_data = fetch_balance_sheet(symbol)
        cash_data = fetch_cash_flow(symbol)
        overview_data = fetch_overview(symbol)
        
        # Debug logging
        logger.info(f"Fetched data for {symbol}: Income={income_data is not None}, Balance={balance_data is not None}, Cash={cash_data is not None}, Overview={overview_data is not None}")
        
        # Check if we got any data
        if not income_data and not balance_data and not cash_data:
            logger.warning(f"No data received for {symbol}")
            return symbol, 0, 'NO_DATA'
        
        # Process and combine data
        records = process_fundamentals_data(symbol, income_data, balance_data, cash_data, overview_data)
        
        if not records:
            logger.warning(f"No records processed for {symbol}")
            return symbol, 0, 'NO_DATA'
        
        logger.info(f"Processed {len(records)} records for {symbol}")
        
        # Upsert to database
        count = upsert_fundamentals(records)
        
        if count > 0:
            logger.info(f"Successfully upserted {count} records for {symbol}")
            return symbol, count, 'SUCCESS'
        else:
            logger.warning(f"Failed to upsert records for {symbol}")
            return symbol, 0, 'FAILED'
            
    except Exception as e:
        logger.error(f"Error processing {symbol}: {e}", exc_info=True)
        return symbol, 0, 'ERROR'

def get_symbols():
    """Get symbols to fetch fundamentals for"""
    query = text("""
        SELECT DISTINCT symbol 
        FROM symbol_universe 
        WHERE is_etf = FALSE 
        AND country = 'USA'
        AND symbol IN (
            SELECT symbol FROM stock_prices 
            GROUP BY symbol 
            HAVING COUNT(*) > 100
        )
        ORDER BY symbol
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query)
        return [row[0] for row in result]

def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch fundamentals data from Alpha Vantage")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to process")
    parser.add_argument("--limit", type=int, help="Limit number of symbols")
    parser.add_argument("--parallel", action="store_true", help="Use parallel processing")
    args = parser.parse_args()
    
    start_time = time.time()
    print("\n" + "=" * 60)
    print("FUNDAMENTALS DATA FETCHER (QUARTERLY FOCUSED)")
    print("=" * 60)
    
    # Get symbols
    if args.symbols:
        symbols = args.symbols
        print(f"[INFO] Processing specific symbols: {symbols}")
    else:
        symbols = get_symbols()
        if args.limit:
            symbols = symbols[:args.limit]
        print(f"[INFO] Found {len(symbols)} symbols to process")
    
    if not symbols:
        print("[ERROR] No symbols to process")
        return
    
    # Shuffle to avoid always hitting same symbols first
    random.shuffle(symbols)
    
    # Process symbols
    results = {'SUCCESS': 0, 'NO_DATA': 0, 'FAILED': 0, 'ERROR': 0}
    total_records = 0
    
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
                        'Failed': results['FAILED'],
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
                    'Failed': results['FAILED'],
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
    print(f"Duration: {duration:.1f}s")
    print(f"Rate: {len(symbols)/duration:.2f} symbols/sec")
    
    logger.info(f"Fundamentals fetch completed: {results}")

if __name__ == "__main__":
    main()