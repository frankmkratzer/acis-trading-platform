#!/usr/bin/env python3
"""
Essential ACIS Database Schema - Data Pipeline Only
Creates only the tables needed by our 9 essential scripts
"""

import os
import sys
import time
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from logging_config import setup_logger, log_script_start, log_script_end

load_dotenv()
logger = setup_logger("setup_essential_schema")

def get_postgres_url():
    """Get PostgreSQL URL from environment."""
    postgres_url = os.getenv("POSTGRES_URL")
    if not postgres_url:
        raise RuntimeError("POSTGRES_URL not found in environment variables.")
    return postgres_url

def create_essential_schema():
    """Create essential tables for data pipeline only."""
    
    ESSENTIAL_SCHEMA_SQL = """
    -- =============================================
    -- ESSENTIAL ACIS DATA PIPELINE TABLES
    -- =============================================
    
    -- 1. Symbol Universe (master symbol list)
    CREATE TABLE IF NOT EXISTS symbol_universe (
        symbol TEXT PRIMARY KEY,
        company_name TEXT,
        exchange TEXT,
        sector TEXT,
        industry TEXT,
        market_cap BIGINT,
        country TEXT DEFAULT 'US',
        currency TEXT DEFAULT 'USD',
        is_etf BOOLEAN DEFAULT FALSE,
        is_active BOOLEAN DEFAULT TRUE,
        ipo_date DATE,
        delisted_date DATE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_symbol_universe_exchange ON symbol_universe(exchange);
    CREATE INDEX IF NOT EXISTS idx_symbol_universe_sector ON symbol_universe(sector);
    CREATE INDEX IF NOT EXISTS idx_symbol_universe_market_cap ON symbol_universe(market_cap);
    CREATE INDEX IF NOT EXISTS idx_universe_sector_industry ON symbol_universe(sector, industry);
    CREATE INDEX IF NOT EXISTS idx_universe_mcap_sector ON symbol_universe(market_cap DESC, sector);
    
    -- 2. Stock Prices (core price data)
    CREATE TABLE IF NOT EXISTS stock_prices (
        symbol TEXT NOT NULL,
        date DATE NOT NULL,
        open_price NUMERIC,
        high_price NUMERIC,
        low_price NUMERIC,
        close_price NUMERIC NOT NULL,
        adjusted_close NUMERIC,
        volume BIGINT,
        dividend_amount NUMERIC DEFAULT 0,
        split_coefficient NUMERIC DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, date)
    );
    
    CREATE INDEX IF NOT EXISTS idx_stock_prices_symbol ON stock_prices(symbol);
    CREATE INDEX IF NOT EXISTS idx_stock_prices_date ON stock_prices(date);
    CREATE INDEX IF NOT EXISTS idx_stock_prices_symbol_date ON stock_prices(symbol, date DESC);
    
    -- 3. Dividend History
    CREATE TABLE IF NOT EXISTS dividend_history (
        symbol TEXT NOT NULL,
        ex_date DATE NOT NULL,
        dividend NUMERIC,
        currency TEXT DEFAULT 'USD',
        fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, ex_date)
    );
    
    CREATE INDEX IF NOT EXISTS idx_dividends_symbol ON dividend_history(symbol);
    CREATE INDEX IF NOT EXISTS idx_dividends_ex_date ON dividend_history(ex_date);
    
    -- 4. S&P 500 Price History
    CREATE TABLE IF NOT EXISTS sp500_price_history (
        trade_date DATE PRIMARY KEY,
        open_price NUMERIC,
        high_price NUMERIC,
        low_price NUMERIC,
        close_price NUMERIC NOT NULL,
        adjusted_close NUMERIC,
        volume BIGINT,
        dividend_amount NUMERIC DEFAULT 0,
        split_coefficient NUMERIC DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_sp500_date ON sp500_price_history(trade_date);
    
    -- 5. Annual Fundamentals
    CREATE TABLE IF NOT EXISTS fundamentals_annual (
        symbol TEXT NOT NULL,
        fiscal_date DATE NOT NULL,
        reported_date DATE,
        period TEXT,
        revenue BIGINT,
        cost_of_revenue BIGINT,
        gross_profit BIGINT,
        operating_expenses BIGINT,
        operating_income BIGINT,
        net_income BIGINT,
        earnings_per_share NUMERIC,
        total_assets BIGINT,
        total_liabilities BIGINT,
        stockholder_equity BIGINT,
        cash_and_equivalents BIGINT,
        total_debt BIGINT,
        free_cash_flow BIGINT,
        shares_outstanding BIGINT,
        book_value_per_share NUMERIC,
        pe_ratio NUMERIC,
        pb_ratio NUMERIC,
        price_to_sales NUMERIC,
        debt_to_equity NUMERIC,
        roe NUMERIC,
        roa NUMERIC,
        current_ratio NUMERIC,
        quick_ratio NUMERIC,
        gross_margin NUMERIC,
        operating_margin NUMERIC,
        net_margin NUMERIC,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, fiscal_date)
    );
    
    CREATE INDEX IF NOT EXISTS idx_fundamentals_symbol_year ON fundamentals_annual(symbol, fiscal_date);
    CREATE INDEX IF NOT EXISTS idx_annual_fiscal_year ON fundamentals_annual(EXTRACT(YEAR FROM fiscal_date));
    CREATE INDEX IF NOT EXISTS idx_annual_pe_ratio ON fundamentals_annual(pe_ratio) WHERE pe_ratio IS NOT NULL;
    CREATE INDEX IF NOT EXISTS idx_annual_roe ON fundamentals_annual(roe) WHERE roe IS NOT NULL;
    CREATE INDEX IF NOT EXISTS idx_annual_debt_equity ON fundamentals_annual(debt_to_equity) WHERE debt_to_equity IS NOT NULL;
    
    -- 6. Quarterly Fundamentals
    CREATE TABLE IF NOT EXISTS fundamentals_quarterly (
        symbol TEXT NOT NULL,
        fiscal_date DATE NOT NULL,
        reported_date DATE,
        period TEXT,
        revenue BIGINT,
        cost_of_revenue BIGINT,
        gross_profit BIGINT,
        operating_expenses BIGINT,
        operating_income BIGINT,
        net_income BIGINT,
        earnings_per_share NUMERIC,
        total_assets BIGINT,
        total_liabilities BIGINT,
        stockholder_equity BIGINT,
        cash_and_equivalents BIGINT,
        total_debt BIGINT,
        free_cash_flow BIGINT,
        shares_outstanding BIGINT,
        book_value_per_share NUMERIC,
        pe_ratio NUMERIC,
        pb_ratio NUMERIC,
        price_to_sales NUMERIC,
        debt_to_equity NUMERIC,
        roe NUMERIC,
        roa NUMERIC,
        current_ratio NUMERIC,
        quick_ratio NUMERIC,
        gross_margin NUMERIC,
        operating_margin NUMERIC,
        net_margin NUMERIC,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, fiscal_date)
    );
    
    CREATE INDEX IF NOT EXISTS idx_fundamentals_quarterly_symbol_date ON fundamentals_quarterly(symbol, fiscal_date);
    CREATE INDEX IF NOT EXISTS idx_quarter_fiscal_quarter ON fundamentals_quarterly(EXTRACT(YEAR FROM fiscal_date), EXTRACT(QUARTER FROM fiscal_date));
    CREATE INDEX IF NOT EXISTS idx_quarterly_pe_ratio ON fundamentals_quarterly(pe_ratio) WHERE pe_ratio IS NOT NULL;
    CREATE INDEX IF NOT EXISTS idx_quarterly_roe ON fundamentals_quarterly(roe) WHERE roe IS NOT NULL;
    CREATE INDEX IF NOT EXISTS idx_quarterly_debt_equity ON fundamentals_quarterly(debt_to_equity) WHERE debt_to_equity IS NOT NULL;
    
    -- 7. Technical Indicators
    CREATE TABLE IF NOT EXISTS technical_indicators (
        symbol TEXT NOT NULL,
        date DATE NOT NULL,
        rsi_14 NUMERIC,
        sma_20 NUMERIC,
        sma_50 NUMERIC,
        sma_200 NUMERIC,
        ema_12 NUMERIC,
        ema_26 NUMERIC,
        macd NUMERIC,
        macd_signal NUMERIC,
        macd_histogram NUMERIC,
        bb_upper NUMERIC,
        bb_middle NUMERIC,
        bb_lower NUMERIC,
        stoch_k NUMERIC,
        stoch_d NUMERIC,
        williams_r NUMERIC,
        adx NUMERIC,
        obv BIGINT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, date)
    );
    
    CREATE INDEX IF NOT EXISTS idx_tech_indicators_symbol ON technical_indicators(symbol);
    CREATE INDEX IF NOT EXISTS idx_tech_indicators_date ON technical_indicators(date);
    CREATE INDEX IF NOT EXISTS idx_tech_indicators_rsi ON technical_indicators(rsi_14) WHERE rsi_14 IS NOT NULL;
    
    -- 8. Forward Returns
    CREATE TABLE IF NOT EXISTS forward_returns (
        symbol TEXT NOT NULL,
        as_of_date DATE NOT NULL,
        return_1m NUMERIC,
        return_3m NUMERIC,
        return_6m NUMERIC,
        return_12m NUMERIC,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, as_of_date)
    );
    
    CREATE INDEX IF NOT EXISTS idx_forward_returns_symbol_date ON forward_returns(symbol, as_of_date);
    CREATE INDEX IF NOT EXISTS idx_forward_returns_date ON forward_returns(as_of_date);
    
    -- =============================================
    -- MATERIALIZED VIEWS
    -- =============================================
    
    -- Latest forward returns per symbol
    CREATE MATERIALIZED VIEW IF NOT EXISTS mv_latest_forward_returns AS
    SELECT DISTINCT ON (symbol) 
        symbol, as_of_date, return_1m, return_3m, return_6m, return_12m
    FROM forward_returns 
    WHERE as_of_date IS NOT NULL
    ORDER BY symbol, as_of_date DESC;
    
    CREATE UNIQUE INDEX IF NOT EXISTS ux_mv_latest_forward_returns_symbol ON mv_latest_forward_returns(symbol);
    
    -- Latest fundamentals per symbol  
    CREATE MATERIALIZED VIEW IF NOT EXISTS mv_latest_fundamentals AS
    SELECT DISTINCT ON (symbol)
        symbol, fiscal_date, revenue, net_income, earnings_per_share, 
        pe_ratio, roe, debt_to_equity, free_cash_flow
    FROM fundamentals_annual
    WHERE fiscal_date IS NOT NULL
    ORDER BY symbol, fiscal_date DESC;
    
    CREATE UNIQUE INDEX IF NOT EXISTS ux_mv_latest_fundamentals_symbol ON mv_latest_fundamentals(symbol);
    """
    
    try:
        postgres_url = get_postgres_url()
        engine = create_engine(postgres_url)
        
        logger.info("Creating essential ACIS data pipeline schema...")
        
        with engine.begin() as conn:
            conn.execute(text(ESSENTIAL_SCHEMA_SQL))
        
        logger.info("Essential schema created successfully!")
        
        # Verify tables were created
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                  AND table_name IN (
                    'symbol_universe', 'stock_prices', 'dividend_history',
                    'sp500_price_history', 'fundamentals_annual', 'fundamentals_quarterly',
                    'technical_indicators', 'forward_returns'
                  )
                ORDER BY table_name
            """))
            
            tables = [row[0] for row in result]
            logger.info(f"Verified {len(tables)} essential tables:")
            for table in tables:
                logger.info(f"  ✓ {table}")
        
        # Verify materialized views
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT matviewname 
                FROM pg_matviews 
                WHERE matviewname LIKE 'mv_%'
                ORDER BY matviewname
            """))
            
            views = [row[0] for row in result]
            logger.info(f"Verified {len(views)} materialized views:")
            for view in views:
                logger.info(f"  ✓ {view}")
        
        return True
        
    except Exception as e:
        logger.error(f"Essential schema creation failed: {e}")
        return False

def main():
    """Main execution function."""
    start_time = time.time()
    log_script_start(logger, "setup_essential_schema", "Create essential ACIS data pipeline schema")
    
    try:
        success = create_essential_schema()
        
        duration = time.time() - start_time
        if success:
            log_script_end(logger, "setup_essential_schema", True, duration, {
                "Tables created": "8 essential tables",
                "Views created": "2 materialized views",
                "Status": "Ready for data pipeline"
            })
        else:
            log_script_end(logger, "setup_essential_schema", False, duration)
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Schema setup failed: {e}")
        log_script_end(logger, "setup_essential_schema", False, time.time() - start_time)
        sys.exit(1)

if __name__ == "__main__":
    main()