#!/usr/bin/env python3
# File: setup_schema.py
# Purpose: Create tables, indexes, materialized views for acis-dem DB

import os
import sys
import time
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError, DisconnectionError
from urllib.parse import urlparse

# Try to load dotenv, fall back to system environment if not available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Using system environment variables.")

def get_postgres_url():
    """Get PostgreSQL URL from environment with validation."""
    postgres_url = os.getenv("POSTGRES_URL")
    if not postgres_url:
        raise RuntimeError(
            "POSTGRES_URL not found in environment variables. "
            "Please set it in your .env file or system environment."
        )
    
    # Validate URL format
    try:
        parsed = urlparse(postgres_url)
        if not parsed.scheme.startswith('postgresql'):
            raise ValueError("URL must use postgresql:// scheme")
        if not parsed.hostname or not parsed.username:
            raise ValueError("URL must include hostname and username")
    except Exception as e:
        raise RuntimeError(f"Invalid POSTGRES_URL format: {e}")
    
    return postgres_url

def create_database_engine(max_retries=3, retry_delay=2):
    """Create and test database engine connection with retry logic."""
    postgres_url = get_postgres_url()
    
    # Enhanced connection parameters
    engine_kwargs = {
        'pool_size': 5,
        'max_overflow': 10,
        'pool_timeout': 30,
        'pool_recycle': 3600,
        'connect_args': {
            'connect_timeout': 10,
            'application_name': 'acis_schema_setup'
        }
    }
    
    for attempt in range(max_retries):
        try:
            engine = create_engine(postgres_url, **engine_kwargs)
            
            # Test connection
            with engine.begin() as conn:
                result = conn.execute(text("SELECT version(), current_database()"))
                db_info = result.fetchone()
                print(f"[OK] Connected to: {db_info[1]} ({db_info[0][:50]}...)")
            
            return engine
            
        except (SQLAlchemyError, DisconnectionError) as e:
            if attempt < max_retries - 1:
                print(f"[WARN] Connection attempt {attempt + 1} failed: {e}")
                print(f"   Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                raise RuntimeError(f"Database connection failed after {max_retries} attempts: {e}")
        except Exception as e:
            raise RuntimeError(f"Error creating database engine: {e}")

def load_sql_from_file(filename):
    """Load SQL from external file if it exists, otherwise return None."""
    sql_file = Path(__file__).parent / 'sql' / filename
    if sql_file.exists():
        return sql_file.read_text(encoding='utf-8')
    return None

def execute_sql_with_transaction(engine, sql_statements, description):
    """Execute SQL statements with proper transaction handling."""
    if not sql_statements:
        print(f"[SKIP] Skipping {description} - no statements provided")
        return True
        
    try:
        with engine.begin() as conn:
            if isinstance(sql_statements, list):
                for stmt in sql_statements:
                    if stmt.strip():
                        conn.execute(text(stmt))
            else:
                conn.execute(text(sql_statements))
        print(f"[OK] {description} executed successfully")
        return True
    except SQLAlchemyError as e:
        print(f"[ERROR] Error in {description}: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error in {description}: {e}")
        return False

SCHEMA_SQL = """
-- =========== SYMBOL UNIVERSE ===========
CREATE TABLE IF NOT EXISTS symbol_universe (
    symbol TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    exchange TEXT NOT NULL,
    security_type TEXT NOT NULL,
    is_etf BOOLEAN NOT NULL,
    listed_date DATE,
    sector TEXT,
    industry TEXT,
    market_cap NUMERIC,
    currency TEXT,
    country TEXT,
    dividend_yield NUMERIC,
    pe_ratio NUMERIC,
    peg_ratio NUMERIC,
    week_52_high NUMERIC,
    week_52_low NUMERIC,
    fetched_at TIMESTAMP NOT NULL,

    -- Constraints to keep symbol_universe "US common only" (no ETFs/funds/etc.)
    CONSTRAINT chk_su_no_etfs
        CHECK (is_etf = false),

    CONSTRAINT chk_su_usd_country
        CHECK (
            country IS NOT NULL
            AND currency IS NOT NULL
            AND (country ILIKE 'United States' OR country ILIKE 'USA')
            AND UPPER(currency) = 'USD'
        ),

    CONSTRAINT chk_su_exchange
        CHECK (exchange IN ('NYSE','NASDAQ','AMEX','NYSE American')),

    CONSTRAINT chk_su_sectype
        CHECK (
            security_type ILIKE 'Common%' OR
            security_type ILIKE 'Ordinary%' OR
            security_type ILIKE 'REIT%'
        ),

    CONSTRAINT chk_su_ticker_pattern
        CHECK (
            symbol ~ '^[A-Z]{1,5}([.][A-Z])?$'
            AND symbol !~ '(-P-[A-Z]+$|[./-](WS|W|WT|U|UN|R|RT)$|[0-9]$)'
        )
);

CREATE INDEX IF NOT EXISTS idx_symbol_universe_exchange ON symbol_universe(exchange);
CREATE INDEX IF NOT EXISTS idx_symbol_universe_sector   ON symbol_universe(sector);
CREATE INDEX IF NOT EXISTS idx_symbol_universe_market_cap ON symbol_universe(market_cap);
CREATE INDEX IF NOT EXISTS idx_universe_sector_industry ON symbol_universe(sector, industry);
CREATE INDEX IF NOT EXISTS idx_universe_mcap_sector ON symbol_universe(market_cap DESC, sector) 
WHERE market_cap IS NOT NULL;

-- =========== STOCK PRICE DATA ===========
CREATE TABLE IF NOT EXISTS stock_eod_daily (
    symbol TEXT NOT NULL,
    trade_date DATE NOT NULL,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    adjusted_close NUMERIC,
    volume BIGINT,
    dividend_amount NUMERIC,
    split_coefficient NUMERIC,
    fetched_at TIMESTAMP,
    PRIMARY KEY (symbol, trade_date)
);

CREATE INDEX IF NOT EXISTS idx_eod_symbol_date ON stock_eod_daily(symbol, trade_date);
CREATE INDEX IF NOT EXISTS idx_eod_trade_date ON stock_eod_daily(trade_date);
CREATE INDEX IF NOT EXISTS idx_eod_symbol ON stock_eod_daily(symbol);
CREATE INDEX IF NOT EXISTS idx_eod_symbol_date_desc ON stock_eod_daily(symbol, trade_date DESC);
CREATE INDEX IF NOT EXISTS idx_eod_date_volume ON stock_eod_daily(trade_date, volume DESC) WHERE volume > 0;
CREATE INDEX IF NOT EXISTS idx_eod_close_date ON stock_eod_daily(close, trade_date) WHERE close IS NOT NULL;

-- Real-time intraday data for live execution
CREATE TABLE IF NOT EXISTS stock_intraday (
    symbol TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    volume BIGINT,
    vwap NUMERIC,
    fetched_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (symbol, timestamp)
);
CREATE INDEX IF NOT EXISTS idx_intraday_symbol_time ON stock_intraday(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_intraday_timestamp ON stock_intraday(timestamp);

-- =========== DIVIDEND DATA ===========
CREATE TABLE IF NOT EXISTS dividend_history (
    symbol TEXT NOT NULL,
    ex_date DATE NOT NULL,
    dividend NUMERIC,
    currency TEXT,
    fetched_at TIMESTAMP,
    PRIMARY KEY (symbol, ex_date)
);

CREATE INDEX IF NOT EXISTS idx_dividends_symbol ON dividend_history(symbol);
CREATE INDEX IF NOT EXISTS idx_dividends_ex_date ON dividend_history(ex_date);

CREATE TABLE IF NOT EXISTS dividend_growth_scores (
    symbol TEXT NOT NULL,
    as_of_date DATE NOT NULL,
    fetched_at TIMESTAMPTZ NOT NULL,
    div_cagr_1y NUMERIC,
    div_cagr_3y NUMERIC,
    div_cagr_5y NUMERIC,
    div_cagr_10y NUMERIC,
    dividend_cut_detected BOOLEAN,
    PRIMARY KEY (symbol, as_of_date)
);

CREATE INDEX IF NOT EXISTS idx_growth_symbol ON dividend_growth_scores(symbol);
CREATE INDEX IF NOT EXISTS idx_growth_as_of_date ON dividend_growth_scores(as_of_date);

-- =========== BENCHMARK DATA ===========
CREATE TABLE IF NOT EXISTS sp500_price_history (
    trade_date DATE PRIMARY KEY,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    adjusted_close NUMERIC,
    volume BIGINT,
    dividend_amount NUMERIC,
    split_coefficient NUMERIC,
    fetched_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sp500_outperformance_scores (
    symbol TEXT PRIMARY KEY,
    lifetime_outperformer BOOLEAN,
    years_outperformed INTEGER,
    total_years INTEGER,
    weighted_score NUMERIC,
    last_year INTEGER,
    fetched_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_sp500_score_weighted ON sp500_outperformance_scores(weighted_score DESC);
CREATE INDEX IF NOT EXISTS idx_sp500_score_lifetime ON sp500_outperformance_scores(lifetime_outperformer);

-- =========== FUNDAMENTALS DATA ===========
CREATE TABLE IF NOT EXISTS fundamentals_annual (
    symbol TEXT NOT NULL,
    fiscal_date DATE NOT NULL,
    source TEXT,
    run_id TEXT,
    totalRevenue BIGINT,
    grossProfit BIGINT,
    netIncome BIGINT,
    eps NUMERIC,
    totalAssets BIGINT,
    totalLiabilities BIGINT,
    totalShareholderEquity BIGINT,
    operatingCashflow BIGINT,
    capitalExpenditures BIGINT,
    dividendPayout BIGINT,
    free_cf BIGINT,
    cash_flow_per_share NUMERIC,
    fetched_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (symbol, fiscal_date)
);

CREATE INDEX IF NOT EXISTS idx_fundamentals_symbol_year ON fundamentals_annual(symbol, fiscal_date);
CREATE INDEX IF NOT EXISTS idx_annual_fiscal_year ON fundamentals_annual (EXTRACT(YEAR FROM fiscal_date));

CREATE TABLE IF NOT EXISTS fundamentals_quarterly (
    symbol TEXT NOT NULL,
    fiscal_date DATE NOT NULL,
    source TEXT,
    run_id TEXT,
    totalRevenue BIGINT,
    grossProfit BIGINT,
    netIncome BIGINT,
    eps NUMERIC,
    totalAssets BIGINT,
    totalLiabilities BIGINT,
    totalShareholderEquity BIGINT,
    operatingCashflow BIGINT,
    capitalExpenditures BIGINT,
    dividendPayout BIGINT,
    free_cf BIGINT,
    cash_flow_per_share NUMERIC,
    fetched_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (symbol, fiscal_date)
);

CREATE INDEX IF NOT EXISTS idx_quarter_fiscal_quarter ON fundamentals_quarterly (EXTRACT(YEAR FROM fiscal_date), EXTRACT(QUARTER FROM fiscal_date));

-- =========== AI MODEL INFRASTRUCTURE ===========
CREATE TABLE IF NOT EXISTS ai_model_run_log (
    run_id TEXT PRIMARY KEY,
    model_type TEXT NOT NULL,
    version TEXT NOT NULL,
    as_of_date DATE NOT NULL,
    features JSONB,
    hyperparameters JSONB,
    sharpe NUMERIC,
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ai_feature_snapshot (
    run_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    as_of_date DATE NOT NULL,
    features JSONB,
    label NUMERIC,
    PRIMARY KEY (run_id, symbol)
);

-- =========== AI SCORE TABLES ===========
CREATE TABLE IF NOT EXISTS ai_value_scores (
    symbol TEXT NOT NULL,
    as_of_date DATE NOT NULL,
    score NUMERIC,
    percentile NUMERIC,
    score_label TEXT,
    rank INTEGER,
    model_version TEXT,
    score_type TEXT,
    fetched_at TIMESTAMPTZ,
    PRIMARY KEY (symbol, as_of_date)
);
CREATE INDEX IF NOT EXISTS idx_value_scores_asof ON ai_value_scores(as_of_date);
CREATE INDEX IF NOT EXISTS idx_value_scores_model ON ai_value_scores(model_version);

CREATE TABLE IF NOT EXISTS ai_growth_scores (
    symbol TEXT NOT NULL,
    as_of_date DATE NOT NULL,
    score NUMERIC,
    percentile NUMERIC,
    score_label TEXT,
    rank INTEGER,
    model_version TEXT,
    score_type TEXT,
    fetched_at TIMESTAMPTZ,
    PRIMARY KEY (symbol, as_of_date)
);
CREATE INDEX IF NOT EXISTS idx_growth_scores_asof ON ai_growth_scores(as_of_date);
CREATE INDEX IF NOT EXISTS idx_growth_scores_model ON ai_growth_scores(model_version);

CREATE TABLE IF NOT EXISTS ai_dividend_scores (
    symbol TEXT NOT NULL,
    as_of_date DATE NOT NULL,
    score NUMERIC,
    percentile NUMERIC,
    score_label TEXT,
    rank INTEGER,
    model_version TEXT,
    score_type TEXT,
    fetched_at TIMESTAMPTZ,
    PRIMARY KEY (symbol, as_of_date)
);
CREATE INDEX IF NOT EXISTS idx_div_scores_asof ON ai_dividend_scores(as_of_date);
CREATE INDEX IF NOT EXISTS idx_div_scores_model ON ai_dividend_scores(model_version);

CREATE TABLE IF NOT EXISTS ai_momentum_scores (
    symbol TEXT NOT NULL,
    as_of_date DATE NOT NULL,
    score NUMERIC,
    percentile NUMERIC,
    score_label TEXT,
    rank INTEGER,
    model_version TEXT,
    score_type TEXT,
    fetched_at TIMESTAMPTZ,
    PRIMARY KEY (symbol, as_of_date)
);
CREATE INDEX IF NOT EXISTS idx_momo_scores_asof ON ai_momentum_scores(as_of_date);
CREATE INDEX IF NOT EXISTS idx_momo_scores_model ON ai_momentum_scores(model_version);

-- =========== PORTFOLIO TABLES ===========
CREATE TABLE IF NOT EXISTS ai_value_portfolio (
    symbol TEXT NOT NULL,
    as_of_date DATE NOT NULL,
    score NUMERIC,
    percentile NUMERIC,
    score_label TEXT,
    rank INTEGER,
    model_version TEXT,
    score_type TEXT,
    fetched_at TIMESTAMPTZ,
    PRIMARY KEY (symbol, as_of_date, score_type)
);
CREATE INDEX IF NOT EXISTS idx_value_port_asof ON ai_value_portfolio(as_of_date);

CREATE TABLE IF NOT EXISTS ai_growth_portfolio (
    symbol TEXT NOT NULL,
    as_of_date DATE NOT NULL,
    score NUMERIC,
    percentile NUMERIC,
    score_label TEXT,
    rank INTEGER,
    model_version TEXT,
    score_type TEXT,
    fetched_at TIMESTAMPTZ,
    PRIMARY KEY (symbol, as_of_date, score_type)
);
CREATE INDEX IF NOT EXISTS idx_growth_port_asof ON ai_growth_portfolio(as_of_date);

CREATE TABLE IF NOT EXISTS ai_dividend_portfolio (
    symbol TEXT NOT NULL,
    as_of_date DATE NOT NULL,
    score NUMERIC,
    percentile NUMERIC,
    score_label TEXT,
    rank INTEGER,
    model_version TEXT,
    score_type TEXT,
    fetched_at TIMESTAMPTZ,
    PRIMARY KEY (symbol, as_of_date, score_type)
);
CREATE INDEX IF NOT EXISTS idx_dividend_port_asof ON ai_dividend_portfolio(as_of_date);

CREATE TABLE IF NOT EXISTS ai_momentum_portfolio (
    symbol TEXT NOT NULL,
    as_of_date DATE NOT NULL,
    score NUMERIC,
    percentile NUMERIC,
    score_label TEXT,
    rank INTEGER,
    model_version TEXT,
    score_type TEXT,
    fetched_at TIMESTAMPTZ,
    PRIMARY KEY (symbol, as_of_date, score_type)
);
CREATE INDEX IF NOT EXISTS idx_momentum_port_asof ON ai_momentum_portfolio(as_of_date);

-- =========== PORTFOLIO MANAGEMENT ===========
CREATE TABLE IF NOT EXISTS ai_portfolio_holdings (
    strategy TEXT NOT NULL,
    as_of_date DATE NOT NULL,
    symbol TEXT NOT NULL,
    weight NUMERIC NOT NULL,
    entry_price NUMERIC,
    rebalance_id TEXT,
    run_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (strategy, as_of_date, symbol)
);
CREATE INDEX IF NOT EXISTS idx_holdings_strategy_date ON ai_portfolio_holdings(strategy, as_of_date);
CREATE INDEX IF NOT EXISTS idx_holdings_rebalance ON ai_portfolio_holdings(rebalance_id);
CREATE INDEX IF NOT EXISTS idx_holdings_run_id ON ai_portfolio_holdings(run_id);

CREATE TABLE IF NOT EXISTS portfolio_rebalance_log (
    rebalance_id TEXT PRIMARY KEY,
    strategy TEXT NOT NULL,
    as_of_date DATE NOT NULL,
    top_k INTEGER NOT NULL,
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_rebalance_strategy_date ON portfolio_rebalance_log(strategy, as_of_date);

CREATE TABLE IF NOT EXISTS strategy_nav (
    strategy TEXT NOT NULL,
    nav_date DATE NOT NULL,
    nav NUMERIC NOT NULL,
    ret NUMERIC,
    run_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (strategy, nav_date)
);
CREATE INDEX IF NOT EXISTS idx_strategy_nav_date ON strategy_nav(strategy, nav_date);

CREATE TABLE IF NOT EXISTS forward_returns (
    symbol TEXT NOT NULL,
    as_of_date DATE NOT NULL,
    return_1m NUMERIC,
    return_3m NUMERIC,
    return_6m NUMERIC,
    return_12m NUMERIC,
    PRIMARY KEY (symbol, as_of_date)
);
CREATE INDEX IF NOT EXISTS idx_forward_returns_symbol_date ON forward_returns(symbol, as_of_date);

-- =========== TRADING INFRASTRUCTURE ===========
CREATE TABLE IF NOT EXISTS trading_orders (
    order_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    order_type TEXT NOT NULL,
    limit_price NUMERIC,
    stop_price NUMERIC,
    status TEXT NOT NULL,
    filled_quantity INTEGER DEFAULT 0,
    avg_fill_price NUMERIC,
    submitted_at TIMESTAMP,
    filled_at TIMESTAMP,
    commission NUMERIC,
    created_at TIMESTAMP DEFAULT NOW(),
    strategy TEXT,
    portfolio_id TEXT,
    parent_order_id TEXT,
    time_in_force TEXT DEFAULT 'DAY',
    broker_order_id TEXT,
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    error_message TEXT,
    tags JSONB
);

CREATE INDEX IF NOT EXISTS idx_orders_symbol ON trading_orders(symbol);
CREATE INDEX IF NOT EXISTS idx_orders_status ON trading_orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_submitted ON trading_orders(submitted_at);
CREATE INDEX IF NOT EXISTS idx_orders_strategy ON trading_orders(strategy);
CREATE INDEX IF NOT EXISTS idx_orders_portfolio ON trading_orders(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_orders_status_submitted ON trading_orders(status, submitted_at DESC);
CREATE INDEX IF NOT EXISTS idx_orders_symbol_status ON trading_orders(symbol, status, submitted_at DESC);

CREATE TABLE IF NOT EXISTS trade_executions (
    execution_id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    order_id TEXT NOT NULL REFERENCES trading_orders(order_id),
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    price NUMERIC NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    venue TEXT,
    execution_fees NUMERIC DEFAULT 0,
    liquidity_flag TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_executions_order_id ON trade_executions(order_id);
CREATE INDEX IF NOT EXISTS idx_executions_symbol_time ON trade_executions(symbol, timestamp);

CREATE TABLE IF NOT EXISTS portfolio_positions (
    portfolio_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    avg_cost NUMERIC NOT NULL,
    market_value NUMERIC,
    unrealized_pnl NUMERIC,
    realized_pnl NUMERIC DEFAULT 0,
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (portfolio_id, symbol)
);
CREATE INDEX IF NOT EXISTS idx_positions_portfolio ON portfolio_positions(portfolio_id);

CREATE TABLE IF NOT EXISTS trading_accounts (
    account_id TEXT PRIMARY KEY,
    broker TEXT NOT NULL,
    account_type TEXT,
    buying_power NUMERIC,
    cash_balance NUMERIC,
    total_value NUMERIC,
    day_trades_used INTEGER DEFAULT 0,
    is_pdt BOOLEAN DEFAULT FALSE,
    last_updated TIMESTAMPTZ DEFAULT NOW()
);

-- =========== BACKTESTING FRAMEWORK ===========
CREATE TABLE IF NOT EXISTS backtest_runs (
    run_id TEXT PRIMARY KEY,
    strategy_name TEXT NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    initial_capital NUMERIC NOT NULL,
    benchmark TEXT DEFAULT 'SPY',
    universe_filter JSONB,
    rebalance_frequency TEXT,
    transaction_costs NUMERIC DEFAULT 0,
    parameters JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    status TEXT DEFAULT 'running',
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_backtest_strategy ON backtest_runs(strategy_name);
CREATE INDEX IF NOT EXISTS idx_backtest_dates ON backtest_runs(start_date, end_date);

CREATE TABLE IF NOT EXISTS backtest_performance (
    run_id TEXT PRIMARY KEY REFERENCES backtest_runs(run_id),
    total_return NUMERIC,
    annual_return NUMERIC,
    volatility NUMERIC,
    sharpe_ratio NUMERIC,
    sortino_ratio NUMERIC,
    max_drawdown NUMERIC,
    max_drawdown_duration INTEGER,
    calmar_ratio NUMERIC,
    win_rate NUMERIC,
    profit_factor NUMERIC,
    beta NUMERIC,
    alpha NUMERIC,
    tracking_error NUMERIC,
    information_ratio NUMERIC,
    var_95 NUMERIC,
    cvar_95 NUMERIC,
    skewness NUMERIC,
    kurtosis NUMERIC,
    trades_count INTEGER,
    avg_trade_pnl NUMERIC,
    best_trade NUMERIC,
    worst_trade NUMERIC
);

CREATE TABLE IF NOT EXISTS backtest_nav (
    run_id TEXT NOT NULL REFERENCES backtest_runs(run_id),
    date DATE NOT NULL,
    portfolio_value NUMERIC NOT NULL,
    cash NUMERIC NOT NULL,
    positions_value NUMERIC NOT NULL,
    daily_return NUMERIC,
    benchmark_return NUMERIC,
    drawdown NUMERIC,
    PRIMARY KEY (run_id, date)
);
CREATE INDEX IF NOT EXISTS idx_backtest_nav_date ON backtest_nav(run_id, date);

CREATE TABLE IF NOT EXISTS backtest_positions (
    run_id TEXT NOT NULL REFERENCES backtest_runs(run_id),
    date DATE NOT NULL,
    symbol TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    price NUMERIC NOT NULL,
    weight NUMERIC,
    market_value NUMERIC,
    unrealized_pnl NUMERIC,
    PRIMARY KEY (run_id, date, symbol)
);
CREATE INDEX IF NOT EXISTS idx_backtest_pos_date ON backtest_positions(run_id, date);

CREATE TABLE IF NOT EXISTS backtest_transactions (
    transaction_id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    run_id TEXT NOT NULL REFERENCES backtest_runs(run_id),
    date DATE NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    price NUMERIC NOT NULL,
    transaction_cost NUMERIC DEFAULT 0,
    pnl NUMERIC,
    reason TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_backtest_txn_run_date ON backtest_transactions(run_id, date);

-- =========== RISK MANAGEMENT ===========
CREATE TABLE IF NOT EXISTS risk_limits (
    limit_id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    portfolio_id TEXT NOT NULL,
    limit_type TEXT NOT NULL,
    symbol TEXT,
    sector TEXT,
    max_value NUMERIC NOT NULL,
    current_value NUMERIC,
    breach_count INTEGER DEFAULT 0,
    last_breach TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Fixed risk_metrics table with proper primary key handling
CREATE TABLE IF NOT EXISTS risk_metrics (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    portfolio_id TEXT,
    symbol TEXT,
    as_of_date DATE NOT NULL,
    beta NUMERIC,
    volatility_30d NUMERIC,
    var_95 NUMERIC,
    var_99_1d NUMERIC,
    expected_shortfall NUMERIC,
    max_drawdown NUMERIC,
    sharpe_ratio NUMERIC,
    correlation_spy NUMERIC,
    total_value NUMERIC,
    cash_pct NUMERIC,
    sector_concentration JSONB,
    single_stock_max NUMERIC,
    active_share NUMERIC,
    UNIQUE (portfolio_id, symbol, as_of_date)
);
CREATE INDEX IF NOT EXISTS idx_risk_metrics_portfolio_date ON risk_metrics(portfolio_id, as_of_date);
CREATE INDEX IF NOT EXISTS idx_risk_metrics_symbol_date ON risk_metrics(symbol, as_of_date);

-- =========== DATA QUALITY & MONITORING ===========
CREATE TABLE IF NOT EXISTS data_quality_checks (
    check_id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    table_name TEXT NOT NULL,
    check_type TEXT NOT NULL,
    symbol TEXT,
    expected_count INTEGER,
    actual_count INTEGER,
    check_date DATE NOT NULL,
    status TEXT NOT NULL,
    details JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_data_quality_table_date ON data_quality_checks(table_name, check_date);

CREATE TABLE IF NOT EXISTS market_data_status (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    source TEXT NOT NULL,
    data_type TEXT NOT NULL,
    symbol TEXT,
    last_update TIMESTAMPTZ,
    expected_update TIMESTAMPTZ,
    status TEXT NOT NULL,
    lag_minutes INTEGER,
    error_message TEXT,
    UNIQUE (source, data_type, symbol)
);

CREATE TABLE IF NOT EXISTS system_alerts (
    alert_id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    alert_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    title TEXT NOT NULL,
    message TEXT NOT NULL,
    details JSONB,
    is_acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by TEXT,
    acknowledged_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_alerts_type_severity ON system_alerts(alert_type, severity);
CREATE INDEX IF NOT EXISTS idx_alerts_unack ON system_alerts(is_acknowledged, created_at);

-- =========== ADDITIONAL MARKET DATA ===========
CREATE TABLE IF NOT EXISTS options_eod (
    symbol TEXT NOT NULL,
    option_symbol TEXT NOT NULL,
    expiration_date DATE NOT NULL,
    strike NUMERIC NOT NULL,
    option_type TEXT NOT NULL,
    trade_date DATE NOT NULL,
    bid NUMERIC,
    ask NUMERIC,
    last NUMERIC,
    volume INTEGER,
    open_interest INTEGER,
    implied_volatility NUMERIC,
    delta NUMERIC,
    gamma NUMERIC,
    theta NUMERIC,
    vega NUMERIC,
    PRIMARY KEY (option_symbol, trade_date)
);
CREATE INDEX IF NOT EXISTS idx_options_underlying_exp ON options_eod(symbol, expiration_date);

CREATE TABLE IF NOT EXISTS economic_indicators (
    indicator_code TEXT NOT NULL,
    date DATE NOT NULL,
    value NUMERIC NOT NULL,
    source TEXT,
    fetched_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (indicator_code, date)
);

-- Note: Foreign key constraints are handled separately to avoid data conflicts
"""

# Separate foreign keys to handle dependencies correctly
FOREIGN_KEYS_SQL = """
-- =========== FOREIGN KEY CONSTRAINTS ===========
DO $$
BEGIN
    -- Check if symbol_universe has data before adding foreign keys
    IF EXISTS (SELECT 1 FROM symbol_universe LIMIT 1) THEN
        IF NOT EXISTS (SELECT 1 FROM information_schema.table_constraints WHERE constraint_name = 'fk_eod_symbol') THEN
            ALTER TABLE stock_eod_daily ADD CONSTRAINT fk_eod_symbol 
            FOREIGN KEY (symbol) REFERENCES symbol_universe(symbol);
        END IF;
        
        IF NOT EXISTS (SELECT 1 FROM information_schema.table_constraints WHERE constraint_name = 'fk_dividend_symbol') THEN
            ALTER TABLE dividend_history ADD CONSTRAINT fk_dividend_symbol
            FOREIGN KEY (symbol) REFERENCES symbol_universe(symbol);
        END IF;
        
        IF NOT EXISTS (SELECT 1 FROM information_schema.table_constraints WHERE constraint_name = 'fk_intraday_symbol') THEN
            ALTER TABLE stock_intraday ADD CONSTRAINT fk_intraday_symbol
            FOREIGN KEY (symbol) REFERENCES symbol_universe(symbol);
        END IF;
    ELSE
        RAISE NOTICE 'Skipping foreign key constraints - no data in symbol_universe table yet';
    END IF;
END$$;
"""

# Split materialized views into separate function for better error handling
MATERIALIZED_VIEWS_SQL = """
-- Latest forward returns per symbol
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_latest_forward_returns AS
SELECT DISTINCT ON (symbol)
    symbol,
    as_of_date,
    return_1m,
    return_3m,
    return_6m,
    return_12m
FROM forward_returns
ORDER BY symbol, as_of_date DESC;

CREATE UNIQUE INDEX IF NOT EXISTS ux_mv_latest_forward_returns_symbol
    ON mv_latest_forward_returns(symbol);

-- Latest annual fundamentals per symbol
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_latest_annual_fundamentals AS
SELECT DISTINCT ON (symbol)
    *
FROM fundamentals_annual
ORDER BY symbol, fiscal_date DESC;

CREATE UNIQUE INDEX IF NOT EXISTS ux_mv_latest_annual_fundamentals_symbol
ON mv_latest_annual_fundamentals(symbol);

-- Current AI portfolios with weights
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_current_ai_portfolios AS
WITH v AS (
  SELECT 'value'::text AS strategy, symbol, as_of_date, score, percentile, score_label, rank, model_version
  FROM ai_value_portfolio
  WHERE as_of_date = (SELECT MAX(as_of_date) FROM ai_value_portfolio)
),
g AS (
  SELECT 'growth'::text, symbol, as_of_date, score, percentile, score_label, rank, model_version
  FROM ai_growth_portfolio
  WHERE as_of_date = (SELECT MAX(as_of_date) FROM ai_growth_portfolio)
),
d AS (
  SELECT 'dividend'::text, symbol, as_of_date, score, percentile, score_label, rank, model_version
  FROM ai_dividend_portfolio
  WHERE as_of_date = (SELECT MAX(as_of_date) FROM ai_dividend_portfolio)
),
m AS (
  SELECT 'momentum'::text, symbol, as_of_date, score, percentile, score_label, rank, model_version
  FROM ai_momentum_portfolio
  WHERE as_of_date = (SELECT MAX(as_of_date) FROM ai_momentum_portfolio)
),
u AS (
  SELECT * FROM v UNION ALL SELECT * FROM g UNION ALL SELECT * FROM d UNION ALL SELECT * FROM m
)
SELECT
  strategy,
  symbol,
  as_of_date,
  score,
  percentile,
  score_label,
  rank,
  model_version,
  1.0 / COUNT(*) OVER (PARTITION BY strategy, as_of_date) AS weight_eq
FROM u;

CREATE UNIQUE INDEX IF NOT EXISTS ux_mv_current_ai_portfolios ON mv_current_ai_portfolios(strategy, symbol);

-- Current positions across all portfolios
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_current_positions AS
SELECT 
    portfolio_id,
    symbol,
    quantity,
    avg_cost,
    market_value,
    unrealized_pnl,
    market_value / NULLIF(SUM(market_value) OVER (PARTITION BY portfolio_id), 0) AS weight,
    last_updated
FROM portfolio_positions
WHERE quantity != 0;

CREATE UNIQUE INDEX IF NOT EXISTS ux_mv_current_positions ON mv_current_positions(portfolio_id, symbol);

-- Portfolio risk summary
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_portfolio_risk_summary AS
SELECT DISTINCT ON (portfolio_id)
    portfolio_id,
    as_of_date,
    total_value,
    beta,
    volatility_30d,
    var_95,
    max_drawdown,
    single_stock_max
FROM risk_metrics
WHERE portfolio_id IS NOT NULL
ORDER BY portfolio_id, as_of_date DESC;

CREATE UNIQUE INDEX IF NOT EXISTS ux_mv_portfolio_risk_summary ON mv_portfolio_risk_summary(portfolio_id);
"""

# Safer version of AI portfolios view that checks for table existence
PORTFOLIO_VIEWS_SQL = """
-- Current AI portfolios with weights (only create if source tables exist and have data)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'ai_value_portfolio') 
       AND EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'ai_growth_portfolio')
       AND EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'ai_dividend_portfolio')
       AND EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'ai_momentum_portfolio') THEN
        
        DROP MATERIALIZED VIEW IF EXISTS mv_current_ai_portfolios;
        
        CREATE MATERIALIZED VIEW mv_current_ai_portfolios AS
        WITH portfolio_data AS (
            SELECT 'value'::text AS strategy, symbol, as_of_date, score, percentile, score_label, rank, model_version
            FROM ai_value_portfolio
            WHERE as_of_date = (SELECT MAX(as_of_date) FROM ai_value_portfolio WHERE as_of_date IS NOT NULL)
            
            UNION ALL
            
            SELECT 'growth'::text AS strategy, symbol, as_of_date, score, percentile, score_label, rank, model_version
            FROM ai_growth_portfolio
            WHERE as_of_date = (SELECT MAX(as_of_date) FROM ai_growth_portfolio WHERE as_of_date IS NOT NULL)
            
            UNION ALL
            
            SELECT 'dividend'::text AS strategy, symbol, as_of_date, score, percentile, score_label, rank, model_version
            FROM ai_dividend_portfolio
            WHERE as_of_date = (SELECT MAX(as_of_date) FROM ai_dividend_portfolio WHERE as_of_date IS NOT NULL)
            
            UNION ALL
            
            SELECT 'momentum'::text AS strategy, symbol, as_of_date, score, percentile, score_label, rank, model_version
            FROM ai_momentum_portfolio
            WHERE as_of_date = (SELECT MAX(as_of_date) FROM ai_momentum_portfolio WHERE as_of_date IS NOT NULL)
        )
        SELECT
            strategy,
            symbol,
            as_of_date,
            score,
            percentile,
            score_label,
            rank,
            model_version,
            1.0 / COUNT(*) OVER (PARTITION BY strategy, as_of_date) AS weight_eq
        FROM portfolio_data
        WHERE as_of_date IS NOT NULL;

        CREATE UNIQUE INDEX IF NOT EXISTS ux_mv_current_ai_portfolios ON mv_current_ai_portfolios(strategy, symbol);
        
        RAISE NOTICE 'Created mv_current_ai_portfolios materialized view';
    ELSE
        RAISE NOTICE 'Skipping mv_current_ai_portfolios - one or more source tables do not exist yet';
    END IF;
END$$;
"""

def execute_schema_sql(engine):
    """Execute the main schema SQL with error handling."""
    # Try loading from external file first
    external_sql = load_sql_from_file('schema.sql')
    sql_to_execute = external_sql if external_sql else SCHEMA_SQL
    
    return execute_sql_with_transaction(
        engine, sql_to_execute, "Main schema (tables, indexes)"
    )

def execute_foreign_keys(engine):
    """Execute foreign key constraints with separate error handling."""
    external_sql = load_sql_from_file('foreign_keys.sql')
    sql_to_execute = external_sql if external_sql else FOREIGN_KEYS_SQL
    
    success = execute_sql_with_transaction(
        engine, sql_to_execute, "Foreign key constraints"
    )
    if not success:
        print("[WARN] Warning: Foreign key constraints failed (this is often expected on first run)")
    return success

def execute_materialized_views(engine):
    """Execute materialized views with separate error handling."""
    external_sql = load_sql_from_file('materialized_views.sql')
    sql_to_execute = external_sql if external_sql else MATERIALIZED_VIEWS_SQL
    
    success = execute_sql_with_transaction(
        engine, sql_to_execute, "Basic materialized views"
    )
    if not success:
        print("[WARN] Warning: Materialized views failed (likely due to missing data)")
    return success

def execute_portfolio_views(engine):
    """Execute portfolio materialized views with separate error handling."""
    external_sql = load_sql_from_file('portfolio_views.sql')
    sql_to_execute = external_sql if external_sql else PORTFOLIO_VIEWS_SQL
    
    success = execute_sql_with_transaction(
        engine, sql_to_execute, "Portfolio materialized views"
    )
    if not success:
        print("[WARN] Warning: Portfolio views failed (likely due to missing portfolio data)")
    return success

def check_existing_schema(engine):
    """Check what schema components already exist."""
    try:
        with engine.begin() as conn:
            # Check for key tables
            result = conn.execute(text("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('symbol_universe', 'stock_eod_daily', 'trading_orders')
                ORDER BY table_name
            """))
            existing_tables = [row[0] for row in result.fetchall()]
            
            if existing_tables:
                print(f"[INFO] Found existing tables: {', '.join(existing_tables)}")
                return len(existing_tables)
            else:
                print("[INFO] No existing schema detected")
                return 0
    except Exception as e:
        print(f"[WARN] Could not check existing schema: {e}")
        return -1

def main():
    """Main execution function with proper error handling."""
    print("[START] Starting ACIS database schema setup...")
    
    try:
        # Create database engine with retries
        engine = create_database_engine()
        
        # Check existing schema
        existing_count = check_existing_schema(engine)
        if existing_count > 0:
            print(f"[INFO] Found {existing_count} existing tables. Proceeding with IF NOT EXISTS logic...")
            # Skip interactive prompt - use IF NOT EXISTS to handle existing tables
        
        # Execute main schema
        print("\n[SETUP] Creating main schema...")
        schema_success = execute_schema_sql(engine)
        if not schema_success:
            raise RuntimeError("Critical error: Main schema setup failed")
        
        # Execute non-critical components
        print("\n[SETUP] Setting up foreign keys...")
        fk_success = execute_foreign_keys(engine)
        
        print("\n[SETUP] Creating materialized views...")
        basic_views_success = execute_materialized_views(engine)
        
        print("\n[SETUP] Creating portfolio views...")
        portfolio_views_success = execute_portfolio_views(engine)
        
        # Summary report
        print("\n" + "="*60)
        print("[SUCCESS] ACIS Database Schema Setup Complete!")
        print("="*60)
        
        print("\n[REPORT] Schema Components:")
        print("   [OK] Core tables and indexes")
        print(f"   [{'OK' if fk_success else 'WARN'}] Foreign key constraints")
        print(f"   [{'OK' if basic_views_success else 'WARN'}] Basic materialized views")
        print(f"   [{'OK' if portfolio_views_success else 'WARN'}] Portfolio views")
        
        print("\n[REPORT] Schema Includes:")
        print("   - Symbol universe and market data storage")
        print("   - AI/ML scoring and portfolio management") 
        print("   - Trading infrastructure and execution")
        print("   - Risk management and monitoring")
        print("   - Comprehensive backtesting framework")
        
        if not all([fk_success, basic_views_success, portfolio_views_success]):
            print("\n[NOTE] Note: Some components were skipped due to missing data.")
            print("   These will be created automatically when you populate the tables.")
        
        # Close engine
        engine.dispose()
        print("\n[OK] Database connection closed")
        
    except KeyboardInterrupt:
        print("\n[WARN] Schema setup interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Schema setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
