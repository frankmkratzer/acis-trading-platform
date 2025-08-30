#!/usr/bin/env python3
"""
ACIS Database Schema Setup - CLEAN VERSION
Only includes tables needed for the Three-Portfolio Strategy
Removed: ML tables, complex rankings, duplicate tables
"""

import os
import sys
import time
from sqlalchemy import create_engine, text
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

load_dotenv()

def get_postgres_url():
    """Get PostgreSQL connection URL from environment"""
    postgres_url = os.getenv("POSTGRES_URL")
    if not postgres_url:
        raise ValueError("POSTGRES_URL not set in .env file")
    return postgres_url

def create_core_tables():
    """Create core trading platform tables"""
    SQL = """
    -- Symbol Universe (Master list of all stocks)
    CREATE TABLE IF NOT EXISTS symbol_universe (
        symbol VARCHAR(10) PRIMARY KEY,
        name VARCHAR(255),
        exchange VARCHAR(50),
        security_type VARCHAR(50),
        is_etf BOOLEAN DEFAULT FALSE,
        sector VARCHAR(100),
        industry VARCHAR(100),
        market_cap NUMERIC(20, 2),
        shares_outstanding NUMERIC(20, 2),
        country VARCHAR(50) DEFAULT 'USA',
        currency VARCHAR(10) DEFAULT 'USD',
        delisted_date DATE,
        overview_fetched_at TIMESTAMP,
        fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- Stock Prices (Daily OHLCV data)
    CREATE TABLE IF NOT EXISTS stock_prices (
        symbol VARCHAR(10) NOT NULL,
        date DATE NOT NULL,
        open_price NUMERIC(12, 4),
        high NUMERIC(12, 4),
        low NUMERIC(12, 4),
        close_price NUMERIC(12, 4),
        adjusted_close NUMERIC(12, 4),
        volume BIGINT,
        dividend_amount NUMERIC(12, 4),
        split_coefficient NUMERIC(12, 4),
        fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, date),
        FOREIGN KEY (symbol) REFERENCES symbol_universe(symbol) ON DELETE CASCADE
    );
    
    -- S&P 500 History (Benchmark data)
    CREATE TABLE IF NOT EXISTS sp500_history (
        date DATE PRIMARY KEY,
        close_price NUMERIC(12, 4),
        open_price NUMERIC(12, 4),
        high NUMERIC(12, 4),
        low NUMERIC(12, 4),
        volume BIGINT,
        fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- Forward Returns (Performance calculations)
    CREATE TABLE IF NOT EXISTS forward_returns (
        symbol VARCHAR(10) NOT NULL,
        date DATE NOT NULL,
        return_1m NUMERIC(10, 4),
        return_3m NUMERIC(10, 4),
        return_6m NUMERIC(10, 4),
        return_12m NUMERIC(10, 4),
        volatility_1m NUMERIC(10, 4),
        volatility_3m NUMERIC(10, 4),
        sharpe_1m NUMERIC(10, 4),
        sharpe_3m NUMERIC(10, 4),
        max_drawdown_1m NUMERIC(10, 4),
        max_drawdown_3m NUMERIC(10, 4),
        calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, date),
        FOREIGN KEY (symbol) REFERENCES symbol_universe(symbol) ON DELETE CASCADE
    );
    """
    return SQL

def create_fundamentals_tables():
    """Create fundamental data tables"""
    SQL = """
    -- Company Fundamentals (Quarterly/Annual financials)
    CREATE TABLE IF NOT EXISTS fundamentals (
        symbol VARCHAR(10) NOT NULL,
        fiscal_date_ending DATE NOT NULL,
        period_type VARCHAR(20) NOT NULL,
        
        -- Income Statement
        total_revenue_ttm NUMERIC(20, 2),
        gross_profit_ttm NUMERIC(20, 2),
        operating_income_ttm NUMERIC(20, 2),
        net_income_ttm NUMERIC(20, 2),
        ebitda NUMERIC(20, 2),
        diluted_eps_ttm NUMERIC(12, 4),
        
        -- Balance Sheet
        total_assets NUMERIC(20, 2),
        total_liabilities NUMERIC(20, 2),
        total_shareholder_equity NUMERIC(20, 2),
        cash_and_cash_equivalents NUMERIC(20, 2),
        total_debt NUMERIC(20, 2),
        
        -- Cash Flow
        operating_cash_flow NUMERIC(20, 2),
        free_cash_flow NUMERIC(20, 2),
        capital_expenditures NUMERIC(20, 2),
        dividends_paid NUMERIC(20, 2),
        
        -- Shares
        shares_outstanding NUMERIC(20, 2),
        
        -- Metadata
        fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, fiscal_date_ending, period_type),
        FOREIGN KEY (symbol) REFERENCES symbol_universe(symbol) ON DELETE CASCADE
    );
    
    -- Dividend History (All dividend payments)
    CREATE TABLE IF NOT EXISTS dividend_history (
        symbol VARCHAR(10) NOT NULL,
        ex_date DATE NOT NULL,
        payment_date DATE,
        record_date DATE,
        declaration_date DATE,
        dividend NUMERIC(12, 4),
        adjusted_dividend NUMERIC(12, 4),
        currency VARCHAR(10) DEFAULT 'USD',
        frequency VARCHAR(20),
        fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, ex_date)
    );
    
    -- Excess Cash Flow Metrics (Core ACIS Investment Metric)
    CREATE TABLE IF NOT EXISTS excess_cash_flow_metrics (
        symbol VARCHAR(10) PRIMARY KEY,
        latest_date DATE,
        cash_flow_per_share NUMERIC(12, 4),
        dividends_per_share NUMERIC(12, 4),
        capex_per_share NUMERIC(12, 4),
        excess_cash_flow NUMERIC(12, 4),
        excess_cash_flow_pct NUMERIC(6, 2),
        quality_rating VARCHAR(20),
        trend_5y VARCHAR(20),
        trend_10y VARCHAR(20),
        avg_excess_cf_5y NUMERIC(6, 2),
        avg_excess_cf_10y NUMERIC(6, 2),
        rank INTEGER,
        percentile NUMERIC(5, 2),
        calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (symbol) REFERENCES symbol_universe(symbol) ON DELETE CASCADE
    );
    
    -- Dividend Sustainability Metrics (ACIS Dividend Growth Strategy)
    CREATE TABLE IF NOT EXISTS dividend_sustainability_metrics (
        symbol VARCHAR(10) PRIMARY KEY,
        payment_streak_years INTEGER,
        increase_streak_years INTEGER,
        avg_growth_rate_5y NUMERIC(8, 2),
        payout_ratio_earnings NUMERIC(8, 2),
        payout_ratio_fcf NUMERIC(8, 2),
        sustainability_score NUMERIC(6, 2),
        safety_rating VARCHAR(20),
        excess_cash_flow_pct NUMERIC(6, 2),
        cash_flow_quality VARCHAR(20),
        dividend_quality_score NUMERIC(6, 2),
        dividend_quality_rating VARCHAR(30),
        rank INTEGER,
        percentile NUMERIC(5, 2),
        calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (symbol) REFERENCES symbol_universe(symbol) ON DELETE CASCADE
    );
    
    -- Breakout Signals (Technical Analysis with Volume)
    CREATE TABLE IF NOT EXISTS breakout_signals (
        symbol VARCHAR(10) NOT NULL,
        signal_date DATE NOT NULL,
        breakout_score NUMERIC(6, 2),
        has_breakout BOOLEAN,
        current_price NUMERIC(12, 4),
        high_52w NUMERIC(12, 4),
        volume_surge NUMERIC(8, 2),
        base_quality NUMERIC(6, 2),
        momentum_score NUMERIC(6, 2),
        relative_strength NUMERIC(8, 4),
        accumulation_ratio NUMERIC(8, 4),
        breakout_strength VARCHAR(20),
        rank INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, signal_date),
        FOREIGN KEY (symbol) REFERENCES symbol_universe(symbol) ON DELETE CASCADE
    );
    
    -- Company Fundamentals Overview (detailed metrics from Alpha Vantage OVERVIEW)
    CREATE TABLE IF NOT EXISTS company_fundamentals_overview (
        symbol VARCHAR(10) PRIMARY KEY,
        
        -- Valuation Metrics
        pe_ratio NUMERIC(10, 2),
        peg_ratio NUMERIC(10, 2),
        book_value NUMERIC(10, 2),
        price_to_book NUMERIC(10, 2),
        price_to_sales NUMERIC(10, 2),
        ev_to_revenue NUMERIC(10, 2),
        ev_to_ebitda NUMERIC(10, 2),
        forward_pe NUMERIC(10, 2),
        trailing_pe NUMERIC(10, 2),
        
        -- Profitability Metrics
        profit_margin NUMERIC(10, 4),
        operating_margin NUMERIC(10, 4),
        return_on_assets NUMERIC(10, 4),
        return_on_equity NUMERIC(10, 4),
        
        -- Revenue & Earnings
        revenue_ttm NUMERIC(20, 2),
        revenue_per_share NUMERIC(10, 2),
        quarterly_revenue_growth NUMERIC(10, 4),
        gross_profit_ttm NUMERIC(20, 2),
        ebitda NUMERIC(20, 2),
        eps NUMERIC(10, 2),
        quarterly_earnings_growth NUMERIC(10, 4),
        
        -- Dividend Information
        dividend_per_share NUMERIC(10, 4),
        dividend_yield NUMERIC(10, 4),
        dividend_date DATE,
        ex_dividend_date DATE,
        
        -- Share Information
        shares_float NUMERIC(20, 2),
        beta NUMERIC(10, 3),
        
        -- Analyst Information
        analyst_target_price NUMERIC(10, 2),
        analyst_rating_strong_buy INTEGER,
        analyst_rating_buy INTEGER,
        analyst_rating_hold INTEGER,
        analyst_rating_sell INTEGER,
        analyst_rating_strong_sell INTEGER,
        
        -- Dates
        fiscal_year_end VARCHAR(10),
        latest_quarter DATE,
        
        -- Metadata
        fetched_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        
        FOREIGN KEY (symbol) REFERENCES symbol_universe(symbol) ON DELETE CASCADE
    );
    
    -- SP500 Outperformance Detail
    CREATE TABLE IF NOT EXISTS sp500_outperformance_detail (
        symbol TEXT NOT NULL,
        calculation_date DATE NOT NULL,
        
        -- Stock returns
        stock_return_1m NUMERIC(10,4),
        stock_return_3m NUMERIC(10,4),
        stock_return_6m NUMERIC(10,4),
        stock_return_12m NUMERIC(10,4),
        stock_return_36m NUMERIC(10,4),
        
        -- SP500 returns
        sp500_return_1m NUMERIC(10,4),
        sp500_return_3m NUMERIC(10,4),
        sp500_return_6m NUMERIC(10,4),
        sp500_return_12m NUMERIC(10,4),
        sp500_return_36m NUMERIC(10,4),
        
        -- Relative performance (alpha)
        relative_return_1m NUMERIC(10,4),
        relative_return_3m NUMERIC(10,4),
        relative_return_6m NUMERIC(10,4),
        relative_return_12m NUMERIC(10,4),
        relative_return_36m NUMERIC(10,4),
        
        -- Weighted composite score
        outperformance_score NUMERIC(10,4),
        
        -- Rankings within universe
        rank_1m INTEGER,
        rank_3m INTEGER,
        rank_6m INTEGER,
        rank_12m INTEGER,
        rank_36m INTEGER,
        rank_composite INTEGER,
        
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, calculation_date)
    );
    """
    return SQL

def create_portfolio_tables():
    """Create portfolio management tables"""
    SQL = """
    -- Portfolio Scores (Three scores for every stock)
    CREATE TABLE IF NOT EXISTS portfolio_scores (
        symbol VARCHAR(10) NOT NULL,
        calculation_date DATE NOT NULL,
        
        -- Individual component scores
        value_score NUMERIC(6, 2),
        growth_score NUMERIC(6, 2),
        dividend_score NUMERIC(6, 2),
        
        -- Component breakdowns for Value Score
        valuation_percentile NUMERIC(6, 2),
        excess_cf_yield NUMERIC(6, 2),
        margin_of_safety NUMERIC(6, 2),
        
        -- Component breakdowns for Growth Score
        sp500_alpha_10y NUMERIC(8, 2),
        fundamental_growth_5y NUMERIC(8, 2),
        forward_growth_estimate NUMERIC(8, 2),
        
        -- Component breakdowns for Dividend Score
        dividend_sustainability NUMERIC(6, 2),
        dividend_growth_rate NUMERIC(8, 2),
        payment_history_score NUMERIC(6, 2),
        
        -- Breakout Signal (Technical Timing)
        breakout_score NUMERIC(6, 2),
        has_active_breakout BOOLEAN DEFAULT FALSE,
        volume_surge_ratio NUMERIC(8, 2),
        days_since_breakout INTEGER,
        
        -- Adjusted scores with breakout bonus
        value_score_adjusted NUMERIC(6, 2),
        growth_score_adjusted NUMERIC(6, 2),
        
        -- Rankings
        value_rank INTEGER,
        growth_rank INTEGER,
        dividend_rank INTEGER,
        
        -- Metadata
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, calculation_date),
        FOREIGN KEY (symbol) REFERENCES symbol_universe(symbol) ON DELETE CASCADE
    );
    
    -- Insider Transactions (Raw transaction data)
    CREATE TABLE IF NOT EXISTS insider_transactions (
        transaction_id SERIAL PRIMARY KEY,
        symbol VARCHAR(10) NOT NULL,
        transaction_date DATE NOT NULL,
        reporting_date DATE,
        
        -- Insider details
        owner_name VARCHAR(200),
        owner_title VARCHAR(200),
        owner_type VARCHAR(50),
        
        -- Transaction details
        transaction_type VARCHAR(10) CHECK (transaction_type IN ('BUY', 'SELL')),
        shares NUMERIC(12, 0),
        price_per_share NUMERIC(12, 4),
        total_value NUMERIC(15, 2),
        shares_owned_after NUMERIC(12, 0),
        
        -- Insider flags
        is_ceo BOOLEAN DEFAULT FALSE,
        is_cfo BOOLEAN DEFAULT FALSE,
        is_director BOOLEAN DEFAULT FALSE,
        is_officer BOOLEAN DEFAULT FALSE,
        is_10b51_plan BOOLEAN DEFAULT FALSE,  -- Scheduled sale
        
        -- Metadata
        fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        
        -- Prevent duplicates
        UNIQUE(symbol, transaction_date, owner_name, transaction_type, shares),
        FOREIGN KEY (symbol) REFERENCES symbol_universe(symbol) ON DELETE CASCADE
    );
    
    -- Insider Signals (Aggregated scores and signals)
    CREATE TABLE IF NOT EXISTS insider_signals (
        symbol VARCHAR(10) PRIMARY KEY,
        calculation_date DATE NOT NULL,
        
        -- 30-day metrics
        buys_30d INTEGER DEFAULT 0,
        sells_30d INTEGER DEFAULT 0,
        buy_value_30d NUMERIC(15, 2),
        sell_value_30d NUMERIC(15, 2),
        
        -- 90-day metrics
        buys_90d INTEGER DEFAULT 0,
        sells_90d INTEGER DEFAULT 0,
        
        -- Key signals
        ceo_buying BOOLEAN DEFAULT FALSE,
        cfo_buying BOOLEAN DEFAULT FALSE,
        officer_cluster_buying BOOLEAN DEFAULT FALSE,  -- 3+ officers buying
        
        -- Net sentiment
        net_insider_buying_30d NUMERIC(15, 2),  -- Positive = net buying
        
        -- Composite score (0-100)
        insider_score NUMERIC(5, 2),
        
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (symbol) REFERENCES symbol_universe(symbol) ON DELETE CASCADE
    );
    
    -- Earnings Estimates (analyst consensus forecasts)
    CREATE TABLE IF NOT EXISTS earnings_estimates (
        symbol VARCHAR(10) NOT NULL,
        fiscal_period DATE NOT NULL,
        period_type VARCHAR(10) CHECK (period_type IN ('quarterly', 'annual')),
        
        -- EPS estimates
        eps_consensus NUMERIC(12, 4),
        eps_high NUMERIC(12, 4),
        eps_low NUMERIC(12, 4),
        eps_num_estimates INTEGER,
        
        -- Revenue estimates
        revenue_consensus NUMERIC(20, 2),
        revenue_high NUMERIC(20, 2),
        revenue_low NUMERIC(20, 2),
        
        -- Actual reported values (for historical surprise)
        reported_eps NUMERIC(12, 4),
        reported_date DATE,
        surprise_amount NUMERIC(12, 4),
        surprise_percentage NUMERIC(12, 2),
        
        fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, fiscal_period, period_type)
    );
    
    -- Altman Z-Score (bankruptcy risk prediction)
    CREATE TABLE IF NOT EXISTS altman_zscores (
        symbol VARCHAR(10) NOT NULL,
        calculation_date DATE NOT NULL,
        fiscal_date DATE,
        
        -- Z-Scores
        z_score NUMERIC(12, 2),  -- Manufacturing companies
        z_prime_score NUMERIC(12, 2),  -- Non-manufacturing companies
        
        -- Components (X1-X5)
        working_capital_ratio NUMERIC(12, 4),  -- X1: Working Capital / Total Assets
        retained_earnings_ratio NUMERIC(12, 4),  -- X2: Retained Earnings / Total Assets
        ebit_ratio NUMERIC(12, 4),  -- X3: EBIT / Total Assets
        market_to_book_liability NUMERIC(12, 4),  -- X4: Market Cap / Total Liabilities
        sales_to_assets NUMERIC(12, 4),  -- X5: Sales / Total Assets
        
        -- Risk assessment
        risk_category VARCHAR(20) CHECK (risk_category IN ('SAFE', 'GREY', 'DISTRESS')),
        is_manufacturing BOOLEAN DEFAULT FALSE,
        
        -- Reference values
        total_assets NUMERIC(20, 2),
        total_liabilities NUMERIC(20, 2),
        market_cap NUMERIC(20, 2),
        
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, calculation_date)
    );
    
    -- Piotroski F-Score (9-point fundamental strength)
    CREATE TABLE IF NOT EXISTS piotroski_scores (
        symbol VARCHAR(10) NOT NULL,
        calculation_date DATE NOT NULL,
        fscore INTEGER CHECK (fscore >= 0 AND fscore <= 9),
        fiscal_date DATE,
        
        -- Profitability components (4 points)
        positive_net_income BOOLEAN DEFAULT FALSE,
        positive_ocf BOOLEAN DEFAULT FALSE,
        increasing_roa BOOLEAN DEFAULT FALSE,
        quality_earnings BOOLEAN DEFAULT FALSE,  -- OCF > Net Income
        
        -- Leverage/Liquidity components (3 points)
        decreasing_leverage BOOLEAN DEFAULT FALSE,
        improving_liquidity BOOLEAN DEFAULT FALSE,
        no_dilution BOOLEAN DEFAULT FALSE,  -- No new shares issued
        
        -- Efficiency components (2 points)
        improving_margin BOOLEAN DEFAULT FALSE,
        improving_efficiency BOOLEAN DEFAULT FALSE,  -- Asset turnover
        
        -- Key metrics
        current_roa NUMERIC(12, 4),
        prior_roa NUMERIC(12, 4),
        current_margin NUMERIC(12, 4),
        prior_margin NUMERIC(12, 4),
        current_debt_ratio NUMERIC(12, 4),
        current_ratio NUMERIC(12, 4),
        current_turnover NUMERIC(12, 4),
        
        -- Classification
        strength_category VARCHAR(20) CHECK (strength_category IN ('STRONG', 'MODERATE', 'WEAK')),
        
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, calculation_date),
        FOREIGN KEY (symbol) REFERENCES symbol_universe(symbol) ON DELETE CASCADE
    );
    
    -- Portfolio Holdings (Current positions)
    CREATE TABLE IF NOT EXISTS portfolio_holdings (
        portfolio_id SERIAL PRIMARY KEY,
        portfolio_type VARCHAR(20) NOT NULL CHECK (portfolio_type IN ('VALUE', 'GROWTH', 'DIVIDEND')),
        symbol VARCHAR(10) NOT NULL,
        selection_date DATE NOT NULL,
        rebalance_date DATE,
        
        -- Selection metrics at time of inclusion
        selection_score NUMERIC(6, 2),
        selection_rank INTEGER,
        
        -- Position sizing
        initial_weight NUMERIC(5, 2) DEFAULT 10.0,
        current_weight NUMERIC(5, 2),
        
        -- Tracking
        entry_price NUMERIC(12, 4),
        current_price NUMERIC(12, 4),
        total_return NUMERIC(8, 2),
        
        -- Status
        is_active BOOLEAN DEFAULT TRUE,
        exit_date DATE,
        exit_reason VARCHAR(100),
        
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (symbol) REFERENCES symbol_universe(symbol) ON DELETE CASCADE
    );
    
    -- Portfolio Rebalances (History of changes)
    CREATE TABLE IF NOT EXISTS portfolio_rebalances (
        rebalance_id SERIAL PRIMARY KEY,
        portfolio_type VARCHAR(20) NOT NULL CHECK (portfolio_type IN ('VALUE', 'GROWTH', 'DIVIDEND')),
        rebalance_date DATE NOT NULL,
        rebalance_type VARCHAR(20) CHECK (rebalance_type IN ('SCHEDULED', 'FORCED', 'OPPORTUNISTIC')),
        
        -- Changes made
        stocks_added TEXT[],
        stocks_removed TEXT[],
        stocks_retained TEXT[],
        
        -- Portfolio metrics before/after
        avg_score_before NUMERIC(6, 2),
        avg_score_after NUMERIC(6, 2),
        
        -- Reason for rebalance
        trigger_reason TEXT,
        
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- Portfolio Performance (Track returns)
    CREATE TABLE IF NOT EXISTS portfolio_performance (
        performance_id SERIAL PRIMARY KEY,
        portfolio_type VARCHAR(20) NOT NULL CHECK (portfolio_type IN ('VALUE', 'GROWTH', 'DIVIDEND')),
        measurement_date DATE NOT NULL,
        
        -- Returns
        daily_return NUMERIC(8, 4),
        mtd_return NUMERIC(8, 4),
        qtd_return NUMERIC(8, 4),
        ytd_return NUMERIC(8, 4),
        total_return NUMERIC(8, 4),
        
        -- vs Benchmark
        sp500_daily_return NUMERIC(8, 4),
        sp500_ytd_return NUMERIC(8, 4),
        alpha_ytd NUMERIC(8, 4),
        
        -- Risk metrics
        volatility_30d NUMERIC(8, 4),
        sharpe_ratio NUMERIC(8, 4),
        max_drawdown NUMERIC(8, 4),
        
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(portfolio_type, measurement_date)
    );
    """
    return SQL

def create_indexes():
    """Create indexes for better query performance"""
    SQL = """
    -- Symbol Universe indexes
    CREATE INDEX IF NOT EXISTS idx_symbol_universe_market_cap 
        ON symbol_universe(market_cap DESC);
    CREATE INDEX IF NOT EXISTS idx_symbol_universe_sector 
        ON symbol_universe(sector);
    CREATE INDEX IF NOT EXISTS idx_symbol_universe_etf 
        ON symbol_universe(is_etf);
    
    -- Stock Prices indexes
    CREATE INDEX IF NOT EXISTS idx_stock_prices_symbol_date 
        ON stock_prices(symbol, trade_date DESC);
    CREATE INDEX IF NOT EXISTS idx_stock_prices_date 
        ON stock_prices(trade_date DESC);
    
    -- Fundamentals indexes
    CREATE INDEX IF NOT EXISTS idx_fundamentals_symbol_date 
        ON fundamentals(symbol, fiscal_date_ending DESC);
    
    -- Dividend History indexes
    CREATE INDEX IF NOT EXISTS idx_dividend_history_symbol_date 
        ON dividend_history(symbol, ex_date DESC);
    
    -- Insider Transactions indexes
    CREATE INDEX IF NOT EXISTS idx_insider_transactions_symbol_date 
        ON insider_transactions(symbol, transaction_date DESC);
    CREATE INDEX IF NOT EXISTS idx_insider_transactions_date 
        ON insider_transactions(transaction_date DESC);
    CREATE INDEX IF NOT EXISTS idx_insider_transactions_type 
        ON insider_transactions(transaction_type);
    CREATE INDEX IF NOT EXISTS idx_insider_transactions_ceo 
        ON insider_transactions(is_ceo) WHERE is_ceo = TRUE;
    CREATE INDEX IF NOT EXISTS idx_insider_transactions_cfo 
        ON insider_transactions(is_cfo) WHERE is_cfo = TRUE;
    
    -- Insider Signals indexes
    CREATE INDEX IF NOT EXISTS idx_insider_signals_score 
        ON insider_signals(insider_score DESC);
    CREATE INDEX IF NOT EXISTS idx_insider_signals_ceo_buying 
        ON insider_signals(ceo_buying) WHERE ceo_buying = TRUE;
    CREATE INDEX IF NOT EXISTS idx_insider_signals_cluster 
        ON insider_signals(officer_cluster_buying) WHERE officer_cluster_buying = TRUE;
    
    -- Earnings Estimates indexes
    CREATE INDEX IF NOT EXISTS idx_earnings_estimates_period 
        ON earnings_estimates(fiscal_period DESC);
    CREATE INDEX IF NOT EXISTS idx_earnings_estimates_symbol 
        ON earnings_estimates(symbol, fiscal_period DESC);
    CREATE INDEX IF NOT EXISTS idx_earnings_estimates_upcoming 
        ON earnings_estimates(fiscal_period) WHERE fiscal_period >= CURRENT_DATE;
    CREATE INDEX IF NOT EXISTS idx_earnings_estimates_high_coverage 
        ON earnings_estimates(eps_num_estimates DESC) WHERE eps_num_estimates >= 5;
    
    -- Altman Z-Score indexes
    CREATE INDEX IF NOT EXISTS idx_altman_zscores_risk 
        ON altman_zscores(risk_category);
    CREATE INDEX IF NOT EXISTS idx_altman_zscores_date 
        ON altman_zscores(calculation_date DESC);
    CREATE INDEX IF NOT EXISTS idx_altman_zscores_safe 
        ON altman_zscores(z_score DESC) WHERE risk_category = 'SAFE';
    CREATE INDEX IF NOT EXISTS idx_altman_zscores_distress 
        ON altman_zscores(z_score) WHERE risk_category = 'DISTRESS';
    
    -- Piotroski F-Score indexes
    CREATE INDEX IF NOT EXISTS idx_piotroski_scores_fscore 
        ON piotroski_scores(fscore DESC);
    CREATE INDEX IF NOT EXISTS idx_piotroski_scores_date 
        ON piotroski_scores(calculation_date DESC);
    CREATE INDEX IF NOT EXISTS idx_piotroski_scores_strong 
        ON piotroski_scores(fscore) WHERE fscore >= 8;
    CREATE INDEX IF NOT EXISTS idx_piotroski_scores_weak 
        ON piotroski_scores(fscore) WHERE fscore <= 2;
    
    -- Portfolio Scores indexes
    CREATE INDEX IF NOT EXISTS idx_portfolio_scores_date 
        ON portfolio_scores(calculation_date DESC);
    CREATE INDEX IF NOT EXISTS idx_portfolio_scores_value_rank 
        ON portfolio_scores(value_rank) WHERE value_rank IS NOT NULL;
    CREATE INDEX IF NOT EXISTS idx_portfolio_scores_growth_rank 
        ON portfolio_scores(growth_rank) WHERE growth_rank IS NOT NULL;
    CREATE INDEX IF NOT EXISTS idx_portfolio_scores_dividend_rank 
        ON portfolio_scores(dividend_rank) WHERE dividend_rank IS NOT NULL;
    
    -- Portfolio Holdings indexes
    CREATE INDEX IF NOT EXISTS idx_portfolio_holdings_active 
        ON portfolio_holdings(portfolio_type, is_active) WHERE is_active = TRUE;
    
    -- Breakout Signals indexes
    CREATE INDEX IF NOT EXISTS idx_breakout_signals_date 
        ON breakout_signals(signal_date DESC);
    CREATE INDEX IF NOT EXISTS idx_breakout_signals_active 
        ON breakout_signals(has_breakout) WHERE has_breakout = TRUE;
    """
    return SQL

def main():
    """Main execution function."""
    start_time = time.time()
    
    print("\n" + "=" * 60)
    print("ACIS DATABASE SCHEMA SETUP - CLEAN VERSION")
    print("Three-Portfolio Strategy Tables Only")
    print("=" * 60)
    
    try:
        postgres_url = get_postgres_url()
        engine = create_engine(postgres_url)
        
        # Create tables in logical groups
        chunks = [
            ("Core Tables", create_core_tables()),
            ("Fundamentals Tables", create_fundamentals_tables()),
            ("Portfolio Tables", create_portfolio_tables()),
            ("Indexes", create_indexes())
        ]
        
        for name, sql in chunks:
            print(f"\n[INFO] Creating {name}...")
            try:
                with engine.begin() as conn:
                    conn.execute(text(sql))
                print(f"[SUCCESS] {name} created")
            except Exception as e:
                print(f"[ERROR] Failed to create {name}: {e}")
                # Continue with other chunks even if one fails
        
        # Verify tables were created
        print("\n[INFO] Verifying tables...")
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                  AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """))
            
            tables = [row[0] for row in result]
            print(f"\n[SUCCESS] Found {len(tables)} tables:")
            
            # Group tables by category
            core_tables = ['symbol_universe', 'stock_prices', 'sp500_history', 'forward_returns']
            fundamental_tables = ['fundamentals', 'company_fundamentals_overview', 'dividend_history']
            strategy_tables = ['excess_cash_flow_metrics', 'dividend_sustainability_metrics', 
                             'breakout_signals', 'sp500_outperformance_detail']
            portfolio_tables = ['portfolio_scores', 'portfolio_holdings', 
                              'portfolio_rebalances', 'portfolio_performance']
            
            print("\nCore Tables:")
            for table in tables:
                if table in core_tables:
                    print(f"  ✓ {table}")
            
            print("\nFundamental Data Tables:")
            for table in tables:
                if table in fundamental_tables:
                    print(f"  ✓ {table}")
            
            print("\nStrategy Analysis Tables:")
            for table in tables:
                if table in strategy_tables:
                    print(f"  ✓ {table}")
            
            print("\nPortfolio Management Tables:")
            for table in tables:
                if table in portfolio_tables:
                    print(f"  ✓ {table}")
            
            print("\nOther Tables (may need removal):")
            for table in tables:
                if table not in (core_tables + fundamental_tables + strategy_tables + portfolio_tables):
                    print(f"  ? {table}")
        
        duration = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"[SUCCESS] Schema setup completed in {duration:.1f} seconds")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[ERROR] Schema setup failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())