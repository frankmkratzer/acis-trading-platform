#!/usr/bin/env python3
"""
ACIS Database Schema with Performance Optimizations
Creates tables with proper columns and performance indexes
"""

import os
import sys
import time
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

def get_postgres_url():
    """Get PostgreSQL URL from environment."""
    postgres_url = os.getenv("POSTGRES_URL")
    if not postgres_url:
        raise RuntimeError("POSTGRES_URL not found in environment variables.")
    return postgres_url

def create_essential_schema():
    """Create essential tables with correct columns and performance indexes."""
    
    ESSENTIAL_SCHEMA_SQL = """
    -- =============================================
    -- ACIS DATA PIPELINE TABLES WITH PERFORMANCE INDEXES
    -- =============================================
    
    -- 1. Symbol Universe (master symbol list)
    CREATE TABLE IF NOT EXISTS symbol_universe (
        symbol TEXT PRIMARY KEY,
        name TEXT,  -- Changed from company_name
        exchange TEXT,
        sector TEXT,
        industry TEXT,
        market_cap BIGINT,
        country TEXT DEFAULT 'USA',  -- Changed from 'US' to 'USA'
        currency TEXT DEFAULT 'USD',
        security_type TEXT DEFAULT 'Common Stock',  -- Added security_type
        is_etf BOOLEAN DEFAULT FALSE,
        ipo_date DATE,
        delisted_date DATE,
        fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- Changed from created_at
        CONSTRAINT chk_su_usd_country CHECK (
            (currency = 'USD' AND country = 'USA') OR 
            currency != 'USD'
        )
    );
    
    -- Performance indexes for symbol_universe
    CREATE INDEX IF NOT EXISTS idx_symbol_universe_exchange ON symbol_universe(exchange);
    CREATE INDEX IF NOT EXISTS idx_symbol_universe_sector ON symbol_universe(sector);
    CREATE INDEX IF NOT EXISTS idx_symbol_universe_market_cap ON symbol_universe(market_cap);
    CREATE INDEX IF NOT EXISTS idx_symbol_universe_is_etf_country ON symbol_universe(is_etf, country);
    CREATE INDEX IF NOT EXISTS idx_universe_sector_industry ON symbol_universe(sector, industry);
    
    -- 2. Stock Prices (core price data)
    CREATE TABLE IF NOT EXISTS stock_prices (
        symbol TEXT NOT NULL,
        trade_date DATE NOT NULL,  -- Changed from date to trade_date
        open NUMERIC,
        high NUMERIC,
        low NUMERIC,
        close NUMERIC NOT NULL,
        adjusted_close NUMERIC,
        volume BIGINT,
        dividend_amount NUMERIC DEFAULT 0,
        split_coefficient NUMERIC DEFAULT 1,
        fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- Changed from created_at
        PRIMARY KEY (symbol, trade_date)
    );
    
    -- Performance indexes for stock_prices
    CREATE INDEX IF NOT EXISTS idx_stock_prices_symbol ON stock_prices(symbol);
    CREATE INDEX IF NOT EXISTS idx_stock_prices_trade_date ON stock_prices(trade_date DESC);
    CREATE INDEX IF NOT EXISTS idx_stock_prices_symbol_date ON stock_prices(symbol, trade_date DESC);
    -- New performance indexes
    CREATE INDEX IF NOT EXISTS idx_stock_prices_date_symbol ON stock_prices(trade_date DESC, symbol);
    CREATE INDEX IF NOT EXISTS idx_stock_prices_volume ON stock_prices(volume DESC) WHERE volume > 0;
    
    -- 3. Technical Indicators
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
    
    -- Performance indexes for technical_indicators
    CREATE INDEX IF NOT EXISTS idx_tech_indicators_symbol ON technical_indicators(symbol);
    CREATE INDEX IF NOT EXISTS idx_tech_indicators_date ON technical_indicators(date DESC);
    CREATE INDEX IF NOT EXISTS idx_tech_indicators_symbol_date ON technical_indicators(symbol, date DESC);
    -- New performance indexes
    CREATE INDEX IF NOT EXISTS idx_tech_indicators_rsi ON technical_indicators(rsi_14) WHERE rsi_14 IS NOT NULL;
    CREATE INDEX IF NOT EXISTS idx_tech_indicators_macd ON technical_indicators(macd) WHERE macd IS NOT NULL;
    
    -- 4. Fundamentals (combined annual and quarterly)
    CREATE TABLE IF NOT EXISTS fundamentals (
        symbol TEXT NOT NULL,
        fiscal_date_ending DATE NOT NULL,
        period_type TEXT NOT NULL CHECK (period_type IN ('annual', 'quarterly')),
        reported_date DATE,
        revenue BIGINT,
        cost_of_revenue BIGINT,
        gross_profit BIGINT,
        operating_expenses BIGINT,
        operating_income BIGINT,
        net_income BIGINT,
        earnings_per_share NUMERIC,
        diluted_earnings_per_share NUMERIC,
        total_assets BIGINT,
        total_liabilities BIGINT,
        total_shareholder_equity BIGINT,
        cash_and_equivalents BIGINT,
        total_debt BIGINT,
        free_cash_flow BIGINT,
        operating_cash_flow BIGINT,
        shares_outstanding BIGINT,
        book_value_per_share NUMERIC,
        pe_ratio NUMERIC,
        peg_ratio NUMERIC,
        pb_ratio NUMERIC,
        ps_ratio NUMERIC,
        ev_to_revenue NUMERIC,
        ev_to_ebitda NUMERIC,
        debt_to_equity NUMERIC,
        current_ratio NUMERIC,
        quick_ratio NUMERIC,
        gross_margin NUMERIC,
        operating_margin NUMERIC,
        net_margin NUMERIC,
        return_on_equity NUMERIC,
        return_on_assets NUMERIC,
        return_on_invested_capital NUMERIC,
        fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, fiscal_date_ending, period_type)
    );
    
    -- Performance indexes for fundamentals
    CREATE INDEX IF NOT EXISTS idx_fundamentals_symbol ON fundamentals(symbol);
    CREATE INDEX IF NOT EXISTS idx_fundamentals_period ON fundamentals(period_type);
    CREATE INDEX IF NOT EXISTS idx_fundamentals_symbol_period ON fundamentals(symbol, fiscal_date_ending DESC, period_type);
    -- New performance indexes
    CREATE INDEX IF NOT EXISTS idx_fundamentals_fiscal_date ON fundamentals(fiscal_date_ending DESC);
    CREATE INDEX IF NOT EXISTS idx_fundamentals_pe_ratio ON fundamentals(pe_ratio) WHERE pe_ratio IS NOT NULL AND pe_ratio > 0;
    CREATE INDEX IF NOT EXISTS idx_fundamentals_roe ON fundamentals(return_on_equity) WHERE return_on_equity IS NOT NULL;
    
    -- 5. Options Data
    CREATE TABLE IF NOT EXISTS options_data (
        symbol TEXT NOT NULL,
        contract_id TEXT NOT NULL,
        option_type TEXT NOT NULL CHECK (option_type IN ('call', 'put')),
        strike_price NUMERIC NOT NULL,
        expiration_date DATE NOT NULL,
        quote_date DATE NOT NULL,
        bid NUMERIC,
        ask NUMERIC,
        last_price NUMERIC,
        volume INTEGER,
        open_interest INTEGER,
        implied_volatility NUMERIC,
        delta NUMERIC,
        gamma NUMERIC,
        theta NUMERIC,
        vega NUMERIC,
        rho NUMERIC,
        in_the_money BOOLEAN,
        time_value NUMERIC,
        intrinsic_value NUMERIC,
        fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (contract_id, quote_date)
    );
    
    -- Performance indexes for options_data
    CREATE INDEX IF NOT EXISTS idx_options_symbol ON options_data(symbol);
    CREATE INDEX IF NOT EXISTS idx_options_expiration ON options_data(expiration_date);
    CREATE INDEX IF NOT EXISTS idx_options_symbol_exp_strike ON options_data(symbol, expiration_date, strike_price);
    -- New performance indexes
    CREATE INDEX IF NOT EXISTS idx_options_quote_date ON options_data(quote_date DESC);
    CREATE INDEX IF NOT EXISTS idx_options_volume ON options_data(volume DESC) WHERE volume > 0;
    CREATE INDEX IF NOT EXISTS idx_options_open_interest ON options_data(open_interest DESC) WHERE open_interest > 0;
    
    -- 6. S&P 500 Price History
    CREATE TABLE IF NOT EXISTS sp500_price_history (
        trade_date DATE PRIMARY KEY,
        open NUMERIC,
        high NUMERIC,
        low NUMERIC,
        close NUMERIC NOT NULL,
        adjusted_close NUMERIC,
        volume BIGINT,
        dividend_amount NUMERIC DEFAULT 0,
        split_coefficient NUMERIC DEFAULT 1,
        fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_sp500_date ON sp500_price_history(trade_date DESC);
    
    -- 7. Dividend History
    CREATE TABLE IF NOT EXISTS dividend_history (
        symbol TEXT NOT NULL,
        ex_date DATE NOT NULL,
        dividend NUMERIC,
        currency TEXT DEFAULT 'USD',
        fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, ex_date)
    );
    
    CREATE INDEX IF NOT EXISTS idx_dividends_symbol ON dividend_history(symbol);
    CREATE INDEX IF NOT EXISTS idx_dividends_ex_date ON dividend_history(ex_date DESC);
    
    -- 8. Forward Returns (for ML features)
    CREATE TABLE IF NOT EXISTS forward_returns (
        symbol TEXT NOT NULL,
        as_of_date DATE NOT NULL,
        return_1m NUMERIC,   -- 1 month (21 trading days)
        return_3m NUMERIC,   -- 3 months (63 trading days)
        return_6m NUMERIC,   -- 6 months (126 trading days)
        return_12m NUMERIC,  -- 12 months (252 trading days)
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, as_of_date)
    );
    
    CREATE INDEX IF NOT EXISTS idx_forward_returns_symbol_date ON forward_returns(symbol, as_of_date DESC);
    CREATE INDEX IF NOT EXISTS idx_forward_returns_date ON forward_returns(as_of_date DESC);
    
    -- =============================================
    -- COMPOSITE INDEXES FOR COMMON QUERIES
    -- =============================================
    
    -- For joining prices with indicators
    CREATE INDEX IF NOT EXISTS idx_prices_indicators_join 
        ON stock_prices(symbol, trade_date) 
        WHERE adjusted_close IS NOT NULL;
    
    -- For finding latest data per symbol (simplified without date filter)
    CREATE INDEX IF NOT EXISTS idx_latest_prices 
        ON stock_prices(symbol, trade_date DESC);
    
    -- For screening stocks by fundamentals
    CREATE INDEX IF NOT EXISTS idx_fundamental_screening 
        ON fundamentals(pe_ratio, return_on_equity, debt_to_equity) 
        WHERE period_type = 'annual' 
        AND pe_ratio BETWEEN 0 AND 100 
        AND return_on_equity > 0;
    
    -- For options chain queries
    CREATE INDEX IF NOT EXISTS idx_options_chain 
        ON options_data(symbol, expiration_date, strike_price, option_type) 
        WHERE volume > 0 OR open_interest > 0;
    
    -- =============================================
    -- MATERIALIZED VIEWS FOR PERFORMANCE
    -- =============================================
    
    -- Latest price per symbol (refreshed daily)
    CREATE MATERIALIZED VIEW IF NOT EXISTS mv_latest_prices AS
    SELECT DISTINCT ON (symbol) 
        symbol, 
        trade_date, 
        open, 
        high, 
        low, 
        close, 
        adjusted_close, 
        volume
    FROM stock_prices 
    WHERE trade_date IS NOT NULL
    ORDER BY symbol, trade_date DESC;
    
    CREATE UNIQUE INDEX IF NOT EXISTS ux_mv_latest_prices ON mv_latest_prices(symbol);
    
    -- Latest fundamentals per symbol
    CREATE MATERIALIZED VIEW IF NOT EXISTS mv_latest_fundamentals AS
    SELECT DISTINCT ON (symbol, period_type)
        symbol, 
        period_type,
        fiscal_date_ending, 
        revenue, 
        net_income, 
        earnings_per_share, 
        pe_ratio, 
        return_on_equity, 
        debt_to_equity, 
        free_cash_flow
    FROM fundamentals
    WHERE fiscal_date_ending IS NOT NULL
    ORDER BY symbol, period_type, fiscal_date_ending DESC;
    
    CREATE UNIQUE INDEX IF NOT EXISTS ux_mv_latest_fundamentals ON mv_latest_fundamentals(symbol, period_type);
    
    -- Active options contracts
    CREATE MATERIALIZED VIEW IF NOT EXISTS mv_active_options AS
    SELECT 
        symbol,
        expiration_date,
        COUNT(DISTINCT contract_id) as contract_count,
        SUM(volume) as total_volume,
        SUM(open_interest) as total_open_interest,
        MAX(quote_date) as latest_quote
    FROM options_data
    WHERE expiration_date >= CURRENT_DATE
    AND (volume > 0 OR open_interest > 0)
    GROUP BY symbol, expiration_date;
    
    CREATE INDEX IF NOT EXISTS ix_mv_active_options ON mv_active_options(symbol, expiration_date);
    
    -- =============================================
    -- STOCK QUALITY RANKING SYSTEM
    -- Initial screening to identify best stocks based on 3 core rankings
    -- =============================================
    
    -- Master quality screening table
    CREATE TABLE IF NOT EXISTS stock_quality_rankings (
        symbol TEXT NOT NULL,
        ranking_date DATE NOT NULL,
        
        -- 1. SP500 Outperformance Ranking
        beat_sp500_ranking INTEGER,  -- 1 = best (beat SP500 most years with recent bias)
        years_beating_sp500 INTEGER,  -- Out of last 20 years
        sp500_weighted_score NUMERIC(10,4),  -- Recent years weighted higher (decay factor)
        avg_annual_excess NUMERIC(10,4),  -- Average excess return vs SP500
        recent_5yr_beat_count INTEGER,  -- How many of last 5 years beat SP500
        recent_1yr_excess NUMERIC(10,4),  -- Most recent year excess return
        
        -- 2. Excess Cash Flow Ranking  
        excess_cash_flow_ranking INTEGER,  -- 1 = best cash generator
        fcf_yield NUMERIC(10,4),  -- Current FCF/Market Cap
        fcf_margin NUMERIC(10,4),  -- FCF/Revenue
        fcf_growth_3yr NUMERIC(10,4),  -- 3-year FCF CAGR
        fcf_consistency_score NUMERIC(10,4),  -- How stable is FCF generation (low volatility = better)
        fcf_to_net_income NUMERIC(10,4),  -- Quality of earnings (FCF/NI ratio)
        
        -- 3. Fundamentals Trend Ranking (momentum-based fundamentals)
        fundamentals_ranking INTEGER,  -- 1 = best fundamental trends
        
        -- Price trends
        price_trend_10yr NUMERIC(10,4),  -- 10-year price CAGR
        price_trend_5yr NUMERIC(10,4),  -- 5-year price CAGR  
        price_trend_1yr NUMERIC(10,4),  -- 1-year return
        price_momentum_score NUMERIC(10,4),  -- Weighted score (recent weighted higher)
        
        -- Revenue trends
        revenue_growth_10yr NUMERIC(10,4),  -- 10-year revenue CAGR
        revenue_growth_5yr NUMERIC(10,4),  -- 5-year revenue CAGR
        revenue_growth_1yr NUMERIC(10,4),  -- Most recent year growth
        revenue_trend TEXT CHECK (revenue_trend IN ('Accelerating', 'Stable Growth', 'Decelerating', 'Declining')),
        
        -- Profitability trends
        margin_trend_5yr TEXT CHECK (margin_trend_5yr IN ('Expanding', 'Stable', 'Contracting')),
        current_net_margin NUMERIC(10,4),
        margin_change_5yr NUMERIC(10,4),  -- Change in net margin over 5 years
        roe_current NUMERIC(10,4),
        roe_trend_5yr TEXT CHECK (roe_trend_5yr IN ('Improving', 'Stable', 'Declining')),
        
        -- Cash flow trends
        operating_cf_growth_5yr NUMERIC(10,4),
        cf_trend TEXT CHECK (cf_trend IN ('Growing', 'Stable', 'Volatile', 'Declining')),
        
        -- Composite Quality Score (for initial filtering)
        composite_quality_score NUMERIC(10,4),  -- Weighted average of all three rankings
        quality_tier TEXT CHECK (quality_tier IN ('Elite', 'Premium', 'Quality', 'Standard', 'Below')),
        
        -- Flags for strategy selection
        is_sp500_beater BOOLEAN DEFAULT FALSE,  -- Beat SP500 >50% of time
        is_cash_generator BOOLEAN DEFAULT FALSE,  -- Top quartile FCF yield
        is_fundamental_grower BOOLEAN DEFAULT FALSE,  -- Positive trends across metrics
        
        -- Additional context
        sector TEXT,
        industry TEXT,
        market_cap BIGINT,
        
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        
        PRIMARY KEY (symbol, ranking_date)
    );
    
    -- Detailed SP500 outperformance history (for calculating rankings)
    CREATE TABLE IF NOT EXISTS sp500_outperformance_detail (
        symbol TEXT NOT NULL,
        year INTEGER NOT NULL,
        
        symbol_return NUMERIC(10,4),
        sp500_return NUMERIC(10,4),
        excess_return NUMERIC(10,4),
        beat_sp500 BOOLEAN,
        
        -- For weighted scoring
        weight_factor NUMERIC(6,4),  -- e.g., 1.0 for current year, 0.95 for year-1, 0.90 for year-2, etc.
        weighted_score NUMERIC(10,4),  -- excess_return * weight_factor
        
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        
        PRIMARY KEY (symbol, year)
    );
    
    -- Indexes for performance
    CREATE INDEX IF NOT EXISTS idx_quality_rankings_symbol ON stock_quality_rankings(symbol);
    CREATE INDEX IF NOT EXISTS idx_quality_rankings_date ON stock_quality_rankings(ranking_date DESC);
    CREATE INDEX IF NOT EXISTS idx_quality_rankings_sp500 ON stock_quality_rankings(beat_sp500_ranking) WHERE beat_sp500_ranking <= 100;
    CREATE INDEX IF NOT EXISTS idx_quality_rankings_cashflow ON stock_quality_rankings(excess_cash_flow_ranking) WHERE excess_cash_flow_ranking <= 100;
    CREATE INDEX IF NOT EXISTS idx_quality_rankings_fundamentals ON stock_quality_rankings(fundamentals_ranking) WHERE fundamentals_ranking <= 100;
    CREATE INDEX IF NOT EXISTS idx_quality_rankings_composite ON stock_quality_rankings(composite_quality_score DESC);
    CREATE INDEX IF NOT EXISTS idx_quality_rankings_tier ON stock_quality_rankings(quality_tier, ranking_date DESC);
    
    -- Composite index for strategy filtering
    CREATE INDEX IF NOT EXISTS idx_quality_elite ON stock_quality_rankings(ranking_date DESC, beat_sp500_ranking, excess_cash_flow_ranking, fundamentals_ranking) 
    WHERE beat_sp500_ranking <= 100 AND excess_cash_flow_ranking <= 100 AND fundamentals_ranking <= 100;
    
    -- View for current top quality stocks
    CREATE OR REPLACE VIEW v_quality_stocks AS
    SELECT 
        q.*,
        s.name,
        s.exchange
    FROM stock_quality_rankings q
    JOIN symbol_universe s ON q.symbol = s.symbol
    WHERE q.ranking_date = (SELECT MAX(ranking_date) FROM stock_quality_rankings)
      AND q.quality_tier IN ('Elite', 'Premium', 'Quality')
    ORDER BY q.composite_quality_score DESC;
    
    -- View for Elite stocks (top in all three categories)
    CREATE OR REPLACE VIEW v_elite_stocks AS
    SELECT 
        symbol,
        beat_sp500_ranking,
        excess_cash_flow_ranking,
        fundamentals_ranking,
        composite_quality_score,
        years_beating_sp500,
        fcf_yield,
        price_momentum_score,
        revenue_trend,
        sector,
        market_cap
    FROM stock_quality_rankings
    WHERE ranking_date = (SELECT MAX(ranking_date) FROM stock_quality_rankings)
      AND beat_sp500_ranking <= 50
      AND excess_cash_flow_ranking <= 50
      AND fundamentals_ranking <= 50
    ORDER BY composite_quality_score DESC;
    
    -- View for strategy base selection (all qualified stocks)
    CREATE OR REPLACE VIEW v_strategy_universe AS
    SELECT 
        symbol,
        beat_sp500_ranking,
        excess_cash_flow_ranking,
        fundamentals_ranking,
        composite_quality_score,
        quality_tier,
        -- Key metrics for strategy filtering
        recent_5yr_beat_count,
        fcf_yield,
        fcf_growth_3yr,
        price_trend_5yr,
        revenue_growth_5yr,
        current_net_margin,
        roe_current,
        sector,
        market_cap
    FROM stock_quality_rankings
    WHERE ranking_date = (SELECT MAX(ranking_date) FROM stock_quality_rankings)
      AND (beat_sp500_ranking <= 200 
           OR excess_cash_flow_ranking <= 200 
           OR fundamentals_ranking <= 200)  -- At least one good ranking
    ORDER BY composite_quality_score DESC;
    """
    
    try:
        postgres_url = get_postgres_url()
        engine = create_engine(postgres_url)
        
        print("[INFO] Creating ACIS schema with performance optimizations...")
        
        with engine.begin() as conn:
            # Drop old views first if they exist
            conn.execute(text("""
                DROP MATERIALIZED VIEW IF EXISTS mv_latest_forward_returns CASCADE;
                DROP MATERIALIZED VIEW IF EXISTS mv_latest_fundamentals CASCADE;
                DROP MATERIALIZED VIEW IF EXISTS mv_latest_prices CASCADE;
                DROP MATERIALIZED VIEW IF EXISTS mv_active_options CASCADE;
            """))
            
            # Create new schema
            conn.execute(text(ESSENTIAL_SCHEMA_SQL))
        
        print("[SUCCESS] Schema created with performance indexes!")
        
        # Verify tables were created
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                  AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """))
            
            tables = [row[0] for row in result]
            print(f"\n[INFO] Created {len(tables)} tables:")
            for table in tables:
                print(f"  - {table}")
        
        # Verify indexes
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT indexname 
                FROM pg_indexes 
                WHERE schemaname = 'public'
                ORDER BY indexname
            """))
            
            indexes = [row[0] for row in result]
            print(f"\n[INFO] Created {len(indexes)} indexes for optimal performance")
        
        # Verify materialized views
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT matviewname 
                FROM pg_matviews 
                WHERE schemaname = 'public'
                ORDER BY matviewname
            """))
            
            views = [row[0] for row in result]
            print(f"\n[INFO] Created {len(views)} materialized views:")
            for view in views:
                print(f"  - {view}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Schema creation failed: {e}")
        return False

def refresh_materialized_views():
    """Refresh all materialized views for better performance."""
    try:
        postgres_url = get_postgres_url()
        engine = create_engine(postgres_url)
        
        print("\n[INFO] Refreshing materialized views...")
        
        with engine.begin() as conn:
            # Check if views have data before refreshing
            result = conn.execute(text("""
                SELECT matviewname 
                FROM pg_matviews 
                WHERE schemaname = 'public'
            """))
            
            views = [row[0] for row in result]
            
            for view in views:
                try:
                    conn.execute(text(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {view}"))
                    print(f"  - Refreshed {view}")
                except:
                    # Try without CONCURRENTLY if unique index doesn't exist yet
                    conn.execute(text(f"REFRESH MATERIALIZED VIEW {view}"))
                    print(f"  - Refreshed {view} (non-concurrent)")
        
        return True
    except Exception as e:
        print(f"[WARNING] Could not refresh views (may be empty): {e}")
        return True  # Don't fail on view refresh

def main():
    """Main execution function."""
    start_time = time.time()
    
    print("\n" + "=" * 60)
    print("ACIS DATABASE SCHEMA SETUP")
    print("=" * 60)
    
    try:
        success = create_essential_schema()
        
        if success:
            # Try to refresh views if there's data
            refresh_materialized_views()
            
            duration = time.time() - start_time
            print("\n" + "=" * 60)
            print(f"[SUCCESS] Schema setup completed in {duration:.1f} seconds")
            print("=" * 60)
            print("\nDatabase is ready for:")
            print("  - fetch_symbol_universe.py")
            print("  - fetch_prices.py") 
            print("  - fetch_technical_indicators.py")
            print("  - fetch_fundamentals.py")
            print("  - fetch_options.py")
            print("  - run_daily_pipeline.py")
        else:
            print("\n[ERROR] Schema setup failed!")
            sys.exit(1)
        
    except Exception as e:
        print(f"\n[ERROR] Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()