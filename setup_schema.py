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
    
    -- 9. ML Forward Returns (for ML model training with risk metrics)
    CREATE TABLE IF NOT EXISTS ml_forward_returns (
        symbol TEXT NOT NULL,
        ranking_date DATE NOT NULL,
        horizon_weeks INTEGER NOT NULL,
        
        -- Return metrics
        forward_return NUMERIC,
        forward_excess_return NUMERIC,  -- vs SP500
        
        -- Risk metrics
        forward_volatility NUMERIC,
        forward_max_drawdown NUMERIC,
        
        -- Additional ML features (optional)
        forward_sharpe_ratio NUMERIC,
        forward_win_rate NUMERIC,  -- %% of positive days
        forward_skewness NUMERIC,
        forward_kurtosis NUMERIC,
        
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        
        PRIMARY KEY (symbol, ranking_date, horizon_weeks)
    );
    
    CREATE INDEX IF NOT EXISTS idx_ml_forward_returns_symbol ON ml_forward_returns(symbol);
    CREATE INDEX IF NOT EXISTS idx_ml_forward_returns_date ON ml_forward_returns(ranking_date DESC);
    CREATE INDEX IF NOT EXISTS idx_ml_forward_returns_horizon ON ml_forward_returns(horizon_weeks);
    CREATE INDEX IF NOT EXISTS idx_ml_forward_returns_symbol_date ON ml_forward_returns(symbol, ranking_date DESC);
    
    -- 10. Ranking Transitions (for ML model training)
    CREATE TABLE IF NOT EXISTS ranking_transitions (
        symbol TEXT NOT NULL,
        from_date DATE NOT NULL,
        to_date DATE NOT NULL,
        horizon_weeks INTEGER NOT NULL,
        
        -- Ranking changes
        from_rank INTEGER,
        to_rank INTEGER,
        rank_change INTEGER,
        
        -- Quality score changes
        from_quality_score NUMERIC,
        to_quality_score NUMERIC,
        quality_score_change NUMERIC,
        
        -- Actual return during period
        actual_return NUMERIC,
        excess_return NUMERIC,
        
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        
        PRIMARY KEY (symbol, from_date, horizon_weeks)
    );
    
    CREATE INDEX IF NOT EXISTS idx_ranking_transitions_symbol ON ranking_transitions(symbol);
    CREATE INDEX IF NOT EXISTS idx_ranking_transitions_from_date ON ranking_transitions(from_date DESC);
    
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
        
        -- 4. News Sentiment Ranking
        sentiment_ranking INTEGER,  -- 1 = best sentiment
        sentiment_score NUMERIC(10,4),  -- Composite sentiment score (-100 to +100)
        sentiment_7d NUMERIC(10,4),  -- 7-day average sentiment
        sentiment_30d NUMERIC(10,4),  -- 30-day average sentiment
        sentiment_momentum NUMERIC(10,4),  -- Change in sentiment (7d vs 30d)
        bull_bear_ratio NUMERIC(10,4),  -- Ratio of bullish to bearish articles
        article_count INTEGER,  -- Total articles in period
        sentiment_confidence TEXT CHECK (sentiment_confidence IN ('Very High', 'High', 'Medium', 'Low', 'Very Low')),
        positive_catalysts TEXT,  -- JSON array of positive catalyst topics
        sentiment_volatility NUMERIC(10,4),  -- Standard deviation of sentiment scores
        days_with_coverage INTEGER,  -- Number of days with news coverage
        
        -- Sentiment flags
        is_sentiment_positive BOOLEAN DEFAULT FALSE,  -- Overall positive sentiment
        is_momentum_star BOOLEAN DEFAULT FALSE,  -- Strong sentiment momentum
        has_catalyst_events BOOLEAN DEFAULT FALSE,  -- Recent catalyst news
        
        -- 5. Value Ranking (Historical Extremes)
        value_ranking INTEGER,  -- 1 = best value (most undervalued historically)
        value_score NUMERIC(10,4),  -- Composite value score based on historical extremes
        
        -- Price multiples vs historical ranges
        price_to_sales_current NUMERIC(10,4),  -- Current P/S ratio
        price_to_sales_percentile NUMERIC(10,4),  -- Percentile in 10-year history (0=lowest ever, 100=highest)
        price_to_sales_zscore NUMERIC(10,4),  -- Standard deviations from historical mean
        
        price_to_book_current NUMERIC(10,4),  -- Current P/B ratio
        price_to_book_percentile NUMERIC(10,4),  -- Percentile in 10-year history
        price_to_book_zscore NUMERIC(10,4),  -- Standard deviations from historical mean
        
        price_to_cashflow_current NUMERIC(10,4),  -- Current P/CF ratio
        price_to_cashflow_percentile NUMERIC(10,4),  -- Percentile in 10-year history
        price_to_cashflow_zscore NUMERIC(10,4),  -- Standard deviations from historical mean
        
        pe_ratio_current NUMERIC(10,4),  -- Current P/E ratio
        pe_ratio_percentile NUMERIC(10,4),  -- Percentile in 10-year history
        pe_ratio_zscore NUMERIC(10,4),  -- Standard deviations from historical mean
        
        ev_to_ebitda_current NUMERIC(10,4),  -- Current EV/EBITDA
        ev_to_ebitda_percentile NUMERIC(10,4),  -- Percentile in 10-year history
        ev_to_ebitda_zscore NUMERIC(10,4),  -- Standard deviations from historical mean
        
        -- Yield metrics
        dividend_yield_current NUMERIC(10,4),  -- Current dividend yield
        dividend_yield_percentile NUMERIC(10,4),  -- Percentile (inverted - high yield = low percentile = good)
        fcf_yield_percentile NUMERIC(10,4),  -- Free cash flow yield percentile
        earnings_yield_percentile NUMERIC(10,4),  -- Earnings yield percentile
        
        -- Relative value metrics
        sector_relative_value NUMERIC(10,4),  -- Value vs sector peers (-100 to +100, negative = cheap)
        market_relative_value NUMERIC(10,4),  -- Value vs overall market
        historical_discount NUMERIC(10,4),  -- Average discount to historical median (negative = cheap)
        
        -- Value confidence and flags
        value_confidence TEXT CHECK (value_confidence IN ('Very High', 'High', 'Medium', 'Low', 'Very Low')),
        years_of_history INTEGER,  -- How many years of data for percentile calculation
        is_deep_value BOOLEAN DEFAULT FALSE,  -- Multiple metrics in bottom quartile
        is_value_trap_risk BOOLEAN DEFAULT FALSE,  -- Cheap but deteriorating fundamentals
        is_historically_cheap BOOLEAN DEFAULT FALSE,  -- Bottom 20%% of historical range
        
        -- 6. Breakout Ranking (Technical Momentum)
        breakout_ranking INTEGER,  -- 1 = strongest breakout
        breakout_score NUMERIC(10,4),  -- Composite breakout score
        
        -- Price breakout metrics (quarterly)
        price_change_3m NUMERIC(10,4),  -- 3-month price change %%
        price_change_1m NUMERIC(10,4),  -- 1-month price change %%
        price_change_1w NUMERIC(10,4),  -- 1-week price change %%
        
        -- Breakout characteristics
        breakout_date DATE,  -- Date of most recent breakout
        breakout_type TEXT,  -- Type: '52w_high', 'resistance', 'range', 'base', 'gap_up'
        breakout_magnitude NUMERIC(10,4),  -- Size of breakout move %%
        days_since_breakout INTEGER,  -- Days since breakout occurred
        
        -- Price levels and patterns
        current_price NUMERIC(12,4),
        resistance_level NUMERIC(12,4),  -- Key resistance that was broken
        support_level NUMERIC(12,4),  -- Current support level
        price_vs_52w_high NUMERIC(10,4),  -- %% below 52-week high
        price_vs_52w_low NUMERIC(10,4),  -- %% above 52-week low
        new_highs_count INTEGER,  -- Number of new highs in quarter
        
        -- Volume analysis
        volume_change_3m NUMERIC(10,4),  -- 3-month volume change %%
        volume_change_1m NUMERIC(10,4),  -- 1-month volume change %%
        volume_surge_days INTEGER,  -- Days with >2x average volume
        avg_volume_3m BIGINT,  -- Average daily volume (3 months)
        avg_volume_prior BIGINT,  -- Average volume before breakout
        volume_ratio NUMERIC(10,4),  -- Current vs historical volume ratio
        
        -- Volume-price relationship
        volume_price_correlation NUMERIC(10,4),  -- Correlation between price and volume
        accumulation_distribution NUMERIC(10,4),  -- A/D line trend
        on_balance_volume_trend NUMERIC(10,4),  -- OBV trend strength
        volume_weighted_price NUMERIC(12,4),  -- VWAP
        
        -- Relative strength
        rs_rating NUMERIC(10,4),  -- Relative strength vs market (0-100)
        rs_vs_sector NUMERIC(10,4),  -- Relative strength vs sector
        outperformance_days INTEGER,  -- Days outperforming market in quarter
        
        -- Breakout quality metrics
        breakout_confidence TEXT CHECK (breakout_confidence IN ('Very High', 'High', 'Medium', 'Low', 'Very Low')),
        consolidation_weeks INTEGER,  -- Weeks of consolidation before breakout
        volatility_contraction NUMERIC(10,4),  -- Volatility decrease before breakout
        follow_through_score NUMERIC(10,4),  -- Post-breakout continuation strength
        
        -- Breakout flags
        is_valid_breakout BOOLEAN DEFAULT FALSE,  -- Meets all breakout criteria
        is_volume_confirmed BOOLEAN DEFAULT FALSE,  -- Volume confirms breakout
        is_52w_high BOOLEAN DEFAULT FALSE,  -- At or near 52-week high
        is_sector_leader BOOLEAN DEFAULT FALSE,  -- Leading sector performance
        is_breakout_star BOOLEAN DEFAULT FALSE,  -- Top breakout candidate
        
        -- 7. Growth Ranking (Long-term Consistent Growth)
        growth_ranking INTEGER,  -- 1 = strongest long-term growth
        growth_score NUMERIC(10,4),  -- Composite growth score
        
        -- Lifetime performance metrics
        total_years_tracked INTEGER,  -- Years of price history available
        lifetime_return NUMERIC(12,4),  -- Total return since inception
        lifetime_annualized_return NUMERIC(10,4),  -- CAGR since inception
        lifetime_sp500_excess NUMERIC(10,4),  -- Lifetime excess return vs SP500
        years_outperforming_sp500 INTEGER,  -- Number of years beating SP500
        outperformance_consistency NUMERIC(10,4),  -- %% of years beating SP500
        
        -- Long-term growth trends
        growth_10yr_cagr NUMERIC(10,4),  -- 10-year compound annual growth rate
        growth_5yr_cagr NUMERIC(10,4),  -- 5-year CAGR
        growth_3yr_cagr NUMERIC(10,4),  -- 3-year CAGR
        growth_acceleration NUMERIC(10,4),  -- Growth rate change (5yr vs 10yr)
        growth_stability NUMERIC(10,4),  -- Consistency of growth (inverse of volatility)
        growth_quality_score NUMERIC(10,4),  -- Quality-adjusted growth score
        
        -- Revenue and earnings growth
        revenue_growth_10yr NUMERIC(10,4),  -- 10-year revenue CAGR
        revenue_growth_consistency NUMERIC(10,4),  -- Std dev of revenue growth
        earnings_growth_10yr NUMERIC(10,4),  -- 10-year earnings CAGR
        earnings_growth_consistency NUMERIC(10,4),  -- Std dev of earnings growth
        fcf_growth_10yr NUMERIC(10,4),  -- 10-year FCF CAGR
        book_value_growth_10yr NUMERIC(10,4),  -- 10-year book value CAGR
        
        -- Long-term trend analysis
        trend_strength_10yr NUMERIC(10,4),  -- R-squared of 10-year trend
        trend_strength_5yr NUMERIC(10,4),  -- R-squared of 5-year trend
        drawdown_recovery_avg NUMERIC(10,4),  -- Avg months to recover from drawdown
        max_drawdown_10yr NUMERIC(10,4),  -- Maximum drawdown in 10 years
        sharpe_ratio_10yr NUMERIC(10,4),  -- Risk-adjusted returns
        sortino_ratio_10yr NUMERIC(10,4),  -- Downside risk-adjusted returns
        
        -- Growth persistence metrics
        consecutive_growth_years INTEGER,  -- Years of consecutive positive growth
        consecutive_beat_years INTEGER,  -- Years consecutively beating SP500
        growth_momentum_score NUMERIC(10,4),  -- Recent vs long-term growth
        mean_reversion_risk NUMERIC(10,4),  -- Risk of growth reverting to mean
        
        -- Relative growth metrics
        sector_growth_percentile NUMERIC(10,4),  -- Growth rank within sector
        market_growth_percentile NUMERIC(10,4),  -- Growth rank within market
        size_adjusted_growth NUMERIC(10,4),  -- Growth adjusted for market cap
        
        -- Growth confidence and quality
        growth_confidence TEXT CHECK (growth_confidence IN ('Very High', 'High', 'Medium', 'Low', 'Very Low')),
        growth_data_quality NUMERIC(10,4),  -- Data completeness score
        is_consistent_grower BOOLEAN DEFAULT FALSE,  -- Steady growth pattern
        is_accelerating_growth BOOLEAN DEFAULT FALSE,  -- Accelerating growth rate
        is_compound_winner BOOLEAN DEFAULT FALSE,  -- Top long-term compounder
        is_lifetime_outperformer BOOLEAN DEFAULT FALSE,  -- Consistently beats SP500
        
        -- Composite Quality Score (for initial filtering)
        base_composite_score NUMERIC(10,4),  -- Raw weighted average of all 7 rankings
        composite_quality_score NUMERIC(10,4),  -- Confidence-adjusted composite score
        sector_neutral_score NUMERIC(10,4),  -- Sector-adjusted composite score
        size_adjusted_score NUMERIC(10,4),  -- Market cap-adjusted score
        final_composite_score NUMERIC(10,4),  -- Final blended composite score
        overall_data_confidence NUMERIC(10,4),  -- Overall data quality score
        quality_tier TEXT CHECK (quality_tier IN ('Elite', 'Premium', 'Quality', 'Standard', 'Below')),
        confidence_tier TEXT CHECK (confidence_tier IN ('Very High', 'High', 'Medium', 'Low', 'Very Low')),
        
        -- Flags for strategy selection
        is_sp500_beater BOOLEAN DEFAULT FALSE,  -- Beat SP500 >50%% of time
        is_cash_generator BOOLEAN DEFAULT FALSE,  -- Top quartile FCF yield
        is_fundamental_grower BOOLEAN DEFAULT FALSE,  -- Positive trends across metrics
        is_deep_value_star BOOLEAN DEFAULT FALSE,  -- Top value stocks at historical extremes
        is_momentum_breakout BOOLEAN DEFAULT FALSE,  -- Strong breakout with volume confirmation
        is_growth_champion BOOLEAN DEFAULT FALSE,  -- Top long-term growth stock
        is_all_star BOOLEAN DEFAULT FALSE,  -- Elite across multiple rankings
        is_momentum_star BOOLEAN DEFAULT FALSE,  -- Top momentum/sentiment combination
        
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
    CREATE INDEX IF NOT EXISTS idx_quality_rankings_sentiment ON stock_quality_rankings(sentiment_ranking) WHERE sentiment_ranking <= 100;
    CREATE INDEX IF NOT EXISTS idx_quality_rankings_value ON stock_quality_rankings(value_ranking) WHERE value_ranking <= 100;
    CREATE INDEX IF NOT EXISTS idx_quality_rankings_breakout ON stock_quality_rankings(breakout_ranking) WHERE breakout_ranking <= 100;
    CREATE INDEX IF NOT EXISTS idx_quality_rankings_composite ON stock_quality_rankings(composite_quality_score DESC);
    CREATE INDEX IF NOT EXISTS idx_quality_rankings_tier ON stock_quality_rankings(quality_tier, ranking_date DESC);
    
    -- Composite index for strategy filtering
    CREATE INDEX IF NOT EXISTS idx_quality_elite ON stock_quality_rankings(
        ranking_date DESC, 
        beat_sp500_ranking, 
        excess_cash_flow_ranking, 
        fundamentals_ranking, 
        sentiment_ranking, 
        value_ranking,
        breakout_ranking,
        growth_ranking
    ) 
    WHERE beat_sp500_ranking <= 100 
      AND excess_cash_flow_ranking <= 100 
      AND fundamentals_ranking <= 100 
      AND sentiment_ranking <= 100 
      AND value_ranking <= 100
      AND breakout_ranking <= 100
      AND growth_ranking <= 100;
    
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
    
    -- =============================================
    -- NEWS SENTIMENT TABLES
    -- =============================================
    
    -- Main news articles table
    CREATE TABLE IF NOT EXISTS news_articles (
        article_id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        summary TEXT,
        source TEXT,
        category TEXT,
        published_at TIMESTAMP NOT NULL,
        url TEXT,
        overall_sentiment_score NUMERIC,
        overall_sentiment_label TEXT,
        fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        
        CONSTRAINT unique_article UNIQUE (article_id)
    );
    
    CREATE INDEX IF NOT EXISTS idx_news_published ON news_articles(published_at DESC);
    CREATE INDEX IF NOT EXISTS idx_news_sentiment ON news_articles(overall_sentiment_score);
    
    -- Symbol-specific sentiment
    CREATE TABLE IF NOT EXISTS news_sentiment_by_symbol (
        id SERIAL PRIMARY KEY,
        article_id TEXT REFERENCES news_articles(article_id),
        symbol TEXT NOT NULL,
        relevance_score NUMERIC,
        sentiment_score NUMERIC,
        sentiment_label TEXT,
        
        CONSTRAINT unique_article_symbol UNIQUE (article_id, symbol)
    );
    
    CREATE INDEX IF NOT EXISTS idx_sentiment_symbol ON news_sentiment_by_symbol(symbol);
    CREATE INDEX IF NOT EXISTS idx_sentiment_date ON news_sentiment_by_symbol(symbol, article_id);
    
    -- Article topics
    CREATE TABLE IF NOT EXISTS news_topics (
        id SERIAL PRIMARY KEY,
        article_id TEXT REFERENCES news_articles(article_id),
        topic TEXT NOT NULL,
        relevance_score NUMERIC
    );
    
    CREATE INDEX IF NOT EXISTS idx_topics_article ON news_topics(article_id);
    CREATE INDEX IF NOT EXISTS idx_topics_topic ON news_topics(topic);
    
    -- Aggregated daily sentiment
    CREATE TABLE IF NOT EXISTS daily_sentiment_summary (
        symbol TEXT NOT NULL,
        date DATE NOT NULL,
        avg_sentiment_score NUMERIC,
        weighted_sentiment_score NUMERIC,  -- Weighted by relevance
        bullish_count INTEGER DEFAULT 0,
        bearish_count INTEGER DEFAULT 0,
        neutral_count INTEGER DEFAULT 0,
        total_articles INTEGER DEFAULT 0,
        avg_relevance_score NUMERIC,
        sentiment_momentum NUMERIC,  -- Change vs previous period
        sentiment_volatility NUMERIC,  -- Std dev of sentiment
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        
        PRIMARY KEY (symbol, date)
    );
    
    CREATE INDEX IF NOT EXISTS idx_daily_sentiment ON daily_sentiment_summary(symbol, date DESC);
    
    -- View for latest sentiment
    CREATE OR REPLACE VIEW v_latest_sentiment AS
    SELECT 
        s.symbol,
        s.date,
        s.weighted_sentiment_score,
        s.sentiment_momentum,
        s.bullish_count,
        s.bearish_count,
        s.total_articles,
        CASE 
            WHEN s.weighted_sentiment_score > 0.15 THEN 'Bullish'
            WHEN s.weighted_sentiment_score < -0.15 THEN 'Bearish'
            ELSE 'Neutral'
        END as sentiment_category
    FROM daily_sentiment_summary s
    WHERE s.date = (
        SELECT MAX(date) 
        FROM daily_sentiment_summary 
        WHERE symbol = s.symbol
    );
    
    -- View for sentiment trends
    CREATE OR REPLACE VIEW v_sentiment_trends AS
    SELECT 
        symbol,
        AVG(CASE WHEN date > CURRENT_DATE - INTERVAL '7 days' 
            THEN weighted_sentiment_score END) as sentiment_7d,
        AVG(CASE WHEN date > CURRENT_DATE - INTERVAL '30 days' 
            THEN weighted_sentiment_score END) as sentiment_30d,
        AVG(CASE WHEN date > CURRENT_DATE - INTERVAL '7 days' 
            THEN sentiment_momentum END) as momentum_7d,
        SUM(CASE WHEN date > CURRENT_DATE - INTERVAL '7 days' 
            THEN total_articles END) as articles_7d,
        SUM(CASE WHEN date > CURRENT_DATE - INTERVAL '30 days' 
            THEN total_articles END) as articles_30d
    FROM daily_sentiment_summary
    WHERE date > CURRENT_DATE - INTERVAL '30 days'
    GROUP BY symbol;
    
    -- =============================================
    -- MACHINE LEARNING SCHEMA
    -- =============================================
    
    -- ML Model Registry
    CREATE TABLE IF NOT EXISTS ml_models (
        model_id SERIAL PRIMARY KEY,
        model_name TEXT NOT NULL,
        model_type TEXT NOT NULL CHECK (model_type IN ('regression', 'classification', 'ensemble', 'deep_learning')),
        version TEXT NOT NULL,
        algorithm TEXT NOT NULL,  -- e.g., 'XGBoost', 'LSTM', 'RandomForest'
        
        -- Model metadata
        target_variable TEXT NOT NULL,  -- e.g., 'return_1m', 'outperform_sp500'
        feature_set TEXT NOT NULL,  -- JSON array of feature names
        hyperparameters TEXT,  -- JSON object of hyperparameters
        
        -- Training info
        training_date TIMESTAMP NOT NULL,
        training_start_date DATE,
        training_end_date DATE,
        validation_start_date DATE,
        validation_end_date DATE,
        
        -- Performance metrics
        train_score NUMERIC,  -- R2 or accuracy on training set
        validation_score NUMERIC,  -- R2 or accuracy on validation set
        test_score NUMERIC,  -- R2 or accuracy on test set
        metrics TEXT,  -- JSON object with detailed metrics (MAE, RMSE, precision, recall, etc.)
        
        -- Feature importance
        feature_importance TEXT,  -- JSON object mapping features to importance scores
        
        -- Model storage
        model_path TEXT,  -- Path to serialized model file
        model_binary BYTEA,  -- Optional: store model binary in DB
        
        -- Status and deployment
        is_active BOOLEAN DEFAULT FALSE,
        is_deployed BOOLEAN DEFAULT FALSE,
        deployment_date TIMESTAMP,
        last_prediction_date TIMESTAMP,
        
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        
        CONSTRAINT unique_model_version UNIQUE (model_name, version)
    );
    
    CREATE INDEX IF NOT EXISTS idx_ml_models_active ON ml_models(is_active) WHERE is_active = TRUE;
    CREATE INDEX IF NOT EXISTS idx_ml_models_type ON ml_models(model_type, target_variable);
    
    -- ML Predictions
    CREATE TABLE IF NOT EXISTS ml_predictions (
        prediction_id SERIAL PRIMARY KEY,
        model_id INTEGER REFERENCES ml_models(model_id),
        symbol TEXT NOT NULL,
        prediction_date DATE NOT NULL,
        
        -- Predictions for different horizons
        predicted_return_1m NUMERIC,
        predicted_return_3m NUMERIC,
        predicted_return_6m NUMERIC,
        predicted_return_12m NUMERIC,
        
        -- Classification predictions
        predicted_outperform BOOLEAN,
        outperform_probability NUMERIC,
        
        -- Prediction confidence/uncertainty
        prediction_confidence NUMERIC,
        prediction_std NUMERIC,  -- Standard deviation for uncertainty
        
        -- Input features snapshot (for reproducibility)
        feature_values TEXT,  -- JSON object of feature values at prediction time
        
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        
        CONSTRAINT unique_model_symbol_date UNIQUE (model_id, symbol, prediction_date)
    );
    
    CREATE INDEX IF NOT EXISTS idx_ml_predictions_symbol_date ON ml_predictions(symbol, prediction_date DESC);
    CREATE INDEX IF NOT EXISTS idx_ml_predictions_model ON ml_predictions(model_id, prediction_date DESC);
    
    -- Strategy Signals (ML-generated trading signals)
    CREATE TABLE IF NOT EXISTS strategy_signals (
        signal_id SERIAL PRIMARY KEY,
        model_id INTEGER REFERENCES ml_models(model_id),
        symbol TEXT NOT NULL,
        signal_date DATE NOT NULL,
        
        -- Signal details
        signal_type TEXT NOT NULL CHECK (signal_type IN ('BUY', 'SELL', 'HOLD', 'STRONG_BUY', 'STRONG_SELL')),
        signal_strength NUMERIC,  -- 0 to 1 or -1 to 1
        
        -- Position sizing
        recommended_allocation NUMERIC,  -- Percentage of portfolio
        position_size_shares INTEGER,
        
        -- Risk metrics
        stop_loss_price NUMERIC,
        take_profit_price NUMERIC,
        risk_reward_ratio NUMERIC,
        
        -- Supporting metrics
        expected_return NUMERIC,
        confidence_score NUMERIC,
        volatility_forecast NUMERIC,
        
        -- Execution tracking
        is_executed BOOLEAN DEFAULT FALSE,
        execution_date TIMESTAMP,
        execution_price NUMERIC,
        
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        
        CONSTRAINT unique_signal UNIQUE (model_id, symbol, signal_date)
    );
    
    CREATE INDEX IF NOT EXISTS idx_strategy_signals_date ON strategy_signals(signal_date DESC);
    CREATE INDEX IF NOT EXISTS idx_strategy_signals_symbol ON strategy_signals(symbol, signal_date DESC);
    CREATE INDEX IF NOT EXISTS idx_strategy_signals_pending ON strategy_signals(is_executed) WHERE is_executed = FALSE;
    
    -- Portfolio Performance Tracking
    CREATE TABLE IF NOT EXISTS portfolio_performance (
        performance_id SERIAL PRIMARY KEY,
        strategy_name TEXT NOT NULL,
        model_id INTEGER REFERENCES ml_models(model_id),
        date DATE NOT NULL,
        
        -- Portfolio value
        portfolio_value NUMERIC NOT NULL,
        cash_balance NUMERIC,
        invested_value NUMERIC,
        
        -- Returns
        daily_return NUMERIC,
        cumulative_return NUMERIC,
        ytd_return NUMERIC,
        
        -- Benchmark comparison
        sp500_return NUMERIC,
        excess_return NUMERIC,
        
        -- Risk metrics
        volatility NUMERIC,
        sharpe_ratio NUMERIC,
        sortino_ratio NUMERIC,
        max_drawdown NUMERIC,
        current_drawdown NUMERIC,
        
        -- Position metrics
        num_positions INTEGER,
        long_positions INTEGER,
        short_positions INTEGER,
        win_rate NUMERIC,
        
        -- Transaction metrics
        trades_today INTEGER,
        turnover_rate NUMERIC,
        
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        
        CONSTRAINT unique_portfolio_date UNIQUE (strategy_name, date)
    );
    
    CREATE INDEX IF NOT EXISTS idx_portfolio_performance_strategy ON portfolio_performance(strategy_name, date DESC);
    CREATE INDEX IF NOT EXISTS idx_portfolio_performance_date ON portfolio_performance(date DESC);
    
    -- Backtesting Results
    CREATE TABLE IF NOT EXISTS backtest_results (
        backtest_id SERIAL PRIMARY KEY,
        model_id INTEGER REFERENCES ml_models(model_id),
        strategy_name TEXT NOT NULL,
        
        -- Backtest period
        start_date DATE NOT NULL,
        end_date DATE NOT NULL,
        
        -- Configuration
        initial_capital NUMERIC NOT NULL,
        position_sizing TEXT,  -- JSON config for position sizing rules
        risk_management TEXT,  -- JSON config for risk management rules
        transaction_costs NUMERIC,
        slippage NUMERIC,
        
        -- Performance metrics
        total_return NUMERIC,
        annual_return NUMERIC,
        volatility NUMERIC,
        sharpe_ratio NUMERIC,
        sortino_ratio NUMERIC,
        calmar_ratio NUMERIC,
        
        -- Risk metrics
        max_drawdown NUMERIC,
        var_95 NUMERIC,  -- Value at Risk at 95%% confidence
        cvar_95 NUMERIC,  -- Conditional VaR at 95%% confidence
        
        -- Trading metrics
        total_trades INTEGER,
        winning_trades INTEGER,
        losing_trades INTEGER,
        win_rate NUMERIC,
        avg_win NUMERIC,
        avg_loss NUMERIC,
        profit_factor NUMERIC,
        
        -- Benchmark comparison
        benchmark_return NUMERIC,
        alpha NUMERIC,
        beta NUMERIC,
        information_ratio NUMERIC,
        
        -- Detailed results
        equity_curve TEXT,  -- JSON array of portfolio values over time
        trade_log TEXT,  -- JSON array of all trades
        
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        
        CONSTRAINT unique_backtest UNIQUE (model_id, strategy_name, start_date, end_date)
    );
    
    CREATE INDEX IF NOT EXISTS idx_backtest_results_model ON backtest_results(model_id);
    CREATE INDEX IF NOT EXISTS idx_backtest_results_performance ON backtest_results(sharpe_ratio DESC, total_return DESC);
    
    -- Feature Engineering Store
    CREATE TABLE IF NOT EXISTS ml_features (
        symbol TEXT NOT NULL,
        date DATE NOT NULL,
        
        -- Price-based features
        price_momentum_1m NUMERIC,
        price_momentum_3m NUMERIC,
        price_momentum_6m NUMERIC,
        price_momentum_12m NUMERIC,
        
        -- Volatility features
        volatility_20d NUMERIC,
        volatility_60d NUMERIC,
        volatility_ratio NUMERIC,
        
        -- Volume features
        volume_ratio_20d NUMERIC,
        volume_trend NUMERIC,
        
        -- Technical features
        rsi_divergence NUMERIC,
        macd_signal_strength NUMERIC,
        bollinger_position NUMERIC,
        
        -- Fundamental features
        pe_percentile NUMERIC,
        fcf_yield_percentile NUMERIC,
        quality_score NUMERIC,
        
        -- Market regime features
        market_regime TEXT,
        sector_momentum NUMERIC,
        correlation_to_market NUMERIC,
        
        -- Sentiment features
        sentiment_score NUMERIC,
        sentiment_momentum NUMERIC,
        news_volume NUMERIC,
        
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        
        PRIMARY KEY (symbol, date)
    );
    
    CREATE INDEX IF NOT EXISTS idx_ml_features_date ON ml_features(date DESC);
    CREATE INDEX IF NOT EXISTS idx_ml_features_symbol ON ml_features(symbol, date DESC);
    
    -- Model Training Queue
    CREATE TABLE IF NOT EXISTS model_training_queue (
        queue_id SERIAL PRIMARY KEY,
        model_name TEXT NOT NULL,
        model_type TEXT NOT NULL,
        priority INTEGER DEFAULT 5,
        
        -- Training configuration
        config TEXT NOT NULL,  -- JSON configuration for training
        
        -- Status tracking
        status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
        started_at TIMESTAMP,
        completed_at TIMESTAMP,
        error_message TEXT,
        
        -- Results
        result_model_id INTEGER REFERENCES ml_models(model_id),
        
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_training_queue_status ON model_training_queue(status, priority DESC) WHERE status IN ('pending', 'running');
    
    -- Views for ML monitoring
    CREATE OR REPLACE VIEW v_active_ml_models AS
    SELECT 
        m.model_id,
        m.model_name,
        m.model_type,
        m.version,
        m.validation_score,
        m.last_prediction_date,
        COUNT(DISTINCT p.symbol) as symbols_predicted,
        AVG(p.prediction_confidence) as avg_confidence
    FROM ml_models m
    LEFT JOIN ml_predictions p ON m.model_id = p.model_id
        AND p.prediction_date >= CURRENT_DATE - INTERVAL '30 days'
    WHERE m.is_active = TRUE
    GROUP BY m.model_id, m.model_name, m.model_type, m.version, m.validation_score, m.last_prediction_date;
    
    CREATE OR REPLACE VIEW v_latest_signals AS
    SELECT 
        s.symbol,
        s.signal_date,
        s.signal_type,
        s.signal_strength,
        s.confidence_score,
        s.expected_return,
        m.model_name,
        m.version
    FROM strategy_signals s
    JOIN ml_models m ON s.model_id = m.model_id
    WHERE s.signal_date = (
        SELECT MAX(signal_date) 
        FROM strategy_signals 
        WHERE symbol = s.symbol
    )
    AND s.is_executed = FALSE;
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