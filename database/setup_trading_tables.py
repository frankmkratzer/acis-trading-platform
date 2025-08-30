#!/usr/bin/env python3
"""
ACIS Automated Trading Tables
Database schema for client management and automated trading with Schwab API
"""

import os
import sys
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

def create_trading_tables():
    """Create automated trading and client management tables"""
    
    SQL = """
    -- ================================================================
    -- AUTOMATED TRADING TABLES FOR SCHWAB API INTEGRATION
    -- ================================================================
    
    -- Clients table (Core client information)
    CREATE TABLE IF NOT EXISTS clients (
        client_id SERIAL PRIMARY KEY,
        -- Personal Information
        first_name VARCHAR(100) NOT NULL,
        last_name VARCHAR(100) NOT NULL,
        email VARCHAR(255) UNIQUE NOT NULL,
        phone VARCHAR(20),
        
        -- Account Preferences
        risk_tolerance VARCHAR(20) CHECK (risk_tolerance IN ('conservative', 'moderate', 'aggressive')),
        investment_horizon_years INTEGER,
        
        -- Trading Preferences
        automated_trading_enabled BOOLEAN DEFAULT FALSE,
        portfolio_strategy VARCHAR(20) CHECK (portfolio_strategy IN ('VALUE', 'GROWTH', 'DIVIDEND', 'BALANCED')),
        max_position_size_pct NUMERIC(5, 2) DEFAULT 10.00,  -- Max % per position
        max_portfolio_positions INTEGER DEFAULT 20,          -- Max number of holdings
        
        -- Notifications
        email_notifications BOOLEAN DEFAULT TRUE,
        sms_notifications BOOLEAN DEFAULT FALSE,
        
        -- Status
        account_status VARCHAR(20) DEFAULT 'PENDING' CHECK (account_status IN ('PENDING', 'ACTIVE', 'SUSPENDED', 'CLOSED')),
        onboarding_completed BOOLEAN DEFAULT FALSE,
        terms_accepted_date TIMESTAMP,
        
        -- Metadata
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login_at TIMESTAMP,
        
        -- Compliance
        kyc_verified BOOLEAN DEFAULT FALSE,
        kyc_verified_date TIMESTAMP,
        accredited_investor BOOLEAN DEFAULT FALSE
    );
    
    -- Trading Accounts (Schwab account credentials and settings)
    CREATE TABLE IF NOT EXISTS trading_accounts (
        account_id SERIAL PRIMARY KEY,
        client_id INTEGER NOT NULL REFERENCES clients(client_id) ON DELETE CASCADE,
        
        -- Schwab Account Information
        schwab_account_number VARCHAR(50) UNIQUE NOT NULL,
        schwab_account_hash VARCHAR(255) UNIQUE NOT NULL,  -- Encrypted account identifier
        account_type VARCHAR(20) CHECK (account_type IN ('INDIVIDUAL', 'IRA', 'ROTH_IRA', 'JOINT')),
        
        -- OAuth Credentials (Encrypted)
        refresh_token TEXT,  -- Encrypted Schwab OAuth refresh token
        access_token TEXT,   -- Encrypted Schwab OAuth access token
        token_expires_at TIMESTAMP,
        
        -- Account Settings
        is_primary BOOLEAN DEFAULT FALSE,
        automated_trading_active BOOLEAN DEFAULT FALSE,
        paper_trading BOOLEAN DEFAULT FALSE,  -- For testing
        
        -- Position Limits
        max_account_allocation_pct NUMERIC(5, 2) DEFAULT 95.00,  -- Max % invested
        min_cash_balance NUMERIC(12, 2) DEFAULT 1000.00,
        
        -- Rebalancing Settings
        rebalance_frequency VARCHAR(20) DEFAULT 'QUARTERLY' 
            CHECK (rebalance_frequency IN ('MONTHLY', 'QUARTERLY', 'SEMI_ANNUAL', 'ANNUAL', 'MANUAL')),
        last_rebalanced_at TIMESTAMP,
        next_rebalance_date DATE,
        
        -- Risk Management
        stop_loss_enabled BOOLEAN DEFAULT FALSE,
        stop_loss_pct NUMERIC(5, 2) DEFAULT 15.00,  -- Trailing stop %
        
        -- Status
        account_active BOOLEAN DEFAULT TRUE,
        last_sync_at TIMESTAMP,
        last_trade_at TIMESTAMP,
        
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- Trade Execution Log (Track all trades executed)
    CREATE TABLE IF NOT EXISTS trade_execution_log (
        trade_id SERIAL PRIMARY KEY,
        account_id INTEGER NOT NULL REFERENCES trading_accounts(account_id),
        
        -- Trade Details
        symbol VARCHAR(10) NOT NULL,
        trade_type VARCHAR(10) NOT NULL CHECK (trade_type IN ('BUY', 'SELL')),
        order_type VARCHAR(20) NOT NULL CHECK (order_type IN ('MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT')),
        
        -- Quantities and Prices
        quantity NUMERIC(12, 4) NOT NULL,
        limit_price NUMERIC(12, 4),
        stop_price NUMERIC(12, 4),
        executed_price NUMERIC(12, 4),
        executed_quantity NUMERIC(12, 4),
        
        -- Costs
        commission NUMERIC(10, 4) DEFAULT 0,
        fees NUMERIC(10, 4) DEFAULT 0,
        total_cost NUMERIC(12, 2),
        
        -- Order Management
        schwab_order_id VARCHAR(100) UNIQUE,
        order_status VARCHAR(20) NOT NULL DEFAULT 'PENDING'
            CHECK (order_status IN ('PENDING', 'SUBMITTED', 'FILLED', 'PARTIAL', 'CANCELLED', 'REJECTED', 'EXPIRED')),
        
        -- Strategy Information
        portfolio_type VARCHAR(20) CHECK (portfolio_type IN ('VALUE', 'GROWTH', 'DIVIDEND')),
        signal_reason TEXT,  -- Why this trade was triggered
        signal_score NUMERIC(5, 2),  -- ACIS score that triggered trade
        
        -- Timestamps
        signal_generated_at TIMESTAMP,
        order_placed_at TIMESTAMP,
        order_filled_at TIMESTAMP,
        
        -- Error Handling
        error_message TEXT,
        retry_count INTEGER DEFAULT 0,
        
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- Account Balances (Track account values over time)
    CREATE TABLE IF NOT EXISTS account_balances (
        balance_id SERIAL PRIMARY KEY,
        account_id INTEGER NOT NULL REFERENCES trading_accounts(account_id),
        
        -- Balance Information
        total_value NUMERIC(15, 2) NOT NULL,
        cash_balance NUMERIC(15, 2) NOT NULL,
        invested_value NUMERIC(15, 2) NOT NULL,
        
        -- Performance Metrics
        daily_pnl NUMERIC(12, 2),
        daily_pnl_pct NUMERIC(8, 4),
        total_pnl NUMERIC(12, 2),
        total_pnl_pct NUMERIC(8, 4),
        
        -- Benchmark Comparison
        spy_return_pct NUMERIC(8, 4),  -- S&P 500 return for comparison
        alpha NUMERIC(8, 4),           -- Excess return vs S&P 500
        
        snapshot_date DATE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        
        UNIQUE(account_id, snapshot_date)
    );
    
    -- Current Holdings (Real-time portfolio positions)
    CREATE TABLE IF NOT EXISTS current_holdings (
        holding_id SERIAL PRIMARY KEY,
        account_id INTEGER NOT NULL REFERENCES trading_accounts(account_id),
        
        symbol VARCHAR(10) NOT NULL,
        quantity NUMERIC(12, 4) NOT NULL,
        
        -- Cost Basis
        avg_cost_basis NUMERIC(12, 4) NOT NULL,
        total_cost_basis NUMERIC(15, 2) NOT NULL,
        
        -- Current Values
        current_price NUMERIC(12, 4),
        market_value NUMERIC(15, 2),
        
        -- Performance
        unrealized_pnl NUMERIC(12, 2),
        unrealized_pnl_pct NUMERIC(8, 4),
        
        -- Portfolio Allocation
        portfolio_weight_pct NUMERIC(5, 2),
        target_weight_pct NUMERIC(5, 2),
        
        -- ACIS Scores
        current_acis_score NUMERIC(5, 2),
        entry_acis_score NUMERIC(5, 2),
        
        -- Dates
        position_opened_date DATE NOT NULL,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        
        UNIQUE(account_id, symbol)
    );
    
    -- Trading Signals Queue (Pending trades from ACIS recommendations)
    CREATE TABLE IF NOT EXISTS trading_signals_queue (
        signal_id SERIAL PRIMARY KEY,
        
        -- Signal Information
        symbol VARCHAR(10) NOT NULL,
        signal_type VARCHAR(10) NOT NULL CHECK (signal_type IN ('BUY', 'SELL', 'REBALANCE')),
        portfolio_type VARCHAR(20) NOT NULL CHECK (portfolio_type IN ('VALUE', 'GROWTH', 'DIVIDEND')),
        
        -- Scoring
        acis_score NUMERIC(5, 2) NOT NULL,
        value_score NUMERIC(5, 2),
        growth_score NUMERIC(5, 2),
        dividend_score NUMERIC(5, 2),
        
        -- Target Allocation
        target_weight_pct NUMERIC(5, 2),
        
        -- Signal Metadata
        signal_strength VARCHAR(10) CHECK (signal_strength IN ('STRONG', 'MODERATE', 'WEAK')),
        expiration_date DATE,
        
        -- Processing Status
        processed BOOLEAN DEFAULT FALSE,
        processed_at TIMESTAMP,
        accounts_notified INTEGER DEFAULT 0,  -- Number of accounts that received this signal
        
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- Automated Trading Log (Track automated trading decisions)
    CREATE TABLE IF NOT EXISTS automated_trading_log (
        log_id SERIAL PRIMARY KEY,
        account_id INTEGER REFERENCES trading_accounts(account_id),
        
        -- Action Taken
        action_type VARCHAR(50) NOT NULL,
        action_details JSONB,
        
        -- Decision Process
        decision_factors JSONB,  -- Factors that led to this decision
        confidence_score NUMERIC(5, 2),
        
        -- Results
        success BOOLEAN,
        error_message TEXT,
        
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- Create indexes for performance
    CREATE INDEX IF NOT EXISTS idx_clients_email ON clients(email);
    CREATE INDEX IF NOT EXISTS idx_clients_status ON clients(account_status);
    CREATE INDEX IF NOT EXISTS idx_trading_accounts_client ON trading_accounts(client_id);
    CREATE INDEX IF NOT EXISTS idx_trading_accounts_schwab ON trading_accounts(schwab_account_number);
    CREATE INDEX IF NOT EXISTS idx_trade_execution_symbol ON trade_execution_log(symbol);
    CREATE INDEX IF NOT EXISTS idx_trade_execution_account ON trade_execution_log(account_id);
    CREATE INDEX IF NOT EXISTS idx_trade_execution_status ON trade_execution_log(order_status);
    CREATE INDEX IF NOT EXISTS idx_account_balances_account_date ON account_balances(account_id, snapshot_date);
    CREATE INDEX IF NOT EXISTS idx_current_holdings_account ON current_holdings(account_id);
    CREATE INDEX IF NOT EXISTS idx_current_holdings_symbol ON current_holdings(symbol);
    CREATE INDEX IF NOT EXISTS idx_trading_signals_processed ON trading_signals_queue(processed);
    CREATE INDEX IF NOT EXISTS idx_trading_signals_symbol ON trading_signals_queue(symbol);
    
    -- Create update trigger for updated_at columns
    CREATE OR REPLACE FUNCTION update_updated_at_column()
    RETURNS TRIGGER AS $$
    BEGIN
        NEW.updated_at = CURRENT_TIMESTAMP;
        RETURN NEW;
    END;
    $$ language 'plpgsql';
    
    CREATE TRIGGER update_clients_updated_at BEFORE UPDATE ON clients
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
        
    CREATE TRIGGER update_trading_accounts_updated_at BEFORE UPDATE ON trading_accounts
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """
    
    return SQL

def main():
    """Create all trading tables"""
    print("\n" + "="*60)
    print("CREATING AUTOMATED TRADING TABLES")
    print("="*60)
    
    try:
        engine = create_engine(get_postgres_url())
        
        # Create tables
        print("\nCreating trading tables...")
        with engine.connect() as conn:
            conn.execute(text(create_trading_tables()))
            conn.commit()
        
        print("✅ Successfully created all trading tables:")
        print("  - clients")
        print("  - trading_accounts") 
        print("  - trade_execution_log")
        print("  - account_balances")
        print("  - current_holdings")
        print("  - trading_signals_queue")
        print("  - automated_trading_log")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())