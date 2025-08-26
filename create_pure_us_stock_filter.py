#!/usr/bin/env python3
"""
Create Pure US Stock Filter
Filter out ETFs, REITs, foreign stocks, ADRs, and other non-common stock securities
"""

import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

def create_pure_us_stock_filter():
    """Create filtered view for pure US common stocks only"""
    print("CREATING PURE US STOCK FILTER")
    print("=" * 50)
    
    load_dotenv()
    engine = create_engine(os.getenv('POSTGRES_URL'))
    
    with engine.connect() as conn:
        # First, analyze what we're filtering out
        print("Analyzing securities to filter...")
        
        # Check foreign/ADR stocks
        result = conn.execute(text("""
            SELECT COUNT(*) as count
            FROM symbol_universe 
            WHERE security_type = 'Common Stock'
              AND (name ILIKE '%ADR%' 
                OR name ILIKE '%ADS%'
                OR name ILIKE '%Ltd%'
                OR name ILIKE '%Limited%'
                OR name ILIKE '%SA%'
                OR name ILIKE '%NV%'  
                OR name ILIKE '%SE%'
                OR name ILIKE '%PLC%'
                OR name ILIKE '%Corp Ltd%'
                OR name ILIKE '%Holdings Ltd%'
                OR symbol LIKE '%.%')  -- Foreign tickers often have dots
        """))
        foreign_count = result.fetchone()[0]
        
        # Check REITs and Trusts  
        result = conn.execute(text("""
            SELECT COUNT(*) as count
            FROM symbol_universe 
            WHERE security_type = 'Common Stock'
              AND (name ILIKE '%REIT%'
                OR name ILIKE '%Trust%' 
                OR sector ILIKE '%REIT%'
                OR name ILIKE '%Trust Fund%'
                OR name ILIKE '%Investment Trust%')
        """))
        reit_count = result.fetchone()[0]
        
        # Check unusual securities (warrants, units, etc.)
        result = conn.execute(text("""
            SELECT COUNT(*) as count  
            FROM symbol_universe 
            WHERE security_type = 'Common Stock'
              AND (name ILIKE '%warrant%'
                OR name ILIKE '%unit%'
                OR name ILIKE '%rights%'
                OR symbol LIKE '%WS'     -- Warrants
                OR symbol LIKE '%WT'     -- Warrants  
                OR symbol LIKE '%U'      -- Units
                OR symbol LIKE '%R'      -- Rights
                OR symbol LIKE '%+'      -- Special symbols
                OR symbol LIKE '%-%'     -- Special symbols
                OR symbol LIKE '%/%')    -- Special symbols
        """))
        unusual_count = result.fetchone()[0]
        
        # Check penny stocks (often problematic)
        result = conn.execute(text("""
            SELECT COUNT(DISTINCT s.symbol) as count
            FROM symbol_universe s
            JOIN stock_eod_daily p ON s.symbol = p.symbol
            WHERE s.security_type = 'Common Stock'
              AND p.trade_date >= CURRENT_DATE - INTERVAL '30 days'
              AND p.adjusted_close < 1.0  -- Under $1
        """))
        penny_count = result.fetchone()[0]
        
        print(f"Securities to filter out:")
        print(f"  Foreign/ADR stocks: {foreign_count}")
        print(f"  REITs and Trusts: {reit_count}")
        print(f"  Warrants/Units/Rights: {unusual_count}")  
        print(f"  Penny stocks (<$1): {penny_count}")
        
        # Get current total
        result = conn.execute(text("""
            SELECT COUNT(*) as total FROM symbol_universe WHERE security_type = 'Common Stock'
        """))
        current_total = result.fetchone()[0]
        
        estimated_filtered = current_total - (foreign_count + reit_count + unusual_count + penny_count)
        print(f"  Current total: {current_total}")
        print(f"  Estimated after filtering: {estimated_filtered}")
        
        # Create the pure US stock filter table
        print("\nCreating pure US stock filter...")
        
        conn.execute(text("""
            DROP TABLE IF EXISTS pure_us_stocks
        """))
        
        conn.execute(text("""
            CREATE TABLE pure_us_stocks AS
            SELECT DISTINCT s.symbol, s.name, s.exchange, s.sector, s.industry, s.market_cap
            FROM symbol_universe s
            JOIN stock_eod_daily p ON s.symbol = p.symbol
            WHERE s.security_type = 'Common Stock'
              
              -- Exclude foreign/ADR stocks
              AND NOT (s.name ILIKE '%ADR%' 
                    OR s.name ILIKE '%ADS%'
                    OR s.name ILIKE '%Ltd%'
                    OR s.name ILIKE '%Limited%'
                    OR s.name ILIKE '%SA%'
                    OR s.name ILIKE '%NV%'  
                    OR s.name ILIKE '%SE%'
                    OR s.name ILIKE '%PLC%'
                    OR s.name ILIKE '%Corp Ltd%'
                    OR s.name ILIKE '%Holdings Ltd%'
                    OR s.symbol LIKE '%.%')
              
              -- Exclude REITs and Trusts
              AND NOT (s.name ILIKE '%REIT%'
                    OR s.name ILIKE '%Trust%'
                    OR s.sector ILIKE '%REIT%' 
                    OR s.name ILIKE '%Trust Fund%'
                    OR s.name ILIKE '%Investment Trust%')
              
              -- Exclude warrants, units, rights
              AND NOT (s.name ILIKE '%warrant%'
                    OR s.name ILIKE '%unit%'
                    OR s.name ILIKE '%rights%'
                    OR s.symbol LIKE '%WS'
                    OR s.symbol LIKE '%WT'  
                    OR s.symbol LIKE '%U'
                    OR s.symbol LIKE '%R'
                    OR s.symbol LIKE '%+'
                    OR s.symbol LIKE '%-%'
                    OR s.symbol LIKE '%/%')
              
              -- Only NYSE, NASDAQ, AMEX (exclude OTC)
              AND s.exchange IN ('NYSE', 'NASDAQ', 'AMEX')
              
              -- Must have recent price data (active trading)
              AND p.trade_date >= CURRENT_DATE - INTERVAL '30 days'
              AND p.adjusted_close >= 1.00  -- No penny stocks
              AND p.volume > 0  -- Must have volume
        """))
        
        # Add primary key
        conn.execute(text("""
            ALTER TABLE pure_us_stocks ADD PRIMARY KEY (symbol)
        """))
        
        # Add indexes
        conn.execute(text("""
            CREATE INDEX idx_pure_us_exchange ON pure_us_stocks(exchange);
            CREATE INDEX idx_pure_us_sector ON pure_us_stocks(sector);
            CREATE INDEX idx_pure_us_market_cap ON pure_us_stocks(market_cap DESC);
        """))
        
        conn.commit()
        
        # Verify results
        result = conn.execute(text("SELECT COUNT(*) FROM pure_us_stocks"))
        final_count = result.fetchone()[0]
        
        print(f"\nPure US stocks filter created: {final_count} symbols")
        
        # Show sample of filtered stocks
        result = conn.execute(text("""
            SELECT symbol, name, exchange, sector
            FROM pure_us_stocks 
            ORDER BY market_cap DESC NULLS LAST
            LIMIT 10
        """))
        
        print(f"\nSample filtered stocks (largest by market cap):")
        for row in result:
            print(f"  {row[0]}: {row[1]} ({row[2]}) - {row[3]}")
        
        # Show what was filtered out (examples)
        print(f"\nExamples of filtered OUT securities:")
        
        result = conn.execute(text("""
            SELECT symbol, name, 'Foreign/ADR' as reason
            FROM symbol_universe
            WHERE security_type = 'Common Stock'
              AND (name ILIKE '%ADR%' OR name ILIKE '%Ltd%' OR name ILIKE '%PLC%')
              AND symbol NOT IN (SELECT symbol FROM pure_us_stocks)
            LIMIT 3
            
            UNION ALL
            
            SELECT symbol, name, 'REIT' as reason  
            FROM symbol_universe
            WHERE security_type = 'Common Stock'
              AND (name ILIKE '%REIT%' OR name ILIKE '%Trust%')
              AND symbol NOT IN (SELECT symbol FROM pure_us_stocks)
            LIMIT 3
        """))
        
        for row in result:
            print(f"  {row[0]}: {row[1]} ({row[2]})")
        
        return final_count

def update_strategy_queries_for_pure_stocks():
    """Update strategy system to use pure_us_stocks filter"""
    print(f"\n" + "=" * 50)
    print("UPDATING STRATEGY SYSTEM FOR PURE US STOCKS")
    print("=" * 50)
    
    load_dotenv()
    engine = create_engine(os.getenv('POSTGRES_URL'))
    
    with engine.connect() as conn:
        # Clear existing strategy scores to force regeneration
        strategy_tables = [
            'ai_value_small_cap_scores', 'ai_growth_small_cap_scores', 'ai_momentum_small_cap_scores', 'ai_dividend_small_cap_scores',
            'ai_value_mid_cap_scores', 'ai_growth_mid_cap_scores', 'ai_momentum_mid_cap_scores', 'ai_dividend_mid_cap_scores', 
            'ai_value_large_cap_scores', 'ai_growth_large_cap_scores', 'ai_momentum_large_cap_scores', 'ai_dividend_large_cap_scores'
        ]
        
        cleared_count = 0
        for table in strategy_tables:
            try:
                result = conn.execute(text(f"DELETE FROM {table} WHERE as_of_date = CURRENT_DATE"))
                cleared_count += result.rowcount
            except:
                pass  # Table might not exist
        
        conn.commit()
        print(f"Cleared {cleared_count} existing strategy scores for regeneration")
        
        return True

def main():
    """Create pure US stock filter and update system"""
    
    # Create the filter
    stock_count = create_pure_us_stock_filter()
    
    # Update strategy system
    update_strategy_queries_for_pure_stocks()
    
    print(f"\n" + "=" * 50)
    print("PURE US STOCK FILTER COMPLETE")  
    print(f"Ready to regenerate strategies with {stock_count} pure US stocks")
    print("Run create_12_strategy_system.py next with --pure-stocks flag")
    print("=" * 50)
    
    return stock_count

if __name__ == "__main__":
    main()