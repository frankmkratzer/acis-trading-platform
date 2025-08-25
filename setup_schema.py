#!/usr/bin/env python3
"""
File: setup_schema.py
Purpose: Create all database tables, indexes, and materialized views for the trading system
"""

import os
import logging
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
engine = create_engine(os.getenv("POSTGRES_URL"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# =====================================
# TABLE DEFINITIONS
# =====================================

TABLES = {
    # Core symbol data
    "symbol_universe": """
                       CREATE TABLE IF NOT EXISTS symbol_universe
                       (
                           symbol
                           TEXT
                           PRIMARY
                           KEY,
                           name
                           TEXT,
                           exchange
                           TEXT,
                           security_type
                           TEXT
                           CHECK
                       (
                           security_type
                           LIKE
                           'Common%'
                           OR
                           security_type
                           LIKE
                           'Ordinary%'
                           OR
                           security_type
                           LIKE
                           'REIT%'
                       ),
                           is_etf BOOLEAN DEFAULT FALSE,
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
                           fetched_at TIMESTAMPTZ DEFAULT NOW
                       (
                       )
                           )
                       """,

    "stock_metadata": """
                      CREATE TABLE IF NOT EXISTS stock_metadata
                      (
                          symbol
                          TEXT
                          PRIMARY
                          KEY,
                          name
                          TEXT,
                          exchange
                          TEXT,
                          currency
                          TEXT,
                          country
                          TEXT,
                          sector
                          TEXT,
                          industry
                          TEXT,
                          market_cap
                          NUMERIC,
                          market_cap_class
                          TEXT,
                          is_etf
                          BOOLEAN
                          DEFAULT
                          FALSE,
                          fetched_at
                          TIMESTAMPTZ
                      )
                      """,

    # Price data
    "stock_eod_daily": """
                       CREATE TABLE IF NOT EXISTS stock_eod_daily
                       (
                           symbol
                           TEXT
                           NOT
                           NULL,
                           trade_date
                           DATE
                           NOT
                           NULL,
                           open
                           NUMERIC,
                           high
                           NUMERIC,
                           low
                           NUMERIC,
                           close
                           NUMERIC,
                           adjusted_close
                           NUMERIC,
                           volume
                           BIGINT,
                           dividend_amount
                           NUMERIC
                           DEFAULT
                           0,
                           split_coefficient
                           NUMERIC
                           DEFAULT
                           1,
                           fetched_at
                           TIMESTAMPTZ
                           DEFAULT
                           NOW
                       (
                       ),
                           PRIMARY KEY
                       (
                           symbol,
                           trade_date
                       )
                           )
                       """,

    "sp500_price_history": """
                           CREATE TABLE IF NOT EXISTS sp500_price_history
                           (
                               trade_date
                               DATE
                               PRIMARY
                               KEY,
                               open
                               NUMERIC,
                               high
                               NUMERIC,
                               low
                               NUMERIC,
                               close
                               NUMERIC,
                               adjusted_close
                               NUMERIC,
                               volume
                               BIGINT,
                               dividend_amount
                               NUMERIC
                               DEFAULT
                               0,
                               split_coefficient
                               NUMERIC
                               DEFAULT
                               1,
                               fetched_at
                               TIMESTAMPTZ
                               DEFAULT
                               NOW
                           (
                           )
                               )
                           """,

    # Dividend data
    "dividend_history": """
                        CREATE TABLE IF NOT EXISTS dividend_history
                        (
                            symbol
                            TEXT
                            NOT
                            NULL,
                            ex_date
                            DATE
                            NOT
                            NULL,
                            dividend
                            NUMERIC,
                            currency
                            TEXT
                            DEFAULT
                            'USD',
                            fetched_at
                            TIMESTAMPTZ
                            DEFAULT
                            NOW
                        (
                        ),
                            PRIMARY KEY
                        (
                            symbol,
                            ex_date
                        )
                            )
                        """,

    "dividend_growth_scores": """
                              CREATE TABLE IF NOT EXISTS dividend_growth_scores
                              (
                                  symbol
                                  TEXT
                                  NOT
                                  NULL,
                                  as_of_date
                                  DATE
                                  NOT
                                  NULL,
                                  div_cagr_1y
                                  NUMERIC,
                                  div_cagr_3y
                                  NUMERIC,
                                  div_cagr_5y
                                  NUMERIC,
                                  div_cagr_10y
                                  NUMERIC,
                                  dividend_cut_detected
                                  BOOLEAN
                                  DEFAULT
                                  FALSE,
                                  fetched_at
                                  TIMESTAMPTZ
                                  DEFAULT
                                  NOW
                              (
                              ),
                                  PRIMARY KEY
                              (
                                  symbol,
                                  as_of_date
                              )
                                  )
                              """,

    # Fundamental data
    "fundamentals_annual": """
                           CREATE TABLE IF NOT EXISTS fundamentals_annual
                           (
                               symbol
                               TEXT
                               NOT
                               NULL,
                               fiscal_date
                               DATE
                               NOT
                               NULL,
                               source
                               TEXT,
                               run_id
                               TEXT,
                               totalrevenue
                               BIGINT,
                               grossprofit
                               BIGINT,
                               netincome
                               BIGINT,
                               eps
                               NUMERIC,
                               totalassets
                               BIGINT,
                               totalliabilities
                               BIGINT,
                               totalshareholderequity
                               BIGINT,
                               operatingcashflow
                               BIGINT,
                               capitalexpenditures
                               BIGINT,
                               dividendpayout
                               BIGINT,
                               free_cf
                               BIGINT,
                               cash_flow_per_share
                               NUMERIC,
                               fetched_at
                               TIMESTAMPTZ
                               DEFAULT
                               NOW
                           (
                           ),
                               PRIMARY KEY
                           (
                               symbol,
                               fiscal_date
                           )
                               )
                           """,

    "fundamentals_quarterly": """
                              CREATE TABLE IF NOT EXISTS fundamentals_quarterly
                              (
                                  symbol
                                  TEXT
                                  NOT
                                  NULL,
                                  fiscal_date
                                  DATE
                                  NOT
                                  NULL,
                                  source
                                  TEXT,
                                  run_id
                                  TEXT,
                                  totalrevenue
                                  BIGINT,
                                  grossprofit
                                  BIGINT,
                                  netincome
                                  BIGINT,
                                  eps
                                  NUMERIC,
                                  totalassets
                                  BIGINT,
                                  totalliabilities
                                  BIGINT,
                                  totalshareholderequity
                                  BIGINT,
                                  operatingcashflow
                                  BIGINT,
                                  capitalexpenditures
                                  BIGINT,
                                  dividendpayout
                                  BIGINT,
                                  free_cf
                                  BIGINT,
                                  cash_flow_per_share
                                  NUMERIC,
                                  fetched_at
                                  TIMESTAMPTZ
                                  DEFAULT
                                  NOW
                              (
                              ),
                                  PRIMARY KEY
                              (
                                  symbol,
                                  fiscal_date
                              )
                                  )
                              """,

    # Returns and performance
    "forward_returns": """
                       CREATE TABLE IF NOT EXISTS forward_returns
                       (
                           symbol
                           TEXT
                           NOT
                           NULL,
                           as_of_date
                           DATE
                           NOT
                           NULL,
                           return_1m
                           NUMERIC,
                           return_3m
                           NUMERIC,
                           return_6m
                           NUMERIC,
                           return_12m
                           NUMERIC,
                           fetched_at
                           TIMESTAMPTZ
                           DEFAULT
                           NOW
                       (
                       ),
                           PRIMARY KEY
                       (
                           symbol,
                           as_of_date
                       )
                           )
                       """,

    "sp500_outperformance_scores": """
                                   CREATE TABLE IF NOT EXISTS sp500_outperformance_scores
                                   (
                                       symbol
                                       TEXT
                                       PRIMARY
                                       KEY,
                                       lifetime_outperformer
                                       BOOLEAN,
                                       years_outperformed
                                       INTEGER,
                                       total_years
                                       INTEGER,
                                       weighted_score
                                       NUMERIC,
                                       last_year
                                       INTEGER,
                                       fetched_at
                                       TIMESTAMPTZ
                                       DEFAULT
                                       NOW
                                   (
                                   )
                                       )
                                   """,

    # AI scoring tables
    "ai_value_scores": """
                       CREATE TABLE IF NOT EXISTS ai_value_scores
                       (
                           symbol
                           TEXT
                           NOT
                           NULL,
                           as_of_date
                           DATE
                           NOT
                           NULL,
                           score
                           NUMERIC,
                           percentile
                           NUMERIC,
                           score_label
                           TEXT,
                           rank
                           INTEGER,
                           model_version
                           TEXT,
                           score_type
                           TEXT,
                           fetched_at
                           TIMESTAMPTZ
                           DEFAULT
                           NOW
                       (
                       ),
                           PRIMARY KEY
                       (
                           symbol,
                           as_of_date
                       )
                           )
                       """,

    "ai_growth_scores": """
                        CREATE TABLE IF NOT EXISTS ai_growth_scores
                        (
                            symbol
                            TEXT
                            NOT
                            NULL,
                            as_of_date
                            DATE
                            NOT
                            NULL,
                            score
                            NUMERIC,
                            percentile
                            NUMERIC,
                            score_label
                            TEXT,
                            rank
                            INTEGER,
                            model_version
                            TEXT,
                            score_type
                            TEXT,
                            fetched_at
                            TIMESTAMPTZ
                            DEFAULT
                            NOW
                        (
                        ),
                            PRIMARY KEY
                        (
                            symbol,
                            as_of_date
                        )
                            )
                        """,

    "ai_momentum_scores": """
                          CREATE TABLE IF NOT EXISTS ai_momentum_scores
                          (
                              symbol
                              TEXT
                              NOT
                              NULL,
                              as_of_date
                              DATE
                              NOT
                              NULL,
                              score
                              NUMERIC,
                              percentile
                              NUMERIC,
                              score_label
                              TEXT,
                              rank
                              INTEGER,
                              model_version
                              TEXT,
                              score_type
                              TEXT,
                              fetched_at
                              TIMESTAMPTZ
                              DEFAULT
                              NOW
                          (
                          ),
                              PRIMARY KEY
                          (
                              symbol,
                              as_of_date
                          )
                              )
                          """,

    "ai_dividend_scores": """
                          CREATE TABLE IF NOT EXISTS ai_dividend_scores
                          (
                              symbol
                              TEXT
                              NOT
                              NULL,
                              as_of_date
                              DATE
                              NOT
                              NULL,
                              score
                              NUMERIC,
                              percentile
                              NUMERIC,
                              score_label
                              TEXT,
                              rank
                              INTEGER,
                              model_version
                              TEXT,
                              score_type
                              TEXT,
                              fetched_at
                              TIMESTAMPTZ
                              DEFAULT
                              NOW
                          (
                          ),
                              PRIMARY KEY
                          (
                              symbol,
                              as_of_date
                          )
                              )
                          """,

    # Portfolio tables
    "ai_value_portfolio": """
                          CREATE TABLE IF NOT EXISTS ai_value_portfolio
                          (
                              symbol
                              TEXT
                              NOT
                              NULL,
                              as_of_date
                              DATE
                              NOT
                              NULL,
                              score
                              NUMERIC,
                              percentile
                              NUMERIC,
                              score_label
                              TEXT,
                              rank
                              INTEGER,
                              model_version
                              TEXT,
                              score_type
                              TEXT,
                              fetched_at
                              TIMESTAMPTZ
                              DEFAULT
                              NOW
                          (
                          ),
                              PRIMARY KEY
                          (
                              symbol,
                              as_of_date,
                              score_type
                          )
                              )
                          """,

    "ai_growth_portfolio": """
                           CREATE TABLE IF NOT EXISTS ai_growth_portfolio
                           (
                               symbol
                               TEXT
                               NOT
                               NULL,
                               as_of_date
                               DATE
                               NOT
                               NULL,
                               score
                               NUMERIC,
                               percentile
                               NUMERIC,
                               score_label
                               TEXT,
                               rank
                               INTEGER,
                               model_version
                               TEXT,
                               score_type
                               TEXT,
                               fetched_at
                               TIMESTAMPTZ
                               DEFAULT
                               NOW
                           (
                           ),
                               PRIMARY KEY
                           (
                               symbol,
                               as_of_date,
                               score_type
                           )
                               )
                           """,

    "ai_momentum_portfolio": """
                             CREATE TABLE IF NOT EXISTS ai_momentum_portfolio
                             (
                                 symbol
                                 TEXT
                                 NOT
                                 NULL,
                                 as_of_date
                                 DATE
                                 NOT
                                 NULL,
                                 score
                                 NUMERIC,
                                 percentile
                                 NUMERIC,
                                 score_label
                                 TEXT,
                                 rank
                                 INTEGER,
                                 model_version
                                 TEXT,
                                 score_type
                                 TEXT,
                                 fetched_at
                                 TIMESTAMPTZ
                                 DEFAULT
                                 NOW
                             (
                             ),
                                 PRIMARY KEY
                             (
                                 symbol,
                                 as_of_date,
                                 score_type
                             )
                                 )
                             """,

    "ai_dividend_portfolio": """
                             CREATE TABLE IF NOT EXISTS ai_dividend_portfolio
                             (
                                 symbol
                                 TEXT
                                 NOT
                                 NULL,
                                 as_of_date
                                 DATE
                                 NOT
                                 NULL,
                                 score
                                 NUMERIC,
                                 percentile
                                 NUMERIC,
                                 score_label
                                 TEXT,
                                 rank
                                 INTEGER,
                                 model_version
                                 TEXT,
                                 score_type
                                 TEXT,
                                 fetched_at
                                 TIMESTAMPTZ
                                 DEFAULT
                                 NOW
                             (
                             ),
                                 PRIMARY KEY
                             (
                                 symbol,
                                 as_of_date,
                                 score_type
                             )
                                 )
                             """,

    # Model and trading tables
    "ai_model_run_log": """
                        CREATE TABLE IF NOT EXISTS ai_model_run_log
                        (
                            run_id
                            TEXT
                            PRIMARY
                            KEY,
                            model_type
                            TEXT,
                            version
                            TEXT,
                            as_of_date
                            DATE,
                            features
                            JSONB,
                            hyperparameters
                            JSONB,
                            notes
                            TEXT,
                            created_at
                            TIMESTAMPTZ
                            DEFAULT
                            NOW
                        (
                        )
                            )
                        """,

    "trading_orders": """
                      CREATE TABLE IF NOT EXISTS trading_orders
                      (
                          order_id
                          TEXT
                          PRIMARY
                          KEY,
                          symbol
                          TEXT
                          NOT
                          NULL,
                          side
                          TEXT,
                          quantity
                          INTEGER,
                          order_type
                          TEXT,
                          limit_price
                          NUMERIC,
                          stop_price
                          NUMERIC,
                          status
                          TEXT,
                          filled_quantity
                          INTEGER
                          DEFAULT
                          0,
                          avg_fill_price
                          NUMERIC,
                          submitted_at
                          TIMESTAMPTZ,
                          filled_at
                          TIMESTAMPTZ,
                          commission
                          NUMERIC
                          DEFAULT
                          0,
                          metadata
                          JSONB,
                          created_at
                          TIMESTAMPTZ
                          DEFAULT
                          NOW
                      (
                      )
                          )
                      """
}

# =====================================
# INDEXES
# =====================================

INDEXES = [
    # Performance indexes
    "CREATE INDEX IF NOT EXISTS idx_stock_eod_symbol ON stock_eod_daily(symbol)",
    "CREATE INDEX IF NOT EXISTS idx_stock_eod_date ON stock_eod_daily(trade_date)",
    "CREATE INDEX IF NOT EXISTS idx_stock_eod_symbol_date ON stock_eod_daily(symbol, trade_date DESC)",

    # Fundamentals indexes
    "CREATE INDEX IF NOT EXISTS idx_fund_annual_symbol ON fundamentals_annual(symbol)",
    "CREATE INDEX IF NOT EXISTS idx_fund_annual_date ON fundamentals_annual(fiscal_date)",
    "CREATE INDEX IF NOT EXISTS idx_fund_quarterly_symbol ON fundamentals_quarterly(symbol)",
    "CREATE INDEX IF NOT EXISTS idx_fund_quarterly_date ON fundamentals_quarterly(fiscal_date)",

    # Scoring indexes
    "CREATE INDEX IF NOT EXISTS idx_value_scores_date ON ai_value_scores(as_of_date)",
    "CREATE INDEX IF NOT EXISTS idx_growth_scores_date ON ai_growth_scores(as_of_date)",
    "CREATE INDEX IF NOT EXISTS idx_momentum_scores_date ON ai_momentum_scores(as_of_date)",
    "CREATE INDEX IF NOT EXISTS idx_dividend_scores_date ON ai_dividend_scores(as_of_date)",

    # Forward returns
    "CREATE INDEX IF NOT EXISTS idx_forward_returns_symbol ON forward_returns(symbol)",
    "CREATE INDEX IF NOT EXISTS idx_forward_returns_date ON forward_returns(as_of_date)",

    # Trading orders
    "CREATE INDEX IF NOT EXISTS idx_orders_symbol ON trading_orders(symbol)",
    "CREATE INDEX IF NOT EXISTS idx_orders_status ON trading_orders(status)",
    "CREATE INDEX IF NOT EXISTS idx_orders_submitted ON trading_orders(submitted_at)"
]

# =====================================
# MATERIALIZED VIEWS
# =====================================

MATERIALIZED_VIEWS = {
    "mv_symbol_with_metadata": """
        CREATE MATERIALIZED VIEW IF NOT EXISTS mv_symbol_with_metadata AS
        SELECT 
            u.symbol,
            u.name,
            u.exchange,
            u.sector,
            u.industry,
            u.market_cap,
            u.pe_ratio,
            u.dividend_yield,
            u.currency,
            u.country,
            u.is_etf
        FROM symbol_universe u
        WHERE u.market_cap > 0
    """,

    "mv_latest_annual_fundamentals": """
        CREATE MATERIALIZED VIEW IF NOT EXISTS mv_latest_annual_fundamentals AS
        SELECT DISTINCT ON (symbol)
            symbol,
            fiscal_date,
            totalrevenue,
            netincome,
            free_cf,
            totalshareholderequity,
            totalassets
        FROM fundamentals_annual
        ORDER BY symbol, fiscal_date DESC
    """,

    "mv_latest_forward_returns": """
        CREATE MATERIALIZED VIEW IF NOT EXISTS mv_latest_forward_returns AS
        SELECT DISTINCT ON (symbol)
            symbol,
            as_of_date,
            return_1m,
            return_3m,
            return_6m,
            return_12m
        FROM forward_returns
        ORDER BY symbol, as_of_date DESC
    """,

    "mv_current_ai_portfolios": """
        CREATE MATERIALIZED VIEW IF NOT EXISTS mv_current_ai_portfolios AS
        WITH latest_dates AS (
            SELECT 'value' as strategy, MAX(as_of_date) as max_date FROM ai_value_portfolio
            UNION ALL
            SELECT 'growth', MAX(as_of_date) FROM ai_growth_portfolio
            UNION ALL
            SELECT 'momentum', MAX(as_of_date) FROM ai_momentum_portfolio
            UNION ALL
            SELECT 'dividend', MAX(as_of_date) FROM ai_dividend_portfolio
        )
        SELECT 'value' as strategy, v.* 
        FROM ai_value_portfolio v
        JOIN latest_dates l ON l.strategy = 'value' AND v.as_of_date = l.max_date
        UNION ALL
        SELECT 'growth', g.* 
        FROM ai_growth_portfolio g
        JOIN latest_dates l ON l.strategy = 'growth' AND g.as_of_date = l.max_date
        UNION ALL
        SELECT 'momentum', m.* 
        FROM ai_momentum_portfolio m
        JOIN latest_dates l ON l.strategy = 'momentum' AND m.as_of_date = l.max_date
        UNION ALL
        SELECT 'dividend', d.* 
        FROM ai_dividend_portfolio d
        JOIN latest_dates l ON l.strategy = 'dividend' AND d.as_of_date = l.max_date
    """
}


# =====================================
# MAIN EXECUTION
# =====================================

def create_tables():
    """Create all tables"""
    logger.info("Creating tables...")

    with engine.begin() as conn:
        for table_name, ddl in TABLES.items():
            try:
                conn.execute(text(ddl))
                logger.info(f"✅ Created table: {table_name}")
            except Exception as e:
                logger.error(f"❌ Error creating table {table_name}: {e}")


def create_indexes():
    """Create all indexes"""
    logger.info("Creating indexes...")

    with engine.begin() as conn:
        for index_ddl in INDEXES:
            try:
                conn.execute(text(index_ddl))
                # Extract index name for logging
                index_name = index_ddl.split("INDEX IF NOT EXISTS")[1].split("ON")[0].strip()
                logger.info(f"✅ Created index: {index_name}")
            except Exception as e:
                logger.error(f"❌ Error creating index: {e}")


def create_materialized_views():
    """Create all materialized views"""
    logger.info("Creating materialized views...")

    with engine.begin() as conn:
        for view_name, ddl in MATERIALIZED_VIEWS.items():
            try:
                conn.execute(text(ddl))

                # Create unique index for concurrent refresh
                if view_name == "mv_symbol_with_metadata":
                    conn.execute(text(f"CREATE UNIQUE INDEX IF NOT EXISTS {view_name}_idx ON {view_name}(symbol)"))
                elif view_name == "mv_latest_annual_fundamentals":
                    conn.execute(text(f"CREATE UNIQUE INDEX IF NOT EXISTS {view_name}_idx ON {view_name}(symbol)"))
                elif view_name == "mv_latest_forward_returns":
                    conn.execute(text(f"CREATE UNIQUE INDEX IF NOT EXISTS {view_name}_idx ON {view_name}(symbol)"))

                logger.info(f"✅ Created materialized view: {view_name}")
            except Exception as e:
                logger.error(f"❌ Error creating view {view_name}: {e}")


def refresh_materialized_views():
    """Refresh all materialized views"""
    logger.info("Refreshing materialized views...")

    with engine.begin() as conn:
        for view_name in MATERIALIZED_VIEWS.keys():
            try:
                # Try concurrent refresh first
                conn.execute(text(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {view_name}"))
                logger.info(f"✅ Refreshed view: {view_name}")
            except Exception:
                # Fallback to regular refresh
                try:
                    conn.execute(text(f"REFRESH MATERIALIZED VIEW {view_name}"))
                    logger.info(f"✅ Refreshed view (non-concurrent): {view_name}")
                except Exception as e:
                    logger.error(f"❌ Error refreshing view {view_name}: {e}")


def verify_schema():
    """Verify all tables exist"""
    logger.info("Verifying schema...")

    with engine.connect() as conn:
        result = conn.execute(text("""
                                   SELECT table_name
                                   FROM information_schema.tables
                                   WHERE table_schema = 'public'
                                   ORDER BY table_name
                                   """))

        existing_tables = [row[0] for row in result]

        logger.info(f"Found {len(existing_tables)} tables in database")

        # Check for missing tables
        expected_tables = set(TABLES.keys())
        existing_set = set(existing_tables)
        missing = expected_tables - existing_set

        if missing:
            logger.warning(f"⚠️ Missing tables: {missing}")
        else:
            logger.info("✅ All expected tables exist")

        return len(missing) == 0


def main():
    """Main execution"""
    print("🚀 Setting up database schema for trading system...")

    # Create everything
    create_tables()
    create_indexes()
    create_materialized_views()

    # Verify
    if verify_schema():
        print("✅ Schema setup complete!")

        # Initial refresh of views
        refresh_materialized_views()
        print("✅ Materialized views refreshed!")
    else:
        print("⚠️ Some tables are missing. Please check the logs.")

    print("\n📊 Database ready for trading system!")


if __name__ == "__main__":
    main()