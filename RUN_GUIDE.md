# ACIS Trading Platform - Complete Execution Guide

## Quick Start - Run Everything

```bash
# First time setup - initialize database
python database/setup_schema_clean.py

# Run complete daily pipeline (fastest, ~30-45 min)
python main.py --daily

# OR run weekly pipeline with fundamentals (slower, ~2-3 hours)
python main.py --weekly
```

## Step-by-Step First Time Setup

### 1. Environment Setup
```bash
# Create .env file with your credentials
POSTGRES_URL=postgresql://user:pass@host:port/dbname
ALPHA_VANTAGE_API_KEY=your_premium_key_here
```

### 2. Database Setup
```bash
# Create all 15 essential tables
python database/setup_schema_clean.py

# Verify tables were created
python database/verify_tables.py
```

### 3. Initial Data Population

#### Step 1: Fetch Stock Universe (~1-2 hours)
```bash
# Get all mid/large cap US common stocks ($2B+)
python data_fetch/market_data/fetch_quality_stocks.py
```

#### Step 2: Fetch S&P 500 History
```bash
# Get S&P 500 constituents and historical changes
python data_fetch/market_data/fetch_sp500_history.py
```

#### Step 3: Fetch Price Data (~2-3 hours)
```bash
# Get daily OHLCV for all stocks
python data_fetch/market_data/fetch_prices.py
```

#### Step 4: Fetch Fundamentals (optional, ~2-3 hours)
```bash
# Get financial statements and metrics
python data_fetch/fundamentals/fetch_fundamentals.py
python data_fetch/fundamentals/fetch_company_overview.py
python data_fetch/fundamentals/calculate_enterprise_value.py
```

### 4. Calculate Rankings & Metrics

#### Option A: Run All Rankings (~30-45 min)
```bash
python rankings/orchestrator.py --phase all
```

#### Option B: Run Specific Components
```bash
# Populate base data (steps 01-09)
python rankings/orchestrator.py --phase populate

# Calculate rankings (steps 10-17)
python rankings/orchestrator.py --phase calculate

# Run validation
python rankings/orchestrator.py --phase validate
```

### 5. Portfolio Analysis
```bash
# Calculate portfolio scores for three strategies
python portfolios/calculate_portfolio_scores.py

# Generate portfolio holdings
python portfolios/portfolio_manager.py
```

## Daily Operations (After Initial Setup)

### Morning Pipeline (Before Market Open)
```bash
# Quick update - prices and rankings only
python main.py --daily
```

This runs:
1. Updates stock prices for previous trading day
2. Recalculates technical indicators
3. Updates breakout signals
4. Refreshes portfolio scores

### Weekly Pipeline (Weekend)
```bash
# Full update including fundamentals
python main.py --weekly
```

This runs everything in daily PLUS:
1. Updates company fundamentals
2. Recalculates all value metrics
3. Full ranking recalculation
4. Portfolio rebalancing check

## Individual Components (As Needed)

### Update Specific Data
```bash
# Just prices
python data_fetch/market_data/fetch_prices.py

# Just fundamentals  
python data_fetch/fundamentals/fetch_fundamentals.py

# Just technical indicators
python data_fetch/market_data/fetch_technical_indicators.py
```

### Recalculate Specific Metrics
```bash
# Excess cash flow
python rankings/calculate/15_calculate_excess_cash_flow.py

# Dividend sustainability
python rankings/calculate/16_calculate_dividend_sustainability.py

# Breakout signals
python strategies/detect_breakouts.py
```

### Portfolio Operations
```bash
# View current portfolios
python portfolios/view_portfolios.py

# Check for rebalancing needs
python portfolios/check_rebalance.py

# Export portfolio to CSV
python portfolios/export_portfolios.py
```

## Monitoring & Verification

### Check Data Freshness
```bash
# See what data needs updating
python utils/check_data_staleness.py
```

### Verify Data Quality
```bash
# Run data validation
python rankings/validate/19_comprehensive_validation.py
```

### Database Health
```bash
# Check table sizes and row counts
python database/verify_tables.py
```

## Resume After Interruption

All major scripts support resuming:
```bash
# Resume daily pipeline
python main.py --daily --skip-completed

# Resume rankings calculation
python rankings/orchestrator.py --phase all --skip-completed

# Resume data fetching
python data_fetch/orchestrator.py --mode daily --skip-completed
```

## Three-Portfolio Strategy Outputs

After running the complete pipeline, you'll have:

1. **VALUE Portfolio**: Top 10 undervalued quality stocks
2. **GROWTH Portfolio**: Top 10 consistent outperformers  
3. **DIVIDEND Portfolio**: Top 10 sustainable dividend growers

Access results:
```sql
-- View portfolio holdings
SELECT * FROM portfolio_holdings WHERE portfolio_name IN ('VALUE', 'GROWTH', 'DIVIDEND');

-- View portfolio scores
SELECT * FROM portfolio_scores ORDER BY composite_score DESC LIMIT 30;
```

## Performance Expectations

- **Initial full setup**: 4-6 hours (one time)
- **Daily update**: 30-45 minutes
- **Weekly update**: 2-3 hours
- **API rate limit**: 600 calls/minute (Premium Alpha Vantage)
- **Database size**: ~5-10GB after full population

## Troubleshooting

### If fetching fails:
```bash
# Check API key
echo $ALPHA_VANTAGE_API_KEY

# Test API connection
python utils/test_alpha_vantage.py
```

### If rankings timeout:
```bash
# Run in smaller phases
python rankings/orchestrator.py --phase populate
python rankings/orchestrator.py --phase calculate --skip-completed
```

### If database connection fails:
```bash
# Test connection
python database/test_connection.py

# Check PostgreSQL is running
psql -U your_user -d your_db -c "SELECT 1;"
```

## Important Notes

1. **Market Hours**: Run daily pipeline AFTER market close (4:30 PM ET or later)
2. **Weekends**: Best time for weekly pipeline with fundamentals
3. **API Limits**: Premium API allows 600 calls/min, script auto-throttles
4. **Disk Space**: Ensure 10GB+ free space for database growth
5. **Memory**: Rankings calculation needs 4-8GB RAM for 15M+ records