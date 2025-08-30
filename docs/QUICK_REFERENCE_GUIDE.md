# ACIS Trading Platform - Quick Reference Guide

## 🚀 Quick Start Commands

```bash
# Initial setup
python setup_schema_chunked.py        # Create database tables
python fetch_symbol_universe.py       # Load stock symbols
python run_daily_pipeline.py          # Run daily data collection
python run_weekly_pipeline.py         # Run weekly analysis

# Daily operations
python run_eod_full_pipeline.py       # Complete end-of-day pipeline
```

## 📊 Critical Tables to Remember

### The Two Forward Returns Tables (DO NOT CONFUSE!)

| Table | Structure | Used By | Purpose |
|-------|-----------|---------|---------|
| `forward_returns` | Wide (return_1m, return_3m, etc.) | compute_forward_returns.py | Simple time series |
| `ml_forward_returns` | Long (horizon_weeks, risk metrics) | calculate_forward_returns.py, ml_strategy_framework.py | ML training |

**⚠️ NEVER mix these tables - they have different structures!**

## 🎯 The 7 Rankings (Lower = Better)

1. **SP500 Outperformance** → Consistent market beaters
2. **Excess Cash Flow** → Cash generation machines  
3. **Fundamentals Trend** → Improving business metrics
4. **News Sentiment** → Positive market perception
5. **Value** → Historically cheap stocks
6. **Breakout** → Technical momentum plays
7. **Growth** → Long-term compounders

**Composite Score** = Average of all 7 (Elite < 100, Premium < 250, Quality < 500)

## 🔄 Pipeline Order (Dependencies Matter!)

### Daily Pipeline
```
1. fetch_symbol_universe.py
2. fetch_sp500_history.py  
3. fetch_prices.py           ← Takes longest (4600+ stocks)
4. fetch_technical_indicators.py
5. fetch_dividend_history.py
```

### Weekly Pipeline  
```
1. fetch_company_overview.py
2. fetch_fundamentals.py
3. fetch_options.py
4. fetch_news_sentiment.py
5. compute_forward_returns.py     → forward_returns table
6. calculate_quality_rankings.py   ← Creates 7 rankings
7. calculate_historical_rankings.py → ml_forward_returns table
8. calculate_forward_returns.py    → ml_forward_returns table
9. ml_strategy_framework.py        ← Reads ml_forward_returns
10. generate_trading_signals.py
```

## 🐛 Common Issues & Fixes

### Issue: "column X is of type Y but expression is of type Z"
```python
# Fix: Check date type conversions in fetch scripts
df['date_column'] = pd.to_datetime(df['date_column'])
```

### Issue: "forward_returns table structure mismatch"
```bash
# Fix: Scripts are using wrong table
# Check which table the script should use:
grep "FROM forward_returns" script_name.py
# Should be ml_forward_returns for ML scripts
```

### Issue: Pipeline timeout
```bash
# Fix: Run scripts individually or in smaller batches
python fetch_prices.py --batch_size 100
```

### Issue: Missing IPO date column
```bash
# Fix: IPO date has been removed from schema
# No action needed - this is intentional
```

## 📁 Key Files

### Core Schema
- `setup_schema.py` - Complete schema (may timeout)
- `setup_schema_chunked.py` - Simplified schema (faster)

### Data Collection
- `fetch_*.py` - Various data fetchers
- `run_daily_pipeline.py` - Daily orchestrator
- `run_weekly_pipeline.py` - Weekly orchestrator

### Analysis & ML
- `calculate_quality_rankings.py` - Generate 7 rankings
- `calculate_forward_returns.py` - ML forward returns
- `compute_forward_returns.py` - Simple forward returns
- `ml_strategy_framework.py` - ML models

## 🎨 Strategy Flags

Stocks are automatically flagged for strategies:
- `is_sp500_beater` - Beat S&P >50% of time
- `is_cash_generator` - Top quartile FCF yield
- `is_fundamental_grower` - Positive trends
- `is_deep_value_star` - Multiple value extremes
- `is_momentum_breakout` - Volume-confirmed breakout
- `is_growth_champion` - Long-term compounder
- `is_all_star` - Elite across multiple rankings

## 📈 SQL Quick Queries

```sql
-- Top 10 stocks overall
SELECT symbol, composite_quality_score, quality_tier 
FROM stock_quality_rankings
WHERE ranking_date = (SELECT MAX(ranking_date) FROM stock_quality_rankings)
ORDER BY composite_quality_score
LIMIT 10;

-- Best cash generators
SELECT symbol, fcf_yield, excess_cash_flow_ranking
FROM stock_quality_rankings  
WHERE ranking_date = (SELECT MAX(ranking_date) FROM stock_quality_rankings)
  AND fcf_yield > 0.10
ORDER BY excess_cash_flow_ranking;

-- Recent breakouts
SELECT symbol, breakout_ranking, price_change_3m
FROM stock_quality_rankings
WHERE ranking_date = (SELECT MAX(ranking_date) FROM stock_quality_rankings)
  AND is_volume_confirmed = true
ORDER BY breakout_ranking
LIMIT 20;
```

## 🔧 Environment Variables (.env)

```bash
POSTGRES_URL=postgresql://user:pass@host:port/dbname
ALPHA_VANTAGE_API_KEY=your_key_here
FMP_API_KEY=optional_key_here
```

## 📝 Important Notes

1. **API Limits**: Alpha Vantage = 600 calls/min (we use 595 for safety)
2. **Processing Time**: Daily pipeline ~30-60 min, Weekly ~2-4 hours
3. **Stock Universe**: ~4,600 US common stocks (NYSE, NASDAQ, AMEX)
4. **No ETFs**: System filters out ETFs, REITs, funds, etc.
5. **Rankings**: 1 = Best, Higher numbers = Worse

## 🚦 Health Checks

```bash
# Check latest data
psql -c "SELECT MAX(fetched_at) FROM stock_prices;"

# Check ranking date
psql -c "SELECT MAX(ranking_date) FROM stock_quality_rankings;"

# Count stocks
psql -c "SELECT COUNT(DISTINCT symbol) FROM symbol_universe WHERE is_etf = false;"
```

## 💡 Pro Tips

1. Run weekly pipeline on weekends (no market data changes)
2. Always run daily pipeline before weekly (dependencies)
3. Monitor `logs/` directory for errors
4. Use `setup_schema_chunked.py` for faster setup
5. Keep IPO date removed (not needed, causes issues)

---

**Remember**: When in doubt, check `ACIS_COMPLETE_STRATEGY_DOCUMENTATION.md` for full details!