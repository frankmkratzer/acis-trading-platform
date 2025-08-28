# ACIS Trading Platform - Complete Strategy Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Core Philosophy](#core-philosophy)
3. [Data Architecture](#data-architecture)
4. [Ranking System](#ranking-system)
5. [Pipeline Structure](#pipeline-structure)
6. [ML Integration](#ml-integration)
7. [Operational Guide](#operational-guide)
8. [Technical Decisions](#technical-decisions)

---

## System Overview

### What is ACIS?
ACIS (Advanced Capital Intelligence System) is a comprehensive stock analysis and trading platform that:
- Ranks ~4,600 US stocks across 7 quality dimensions
- Generates ML-based trading signals
- Tracks performance against S&P 500 benchmark
- Provides risk-adjusted return predictions

### Key Components
1. **Data Collection Pipeline** - Daily/weekly fetching of prices, fundamentals, news
2. **Quality Ranking System** - Multi-factor stock ranking across 7 dimensions
3. **Forward Returns Calculation** - Two systems: simple and ML-focused
4. **ML Strategy Framework** - Machine learning models for return prediction
5. **Signal Generation** - Trading signals based on rankings and ML predictions

---

## Core Philosophy

### Investment Approach
The system is built on the principle that **quality beats the market over time**. We identify quality through:

1. **Historical Outperformance** - Stocks that consistently beat S&P 500
2. **Cash Generation** - Strong free cash flow metrics
3. **Fundamental Momentum** - Improving business metrics
4. **Market Sentiment** - Positive news and analyst coverage
5. **Historical Value** - Trading below historical averages
6. **Technical Breakouts** - Price and volume momentum
7. **Long-term Growth** - Consistent compounders

### Why 7 Rankings?
Each ranking captures a different aspect of quality:
- Some investors care about value (Value Ranking)
- Others focus on momentum (Breakout Ranking)
- Long-term investors want consistency (Growth Ranking)
- The composite score identifies stocks strong across multiple dimensions

---

## Data Architecture

### Database Tables Structure

```
┌─────────────────────────────────────────────────────────┐
│                    Core Data Tables                      │
├─────────────────────────────────────────────────────────┤
│ symbol_universe      │ Master list of US stocks          │
│ stock_prices         │ Daily OHLCV data                  │
│ sp500_price_history  │ Benchmark data                    │
│ fundamentals         │ Financial statements              │
│ technical_indicators │ RSI, MACD, Bollinger, etc.       │
│ dividend_history     │ Dividend payments                 │
│ options_data         │ Options chains                    │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                  Analytics Tables                        │
├─────────────────────────────────────────────────────────┤
│ stock_quality_rankings    │ 7 ranking dimensions + composite │
│ sp500_outperformance_detail │ Year-by-year excess returns    │
│ forward_returns           │ Simple 1m,3m,6m,12m returns     │
│ ml_forward_returns        │ ML returns with risk metrics     │
│ ranking_transitions       │ Ranking changes over time        │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                  News & Sentiment Tables                 │
├─────────────────────────────────────────────────────────┤
│ news_articles            │ Raw news articles               │
│ news_sentiment_by_symbol │ Symbol-specific sentiment       │
│ news_topics             │ Article topics                   │
│ daily_sentiment_summary │ Aggregated daily sentiment       │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                     ML Tables                            │
├─────────────────────────────────────────────────────────┤
│ ml_models               │ Model registry                   │
│ ml_predictions          │ Model predictions                │
│ strategy_signals        │ Trading signals                  │
│ portfolio_performance   │ Performance tracking             │
│ backtest_results        │ Backtest metrics                 │
│ ml_features            │ Feature store                    │
│ model_training_queue   │ Training queue                   │
└─────────────────────────────────────────────────────────┘
```

### Critical Table Relationships

```sql
-- Forward Returns: TWO SEPARATE SYSTEMS
-- 1. Simple returns (wide format)
forward_returns: symbol, as_of_date, return_1m, return_3m, return_6m, return_12m

-- 2. ML returns (long format with risk)  
ml_forward_returns: symbol, ranking_date, horizon_weeks, forward_return, 
                   forward_excess_return, forward_volatility, forward_max_drawdown
```

**IMPORTANT**: Never mix these two tables! They serve different purposes:
- `forward_returns` → Daily time series, simple calculations
- `ml_forward_returns` → ML training data with risk metrics

---

## Ranking System

### The 7 Quality Rankings

Each ranking is from 1 (best) to N (worst) where N = number of stocks analyzed.

#### 1. SP500 Outperformance Ranking (`beat_sp500_ranking`)
**Purpose**: Find consistent market beaters
**Key Metrics**:
- `years_beating_sp500` - How many of last 20 years beat S&P
- `sp500_weighted_score` - Recent years weighted higher
- `recent_5yr_beat_count` - Recent consistency
**Top Stocks**: Those beating S&P 500 most consistently with recent bias

#### 2. Excess Cash Flow Ranking (`excess_cash_flow_ranking`)
**Purpose**: Identify cash generation machines
**Key Metrics**:
- `fcf_yield` - Free cash flow / Market cap
- `fcf_margin` - FCF / Revenue  
- `fcf_growth_3yr` - 3-year FCF growth rate
- `fcf_consistency_score` - Stability of cash generation
**Top Stocks**: High, growing, consistent cash generators

#### 3. Fundamentals Trend Ranking (`fundamentals_ranking`)
**Purpose**: Find companies with improving fundamentals
**Key Metrics**:
- Price trends (10yr, 5yr, 1yr CAGR)
- Revenue growth and acceleration
- Margin expansion/contraction
- ROE trends
- Operating cash flow growth
**Top Stocks**: Accelerating growth, expanding margins, improving returns

#### 4. News Sentiment Ranking (`sentiment_ranking`)
**Purpose**: Capture market sentiment and catalysts
**Key Metrics**:
- `sentiment_score` - Composite sentiment (-100 to +100)
- `sentiment_momentum` - Change in sentiment
- `bull_bear_ratio` - Ratio of positive to negative articles
- Catalyst identification
**Top Stocks**: Positive, improving sentiment with catalysts

#### 5. Value Ranking (`value_ranking`)
**Purpose**: Find historically cheap stocks
**Key Metrics**:
- P/S, P/B, P/CF, P/E, EV/EBITDA percentiles vs 10-year history
- Z-scores from historical mean
- Sector and market relative value
**Top Stocks**: Trading at historical lows across multiple metrics

#### 6. Breakout Ranking (`breakout_ranking`)
**Purpose**: Identify technical momentum
**Key Metrics**:
- Price changes (3m, 1m, 1w)
- Volume surges and trends
- 52-week high proximity
- Relative strength vs market/sector
- Breakout confirmation signals
**Top Stocks**: Breaking out on volume with momentum

#### 7. Growth Ranking (`growth_ranking`)
**Purpose**: Find long-term compounders
**Key Metrics**:
- Lifetime returns and CAGR
- Years outperforming S&P 500
- Revenue/earnings/FCF growth consistency
- Trend strength (R-squared)
- Risk-adjusted returns (Sharpe/Sortino)
**Top Stocks**: Consistent long-term outperformers

### Composite Scoring

```python
# Base composite (simple average)
base_composite = (rank1 + rank2 + ... + rank7) / 7

# Confidence adjustment
confidence_factor = data_quality_score * recency_weight
composite_quality_score = base_composite * confidence_factor

# Tier assignment
if composite_score <= 100: tier = "Elite"
elif composite_score <= 250: tier = "Premium"  
elif composite_score <= 500: tier = "Quality"
elif composite_score <= 1000: tier = "Standard"
else: tier = "Below"
```

### Strategy Selection Flags

Stocks are flagged for specific strategies:
- `is_sp500_beater` - Consistently beats market
- `is_cash_generator` - Top quartile FCF yield
- `is_fundamental_grower` - Positive trend momentum
- `is_deep_value_star` - Multiple value metrics at extremes
- `is_momentum_breakout` - Technical breakout confirmed
- `is_growth_champion` - Long-term compounder
- `is_all_star` - Elite across multiple rankings

---

## Pipeline Structure

### Daily Pipeline (`run_daily_pipeline.py`)
**Frequency**: Every market day after close
**Duration**: ~30-60 minutes

```python
Daily Scripts (in order):
1. fetch_symbol_universe.py    # Update stock list
2. fetch_sp500_history.py      # Benchmark data
3. fetch_prices.py             # Stock prices (biggest task)
4. fetch_technical_indicators.py # Technical analysis
5. fetch_dividend_history.py   # Dividend data
```

### Weekly Pipeline (`run_weekly_pipeline.py`)
**Frequency**: Weekly (weekends preferred)
**Duration**: 2-4 hours

```python
Weekly Scripts (in order):
1. fetch_company_overview.py   # Company fundamentals
2. fetch_fundamentals.py       # Financial statements
3. fetch_options.py            # Options data
4. fetch_news_sentiment.py     # News analysis
5. compute_forward_returns.py  # Simple returns → forward_returns table
6. calculate_quality_rankings.py # Generate 7 rankings
7. calculate_historical_rankings.py # Historical analysis → ml_forward_returns
8. calculate_forward_returns.py # ML returns → ml_forward_returns table
9. ml_strategy_framework.py    # Train ML models
10. generate_trading_signals.py # Generate signals
```

### Full Pipeline (`run_eod_full_pipeline.py`)
Runs both daily and weekly in sequence.

---

## ML Integration

### Two Forward Returns Systems

#### System 1: Simple Forward Returns
**Table**: `forward_returns`
**Script**: `compute_forward_returns.py`
**Structure**: Wide format (one row per date)
```sql
symbol | as_of_date | return_1m | return_3m | return_6m | return_12m
AAPL   | 2024-01-01 | 5.2%      | 12.1%     | 18.3%     | 35.2%
```
**Use Case**: Time series analysis, quick lookups

#### System 2: ML Forward Returns
**Table**: `ml_forward_returns`
**Script**: `calculate_forward_returns.py`
**Structure**: Long format (one row per horizon)
```sql
symbol | ranking_date | horizon_weeks | forward_return | forward_excess_return | volatility | max_drawdown
AAPL   | 2024-01-01  | 4            | 5.2%          | 2.1%                  | 18.5%     | -3.2%
AAPL   | 2024-01-01  | 12           | 12.1%         | 4.3%                  | 21.2%     | -5.1%
```
**Use Case**: ML training with risk-adjusted metrics

### ML Model Pipeline

```python
1. Feature Engineering (from multiple tables)
   ↓
2. Target Variables (from ml_forward_returns)
   ↓
3. Model Training (XGBoost, LSTM, etc.)
   ↓
4. Predictions → ml_predictions table
   ↓
5. Signal Generation → strategy_signals table
   ↓
6. Portfolio Construction
```

### Key ML Features

**Ranking Features**:
- All 7 ranking scores
- Ranking changes (momentum)
- Cross-ranking correlations

**Fundamental Features**:
- Growth rates
- Profitability metrics
- Valuation ratios
- Quality scores

**Technical Features**:
- Price momentum
- Volume patterns
- Volatility regimes
- Support/resistance levels

**Sentiment Features**:
- News sentiment scores
- Sentiment momentum
- Article volume
- Catalyst events

---

## Operational Guide

### Initial Setup

```bash
# 1. Create .env file with credentials
POSTGRES_URL=postgresql://...
ALPHA_VANTAGE_API_KEY=...
FMP_API_KEY=...  # Optional

# 2. Create database schema
python setup_schema_chunked.py  # Quick setup
# OR
python setup_schema.py  # Full schema (may timeout)

# 3. Initial data load
python fetch_symbol_universe.py
python fetch_sp500_history.py
python run_daily_pipeline.py
python run_weekly_pipeline.py
```

### Daily Operations

```bash
# Option 1: Run everything
python run_eod_full_pipeline.py

# Option 2: Run separately
python run_daily_pipeline.py   # After market close
python run_weekly_pipeline.py  # Weekly (weekend)

# Option 3: Specific updates
python fetch_prices.py --symbols "AAPL,MSFT,GOOGL"
python calculate_quality_rankings.py
```

### Monitoring

```python
# Check data freshness
SELECT MAX(fetched_at) FROM stock_prices;
SELECT MAX(ranking_date) FROM stock_quality_rankings;

# Check pipeline health  
SELECT * FROM script_health_monitor ORDER BY last_run DESC;

# View top stocks
SELECT * FROM v_elite_stocks;  # Top 50 in all categories
SELECT * FROM v_quality_stocks WHERE quality_tier = 'Elite';
```

### Troubleshooting

**Issue**: Pipeline timeout
```bash
# Use chunked approach
python fetch_prices.py --batch_size 100
```

**Issue**: Table conflict errors
```bash
# Check table structure
python -c "from sqlalchemy import inspect; ..."

# Recreate ML tables
python create_ml_forward_returns_table.py
```

**Issue**: Missing data
```bash
# Backfill specific date range
python fetch_prices.py --start_date 2024-01-01 --end_date 2024-01-31
```

---

## Technical Decisions

### Why Two Forward Returns Tables?

**Problem**: Two incompatible data structures needed:
1. Wide format for time series (return_1m, return_3m columns)
2. Long format for ML (one row per horizon with risk metrics)

**Solution**: Separate tables optimized for each use case
- `forward_returns` - Simple, fast queries
- `ml_forward_returns` - Rich ML features

**Alternative Considered**: Single unified table
- Rejected due to: Too many columns, slower queries, complex updates

### Why 7 Rankings Instead of One?

**Reasoning**: 
- Different strategies work in different market regimes
- Investors have different preferences (value vs growth)
- Composite score identifies stocks good across multiple dimensions
- Allows for strategy rotation based on market conditions

### Why Rankings Instead of Raw Scores?

**Benefits**:
- Comparable across different metrics
- Robust to outliers
- Easy to understand (1 = best)
- Stable over time

### API Rate Limiting Strategy

**Alpha Vantage Premium**: 600 calls/minute
- Rate limiter: 595 calls/min (safety margin)
- Batch processing with retries
- Incremental updates to minimize API calls

### Schema Design Principles

1. **Normalization**: Separate tables for different data types
2. **Performance**: Extensive indexing on common query patterns
3. **Flexibility**: JSON fields for variable data
4. **Audit Trail**: fetched_at/created_at on all tables
5. **Idempotency**: UPSERT operations prevent duplicates

---

## Performance Optimization

### Database Indexes

**Critical Indexes**:
```sql
-- Most important for performance
idx_stock_prices_symbol_date
idx_quality_rankings_date
idx_forward_returns_symbol_date
idx_ml_forward_returns_symbol_date
```

### Materialized Views

```sql
mv_latest_prices       -- Current prices
mv_latest_fundamentals -- Recent fundamentals
mv_active_options      -- Active options
```

### Query Optimization Tips

1. **Use date filters**: Always filter by date first
2. **Limit symbols**: Process in batches when possible
3. **Use EXISTS**: Faster than COUNT for existence checks
4. **Avoid SELECT ***: Specify needed columns
5. **Use UPSERT**: ON CONFLICT DO UPDATE prevents duplicates

---

## Future Enhancements

### Planned Features

1. **Real-time Integration**
   - WebSocket feeds for prices
   - Intraday signal generation

2. **Advanced ML**
   - Ensemble models
   - Deep learning (LSTM/Transformer)
   - Reinforcement learning for portfolio optimization

3. **Risk Management**
   - VaR calculations
   - Portfolio optimization
   - Hedging strategies

4. **Backtesting Framework**
   - Walk-forward analysis
   - Monte Carlo simulations
   - Transaction cost modeling

5. **UI/Dashboard**
   - Web interface for monitoring
   - Real-time P&L tracking
   - Alert system

### Scaling Considerations

**Current Capacity**: ~5,000 stocks
**Bottlenecks**: API rate limits, price fetching time
**Solutions**: 
- Parallel processing
- Multiple API keys
- Redis caching layer
- Dedicated database server

---

## Appendix: Key SQL Queries

### Find Elite Stocks
```sql
SELECT * FROM stock_quality_rankings
WHERE ranking_date = (SELECT MAX(ranking_date) FROM stock_quality_rankings)
  AND composite_quality_score <= 100
ORDER BY composite_quality_score;
```

### Recent Breakouts
```sql
SELECT * FROM stock_quality_rankings
WHERE ranking_date = (SELECT MAX(ranking_date) FROM stock_quality_rankings)
  AND breakout_ranking <= 50
  AND is_volume_confirmed = true
ORDER BY breakout_ranking;
```

### Cash Generators
```sql
SELECT symbol, fcf_yield, fcf_growth_3yr, excess_cash_flow_ranking
FROM stock_quality_rankings
WHERE ranking_date = (SELECT MAX(ranking_date) FROM stock_quality_rankings)
  AND fcf_yield > 0.10  -- 10%+ FCF yield
ORDER BY excess_cash_flow_ranking;
```

---

## Contact & Support

**Created by**: ACIS Development Team
**Last Updated**: November 2024
**Version**: 2.0

For questions or issues, check:
1. This documentation
2. Individual script docstrings
3. Error logs in `logs/` directory

---

END OF DOCUMENTATION