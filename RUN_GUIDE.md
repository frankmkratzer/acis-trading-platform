# ACIS Platform Run Guide
**Last Updated**: December 2024  
**Version**: TOP 1% Strategy Implementation v2.0

## 🚀 Quick Start

### First Time Setup
```bash
# 1. Verify environment variables in .env
POSTGRES_URL=postgresql://user:pass@host:port/dbname
ALPHA_VANTAGE_API_KEY=your_key_here

# 2. Initialize database
python master_control.py --setup

# 3. Verify all tables are created
python master_control.py --verify

# 4. Run initial data population (takes 6-8 hours)
python master_control.py --weekly

# 5. Check portfolio status
python master_control.py --status
```

## 📅 Daily Operations

### Option 1: Full TOP 1% Pipeline (Recommended)
**Runtime**: 4-5 hours  
**Best Time**: 5:00 AM - 10:00 AM (before market open)

```bash
python master_control.py --daily
```

**What it does:**
1. Updates stock prices and S&P 500 benchmark
2. Fetches insider transactions
3. Updates institutional holdings
4. Calculates all fundamental scores (Piotroski, Altman, Beneish)
5. Updates risk metrics (Sharpe, Sortino, VaR)
6. Detects technical breakouts
7. Updates sector rotation matrix
8. Calculates Kelly Criterion position sizes
9. Generates master composite scores
10. Updates portfolio selections

### Option 2: Basic Daily Pipeline
**Runtime**: 2 hours  
**Use When**: Time constrained or API limits reached

```bash
python master_control.py --basic-daily
```

**What it does:**
1. Updates stock prices only
2. Recalculates basic metrics
3. Updates portfolio scores

## 📅 Weekly Operations

### Option 1: Full TOP 1% Weekly Pipeline
**Runtime**: 6-8 hours  
**Best Day**: Saturday or Sunday

```bash
python master_control.py --weekly

# With optional features enabled (adds 2-3 hours):
ENABLE_ML=true ENABLE_OPTIMIZATION=true python master_control.py --weekly
```

**What it does:**
1. Everything from daily pipeline PLUS:
2. Updates complete fundamentals (income, balance, cash flow)
3. Updates dividend history
4. Updates earnings estimates
5. Runs strategy backtests
6. Performs walk-forward optimization (if enabled)
7. Calculates performance attribution
8. Retrains ML models (if enabled)

### Option 2: Basic Weekly Pipeline
**Runtime**: 3-4 hours

```bash
python master_control.py --basic-weekly
```

## 🔍 Monitoring & Analysis

### Check Portfolio Status
```bash
python master_control.py --status
```

### Run Specific Analysis
```bash
# Fundamental Scores
python master_control.py --analyze piotroski    # F-Score calculation
python master_control.py --analyze altman       # Z-Score calculation
python master_control.py --analyze beneish      # M-Score calculation

# Smart Money
python master_control.py --analyze insider      # Insider transactions
python master_control.py --analyze institutional # Institutional holdings

# Risk & Optimization
python master_control.py --analyze risk         # Risk metrics
python master_control.py --analyze kelly        # Position sizing
python master_control.py --analyze sector       # Sector rotation

# Performance
python master_control.py --analyze backtest     # Strategy backtesting
python master_control.py --analyze attribution  # Performance attribution
python master_control.py --analyze optimize     # Walk-forward optimization

# Machine Learning
python master_control.py --analyze ml          # ML predictions
```

### Database Verification
```bash
# Check all tables are present
python database/verify_tables_enhanced.py

# Basic verification
python master_control.py --verify
```

## 📊 Understanding the Outputs

### Portfolio Files Location
```
acis-trading-platform/
├── logs/                    # Execution logs
│   ├── daily_pipeline.log
│   ├── weekly_pipeline.log
│   └── [script_name].log
├── outputs/                 # Analysis results (if configured)
│   ├── portfolios/
│   ├── backtests/
│   └── reports/
```

### Database Tables to Monitor

**Primary Portfolio Tables:**
- `portfolio_holdings` - Current portfolio positions
- `portfolio_scores` - Stock scores for each portfolio
- `portfolio_performance` - Historical performance
- `master_scores` - Composite scoring for all stocks

**Key Analytics Tables:**
- `piotroski_scores` - F-Score (8-9 = Strong Buy)
- `altman_zscores` - Bankruptcy risk (Z > 3 = Safe)
- `beneish_mscores` - Manipulation risk (M < -2.22 = Clean)
- `kelly_criterion` - Optimal position sizes
- `risk_metrics` - Sharpe, Sortino, Beta, VaR

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### 1. Pipeline Timeout
**Problem**: Script exceeds timeout limit  
**Solution**: 
```bash
# Use basic mode instead
python master_control.py --basic-daily

# Or run specific components
python master_control.py --analyze piotroski
python master_control.py --analyze risk
```

#### 2. API Rate Limit
**Problem**: Alpha Vantage 600 calls/min exceeded  
**Solution**: Built-in rate limiter handles this, but if persistent:
```bash
# Reduce batch size in scripts
# Wait 1 minute between runs
# Use basic pipeline mode
```

#### 3. Missing Tables
**Problem**: Database tables not found  
**Solution**:
```bash
# Recreate all tables
python master_control.py --setup

# Verify
python master_control.py --verify
```

#### 4. Memory Issues
**Problem**: Out of memory errors  
**Solution**:
```bash
# Run components separately
python analysis/calculate_piotroski_fscore.py
python analysis/calculate_risk_metrics.py

# Reduce batch sizes in scripts
```

#### 5. Import Errors
**Problem**: Module not found errors  
**Solution**: Check sys.path.append() in affected script
```python
# Should be at top of every script
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
```

## 📈 Performance Monitoring

### Daily Metrics to Track
```sql
-- Check latest portfolio update
SELECT portfolio_type, COUNT(*), MAX(last_updated) 
FROM portfolio_holdings 
WHERE is_current = true 
GROUP BY portfolio_type;

-- Top stocks by composite score
SELECT symbol, composite_score, value_score, growth_score, dividend_score
FROM master_scores
ORDER BY composite_score DESC
LIMIT 10;

-- Check for data freshness
SELECT MAX(trade_date) FROM stock_prices;
SELECT MAX(fiscal_date_ending) FROM fundamentals;
```

### Weekly Performance Review
```bash
# Run attribution analysis
python master_control.py --analyze attribution

# Review backtest results
python master_control.py --analyze backtest

# Check risk metrics
python master_control.py --analyze risk
```

## 🤖 Automated Trading with Schwab API

### Initial Setup (One-time)
```bash
# 1. Create trading database tables
python database/setup_trading_tables.py

# 2. Set up environment variables in .env
SCHWAB_APP_KEY=your_app_key
SCHWAB_APP_SECRET=your_app_secret
SCHWAB_REDIRECT_URI=https://localhost:8443/callback
TRADING_ENCRYPTION_KEY=your_32_byte_key_base64

# 3. Onboard a client (example)
python trading/client_onboarding.py

# 4. Link Schwab account
python trading/schwab_account_setup.py --client-id 1
```

### Daily Automated Trading Operations
```bash
# Complete automated trading workflow (runs after market close)
# 1. Run ACIS analysis (4-5 hours)
python master_control.py --daily

# 2. Generate trading signals from portfolio recommendations
python trading/automated_trading_manager.py --generate-signals

# 3. Execute trades for all active accounts (run before market open)
python trading/automated_trading_manager.py --mode production --process-trades

# 4. Update account balances and holdings
python trading/update_account_balances.py

# 5. Send daily reports to clients
python trading/daily_reports.py
```

### Paper Trading Mode (Testing)
```bash
# Test trading logic without real money
python trading/automated_trading_manager.py --mode paper --process-trades

# View paper trading results
python trading/view_paper_trades.py
```

### Monitoring Automated Trading
```sql
-- Check active trading accounts
SELECT c.email, ta.automated_trading_active, ta.last_sync_at
FROM clients c
JOIN trading_accounts ta ON c.client_id = ta.client_id
WHERE ta.automated_trading_active = true;

-- View today's executed trades
SELECT * FROM trade_execution_log
WHERE DATE(order_placed_at) = CURRENT_DATE
ORDER BY order_placed_at DESC;

-- Check current holdings across all accounts
SELECT ta.schwab_account_number, ch.symbol, ch.quantity, ch.market_value
FROM current_holdings ch
JOIN trading_accounts ta ON ch.account_id = ta.account_id
ORDER BY ta.account_id, ch.market_value DESC;

-- Monitor trading signals queue
SELECT * FROM trading_signals_queue
WHERE processed = false
ORDER BY acis_score DESC;
```

### Trading Safety Checks
- Maximum 10% per position (configurable per client)
- Minimum $1,000 cash balance maintained
- Maximum 20 positions per portfolio
- Optional trailing stop-losses (15% default)
- Paper trading mode for testing

### Troubleshooting Trading Issues

#### 1. OAuth Token Expired
**Problem**: Schwab API returns 401 Unauthorized
**Solution**: 
```bash
# Refresh tokens for all accounts
python trading/refresh_all_tokens.py
```

#### 2. Trade Execution Failed
**Problem**: Order rejected by Schwab
**Solution**: Check `trade_execution_log` for error messages:
```sql
SELECT * FROM trade_execution_log 
WHERE order_status = 'REJECTED' 
  AND DATE(created_at) = CURRENT_DATE;
```

#### 3. Insufficient Funds
**Problem**: Cannot execute buy orders
**Solution**: Check account balances and adjust position sizes:
```sql
SELECT account_id, cash_balance, min_cash_balance 
FROM account_balances ab
JOIN trading_accounts ta ON ab.account_id = ta.account_id
WHERE DATE(snapshot_date) = CURRENT_DATE;
```

## 🔄 Maintenance Schedule

### Daily (Weekdays)
- [ ] Run daily pipeline at 5 AM
- [ ] Generate trading signals at 10 AM
- [ ] Execute trades at 9:25 AM (before market open)
- [ ] Update account balances at 4:30 PM (after market close)
- [ ] Check logs for errors
- [ ] Verify portfolio status
- [ ] Monitor any position alerts
- [ ] Review executed trades

### Weekly (Weekends)
- [ ] Run full weekly pipeline
- [ ] Review backtest performance
- [ ] Check performance attribution
- [ ] Review sector rotation recommendations
- [ ] Clean old log files (>30 days)

### Monthly
- [ ] Retrain ML models (set ENABLE_ML=true)
- [ ] Run walk-forward optimization
- [ ] Database vacuum/analyze
- [ ] Review and archive old data
- [ ] Update this RUN_GUIDE.md if needed

### Quarterly
- [ ] Full system performance review
- [ ] Strategy parameter tuning
- [ ] Database optimization
- [ ] Review API usage and costs

## 🚨 Critical Warnings

1. **NEVER** modify database schema outside of `database/setup_schema.py`
2. **NEVER** run multiple pipelines simultaneously (database locks)
3. **ALWAYS** check logs after pipeline runs
4. **ALWAYS** use DatabaseConnectionManager for DB operations
5. **ALWAYS** update this guide when changing operational procedures

## 📞 Support & Resources

### Log Files
- Location: `acis-trading-platform/logs/`
- Retention: 30 days recommended
- Format: `[timestamp] [level] [module] message`

### Configuration Files
- `.env` - Environment variables (API keys, database)
- `database/setup_schema.py` - Database schema definition
- `utils/config_manager.py` - System configuration

### Key Scripts to Understand
1. `master_control.py` - Main control interface
2. `pipelines/run_daily_pipeline_top1pct.py` - Daily operations
3. `pipelines/run_weekly_pipeline_top1pct.py` - Weekly operations
4. `analysis/calculate_master_scores.py` - Scoring system
5. `database/verify_tables_enhanced.py` - Database verification

## 📝 Change Log

### Version 2.1 (December 2024)
- Added automated trading system with Schwab API integration
- Created 7 new database tables for client and trade management
- Implemented automated_trading_manager.py for trade execution
- Added paper trading mode for testing
- Integrated encrypted credential storage
- Added position sizing and risk management controls

### Version 2.0 (December 2024)
- Implemented all 15 TOP 1% strategy enhancements
- Added master_control.py unified interface
- Created comprehensive pipeline system
- Added ML predictions and backtesting

### Update Checklist
When making changes that affect operations:
- [ ] Update this RUN_GUIDE.md
- [ ] Update CLAUDE.md if architectural
- [ ] Test changes in development first
- [ ] Document in change log above
- [ ] Notify team of operational changes

---
**Remember**: This guide is the source of truth for operating the ACIS platform. Keep it updated!
