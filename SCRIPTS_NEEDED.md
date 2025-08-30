# Scripts Alignment with Three-Portfolio Strategy

## ✅ ESSENTIAL Scripts for Our Strategy

### Data Fetching (Keep All)
- `data_fetch/market_data/fetch_symbol_universe.py` - Stock list
- `data_fetch/market_data/fetch_stock_prices.py` - Daily prices
- `data_fetch/market_data/fetch_sp500_history.py` - Benchmark
- `data_fetch/fundamentals/fetch_fundamentals.py` - Financials
- `data_fetch/fundamentals/fetch_company_overview.py` - Company metrics
- `data_fetch/fundamentals/fetch_dividend_history.py` - Dividends

### Core Analysis (Our New Modules)
- `analysis/excess_cash_flow.py` ✅ - Primary quality metric
- `analysis/dividend_sustainability.py` ✅ - Dividend analysis
- `portfolios/portfolio_manager.py` ✅ - Three-portfolio system

### Portfolio Support
- `analysis/calculate_forward_returns.py` - Needed for performance
- `rankings/calculate/17_calculate_composite_rankings.py` - Could adapt for portfolio scores

## ❌ Scripts We DON'T Need

### ML-Related (Not using ML)
- All scripts in `ml_analysis/strategies/`
- `ml_analysis/train_models.py`
- `ml_analysis/generate_signals.py`
- `ml_analysis/backtest.py`

### Complex Rankings (Replaced by portfolio_scores)
- Most of `rankings/populate/` (01-09 scripts)
- Most of `rankings/calculate/` (10-16 scripts)
- These are replaced by our simpler three-score system

### Technical Analysis (Not in our strategy)
- `data_fetch/market_data/fetch_technical_indicators.py`
- Any momentum or technical pattern scripts

## 📝 New Scripts to Create

### Immediate Needs
1. **Portfolio Dashboard** (`portfolios/dashboard.py`)
   - Show current holdings
   - Performance vs benchmark
   - Quality score changes

2. **Rebalance Scheduler** (`portfolios/rebalance_scheduler.py`)
   - Automated quarterly/annual rebalancing
   - Email notifications
   - Pre-rebalance checks

3. **Performance Reporter** (`portfolios/performance_report.py`)
   - Monthly performance summaries
   - Attribution analysis
   - Risk metrics

### Future Enhancements
1. **Tax Optimizer** (`portfolios/tax_optimizer.py`)
   - Tax-loss harvesting
   - Wash sale avoidance
   - Tax-efficient rebalancing

2. **Risk Manager** (`portfolios/risk_manager.py`)
   - Position concentration alerts
   - Quality deterioration warnings
   - Sector exposure monitoring

## 🔄 Modified Data Flow

### OLD Complex Flow:
```
Symbol Universe → Prices → 19 Ranking Scripts → ML Features → 
ML Models → Signals → Backtest → Trading
```

### NEW Simplified Flow:
```
Symbol Universe → Prices + Fundamentals → 
Excess CF + Dividends → Portfolio Scores → 
Three Portfolios → Rebalance → Track Performance
```

## 📊 Database Tables Usage

### Tables We Use:
| Table | Purpose | Update Frequency |
|-------|---------|------------------|
| symbol_universe | Stock list | Daily |
| stock_prices | Price data | Daily |
| fundamentals | Financials | Quarterly |
| company_fundamentals_overview | Metrics | Weekly |
| dividend_history | Dividends | Daily |
| sp500_history | Benchmark | Daily |
| excess_cash_flow_metrics | Quality | Weekly |
| dividend_sustainability_metrics | Dividends | Monthly |
| portfolio_scores | Scoring | Weekly |
| portfolio_holdings | Positions | As needed |
| portfolio_rebalances | History | Quarterly/Annual |
| portfolio_performance | Returns | Daily |

### Tables to Remove:
- ml_models
- ml_predictions
- ml_features
- ml_forward_returns
- trading_signals
- backtest_results
- technical_indicators_advanced

## 🚀 Simplified Pipeline

### Daily Pipeline:
```bash
# Just update prices and track performance
python data_fetch/market_data/fetch_stock_prices.py
python portfolios/update_performance.py
```

### Weekly Pipeline:
```bash
# Update fundamentals and scores
python data_fetch/fundamentals/fetch_company_overview.py
python analysis/excess_cash_flow.py
python portfolios/portfolio_manager.py --update-scores
```

### Quarterly Pipeline (Value/Growth):
```bash
# Full rebalance
python portfolios/portfolio_manager.py --rebalance VALUE
python portfolios/portfolio_manager.py --rebalance GROWTH
```

### Annual Pipeline (Dividend):
```bash
# Dividend portfolio rebalance
python analysis/dividend_sustainability.py
python portfolios/portfolio_manager.py --rebalance DIVIDEND
```

## 💡 Key Simplifications

1. **No ML Complexity** - Rule-based scoring is transparent and reliable
2. **Three Clear Strategies** - Each portfolio has a specific objective
3. **Quality First** - Excess cash flow gates all portfolios
4. **Systematic Rebalancing** - Quarterly/annual on schedule
5. **Simple Equal Weighting** - No complex position sizing

## 📈 Implementation Priority

### Phase 1 (Now):
1. ✅ Excess cash flow calculator
2. ✅ Dividend sustainability analyzer
3. ✅ Portfolio manager
4. ✅ Database schema updates

### Phase 2 (Next):
1. Fetch missing fundamentals data
2. Calculate initial scores
3. Select first portfolios
4. Set up performance tracking

### Phase 3 (Future):
1. Automate rebalancing
2. Add reporting dashboard
3. Implement risk monitoring
4. Add tax optimization

This simplified approach focuses on your core investment philosophy while eliminating unnecessary complexity from ML and excessive technical indicators.