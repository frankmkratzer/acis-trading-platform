# ACIS Trading Platform - Modular Architecture

## Overview
The ACIS (Algorithmic Capital Investment System) Trading Platform is a comprehensive stock analysis and ranking system that processes ~4,600 US stocks daily. The platform has been reorganized into a modular architecture for better maintainability and scalability.

## 🏗️ New Folder Structure

```
acis-trading-platform/
├── main.py                 # Main entry point for all operations
├── requirements.txt        # Python dependencies
│
├── data_fetch/            # Data acquisition module
│   ├── orchestrator.py    # Coordinates all data fetching
│   ├── base/              # Core utilities
│   │   ├── rate_limiter.py
│   │   └── fetch_manager.py
│   ├── market_data/       # Market data fetchers
│   │   ├── fetch_symbol_universe.py
│   │   ├── fetch_prices.py
│   │   ├── fetch_sp500_history.py
│   │   └── fetch_technical_indicators.py
│   └── fundamentals/      # Company fundamentals
│       ├── fetch_company_overview.py
│       ├── fetch_fundamentals.py
│       └── fetch_dividend_history.py
│
├── rankings/              # Ranking calculation system
│   ├── orchestrator.py   # Coordinates ranking pipeline
│   ├── base/              # Shared utilities
│   ├── populate/          # Data preparation (9 scripts)
│   │   ├── 01_initialize_table.py
│   │   ├── 02_price_returns.py
│   │   └── ... (through 09_technical_indicators.py)
│   ├── calculate/         # Ranking generation (8 scripts)
│   │   ├── 10_momentum_ranking.py
│   │   ├── 11_value_ranking.py
│   │   └── ... (through 17_composite_ranking.py)
│   └── validate/          # Quality checks
│       ├── data_quality.py
│       └── report_generator.py
│
├── ml_analysis/           # Machine learning module
│   ├── forward_returns/   # Return calculations
│   ├── strategies/        # ML models and signals
│   └── historical/        # Historical analysis
│
├── pipelines/             # High-level orchestrators
│   ├── run_daily_pipeline_v2.py
│   ├── run_weekly_pipeline.py
│   └── run_eod_full_pipeline.py
│
├── database/              # Database management
│   ├── setup_schema.py
│   ├── migrate_schema.py
│   └── data_validator.py
│
├── utils/                 # Shared utilities
│   ├── config_manager.py
│   ├── logging_config.py
│   └── batch_processor.py
│
├── tests/                 # Test suite
├── docs/                  # Documentation
└── logs/                  # Log files
```

## 🚀 Quick Start

### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export POSTGRES_URL=postgresql://user:pass@host:port/dbname
export ALPHA_VANTAGE_API_KEY=your_key_here
```

### Database Setup
```bash
# Initialize database schema
python main.py database --operation setup

# Or run migration if updating
python main.py database --operation migrate
```

## 📊 Daily Operations

### Using Main Orchestrator (Recommended)
```bash
# Run complete daily pipeline
python main.py --daily

# Run weekly pipeline (includes fundamentals)
python main.py --weekly

# Run specific components
python main.py fetch --mode daily
python main.py rankings --phase all
python main.py ml --type strategy
```

### Using Module Orchestrators
```bash
# Data fetching only
python data_fetch/orchestrator.py --mode daily

# Rankings calculation only
python rankings/orchestrator.py --phase all

# Custom pipeline
python pipelines/run_daily_pipeline_v2.py --phase all
```

## 🎯 Key Features

### 1. Data Fetching Module
- **Rate Limiting**: Respects Alpha Vantage 600 calls/min limit
- **Incremental Updates**: Only fetches changed data
- **Error Recovery**: Automatic retry with exponential backoff
- **Parallel Processing**: Batch processing for efficiency

### 2. Ranking System
- **7 Ranking Systems**:
  - Momentum (price performance)
  - Value (valuation metrics)
  - Growth (revenue/earnings growth)
  - Quality (profitability metrics)
  - Fundamentals (comprehensive)
  - Earnings Stability (consistency)
  - Size (market cap factor)
- **Composite Score**: Weighted combination of all rankings
- **Modular Design**: 19 focused scripts avoid timeouts

### 3. Machine Learning
- **Forward Returns**: Multiple time horizons (1M, 3M, 6M, 1Y)
- **Risk Metrics**: Volatility, Sharpe, maximum drawdown
- **ML Models**: Random Forest, XGBoost support
- **Signal Generation**: Buy/sell/hold signals

## 📈 Pipeline Dependencies

### Daily Pipeline Order:
1. **Data Fetch Phase**:
   - Symbol universe (foundation)
   - S&P 500 history (benchmark)
   - Stock prices (core data)
   - Technical indicators

2. **Rankings Phase**:
   - Populate metrics (9 scripts)
   - Calculate rankings (8 scripts)
   - Validate results

3. **ML Phase**:
   - Forward returns calculation
   - Strategy execution
   - Signal generation

### Weekly Pipeline (Additional):
- Company fundamentals
- Financial statements
- Dividend history
- News sentiment

## 🛠️ Advanced Usage

### Skip Completed Scripts
```bash
# Resume interrupted pipeline
python main.py pipeline --type daily --skip-completed

# Skip specific phases
python main.py pipeline --type daily --skip-fetch --skip-rankings
```

### Custom Script Execution
```bash
# Run specific data fetchers
python data_fetch/orchestrator.py --mode custom \
  --scripts market_data/fetch_prices.py fundamentals/fetch_dividends.py

# Run specific ranking phase
python rankings/orchestrator.py --phase calculate
```

### Validation and Reports
```bash
# Check data quality
python rankings/validate/data_quality.py

# Generate comprehensive reports
python rankings/validate/report_generator.py
```

## 📊 Database Schema

### Main Tables:
- `symbol_universe`: All tradeable US stocks
- `stock_prices`: Daily OHLCV data (15M+ records)
- `stock_quality_rankings`: 7 rankings + composite scores
- `company_overview`: Fundamentals and company info
- `forward_returns`: Wide format for time series
- `ml_forward_returns`: Long format for ML training

## 🔧 Configuration

### Environment Variables:
```bash
POSTGRES_URL=postgresql://user:pass@host:port/dbname
ALPHA_VANTAGE_API_KEY=your_key_here
```

### Key Settings (rankings/base/config.py):
- `BATCH_SIZE`: 500 records per batch
- `QUERY_TIMEOUT`: 120 seconds
- `API_RATE_LIMIT`: 595 calls/min (safety margin)

## 📝 Logging

Logs are stored in `logs/` directory:
- `daily_pipeline.log`: Daily operations
- `weekly_pipeline.log`: Weekly operations
- `fetch_*.log`: Individual fetcher logs
- `rankings_*.log`: Ranking calculation logs

## 🚨 Troubleshooting

### Common Issues:

1. **Database Timeout**:
   - Use modular scripts instead of monolithic ones
   - Adjust `BATCH_SIZE` in config

2. **API Rate Limits**:
   - Scripts handle automatically with delays
   - Check `alpha_vantage_rate_limiter.py` settings

3. **Pipeline Failures**:
   - Use `--skip-completed` to resume
   - Check logs for specific errors
   - Run individual modules to isolate issues

## 📈 Performance

- **Daily Pipeline**: ~2-3 hours (full)
- **Weekly Pipeline**: ~4-5 hours (with fundamentals)
- **Rankings Only**: ~30-45 minutes
- **Database Size**: ~5-10 GB for 1 year of data

## 🔄 Migration from Old Structure

The platform has been migrated from a flat structure with 30+ scripts in root to a modular architecture. Old scripts have been:
- Reorganized into logical folders
- Updated with improved error handling
- Enhanced with batch processing
- Optimized for large datasets (15M+ records)

## 📚 Documentation

See `docs/` folder for:
- `QUICK_REFERENCE_GUIDE.md`: Command reference
- `REORGANIZATION_PLAN.md`: Migration details
- `CLAUDE.md`: AI assistant instructions

## 🤝 Contributing

When adding new features:
1. Place scripts in appropriate module folder
2. Update module's orchestrator if needed
3. Add to main.py for unified access
4. Update this README

## 📄 License

Proprietary - ACIS Trading Platform