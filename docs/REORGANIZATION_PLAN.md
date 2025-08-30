# Script Reorganization Plan

## Current Structure Analysis
The project has ~30+ scripts in the root directory, making it hard to navigate and maintain.

## Proposed Folder Structure

```
acis-trading-platform/
│
├── data_fetch/          # All data fetching scripts
│   ├── __init__.py
│   ├── orchestrator.py  # Main data fetch coordinator
│   ├── base/
│   │   ├── __init__.py
│   │   ├── rate_limiter.py (from alpha_vantage_rate_limiter.py)
│   │   └── fetch_manager.py (from incremental_fetch_manager.py)
│   ├── market_data/
│   │   ├── fetch_symbol_universe.py
│   │   ├── fetch_prices.py
│   │   ├── fetch_sp500_history.py
│   │   └── fetch_technical_indicators.py
│   ├── fundamentals/
│   │   ├── fetch_company_overview.py
│   │   ├── fetch_fundamentals.py
│   │   ├── fetch_dividend_history.py
│   │   └── fetch_news_sentiment.py (if exists)
│   └── options/
│       └── fetch_options.py (if exists)
│
├── rankings/            # Already organized ✓
│   ├── populate/
│   ├── calculate/
│   ├── validate/
│   └── orchestrator.py
│
├── ml_analysis/         # Machine learning and analysis
│   ├── __init__.py
│   ├── orchestrator.py
│   ├── forward_returns/
│   │   ├── compute_forward_returns.py
│   │   ├── calculate_forward_returns.py
│   │   └── create_ml_forward_returns_table.py
│   ├── strategies/
│   │   ├── ml_strategy_framework.py
│   │   └── generate_trading_signals.py
│   └── historical/
│       ├── calculate_historical_rankings.py
│       └── calculate_point_in_time_rankings.py
│
├── pipelines/           # Pipeline orchestrators
│   ├── __init__.py
│   ├── run_daily_pipeline.py
│   ├── run_weekly_pipeline.py
│   ├── run_eod_full_pipeline.py
│   └── run_weekly_rankings.py
│
├── database/            # Database management
│   ├── __init__.py
│   ├── setup_schema.py
│   ├── migrate_schema.py
│   ├── db_connection_manager.py
│   └── data_validator.py
│
├── utils/               # Utility modules
│   ├── __init__.py
│   ├── config_manager.py
│   ├── logging_config.py
│   ├── batch_processor.py
│   ├── reliability_manager.py
│   ├── smart_scheduler.py
│   └── monitoring_dashboard.py
│
├── tests/               # Test scripts
│   ├── test_schema_syntax.py
│   ├── test_quality_rankings_save.py
│   └── test_run.log
│
├── docs/                # Documentation
│   ├── QUICK_REFERENCE_GUIDE.md
│   └── SAVE_TO_DATABASE_FIX.md
│
├── logs/                # Already exists ✓
│
└── requirements.txt     # Keep in root

## Scripts to Remove (Obsolete/Redundant)

1. **populate_rankings_basic.py** - Replaced by rankings/populate/ scripts
2. **fix_revenue_growth.py** - Fixed in rankings/populate/04_revenue_growth.py
3. **calculate_quality_rankings.py** - Replaced by entire rankings/ system
4. **calculate_quality_rankings_complete.py** - Replaced by rankings/ system
5. **calculate_quality_rankings_optimized.py** - Replaced by rankings/ system
6. **calculate_quality_rankings_simplified.py** - Replaced by rankings/ system
7. **calculate_quality_rankings_worldclass.py** - Replaced by rankings/ system
8. **test_run.log** - Log file, not needed
9. **NUL** - Windows null file, can be deleted

## Migration Benefits

1. **Better Organization**: Related scripts grouped together
2. **Easier Navigation**: Clear folder structure
3. **Reduced Clutter**: Root directory only has essential files
4. **Modular Design**: Each folder can have its own orchestrator
5. **Cleaner Imports**: Organized module structure
6. **Easier Testing**: Test files separated
7. **Better Documentation**: Docs in dedicated folder

## Implementation Order

1. Create folder structure
2. Move data fetching scripts
3. Move ML/analysis scripts
4. Move pipeline scripts
5. Move database scripts
6. Move utility scripts
7. Update all imports
8. Remove obsolete scripts
9. Test all pipelines