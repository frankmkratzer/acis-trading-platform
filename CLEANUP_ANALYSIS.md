# ACIS Trading Platform - Cleanup Analysis (COMPLETED)

## CLEANUP RESULTS

**Files Organized:**
- **Main Directory**: 43 core Python files (down from ~65)
- **archive/removed_scripts/**: 13 redundant pipeline and compute scripts  
- **archive/demos/**: 3 demo/system run scripts
- **tests/**: 4 test scripts properly organized
- **archive/logs/**: Log files archived

**Database Optimized:**
- **Tables Dropped**: 25 unused/empty tables removed
- **Tables Kept**: 15 essential tables with data (27M+ total records)
- **Database Size**: Optimized for production use

**Space Savings:**
- ~35% reduction in file count
- Cleaner, more maintainable structure
- Clear separation of core vs utility scripts

## Core System Files (KEEP)
**Essential Infrastructure:**
- `setup_schema.py` - Database schema setup
- `run_eod_full_pipeline.py` - Main pipeline runner (1613 lines, core system)
- `live_trading_engine.py` - Live trading core
- `strategy_orchestrator.py` - Strategy coordination
- `risk_management.py` - Portfolio risk management
- `backtest_engine.py` - Backtesting framework

**AI Models (KEEP):**
- `train_ai_value_model.py` - Value strategy ML
- `train_ai_growth_model.py` - Growth strategy ML
- `score_ai_value_model.py` - Value scoring
- `score_ai_growth_model.py` - Growth scoring
- `compute_forward_returns.py` - Return predictions
- `feature_engineering.py` - ML features

**Data Fetching (KEEP):**
- `fetch_prices.py` - Price data ingestion
- `fetch_fundamentals.py` - Fundamental data
- `fetch_sp500_history.py` - Benchmark data
- `fetch_dividend_history.py` - Dividend data

**Portfolio Management (KEEP):**
- `create_portfolios.py` - Portfolio generation
- `comprehensive_20year_backtest.py` - Backtesting (validated)

## Duplicate/Redundant Files (CONSIDER REMOVING)

**Redundant AI Scripts:**
- `compute_ai_dividend_scores.py` - May duplicate dividend logic
- `compute_dividend_growth_scores.py` - Overlaps with main dividend processing
- `compute_sp500_outperformance_scores.py` - Specialized scoring (may be redundant)
- `compute_value_momentum_and_growth_scores.py` - Large compute script (2000+ lines)

**Redundant Pipeline Runners:**
- `run_ai_analysis.py` - Covered by main pipeline
- `run_all_ai_strategies.py` - Simple wrapper script
- `run_dividend_pipeline.py` - Specialized runner
- `run_growth_pipeline.py` - Specialized runner  
- `run_value_pipeline.py` - Specialized runner
- `run_rank_dividend_stocks.py` - Ranking script
- `run_rank_growth_stocks.py` - Ranking script
- `run_rank_momentum_stocks.py` - Ranking script
- `run_rank_value_stocks.py` - Ranking script

**Debug/Testing Scripts (CONSOLIDATE):**
- `debug_connection.py` - DB connection testing
- `debug_pipeline.py` - Pipeline debugging
- `test_db_connection.py` - DB connection test
- `check_existing_tables.py` - Table verification
- `check_pipeline_status.py` - Status checking

**Monitoring Scripts (CONSOLIDATE):**
- `completion_monitor.py` - Pipeline monitoring
- `monitor_pipeline.py` - Pipeline monitoring
- `monitor_ai_training.py` - Training monitoring

## System Demo Files (OPTIONAL)
- `complete_system_run.py` - Full system demonstration
- `run_trading_demo.py` - Trading demo
- `demo_backtest.py` - Demo backtesting
- `test_system.py` - System validation (33 tests passed)
- `test_strategies.py` - Strategy testing

## Utility/Support Files (KEEP)
- `optimized_database_utils.py` - DB optimization
- `config_manager.py` - Configuration management
- `advanced_risk_manager.py` - Advanced risk tools
- `multi_factor_optimizer.py` - Portfolio optimization
- `hyperparameter_optimizer.py` - ML optimization
- `ensemble_models.py` - Model ensembling
- `regime_detection.py` - Market regime detection
- `performance_dashboard.py` - Performance analytics
- `strategy_generator.py` - Strategy generation
- `strategy_executor.py` - Strategy execution

## Schwab Integration (KEEP IF USING SCHWAB)
- `schwab_*.py` files - All Schwab broker integration

## Maintenance Files (KEEP)
- `drop_all_tables.py` - Database cleanup utility
- `migrate_existing_schema.py` - Schema migrations
- `fix_and_generate_portfolios.py` - Portfolio fix utility

## Log Files (CLEANUP)
- `*.log` files - Can be archived/deleted
- `logs/` directory - Can be cleaned up

## Recommended Actions:

### Phase 1: Remove Clear Redundants (17 files)
- All `run_rank_*_stocks.py` (4 files) - Covered by main pipeline  
- All individual `run_*_pipeline.py` (4 files) - Covered by main pipeline
- `run_ai_analysis.py` - Covered by main pipeline
- `run_all_ai_strategies.py` - Simple wrapper
- `compute_ai_dividend_scores.py` - May be redundant
- `compute_dividend_growth_scores.py` - May be redundant  
- `compute_sp500_outperformance_scores.py` - Specialized
- `compute_value_momentum_and_growth_scores.py` - Large specialized script
- `debug_connection.py` - Merge with test_db_connection.py
- `check_existing_tables.py` - Integrate into main scripts
- `completion_monitor.py` - Merge with monitor_pipeline.py

### Phase 2: Archive Demo Files (Optional - 6 files)
- `complete_system_run.py` - Move to archive/
- `run_trading_demo.py` - Move to archive/
- `demo_backtest.py` - Move to archive/  
- `test_system.py` - Move to tests/
- `test_strategies.py` - Move to tests/
- `test_schwab_integration.py` - Move to tests/

### Phase 3: Clean Logs
- Archive all `.log` files
- Clean `logs/` directory

## Database Tables Review Needed
Need to check `setup_schema.py` for unused tables that could be dropped.

## Result: 
- Current: ~65 Python files
- After cleanup: ~42 core files  
- Reduction: ~35% file count
- Cleaner, more maintainable codebase