# backend/models.py
# SQLAlchemy models for ACIS Trading Platform

from sqlalchemy import Column, Integer, String, Numeric, Date, DateTime, Boolean, Text, BigInteger
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from .database import Base
from datetime import datetime


class SymbolUniverse(Base):
    __tablename__ = "symbol_universe"

    symbol = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    exchange = Column(String, nullable=False)
    security_type = Column(String, nullable=False)
    is_etf = Column(Boolean, nullable=False)
    listed_date = Column(Date)
    sector = Column(String)
    industry = Column(String)
    market_cap = Column(Numeric)
    currency = Column(String)
    country = Column(String)
    dividend_yield = Column(Numeric)
    pe_ratio = Column(Numeric)
    peg_ratio = Column(Numeric)
    week_52_high = Column(Numeric)
    week_52_low = Column(Numeric)
    fetched_at = Column(DateTime, nullable=False)


class StockEODDaily(Base):
    __tablename__ = "stock_eod_daily"

    symbol = Column(String, primary_key=True)
    trade_date = Column(Date, primary_key=True)
    open = Column(Numeric)
    high = Column(Numeric)
    low = Column(Numeric)
    close = Column(Numeric)
    adjusted_close = Column(Numeric)
    volume = Column(BigInteger)
    dividend_amount = Column(Numeric)
    split_coefficient = Column(Numeric)
    fetched_at = Column(DateTime)


class StockIntraday(Base):
    __tablename__ = "stock_intraday"

    symbol = Column(String, primary_key=True)
    timestamp = Column(DateTime, primary_key=True)
    open = Column(Numeric)
    high = Column(Numeric)
    low = Column(Numeric)
    close = Column(Numeric)
    volume = Column(BigInteger)
    vwap = Column(Numeric)
    fetched_at = Column(DateTime, default=func.now())


class TradingOrders(Base):
    __tablename__ = "trading_orders"

    order_id = Column(String, primary_key=True)
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False)
    quantity = Column(Integer, nullable=False)
    order_type = Column(String, nullable=False)
    limit_price = Column(Numeric)
    stop_price = Column(Numeric)
    status = Column(String, nullable=False)
    filled_quantity = Column(Integer, default=0)
    avg_fill_price = Column(Numeric)
    submitted_at = Column(DateTime)
    filled_at = Column(DateTime)
    commission = Column(Numeric)
    created_at = Column(DateTime, default=func.now())

    # Enhanced fields for live trading
    strategy = Column(String)
    portfolio_id = Column(String)
    parent_order_id = Column(String)
    time_in_force = Column(String, default='DAY')
    broker_order_id = Column(String)
    last_updated = Column(DateTime, default=func.now())
    error_message = Column(Text)
    tags = Column(JSONB)


class TradeExecutions(Base):
    __tablename__ = "trade_executions"

    execution_id = Column(String, primary_key=True)
    order_id = Column(String, nullable=False)
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False)
    quantity = Column(Integer, nullable=False)
    price = Column(Numeric, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    venue = Column(String)
    execution_fees = Column(Numeric, default=0)
    liquidity_flag = Column(String)  # 'maker', 'taker'
    created_at = Column(DateTime, default=func.now())


class PortfolioPositions(Base):
    __tablename__ = "portfolio_positions"

    portfolio_id = Column(String, primary_key=True)
    symbol = Column(String, primary_key=True)
    quantity = Column(Integer, nullable=False)
    avg_cost = Column(Numeric, nullable=False)
    market_value = Column(Numeric)
    unrealized_pnl = Column(Numeric)
    realized_pnl = Column(Numeric, default=0)
    last_updated = Column(DateTime, default=func.now())


class TradingAccounts(Base):
    __tablename__ = "trading_accounts"

    account_id = Column(String, primary_key=True)
    broker = Column(String, nullable=False)
    account_type = Column(String)  # 'paper', 'live', 'margin'
    buying_power = Column(Numeric)
    cash_balance = Column(Numeric)
    total_value = Column(Numeric)
    day_trades_used = Column(Integer, default=0)
    is_pdt = Column(Boolean, default=False)  # Pattern Day Trader
    last_updated = Column(DateTime, default=func.now())


class AIValueScores(Base):
    __tablename__ = "ai_value_scores"

    symbol = Column(String, primary_key=True)
    as_of_date = Column(Date, primary_key=True)
    score = Column(Numeric)
    percentile = Column(Numeric)
    score_label = Column(String)
    rank = Column(Integer)
    model_version = Column(String)
    score_type = Column(String)
    fetched_at = Column(DateTime)


class AIGrowthScores(Base):
    __tablename__ = "ai_growth_scores"

    symbol = Column(String, primary_key=True)
    as_of_date = Column(Date, primary_key=True)
    score = Column(Numeric)
    percentile = Column(Numeric)
    score_label = Column(String)
    rank = Column(Integer)
    model_version = Column(String)
    score_type = Column(String)
    fetched_at = Column(DateTime)


class AIDividendScores(Base):
    __tablename__ = "ai_dividend_scores"

    symbol = Column(String, primary_key=True)
    as_of_date = Column(Date, primary_key=True)
    score = Column(Numeric)
    percentile = Column(Numeric)
    score_label = Column(String)
    rank = Column(Integer)
    model_version = Column(String)
    score_type = Column(String)
    fetched_at = Column(DateTime)


class AIMomentumScores(Base):
    __tablename__ = "ai_momentum_scores"

    symbol = Column(String, primary_key=True)
    as_of_date = Column(Date, primary_key=True)
    score = Column(Numeric)
    percentile = Column(Numeric)
    score_label = Column(String)
    rank = Column(Integer)
    model_version = Column(String)
    score_type = Column(String)
    fetched_at = Column(DateTime)


class BacktestRuns(Base):
    __tablename__ = "backtest_runs"

    run_id = Column(String, primary_key=True)
    strategy_name = Column(String, nullable=False)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    initial_capital = Column(Numeric, nullable=False)
    benchmark = Column(String, default='SPY')
    universe_filter = Column(JSONB)
    rebalance_frequency = Column(String)
    transaction_costs = Column(Numeric, default=0)
    parameters = Column(JSONB)
    created_at = Column(DateTime, default=func.now())
    status = Column(String, default='running')
    notes = Column(Text)


class BacktestPerformance(Base):
    __tablename__ = "backtest_performance"

    run_id = Column(String, primary_key=True)
    total_return = Column(Numeric)
    annual_return = Column(Numeric)
    volatility = Column(Numeric)
    sharpe_ratio = Column(Numeric)
    sortino_ratio = Column(Numeric)
    max_drawdown = Column(Numeric)
    max_drawdown_duration = Column(Integer)
    calmar_ratio = Column(Numeric)
    win_rate = Column(Numeric)
    profit_factor = Column(Numeric)
    beta = Column(Numeric)
    alpha = Column(Numeric)
    tracking_error = Column(Numeric)
    information_ratio = Column(Numeric)
    var_95 = Column(Numeric)
    cvar_95 = Column(Numeric)
    skewness = Column(Numeric)
    kurtosis = Column(Numeric)
    trades_count = Column(Integer)
    avg_trade_pnl = Column(Numeric)
    best_trade = Column(Numeric)
    worst_trade = Column(Numeric)


class RiskMetrics(Base):
    __tablename__ = "risk_metrics"

    portfolio_id = Column(String)
    symbol = Column(String)
    as_of_date = Column(Date, primary_key=True)
    beta = Column(Numeric)
    volatility_30d = Column(Numeric)
    var_95 = Column(Numeric)
    var_99_1d = Column(Numeric)
    expected_shortfall = Column(Numeric)
    max_drawdown = Column(Numeric)
    sharpe_ratio = Column(Numeric)
    correlation_spy = Column(Numeric)

    # Portfolio-specific fields
    total_value = Column(Numeric)
    cash_pct = Column(Numeric)
    sector_concentration = Column(JSONB)
    single_stock_max = Column(Numeric)
    active_share = Column(Numeric)


class SystemAlerts(Base):
    __tablename__ = "system_alerts"

    alert_id = Column(String, primary_key=True)
    alert_type = Column(String, nullable=False)
    severity = Column(String, nullable=False)
    title = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    details = Column(JSONB)
    is_acknowledged = Column(Boolean, default=False)
    acknowledged_by = Column(String)
    acknowledged_at = Column(DateTime)
    created_at = Column(DateTime, default=func.now())


class DataQualityChecks(Base):
    __tablename__ = "data_quality_checks"

    check_id = Column(String, primary_key=True)
    table_name = Column(String, nullable=False)
    check_type = Column(String, nullable=False)
    symbol = Column(String)
    expected_count = Column(Integer)
    actual_count = Column(Integer)
    check_date = Column(Date, nullable=False)
    status = Column(String, nullable=False)
    details = Column(JSONB)
    created_at = Column(DateTime, default=func.now())