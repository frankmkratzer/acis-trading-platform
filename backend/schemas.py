# backend/schemas.py
# Pydantic schemas for ACIS Trading Platform API

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from decimal import Decimal


# ═══════════ AUTHENTICATION SCHEMAS ═══════════

class UserLogin(BaseModel):
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    username: Optional[str] = None


# ═══════════ STOCK & MARKET DATA SCHEMAS ═══════════

class StockInfo(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    symbol: str
    name: str
    exchange: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[Decimal] = None
    pe_ratio: Optional[Decimal] = None
    dividend_yield: Optional[Decimal] = None
    week_52_high: Optional[Decimal] = None
    week_52_low: Optional[Decimal] = None


class StockSearch(BaseModel):
    symbol: str
    name: str
    exchange: str
    market_cap: Optional[Decimal] = None


# ═══════════ TRADING SCHEMAS ═══════════

class CreateOrderRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    side: str = Field(..., description="'buy' or 'sell'")
    quantity: int = Field(..., gt=0, description="Number of shares")
    order_type: str = Field(..., description="'market', 'limit', or 'stop'")
    limit_price: Optional[Decimal] = Field(None, description="Limit price for limit orders")
    stop_price: Optional[Decimal] = Field(None, description="Stop price for stop orders")
    time_in_force: str = Field("DAY", description="'DAY', 'GTC', 'IOC', or 'FOK'")
    strategy: Optional[str] = Field(None, description="Associated strategy")
    portfolio_id: Optional[str] = Field(None, description="Portfolio ID")


class OrderInfo(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    order_id: str
    symbol: str
    side: str
    quantity: int
    order_type: str
    status: str
    limit_price: Optional[Decimal] = None
    filled_quantity: int = 0
    avg_fill_price: Optional[Decimal] = None
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None


# ═══════════ PORTFOLIO SCHEMAS ═══════════

class PortfolioSummary(BaseModel):
    portfolio_id: str
    name: str
    total_value: Decimal
    cash_balance: Decimal
    positions_count: int
    daily_pnl: Decimal
    daily_return: Decimal
    inception_date: date
    last_updated: datetime


# ═══════════ RESPONSE MODELS ═══════════

class SuccessResponse(BaseModel):
    success: bool = True
    message: str
    data: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    details: Optional[Dict[str, Any]] = None