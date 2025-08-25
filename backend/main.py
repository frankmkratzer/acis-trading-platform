# backend/main.py - ACIS Trading Platform FastAPI Application
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from contextlib import asynccontextmanager
import uvicorn
import logging
from typing import Optional
from datetime import datetime
import os
import uuid

from .database import get_db, check_database_connection, get_database_info
from .schemas import UserLogin, Token, CreateOrderRequest
from .auth import get_current_user, create_access_token, authenticate_user

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting ACIS (Alpha Centauri Investment Strategies) Platform API...")

    # Check database connection
    if check_database_connection():
        db_info = get_database_info()
        logger.info(f"Connected to database: {db_info.get('database')}")
        logger.info(f"PostgreSQL version: {db_info.get('version', 'Unknown')}")
    else:
        logger.error("Failed to connect to database!")

    yield
    # Shutdown
    logger.info("Shutting down ACIS Platform API...")


# Create FastAPI app
app = FastAPI(
    title="ACIS Trading Platform",
    description="Alpha Centauri Investment Strategies - AI-Powered Quantitative Trading & Portfolio Management",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


# Health check endpoint
@app.get("/health")
async def health_check():
    db_status = check_database_connection()
    return {
        "status": "healthy" if db_status else "unhealthy",
        "timestamp": datetime.utcnow(),
        "service": "ACIS Trading Platform",
        "database": "connected" if db_status else "disconnected"
    }


# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "ACIS Trading Platform API",
        "company": "Alpha Centauri Investment Strategies",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs"
    }


# Authentication endpoints
@app.post("/auth/login", response_model=Token)
async def login(credentials: UserLogin):
    """Login endpoint for ACIS platform"""
    user = authenticate_user(credentials.username, credentials.password)
    if not user:
        logger.warning(f"Failed login attempt for username: {credentials.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(data={"sub": user.username})
    logger.info(f"User '{user.username}' logged in successfully")

    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/auth/me")
async def get_current_user_info(current_user=Depends(get_current_user)):
    """Get current user information"""
    return {
        "username": current_user.username,
        "email": current_user.email,
        "role": current_user.role,
        "disabled": current_user.disabled
    }


# Portfolio endpoints
@app.get("/portfolios")
async def get_portfolios(current_user=Depends(get_current_user)):
    """Get all portfolio summaries"""
    return [
        {
            "portfolio_id": "ai-value",
            "name": "AI Value Strategy",
            "total_value": 125000.00,
            "daily_pnl": 2341.50,
            "daily_return": 0.0187,
            "positions_count": 18
        },
        {
            "portfolio_id": "ai-growth",
            "name": "AI Growth Strategy",
            "total_value": 98500.00,
            "daily_pnl": -856.20,
            "daily_return": -0.0087,
            "positions_count": 22
        }
    ]


# Stock search endpoint
@app.get("/stocks/search")
async def search_stocks(query: str, limit: int = 20, current_user=Depends(get_current_user)):
    """Search stocks by symbol or name"""
    mock_results = [
        {"symbol": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ", "market_cap": 3000000000000},
        {"symbol": "MSFT", "name": "Microsoft Corporation", "exchange": "NASDAQ", "market_cap": 2800000000000},
        {"symbol": "GOOGL", "name": "Alphabet Inc.", "exchange": "NASDAQ", "market_cap": 1700000000000}
    ]

    filtered = [stock for stock in mock_results if
                query.upper() in stock["symbol"] or query.lower() in stock["name"].lower()]
    return filtered[:limit]


# Trading endpoints
@app.get("/orders")
async def get_orders(status: Optional[str] = None, limit: int = 100, current_user=Depends(get_current_user)):
    """Get trading orders"""
    return [
        {
            "order_id": "order_001",
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 100,
            "order_type": "market",
            "status": "filled",
            "filled_quantity": 100,
            "avg_fill_price": 175.25,
            "submitted_at": datetime.utcnow()
        }
    ]


@app.post("/orders")
async def create_order(order: CreateOrderRequest, current_user=Depends(get_current_user)):
    """Create a new trading order"""
    order_id = f"order_{str(uuid.uuid4())[:8]}"

    logger.info(f"Order created: {order_id} - {order.side} {order.quantity} {order.symbol}")

    return {
        "order_id": order_id,
        "symbol": order.symbol,
        "side": order.side,
        "quantity": order.quantity,
        "order_type": order.order_type,
        "status": "pending",
        "submitted_at": datetime.utcnow()
    }


# System status endpoint
@app.get("/system/status")
async def get_system_status(current_user=Depends(get_current_user)):
    """Get system health status"""
    db_info = get_database_info()

    return {
        "database": db_info,
        "data_feeds": {"status": "healthy", "last_update": datetime.utcnow()},
        "trading_connection": {"status": "connected", "broker": "Schwab"},
        "last_updated": datetime.utcnow()
    }


# Dashboard data endpoint
@app.get("/analytics/dashboard")
async def get_dashboard_data(current_user=Depends(get_current_user)):
    """Get dashboard summary data"""
    return {
        "total_portfolio_value": 310700.00,
        "daily_pnl": 1485.30,
        "daily_return": 0.0048,
        "active_positions": 55,
        "cash_percentage": 8.5,
        "strategy_performance": {
            "AI Value": 0.187,
            "AI Growth": 0.243,
            "AI Dividend": 0.092,
            "AI Momentum": 0.156
        }
    }


# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=os.getenv("ENVIRONMENT") != "production",
        log_level="info"
    )