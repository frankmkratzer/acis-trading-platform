#!/usr/bin/env python3
# Purpose: Compare stock annual performance to SPY and score lifetime outperformance with parallelism and retries

import os
import pandas as pd
import time
from datetime import datetime, timezone
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─── Setup ───────────────────────────────────────────────────────
load_dotenv()
POSTGRES_URL = os.getenv("POSTGRES_URL")
engine = create_engine(POSTGRES_URL)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sp500_score")

DECAY_RATE = 0.9  # more recent years matter more
MAX_RETRIES = 3

# ─── Scoring Function ────────────────────────────────────────────
def weighted_score(years, flags):
    weights = [(DECAY_RATE ** (max(years) - y)) for y in years]
    return sum(w for w, f in zip(weights, flags) if f)

# ─── Annual Returns ───────────────────────────────────────────────
def compute_annual_returns(df, symbol_col):
    df["year"] = df["trade_date"].dt.year
    df = df.sort_values([symbol_col, "trade_date"])
    return (
        df.groupby([symbol_col, "year"])
          .agg(first_price=("adjusted_close", "first"),
               last_price=("adjusted_close", "last"))
          .reset_index()
          .assign(annual_return=lambda d: d["last_price"] / d["first_price"] - 1)
    )

# ─── Data Fetch ──────────────────────────────────────────────────
def fetch_data():
    stock_prices = pd.read_sql("SELECT symbol, trade_date, adjusted_close FROM stock_eod_daily", engine, parse_dates=["trade_date"])
    spy_prices = pd.read_sql("SELECT trade_date, adjusted_close FROM sp500_price_history", engine, parse_dates=["trade_date"])
    stock_prices = stock_prices[stock_prices["adjusted_close"] > 0]
    spy_prices["symbol"] = "SPY"
    return stock_prices, spy_prices

# ─── Retry Wrapper ───────────────────────────────────────────────
def retryable(fn, *args, **kwargs):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            logger.warning(f"⚠️ Attempt {attempt} failed for {args[0]}: {e}")
            time.sleep(2 * attempt)
    logger.error(f"❌ Giving up after {MAX_RETRIES} retries for {args[0]}")
    return None

# ─── Worker Function ─────────────────────────────────────────────
def process_symbol(symbol, group):
    try:
        years = group["year"].tolist()
        flags = group["outperformed"].tolist()
        score = weighted_score(years, flags)
        return {
            "symbol": symbol,
            "lifetime_outperformer": all(flags),
            "years_outperformed": sum(flags),
            "total_years": len(flags),
            "weighted_score": round(score, 5),
            "last_year": max(years),
            "fetched_at": datetime.now(timezone.utc)
        }
    except Exception as e:
        logger.error(f"❌ Error scoring {symbol}: {e}")
        return None

# ─── Main ────────────────────────────────────────────────────────
def main():
    stock_prices, spy_prices = fetch_data()

    stock_returns = compute_annual_returns(stock_prices, "symbol")
    spy_returns = compute_annual_returns(spy_prices, "symbol").rename(columns={"annual_return": "spy_return"}).drop(columns=["symbol"])

    merged = pd.merge(stock_returns, spy_returns, on="year", how="inner")
    merged["outperformed"] = merged["annual_return"] > merged["spy_return"]

    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(retryable, process_symbol, symbol, group): symbol for symbol, group in merged.groupby("symbol")}
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    df = pd.DataFrame(results)
    df.drop_duplicates(subset=["symbol"], inplace=True)
    df.to_sql("sp500_outperformance_scores", engine, if_exists="replace", index=False)
    logger.info(f"✅ Scored {len(df)} stocks into sp500_outperformance_scores")

if __name__ == "__main__":
    main()
