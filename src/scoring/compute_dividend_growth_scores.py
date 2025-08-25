#!/usr/bin/env python3
# Purpose: Calculate dividend CAGR and detect dividend cuts with retries and parallel processing

import os
import time
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timezone
from sqlalchemy import create_engine
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# ─── Setup ──────────────────────────────────────────────────────────
load_dotenv()
POSTGRES_URL = os.getenv("POSTGRES_URL")
engine = create_engine(POSTGRES_URL)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("div_growth")

MAX_WORKERS = 8
MAX_RETRIES = 3

# ─── CAGR Calculation ──────────────────────────────────────────────────

def calc_cagr(start, end, years):
    try:
        if start > 0 and end >= 0 and years > 0:
            return (end / start) ** (1 / years) - 1
    except:
        pass
    return None

# ─── Scoring Function ──────────────────────────────────────────────────

def compute_growth(symbol, df):
    try:
        df = df.sort_values("ex_date")
        df["year"] = df["ex_date"].dt.year
        yearly = df.groupby("year")["dividend"].sum().reset_index()

        result = {
            "symbol": symbol,
            "as_of_date": df["ex_date"].max(),
            "fetched_at": datetime.now(timezone.utc),
        }

        for period in [1, 3, 5, 10]:
            if len(yearly) >= period + 1:
                start = yearly.iloc[-period - 1]["dividend"]
                end = yearly.iloc[-1]["dividend"]
                result[f"div_cagr_{period}y"] = calc_cagr(start, end, period)
            else:
                result[f"div_cagr_{period}y"] = None

        result["dividend_cut_detected"] = any(yearly["dividend"].diff().fillna(0) < 0)
        return result
    except Exception as e:
        logger.error(f"❌ Error computing growth for {symbol}: {e}")
        return None

# ─── Retry Wrapper ───────────────────────────────────────────────

def retryable(fn, *args):
    for attempt in range(1, MAX_RETRIES + 1):
        result = fn(*args)
        if result is not None:
            return result
        logger.warning(f"⚠️ Retry {attempt} for {args[0]}")
        time.sleep(2 * attempt)
    logger.error(f"❌ Failed after {MAX_RETRIES} retries for {args[0]}")
    return None

# ─── Main ──────────────────────────────────────────────────

def main():
    logger.info("📅 Loading dividend history...")
    query = "SELECT symbol, ex_date, dividend FROM dividend_history"
    df_all = pd.read_sql(query, engine, parse_dates=["ex_date"])

    grouped = df_all.groupby("symbol")
    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(retryable, compute_growth, symbol, group): symbol
            for symbol, group in grouped
        }
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    df = pd.DataFrame(results)
    df.drop_duplicates(subset=["symbol", "as_of_date"], inplace=True)

    logger.info(f"📊 Writing {len(df)} rows to dividend_growth_scores...")
    df.to_sql("dividend_growth_scores", engine, if_exists="replace", index=False)
    logger.info("✅ Done.")

if __name__ == "__main__":
    main()
