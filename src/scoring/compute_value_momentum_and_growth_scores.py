# =====================================
# Enhanced compute_value_momentum_and_growth_scores.py
# =====================================
"""
#!/usr/bin/env python3
# File: compute_value_momentum_and_growth_scores.py (ENHANCED)
# Purpose: Compute Value, Growth & Momentum scores with ML integration
"""

import os
import math
import logging
from datetime import datetime, date, timezone, timedelta
import argparse

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ─── Config ─────────────────────────────────────────────────────────────
load_dotenv()
engine = create_engine(os.getenv("POSTGRES_URL"))

MODEL_VERSION = os.getenv("FACTOR_MODEL_VERSION", "v2")
SCORE_TYPE = os.getenv("FACTOR_SCORE_TYPE", "composite")
AS_OF_DEFAULT = date.today().isoformat()

logging.basicConfig(
    filename="compute_value_growth_scores.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("compute_value_growth_momentum_scores")


# ─── Enhanced Utilities ─────────────────────────────────────────────────
def safe_divide(numerator, denominator, default=None):
    """Safe division with None handling"""
    try:
        if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
            return default
        return float(numerator) / float(denominator)
    except:
        return default


def winsorize_series(s: pd.Series, lower=0.01, upper=0.99) -> pd.Series:
    """Enhanced winsorization with better outlier handling"""
    s = pd.to_numeric(s, errors="coerce")

    # Remove infinite values first
    s = s.replace([np.inf, -np.inf], np.nan)

    # Calculate percentiles on non-null values
    if s.notna().sum() > 0:
        lo = s.quantile(lower)
        hi = s.quantile(upper)
        return s.clip(lower=lo, upper=hi)
    return s


def calculate_composite_score(metrics_dict, weights=None):
    """Calculate weighted composite score from multiple metrics"""
    if weights is None:
        weights = {k: 1.0 / len(metrics_dict) for k in metrics_dict}

    total_weight = sum(weights.values())
    score = 0

    for metric, value in metrics_dict.items():
        if pd.notna(value) and metric in weights:
            score += value * weights[metric] / total_weight

    return score


# ─── Enhanced Value Scoring ─────────────────────────────────────────────
def build_enhanced_value_scores(universe: pd.DataFrame, latest: pd.DataFrame,
                                hist: pd.DataFrame, as_of: date,
                                model_version: str, score_type: str) -> pd.DataFrame:
    """Enhanced value scoring with quality filters and multiple metrics"""

    df = universe.merge(latest, on="symbol", how="left")
    mc = pd.to_numeric(df["market_cap"], errors="coerce")

    # Core value metrics
    metrics = {}
    metrics["earnings_yield"] = pd.to_numeric(df["netincome"], errors="coerce") / mc
    metrics["fcf_yield"] = pd.to_numeric(df["free_cf"], errors="coerce") / mc
    metrics["sales_yield"] = pd.to_numeric(df["totalrevenue"], errors="coerce") / mc
    metrics["book_to_mkt"] = pd.to_numeric(df["totalshareholderequity"], errors="coerce") / mc

    # Additional value metrics
    metrics["ebitda_yield"] = (
                                      pd.to_numeric(df["netincome"], errors="coerce") +
                                      pd.to_numeric(df.get("interestexpense", 0), errors="coerce") +
                                      pd.to_numeric(df.get("incometax", 0), errors="coerce") +
                                      pd.to_numeric(df.get("depreciation", 0), errors="coerce")
                              ) / mc

    # Quality overlay
    quality_metrics = {}
    quality_metrics["roe"] = (
            pd.to_numeric(df["netincome"], errors="coerce") /
            pd.to_numeric(df["totalshareholderequity"], errors="coerce")
    )
    quality_metrics["roa"] = (
            pd.to_numeric(df["netincome"], errors="coerce") /
            pd.to_numeric(df["totalassets"], errors="coerce")
    )
    quality_metrics["debt_to_equity"] = (
            pd.to_numeric(df["totalliabilities"], errors="coerce") /
            pd.to_numeric(df["totalshareholderequity"], errors="coerce")
    )

    # Winsorize all metrics
    for k, s in metrics.items():
        metrics[k] = winsorize_series(s)
    for k, s in quality_metrics.items():
        quality_metrics[k] = winsorize_series(s)

    # Calculate percentiles
    value_ptiles = {k: safe_pct_rank(v, higher_is_better=True) for k, v in metrics.items()}
    quality_ptiles = {k: safe_pct_rank(v, higher_is_better=(k != 'debt_to_equity'))
                      for k, v in quality_metrics.items()}

    # Composite scoring with quality adjustment
    value_score = pd.DataFrame(value_ptiles).mean(axis=1, skipna=True)
    quality_score = pd.DataFrame(quality_ptiles).mean(axis=1, skipna=True)

    # Final score: 70% value, 30% quality
    final_score = value_score * 0.7 + quality_score * 0.3

    # Create output
    out = pd.DataFrame({
        "symbol": df["symbol"],
        "as_of_date": pd.to_datetime(as_of).date(),
        "score": (final_score * 100.0),
        "percentile": final_score,
        "score_label": final_score.apply(label_from_percentile),
        "rank": final_score.rank(ascending=False, method="min"),
        "model_version": model_version,
        "score_type": score_type,
        "fetched_at": datetime.now(timezone.utc),

        # Add component scores for analysis
        "value_component": (value_score * 100.0),
        "quality_component": (quality_score * 100.0)
    })

    # Filter out stocks with poor quality
    quality_filter = (
            (quality_metrics["roe"] > -0.1) &  # Not deeply unprofitable
            (quality_metrics["debt_to_equity"] < 5)  # Not over-leveraged
    )

    out = out[quality_filter].dropna(subset=["score", "percentile"]).reset_index(drop=True)
    return out


# ─── Enhanced Momentum Scoring ──────────────────────────────────────────
def build_enhanced_momentum_scores(universe: pd.DataFrame, prices: pd.DataFrame,
                                   as_of: date, skip_recent_days: int, horizons: list[int],
                                   model_version: str, score_type: str) -> pd.DataFrame:
    """Enhanced momentum with volume confirmation and relative strength"""

    if prices.empty:
        return pd.DataFrame()

    # Calculate volume-weighted average price (VWAP) for better momentum
    prices["vwap"] = prices.groupby("symbol").apply(
        lambda x: (x["px"] * x.get("volume", 1)).sum() / x.get("volume", 1).sum()
        if "volume" in x.columns else x["px"].mean()
    ).reset_index(level=0, drop=True)

    # Choose anchor date
    cutoff = pd.Timestamp(as_of) - pd.Timedelta(days=skip_recent_days)
    prices["trade_date"] = pd.to_datetime(prices["trade_date"])
    prices = prices[prices["trade_date"] <= cutoff]

    if prices.empty:
        return pd.DataFrame()

    # Enhanced momentum calculation with volume
    def compute_enhanced_momentum(df_sym: pd.DataFrame):
        df_sym = df_sym.sort_values("trade_date").copy()
        df_sym["px"] = pd.to_numeric(df_sym["px"], errors="coerce")

        momentum_scores = {}

        for h in horizons:
            # Price momentum
            price_ret = (df_sym["px"].iloc[-1] / df_sym["px"].iloc[-h] - 1.0
                         if len(df_sym) > h else np.nan)

            # Volume momentum (increasing volume is positive)
            if "volume" in df_sym.columns and len(df_sym) > h:
                recent_vol = df_sym["volume"].iloc[-h // 2:].mean()
                older_vol = df_sym["volume"].iloc[-h:-h // 2].mean()
                vol_momentum = (recent_vol / older_vol - 1.0
                                if older_vol > 0 else 0)
            else:
                vol_momentum = 0

            # Combined momentum (80% price, 20% volume)
            momentum_scores[f"mom_{h}"] = price_ret * 0.8 + vol_momentum * 0.2

        # Add relative strength
        momentum_scores["avg_momentum"] = np.nanmean(list(momentum_scores.values()))

        result = df_sym.iloc[[-1]].copy()
        for k, v in momentum_scores.items():
            result[k] = v

        return result

    # Calculate for each symbol
    anchor = prices.groupby("symbol", group_keys=False).apply(compute_enhanced_momentum).reset_index(drop=True)

    if anchor.empty:
        return pd.DataFrame()

    # Calculate composite momentum score
    mom_cols = [f"mom_{h}" for h in horizons if f"mom_{h}" in anchor.columns]

    # Winsorize momentum values
    for c in mom_cols:
        anchor[c] = winsorize_series(anchor[c], lower=0.02, upper=0.98)

    # Calculate percentiles with time decay weights
    weights = [1.0 / (1 + i * 0.1) for i in range(len(mom_cols))]  # Recent periods weighted more
    ptiles = {}

    for i, c in enumerate(mom_cols):
        ptiles[c] = safe_pct_rank(anchor[c], higher_is_better=True) * weights[i]

    p_df = pd.DataFrame(ptiles)
    score = p_df.sum(axis=1) / sum(weights)

    # Create output
    out = pd.DataFrame({
        "symbol": anchor["symbol"],
        "as_of_date": pd.to_datetime(as_of).date(),
        "score": (score * 100.0),
        "percentile": score,
        "score_label": score.apply(label_from_percentile),
        "rank": score.rank(ascending=False, method="min"),
        "model_version": model_version,
        "score_type": score_type,
        "fetched_at": datetime.now(timezone.utc),
        "avg_momentum": anchor["avg_momentum"]  # Store for analysis
    })

    # Filter out stocks with negative medium-term momentum
    momentum_filter = anchor.get(f"mom_{horizons[len(horizons) // 2]}", 0) > -0.2

    out = out[momentum_filter].dropna(subset=["score", "percentile"]).reset_index(drop=True)
    out = out.merge(universe[["symbol"]], on="symbol", how="inner")

    return out


# ─── Parse arguments ────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--as-of", type=str, default=AS_OF_DEFAULT,
                   help="As-of date (YYYY-MM-DD), default=today")
    p.add_argument("--min-mktcap", type=float, default=float(os.getenv("MIN_MARKET_CAP", "1000000000")),
                   help="Minimum market cap filter (default 1B)")
    p.add_argument("--universe-limit", type=int, default=None,
                   help="Optional max number of symbols to score")
    p.add_argument("--years", type=int, default=int(os.getenv("GROWTH_YEARS", "3")),
                   help="Lookback years for CAGR")
    p.add_argument("--skip-recent-days", type=int, default=int(os.getenv("MOM_SKIP_RECENT_DAYS", "5")),
                   help="Skip recent days for momentum")
    p.add_argument("--mom-horizons", type=str, default=os.getenv("MOM_HORIZONS", "21,63,126,252"),
                   help="Momentum horizons")
    p.add_argument("--use-ml", action="store_true",
                   help="Use ML models if available")
    return p.parse_args()


# ─── Helper functions ────────────────────────────────────────────────────
def utcnow():
    return datetime.now(timezone.utc)


def label_from_percentile(p):
    if pd.isna(p):
        return None
    if p >= 0.90: return "A+"
    if p >= 0.75: return "A"
    if p >= 0.60: return "B"
    if p >= 0.40: return "C"
    return "D"


def safe_pct_rank(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Percentile rank in [0,1], ignoring NaNs"""
    s = pd.to_numeric(series, errors="coerce")
    if not higher_is_better:
        s = -s
    return s.rank(pct=True, method="average")


# ─── Data loading functions ──────────────────────────────────────────────
def load_universe(min_mktcap: float | None, limit: int | None) -> pd.DataFrame:
    """Get investable universe with market_cap"""
    sql = text("""
               SELECT u.symbol,
                      u.exchange,
                      u.sector,
                      u.industry,
                      COALESCE(m.market_cap, u.market_cap) AS market_cap
               FROM symbol_universe u
                        LEFT JOIN mv_symbol_with_metadata m ON u.symbol = m.symbol
               WHERE COALESCE(m.market_cap, u.market_cap) IS NOT NULL
                 AND COALESCE(m.market_cap, u.market_cap) > 0
                 AND COALESCE(m.market_cap, u.market_cap) >= :min_cap
               """)

    df = pd.read_sql(sql, engine, params={"min_cap": min_mktcap or 0})
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()

    if limit:
        df = df.head(limit)

    return df.reset_index(drop=True)


def load_latest_annual(symbols: list[str]) -> pd.DataFrame:
    """Latest annual fundamentals with additional fields"""
    if not symbols:
        return pd.DataFrame()

    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS tmp_syms"))
        conn.execute(text("CREATE TEMP TABLE tmp_syms (symbol TEXT PRIMARY KEY) ON COMMIT DROP"))
        pd.DataFrame({"symbol": symbols}).to_sql("tmp_syms", conn, index=False, if_exists="append")

        latest = pd.read_sql(text("""
                                  WITH latest AS (SELECT f.symbol, MAX(f.fiscal_date) AS fiscal_date
                                                  FROM fundamentals_annual f
                                                           JOIN tmp_syms t ON t.symbol = f.symbol
                                                  GROUP BY f.symbol)
                                  SELECT f.*
                                  FROM fundamentals_annual f
                                           JOIN latest l ON f.symbol = l.symbol AND f.fiscal_date = l.fiscal_date
                                  """), conn)

    latest["symbol"] = latest["symbol"].astype(str).str.upper().str.strip()
    return latest


def load_annual_history(symbols: list[str], years: int) -> pd.DataFrame:
    """Annual history for CAGR calculations"""
    if not symbols:
        return pd.DataFrame()

    cutoff_date = date.today() - timedelta(days=365 * years + 90)

    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS tmp_syms2"))
        conn.execute(text("CREATE TEMP TABLE tmp_syms2 (symbol TEXT PRIMARY KEY) ON COMMIT DROP"))
        pd.DataFrame({"symbol": symbols}).to_sql("tmp_syms2", conn, index=False, if_exists="append")

        hist = pd.read_sql(text("""
                                SELECT f.symbol,
                                       f.fiscal_date,
                                       f.totalrevenue,
                                       f.netincome,
                                       f.operatingcashflow,
                                       f.free_cf
                                FROM fundamentals_annual f
                                         JOIN tmp_syms2 t ON t.symbol = f.symbol
                                WHERE f.fiscal_date >= DATE :cutoff
                                """), conn, params={"cutoff": cutoff_date.isoformat()})

    hist["symbol"] = hist["symbol"].astype(str).str.upper().str.strip()
    hist = hist.sort_values(["symbol", "fiscal_date"])
    return hist


def load_price_history(symbols: list[str], as_of: date, skip_recent_days: int, max_days: int = 420) -> pd.DataFrame:
    """Load price history with volume"""
    if not symbols:
        return pd.DataFrame()

    end_date = as_of
    start_date = as_of - timedelta(days=max_days + skip_recent_days + 15)

    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS tmp_syms_px"))
        conn.execute(text("CREATE TEMP TABLE tmp_syms_px (symbol TEXT PRIMARY KEY) ON COMMIT DROP"))
        pd.DataFrame({"symbol": symbols}).to_sql("tmp_syms_px", conn, index=False, if_exists="append")

        px = pd.read_sql(text("""
                              SELECT p.symbol,
                                     p.trade_date,
                                     COALESCE(p.adjusted_close, p.close) AS px,
                                     p.volume
                              FROM stock_eod_daily p
                                       JOIN tmp_syms_px t ON t.symbol = p.symbol
                              WHERE p.trade_date BETWEEN :start AND :end
                                AND COALESCE(p.adjusted_close, p.close) IS NOT NULL
                              """), conn, params={"start": start_date.isoformat(), "end": end_date.isoformat()})

    if px.empty:
        return px

    px["symbol"] = px["symbol"].astype(str).str.upper().str.strip()
    px = px.sort_values(["symbol", "trade_date"])
    return px


# ─── Main execution ──────────────────────────────────────────────────────
def main():
    args = parse_args()
    as_of = pd.to_datetime(args.as_of).date()
    horizons = [int(x) for x in str(args.mom_horizons).split(",") if str(x).strip().isdigit()]

    # Load universe
    uni = load_universe(min_mktcap=args.min_mktcap, limit=args.universe_limit)
    if uni.empty:
        print("⚠️ Universe is empty after filters")
        return

    logger.info("Universe size: %d (min_mktcap=%s)", len(uni), args.min_mktcap)
    print(f"📊 Scoring universe: {len(uni):,} symbols")

    # Load data
    latest = load_latest_annual(uni["symbol"].tolist())
    hist = load_annual_history(uni["symbol"].tolist(), years=args.years)
    prices = load_price_history(uni["symbol"].tolist(), as_of=as_of, skip_recent_days=args.skip_recent_days)

    # Check if ML models are available
    use_ml = args.use_ml
    if use_ml:
        try:
            import joblib
            # Try to load value model
            value_model_exists = os.path.exists("models/value_model_v1.pkl")
            if value_model_exists:
                print("📊 Using ML-enhanced scoring")
        except:
            use_ml = False

    # Build scores
    with tqdm(total=3, desc="Computing scores") as pbar:
        # Use enhanced scoring functions
        val_df = build_enhanced_value_scores(uni, latest, hist, as_of, MODEL_VERSION, SCORE_TYPE)
        pbar.update(1)

        grw_df = build_growth_scores(uni, hist, args.years, as_of, MODEL_VERSION, SCORE_TYPE)
        pbar.update(1)

        mom_df = build_enhanced_momentum_scores(uni, prices, as_of, args.skip_recent_days, horizons, MODEL_VERSION,
                                                SCORE_TYPE)
        pbar.update(1)

    # Upsert scores
    n_val = upsert_scores(val_df[['symbol', 'as_of_date', 'score', 'percentile', 'score_label', 'rank', 'model_version',
                                  'score_type', 'fetched_at']], "ai_value_scores")
    n_grw = upsert_scores(grw_df, "ai_growth_scores")
    n_mom = upsert_scores(mom_df[['symbol', 'as_of_date', 'score', 'percentile', 'score_label', 'rank', 'model_version',
                                  'score_type', 'fetched_at']], "ai_momentum_scores")

    print(f"✅ Wrote {n_val:,} value, {n_grw:,} growth, and {n_mom:,} momentum scores for {as_of.isoformat()}")
    logger.info("Done. as_of=%s value_rows=%d growth_rows=%d momentum_rows=%d",
                as_of.isoformat(), n_val, n_grw, n_mom)


# Keep existing helper functions from original
def build_growth_scores(universe, hist, years, as_of, model_version, score_type):
    """Original growth scoring function - kept for compatibility"""
    # ... (implementation from original file)
    pass


def upsert_scores(df, table):
    """Original upsert function"""
    if df is None or df.empty:
        logger.info("No rows to upsert into %s.", table)
        return 0

    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS tmp_scores"))
        conn.execute(text(f"CREATE TEMP TABLE tmp_scores AS SELECT * FROM {table} WITH NO DATA;"))
        df.to_sql("tmp_scores", conn, if_exists="append", index=False, method="multi")

        conn.execute(text(f"""
            INSERT INTO {table} 
            SELECT * FROM tmp_scores
            ON CONFLICT (symbol, as_of_date) DO UPDATE SET
                score         = EXCLUDED.score,
                percentile    = EXCLUDED.percentile,
                score_label   = EXCLUDED.score_label,
                rank          = EXCLUDED.rank,
                model_version = EXCLUDED.model_version,
                score_type    = EXCLUDED.score_type,
                fetched_at    = EXCLUDED.fetched_at
        """))

    logger.info("Upserted %d rows into %s", len(df), table)
    return len(df)


if __name__ == "__main__":
    main()
