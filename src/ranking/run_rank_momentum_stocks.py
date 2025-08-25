# ===============================
# File: run_rank_momentum_stocks.py
# Purpose: Select and persist Top-K momentum names from ai_momentum_scores
# ===============================

#!/usr/bin/env python3
import os
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
engine = create_engine(os.getenv("POSTGRES_URL"))
MODEL_VERSION = os.getenv("MOMENTUM_MODEL_VERSION", datetime.now().strftime("v%Y%m%d"))
TOP_K = int(os.getenv("MOMENTUM_TOP_K", 20))
SCORE_TABLE = "ai_momentum_scores"
OUT_TABLE = "ai_momentum_portfolio"
SCORE_TYPE = "momentum"

DDL = f"""
CREATE TABLE IF NOT EXISTS {OUT_TABLE} (
    symbol TEXT NOT NULL,
    as_of_date DATE NOT NULL,
    score NUMERIC,
    percentile NUMERIC,
    score_label TEXT,
    rank INTEGER,
    model_version TEXT,
    score_type TEXT,
    fetched_at TIMESTAMPTZ,
    PRIMARY KEY (symbol, as_of_date, score_type)
);
"""


def main():
    with engine.begin() as conn:
        conn.execute(text(DDL))

    df = pd.read_sql(
        text(f"""
            SELECT symbol, as_of_date, score, percentile, score_label, model_version
            FROM {SCORE_TABLE}
            WHERE model_version = :mv
        """),
        engine,
        params={"mv": MODEL_VERSION},
        parse_dates=["as_of_date"],
    )

    if df.empty:
        print("⚠️ No momentum scores for model_version", MODEL_VERSION)
        return

    as_of = df["as_of_date"].max()
    top = (
        df[df["as_of_date"] == as_of]
          .sort_values("score", ascending=False)
          .head(TOP_K)
          .copy()
    )
    top["rank"] = range(1, len(top) + 1)
    top["score_type"] = SCORE_TYPE
    top["fetched_at"] = datetime.utcnow()

    with engine.begin() as conn:
        conn.execute(text(f"DELETE FROM {OUT_TABLE} WHERE as_of_date = :d AND score_type = :t"),
                     {"d": as_of.date(), "t": SCORE_TYPE})
        top[[
            "symbol", "as_of_date", "score", "percentile", "score_label",
            "rank", "model_version", "score_type", "fetched_at"
        ]].to_sql(OUT_TABLE, conn, if_exists="append", index=False, method="multi")

    print(f"✅ Saved Top-{len(top)} momentum picks for {as_of.date()} → {OUT_TABLE}")


if __name__ == "__main__":
    main()
