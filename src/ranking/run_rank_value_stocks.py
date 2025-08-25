#!/usr/bin/env python3
# File: run_rank_value_stocks.py
# Purpose: Compute and rank value stocks using latest AI value scores and write top picks

import os
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# ─── Setup ─────────────────────────────────────────────────────────────
load_dotenv()
engine = create_engine(os.getenv("POSTGRES_URL"))

# ─── Constants ──────────────────────────────────────────────────────────
MODEL_VERSION = os.getenv("VALUE_MODEL_VERSION", datetime.now().strftime("v%Y%m%d"))
TOP_K = 20
RANK_TABLE = "ai_value_portfolio"
SCORE_TABLE = "ai_value_scores"

# ─── Main ──────────────────────────────────────────────────────────────
def main():
    print("🚀 Loading latest AI value scores...")

    df = pd.read_sql(text(f"""
        SELECT * FROM {SCORE_TABLE}
        WHERE model_version = :model_version
        ORDER BY as_of_date DESC, ai_score DESC
        LIMIT 1000
    """), engine, params={"model_version": MODEL_VERSION})

    if df.empty:
        print("⚠️ No scores found for current model version.")
        return

    latest_date = df['as_of_date'].max()
    top_df = df[df['as_of_date'] == latest_date].nlargest(TOP_K, 'ai_score').copy()
    top_df['rank'] = range(1, len(top_df)+1)
    top_df['score_type'] = 'value'
    top_df['model_version'] = MODEL_VERSION

    print(f"🏆 Selected top {len(top_df)} value stocks for {latest_date}")

    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {RANK_TABLE} (
                symbol TEXT NOT NULL,
                as_of_date DATE NOT NULL,
                ai_score NUMERIC,
                ai_percentile NUMERIC,
                ai_score_label TEXT,
                rank INTEGER,
                model_version TEXT,
                score_type TEXT,
                PRIMARY KEY (symbol, as_of_date, score_type)
            )
        """))

        top_df[['symbol','as_of_date','ai_score','ai_percentile','ai_score_label','rank','model_version','score_type']] \
            .to_sql(RANK_TABLE, con=conn, if_exists="append", index=False, method="multi")

    print("✅ Ranking complete and saved.")

if __name__ == "__main__":
    main()
