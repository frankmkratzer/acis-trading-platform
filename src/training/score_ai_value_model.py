# =====================================
# 2. score_ai_value_model.py
# =====================================
"""
#!/usr/bin/env python3
# File: score_ai_value_model.py
# Purpose: Score stocks using trained value model
"""

import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import logging

load_dotenv()
engine = create_engine(os.getenv("POSTGRES_URL"))

logging.basicConfig(
    filename="score_value_model.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class ValueScorer:
    def __init__(self, model_version=None):
        self.model_version = model_version or self._get_latest_version()
        self.model = None
        self.scaler = None
        self.load_model()

    def _get_latest_version(self):
        """Get the latest model version from the database"""
        query = text("""
                     SELECT version
                     FROM ai_model_run_log
                     WHERE model_type = 'value'
                     ORDER BY created_at DESC LIMIT 1
                     """)
        with engine.connect() as conn:
            result = conn.execute(query).fetchone()
            if result:
                return result[0]

        # Fallback to file system
        import glob
        models = glob.glob("models/value_model_*.pkl")
        if models:
            latest = sorted(models)[-1]
            return latest.split("_")[-1].replace(".pkl", "")

        raise ValueError("No value model found")

    def load_model(self):
        """Load the trained model and scaler"""
        model_path = f"models/value_model_{self.model_version}.pkl"
        scaler_path = f"models/value_scaler_{self.model_version}.pkl"

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        logger.info(f"Loaded model version: {self.model_version}")

    def load_current_features(self):
        """Load current features for all stocks"""
        query = text("""
                     WITH latest_fundamentals AS (SELECT DISTINCT
                     ON (symbol)
                         symbol,
                         fiscal_date,
                         netincome,
                         free_cf,
                         totalrevenue,
                         totalshareholderequity,
                         totalassets,
                         totalliabilities,
                         operatingcashflow
                     FROM fundamentals_annual
                     ORDER BY symbol, fiscal_date DESC
                         ),
                         features AS (
                     SELECT
                         f.symbol, CURRENT_DATE as as_of_date,

                         -- Value metrics
                         f.netincome:: float / NULLIF (m.market_cap, 0) as earnings_yield, f.free_cf:: float / NULLIF (m.market_cap, 0) as fcf_yield, f.totalrevenue:: float / NULLIF (m.market_cap, 0) as sales_yield, f.totalshareholderequity:: float / NULLIF (m.market_cap, 0) as book_to_market,

                         -- Quality metrics
                         f.netincome:: float / NULLIF (f.totalshareholderequity, 0) as roe, f.netincome:: float / NULLIF (f.totalassets, 0) as roa, (f.totalassets - f.totalliabilities):: float / NULLIF (f.totalassets, 0) as equity_ratio, f.operatingcashflow:: float / NULLIF (f.netincome, 0) as cash_conversion,

                         -- Growth placeholders (would need historical data)
                         0 as revenue_growth, 0 as earnings_growth,

                         -- Market metrics
                         m.pe_ratio, m.dividend_yield, LOG(m.market_cap + 1) as log_market_cap,

                         -- Sector dummies
                         CASE WHEN m.sector = 'Technology' THEN 1 ELSE 0 END as is_tech, CASE WHEN m.sector = 'Financial Services' THEN 1 ELSE 0 END as is_financial, CASE WHEN m.sector = 'Healthcare' THEN 1 ELSE 0 END as is_healthcare,

                         -- Additional context
                         m.sector, m.industry, m.market_cap

                     FROM latest_fundamentals f
                         JOIN mv_symbol_with_metadata m
                     ON f.symbol = m.symbol
                     WHERE m.market_cap
                         > 1000000000 -- $1B minimum
                       AND f.netincome IS NOT NULL
                       AND f.free_cf IS NOT NULL
                         )
                     SELECT *
                     FROM features
                     WHERE earnings_yield IS NOT NULL
                       AND fcf_yield IS NOT NULL
                       AND ABS(earnings_yield) < 1
                       AND ABS(fcf_yield) < 1
                     """)

        df = pd.read_sql(query, engine)

        # Handle infinities and missing values
        df = df.replace([np.inf, -np.inf], np.nan)

        # Fill NaNs with median (calculated from training data ideally)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['symbol', 'as_of_date']:
                df[col] = df[col].fillna(df[col].median())

        logger.info(f"Loaded features for {len(df)} stocks")
        return df

    def score_stocks(self, df):
        """Generate scores for all stocks"""
        feature_cols = [
            'earnings_yield', 'fcf_yield', 'sales_yield', 'book_to_market',
            'roe', 'roa', 'equity_ratio', 'cash_conversion',
            'revenue_growth', 'earnings_growth',
            'pe_ratio', 'dividend_yield', 'log_market_cap',
            'is_tech', 'is_financial', 'is_healthcare'
        ]

        # Prepare features
        X = df[feature_cols].values

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Generate predictions
        scores = self.model.predict(X_scaled)

        # Create output dataframe
        results = pd.DataFrame({
            'symbol': df['symbol'],
            'as_of_date': df['as_of_date'],
            'raw_score': scores,
            'sector': df['sector'],
            'industry': df['industry'],
            'market_cap': df['market_cap']
        })

        # Normalize scores to 0-100 scale
        results['score'] = 100 * (results['raw_score'].rank(pct=True))

        # Calculate percentiles
        results['percentile'] = results['raw_score'].rank(pct=True)

        # Assign labels
        results['score_label'] = pd.cut(
            results['percentile'],
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=['D', 'C', 'B', 'A', 'A+']
        )

        # Add rank
        results['rank'] = results['raw_score'].rank(ascending=False, method='min').astype(int)

        # Add metadata
        results['model_version'] = self.model_version
        results['score_type'] = 'value'
        results['fetched_at'] = datetime.now()

        return results

    def save_scores(self, scores_df):
        """Save scores to database"""
        # Select columns for database
        db_columns = [
            'symbol', 'as_of_date', 'score', 'percentile',
            'score_label', 'rank', 'model_version', 'score_type', 'fetched_at'
        ]

        scores_to_save = scores_df[db_columns].copy()

        # Upsert to database
        with engine.begin() as conn:
            # Delete existing scores for this date
            conn.execute(text("""
                              DELETE
                              FROM ai_value_scores
                              WHERE as_of_date = :as_of_date
                              """), {"as_of_date": scores_df['as_of_date'].iloc[0]})

            # Insert new scores
            scores_to_save.to_sql(
                'ai_value_scores',
                conn,
                if_exists='append',
                index=False,
                method='multi'
            )

        logger.info(f"Saved {len(scores_to_save)} value scores to database")

        # Also save detailed scores to CSV for analysis
        detailed_path = f"scores/value_scores_{datetime.now().strftime('%Y%m%d')}.csv"
        os.makedirs("scores", exist_ok=True)
        scores_df.to_csv(detailed_path, index=False)

        return len(scores_to_save)


def main():
    scorer = ValueScorer()

    # Load current features
    df = scorer.load_current_features()

    if df.empty:
        logger.error("No stocks to score")
        return

    # Score stocks
    scores = scorer.score_stocks(df)

    # Save scores
    saved_count = scorer.save_scores(scores)

    # Print top stocks
    print("\n📊 Top 20 Value Stocks:")
    print("=" * 70)
    top_stocks = scores.nsmallest(20, 'rank')[['rank', 'symbol', 'score', 'sector', 'score_label']]
    for _, row in top_stocks.iterrows():
        print(
            f"{int(row['rank']):3d}. {row['symbol']:6s} | Score: {row['score']:6.2f} | {row['sector']:20s} | {row['score_label']}")

    print(f"\n✅ Scored {saved_count} stocks using model version: {scorer.model_version}")


if __name__ == "__main__":
    main()
