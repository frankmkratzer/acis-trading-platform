# =====================================
# score_ai_growth_model.py
# =====================================
"""
#!/usr/bin/env python3
# File: score_ai_growth_model.py
# Purpose: Score stocks using trained growth model
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
    filename="score_growth_model.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class GrowthScorer:
    def __init__(self, model_version=None):
        self.model_version = model_version or self._get_latest_version()
        self.model = None
        self.scaler = None
        self.load_model()

    def _get_latest_version(self):
        """Get the latest growth model version"""
        query = text("""
                     SELECT version
                     FROM ai_model_run_log
                     WHERE model_type = 'growth'
                     ORDER BY created_at DESC LIMIT 1
                     """)
        with engine.connect() as conn:
            result = conn.execute(query).fetchone()
            if result:
                return result[0]

        # Fallback to file system
        import glob
        models = glob.glob("models/growth_model_*.pkl")
        if models:
            latest = sorted(models)[-1]
            return latest.split("_")[-1].replace(".pkl", "")

        # If no model exists, use value model for now
        return "v1"

    def load_model(self):
        """Load the trained growth model and scaler"""
        model_path = f"models/growth_model_{self.model_version}.pkl"
        scaler_path = f"models/growth_scaler_{self.model_version}.pkl"

        # If growth model doesn't exist, create a simple one
        if not os.path.exists(model_path):
            logger.warning(f"Growth model not found, using fallback scoring")
            self.model = None
            self.scaler = None
            return

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        logger.info(f"Loaded growth model version: {self.model_version}")

    def load_current_features(self):
        """Load current growth features for all stocks"""
        query = text("""
                     WITH latest_fundamentals AS (SELECT DISTINCT
                     ON (f1.symbol)
                         f1.symbol,
                         f1.fiscal_date,
                         f1.totalrevenue,
                         f1.netincome,
                         f1.grossprofit,
                         f1.operatingcashflow,
                         f1.free_cf,
                         f1.totalshareholderequity,
                         f1.totalassets,
                         f1.totalliabilities,
                         f2.totalrevenue as revenue_1y_ago,
                         f2.netincome as netincome_1y_ago,
                         f3.totalrevenue as revenue_3y_ago,
                         f3.netincome as netincome_3y_ago
                     FROM fundamentals_annual f1
                         LEFT JOIN fundamentals_annual f2
                     ON f1.symbol = f2.symbol
                         AND f2.fiscal_date = f1.fiscal_date - INTERVAL '1 year'
                         LEFT JOIN fundamentals_annual f3 ON f1.symbol = f3.symbol
                         AND f3.fiscal_date = f1.fiscal_date - INTERVAL '3 years'
                     ORDER BY f1.symbol, f1.fiscal_date DESC
                         ),
                         features AS (
                     SELECT
                         f.symbol, CURRENT_DATE as as_of_date,

                         -- Growth rates
                         (f.totalrevenue - f.revenue_1y_ago):: float / NULLIF (f.revenue_1y_ago, 0) as revenue_growth_1y, CASE
                         WHEN f.revenue_3y_ago > 0 AND f.totalrevenue > 0
                         THEN (POWER(f.totalrevenue:: float / f.revenue_3y_ago, 1.0/3) - 1)
                         ELSE 0
                         END as revenue_cagr_3y, (f.netincome - f.netincome_1y_ago):: float / NULLIF (ABS(f.netincome_1y_ago), 0) as earnings_growth_1y, CASE
                         WHEN f.netincome_3y_ago > 0 AND f.netincome > 0
                         THEN (POWER(f.netincome:: float / f.netincome_3y_ago, 1.0/3) - 1)
                         ELSE 0
                         END as earnings_cagr_3y, 0 as ocf_growth_1y, -- Placeholder
                         0 as fcf_growth_1y,                          -- Placeholder

                         -- Margins
                         f.grossprofit:: float / NULLIF (f.totalrevenue, 0) as gross_margin, f.netincome:: float / NULLIF (f.totalrevenue, 0) as net_margin, f.operatingcashflow:: float / NULLIF (f.totalrevenue, 0) as ocf_margin,

                         -- Quality metrics
                         f.netincome:: float / NULLIF (f.totalshareholderequity, 0) as roe, f.netincome:: float / NULLIF (f.totalassets, 0) as roa, f.totalrevenue:: float / NULLIF (f.totalassets, 0) as asset_turnover,

                         -- Balance sheet
                         f.totalshareholderequity:: float / NULLIF (f.totalassets, 0) as equity_ratio, (f.totalassets - f.totalliabilities):: float / NULLIF (f.totalassets, 0) as solvency_ratio,

                         -- Market metrics
                         m.pe_ratio, m.peg_ratio, LOG(m.market_cap + 1) as log_market_cap,

                         -- Sectors
                         CASE WHEN m.sector = 'Technology' THEN 1 ELSE 0 END as is_tech, CASE WHEN m.sector = 'Healthcare' THEN 1 ELSE 0 END as is_healthcare, CASE WHEN m.sector = 'Consumer Cyclical' THEN 1 ELSE 0 END as is_consumer,

                         -- Context
                         m.sector, m.industry, m.market_cap

                     FROM latest_fundamentals f
                         JOIN mv_symbol_with_metadata m
                     ON f.symbol = m.symbol
                     WHERE m.market_cap
                         > 1000000000 -- $1B minimum
                       AND f.totalrevenue
                         > 0
                       AND f.revenue_1y_ago
                         > 0
                         )
                     SELECT *
                     FROM features
                     WHERE revenue_growth_1y > -0.5 -- Not declining too much
                       AND revenue_growth_1y < 5 -- Not unrealistic
                     """)

        df = pd.read_sql(query, engine)

        # Handle missing values
        df = df.replace([np.inf, -np.inf], np.nan)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['symbol', 'as_of_date']:
                df[col] = df[col].fillna(df[col].median() if len(df) > 0 else 0)

        logger.info(f"Loaded features for {len(df)} growth stocks")
        return df

    def score_stocks(self, df):
        """Generate growth scores for all stocks"""

        # If no ML model, use simple scoring
        if self.model is None:
            return self._fallback_scoring(df)

        feature_cols = [
            'revenue_growth_1y', 'revenue_cagr_3y',
            'earnings_growth_1y', 'earnings_cagr_3y',
            'ocf_growth_1y', 'fcf_growth_1y',
            'gross_margin', 'net_margin', 'ocf_margin',
            'roe', 'roa', 'asset_turnover',
            'equity_ratio', 'solvency_ratio',
            'pe_ratio', 'peg_ratio', 'log_market_cap',
            'is_tech', 'is_healthcare', 'is_consumer'
        ]

        # Use available features
        available_features = [col for col in feature_cols if col in df.columns]

        # Prepare features
        X = df[available_features].values

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

        # Normalize scores
        results['score'] = 100 * results['raw_score'].rank(pct=True)
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
        results['score_type'] = 'growth'
        results['fetched_at'] = datetime.now()

        return results

    def _fallback_scoring(self, df):
        """Simple growth scoring when no ML model is available"""

        # Calculate composite growth score
        df['growth_score'] = (
                df['revenue_growth_1y'].fillna(0) * 0.3 +
                df['revenue_cagr_3y'].fillna(0) * 0.2 +
                df['earnings_growth_1y'].fillna(0).clip(-1, 2) * 0.3 +
                df['roe'].fillna(0) * 0.2
        )

        # Create results
        results = pd.DataFrame({
            'symbol': df['symbol'],
            'as_of_date': df['as_of_date'],
            'raw_score': df['growth_score'],
            'sector': df['sector'],
            'industry': df['industry'],
            'market_cap': df['market_cap']
        })

        # Normalize scores
        results['score'] = 100 * results['raw_score'].rank(pct=True)
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
        results['model_version'] = 'fallback_v1'
        results['score_type'] = 'growth'
        results['fetched_at'] = datetime.now()

        return results

    def save_scores(self, scores_df):
        """Save growth scores to database"""
        db_columns = [
            'symbol', 'as_of_date', 'score', 'percentile',
            'score_label', 'rank', 'model_version', 'score_type', 'fetched_at'
        ]

        scores_to_save = scores_df[db_columns].copy()

        with engine.begin() as conn:
            # Delete existing scores for this date
            conn.execute(text("""
                              DELETE
                              FROM ai_growth_scores
                              WHERE as_of_date = :as_of_date
                              """), {"as_of_date": scores_df['as_of_date'].iloc[0]})

            # Insert new scores
            scores_to_save.to_sql(
                'ai_growth_scores',
                conn,
                if_exists='append',
                index=False,
                method='multi'
            )

        logger.info(f"Saved {len(scores_to_save)} growth scores to database")

        # Save to CSV
        csv_path = f"scores/growth_scores_{datetime.now().strftime('%Y%m%d')}.csv"
        os.makedirs("scores", exist_ok=True)
        scores_df.to_csv(csv_path, index=False)

        return len(scores_to_save)


def main():
    scorer = GrowthScorer()

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
    print("\n📈 Top 20 Growth Stocks:")
    print("=" * 70)
    top_stocks = scores.nsmallest(20, 'rank')[['rank', 'symbol', 'score', 'sector', 'score_label']]
    for _, row in top_stocks.iterrows():
        print(
            f"{int(row['rank']):3d}. {row['symbol']:6s} | Score: {row['score']:6.2f} | {row['sector']:20s} | {row['score_label']}")

    print(f"\n✅ Scored {saved_count} stocks for growth strategy")


if __name__ == "__main__":
    main()
