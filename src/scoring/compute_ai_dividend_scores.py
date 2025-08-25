# =====================================
# compute_ai_dividend_scores.py
# =====================================
"""
#!/usr/bin/env python3
# File: compute_ai_dividend_scores.py
# Purpose: Score dividend stocks based on yield, growth, and sustainability
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, date
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import logging

load_dotenv()
engine = create_engine(os.getenv("POSTGRES_URL"))

logging.basicConfig(
    filename="dividend_scores.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class DividendScorer:
    def __init__(self):
        self.min_market_cap = 5e9  # $5B for dividend stocks
        self.min_yield = 0.01  # 1% minimum yield

    def load_dividend_data(self):
        """Load comprehensive dividend data"""
        query = text("""
                     WITH dividend_metrics AS (SELECT d.symbol,
                                                      COUNT(DISTINCT EXTRACT(YEAR FROM d.ex_date)) as years_paying,
                                                      MAX(d.ex_date)                               as last_dividend_date,
                                                      SUM(CASE WHEN d.ex_date >= CURRENT_DATE - INTERVAL '1 year' THEN
                                                          d.dividend ELSE 0 END)                   as dividends_ttm,
                                                      SUM(CASE WHEN d.ex_date >= CURRENT_DATE - INTERVAL '3 years' THEN
                                                          d.dividend ELSE 0 END) / 3               as avg_dividend_3y,

                                                      -- Check for cuts
                                                      BOOL_OR(
                                                              LAG(d.dividend) OVER (PARTITION BY d.symbol ORDER BY d.ex_date) > d.dividend
                                                      )                                            as has_cut_dividend

                                               FROM dividend_history d
                                               WHERE d.ex_date >= CURRENT_DATE - INTERVAL '10 years'
                     GROUP BY d.symbol
                         ),
                         growth_metrics AS (
                     SELECT
                         symbol, div_cagr_1y, div_cagr_3y, div_cagr_5y, dividend_cut_detected
                     FROM dividend_growth_scores
                     WHERE as_of_date = (SELECT MAX (as_of_date) FROM dividend_growth_scores)
                         )
                         , fundamental_metrics AS (
                     SELECT DISTINCT
                     ON (symbol)
                         symbol,
                         netincome,
                         operatingcashflow,
                         free_cf,
                         totalrevenue,
                         totalshareholderequity
                     FROM fundamentals_annual
                     ORDER BY symbol, fiscal_date DESC
                         ),
                         combined AS (
                     SELECT
                         m.symbol, m.sector, m.industry, m.market_cap,

                         -- Dividend metrics
                         d.dividends_ttm, d.years_paying, d.last_dividend_date, COALESCE (d.has_cut_dividend, FALSE) as has_cut_dividend,

                         -- Yield
                         d.dividends_ttm / (p.adjusted_close * s.shares_outstanding) as current_yield, m.dividend_yield as reported_yield,

                         -- Growth
                         COALESCE (g.div_cagr_1y, 0) as div_growth_1y, COALESCE (g.div_cagr_3y, 0) as div_growth_3y, COALESCE (g.div_cagr_5y, 0) as div_growth_5y,

                         -- Payout ratios
                         d.dividends_ttm * s.shares_outstanding / NULLIF (f.netincome, 0) as payout_ratio, d.dividends_ttm * s.shares_outstanding / NULLIF (f.free_cf, 0) as fcf_payout_ratio,

                         -- Quality
                         f.netincome:: float / NULLIF (f.totalshareholderequity, 0) as roe, f.free_cf:: float / NULLIF (f.totalrevenue, 0) as fcf_margin,

                         -- Price
                         p.adjusted_close as current_price

                     FROM mv_symbol_with_metadata m
                         LEFT JOIN dividend_metrics d
                     ON m.symbol = d.symbol
                         LEFT JOIN growth_metrics g ON m.symbol = g.symbol
                         LEFT JOIN fundamental_metrics f ON m.symbol = f.symbol
                         LEFT JOIN (
                         SELECT DISTINCT ON (symbol)
                         symbol,
                         adjusted_close
                         FROM stock_eod_daily
                         ORDER BY symbol, trade_date DESC
                         ) p ON m.symbol = p.symbol
                         LEFT JOIN (
                         -- Approximate shares outstanding
                         SELECT
                         symbol,
                         market_cap / NULLIF (adjusted_close, 0) as shares_outstanding
                         FROM (
                         SELECT DISTINCT ON (m.symbol)
                         m.symbol,
                         m.market_cap,
                         p.adjusted_close
                         FROM mv_symbol_with_metadata m
                         JOIN stock_eod_daily p ON m.symbol = p.symbol
                         ORDER BY m.symbol, p.trade_date DESC
                         ) t
                         ) s ON m.symbol = s.symbol

                     WHERE m.market_cap >= :min_cap
                       AND d.dividends_ttm
                         > 0
                       AND d.years_paying >= 3 -- At least 3 years of dividends
                         )
                     SELECT *
                     FROM combined
                     WHERE current_yield >= :min_yield
                        OR reported_yield >= :min_yield
                     """)

        df = pd.read_sql(
            query,
            engine,
            params={
                'min_cap': self.min_market_cap,
                'min_yield': self.min_yield
            }
        )

        logger.info(f"Loaded {len(df)} dividend-paying stocks")
        return df

    def calculate_scores(self, df):
        """Calculate dividend scores"""

        # Use reported yield if current yield calculation failed
        df['dividend_yield'] = df['current_yield'].fillna(df['reported_yield'])

        # Cap extreme values
        df['dividend_yield'] = df['dividend_yield'].clip(0, 0.20)  # Cap at 20%
        df['payout_ratio'] = df['payout_ratio'].clip(0, 2)  # Cap at 200%

        # Score components
        scores = pd.DataFrame(index=df.index)

        # 1. Yield Score (30% weight)
        # Higher yield is better, but extreme yields are suspicious
        optimal_yield = 0.04  # 4% is optimal
        scores['yield_score'] = df['dividend_yield'].apply(
            lambda x: min(100, 100 * (x / optimal_yield)) if x <= optimal_yield
            else max(0, 100 - 20 * (x - optimal_yield))  # Penalty for very high yields
        )

        # 2. Growth Score (30% weight)
        # Average of different growth periods
        scores['growth_score'] = (
                df['div_growth_1y'].clip(-0.2, 0.5) * 100 * 0.4 +
                df['div_growth_3y'].clip(-0.1, 0.3) * 100 * 0.4 +
                df['div_growth_5y'].clip(-0.05, 0.2) * 100 * 0.2
        )

        # 3. Sustainability Score (25% weight)
        # Based on payout ratio and FCF coverage
        scores['sustainability_score'] = 0

        # Payout ratio component (lower is better)
        payout_component = df['payout_ratio'].apply(
            lambda x: 100 if x <= 0.4
            else 80 if x <= 0.6
            else 60 if x <= 0.8
            else 20 if x <= 1.0
            else 0
        )

        # FCF coverage component
        fcf_component = df['fcf_payout_ratio'].apply(
            lambda x: 100 if pd.notna(x) and 0 < x <= 0.5
            else 70 if pd.notna(x) and x <= 0.7
            else 40 if pd.notna(x) and x <= 1.0
            else 0
        )

        scores['sustainability_score'] = (payout_component * 0.6 + fcf_component * 0.4)

        # 4. Consistency Score (15% weight)
        scores['consistency_score'] = (
                df['years_paying'].clip(0, 20) * 5 +  # Up to 100 for 20+ years
                (~df['has_cut_dividend']).astype(int) * 20  # Bonus for no cuts
        ).clip(0, 100)

        # Calculate composite score
        df['score'] = (
                scores['yield_score'] * 0.30 +
                scores['growth_score'] * 0.30 +
                scores['sustainability_score'] * 0.25 +
                scores['consistency_score'] * 0.15
        )

        # Add score components to dataframe
        for col in scores.columns:
            df[col] = scores[col]

        # Calculate percentiles and ranks
        df['percentile'] = df['score'].rank(pct=True)
        df['rank'] = df['score'].rank(ascending=False, method='min').astype(int)

        # Assign labels
        df['score_label'] = pd.cut(
            df['percentile'],
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=['D', 'C', 'B', 'A', 'A+']
        )

        return df

    def save_scores(self, df):
        """Save dividend scores to database"""

        # Prepare data for saving
        scores_df = pd.DataFrame({
            'symbol': df['symbol'],
            'as_of_date': date.today(),
            'score': df['score'],
            'percentile': df['percentile'],
            'score_label': df['score_label'],
            'rank': df['rank'],
            'model_version': 'v1',
            'score_type': 'dividend',
            'fetched_at': datetime.now()
        })

        with engine.begin() as conn:
            # Delete existing scores for today
            conn.execute(text("""
                              DELETE
                              FROM ai_dividend_scores
                              WHERE as_of_date = :as_of_date
                              """), {"as_of_date": date.today()})

            # Insert new scores
            scores_df.to_sql(
                'ai_dividend_scores',
                conn,
                if_exists='append',
                index=False,
                method='multi'
            )

        logger.info(f"Saved {len(scores_df)} dividend scores")

        # Save detailed results to CSV
        csv_path = f"scores/dividend_scores_{datetime.now().strftime('%Y%m%d')}.csv"
        os.makedirs("scores", exist_ok=True)
        df.to_csv(csv_path, index=False)

        return len(scores_df)


def main():
    scorer = DividendScorer()

    # Load dividend data
    df = scorer.load_dividend_data()

    if df.empty:
        logger.error("No dividend stocks found")
        return

    # Calculate scores
    scored_df = scorer.calculate_scores(df)

    # Save scores
    saved_count = scorer.save_scores(scored_df)

    # Print top dividend stocks
    print("\n💰 Top 20 Dividend Stocks:")
    print("=" * 80)
    print(f"{'Rank':>4} {'Symbol':<8} {'Score':>7} {'Yield':>7} {'Growth':>7} {'Payout':>7} {'Sector':<20}")
    print("-" * 80)

    top_stocks = scored_df.nsmallest(20, 'rank')
    for _, row in top_stocks.iterrows():
        print(f"{int(row['rank']):4d}. {row['symbol']:<8} "
              f"{row['score']:6.1f} "
              f"{row['dividend_yield'] * 100:6.2f}% "
              f"{row['div_growth_3y'] * 100:6.1f}% "
              f"{row['payout_ratio'] * 100 if pd.notna(row['payout_ratio']) else 0:6.1f}% "
              f"{row['sector'][:20] if pd.notna(row['sector']) else 'N/A':<20}")

    print(f"\n✅ Scored {saved_count} dividend stocks")


if __name__ == "__main__":
    main()