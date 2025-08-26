#!/usr/bin/env python3
"""
AI Dividend Model Scoring
Score stocks using trained dividend model for dividend-focused portfolio selection
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
    filename="score_dividend_model.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

class DividendScorer:
    def __init__(self, model_version=None):
        self.model_version = model_version or self._get_latest_version()
        self.model = None
        self.scaler = None
        self.load_model()

    def _get_latest_version(self):
        """Get the latest dividend model version"""
        query = text("""
        SELECT version
        FROM ai_model_run_log
        WHERE model_type = 'dividend'
        ORDER BY created_at DESC LIMIT 1
        """)
        
        with engine.connect() as conn:
            result = conn.execute(query).fetchone()
            if result:
                return result[0]

        # Fallback to file system
        import glob
        models = glob.glob("models/dividend_model_*.pkl")
        if models:
            latest = sorted(models)[-1]
            return latest.split("_")[-1].replace(".pkl", "")

        # No dividend model exists yet
        return "v1"

    def load_model(self):
        """Load the trained dividend model and scaler"""
        model_path = f"models/dividend_model_{self.model_version}.pkl"
        scaler_path = f"models/dividend_scaler_{self.model_version}.pkl"

        if not os.path.exists(model_path):
            logger.warning(f"Dividend model not found at {model_path}")
            logger.info("Will use fallback dividend scoring methodology")
            self.model = None
            self.scaler = None
            return

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        logger.info(f"Loaded dividend model version: {self.model_version}")

    def load_current_features(self):
        """Load current dividend features for all dividend-paying stocks"""
        query = text("""
        WITH current_dividend_features AS (
            SELECT DISTINCT ON (f.symbol)
                f.symbol,
                CURRENT_DATE as as_of_date,
                
                -- Dividend Metrics
                f.dividendpayout::float / NULLIF(f.netincome, 0) as payout_ratio,
                CASE 
                    WHEN f.dividendpayout > 0 AND m.current_price > 0 
                    THEN (f.dividendpayout::float * 4) / m.current_price  -- Annualized yield
                    ELSE 0 
                END as dividend_yield,
                f.free_cf::float / NULLIF(f.dividendpayout, 0) as fcf_dividend_coverage,
                f.operatingcashflow::float / NULLIF(f.dividendpayout, 0) as ocf_dividend_coverage,
                
                -- Dividend Growth (year-over-year)
                (f.dividendpayout - f2.dividendpayout)::float / NULLIF(f2.dividendpayout, 0) as dividend_growth_1yr,
                
                -- Quality Metrics for Dividend Sustainability
                f.netincome::float / NULLIF(f.totalshareholderequity, 0) as roe,
                f.netincome::float / NULLIF(f.totalassets, 0) as roa,
                f.netincome::float / NULLIF(f.totalrevenue, 0) as net_margin,
                f.grossprofit::float / NULLIF(f.totalrevenue, 0) as gross_margin,
                f.operatingcashflow::float / NULLIF(f.totalrevenue, 0) as ocf_margin,
                
                -- Financial Stability 
                f.totalliabilities::float / NULLIF(f.totalshareholderequity, 0) as debt_to_equity,
                f.totalshareholderequity::float / NULLIF(f.totalassets, 0) as equity_ratio,
                (f.totalassets - f.totalliabilities)::float / NULLIF(f.totalassets, 0) as solvency_ratio,
                
                -- Business Quality
                f.free_cf::float / NULLIF(f.totalrevenue, 0) as fcf_margin,
                f.totalrevenue::float / NULLIF(f.totalassets, 0) as asset_turnover,
                
                -- Growth Sustainability
                (f.totalrevenue - f2.totalrevenue)::float / NULLIF(f2.totalrevenue, 0) as revenue_growth,
                (f.netincome - f2.netincome)::float / NULLIF(ABS(f2.netincome), 0) as earnings_growth,
                
                -- Market Context
                m.pe_ratio,
                LOG(m.market_cap + 1) as log_market_cap,
                m.beta,
                
                -- Sector Classifications (dividend-paying sectors)
                CASE WHEN m.sector IN ('Utilities', 'Consumer Staples', 'Real Estate') THEN 1 ELSE 0 END as is_dividend_sector,
                CASE WHEN m.sector = 'Financial Services' THEN 1 ELSE 0 END as is_financial,
                CASE WHEN m.sector = 'Energy' THEN 1 ELSE 0 END as is_energy,
                CASE WHEN m.sector = 'Telecommunications' THEN 1 ELSE 0 END as is_telecom,
                
                -- Dividend History (stability indicator)
                COUNT(f3.dividendpayout) as dividend_history_years,
                
                -- Context for display
                m.company_name,
                m.sector,
                m.market_cap,
                m.current_price
                
            FROM fundamentals_annual f
            JOIN mv_symbol_with_metadata m ON f.symbol = m.symbol
            LEFT JOIN fundamentals_annual f2 ON f.symbol = f2.symbol 
                AND f2.fiscal_date = (
                    SELECT MAX(fiscal_date) 
                    FROM fundamentals_annual ff2 
                    WHERE ff2.symbol = f.symbol 
                    AND ff2.fiscal_date < f.fiscal_date
                )
            LEFT JOIN fundamentals_annual f3 ON f.symbol = f3.symbol 
                AND f3.dividendpayout > 0
                AND f3.fiscal_date >= f.fiscal_date - INTERVAL '10 years'
            WHERE f.dividendpayout IS NOT NULL
                AND f.dividendpayout > 0  -- Only dividend-paying companies
                AND f.netincome IS NOT NULL
                AND f.netincome > 0
                AND f.free_cf IS NOT NULL
                AND m.market_cap > 1000000000  -- $1B minimum
                AND m.current_price > 5  -- Avoid penny stocks
            GROUP BY f.symbol, f.fiscal_date, f.dividendpayout, f.netincome, f.free_cf, 
                     f.operatingcashflow, f.totalshareholderequity, f.totalassets, 
                     f.totalrevenue, f.grossprofit, f.totalliabilities,
                     f2.dividendpayout, f2.totalrevenue, f2.netincome,
                     m.pe_ratio, m.market_cap, m.beta, m.company_name, 
                     m.sector, m.current_price
            HAVING COUNT(f3.dividendpayout) >= 3  -- At least 3 years of dividends
            ORDER BY f.symbol, f.fiscal_date DESC
        )
        SELECT *
        FROM current_dividend_features
        WHERE payout_ratio BETWEEN 0.05 AND 0.95  -- Reasonable payout ratios
          AND dividend_yield BETWEEN 0.01 AND 0.15  -- 1% to 15% yield
          AND roe > 0.05  -- At least 5% ROE
          AND debt_to_equity < 3  -- Not overleveraged
        ORDER BY symbol
        """)

        df = pd.read_sql(query, engine)
        
        if len(df) == 0:
            logger.warning("No current dividend stocks found for scoring")
            return pd.DataFrame()
        
        logger.info(f"Loaded {len(df)} dividend stocks for scoring")
        
        # Handle missing values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with median
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in ['symbol', 'as_of_date']:
                df[col] = df[col].fillna(df[col].median())
        
        return df

    def calculate_dividend_scores(self, df):
        """Calculate dividend scores using model or fallback methodology"""
        
        if self.model is not None and self.scaler is not None:
            return self._calculate_ml_scores(df)
        else:
            return self._calculate_fallback_scores(df)

    def _calculate_ml_scores(self, df):
        """Calculate scores using trained ML model"""
        # Prepare features (same as training)
        feature_columns = [col for col in df.columns 
                          if col not in ['symbol', 'as_of_date', 'company_name', 'sector', 'market_cap', 'current_price']]
        
        X = df[feature_columns].values
        X_scaled = self.scaler.transform(X)
        
        # Predict forward returns
        predicted_returns = self.model.predict(X_scaled)
        
        # Convert to scores (0-100 scale)
        scores = ((predicted_returns - predicted_returns.min()) / 
                 (predicted_returns.max() - predicted_returns.min()) * 100)
        
        result_df = df[['symbol', 'as_of_date', 'company_name', 'sector', 'market_cap', 
                       'dividend_yield', 'payout_ratio', 'roe', 'debt_to_equity']].copy()
        result_df['score'] = scores
        result_df['predicted_return'] = predicted_returns
        
        logger.info(f"ML dividend scores calculated for {len(result_df)} stocks")
        return result_df

    def _calculate_fallback_scores(self, df):
        """Fallback dividend scoring when no ML model is available"""
        logger.info("Using fallback dividend scoring methodology")
        
        scores = []
        
        for _, row in df.iterrows():
            score = 0
            
            # Dividend Yield (higher is better, but not too high)
            if 0.02 <= row['dividend_yield'] <= 0.08:  # 2-8% sweet spot
                score += min(row['dividend_yield'] * 100, 30)
            elif row['dividend_yield'] > 0.08:  # Penalty for unsustainable high yields
                score += 20 - (row['dividend_yield'] - 0.08) * 200
            
            # Payout Ratio (sustainable range)
            if 0.3 <= row['payout_ratio'] <= 0.7:  # 30-70% optimal
                score += (0.6 - abs(row['payout_ratio'] - 0.5)) * 40
            
            # ROE Quality (earnings support)
            if row['roe'] > 0.10:  # At least 10% ROE
                score += min(row['roe'] * 100, 30) * 0.3
            
            # Free Cash Flow Coverage
            if row['fcf_dividend_coverage'] > 1.5:  # Good coverage
                score += min(row['fcf_dividend_coverage'] * 5, 20)
            
            # Financial Stability (low debt)
            if row['debt_to_equity'] < 0.6:
                score += (0.6 - row['debt_to_equity']) * 15
            elif row['debt_to_equity'] > 1.5:  # Penalty for high debt
                score -= (row['debt_to_equity'] - 1.5) * 10
            
            # Dividend Growth Bonus
            if row['dividend_growth_1yr'] > 0.05:  # >5% dividend growth
                score += min(row['dividend_growth_1yr'] * 50, 15)
            
            # Sector Bonus (traditional dividend sectors)
            if row['is_dividend_sector'] == 1:
                score += 5
            
            # Size stability bonus (large caps more stable)
            if row['market_cap'] > 10000000000:  # >$10B
                score += 3
            
            scores.append(max(score, 0))  # No negative scores
        
        result_df = df[['symbol', 'as_of_date', 'company_name', 'sector', 'market_cap',
                       'dividend_yield', 'payout_ratio', 'roe', 'debt_to_equity']].copy()
        result_df['score'] = scores
        result_df['predicted_return'] = None  # No ML prediction
        
        logger.info(f"Fallback dividend scores calculated for {len(result_df)} stocks")
        return result_df

    def save_scores(self, scored_df):
        """Save dividend scores to database"""
        if len(scored_df) == 0:
            logger.warning("No dividend scores to save")
            return False

        try:
            with engine.connect() as conn:
                # Clear existing scores for today
                conn.execute(text("""
                    DELETE FROM ai_dividend_scores 
                    WHERE as_of_date = CURRENT_DATE
                """))
                
                # Calculate percentiles
                scored_df['percentile'] = scored_df['score'].rank(pct=True)
                
                # Insert new scores
                for _, row in scored_df.iterrows():
                    conn.execute(text("""
                        INSERT INTO ai_dividend_scores (
                            symbol, as_of_date, score, percentile, 
                            predicted_return, model_version, created_at
                        ) VALUES (
                            :symbol, :as_of_date, :score, :percentile,
                            :predicted_return, :model_version, CURRENT_TIMESTAMP
                        )
                    """), {
                        'symbol': row['symbol'],
                        'as_of_date': row['as_of_date'],
                        'score': float(row['score']),
                        'percentile': float(row['percentile']),
                        'predicted_return': float(row['predicted_return']) if row['predicted_return'] is not None else None,
                        'model_version': self.model_version
                    })
                
                conn.commit()
                logger.info(f"Saved {len(scored_df)} dividend scores to database")
                return True
                
        except Exception as e:
            logger.error(f"Error saving dividend scores: {e}")
            return False

    def display_top_picks(self, scored_df, top_n=20):
        """Display top dividend picks"""
        if len(scored_df) == 0:
            print("No dividend scores available")
            return
        
        top_picks = scored_df.nlargest(top_n, 'score')
        
        print(f"\nTOP {top_n} DIVIDEND PICKS:")
        print("=" * 80)
        print(f"{'Rank':>4} {'Symbol':>8} {'Company':>25} {'Score':>8} {'Yield':>7} {'Payout':>7} {'ROE':>6}")
        print("-" * 80)
        
        for i, (_, row) in enumerate(top_picks.iterrows(), 1):
            company = row['company_name'][:22] + "..." if len(row['company_name']) > 25 else row['company_name']
            yield_str = f"{row['dividend_yield']:.1%}" if row['dividend_yield'] else 'N/A'
            payout_str = f"{row['payout_ratio']:.1%}" if row['payout_ratio'] else 'N/A'
            roe_str = f"{row['roe']:.1%}" if row['roe'] else 'N/A'
            
            print(f"{i:>4d} {row['symbol']:>8} {company:>25} {row['score']:>8.1f} {yield_str:>7} {payout_str:>7} {roe_str:>6}")

def main():
    """Score all dividend stocks and save results"""
    print("AI DIVIDEND MODEL SCORING")
    print("=" * 50)
    
    scorer = DividendScorer()
    
    try:
        # Load current features
        print("Loading current dividend stock features...")
        df = scorer.load_current_features()
        
        if len(df) == 0:
            print("No dividend-paying stocks found for scoring")
            return False
        
        print(f"Found {len(df)} dividend-paying stocks")
        
        # Calculate scores
        print("Calculating dividend scores...")
        scored_df = scorer.calculate_dividend_scores(df)
        
        # Display results
        scorer.display_top_picks(scored_df, top_n=20)
        
        # Save to database
        print(f"\nSaving dividend scores to database...")
        success = scorer.save_scores(scored_df)
        
        if success:
            print(f"Successfully saved {len(scored_df)} dividend scores!")
            print("Ready for dividend portfolio generation.")
        else:
            print("Failed to save dividend scores to database")
            return False
        
        return True
        
    except Exception as e:
        print(f"Scoring error: {e}")
        logger.error(f"Dividend scoring failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()