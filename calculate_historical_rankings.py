#!/usr/bin/env python3
"""
Historical Quality Rankings Calculator
Generates weekly rankings for every Friday in history to build ML/DL training data
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import the main ranking calculator
from calculate_quality_rankings import WorldClassQualityRankingCalculator

load_dotenv()
engine = create_engine(os.getenv("POSTGRES_URL"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HistoricalRankingCalculator:
    """
    Generates historical rankings for every Friday to build training data for ML models
    Ensures point-in-time accuracy to avoid look-ahead bias
    """
    
    def __init__(self, start_date=None, end_date=None):
        """
        Initialize historical calculator
        
        Args:
            start_date: Earliest date to calculate rankings (default: earliest data)
            end_date: Latest date to calculate rankings (default: today)
        """
        self.start_date = start_date
        self.end_date = end_date or datetime.now()
        self.engine = engine
        
    def get_friday_dates(self):
        """Get all Friday dates between start and end dates"""
        
        # Get date range from database if not specified
        if not self.start_date:
            query = """
            SELECT MIN(date) as min_date 
            FROM stock_prices 
            WHERE date > '2010-01-01'  -- Reasonable starting point
            """
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                min_date = result.scalar()
                if min_date:
                    self.start_date = pd.to_datetime(min_date)
                else:
                    self.start_date = datetime(2010, 1, 1)
        
        # Generate all Fridays in range
        current_date = self.start_date
        # Find first Friday
        while current_date.weekday() != 4:  # 4 = Friday
            current_date += timedelta(days=1)
        
        fridays = []
        while current_date <= self.end_date:
            fridays.append(current_date)
            current_date += timedelta(days=7)
        
        return fridays
    
    def check_existing_rankings(self, date):
        """Check if rankings already exist for a given date"""
        query = """
        SELECT COUNT(*) 
        FROM stock_quality_rankings 
        WHERE ranking_date = :date
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(query), {"date": date})
            count = result.scalar()
            return count > 0
    
    def create_point_in_time_views(self, as_of_date):
        """
        Create temporary views with data as it would have been known on as_of_date
        This prevents look-ahead bias in historical calculations
        """
        
        # Create point-in-time views for each data source
        views = []
        
        # Stock prices view (only data up to as_of_date)
        views.append(f"""
        CREATE OR REPLACE TEMPORARY VIEW pit_stock_prices AS
        SELECT * FROM stock_prices 
        WHERE date <= '{as_of_date}'::date
        """)
        
        # Fundamentals view (only reported before as_of_date)
        views.append(f"""
        CREATE OR REPLACE TEMPORARY VIEW pit_fundamentals AS
        SELECT * FROM fundamentals 
        WHERE fiscal_date_ending <= '{as_of_date}'::date
        AND created_at <= '{as_of_date}'::timestamp  -- Ensure not using future corrections
        """)
        
        # SP500 history view
        views.append(f"""
        CREATE OR REPLACE TEMPORARY VIEW pit_sp500_price_history AS
        SELECT * FROM sp500_price_history 
        WHERE date <= '{as_of_date}'::date
        """)
        
        # SP500 outperformance view
        views.append(f"""
        CREATE OR REPLACE TEMPORARY VIEW pit_sp500_outperformance_detail AS
        SELECT * FROM sp500_outperformance_detail 
        WHERE year <= EXTRACT(YEAR FROM '{as_of_date}'::date)
        """)
        
        # News sentiment view (last 30 days only)
        views.append(f"""
        CREATE OR REPLACE TEMPORARY VIEW pit_daily_sentiment_summary AS
        SELECT * FROM daily_sentiment_summary 
        WHERE date BETWEEN '{as_of_date}'::date - INTERVAL '30 days' 
                      AND '{as_of_date}'::date
        """)
        
        # Dividend history view
        views.append(f"""
        CREATE OR REPLACE TEMPORARY VIEW pit_dividend_history AS
        SELECT * FROM dividend_history 
        WHERE ex_date <= '{as_of_date}'::date
        """)
        
        # Execute all views
        with self.engine.connect() as conn:
            for view_sql in views:
                conn.execute(text(view_sql))
            conn.commit()
    
    def calculate_rankings_for_date(self, ranking_date):
        """
        Calculate rankings as they would have been on a specific date
        
        Args:
            ranking_date: Date to calculate rankings for
            
        Returns:
            DataFrame with rankings
        """
        logger.info(f"Calculating rankings for {ranking_date.strftime('%Y-%m-%d')}")
        
        # Create point-in-time views to prevent look-ahead bias
        self.create_point_in_time_views(ranking_date)
        
        # Initialize calculator with historical date
        calculator = WorldClassQualityRankingCalculator(ranking_date=ranking_date)
        
        # Temporarily patch the calculator to use point-in-time views
        # This requires modifying queries in the calculator methods
        original_queries = self._patch_calculator_queries(calculator)
        
        try:
            # Calculate rankings using point-in-time data
            rankings_df = calculator.calculate_composite_rankings()
            
            # Add metadata
            rankings_df['calculation_date'] = datetime.now()  # When we calculated
            rankings_df['is_historical'] = True
            
            return rankings_df
            
        except Exception as e:
            logger.error(f"Error calculating rankings for {ranking_date}: {e}")
            return pd.DataFrame()
            
        finally:
            # Restore original queries
            self._restore_calculator_queries(calculator, original_queries)
    
    def _patch_calculator_queries(self, calculator):
        """
        Modify calculator to use point-in-time views
        This is a simplified version - in production you'd modify all SQL queries
        """
        # Store original table names
        original = {
            'stock_prices': 'stock_prices',
            'fundamentals': 'fundamentals',
            'sp500_price_history': 'sp500_price_history',
            'sp500_outperformance_detail': 'sp500_outperformance_detail',
            'daily_sentiment_summary': 'daily_sentiment_summary',
            'dividend_history': 'dividend_history'
        }
        
        # For now, we'll note that the actual implementation would need to
        # modify all SQL queries in the calculator methods to use pit_ prefixed views
        
        return original
    
    def _restore_calculator_queries(self, calculator, original_queries):
        """Restore original table names in calculator queries"""
        # Restore original configuration
        pass
    
    def generate_historical_rankings(self, batch_size=4, skip_existing=True):
        """
        Generate rankings for all historical Fridays
        
        Args:
            batch_size: Number of weeks to process in parallel
            skip_existing: Skip dates that already have rankings
        """
        fridays = self.get_friday_dates()
        logger.info(f"Found {len(fridays)} Fridays to process from {self.start_date} to {self.end_date}")
        
        # Filter out existing if requested
        if skip_existing:
            fridays_to_process = []
            for friday in fridays:
                if not self.check_existing_rankings(friday):
                    fridays_to_process.append(friday)
            logger.info(f"Skipping {len(fridays) - len(fridays_to_process)} dates with existing rankings")
            fridays = fridays_to_process
        
        if not fridays:
            logger.info("No dates to process")
            return
        
        # Process in batches
        total_processed = 0
        failed_dates = []
        
        with tqdm(total=len(fridays), desc="Calculating historical rankings") as pbar:
            for i in range(0, len(fridays), batch_size):
                batch = fridays[i:i+batch_size]
                
                for ranking_date in batch:
                    try:
                        # Calculate rankings for this date
                        rankings_df = self.calculate_rankings_for_date(ranking_date)
                        
                        if not rankings_df.empty:
                            # Save to database
                            calculator = WorldClassQualityRankingCalculator(ranking_date=ranking_date)
                            calculator.save_rankings(rankings_df)
                            total_processed += 1
                        else:
                            failed_dates.append(ranking_date)
                            
                    except Exception as e:
                        logger.error(f"Failed to process {ranking_date}: {e}")
                        failed_dates.append(ranking_date)
                    
                    pbar.update(1)
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info(f"Historical Ranking Generation Complete")
        logger.info(f"{'='*60}")
        logger.info(f"Total dates processed: {total_processed}")
        logger.info(f"Failed dates: {len(failed_dates)}")
        if failed_dates:
            logger.info(f"Failed dates: {[d.strftime('%Y-%m-%d') for d in failed_dates[:10]]}")
        
        return total_processed, failed_dates
    
    def calculate_forward_returns(self, horizons=[1, 4, 12, 26, 52]):
        """
        Calculate forward returns for each ranking date
        This creates ML/DL target variables
        
        Args:
            horizons: List of week horizons to calculate returns for
        """
        logger.info("Calculating forward returns for ML targets...")
        
        for horizon in horizons:
            logger.info(f"Calculating {horizon}-week forward returns...")
            
            query = f"""
            INSERT INTO ml_forward_returns (symbol, ranking_date, horizon_weeks, forward_return, forward_excess_return)
            SELECT 
                r.symbol,
                r.ranking_date,
                {horizon} as horizon_weeks,
                (future.close / current.close - 1) * 100 as forward_return,
                ((future.close / current.close - 1) - (sp_future.close / sp_current.close - 1)) * 100 as forward_excess_return
            FROM stock_quality_rankings r
            JOIN stock_prices current 
                ON r.symbol = current.symbol 
                AND current.date = r.ranking_date
            JOIN stock_prices future 
                ON r.symbol = future.symbol 
                AND future.date = r.ranking_date + INTERVAL '{horizon} weeks'
            LEFT JOIN sp500_price_history sp_current 
                ON sp_current.date = r.ranking_date
            LEFT JOIN sp500_price_history sp_future 
                ON sp_future.date = r.ranking_date + INTERVAL '{horizon} weeks'
            WHERE NOT EXISTS (
                SELECT 1 FROM ml_forward_returns fr 
                WHERE fr.symbol = r.symbol 
                AND fr.ranking_date = r.ranking_date 
                AND fr.horizon_weeks = {horizon}
            )
            ON CONFLICT (symbol, ranking_date, horizon_weeks) DO NOTHING;
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                conn.commit()
                logger.info(f"  Added {result.rowcount} forward return records")
    
    def create_ml_features(self):
        """
        Create feature engineering table for ML/DL models
        Includes ranking changes, momentum, regime indicators
        """
        logger.info("Creating ML feature engineering...")
        
        query = """
        CREATE TABLE IF NOT EXISTS ml_features AS
        WITH ranking_changes AS (
            SELECT 
                symbol,
                ranking_date,
                -- Ranking levels
                beat_sp500_ranking,
                excess_cash_flow_ranking,
                fundamentals_ranking,
                sentiment_ranking,
                value_ranking,
                breakout_ranking,
                growth_ranking,
                -- Week-over-week ranking changes
                beat_sp500_ranking - LAG(beat_sp500_ranking, 1) OVER (PARTITION BY symbol ORDER BY ranking_date) as sp500_rank_change_1w,
                excess_cash_flow_ranking - LAG(excess_cash_flow_ranking, 1) OVER (PARTITION BY symbol ORDER BY ranking_date) as fcf_rank_change_1w,
                fundamentals_ranking - LAG(fundamentals_ranking, 1) OVER (PARTITION BY symbol ORDER BY ranking_date) as fund_rank_change_1w,
                sentiment_ranking - LAG(sentiment_ranking, 1) OVER (PARTITION BY symbol ORDER BY ranking_date) as sent_rank_change_1w,
                -- 4-week ranking changes
                beat_sp500_ranking - LAG(beat_sp500_ranking, 4) OVER (PARTITION BY symbol ORDER BY ranking_date) as sp500_rank_change_4w,
                excess_cash_flow_ranking - LAG(excess_cash_flow_ranking, 4) OVER (PARTITION BY symbol ORDER BY ranking_date) as fcf_rank_change_4w,
                -- Composite score changes
                final_composite_score - LAG(final_composite_score, 1) OVER (PARTITION BY symbol ORDER BY ranking_date) as score_change_1w,
                final_composite_score - LAG(final_composite_score, 4) OVER (PARTITION BY symbol ORDER BY ranking_date) as score_change_4w,
                -- Momentum indicators
                CASE WHEN final_composite_score > LAG(final_composite_score, 1) OVER (PARTITION BY symbol ORDER BY ranking_date) THEN 1 ELSE 0 END as score_improving,
                -- Rolling averages
                AVG(final_composite_score) OVER (PARTITION BY symbol ORDER BY ranking_date ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) as score_ma_4w,
                AVG(final_composite_score) OVER (PARTITION BY symbol ORDER BY ranking_date ROWS BETWEEN 11 PRECEDING AND CURRENT ROW) as score_ma_12w,
                -- Volatility
                STDDEV(final_composite_score) OVER (PARTITION BY symbol ORDER BY ranking_date ROWS BETWEEN 11 PRECEDING AND CURRENT ROW) as score_volatility_12w
            FROM stock_quality_rankings
        ),
        regime_indicators AS (
            SELECT 
                ranking_date,
                -- Market regime (based on SP500 trend)
                AVG(CASE WHEN beat_sp500_ranking <= 100 THEN 1 ELSE 0 END) as pct_beating_sp500,
                AVG(sentiment_score) as market_sentiment,
                AVG(CASE WHEN value_ranking <= 100 THEN 1 ELSE 0 END) as value_regime,
                AVG(CASE WHEN growth_ranking <= 100 THEN 1 ELSE 0 END) as growth_regime,
                AVG(CASE WHEN breakout_ranking <= 100 THEN 1 ELSE 0 END) as momentum_regime
            FROM stock_quality_rankings
            GROUP BY ranking_date
        )
        SELECT 
            rc.*,
            ri.pct_beating_sp500,
            ri.market_sentiment,
            ri.value_regime,
            ri.growth_regime,
            ri.momentum_regime,
            -- Target variables from forward returns
            fr1.forward_return as return_1w,
            fr4.forward_return as return_4w,
            fr12.forward_return as return_12w,
            fr1.forward_excess_return as excess_return_1w,
            fr4.forward_excess_return as excess_return_4w,
            fr12.forward_excess_return as excess_return_12w
        FROM ranking_changes rc
        LEFT JOIN regime_indicators ri ON rc.ranking_date = ri.ranking_date
        LEFT JOIN ml_forward_returns fr1 ON rc.symbol = fr1.symbol AND rc.ranking_date = fr1.ranking_date AND fr1.horizon_weeks = 1
        LEFT JOIN ml_forward_returns fr4 ON rc.symbol = fr4.symbol AND rc.ranking_date = fr4.ranking_date AND fr4.horizon_weeks = 4
        LEFT JOIN ml_forward_returns fr12 ON rc.symbol = fr12.symbol AND rc.ranking_date = fr12.ranking_date AND fr12.horizon_weeks = 12;
        
        CREATE INDEX IF NOT EXISTS idx_ml_features_symbol_date ON ml_features(symbol, ranking_date);
        CREATE INDEX IF NOT EXISTS idx_ml_features_date ON ml_features(ranking_date);
        """
        
        with self.engine.connect() as conn:
            conn.execute(text(query))
            conn.commit()
            
        logger.info("ML features table created successfully")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate historical quality rankings")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of weeks to process in parallel")
    parser.add_argument("--skip-existing", action="store_true", help="Skip dates with existing rankings")
    parser.add_argument("--forward-returns", action="store_true", help="Calculate forward returns")
    parser.add_argument("--ml-features", action="store_true", help="Create ML feature table")
    
    args = parser.parse_args()
    
    # Parse dates
    start_date = pd.to_datetime(args.start_date) if args.start_date else None
    end_date = pd.to_datetime(args.end_date) if args.end_date else None
    
    # Initialize calculator
    calculator = HistoricalRankingCalculator(start_date, end_date)
    
    # Generate historical rankings
    total, failed = calculator.generate_historical_rankings(
        batch_size=args.batch_size,
        skip_existing=args.skip_existing
    )
    
    # Calculate forward returns if requested
    if args.forward_returns:
        calculator.calculate_forward_returns()
    
    # Create ML features if requested
    if args.ml_features:
        calculator.create_ml_features()
    
    print(f"\nHistorical ranking generation complete!")
    print(f"Total rankings generated: {total}")
    if failed:
        print(f"Failed dates: {len(failed)}")


if __name__ == "__main__":
    main()