#!/usr/bin/env python3
"""
Calculate ML-focused forward returns with risk metrics
Uses ml_forward_returns table (not the simple forward_returns table)
Computes returns, volatility, drawdown over multiple horizons for ranked stocks
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import logging

load_dotenv()
engine = create_engine(os.getenv("POSTGRES_URL"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_forward_returns(horizons=[1, 4, 12, 26, 52], batch_size=1000):
    """
    Calculate forward returns for all stocks with rankings
    
    Args:
        horizons: List of week horizons to calculate
        batch_size: Number of records to process at once
    """
    logger.info(f"Calculating forward returns for horizons: {horizons} weeks")
    
    total_inserted = 0
    
    for horizon in horizons:
        logger.info(f"\nProcessing {horizon}-week forward returns...")
        
        # Calculate forward returns using window functions for efficiency
        query = f"""
        INSERT INTO ml_forward_returns (
            symbol, 
            ranking_date, 
            horizon_weeks, 
            forward_return,
            forward_excess_return,
            forward_volatility,
            forward_max_drawdown
        )
        WITH return_calc AS (
            SELECT 
                r.symbol,
                r.ranking_date,
                {horizon} as horizon_weeks,
                -- Calculate forward return
                (lead_price.close / current_price.close - 1) * 100 as forward_return,
                -- Calculate excess return vs SP500
                ((lead_price.close / current_price.close) - 
                 (sp_lead.close / sp_current.close)) * 100 as forward_excess_return,
                -- Calculate volatility during period
                (
                    SELECT STDDEV(daily_return) * SQRT(252/{horizon}) 
                    FROM (
                        SELECT (close / LAG(close) OVER (ORDER BY date) - 1) as daily_return
                        FROM stock_prices sp
                        WHERE sp.symbol = r.symbol
                          AND sp.date > r.ranking_date
                          AND sp.date <= r.ranking_date + INTERVAL '{horizon} weeks'
                    ) vol
                ) as forward_volatility,
                -- Calculate max drawdown during period
                (
                    SELECT MIN((trough.close - peak.close) / peak.close) * 100
                    FROM stock_prices peak
                    JOIN stock_prices trough 
                        ON peak.symbol = trough.symbol 
                        AND trough.date > peak.date
                    WHERE peak.symbol = r.symbol
                      AND peak.date >= r.ranking_date
                      AND trough.date <= r.ranking_date + INTERVAL '{horizon} weeks'
                ) as forward_max_drawdown
                
            FROM stock_quality_rankings r
            
            -- Join current price
            JOIN stock_prices current_price 
                ON r.symbol = current_price.symbol 
                AND current_price.date = r.ranking_date
            
            -- Join future price
            LEFT JOIN stock_prices lead_price 
                ON r.symbol = lead_price.symbol 
                AND lead_price.date >= r.ranking_date + INTERVAL '{horizon} weeks' - INTERVAL '2 days'
                AND lead_price.date <= r.ranking_date + INTERVAL '{horizon} weeks' + INTERVAL '2 days'
            
            -- Join SP500 current
            LEFT JOIN sp500_price_history sp_current 
                ON sp_current.date = r.ranking_date
            
            -- Join SP500 future  
            LEFT JOIN sp500_price_history sp_lead 
                ON sp_lead.date >= r.ranking_date + INTERVAL '{horizon} weeks' - INTERVAL '2 days'
                AND sp_lead.date <= r.ranking_date + INTERVAL '{horizon} weeks' + INTERVAL '2 days'
            
            WHERE lead_price.close IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM ml_forward_returns fr 
                  WHERE fr.symbol = r.symbol 
                    AND fr.ranking_date = r.ranking_date 
                    AND fr.horizon_weeks = {horizon}
              )
            
            -- Process in batches
            LIMIT {batch_size}
        )
        SELECT * FROM return_calc
        ON CONFLICT (symbol, ranking_date, horizon_weeks) 
        DO UPDATE SET
            forward_return = EXCLUDED.forward_return,
            forward_excess_return = EXCLUDED.forward_excess_return,
            forward_volatility = EXCLUDED.forward_volatility,
            forward_max_drawdown = EXCLUDED.forward_max_drawdown
        """
        
        try:
            with engine.connect() as conn:
                result = conn.execute(text(query))
                rows_inserted = result.rowcount
                conn.commit()
                
                logger.info(f"  Inserted {rows_inserted} records for {horizon}-week horizon")
                total_inserted += rows_inserted
                
        except Exception as e:
            logger.error(f"Error calculating {horizon}-week returns: {e}")
            continue
    
    # Update additional risk metrics
    update_risk_metrics()
    
    return total_inserted


def update_risk_metrics():
    """Update stop-loss and take-profit hit indicators"""
    logger.info("Updating risk metrics...")
    
    query = """
    UPDATE ml_forward_returns fr
    SET 
        hit_stop_loss = CASE 
            WHEN fr.forward_max_drawdown < -10 THEN TRUE 
            ELSE FALSE 
        END,
        hit_take_profit = CASE 
            WHEN fr.forward_return > 20 THEN TRUE 
            ELSE FALSE 
        END,
        forward_sharpe = CASE 
            WHEN fr.forward_volatility > 0 THEN 
                (fr.forward_return - 2) / fr.forward_volatility  -- Assuming 2% risk-free rate
            ELSE NULL 
        END
    WHERE fr.forward_sharpe IS NULL
       OR fr.hit_stop_loss IS NULL
       OR fr.hit_take_profit IS NULL
    """
    
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query))
            conn.commit()
            logger.info(f"  Updated risk metrics for {result.rowcount} records")
    except Exception as e:
        logger.error(f"Error updating risk metrics: {e}")


def create_ranking_transitions():
    """Create ranking transition table for sequence models"""
    logger.info("Creating ranking transitions...")
    
    query = """
    INSERT INTO ranking_transitions (
        symbol, from_date, to_date, weeks_between,
        sp500_rank_from, sp500_rank_to, sp500_rank_change,
        fcf_rank_from, fcf_rank_to, fcf_rank_change,
        growth_rank_from, growth_rank_to, growth_rank_change,
        composite_score_from, composite_score_to, composite_score_change,
        quality_tier_from, quality_tier_to,
        tier_improved, tier_degraded,
        period_return, period_excess_return
    )
    SELECT 
        r1.symbol,
        r1.ranking_date as from_date,
        r2.ranking_date as to_date,
        EXTRACT(EPOCH FROM (r2.ranking_date - r1.ranking_date)) / 604800 as weeks_between,
        
        r1.beat_sp500_ranking as sp500_rank_from,
        r2.beat_sp500_ranking as sp500_rank_to,
        r2.beat_sp500_ranking - r1.beat_sp500_ranking as sp500_rank_change,
        
        r1.excess_cash_flow_ranking as fcf_rank_from,
        r2.excess_cash_flow_ranking as fcf_rank_to,
        r2.excess_cash_flow_ranking - r1.excess_cash_flow_ranking as fcf_rank_change,
        
        r1.growth_ranking as growth_rank_from,
        r2.growth_ranking as growth_rank_to,
        r2.growth_ranking - r1.growth_ranking as growth_rank_change,
        
        r1.final_composite_score as composite_score_from,
        r2.final_composite_score as composite_score_to,
        r2.final_composite_score - r1.final_composite_score as composite_score_change,
        
        r1.quality_tier as quality_tier_from,
        r2.quality_tier as quality_tier_to,
        
        CASE WHEN r2.quality_tier < r1.quality_tier THEN TRUE ELSE FALSE END as tier_improved,
        CASE WHEN r2.quality_tier > r1.quality_tier THEN TRUE ELSE FALSE END as tier_degraded,
        
        fr.forward_return as period_return,
        fr.forward_excess_return as period_excess_return
        
    FROM stock_quality_rankings r1
    JOIN stock_quality_rankings r2 
        ON r1.symbol = r2.symbol 
        AND r2.ranking_date > r1.ranking_date
        AND r2.ranking_date <= r1.ranking_date + INTERVAL '4 weeks'
    LEFT JOIN ml_forward_returns fr
        ON r1.symbol = fr.symbol
        AND r1.ranking_date = fr.ranking_date
        AND fr.horizon_weeks = EXTRACT(EPOCH FROM (r2.ranking_date - r1.ranking_date)) / 604800
    
    WHERE NOT EXISTS (
        SELECT 1 FROM ranking_transitions rt
        WHERE rt.symbol = r1.symbol
          AND rt.from_date = r1.ranking_date
          AND rt.to_date = r2.ranking_date
    )
    ON CONFLICT (symbol, from_date, to_date) DO NOTHING
    """
    
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query))
            conn.commit()
            logger.info(f"  Created {result.rowcount} ranking transitions")
    except Exception as e:
        logger.error(f"Error creating ranking transitions: {e}")


def generate_summary_statistics():
    """Generate summary statistics for forward returns"""
    logger.info("\nGenerating summary statistics...")
    
    query = """
    SELECT 
        horizon_weeks,
        COUNT(*) as total_records,
        AVG(forward_return) as avg_return,
        STDDEV(forward_return) as std_return,
        AVG(forward_excess_return) as avg_excess,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY forward_return) as q25_return,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY forward_return) as median_return,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY forward_return) as q75_return,
        AVG(CASE WHEN hit_stop_loss THEN 1.0 ELSE 0.0 END) * 100 as pct_hit_stop_loss,
        AVG(CASE WHEN hit_take_profit THEN 1.0 ELSE 0.0 END) * 100 as pct_hit_profit
    FROM ml_forward_returns
    GROUP BY horizon_weeks
    ORDER BY horizon_weeks
    """
    
    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn)
        
    print("\n" + "="*80)
    print("FORWARD RETURNS SUMMARY STATISTICS")
    print("="*80)
    
    if not df.empty:
        for _, row in df.iterrows():
            print(f"\n{int(row['horizon_weeks'])}-Week Horizon:")
            print(f"  Records: {int(row['total_records']):,}")
            print(f"  Avg Return: {row['avg_return']:.2f}%")
            print(f"  Std Dev: {row['std_return']:.2f}%")
            print(f"  Avg Excess: {row['avg_excess']:.2f}%")
            print(f"  Median: {row['median_return']:.2f}%")
            print(f"  Q25/Q75: {row['q25_return']:.2f}% / {row['q75_return']:.2f}%")
            print(f"  Hit Stop Loss (-10%): {row['pct_hit_stop_loss']:.1f}%")
            print(f"  Hit Take Profit (+20%): {row['pct_hit_profit']:.1f}%")
    else:
        print("No forward returns data available")
    
    print("="*80)


def main():
    """Main execution"""
    start_time = datetime.now()
    
    print("\n" + "="*60)
    print("FORWARD RETURNS CALCULATOR")
    print("="*60)
    print(f"Start time: {start_time}")
    
    try:
        # Calculate forward returns
        total = calculate_forward_returns(
            horizons=[1, 4, 12, 26, 52],
            batch_size=5000
        )
        
        # Create ranking transitions
        create_ranking_transitions()
        
        # Generate summary
        generate_summary_statistics()
        
        duration = (datetime.now() - start_time).total_seconds()
        print(f"\n[SUCCESS] Calculated {total} forward returns in {duration:.1f} seconds")
        
        return 0
        
    except Exception as e:
        logger.error(f"Forward returns calculation failed: {e}")
        print(f"\n[ERROR] Script failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())