#!/usr/bin/env python3
"""
Point-in-Time Quality Rankings Calculator
Ensures no look-ahead bias in historical calculations by using only data available as of ranking date
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import logging
from calculate_quality_rankings import WorldClassQualityRankingCalculator

logger = logging.getLogger(__name__)


class PointInTimeRankingCalculator(WorldClassQualityRankingCalculator):
    """
    Modified ranking calculator that ensures point-in-time accuracy
    All queries are modified to use only data available as of ranking_date
    """
    
    def __init__(self, ranking_date=None):
        super().__init__(ranking_date)
        self.as_of_date = ranking_date or datetime.now()
        logger.info(f"Point-in-time calculator initialized for {self.as_of_date}")
    
    def calculate_sp500_outperformance(self):
        """Calculate SP500 outperformance using only historical data up to ranking date"""
        
        query = f"""
        WITH yearly_returns AS (
            SELECT 
                s.symbol,
                s.year,
                s.symbol_return,
                s.sp500_return,
                s.excess_return,
                s.beat_sp500,
                -- Only use data up to the ranking year
                CASE 
                    WHEN s.year = EXTRACT(YEAR FROM '{self.as_of_date}'::date) THEN
                        -- For current year, calculate YTD returns
                        (
                            SELECT (sp2.close / sp1.close - 1) * 100
                            FROM stock_prices sp1
                            JOIN stock_prices sp2 ON sp1.symbol = sp2.symbol
                            WHERE sp1.symbol = s.symbol
                              AND sp1.date = DATE_TRUNC('year', '{self.as_of_date}'::date)
                              AND sp2.date = '{self.as_of_date}'::date
                        )
                    ELSE s.symbol_return
                END as adjusted_return
            FROM sp500_outperformance_detail s
            WHERE s.year <= EXTRACT(YEAR FROM '{self.as_of_date}'::date)
        ),
        recent_performance AS (
            SELECT 
                symbol,
                COUNT(*) FILTER (WHERE beat_sp500 = true AND year >= EXTRACT(YEAR FROM '{self.as_of_date}'::date) - 10) as beat_count_10yr,
                COUNT(*) FILTER (WHERE year >= EXTRACT(YEAR FROM '{self.as_of_date}'::date) - 10) as total_years_10yr,
                COUNT(*) FILTER (WHERE beat_sp500 = true AND year >= EXTRACT(YEAR FROM '{self.as_of_date}'::date) - 5) as beat_count_5yr,
                COUNT(*) FILTER (WHERE year >= EXTRACT(YEAR FROM '{self.as_of_date}'::date) - 5) as total_years_5yr,
                AVG(excess_return) FILTER (WHERE year >= EXTRACT(YEAR FROM '{self.as_of_date}'::date) - 10) as avg_excess_10yr,
                AVG(excess_return) FILTER (WHERE year >= EXTRACT(YEAR FROM '{self.as_of_date}'::date) - 5) as avg_excess_5yr,
                MAX(excess_return) FILTER (WHERE year = EXTRACT(YEAR FROM '{self.as_of_date}'::date)) as current_year_excess
            FROM yearly_returns
            GROUP BY symbol
        )
        SELECT 
            symbol,
            COALESCE(beat_count_10yr, 0) as years_beating_sp500,
            COALESCE(total_years_10yr, 0) as years_of_data,
            CASE 
                WHEN total_years_10yr > 0 THEN beat_count_10yr * 100.0 / total_years_10yr 
                ELSE 0 
            END as consistency_score,
            COALESCE(avg_excess_10yr, 0) as avg_annual_excess,
            COALESCE(beat_count_5yr, 0) as recent_5yr_beat_count,
            COALESCE(current_year_excess, 0) as recent_1yr_excess,
            -- Weighted score calculation
            (
                COALESCE(beat_count_10yr * 10, 0) + 
                COALESCE(avg_excess_10yr * 2, 0) + 
                COALESCE(beat_count_5yr * 15, 0) + 
                COALESCE(current_year_excess, 0)
            ) as sp500_weighted_score
        FROM recent_performance
        WHERE total_years_10yr >= 2  -- Minimum data requirement
        """
        
        return pd.read_sql_query(query, self.engine)
    
    def calculate_fundamentals_trend(self):
        """Calculate fundamental trends using only data available as of ranking date"""
        
        query = f"""
        WITH price_momentum AS (
            SELECT 
                sp.symbol,
                -- Price trends calculated up to ranking date
                (LAST_VALUE(sp.close) OVER (PARTITION BY sp.symbol ORDER BY sp.date 
                    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) / 
                 FIRST_VALUE(sp.close) OVER (PARTITION BY sp.symbol ORDER BY sp.date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) - 1) * 100 as price_trend_10yr,
                     
                (SELECT (sp2.close / sp1.close - 1) * 100
                 FROM stock_prices sp1, stock_prices sp2
                 WHERE sp1.symbol = sp.symbol AND sp2.symbol = sp.symbol
                   AND sp1.date = '{self.as_of_date}'::date - INTERVAL '5 years'
                   AND sp2.date = '{self.as_of_date}'::date) as price_trend_5yr,
                   
                (SELECT (sp2.close / sp1.close - 1) * 100
                 FROM stock_prices sp1, stock_prices sp2
                 WHERE sp1.symbol = sp.symbol AND sp2.symbol = sp.symbol
                   AND sp1.date = '{self.as_of_date}'::date - INTERVAL '1 year'
                   AND sp2.date = '{self.as_of_date}'::date) as price_trend_1yr
                   
            FROM stock_prices sp
            WHERE sp.date <= '{self.as_of_date}'::date
            GROUP BY sp.symbol, sp.date, sp.close
        ),
        fundamental_trends AS (
            SELECT 
                f.symbol,
                -- Only use fundamentals reported before ranking date
                AVG(f.net_margin) FILTER (
                    WHERE f.fiscal_date_ending <= '{self.as_of_date}'::date 
                    AND f.fiscal_date_ending > '{self.as_of_date}'::date - INTERVAL '1 year'
                ) as current_net_margin,
                
                AVG(f.gross_margin) FILTER (
                    WHERE f.fiscal_date_ending <= '{self.as_of_date}'::date 
                    AND f.fiscal_date_ending > '{self.as_of_date}'::date - INTERVAL '1 year'
                ) as current_gross_margin,
                
                AVG(f.operating_margin) FILTER (
                    WHERE f.fiscal_date_ending <= '{self.as_of_date}'::date 
                    AND f.fiscal_date_ending > '{self.as_of_date}'::date - INTERVAL '1 year'
                ) as current_operating_margin,
                
                -- Historical comparisons
                AVG(f.net_margin) FILTER (
                    WHERE f.fiscal_date_ending <= '{self.as_of_date}'::date - INTERVAL '5 years'
                    AND f.fiscal_date_ending > '{self.as_of_date}'::date - INTERVAL '6 years'
                ) as net_margin_5yr_ago,
                
                -- Revenue growth
                MAX(f.total_revenue) FILTER (
                    WHERE f.fiscal_date_ending <= '{self.as_of_date}'::date 
                    AND f.fiscal_date_ending > '{self.as_of_date}'::date - INTERVAL '1 year'
                ) as current_revenue,
                
                MAX(f.total_revenue) FILTER (
                    WHERE f.fiscal_date_ending <= '{self.as_of_date}'::date - INTERVAL '10 years'
                    AND f.fiscal_date_ending > '{self.as_of_date}'::date - INTERVAL '11 years'
                ) as revenue_10yr_ago,
                
                -- ROE and ROA
                AVG(f.return_on_equity) FILTER (
                    WHERE f.fiscal_date_ending <= '{self.as_of_date}'::date 
                    AND f.fiscal_date_ending > '{self.as_of_date}'::date - INTERVAL '1 year'
                ) as roe_current,
                
                AVG(f.return_on_assets) FILTER (
                    WHERE f.fiscal_date_ending <= '{self.as_of_date}'::date 
                    AND f.fiscal_date_ending > '{self.as_of_date}'::date - INTERVAL '1 year'
                ) as roa_current
                
            FROM fundamentals f
            WHERE f.fiscal_date_ending <= '{self.as_of_date}'::date
              AND f.created_at <= '{self.as_of_date}'::timestamp  -- Prevent using future corrections
            GROUP BY f.symbol
        )
        SELECT 
            COALESCE(pm.symbol, ft.symbol) as symbol,
            pm.price_trend_10yr,
            pm.price_trend_5yr,
            pm.price_trend_1yr,
            ft.current_net_margin,
            ft.current_gross_margin,
            ft.current_operating_margin,
            ft.current_net_margin - ft.net_margin_5yr_ago as margin_change_5yr,
            CASE 
                WHEN ft.revenue_10yr_ago > 0 THEN 
                    (POWER(ft.current_revenue / ft.revenue_10yr_ago, 1.0/10) - 1) * 100
                ELSE 0
            END as revenue_growth_10yr,
            ft.roe_current,
            ft.roa_current
        FROM price_momentum pm
        FULL OUTER JOIN fundamental_trends ft ON pm.symbol = ft.symbol
        WHERE pm.symbol IS NOT NULL OR ft.symbol IS NOT NULL
        """
        
        return pd.read_sql_query(query, self.engine)
    
    def calculate_news_sentiment_score(self):
        """Calculate sentiment using only news available as of ranking date"""
        
        # For historical dates, use a 30-day lookback window
        sentiment_window_start = self.as_of_date - timedelta(days=30)
        
        query = f"""
        WITH sentiment_data AS (
            SELECT 
                ds.symbol,
                AVG(ds.weighted_sentiment_score) FILTER (
                    WHERE ds.date >= '{sentiment_window_start}'::date
                    AND ds.date <= '{self.as_of_date}'::date
                ) as sentiment_30d,
                
                AVG(ds.weighted_sentiment_score) FILTER (
                    WHERE ds.date >= '{self.as_of_date}'::date - INTERVAL '7 days'
                    AND ds.date <= '{self.as_of_date}'::date
                ) as sentiment_7d,
                
                SUM(ds.total_articles) FILTER (
                    WHERE ds.date >= '{sentiment_window_start}'::date
                    AND ds.date <= '{self.as_of_date}'::date
                ) as article_count,
                
                AVG(ds.bull_bear_ratio) FILTER (
                    WHERE ds.date >= '{sentiment_window_start}'::date
                    AND ds.date <= '{self.as_of_date}'::date
                ) as bull_bear_ratio,
                
                -- Momentum: compare recent vs older sentiment
                AVG(ds.weighted_sentiment_score) FILTER (
                    WHERE ds.date >= '{self.as_of_date}'::date - INTERVAL '7 days'
                    AND ds.date <= '{self.as_of_date}'::date
                ) - 
                AVG(ds.weighted_sentiment_score) FILTER (
                    WHERE ds.date >= '{self.as_of_date}'::date - INTERVAL '30 days'
                    AND ds.date <= '{self.as_of_date}'::date - INTERVAL '7 days'
                ) as sentiment_momentum
                
            FROM daily_sentiment_summary ds
            WHERE ds.date <= '{self.as_of_date}'::date
            GROUP BY ds.symbol
        )
        SELECT 
            symbol,
            COALESCE(sentiment_30d, 0) as sentiment_30d,
            COALESCE(sentiment_7d, 0) as sentiment_7d,
            COALESCE(sentiment_momentum, 0) as sentiment_momentum,
            COALESCE(article_count, 0) as article_count,
            COALESCE(bull_bear_ratio, 1) as bull_bear_ratio,
            -- Composite sentiment score
            (
                COALESCE(sentiment_7d * 0.6, 0) +
                COALESCE(sentiment_30d * 0.4, 0) +
                COALESCE(sentiment_momentum * 10, 0) +
                CASE 
                    WHEN bull_bear_ratio > 2 THEN 10
                    WHEN bull_bear_ratio > 1.5 THEN 5
                    WHEN bull_bear_ratio < 0.5 THEN -10
                    WHEN bull_bear_ratio < 0.67 THEN -5
                    ELSE 0
                END
            ) as sentiment_score
        FROM sentiment_data
        WHERE article_count > 0
        """
        
        return pd.read_sql_query(query, self.engine)
    
    def calculate_breakout_ranking(self):
        """Calculate breakout rankings using price/volume data up to ranking date"""
        
        # Use 90-day lookback for breakout detection
        lookback_start = self.as_of_date - timedelta(days=90)
        
        query = f"""
        WITH price_data AS (
            SELECT 
                symbol,
                date,
                close,
                volume,
                high,
                low,
                -- Moving averages calculated up to ranking date
                AVG(close) OVER (PARTITION BY symbol ORDER BY date 
                    ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as sma_20,
                AVG(close) OVER (PARTITION BY symbol ORDER BY date 
                    ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) as sma_50,
                -- 52-week high/low up to ranking date
                MAX(high) OVER (PARTITION BY symbol ORDER BY date 
                    ROWS BETWEEN 251 PRECEDING AND CURRENT ROW) as high_52w,
                MIN(low) OVER (PARTITION BY symbol ORDER BY date 
                    ROWS BETWEEN 251 PRECEDING AND CURRENT ROW) as low_52w
            FROM stock_prices
            WHERE date <= '{self.as_of_date}'::date
              AND date >= '{lookback_start}'::date
        ),
        breakout_metrics AS (
            SELECT 
                pd.symbol,
                -- Price changes
                (LAST_VALUE(pd.close) OVER w / FIRST_VALUE(pd.close) OVER w - 1) * 100 as price_change_3m,
                -- Volume analysis
                AVG(pd.volume) OVER w as avg_volume_3m,
                -- Current position
                LAST_VALUE(pd.close) OVER w as current_price,
                LAST_VALUE(pd.high_52w) OVER w as high_52w,
                LAST_VALUE(pd.low_52w) OVER w as low_52w,
                -- Technical position
                CASE 
                    WHEN LAST_VALUE(pd.close) OVER w >= LAST_VALUE(pd.high_52w) OVER w * 0.95 THEN true
                    ELSE false
                END as near_52w_high
            FROM price_data pd
            WHERE pd.date = '{self.as_of_date}'::date
            WINDOW w AS (PARTITION BY pd.symbol ORDER BY pd.date 
                         ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)
        )
        SELECT 
            symbol,
            price_change_3m,
            avg_volume_3m,
            current_price,
            (current_price - low_52w) / NULLIF(high_52w - low_52w, 0) * 100 as price_vs_52w_range,
            near_52w_high,
            -- Breakout score
            GREATEST(0, LEAST(100,
                price_change_3m * 0.4 +
                CASE WHEN near_52w_high THEN 30 ELSE 0 END +
                ((current_price - low_52w) / NULLIF(high_52w - low_52w, 0) * 30)
            )) as breakout_score
        FROM breakout_metrics
        """
        
        df = pd.read_sql_query(query, self.engine)
        if not df.empty:
            df['breakout_ranking'] = df['breakout_score'].rank(ascending=False, method='min').astype(int)
        return df
    
    def calculate_growth_ranking(self):
        """Calculate long-term growth using only historical data up to ranking date"""
        
        query = f"""
        WITH lifetime_performance AS (
            SELECT 
                sp.symbol,
                MIN(sp.date) FILTER (WHERE sp.date <= '{self.as_of_date}'::date) as first_date,
                '{self.as_of_date}'::date as last_date,
                COUNT(DISTINCT EXTRACT(YEAR FROM sp.date)) FILTER (
                    WHERE sp.date <= '{self.as_of_date}'::date
                ) as years_tracked,
                FIRST_VALUE(sp.close) OVER (
                    PARTITION BY sp.symbol 
                    ORDER BY sp.date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                ) as first_price,
                LAST_VALUE(sp.close) OVER (
                    PARTITION BY sp.symbol 
                    ORDER BY sp.date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                ) FILTER (WHERE sp.date <= '{self.as_of_date}'::date) as last_price
            FROM stock_prices sp
            WHERE sp.date <= '{self.as_of_date}'::date
            GROUP BY sp.symbol, sp.close, sp.date
        ),
        growth_metrics AS (
            SELECT 
                lp.symbol,
                lp.years_tracked,
                CASE 
                    WHEN lp.first_price > 0 AND lp.last_price > 0 AND lp.years_tracked > 1
                    THEN (POWER(lp.last_price / lp.first_price, 1.0 / NULLIF(lp.years_tracked, 0)) - 1) * 100
                    ELSE 0
                END as lifetime_cagr,
                -- Get SP500 outperformance data
                COUNT(*) FILTER (
                    WHERE spo.beat_sp500 = true 
                    AND spo.year <= EXTRACT(YEAR FROM '{self.as_of_date}'::date)
                ) as years_beating_sp500,
                COUNT(*) FILTER (
                    WHERE spo.year <= EXTRACT(YEAR FROM '{self.as_of_date}'::date)
                ) as total_years
            FROM lifetime_performance lp
            LEFT JOIN sp500_outperformance_detail spo ON lp.symbol = spo.symbol
            GROUP BY lp.symbol, lp.years_tracked, lp.first_price, lp.last_price
        )
        SELECT 
            symbol,
            years_tracked as total_years_tracked,
            lifetime_cagr as growth_score,
            years_beating_sp500,
            CASE 
                WHEN total_years > 0 THEN years_beating_sp500 * 100.0 / total_years
                ELSE 0
            END as outperformance_consistency
        FROM growth_metrics
        WHERE years_tracked >= 2
        """
        
        df = pd.read_sql_query(query, self.engine)
        if not df.empty:
            df['growth_ranking'] = df['growth_score'].rank(ascending=False, method='min').astype(int)
        return df


def calculate_historical_rankings_batch(dates_list, save_to_db=True):
    """
    Calculate rankings for a batch of historical dates
    
    Args:
        dates_list: List of dates to calculate rankings for
        save_to_db: Whether to save results to database
    
    Returns:
        Dictionary of DataFrames with rankings for each date
    """
    results = {}
    
    for ranking_date in dates_list:
        try:
            logger.info(f"Calculating point-in-time rankings for {ranking_date}")
            
            # Initialize point-in-time calculator
            calc = PointInTimeRankingCalculator(ranking_date=ranking_date)
            
            # Calculate rankings
            rankings_df = calc.calculate_composite_rankings()
            
            # Store results
            results[ranking_date] = rankings_df
            
            # Save to database if requested
            if save_to_db and not rankings_df.empty:
                calc.save_rankings(rankings_df)
                logger.info(f"Saved {len(rankings_df)} rankings for {ranking_date}")
            
        except Exception as e:
            logger.error(f"Error processing {ranking_date}: {e}")
            results[ranking_date] = pd.DataFrame()
    
    return results


if __name__ == "__main__":
    # Example usage
    from datetime import datetime, timedelta
    
    # Calculate rankings for the last 4 Fridays
    end_date = datetime.now()
    dates = []
    
    # Find most recent Friday
    days_until_friday = (4 - end_date.weekday()) % 7
    if days_until_friday == 0 and end_date.hour < 16:  # If today is Friday but market not closed
        days_until_friday = 7
    last_friday = end_date - timedelta(days=days_until_friday)
    
    # Get last 4 Fridays
    for i in range(4):
        dates.append(last_friday - timedelta(weeks=i))
    
    print(f"Calculating rankings for dates: {[d.strftime('%Y-%m-%d') for d in dates]}")
    
    # Calculate rankings
    results = calculate_historical_rankings_batch(dates, save_to_db=False)
    
    # Display results
    for date, df in results.items():
        if not df.empty:
            print(f"\nTop 10 stocks for {date.strftime('%Y-%m-%d')}:")
            print(df.nlargest(10, 'final_composite_score')[['symbol', 'final_composite_score']])