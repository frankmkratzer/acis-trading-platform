#!/usr/bin/env python3
"""
Excess Cash Flow Calculator
Core metric for ACIS investment strategy

Formula: Excess Cash Flow = (Cash Flow per Share - Dividends per Share - CapEx per Share)
Quality Metric: Excess Cash Flow / Cash Flow per Share × 100%

This module calculates the most important metric in our investment philosophy:
how much cash a company generates after all obligations.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import text
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.logging_config import setup_logger
from database.db_connection_manager import DatabaseConnectionManager

logger = setup_logger("excess_cash_flow")
db_manager = DatabaseConnectionManager()
engine = db_manager.get_engine()


class ExcessCashFlowAnalyzer:
    """Calculate and analyze Excess Cash Flow metrics"""
    
    def __init__(self):
        self.engine = engine
        
    def fetch_cash_flow_data(self, symbols=None, years_back=5):
        """
        Fetch cash flow, dividend, and capex data for analysis
        
        Args:
            symbols: List of symbols to analyze (None = all)
            years_back: Number of years of data to fetch
        """
        
        query = text("""
            WITH latest_fundamentals AS (
                SELECT 
                    f.symbol,
                    f.fiscal_date_ending,
                    f.period_type,
                    f.operating_cash_flow,
                    f.free_cash_flow,
                    f.shares_outstanding,
                    -- Capital Expenditures = Operating Cash Flow - Free Cash Flow
                    (f.operating_cash_flow - f.free_cash_flow) as capital_expenditures
                FROM fundamentals f
                WHERE f.fiscal_date_ending >= CURRENT_DATE - INTERVAL ':years years'
                    AND f.period_type = 'annual'
                    AND f.operating_cash_flow IS NOT NULL
                    AND f.free_cash_flow IS NOT NULL
                    AND f.shares_outstanding IS NOT NULL
                    AND f.shares_outstanding > 0
                    AND (:symbols IS NULL OR f.symbol = ANY(:symbols))
            ),
            dividend_data AS (
                SELECT 
                    dh.symbol,
                    EXTRACT(YEAR FROM dh.ex_date) as dividend_year,
                    SUM(dh.dividend) as total_dividends_per_share
                FROM dividend_history dh
                WHERE dh.ex_date >= CURRENT_DATE - INTERVAL ':years years'
                GROUP BY dh.symbol, EXTRACT(YEAR FROM dh.ex_date)
            ),
            combined_data AS (
                SELECT 
                    lf.symbol,
                    lf.fiscal_date_ending,
                    lf.operating_cash_flow,
                    lf.free_cash_flow,
                    lf.capital_expenditures,
                    lf.shares_outstanding,
                    -- Per share calculations
                    lf.operating_cash_flow::NUMERIC / lf.shares_outstanding as cash_flow_per_share,
                    lf.capital_expenditures::NUMERIC / lf.shares_outstanding as capex_per_share,
                    COALESCE(dd.total_dividends_per_share, 0) as dividends_per_share
                FROM latest_fundamentals lf
                LEFT JOIN dividend_data dd 
                    ON lf.symbol = dd.symbol 
                    AND EXTRACT(YEAR FROM lf.fiscal_date_ending) = dd.dividend_year
            )
            SELECT 
                symbol,
                fiscal_date_ending,
                cash_flow_per_share,
                capex_per_share,
                dividends_per_share,
                -- Calculate Excess Cash Flow
                (cash_flow_per_share - dividends_per_share - capex_per_share) as excess_cash_flow_per_share,
                -- Calculate quality percentage
                CASE 
                    WHEN cash_flow_per_share > 0 THEN
                        ((cash_flow_per_share - dividends_per_share - capex_per_share) / cash_flow_per_share * 100)
                    ELSE NULL
                END as excess_cash_flow_percentage
            FROM combined_data
            ORDER BY symbol, fiscal_date_ending DESC
        """)
        
        params = {
            'years': years_back,
            'symbols': symbols if symbols else None
        }
        
        with self.engine.connect() as conn:
            result = conn.execute(query, params)
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            
        return df
    
    def calculate_excess_cash_flow(self, symbol):
        """
        Calculate Excess Cash Flow metrics for a single company
        
        Returns:
            dict: Contains current and historical excess cash flow metrics
        """
        
        # Fetch data for this symbol
        df = self.fetch_cash_flow_data(symbols=[symbol], years_back=10)
        
        if df.empty:
            logger.warning(f"No cash flow data available for {symbol}")
            return None
        
        # Get the most recent data
        latest = df.iloc[0]
        
        # Calculate trends (10y, 5y, 1y)
        metrics = {
            'symbol': symbol,
            'latest_date': latest['fiscal_date_ending'],
            
            # Current metrics
            'cash_flow_per_share': float(latest['cash_flow_per_share']) if pd.notna(latest['cash_flow_per_share']) else None,
            'dividends_per_share': float(latest['dividends_per_share']) if pd.notna(latest['dividends_per_share']) else None,
            'capex_per_share': float(latest['capex_per_share']) if pd.notna(latest['capex_per_share']) else None,
            'excess_cash_flow': float(latest['excess_cash_flow_per_share']) if pd.notna(latest['excess_cash_flow_per_share']) else None,
            'excess_cash_flow_pct': float(latest['excess_cash_flow_percentage']) if pd.notna(latest['excess_cash_flow_percentage']) else None,
            
            # Quality rating based on percentage
            'quality_rating': self._get_quality_rating(latest['excess_cash_flow_percentage']),
            
            # Historical analysis
            'trend_5y': self._calculate_trend(df.head(5)),
            'trend_10y': self._calculate_trend(df.head(10)),
            'avg_excess_cf_5y': df.head(5)['excess_cash_flow_percentage'].mean() if len(df) >= 5 else None,
            'avg_excess_cf_10y': df.head(10)['excess_cash_flow_percentage'].mean() if len(df) >= 10 else None,
        }
        
        return metrics
    
    def _get_quality_rating(self, excess_pct):
        """
        Classify company quality based on excess cash flow percentage
        
        80-100%: Excellent - Exceptional cash generation
        60-80%: Very Good - Strong cash generation
        40-60%: Good - Solid cash generation
        20-40%: Fair - Moderate cash generation
        0-20%: Poor - Weak cash generation
        <0%: Warning - Negative excess cash flow
        """
        if excess_pct is None:
            return "Unknown"
        elif excess_pct >= 80:
            return "Excellent"
        elif excess_pct >= 60:
            return "Very Good"
        elif excess_pct >= 40:
            return "Good"
        elif excess_pct >= 20:
            return "Fair"
        elif excess_pct >= 0:
            return "Poor"
        else:
            return "Warning"
    
    def _calculate_trend(self, df):
        """
        Calculate trend (Advancing, Stable, Decreasing) for excess cash flow
        """
        if len(df) < 3:
            return "Insufficient Data"
        
        # Get excess cash flow percentages
        values = df['excess_cash_flow_percentage'].dropna()
        
        if len(values) < 3:
            return "Insufficient Data"
        
        # Calculate linear regression slope
        x = np.arange(len(values))
        y = values.values
        
        # Simple linear regression
        slope = np.polyfit(x, y, 1)[0]
        
        # Classify based on slope
        if slope > 2:  # Growing more than 2% per year
            return "Advancing"
        elif slope < -2:  # Declining more than 2% per year
            return "Decreasing"
        else:
            return "Stable"
    
    def rank_by_excess_cash_flow(self, min_market_cap=2_000_000_000):
        """
        Rank all companies by their excess cash flow metrics
        
        Args:
            min_market_cap: Minimum market cap filter
        
        Returns:
            DataFrame with companies ranked by excess cash flow quality
        """
        
        # Get list of symbols to analyze
        query = text("""
            SELECT symbol 
            FROM symbol_universe 
            WHERE market_cap >= :min_cap
                AND is_etf = FALSE
                AND security_type = 'Common Stock'
        """)
        
        with self.engine.connect() as conn:
            result = conn.execute(query, {'min_cap': min_market_cap})
            symbols = [row[0] for row in result.fetchall()]
        
        logger.info(f"Analyzing excess cash flow for {len(symbols)} companies...")
        
        # Calculate metrics for all symbols
        all_metrics = []
        for symbol in symbols:
            metrics = self.calculate_excess_cash_flow(symbol)
            if metrics and metrics['excess_cash_flow_pct'] is not None:
                all_metrics.append(metrics)
        
        # Convert to DataFrame and rank
        df = pd.DataFrame(all_metrics)
        
        if not df.empty:
            # Sort by excess cash flow percentage (higher is better)
            df = df.sort_values('excess_cash_flow_pct', ascending=False)
            df['rank'] = range(1, len(df) + 1)
            
            # Add percentile
            df['percentile'] = 100 - (df['rank'] - 1) / len(df) * 100
        
        return df
    
    def save_to_database(self, df):
        """Save excess cash flow metrics to database"""
        
        if df.empty:
            logger.warning("No data to save")
            return
        
        # Create table if it doesn't exist
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS excess_cash_flow_metrics (
            symbol VARCHAR(10) PRIMARY KEY,
            latest_date DATE,
            cash_flow_per_share NUMERIC(12, 4),
            dividends_per_share NUMERIC(12, 4),
            capex_per_share NUMERIC(12, 4),
            excess_cash_flow NUMERIC(12, 4),
            excess_cash_flow_pct NUMERIC(6, 2),
            quality_rating VARCHAR(20),
            trend_5y VARCHAR(20),
            trend_10y VARCHAR(20),
            avg_excess_cf_5y NUMERIC(6, 2),
            avg_excess_cf_10y NUMERIC(6, 2),
            rank INTEGER,
            percentile NUMERIC(5, 2),
            calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (symbol) REFERENCES symbol_universe(symbol) ON DELETE CASCADE
        )
        """
        
        with self.engine.begin() as conn:
            conn.execute(text(create_table_sql))
            
            # Save data
            df['calculated_at'] = datetime.now()
            df.to_sql('excess_cash_flow_metrics', conn, if_exists='replace', index=False)
            
        logger.info(f"Saved {len(df)} excess cash flow metrics to database")


def main():
    """Main execution"""
    
    analyzer = ExcessCashFlowAnalyzer()
    
    # Example: Analyze a specific stock
    print("\n" + "="*60)
    print("EXCESS CASH FLOW ANALYSIS")
    print("="*60)
    
    # Analyze Apple as an example
    symbol = 'AAPL'
    metrics = analyzer.calculate_excess_cash_flow(symbol)
    
    if metrics:
        print(f"\n{symbol} Excess Cash Flow Analysis:")
        print(f"  Latest Date: {metrics['latest_date']}")
        print(f"  Cash Flow/Share: ${metrics['cash_flow_per_share']:.2f}")
        print(f"  Dividends/Share: ${metrics['dividends_per_share']:.2f}")
        print(f"  CapEx/Share: ${metrics['capex_per_share']:.2f}")
        print(f"  Excess Cash Flow: ${metrics['excess_cash_flow']:.2f}")
        print(f"  Excess CF %: {metrics['excess_cash_flow_pct']:.1f}%")
        print(f"  Quality Rating: {metrics['quality_rating']}")
        print(f"  5-Year Trend: {metrics['trend_5y']}")
        print(f"  10-Year Trend: {metrics['trend_10y']}")
    
    # Rank all companies
    print("\n" + "="*60)
    print("RANKING ALL COMPANIES BY EXCESS CASH FLOW")
    print("="*60)
    
    df_ranked = analyzer.rank_by_excess_cash_flow()
    
    if not df_ranked.empty:
        # Save to database
        analyzer.save_to_database(df_ranked)
        
        # Show top 10
        print("\nTop 10 Companies by Excess Cash Flow:")
        for _, row in df_ranked.head(10).iterrows():
            print(f"  #{row['rank']} {row['symbol']}: {row['excess_cash_flow_pct']:.1f}% ({row['quality_rating']}) - {row['trend_5y']}")
        
        # Show bottom 10
        print("\nBottom 10 Companies by Excess Cash Flow:")
        for _, row in df_ranked.tail(10).iterrows():
            print(f"  #{row['rank']} {row['symbol']}: {row['excess_cash_flow_pct']:.1f}% ({row['quality_rating']}) - {row['trend_5y']}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())