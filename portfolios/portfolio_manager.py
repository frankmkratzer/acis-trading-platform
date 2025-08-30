#!/usr/bin/env python3
"""
Portfolio Manager - Three-Portfolio Strategy Implementation
Manages Value, Growth, and Dividend portfolios with automated selection and rebalancing

This module implements the ACIS three-portfolio strategy:
1. VALUE: Top 10 undervalued quality stocks (quarterly rebalance)
2. GROWTH: Top 10 long-term outperformers (quarterly rebalance)
3. DIVIDEND: Top 10 sustainable dividend growers (annual rebalance)
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import text
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.logging_config import setup_logger
from database.db_connection_manager import DatabaseConnectionManager

logger = setup_logger("portfolio_manager")
db_manager = DatabaseConnectionManager()
engine = db_manager.get_engine()


class PortfolioManager:
    """Manages the three ACIS portfolios"""
    
    # Portfolio configuration
    PORTFOLIO_SIZE = 10
    MAX_SECTOR_WEIGHT = 0.30
    MAX_INDUSTRY_COUNT = 2
    MIN_MARKET_CAP = 2_000_000_000  # $2B minimum
    
    # Score thresholds
    MIN_VALUE_EXCESS_CF = 20  # Minimum 20% excess cash flow for value
    MIN_GROWTH_EXCESS_CF = 40  # Minimum 40% for growth
    MIN_DIVIDEND_EXCESS_CF = 30  # Minimum 30% for dividend
    MIN_DIVIDEND_SCORE = 60  # Minimum dividend quality score
    
    def __init__(self):
        self.engine = engine
        
    def calculate_value_score(self, symbol: str) -> Dict:
        """
        Calculate value score for a stock
        
        Components:
        - 40% Historical valuation percentile
        - 30% Excess cash flow yield
        - 30% Price/Intrinsic value estimate
        """
        
        query = text("""
            WITH current_metrics AS (
                SELECT 
                    sp.symbol,
                    sp.close_price as current_price,
                    cfo.pe_ratio,
                    cfo.price_to_book,
                    cfo.price_to_sales,
                    ecf.excess_cash_flow_pct,
                    ecf.excess_cash_flow,
                    su.shares_outstanding,
                    su.market_cap
                FROM stock_prices sp
                JOIN company_fundamentals_overview cfo ON sp.symbol = cfo.symbol
                JOIN excess_cash_flow_metrics ecf ON sp.symbol = ecf.symbol
                JOIN symbol_universe su ON sp.symbol = su.symbol
                WHERE sp.symbol = :symbol
                    AND sp.date = (SELECT MAX(date) FROM stock_prices WHERE symbol = :symbol)
            ),
            historical_valuations AS (
                SELECT 
                    symbol,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY pe_ratio) as median_pe,
                    PERCENTILE_CONT(0.2) WITHIN GROUP (ORDER BY pe_ratio) as p20_pe,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price_to_book) as median_pb,
                    PERCENTILE_CONT(0.2) WITHIN GROUP (ORDER BY price_to_book) as p20_pb,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price_to_sales) as median_ps,
                    PERCENTILE_CONT(0.2) WITHIN GROUP (ORDER BY price_to_sales) as p20_ps
                FROM (
                    SELECT 
                        sp.symbol,
                        sp.close_price / NULLIF(f.diluted_eps_ttm, 0) as pe_ratio,
                        su.market_cap / NULLIF(f.total_shareholder_equity, 0) as price_to_book,
                        su.market_cap / NULLIF(f.total_revenue_ttm, 0) as price_to_sales
                    FROM stock_prices sp
                    JOIN fundamentals f ON sp.symbol = f.symbol 
                        AND DATE_TRUNC('quarter', sp.date) = DATE_TRUNC('quarter', f.fiscal_date_ending)
                    JOIN symbol_universe su ON sp.symbol = su.symbol
                    WHERE sp.symbol = :symbol
                        AND sp.date >= CURRENT_DATE - INTERVAL '10 years'
                ) hist
                GROUP BY symbol
            )
            SELECT 
                cm.*,
                hv.*,
                -- Valuation percentile (lower is better for value)
                CASE 
                    WHEN cm.pe_ratio < hv.p20_pe THEN 100
                    WHEN cm.pe_ratio < hv.median_pe THEN 60 + (hv.median_pe - cm.pe_ratio) / NULLIF(hv.median_pe - hv.p20_pe, 1) * 40
                    ELSE 60 * (2 - cm.pe_ratio / NULLIF(hv.median_pe, 1))
                END as pe_percentile_score,
                -- Excess CF yield
                CASE 
                    WHEN cm.shares_outstanding > 0 AND cm.current_price > 0 THEN
                        (cm.excess_cash_flow / cm.shares_outstanding) / cm.current_price * 100
                    ELSE 0
                END as excess_cf_yield
            FROM current_metrics cm
            CROSS JOIN historical_valuations hv
        """)
        
        with self.engine.connect() as conn:
            result = conn.execute(query, {'symbol': symbol})
            row = result.fetchone()
        
        if not row:
            return {
                'symbol': symbol,
                'value_score': 0,
                'valuation_percentile': 0,
                'excess_cf_yield': 0,
                'margin_of_safety': 0
            }
        
        # Calculate components
        valuation_score = float(row['pe_percentile_score']) if row['pe_percentile_score'] else 0
        cf_yield = float(row['excess_cf_yield']) if row['excess_cf_yield'] else 0
        
        # Simple margin of safety based on PE vs historical
        margin_of_safety = 0
        if row['pe_ratio'] and row['median_pe']:
            if row['pe_ratio'] < row['median_pe']:
                margin_of_safety = min(100, (1 - row['pe_ratio'] / row['median_pe']) * 200)
        
        # Weighted value score
        value_score = (
            valuation_score * 0.40 +
            min(100, cf_yield * 10) * 0.30 +  # Cap CF yield contribution at 100
            margin_of_safety * 0.30
        )
        
        return {
            'symbol': symbol,
            'value_score': value_score,
            'valuation_percentile': valuation_score,
            'excess_cf_yield': cf_yield,
            'margin_of_safety': margin_of_safety
        }
    
    def calculate_growth_score(self, symbol: str) -> Dict:
        """
        Calculate growth score for a stock
        
        Components:
        - 35% 10-year S&P 500 alpha
        - 35% 5-year fundamental growth rate
        - 30% Forward growth indicators
        """
        
        query = text("""
            WITH stock_returns AS (
                SELECT 
                    symbol,
                    -- 10-year return
                    (SELECT close_price FROM stock_prices 
                     WHERE symbol = :symbol AND date = (SELECT MAX(date) FROM stock_prices WHERE symbol = :symbol)) /
                    NULLIF((SELECT close_price FROM stock_prices 
                     WHERE symbol = :symbol AND date <= CURRENT_DATE - INTERVAL '10 years'
                     ORDER BY date DESC LIMIT 1), 0) - 1 as return_10y,
                    -- 5-year return
                    (SELECT close_price FROM stock_prices 
                     WHERE symbol = :symbol AND date = (SELECT MAX(date) FROM stock_prices WHERE symbol = :symbol)) /
                    NULLIF((SELECT close_price FROM stock_prices 
                     WHERE symbol = :symbol AND date <= CURRENT_DATE - INTERVAL '5 years'
                     ORDER BY date DESC LIMIT 1), 0) - 1 as return_5y
            ),
            sp500_returns AS (
                SELECT 
                    -- 10-year S&P return
                    (SELECT close_price FROM sp500_history 
                     WHERE date = (SELECT MAX(date) FROM sp500_history)) /
                    NULLIF((SELECT close_price FROM sp500_history 
                     WHERE date <= CURRENT_DATE - INTERVAL '10 years'
                     ORDER BY date DESC LIMIT 1), 0) - 1 as sp500_return_10y,
                    -- 5-year S&P return
                    (SELECT close_price FROM sp500_history 
                     WHERE date = (SELECT MAX(date) FROM sp500_history)) /
                    NULLIF((SELECT close_price FROM sp500_history 
                     WHERE date <= CURRENT_DATE - INTERVAL '5 years'
                     ORDER BY date DESC LIMIT 1), 0) - 1 as sp500_return_5y
            ),
            fundamental_growth AS (
                SELECT 
                    symbol,
                    -- 5-year revenue CAGR
                    POWER(
                        NULLIF((SELECT total_revenue_ttm FROM fundamentals 
                         WHERE symbol = :symbol AND period_type = 'annual'
                         ORDER BY fiscal_date_ending DESC LIMIT 1), 0) /
                        NULLIF((SELECT total_revenue_ttm FROM fundamentals 
                         WHERE symbol = :symbol AND period_type = 'annual'
                         AND fiscal_date_ending <= CURRENT_DATE - INTERVAL '5 years'
                         ORDER BY fiscal_date_ending DESC LIMIT 1), 0),
                        0.2
                    ) - 1 as revenue_cagr_5y,
                    -- Excess cash flow growth
                    (SELECT trend_5y FROM excess_cash_flow_metrics WHERE symbol = :symbol) as ecf_trend
                FROM (SELECT :symbol as symbol) s
            )
            SELECT 
                sr.symbol,
                sr.return_10y,
                sr.return_5y,
                sp.sp500_return_10y,
                sp.sp500_return_5y,
                sr.return_10y - sp.sp500_return_10y as alpha_10y,
                sr.return_5y - sp.sp500_return_5y as alpha_5y,
                fg.revenue_cagr_5y,
                fg.ecf_trend,
                cfo.quarterly_revenue_growth,
                cfo.quarterly_earnings_growth
            FROM stock_returns sr
            CROSS JOIN sp500_returns sp
            LEFT JOIN fundamental_growth fg ON sr.symbol = fg.symbol
            LEFT JOIN company_fundamentals_overview cfo ON sr.symbol = cfo.symbol
        """)
        
        with self.engine.connect() as conn:
            result = conn.execute(query, {'symbol': symbol})
            row = result.fetchone()
        
        if not row:
            return {
                'symbol': symbol,
                'growth_score': 0,
                'sp500_alpha_10y': 0,
                'fundamental_growth_5y': 0,
                'forward_growth_estimate': 0
            }
        
        # Calculate components
        # 10-year alpha score (convert to 0-100)
        alpha_10y = float(row['alpha_10y']) * 100 if row['alpha_10y'] else 0
        alpha_score = min(100, max(0, 50 + alpha_10y * 2))  # Center at 50, scale appropriately
        
        # Fundamental growth score
        revenue_growth = float(row['revenue_cagr_5y']) * 100 if row['revenue_cagr_5y'] else 0
        ecf_trend_score = 100 if row['ecf_trend'] == 'Advancing' else 50 if row['ecf_trend'] == 'Stable' else 0
        fundamental_score = (min(100, revenue_growth * 5) + ecf_trend_score) / 2
        
        # Forward growth indicators
        qtr_revenue = float(row['quarterly_revenue_growth']) * 100 if row['quarterly_revenue_growth'] else 0
        qtr_earnings = float(row['quarterly_earnings_growth']) * 100 if row['quarterly_earnings_growth'] else 0
        forward_score = min(100, (max(0, qtr_revenue) + max(0, qtr_earnings)) / 2 * 2)
        
        # Weighted growth score
        growth_score = (
            alpha_score * 0.35 +
            fundamental_score * 0.35 +
            forward_score * 0.30
        )
        
        return {
            'symbol': symbol,
            'growth_score': growth_score,
            'sp500_alpha_10y': alpha_10y,
            'fundamental_growth_5y': fundamental_score,
            'forward_growth_estimate': forward_score
        }
    
    def calculate_dividend_score(self, symbol: str) -> Dict:
        """
        Calculate dividend score for a stock
        Uses existing dividend_sustainability_metrics
        """
        
        query = text("""
            SELECT 
                symbol,
                dividend_quality_score,
                sustainability_score,
                avg_growth_rate_5y,
                payment_streak_years
            FROM dividend_sustainability_metrics
            WHERE symbol = :symbol
        """)
        
        with self.engine.connect() as conn:
            result = conn.execute(query, {'symbol': symbol})
            row = result.fetchone()
        
        if not row:
            return {
                'symbol': symbol,
                'dividend_score': 0,
                'dividend_sustainability': 0,
                'dividend_growth_rate': 0,
                'payment_history_score': 0
            }
        
        # Use the pre-calculated dividend quality score
        dividend_score = float(row['dividend_quality_score']) if row['dividend_quality_score'] else 0
        
        return {
            'symbol': symbol,
            'dividend_score': dividend_score,
            'dividend_sustainability': float(row['sustainability_score']) if row['sustainability_score'] else 0,
            'dividend_growth_rate': float(row['avg_growth_rate_5y']) if row['avg_growth_rate_5y'] else 0,
            'payment_history_score': min(100, row['payment_streak_years'] * 4) if row['payment_streak_years'] else 0
        }
    
    def calculate_all_scores(self) -> pd.DataFrame:
        """Calculate all three scores for all eligible stocks"""
        
        # Get eligible stocks
        query = text("""
            SELECT DISTINCT su.symbol
            FROM symbol_universe su
            JOIN excess_cash_flow_metrics ecf ON su.symbol = ecf.symbol
            WHERE su.market_cap >= :min_cap
                AND su.is_etf = FALSE
                AND su.security_type = 'Common Stock'
                AND su.country = 'USA'
                AND ecf.excess_cash_flow_pct IS NOT NULL
        """)
        
        with self.engine.connect() as conn:
            result = conn.execute(query, {'min_cap': self.MIN_MARKET_CAP})
            symbols = [row[0] for row in result.fetchall()]
        
        logger.info(f"Calculating scores for {len(symbols)} eligible stocks...")
        
        all_scores = []
        for symbol in symbols:
            value_data = self.calculate_value_score(symbol)
            growth_data = self.calculate_growth_score(symbol)
            dividend_data = self.calculate_dividend_score(symbol)
            
            all_scores.append({
                'symbol': symbol,
                'calculation_date': datetime.now().date(),
                # Scores
                'value_score': value_data['value_score'],
                'growth_score': growth_data['growth_score'],
                'dividend_score': dividend_data['dividend_score'],
                # Value components
                'valuation_percentile': value_data['valuation_percentile'],
                'excess_cf_yield': value_data['excess_cf_yield'],
                'margin_of_safety': value_data['margin_of_safety'],
                # Growth components
                'sp500_alpha_10y': growth_data['sp500_alpha_10y'],
                'fundamental_growth_5y': growth_data['fundamental_growth_5y'],
                'forward_growth_estimate': growth_data['forward_growth_estimate'],
                # Dividend components
                'dividend_sustainability': dividend_data['dividend_sustainability'],
                'dividend_growth_rate': dividend_data['dividend_growth_rate'],
                'payment_history_score': dividend_data['payment_history_score']
            })
        
        df = pd.DataFrame(all_scores)
        
        # Check if we have any data
        if df.empty:
            logger.warning("No scores calculated - returning empty DataFrame")
            return df
        
        # Calculate rankings only if we have data
        if 'value_score' in df.columns and not df['value_score'].isna().all():
            df['value_rank'] = df['value_score'].rank(ascending=False, method='min').astype(int)
        else:
            df['value_rank'] = None
            
        if 'growth_score' in df.columns and not df['growth_score'].isna().all():
            df['growth_rank'] = df['growth_score'].rank(ascending=False, method='min').astype(int)
        else:
            df['growth_rank'] = None
            
        if 'dividend_score' in df.columns and not df['dividend_score'].isna().all():
            df['dividend_rank'] = df['dividend_score'].rank(ascending=False, method='min').astype(int)
        else:
            df['dividend_rank'] = None
        
        return df
    
    def select_portfolio(self, portfolio_type: str, scores_df: pd.DataFrame) -> List[str]:
        """
        Select top 10 stocks for a portfolio with diversification constraints
        
        Args:
            portfolio_type: 'VALUE', 'GROWTH', or 'DIVIDEND'
            scores_df: DataFrame with all scores
        
        Returns:
            List of selected symbols
        """
        
        # Get sector/industry data
        query = text("""
            SELECT symbol, sector, industry, market_cap
            FROM symbol_universe
            WHERE symbol = ANY(:symbols)
        """)
        
        with self.engine.connect() as conn:
            result = conn.execute(query, {'symbols': scores_df['symbol'].tolist()})
            sector_df = pd.DataFrame(result.fetchall(), columns=['symbol', 'sector', 'industry', 'market_cap'])
        
        # Merge with scores
        df = scores_df.merge(sector_df, on='symbol')
        
        # Filter by minimum excess cash flow requirements
        ecf_query = text("""
            SELECT symbol, excess_cash_flow_pct
            FROM excess_cash_flow_metrics
            WHERE symbol = ANY(:symbols)
        """)
        
        with self.engine.connect() as conn:
            result = conn.execute(ecf_query, {'symbols': df['symbol'].tolist()})
            ecf_df = pd.DataFrame(result.fetchall(), columns=['symbol', 'excess_cash_flow_pct'])
        
        df = df.merge(ecf_df, on='symbol')
        
        # Apply portfolio-specific filters
        if portfolio_type == 'VALUE':
            df = df[df['excess_cash_flow_pct'] >= self.MIN_VALUE_EXCESS_CF]
            df = df.sort_values('value_rank')
            score_col = 'value_score'
        elif portfolio_type == 'GROWTH':
            df = df[df['excess_cash_flow_pct'] >= self.MIN_GROWTH_EXCESS_CF]
            df = df.sort_values('growth_rank')
            score_col = 'growth_score'
        elif portfolio_type == 'DIVIDEND':
            df = df[df['excess_cash_flow_pct'] >= self.MIN_DIVIDEND_EXCESS_CF]
            df = df[df['dividend_score'] >= self.MIN_DIVIDEND_SCORE]
            df = df.sort_values('dividend_rank')
            score_col = 'dividend_score'
        else:
            raise ValueError(f"Invalid portfolio type: {portfolio_type}")
        
        # Select with diversification constraints
        selected = []
        sector_counts = {}
        industry_counts = {}
        
        for _, row in df.iterrows():
            if len(selected) >= self.PORTFOLIO_SIZE:
                break
            
            sector = row['sector']
            industry = row['industry']
            
            # Check sector concentration
            if sector in sector_counts:
                if sector_counts[sector] >= 3:  # Max 3 per sector for 10 stocks
                    continue
            
            # Check industry concentration
            if industry in industry_counts:
                if industry_counts[industry] >= 2:  # Max 2 per industry
                    continue
            
            # Add to portfolio
            selected.append(row['symbol'])
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
            industry_counts[industry] = industry_counts.get(industry, 0) + 1
        
        logger.info(f"Selected {len(selected)} stocks for {portfolio_type} portfolio")
        return selected
    
    def rebalance_portfolio(self, portfolio_type: str, 
                          rebalance_type: str = 'SCHEDULED') -> Dict:
        """
        Rebalance a portfolio
        
        Args:
            portfolio_type: 'VALUE', 'GROWTH', or 'DIVIDEND'
            rebalance_type: 'SCHEDULED', 'FORCED', or 'OPPORTUNISTIC'
        
        Returns:
            Dictionary with rebalance details
        """
        
        # Calculate current scores
        scores_df = self.calculate_all_scores()
        
        # Save scores to database
        self.save_scores(scores_df)
        
        # Get current holdings
        current_holdings = self.get_current_holdings(portfolio_type)
        
        # Select new portfolio
        new_holdings = self.select_portfolio(portfolio_type, scores_df)
        
        # Determine changes
        stocks_added = list(set(new_holdings) - set(current_holdings))
        stocks_removed = list(set(current_holdings) - set(new_holdings))
        stocks_retained = list(set(current_holdings) & set(new_holdings))
        
        # Calculate average scores
        old_scores = scores_df[scores_df['symbol'].isin(current_holdings)][f"{portfolio_type.lower()}_score"].mean() if current_holdings else 0
        new_scores = scores_df[scores_df['symbol'].isin(new_holdings)][f"{portfolio_type.lower()}_score"].mean()
        
        # Record rebalance
        rebalance_data = {
            'portfolio_type': portfolio_type,
            'rebalance_date': datetime.now().date(),
            'rebalance_type': rebalance_type,
            'stocks_added': stocks_added,
            'stocks_removed': stocks_removed,
            'stocks_retained': stocks_retained,
            'avg_score_before': old_scores,
            'avg_score_after': new_scores,
            'trigger_reason': f"{rebalance_type} rebalance"
        }
        
        # Save to database
        self.save_rebalance(rebalance_data)
        self.update_holdings(portfolio_type, new_holdings, scores_df)
        
        logger.info(f"Rebalanced {portfolio_type} portfolio: "
                   f"Added {len(stocks_added)}, Removed {len(stocks_removed)}, "
                   f"Retained {len(stocks_retained)}")
        
        return rebalance_data
    
    def get_current_holdings(self, portfolio_type: str) -> List[str]:
        """Get current active holdings for a portfolio"""
        
        query = text("""
            SELECT symbol
            FROM portfolio_holdings
            WHERE portfolio_type = :portfolio_type
                AND is_active = TRUE
            ORDER BY selection_score DESC
        """)
        
        with self.engine.connect() as conn:
            result = conn.execute(query, {'portfolio_type': portfolio_type})
            return [row[0] for row in result.fetchall()]
    
    def save_scores(self, scores_df: pd.DataFrame):
        """Save portfolio scores to database"""
        
        scores_df['created_at'] = datetime.now()
        
        with self.engine.begin() as conn:
            scores_df.to_sql('portfolio_scores', conn, if_exists='append', index=False)
        
        logger.info(f"Saved {len(scores_df)} portfolio scores to database")
    
    def save_rebalance(self, rebalance_data: Dict):
        """Save rebalance record to database"""
        
        with self.engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO portfolio_rebalances (
                    portfolio_type, rebalance_date, rebalance_type,
                    stocks_added, stocks_removed, stocks_retained,
                    avg_score_before, avg_score_after, trigger_reason
                ) VALUES (
                    :portfolio_type, :rebalance_date, :rebalance_type,
                    :stocks_added, :stocks_removed, :stocks_retained,
                    :avg_score_before, :avg_score_after, :trigger_reason
                )
            """), rebalance_data)
    
    def update_holdings(self, portfolio_type: str, new_holdings: List[str], 
                       scores_df: pd.DataFrame):
        """Update portfolio holdings in database"""
        
        # Mark old holdings as inactive
        with self.engine.begin() as conn:
            conn.execute(text("""
                UPDATE portfolio_holdings
                SET is_active = FALSE,
                    exit_date = CURRENT_DATE,
                    exit_reason = 'Rebalance'
                WHERE portfolio_type = :portfolio_type
                    AND is_active = TRUE
            """), {'portfolio_type': portfolio_type})
            
            # Add new holdings
            score_col = f"{portfolio_type.lower()}_score"
            rank_col = f"{portfolio_type.lower()}_rank"
            
            for symbol in new_holdings:
                row = scores_df[scores_df['symbol'] == symbol].iloc[0]
                
                # Get current price
                price_result = conn.execute(text("""
                    SELECT close_price
                    FROM stock_prices
                    WHERE symbol = :symbol
                    ORDER BY date DESC
                    LIMIT 1
                """), {'symbol': symbol})
                current_price = price_result.fetchone()[0]
                
                conn.execute(text("""
                    INSERT INTO portfolio_holdings (
                        portfolio_type, symbol, selection_date,
                        selection_score, selection_rank,
                        initial_weight, current_weight,
                        entry_price, current_price,
                        is_active
                    ) VALUES (
                        :portfolio_type, :symbol, CURRENT_DATE,
                        :score, :rank,
                        10.0, 10.0,
                        :price, :price,
                        TRUE
                    )
                """), {
                    'portfolio_type': portfolio_type,
                    'symbol': symbol,
                    'score': row[score_col],
                    'rank': row[rank_col],
                    'price': current_price
                })


def main():
    """Main execution"""
    
    manager = PortfolioManager()
    
    print("\n" + "="*70)
    print("ACIS THREE-PORTFOLIO STRATEGY")
    print("="*70)
    
    # Calculate all scores
    print("\nCalculating portfolio scores for all stocks...")
    scores_df = manager.calculate_all_scores()
    
    # Check if we have any data
    if scores_df.empty:
        print("\nNo eligible stocks found for portfolio scoring.")
        print("This usually means:")
        print("  1. Fundamental data hasn't been fetched yet")
        print("  2. Stock prices are missing")
        print("  3. Excess cash flow metrics haven't been calculated")
        print("\nRun the following to populate data:")
        print("  python master_control.py --daily")
        print("  # Or for specific data:")
        print("  python data_fetch/fundamentals/fetch_fundamentals.py")
        print("  python data_fetch/market_data/fetch_prices.py")
        print("  python analysis/excess_cash_flow.py")
        return 1
    
    # Display top candidates for each portfolio
    print("\n" + "="*70)
    print("TOP 10 VALUE STOCKS:")
    print("="*70)
    value_picks = manager.select_portfolio('VALUE', scores_df)
    for i, symbol in enumerate(value_picks, 1):
        row = scores_df[scores_df['symbol'] == symbol].iloc[0]
        print(f"  {i}. {symbol}: Score {row['value_score']:.1f} "
              f"(Valuation: {row['valuation_percentile']:.1f}, "
              f"CF Yield: {row['excess_cf_yield']:.1f}%)")
    
    print("\n" + "="*70)
    print("TOP 10 GROWTH STOCKS:")
    print("="*70)
    growth_picks = manager.select_portfolio('GROWTH', scores_df)
    for i, symbol in enumerate(growth_picks, 1):
        row = scores_df[scores_df['symbol'] == symbol].iloc[0]
        print(f"  {i}. {symbol}: Score {row['growth_score']:.1f} "
              f"(10Y Alpha: {row['sp500_alpha_10y']:.1f}%, "
              f"Growth: {row['fundamental_growth_5y']:.1f})")
    
    print("\n" + "="*70)
    print("TOP 10 DIVIDEND STOCKS:")
    print("="*70)
    dividend_picks = manager.select_portfolio('DIVIDEND', scores_df)
    for i, symbol in enumerate(dividend_picks, 1):
        row = scores_df[scores_df['symbol'] == symbol].iloc[0]
        print(f"  {i}. {symbol}: Score {row['dividend_score']:.1f} "
              f"(Sustainability: {row['dividend_sustainability']:.1f}, "
              f"Growth: {row['dividend_growth_rate']:.1f}%)")
    
    # Perform rebalancing
    print("\n" + "="*70)
    print("REBALANCING PORTFOLIOS")
    print("="*70)
    
    for portfolio_type in ['VALUE', 'GROWTH', 'DIVIDEND']:
        result = manager.rebalance_portfolio(portfolio_type)
        print(f"\n{portfolio_type} Portfolio:")
        print(f"  Added: {result['stocks_added']}")
        print(f"  Removed: {result['stocks_removed']}")
        print(f"  Score improved from {result['avg_score_before']:.1f} to {result['avg_score_after']:.1f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())