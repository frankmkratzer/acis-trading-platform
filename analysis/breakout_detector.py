#!/usr/bin/env python3
"""
Breakout Detector with Volume Confirmation
Identifies stocks breaking out of consolidation patterns with increasing volume

Key Signals:
1. Price breakouts above resistance/52-week highs
2. Volume surge (>150% of average)
3. Base formation quality
4. Momentum sustainability
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import text
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.logging_config import setup_logger
from database.db_connection_manager import DatabaseConnectionManager

logger = setup_logger("breakout_detector")
db_manager = DatabaseConnectionManager()
engine = db_manager.get_engine()


class BreakoutDetector:
    """Detect technical breakouts with volume confirmation"""
    
    # Configuration
    MIN_BASE_DAYS = 60  # Minimum consolidation period (3 months)
    VOLUME_SURGE_THRESHOLD = 1.5  # 150% of average volume
    BREAKOUT_THRESHOLD = 0.98  # Within 2% of 52-week high
    MAX_EXTENSION = 1.10  # Not more than 10% above breakout
    
    def __init__(self):
        self.engine = engine
        
    def calculate_breakout_score(self, symbol: str, lookback_days: int = 252) -> Dict:
        """
        Calculate comprehensive breakout score for a stock
        
        Components:
        1. Proximity to 52-week high
        2. Volume surge ratio
        3. Base formation quality
        4. Momentum strength
        5. Relative strength vs market
        """
        
        # Fetch price and volume data
        query = text("""
            WITH price_data AS (
                SELECT 
                    date,
                    close_price,
                    high,
                    low,
                    volume,
                    -- Calculate moving averages
                    AVG(close_price) OVER (ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as ma20,
                    AVG(close_price) OVER (ORDER BY date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) as ma50,
                    AVG(close_price) OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW) as ma200,
                    -- Volume averages
                    AVG(volume) OVER (ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as avg_volume_20,
                    AVG(volume) OVER (ORDER BY date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) as avg_volume_50,
                    -- 52-week high/low
                    MAX(high) OVER (ORDER BY date ROWS BETWEEN 251 PRECEDING AND CURRENT ROW) as high_52w,
                    MIN(low) OVER (ORDER BY date ROWS BETWEEN 251 PRECEDING AND CURRENT ROW) as low_52w,
                    -- Price change
                    close_price - LAG(close_price, 1) OVER (ORDER BY date) as daily_change,
                    (close_price - LAG(close_price, 1) OVER (ORDER BY date)) / 
                        NULLIF(LAG(close_price, 1) OVER (ORDER BY date), 0) as daily_return
                FROM stock_prices
                WHERE symbol = :symbol
                    AND date >= CURRENT_DATE - INTERVAL ':lookback days'
                ORDER BY date DESC
            ),
            volume_analysis AS (
                SELECT 
                    date,
                    volume,
                    avg_volume_20,
                    daily_change,
                    -- Accumulation/Distribution
                    CASE 
                        WHEN daily_change > 0 THEN volume
                        WHEN daily_change < 0 THEN -volume
                        ELSE 0
                    END as volume_direction,
                    -- Volume surge
                    volume / NULLIF(avg_volume_20, 0) as volume_ratio
                FROM price_data
                WHERE avg_volume_20 IS NOT NULL
            ),
            recent_data AS (
                SELECT * FROM price_data 
                WHERE date >= CURRENT_DATE - INTERVAL '30 days'
            )
            SELECT 
                -- Current metrics
                (SELECT close_price FROM price_data LIMIT 1) as current_price,
                (SELECT volume FROM price_data LIMIT 1) as current_volume,
                (SELECT high_52w FROM price_data LIMIT 1) as high_52w,
                (SELECT low_52w FROM price_data LIMIT 1) as low_52w,
                (SELECT ma20 FROM price_data LIMIT 1) as ma20,
                (SELECT ma50 FROM price_data LIMIT 1) as ma50,
                (SELECT ma200 FROM price_data LIMIT 1) as ma200,
                (SELECT avg_volume_20 FROM price_data LIMIT 1) as avg_volume,
                
                -- Breakout metrics
                (SELECT close_price FROM price_data LIMIT 1) / 
                    NULLIF((SELECT high_52w FROM price_data LIMIT 1), 0) as pct_of_52w_high,
                (SELECT volume_ratio FROM volume_analysis LIMIT 1) as current_volume_ratio,
                
                -- Base formation (volatility over past 3 months)
                (SELECT STDDEV(close_price) / AVG(close_price) 
                 FROM price_data 
                 WHERE date >= CURRENT_DATE - INTERVAL '90 days') as base_volatility,
                
                -- Momentum (consecutive up days)
                (SELECT COUNT(*) 
                 FROM price_data 
                 WHERE daily_return > 0 
                   AND date >= CURRENT_DATE - INTERVAL '10 days') as up_days_10d,
                
                -- Volume trend (accumulation vs distribution)
                (SELECT SUM(volume_direction) / SUM(ABS(volume_direction))
                 FROM volume_analysis
                 WHERE date >= CURRENT_DATE - INTERVAL '20 days') as accumulation_ratio,
                
                -- Relative strength vs S&P 500
                (SELECT 
                    ((SELECT close_price FROM price_data LIMIT 1) / 
                     (SELECT close_price FROM price_data WHERE date <= CURRENT_DATE - INTERVAL '20 days' LIMIT 1) - 1) -
                    ((SELECT close_price FROM sp500_history WHERE date = (SELECT MAX(date) FROM sp500_history)) /
                     (SELECT close_price FROM sp500_history WHERE date <= CURRENT_DATE - INTERVAL '20 days' ORDER BY date DESC LIMIT 1) - 1)
                 ) as relative_strength_20d
        """)
        
        with self.engine.connect() as conn:
            result = conn.execute(query, {
                'symbol': symbol,
                'lookback': lookback_days
            })
            data = result.fetchone()
        
        if not data:
            return {
                'symbol': symbol,
                'breakout_score': 0,
                'has_breakout': False
            }
        
        # Calculate component scores
        scores = self._calculate_component_scores(data)
        
        # Determine if this is a valid breakout
        has_breakout = self._is_valid_breakout(data)
        
        # Calculate weighted breakout score
        breakout_score = self._calculate_weighted_score(scores)
        
        return {
            'symbol': symbol,
            'breakout_score': breakout_score,
            'has_breakout': has_breakout,
            'current_price': float(data['current_price']) if data['current_price'] else None,
            'high_52w': float(data['high_52w']) if data['high_52w'] else None,
            'volume_surge': float(data['current_volume_ratio']) if data['current_volume_ratio'] else None,
            'base_quality': scores['base_quality'],
            'momentum_score': scores['momentum'],
            'relative_strength': float(data['relative_strength_20d']) if data['relative_strength_20d'] else None,
            'accumulation_ratio': float(data['accumulation_ratio']) if data['accumulation_ratio'] else None
        }
    
    def _calculate_component_scores(self, data) -> Dict[str, float]:
        """Calculate individual component scores"""
        
        scores = {}
        
        # 1. Proximity to 52-week high (0-100)
        if data['pct_of_52w_high']:
            if data['pct_of_52w_high'] >= 0.98:  # Within 2% of high
                scores['proximity'] = 100
            elif data['pct_of_52w_high'] >= 0.95:  # Within 5%
                scores['proximity'] = 80
            elif data['pct_of_52w_high'] >= 0.90:  # Within 10%
                scores['proximity'] = 60
            else:
                scores['proximity'] = max(0, data['pct_of_52w_high'] * 100 - 40)
        else:
            scores['proximity'] = 0
        
        # 2. Volume surge (0-100)
        if data['current_volume_ratio']:
            if data['current_volume_ratio'] >= 2.0:  # 200%+ of average
                scores['volume'] = 100
            elif data['current_volume_ratio'] >= 1.5:  # 150%+
                scores['volume'] = 80
            elif data['current_volume_ratio'] >= 1.2:  # 120%+
                scores['volume'] = 60
            else:
                scores['volume'] = data['current_volume_ratio'] * 50
        else:
            scores['volume'] = 0
        
        # 3. Base formation quality (tighter = better)
        if data['base_volatility']:
            # Lower volatility = better base
            if data['base_volatility'] <= 0.15:  # Very tight base
                scores['base_quality'] = 100
            elif data['base_volatility'] <= 0.25:
                scores['base_quality'] = 80
            elif data['base_volatility'] <= 0.35:
                scores['base_quality'] = 60
            else:
                scores['base_quality'] = max(0, 100 - data['base_volatility'] * 200)
        else:
            scores['base_quality'] = 0
        
        # 4. Momentum strength
        if data['up_days_10d']:
            scores['momentum'] = min(100, data['up_days_10d'] * 15)  # 7+ up days = 100
        else:
            scores['momentum'] = 0
        
        # 5. Accumulation (positive = buying pressure)
        if data['accumulation_ratio']:
            scores['accumulation'] = 50 + data['accumulation_ratio'] * 50  # -1 to 1 mapped to 0-100
        else:
            scores['accumulation'] = 50
        
        # 6. Relative strength vs market
        if data['relative_strength_20d']:
            scores['relative_strength'] = 50 + min(50, max(-50, data['relative_strength_20d'] * 500))
        else:
            scores['relative_strength'] = 50
        
        return scores
    
    def _is_valid_breakout(self, data) -> bool:
        """Determine if stock has a valid breakout signal"""
        
        if not data['pct_of_52w_high'] or not data['current_volume_ratio']:
            return False
        
        # Breakout criteria
        is_near_high = data['pct_of_52w_high'] >= self.BREAKOUT_THRESHOLD
        has_volume = data['current_volume_ratio'] >= self.VOLUME_SURGE_THRESHOLD
        not_extended = data['pct_of_52w_high'] <= self.MAX_EXTENSION
        
        # Above key moving averages
        above_ma = True
        if data['current_price'] and data['ma50'] and data['ma200']:
            above_ma = (data['current_price'] > data['ma50'] and 
                       data['current_price'] > data['ma200'])
        
        return is_near_high and has_volume and not_extended and above_ma
    
    def _calculate_weighted_score(self, scores: Dict[str, float]) -> float:
        """Calculate weighted breakout score"""
        
        weights = {
            'proximity': 0.25,      # 25% - How close to 52-week high
            'volume': 0.25,         # 25% - Volume surge strength
            'base_quality': 0.15,   # 15% - Consolidation pattern quality
            'momentum': 0.15,       # 15% - Recent momentum
            'accumulation': 0.10,   # 10% - Buy/sell pressure
            'relative_strength': 0.10  # 10% - Outperformance vs market
        }
        
        weighted_score = sum(scores.get(key, 0) * weight 
                           for key, weight in weights.items())
        
        return round(weighted_score, 2)
    
    def scan_for_breakouts(self, min_market_cap: float = 2_000_000_000) -> pd.DataFrame:
        """
        Scan entire universe for breakout candidates
        
        Args:
            min_market_cap: Minimum market cap filter
        
        Returns:
            DataFrame with breakout candidates ranked by score
        """
        
        # Get eligible symbols
        query = text("""
            SELECT DISTINCT su.symbol
            FROM symbol_universe su
            WHERE su.market_cap >= :min_cap
                AND su.is_etf = FALSE
                AND su.security_type = 'Common Stock'
                AND su.country = 'USA'
                AND EXISTS (
                    SELECT 1 FROM stock_prices sp 
                    WHERE sp.symbol = su.symbol 
                    AND sp.date >= CURRENT_DATE - INTERVAL '1 year'
                )
        """)
        
        with self.engine.connect() as conn:
            result = conn.execute(query, {'min_cap': min_market_cap})
            symbols = [row[0] for row in result.fetchall()]
        
        logger.info(f"Scanning {len(symbols)} stocks for breakouts...")
        
        # Calculate breakout scores for all symbols
        breakout_data = []
        for i, symbol in enumerate(symbols):
            if i % 100 == 0:
                logger.info(f"Progress: {i}/{len(symbols)} stocks scanned")
            
            score_data = self.calculate_breakout_score(symbol)
            if score_data['breakout_score'] > 0:
                breakout_data.append(score_data)
        
        # Convert to DataFrame and rank
        df = pd.DataFrame(breakout_data)
        
        if not df.empty:
            # Sort by breakout score
            df = df.sort_values('breakout_score', ascending=False)
            df['rank'] = range(1, len(df) + 1)
            
            # Filter for actual breakouts
            df['breakout_signal'] = df['has_breakout']
            
            # Add breakout strength category
            df['breakout_strength'] = pd.cut(
                df['breakout_score'],
                bins=[0, 40, 60, 80, 100],
                labels=['Weak', 'Moderate', 'Strong', 'Very Strong']
            )
        
        return df
    
    def save_to_database(self, df: pd.DataFrame):
        """Save breakout signals to database"""
        
        if df.empty:
            logger.warning("No breakout data to save")
            return
        
        # Create table if it doesn't exist
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS breakout_signals (
            symbol VARCHAR(10) NOT NULL,
            signal_date DATE NOT NULL,
            breakout_score NUMERIC(6, 2),
            has_breakout BOOLEAN,
            current_price NUMERIC(12, 4),
            high_52w NUMERIC(12, 4),
            volume_surge NUMERIC(8, 2),
            base_quality NUMERIC(6, 2),
            momentum_score NUMERIC(6, 2),
            relative_strength NUMERIC(8, 4),
            accumulation_ratio NUMERIC(8, 4),
            breakout_strength VARCHAR(20),
            rank INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, signal_date),
            FOREIGN KEY (symbol) REFERENCES symbol_universe(symbol) ON DELETE CASCADE
        )
        """
        
        with self.engine.begin() as conn:
            conn.execute(text(create_table_sql))
            
            # Prepare data for insert
            df['signal_date'] = datetime.now().date()
            df['created_at'] = datetime.now()
            
            # Save to database
            df.to_sql('breakout_signals', conn, if_exists='append', index=False)
            
        logger.info(f"Saved {len(df)} breakout signals to database")


def main():
    """Main execution"""
    
    detector = BreakoutDetector()
    
    print("\n" + "="*70)
    print("BREAKOUT DETECTION WITH VOLUME CONFIRMATION")
    print("="*70)
    
    # Example: Analyze a specific stock
    symbol = 'AAPL'
    result = detector.calculate_breakout_score(symbol)
    
    print(f"\n{symbol} Breakout Analysis:")
    print(f"  Breakout Score: {result['breakout_score']}/100")
    print(f"  Has Valid Breakout: {result['has_breakout']}")
    if result['current_price'] and result['high_52w']:
        print(f"  Price vs 52W High: ${result['current_price']:.2f} / ${result['high_52w']:.2f}")
    if result['volume_surge']:
        print(f"  Volume Surge: {result['volume_surge']:.1f}x average")
    if result['relative_strength']:
        print(f"  Relative Strength: {result['relative_strength']:.2%}")
    
    # Scan for all breakouts
    print("\n" + "="*70)
    print("SCANNING FOR BREAKOUT CANDIDATES")
    print("="*70)
    
    df = detector.scan_for_breakouts()
    
    if not df.empty:
        # Save to database
        detector.save_to_database(df)
        
        # Show top breakouts
        print("\nTop 20 Breakout Candidates:")
        print("-" * 70)
        for _, row in df.head(20).iterrows():
            signal = "🔥 BREAKOUT" if row['has_breakout'] else "  "
            print(f"{signal} #{row['rank']:3d} {row['symbol']:6s}: "
                  f"Score {row['breakout_score']:5.1f} "
                  f"({row['breakout_strength']}) "
                  f"Volume {row['volume_surge']:.1f}x")
        
        # Show statistics
        print(f"\nBreakout Statistics:")
        print(f"  Total candidates: {len(df)}")
        print(f"  Active breakouts: {df['has_breakout'].sum()}")
        print(f"  Very Strong (80+): {len(df[df['breakout_score'] >= 80])}")
        print(f"  Strong (60-80): {len(df[(df['breakout_score'] >= 60) & (df['breakout_score'] < 80)])}")
        print(f"  Average score: {df['breakout_score'].mean():.1f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())