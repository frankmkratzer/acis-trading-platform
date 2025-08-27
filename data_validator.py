#!/usr/bin/env python3
"""
Enhanced Data Validator with Statistical Anomaly Detection
Implements sophisticated validation checks for financial data quality
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from datetime import datetime, timedelta
from scipy import stats
from sqlalchemy import text
from db_connection_manager import get_db_engine

logger = logging.getLogger(__name__)

class DataValidator:
    """Enhanced data validation with statistical anomaly detection"""
    
    def __init__(self, z_score_threshold: float = 3.0, 
                 max_price_change_pct: float = 0.5,
                 min_volume_threshold: int = 100):
        """
        Initialize data validator
        
        Args:
            z_score_threshold: Z-score threshold for outlier detection
            max_price_change_pct: Maximum allowed daily price change percentage
            min_volume_threshold: Minimum volume to consider valid
        """
        self.z_score_threshold = z_score_threshold
        self.max_price_change_pct = max_price_change_pct
        self.min_volume_threshold = min_volume_threshold
        self.engine = get_db_engine()
        
    def validate_price_data(self, df: pd.DataFrame, symbol: Optional[str] = None) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Comprehensive price data validation with statistical checks
        
        Args:
            df: DataFrame with price data
            symbol: Optional symbol for historical comparison
            
        Returns:
            Tuple of (is_valid, issues_list, metrics_dict)
        """
        issues = []
        metrics = {}
        
        if df.empty:
            return False, ["No data provided"], {}
        
        # Basic structural checks
        structural_issues = self._check_structure(df)
        issues.extend(structural_issues)
        
        # Data type checks
        type_issues = self._check_data_types(df)
        issues.extend(type_issues)
        
        # Statistical anomaly detection
        anomaly_issues, anomaly_metrics = self._detect_statistical_anomalies(df)
        issues.extend(anomaly_issues)
        metrics.update(anomaly_metrics)
        
        # Price consistency checks
        price_issues = self._check_price_consistency(df)
        issues.extend(price_issues)
        
        # Volume analysis
        volume_issues, volume_metrics = self._analyze_volume(df)
        issues.extend(volume_issues)
        metrics.update(volume_metrics)
        
        # Historical consistency (if symbol provided)
        if symbol:
            historical_issues = self._check_historical_consistency(df, symbol)
            issues.extend(historical_issues)
        
        # Cross-validation between fields
        cross_issues = self._cross_validate_fields(df)
        issues.extend(cross_issues)
        
        # Calculate overall data quality score
        quality_score = self._calculate_quality_score(df, issues)
        metrics['quality_score'] = quality_score
        
        is_valid = len(issues) == 0 or quality_score >= 0.7  # 70% quality threshold
        
        return is_valid, issues, metrics
    
    def _check_structure(self, df: pd.DataFrame) -> List[str]:
        """Check DataFrame structure and required columns"""
        issues = []
        
        # Required columns for price data
        required_cols = ['symbol', 'trade_date', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        # Check for duplicate dates per symbol
        if 'symbol' in df.columns and 'trade_date' in df.columns:
            duplicates = df.duplicated(subset=['symbol', 'trade_date'])
            if duplicates.any():
                issues.append(f"Duplicate symbol-date combinations: {duplicates.sum()}")
        
        return issues
    
    def _check_data_types(self, df: pd.DataFrame) -> List[str]:
        """Check data types and null values"""
        issues = []
        
        # Check for null values in critical columns
        critical_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in critical_cols:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    null_pct = (null_count / len(df)) * 100
                    issues.append(f"Null values in {col}: {null_count} ({null_pct:.1f}%)")
        
        # Check for negative prices
        price_cols = ['open', 'high', 'low', 'close', 'adjusted_close']
        for col in price_cols:
            if col in df.columns:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    issues.append(f"Negative values in {col}: {negative_count}")
        
        # Check for negative volume
        if 'volume' in df.columns:
            negative_volume = (df['volume'] < 0).sum()
            if negative_volume > 0:
                issues.append(f"Negative volume: {negative_volume} records")
        
        return issues
    
    def _detect_statistical_anomalies(self, df: pd.DataFrame) -> Tuple[List[str], Dict[str, Any]]:
        """Detect statistical anomalies using z-scores and other methods"""
        issues = []
        metrics = {}
        
        if 'close' not in df.columns or len(df) < 10:
            return issues, metrics
        
        # Calculate returns for anomaly detection
        df_sorted = df.sort_values('trade_date') if 'trade_date' in df.columns else df
        returns = df_sorted['close'].pct_change()
        
        # Z-score based outlier detection
        z_scores = np.abs(stats.zscore(returns.dropna()))
        outliers = z_scores > self.z_score_threshold
        outlier_count = outliers.sum()
        
        if outlier_count > 0:
            outlier_pct = (outlier_count / len(returns.dropna())) * 100
            if outlier_pct > 5:  # More than 5% outliers is suspicious
                issues.append(f"Statistical outliers detected: {outlier_count} ({outlier_pct:.1f}%)")
        
        metrics['outlier_count'] = int(outlier_count)
        metrics['max_z_score'] = float(z_scores.max()) if len(z_scores) > 0 else 0
        
        # Detect extreme price movements
        extreme_moves = returns.abs() > self.max_price_change_pct
        if extreme_moves.any():
            extreme_count = extreme_moves.sum()
            issues.append(f"Extreme price movements (>{self.max_price_change_pct*100}%): {extreme_count}")
            metrics['extreme_moves'] = int(extreme_count)
        
        # Check for price stagnation (no movement for extended periods)
        if len(df_sorted) > 5:
            rolling_std = df_sorted['close'].rolling(window=5).std()
            stagnant = rolling_std == 0
            if stagnant.any():
                stagnant_periods = stagnant.sum()
                if stagnant_periods > len(df_sorted) * 0.1:  # More than 10% stagnant
                    issues.append(f"Price stagnation detected: {stagnant_periods} periods")
        
        return issues, metrics
    
    def _check_price_consistency(self, df: pd.DataFrame) -> List[str]:
        """Check OHLC price consistency"""
        issues = []
        
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            return issues
        
        # High should be >= all other prices
        high_violations = ((df['high'] < df['low']) | 
                          (df['high'] < df['open']) | 
                          (df['high'] < df['close'])).sum()
        if high_violations > 0:
            issues.append(f"High price violations: {high_violations} records")
        
        # Low should be <= all other prices
        low_violations = ((df['low'] > df['high']) | 
                         (df['low'] > df['open']) | 
                         (df['low'] > df['close'])).sum()
        if low_violations > 0:
            issues.append(f"Low price violations: {low_violations} records")
        
        # Check for zero prices (suspicious unless stock delisted)
        zero_prices = ((df['open'] == 0) | (df['high'] == 0) | 
                      (df['low'] == 0) | (df['close'] == 0)).sum()
        if zero_prices > 0:
            issues.append(f"Zero prices detected: {zero_prices} records")
        
        return issues
    
    def _analyze_volume(self, df: pd.DataFrame) -> Tuple[List[str], Dict[str, Any]]:
        """Analyze volume patterns for anomalies"""
        issues = []
        metrics = {}
        
        if 'volume' not in df.columns:
            return issues, metrics
        
        # Calculate volume statistics
        mean_volume = df['volume'].mean()
        median_volume = df['volume'].median()
        zero_volume_count = (df['volume'] == 0).sum()
        zero_volume_pct = (zero_volume_count / len(df)) * 100
        
        metrics['mean_volume'] = float(mean_volume)
        metrics['median_volume'] = float(median_volume)
        metrics['zero_volume_pct'] = float(zero_volume_pct)
        
        # Check for excessive zero volume
        if zero_volume_pct > 20:  # More than 20% zero volume is suspicious
            issues.append(f"High zero volume percentage: {zero_volume_pct:.1f}%")
        
        # Check for suspiciously low volume
        low_volume = df['volume'] < self.min_volume_threshold
        if low_volume.any():
            low_volume_pct = (low_volume.sum() / len(df)) * 100
            if low_volume_pct > 50:
                issues.append(f"Suspiciously low volume: {low_volume_pct:.1f}% below {self.min_volume_threshold}")
        
        # Detect volume spikes (potential manipulation)
        if len(df) > 10:
            volume_z_scores = np.abs(stats.zscore(df['volume']))
            volume_spikes = volume_z_scores > 4  # Very high z-score
            if volume_spikes.any():
                spike_count = volume_spikes.sum()
                issues.append(f"Volume spikes detected: {spike_count} records")
                metrics['volume_spike_count'] = int(spike_count)
        
        return issues, metrics
    
    def _check_historical_consistency(self, df: pd.DataFrame, symbol: str) -> List[str]:
        """Check consistency with historical data"""
        issues = []
        
        try:
            # Get historical statistics for comparison
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT 
                        AVG(close_price) as avg_price,
                        STDDEV(close_price) as std_price,
                        AVG(volume) as avg_volume
                    FROM stock_prices
                    WHERE symbol = :symbol
                    AND date >= CURRENT_DATE - INTERVAL '1 year'
                """), {"symbol": symbol})
                
                historical = result.fetchone()
                if historical and historical[0]:
                    hist_avg_price = float(historical[0])
                    hist_std_price = float(historical[1]) if historical[1] else 0
                    hist_avg_volume = float(historical[2]) if historical[2] else 0
                    
                    # Check if new data is consistent with historical patterns
                    if 'close' in df.columns:
                        new_avg_price = df['close'].mean()
                        price_deviation = abs(new_avg_price - hist_avg_price) / hist_avg_price
                        
                        if price_deviation > 0.5:  # 50% deviation from historical average
                            issues.append(f"Large deviation from historical price: {price_deviation*100:.1f}%")
                    
                    if 'volume' in df.columns and hist_avg_volume > 0:
                        new_avg_volume = df['volume'].mean()
                        volume_deviation = abs(new_avg_volume - hist_avg_volume) / hist_avg_volume
                        
                        if volume_deviation > 0.7:  # 70% deviation from historical volume
                            issues.append(f"Large deviation from historical volume: {volume_deviation*100:.1f}%")
        
        except Exception as e:
            logger.warning(f"Failed to check historical consistency: {e}")
        
        return issues
    
    def _cross_validate_fields(self, df: pd.DataFrame) -> List[str]:
        """Cross-validate relationships between fields"""
        issues = []
        
        # Check price-volume correlation during extreme moves
        if 'close' in df.columns and 'volume' in df.columns and len(df) > 10:
            df_sorted = df.sort_values('trade_date') if 'trade_date' in df.columns else df
            returns = df_sorted['close'].pct_change()
            
            # Large price moves should typically have higher volume
            large_moves = returns.abs() > 0.1  # 10% moves
            if large_moves.any():
                large_move_volumes = df_sorted.loc[large_moves, 'volume']
                normal_volumes = df_sorted.loc[~large_moves, 'volume']
                
                if len(large_move_volumes) > 0 and len(normal_volumes) > 0:
                    avg_large_volume = large_move_volumes.mean()
                    avg_normal_volume = normal_volumes.mean()
                    
                    if avg_large_volume < avg_normal_volume:
                        issues.append("Suspicious: Large price moves with lower than average volume")
        
        # Check for split adjustments
        if 'adjusted_close' in df.columns and 'close' in df.columns:
            adjustment_factor = df['adjusted_close'] / df['close']
            
            # Check for unreasonable adjustments
            unreasonable_adjustments = ((adjustment_factor < 0.1) | (adjustment_factor > 10)).sum()
            if unreasonable_adjustments > 0:
                issues.append(f"Unreasonable split adjustments: {unreasonable_adjustments} records")
        
        return issues
    
    def _calculate_quality_score(self, df: pd.DataFrame, issues: List[str]) -> float:
        """Calculate overall data quality score"""
        if df.empty:
            return 0.0
        
        # Start with perfect score
        score = 1.0
        
        # Deduct points for each issue type
        issue_weights = {
            'Missing required columns': 0.3,
            'Duplicate': 0.2,
            'Null values': 0.1,
            'Negative': 0.2,
            'outliers': 0.1,
            'Extreme price': 0.15,
            'violations': 0.15,
            'Zero prices': 0.2,
            'zero volume': 0.1,
            'deviation': 0.1
        }
        
        for issue in issues:
            for keyword, weight in issue_weights.items():
                if keyword.lower() in issue.lower():
                    score -= weight
                    break
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))
    
    def validate_fundamentals(self, df: pd.DataFrame) -> Tuple[bool, List[str], Dict[str, Any]]:
        """Validate fundamental data"""
        issues = []
        metrics = {}
        
        if df.empty:
            return False, ["No fundamentals data provided"], {}
        
        # Check for required columns
        required_cols = ['symbol', 'fiscal_date', 'revenue', 'earnings']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        # Check for reasonable value ranges
        if 'revenue' in df.columns:
            negative_revenue = (df['revenue'] < 0).sum()
            if negative_revenue > 0:
                issues.append(f"Negative revenue: {negative_revenue} records")
        
        # Check for reasonable ratios
        if 'pe_ratio' in df.columns:
            extreme_pe = ((df['pe_ratio'] < -100) | (df['pe_ratio'] > 1000)).sum()
            if extreme_pe > 0:
                issues.append(f"Extreme P/E ratios: {extreme_pe} records")
        
        # Check for data freshness
        if 'fiscal_date' in df.columns:
            df['fiscal_date'] = pd.to_datetime(df['fiscal_date'])
            latest_date = df['fiscal_date'].max()
            days_old = (datetime.now() - latest_date).days
            
            if days_old > 180:  # More than 6 months old
                issues.append(f"Stale fundamental data: {days_old} days old")
            
            metrics['data_age_days'] = days_old
        
        quality_score = self._calculate_quality_score(df, issues)
        metrics['quality_score'] = quality_score
        
        is_valid = len(issues) == 0 or quality_score >= 0.6  # Lower threshold for fundamentals
        
        return is_valid, issues, metrics

# Singleton instance
data_validator = DataValidator()

# Convenience functions
def validate_price_data(df: pd.DataFrame, symbol: Optional[str] = None) -> Tuple[bool, List[str]]:
    """Validate price data (simplified interface)"""
    is_valid, issues, _ = data_validator.validate_price_data(df, symbol)
    return is_valid, issues

def validate_fundamentals_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate fundamentals data (simplified interface)"""
    is_valid, issues, _ = data_validator.validate_fundamentals(df)
    return is_valid, issues