#!/usr/bin/env python3
"""
World-Class Stock Quality Rankings Calculator
Institutional-grade quality rankings with sophisticated null handling,
data quality tracking, and confidence scoring
"""

import os
import sys
import time
import json
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
import joblib

warnings.filterwarnings('ignore')

# Configuration
load_dotenv()
POSTGRES_URL = os.getenv("POSTGRES_URL")
if not POSTGRES_URL:
    print("[ERROR] POSTGRES_URL not set")
    sys.exit(1)

engine = create_engine(POSTGRES_URL)

# Logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/calculate_quality_rankings.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("quality_rankings")

class DataQualityTracker:
    """Track and score data quality for each symbol"""
    
    def __init__(self):
        self.quality_metrics = {}
        self.imputation_log = []
        
    def calculate_completeness(self, df, symbol, required_fields):
        """Calculate data completeness percentage"""
        if symbol not in df.index:
            return 0
        
        symbol_data = df.loc[symbol]
        non_null_count = symbol_data[required_fields].notna().sum()
        completeness = (non_null_count / len(required_fields)) * 100
        
        self.quality_metrics[symbol] = {
            'completeness': completeness,
            'missing_fields': list(symbol_data[required_fields][symbol_data[required_fields].isna()].index)
        }
        
        return completeness
    
    def log_imputation(self, symbol, field, method, original_value, imputed_value):
        """Log all imputation activities for audit trail"""
        self.imputation_log.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'field': field,
            'method': method,
            'original': original_value,
            'imputed': imputed_value
        })
    
    def get_confidence_score(self, symbol, base_score):
        """Adjust score based on data quality"""
        if symbol not in self.quality_metrics:
            return base_score * 0.5  # 50% confidence if no quality data
        
        completeness = self.quality_metrics[symbol]['completeness'] / 100
        # Exponential penalty for low data quality
        confidence_multiplier = completeness ** 2
        
        return base_score * confidence_multiplier

class AdvancedImputer:
    """Sophisticated imputation strategies for financial data"""
    
    def __init__(self, df):
        self.df = df
        self.sector_medians = {}
        self.industry_medians = {}
        self._calculate_benchmarks()
    
    def _calculate_benchmarks(self):
        """Pre-calculate sector and industry benchmarks"""
        if 'sector' in self.df.columns:
            for metric in self.df.select_dtypes(include=[np.number]).columns:
                self.sector_medians[metric] = self.df.groupby('sector')[metric].median()
                if 'industry' in self.df.columns:
                    self.industry_medians[metric] = self.df.groupby('industry')[metric].median()
    
    def impute_by_hierarchy(self, symbol, metric):
        """Hierarchical imputation: Industry -> Sector -> Market"""
        row = self.df.loc[symbol]
        
        # Try industry median first
        if 'industry' in row and row['industry'] in self.industry_medians.get(metric, {}):
            return self.industry_medians[metric][row['industry']]
        
        # Fall back to sector median
        if 'sector' in row and row['sector'] in self.sector_medians.get(metric, {}):
            return self.sector_medians[metric][row['sector']]
        
        # Fall back to market median
        return self.df[metric].median()
    
    def impute_time_series(self, symbol_df, metric, method='linear'):
        """Time-series specific imputation"""
        if method == 'linear':
            return symbol_df[metric].interpolate(method='linear', limit=3)
        elif method == 'forward_fill':
            return symbol_df[metric].fillna(method='ffill', limit=5)
        elif method == 'seasonal':
            # Simple seasonal imputation (same quarter previous year)
            return symbol_df[metric].fillna(symbol_df[metric].shift(4))
        
    def ml_based_imputation(self, df, target_col, feature_cols):
        """Use Random Forest for sophisticated imputation"""
        # Separate complete and incomplete data
        complete_mask = df[target_col].notna()
        train_df = df[complete_mask]
        predict_df = df[~complete_mask]
        
        if len(train_df) < 100 or len(predict_df) == 0:
            # Not enough data for ML
            return df[target_col].fillna(df[target_col].median())
        
        # Prepare features
        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df[target_col]
        X_predict = predict_df[feature_cols].fillna(0)
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_predict_scaled = scaler.transform(X_predict)
        
        # Train model
        rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        rf.fit(X_train_scaled, y_train)
        
        # Predict missing values
        predictions = rf.predict(X_predict_scaled)
        
        # Fill in predictions
        result = df[target_col].copy()
        result[~complete_mask] = predictions
        
        return result

class OutlierHandler:
    """Detect and handle outliers in financial data"""
    
    @staticmethod
    def detect_outliers_iqr(series, multiplier=1.5):
        """Detect outliers using IQR method"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outliers = (series < lower_bound) | (series > upper_bound)
        return outliers, lower_bound, upper_bound
    
    @staticmethod
    def detect_outliers_zscore(series, threshold=3):
        """Detect outliers using z-score method"""
        z_scores = np.abs(stats.zscore(series.dropna()))
        outliers = z_scores > threshold
        return outliers
    
    @staticmethod
    def winsorize(series, limits=(0.05, 0.95)):
        """Cap outliers at specified percentiles"""
        return series.clip(
            lower=series.quantile(limits[0]),
            upper=series.quantile(limits[1])
        )
    
    @staticmethod
    def handle_outliers(df, columns, method='winsorize'):
        """Apply outlier handling to specified columns"""
        df_clean = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
                
            if method == 'winsorize':
                df_clean[col] = OutlierHandler.winsorize(df[col])
            elif method == 'remove':
                outliers, _, _ = OutlierHandler.detect_outliers_iqr(df[col])
                df_clean.loc[outliers, col] = np.nan
            elif method == 'cap':
                outliers, lower, upper = OutlierHandler.detect_outliers_iqr(df[col])
                df_clean[col] = df[col].clip(lower=lower, upper=upper)
        
        return df_clean

class WorldClassQualityRankingCalculator:
    """World-class quality rankings with institutional-grade features"""
    
    def __init__(self):
        self.engine = engine
        self.ranking_date = datetime.now().date()
        self.data_tracker = DataQualityTracker()
        self.min_data_requirements = {
            'min_quarters': 8,
            'min_price_points': 250,
            'min_fundamentals_coverage': 0.6,
            'min_years_history': 2
        }
        
    def validate_data_requirements(self, df, id_col='symbol'):
        """Filter symbols that meet minimum data requirements"""
        valid_symbols = []
        
        for symbol in df[id_col].unique():
            symbol_data = df[df[id_col] == symbol]
            
            # Check minimum data points
            if len(symbol_data) < self.min_data_requirements['min_quarters']:
                logger.info(f"Excluding {symbol}: insufficient quarters ({len(symbol_data)})")
                continue
            
            # Check data completeness
            important_cols = ['revenue', 'net_income', 'free_cash_flow', 'total_assets']
            completeness = symbol_data[important_cols].notna().sum().sum() / (len(important_cols) * len(symbol_data))
            
            if completeness < self.min_data_requirements['min_fundamentals_coverage']:
                logger.info(f"Excluding {symbol}: low data completeness ({completeness:.1%})")
                continue
            
            valid_symbols.append(symbol)
        
        logger.info(f"Data validation: {len(valid_symbols)}/{df[id_col].nunique()} symbols passed")
        return df[df[id_col].isin(valid_symbols)]
    
    def calculate_sp500_outperformance(self):
        """Calculate SP500 outperformance with confidence scoring"""
        print("[INFO] Calculating SP500 outperformance rankings (World-Class)...")
        
        # Enhanced query with data quality metrics
        query = text("""
            WITH data_quality AS (
                SELECT 
                    symbol,
                    COUNT(DISTINCT EXTRACT(YEAR FROM trade_date)) as years_of_data,
                    COUNT(*) as total_data_points,
                    MIN(trade_date) as first_date,
                    MAX(trade_date) as last_date
                FROM stock_prices
                WHERE trade_date >= CURRENT_DATE - INTERVAL '21 years'
                GROUP BY symbol
            ),
            sp500_returns AS (
                SELECT 
                    EXTRACT(YEAR FROM trade_date) as year,
                    (LAST_VALUE(adjusted_close) OVER (PARTITION BY EXTRACT(YEAR FROM trade_date) ORDER BY trade_date
                     ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) /
                     FIRST_VALUE(adjusted_close) OVER (PARTITION BY EXTRACT(YEAR FROM trade_date) ORDER BY trade_date) - 1) * 100 as sp500_return,
                    COUNT(*) OVER (PARTITION BY EXTRACT(YEAR FROM trade_date)) as trading_days
                FROM sp500_price_history
                WHERE trade_date >= CURRENT_DATE - INTERVAL '21 years'
                GROUP BY EXTRACT(YEAR FROM trade_date), trade_date, adjusted_close
            ),
            stock_returns AS (
                SELECT 
                    s.symbol,
                    EXTRACT(YEAR FROM s.trade_date) as year,
                    (LAST_VALUE(s.adjusted_close) OVER (PARTITION BY s.symbol, EXTRACT(YEAR FROM s.trade_date) ORDER BY s.trade_date
                     ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) /
                     FIRST_VALUE(s.adjusted_close) OVER (PARTITION BY s.symbol, EXTRACT(YEAR FROM s.trade_date) ORDER BY s.trade_date) - 1) * 100 as stock_return,
                    COUNT(*) OVER (PARTITION BY s.symbol, EXTRACT(YEAR FROM s.trade_date)) as symbol_trading_days,
                    dq.years_of_data,
                    dq.total_data_points
                FROM stock_prices s
                JOIN data_quality dq ON s.symbol = dq.symbol
                WHERE s.trade_date >= CURRENT_DATE - INTERVAL '21 years'
                  AND s.adjusted_close > 0
                  AND dq.years_of_data >= 3  -- Minimum 3 years required
                  AND dq.total_data_points >= 500  -- Minimum 500 data points
                GROUP BY s.symbol, EXTRACT(YEAR FROM s.trade_date), s.trade_date, s.adjusted_close, 
                         dq.years_of_data, dq.total_data_points
            )
            SELECT 
                s.symbol,
                s.year,
                s.stock_return,
                sp.sp500_return,
                s.stock_return - sp.sp500_return as excess_return,
                CASE WHEN s.stock_return > sp.sp500_return THEN TRUE ELSE FALSE END as beat_sp500,
                s.years_of_data,
                s.total_data_points,
                s.symbol_trading_days::float / NULLIF(sp.trading_days, 0) as data_coverage_ratio
            FROM stock_returns s
            JOIN sp500_returns sp ON s.year = sp.year
            WHERE s.year >= EXTRACT(YEAR FROM CURRENT_DATE) - 20
              AND s.symbol_trading_days >= sp.trading_days * 0.8  -- At least 80% trading day coverage
            ORDER BY s.symbol, s.year DESC
        """)
        
        with engine.connect() as conn:
            df = pd.read_sql_query(query, conn)
        
        if df.empty:
            logger.warning("No data for SP500 outperformance calculation")
            return pd.DataFrame()
        
        # Get sector data for sector-relative performance
        sector_query = text("""
            SELECT symbol, sector, industry, market_cap
            FROM symbol_universe
        """)
        
        with engine.connect() as conn:
            sector_df = pd.read_sql_query(sector_query, conn)
        
        # Calculate sector-relative performance
        df = df.merge(sector_df, on='symbol', how='left')
        
        # Calculate weighted scores with data quality adjustments
        current_year = datetime.now().year
        decay_factor = 0.95
        
        rankings = []
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol].sort_values('year', ascending=False)
            
            if len(symbol_df) < 3:
                continue
            
            # Data quality metrics
            avg_coverage = symbol_df['data_coverage_ratio'].mean()
            years_of_data = symbol_df['years_of_data'].iloc[0]
            
            # Calculate metrics with confidence adjustments
            years_beating = symbol_df['beat_sp500'].sum()
            total_years = len(symbol_df)
            
            # Weighted score with recency bias
            weighted_score = 0
            weight_sum = 0
            confidence_scores = []
            
            for i, row in symbol_df.iterrows():
                years_ago = current_year - row['year']
                weight = decay_factor ** years_ago
                
                # Adjust weight by data coverage
                quality_adjusted_weight = weight * row['data_coverage_ratio']
                
                if row['beat_sp500']:
                    weighted_score += row['excess_return'] * quality_adjusted_weight
                else:
                    weighted_score += row['excess_return'] * quality_adjusted_weight * 0.5
                
                weight_sum += quality_adjusted_weight
                confidence_scores.append(row['data_coverage_ratio'])
            
            weighted_score = weighted_score / weight_sum if weight_sum > 0 else 0
            
            # Calculate rolling consistency
            rolling_3yr_beats = []
            for i in range(len(symbol_df) - 2):
                window = symbol_df.iloc[i:i+3]
                rolling_3yr_beats.append(window['beat_sp500'].sum() / 3)
            
            consistency_score = np.mean(rolling_3yr_beats) * 100 if rolling_3yr_beats else 0
            
            # Recent performance metrics
            recent_5yr = symbol_df.head(5)
            recent_5yr_beat = recent_5yr['beat_sp500'].sum() if len(recent_5yr) >= 5 else 0
            recent_1yr_excess = symbol_df.iloc[0]['excess_return'] if len(symbol_df) > 0 else 0
            
            # Information Ratio (risk-adjusted excess return)
            excess_returns = symbol_df['excess_return'].values
            if len(excess_returns) > 1:
                info_ratio = np.mean(excess_returns) / (np.std(excess_returns) + 1e-6)
            else:
                info_ratio = 0
            
            # Confidence score based on data quality
            confidence = np.mean(confidence_scores) * (min(years_of_data, 20) / 20)
            
            rankings.append({
                'symbol': symbol,
                'years_beating_sp500': int(years_beating),
                'total_years': total_years,
                'sp500_weighted_score': weighted_score,
                'avg_annual_excess': symbol_df['excess_return'].mean(),
                'recent_5yr_beat_count': int(recent_5yr_beat),
                'recent_1yr_excess': recent_1yr_excess,
                'consistency_score': consistency_score,
                'information_ratio': info_ratio,
                'data_confidence': confidence * 100,
                'years_of_data': years_of_data,
                'avg_data_coverage': avg_coverage * 100,
                'sector': symbol_df['sector'].iloc[0] if 'sector' in symbol_df.columns else None
            })
        
        rank_df = pd.DataFrame(rankings)
        
        # Sector-neutral ranking
        if 'sector' in rank_df.columns and rank_df['sector'].notna().any():
            rank_df['sector_relative_score'] = rank_df.groupby('sector')['sp500_weighted_score'].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-6)
            )
        else:
            rank_df['sector_relative_score'] = 0
        
        # Composite ranking with confidence adjustment
        rank_df['confidence_adjusted_score'] = (
            rank_df['sp500_weighted_score'] * (rank_df['data_confidence'] / 100)
        )
        
        rank_df['beat_sp500_ranking'] = rank_df['confidence_adjusted_score'].rank(
            ascending=False, method='min'
        ).astype(int)
        
        return rank_df
    
    def calculate_excess_cash_flow(self):
        """Calculate excess cash flow with ML imputation"""
        print("[INFO] Calculating excess cash flow rankings (World-Class)...")
        
        query = text("""
            WITH latest_fundamentals AS (
                SELECT DISTINCT ON (symbol)
                    symbol,
                    free_cash_flow,
                    operating_cash_flow,
                    net_income,
                    revenue,
                    total_assets,
                    total_liabilities,
                    fiscal_date_ending,
                    period_type
                FROM fundamentals
                WHERE period_type = 'annual'
                  AND fiscal_date_ending >= CURRENT_DATE - INTERVAL '1 year'
                ORDER BY symbol, fiscal_date_ending DESC
            ),
            historical_fcf AS (
                SELECT 
                    symbol,
                    fiscal_date_ending,
                    free_cash_flow,
                    operating_cash_flow,
                    revenue,
                    net_income,
                    EXTRACT(YEAR FROM fiscal_date_ending) as year,
                    LAG(free_cash_flow, 4) OVER (PARTITION BY symbol ORDER BY fiscal_date_ending) as fcf_year_ago,
                    LAG(revenue, 4) OVER (PARTITION BY symbol ORDER BY fiscal_date_ending) as revenue_year_ago
                FROM fundamentals
                WHERE period_type = 'quarterly'
                  AND fiscal_date_ending >= CURRENT_DATE - INTERVAL '6 years'
            ),
            market_caps AS (
                SELECT symbol, market_cap, sector, industry
                FROM symbol_universe
                WHERE market_cap > 0
            ),
            fcf_metrics AS (
                SELECT 
                    lf.symbol,
                    lf.free_cash_flow,
                    lf.operating_cash_flow,
                    lf.net_income,
                    lf.revenue,
                    lf.total_assets,
                    lf.free_cash_flow / NULLIF(mc.market_cap, 0) * 100 as fcf_yield,
                    lf.free_cash_flow / NULLIF(lf.revenue, 0) * 100 as fcf_margin,
                    lf.operating_cash_flow / NULLIF(lf.revenue, 0) * 100 as ocf_margin,
                    lf.free_cash_flow / NULLIF(lf.net_income, 0) as fcf_to_net_income,
                    lf.free_cash_flow / NULLIF(lf.total_assets, 0) * 100 as fcf_to_assets,
                    mc.market_cap,
                    mc.sector,
                    mc.industry
                FROM latest_fundamentals lf
                LEFT JOIN market_caps mc ON lf.symbol = mc.symbol
            )
            SELECT * FROM fcf_metrics
        """)
        
        with engine.connect() as conn:
            current_df = pd.read_sql_query(query, conn)
        
        # Get historical data for advanced metrics
        hist_query = text("""
            SELECT 
                symbol,
                fiscal_date_ending,
                free_cash_flow,
                operating_cash_flow,
                revenue,
                net_income,
                total_assets
            FROM fundamentals
            WHERE period_type = 'quarterly'
              AND fiscal_date_ending >= CURRENT_DATE - INTERVAL '6 years'
            ORDER BY symbol, fiscal_date_ending DESC
        """)
        
        with engine.connect() as conn:
            hist_df = pd.read_sql_query(hist_query, conn)
        
        # Advanced imputation
        imputer = AdvancedImputer(current_df)
        
        # Handle outliers
        outlier_handler = OutlierHandler()
        metric_cols = ['fcf_yield', 'fcf_margin', 'ocf_margin', 'fcf_to_net_income', 'fcf_to_assets']
        current_df = outlier_handler.handle_outliers(current_df, metric_cols, method='winsorize')
        
        rankings = []
        for symbol in current_df['symbol'].unique():
            symbol_current = current_df[current_df['symbol'] == symbol].iloc[0]
            symbol_hist = hist_df[hist_df['symbol'] == symbol].sort_values('fiscal_date_ending')
            
            # Track data quality
            required_fields = ['free_cash_flow', 'revenue', 'operating_cash_flow', 'net_income']
            completeness = self.data_tracker.calculate_completeness(
                current_df.set_index('symbol'), symbol, required_fields
            )
            
            # FCF Growth Metrics
            fcf_growth_3yr = 0
            fcf_growth_1yr = 0
            fcf_volatility = 0
            
            if len(symbol_hist) >= 12:  # 3 years quarterly
                recent_fcf = symbol_hist.iloc[-4:]['free_cash_flow'].sum()  # Last 4 quarters
                three_yr_ago_fcf = symbol_hist.iloc[-16:-12]['free_cash_flow'].sum()  # 3 years ago
                one_yr_ago_fcf = symbol_hist.iloc[-8:-4]['free_cash_flow'].sum()  # 1 year ago
                
                if three_yr_ago_fcf > 0 and recent_fcf > 0:
                    fcf_growth_3yr = (((recent_fcf / three_yr_ago_fcf) ** (1/3)) - 1) * 100
                
                if one_yr_ago_fcf > 0 and recent_fcf > 0:
                    fcf_growth_1yr = ((recent_fcf / one_yr_ago_fcf) - 1) * 100
                
                # FCF Volatility (lower is better)
                quarterly_fcf = symbol_hist['free_cash_flow'].dropna()
                if len(quarterly_fcf) > 4:
                    fcf_volatility = quarterly_fcf.std() / (abs(quarterly_fcf.mean()) + 1e-6)
            
            # FCF Quality Score (consistency + conversion)
            fcf_quality_score = 0
            if len(symbol_hist) >= 8:
                # How many quarters had positive FCF
                positive_fcf_quarters = (symbol_hist['free_cash_flow'] > 0).sum()
                fcf_consistency = positive_fcf_quarters / len(symbol_hist) * 100
                
                # FCF conversion quality
                if symbol_hist['operating_cash_flow'].notna().any():
                    fcf_conversion = (
                        symbol_hist['free_cash_flow'] / 
                        symbol_hist['operating_cash_flow'].replace(0, np.nan)
                    ).mean()
                    fcf_conversion = min(max(fcf_conversion * 100, 0), 100) if not pd.isna(fcf_conversion) else 50
                else:
                    fcf_conversion = 50
                
                fcf_quality_score = (fcf_consistency * 0.6 + fcf_conversion * 0.4)
            
            # Sector-relative metrics
            sector = symbol_current.get('sector')
            if sector and not pd.isna(sector):
                sector_data = current_df[current_df['sector'] == sector]
                if len(sector_data) > 5:
                    sector_median_yield = sector_data['fcf_yield'].median()
                    sector_relative_yield = (
                        (symbol_current['fcf_yield'] - sector_median_yield) / 
                        (sector_data['fcf_yield'].std() + 1e-6)
                    ) if not pd.isna(symbol_current['fcf_yield']) else 0
                else:
                    sector_relative_yield = 0
            else:
                sector_relative_yield = 0
            
            # Impute missing values with hierarchy
            fcf_yield = symbol_current['fcf_yield']
            if pd.isna(fcf_yield):
                fcf_yield = imputer.impute_by_hierarchy(symbol, 'fcf_yield')
                self.data_tracker.log_imputation(
                    symbol, 'fcf_yield', 'hierarchical', None, fcf_yield
                )
            
            rankings.append({
                'symbol': symbol,
                'fcf_yield': fcf_yield if not pd.isna(fcf_yield) else 0,
                'fcf_margin': symbol_current['fcf_margin'] if pd.notna(symbol_current['fcf_margin']) else 0,
                'ocf_margin': symbol_current['ocf_margin'] if pd.notna(symbol_current['ocf_margin']) else 0,
                'fcf_growth_3yr': fcf_growth_3yr,
                'fcf_growth_1yr': fcf_growth_1yr,
                'fcf_volatility': fcf_volatility,
                'fcf_quality_score': fcf_quality_score,
                'fcf_to_net_income': symbol_current['fcf_to_net_income'] if pd.notna(symbol_current['fcf_to_net_income']) else 0,
                'fcf_to_assets': symbol_current['fcf_to_assets'] if pd.notna(symbol_current['fcf_to_assets']) else 0,
                'sector_relative_yield': sector_relative_yield,
                'data_completeness': completeness,
                'sector': sector
            })
        
        rank_df = pd.DataFrame(rankings)
        
        # ML-based imputation for critical missing values
        if len(rank_df) > 100:
            feature_cols = ['fcf_margin', 'ocf_margin', 'fcf_to_assets']
            if 'fcf_yield' in rank_df.columns and rank_df['fcf_yield'].isna().any():
                rank_df['fcf_yield'] = imputer.ml_based_imputation(
                    rank_df, 'fcf_yield', feature_cols
                )
        
        # Normalize with robust scaling
        for col in ['fcf_yield', 'fcf_margin', 'fcf_growth_3yr', 'fcf_quality_score']:
            if col in rank_df.columns and len(rank_df) > 0:
                valid_data = rank_df[col].dropna()
                if len(valid_data) > 0:
                    # Use robust scaling (less sensitive to outliers)
                    median_val = valid_data.median()
                    mad = np.median(np.abs(valid_data - median_val))
                    
                    if mad > 0:
                        rank_df[f'{col}_normalized'] = (rank_df[col] - median_val) / (mad * 1.4826)
                        rank_df[f'{col}_normalized'] = rank_df[f'{col}_normalized'].clip(-3, 3)
                        rank_df[f'{col}_normalized'] = (rank_df[f'{col}_normalized'] + 3) / 6 * 100
                    else:
                        rank_df[f'{col}_normalized'] = 50
        
        # Composite score with data quality weighting
        rank_df['excess_cash_flow_score'] = (
            rank_df.get('fcf_yield_normalized', 0) * 0.30 +
            rank_df.get('fcf_margin_normalized', 0) * 0.25 +
            rank_df.get('fcf_growth_3yr_normalized', 0) * 0.20 +
            rank_df.get('fcf_quality_score', 0) * 0.25
        ) * (rank_df['data_completeness'] / 100) ** 0.5  # Square root for less aggressive penalty
        
        rank_df['excess_cash_flow_ranking'] = rank_df['excess_cash_flow_score'].rank(
            ascending=False, method='min'
        ).astype(int)
        
        return rank_df
    
    def calculate_fundamentals_trend(self):
        """Calculate fundamentals trend with advanced analytics"""
        print("[INFO] Calculating fundamentals trend rankings (World-Class)...")
        
        # Price momentum with volatility adjustment
        price_query = text("""
            WITH price_data AS (
                SELECT 
                    symbol,
                    trade_date,
                    adjusted_close,
                    volume
                FROM stock_prices
                WHERE trade_date >= CURRENT_DATE - INTERVAL '11 years'
                  AND adjusted_close > 0
            ),
            price_metrics AS (
                SELECT 
                    symbol,
                    -- Returns
                    (LAST_VALUE(adjusted_close) FILTER (WHERE trade_date >= CURRENT_DATE - INTERVAL '30 days') 
                        OVER (PARTITION BY symbol) /
                     NULLIF(FIRST_VALUE(adjusted_close) FILTER (WHERE trade_date <= CURRENT_DATE - INTERVAL '10 years' + INTERVAL '30 days')
                        OVER (PARTITION BY symbol), 0) - 1) * 100 as price_change_10yr,
                    (LAST_VALUE(adjusted_close) FILTER (WHERE trade_date >= CURRENT_DATE - INTERVAL '30 days')
                        OVER (PARTITION BY symbol) /
                     NULLIF(FIRST_VALUE(adjusted_close) FILTER (WHERE trade_date <= CURRENT_DATE - INTERVAL '5 years' + INTERVAL '30 days')
                        OVER (PARTITION BY symbol), 0) - 1) * 100 as price_change_5yr,
                    (LAST_VALUE(adjusted_close) FILTER (WHERE trade_date >= CURRENT_DATE - INTERVAL '7 days')
                        OVER (PARTITION BY symbol) /
                     NULLIF(FIRST_VALUE(adjusted_close) FILTER (WHERE trade_date <= CURRENT_DATE - INTERVAL '1 year' + INTERVAL '7 days')
                        OVER (PARTITION BY symbol), 0) - 1) * 100 as price_change_1yr,
                    -- Volatility
                    STDDEV(adjusted_close) OVER (PARTITION BY symbol) as price_volatility,
                    -- Volume trend
                    AVG(volume) FILTER (WHERE trade_date >= CURRENT_DATE - INTERVAL '30 days')
                        OVER (PARTITION BY symbol) /
                    NULLIF(AVG(volume) FILTER (WHERE trade_date <= CURRENT_DATE - INTERVAL '1 year')
                        OVER (PARTITION BY symbol), 0) as volume_trend
                FROM price_data
            )
            SELECT DISTINCT 
                symbol,
                price_change_10yr,
                price_change_5yr,
                price_change_1yr,
                price_volatility,
                volume_trend
            FROM price_metrics
            WHERE price_change_10yr IS NOT NULL OR price_change_5yr IS NOT NULL
        """)
        
        # Enhanced fundamental trends
        fundamental_query = text("""
            WITH fundamental_history AS (
                SELECT 
                    symbol,
                    fiscal_date_ending,
                    revenue,
                    net_income,
                    gross_profit,
                    operating_income,
                    total_assets,
                    total_liabilities,
                    net_margin,
                    gross_margin,
                    operating_margin,
                    return_on_equity as roe,
                    return_on_assets as roa,
                    debt_to_equity,
                    current_ratio,
                    operating_cash_flow,
                    free_cash_flow,
                    period_type
                FROM fundamentals
                WHERE fiscal_date_ending >= CURRENT_DATE - INTERVAL '11 years'
            ),
            quarterly_metrics AS (
                SELECT 
                    symbol,
                    fiscal_date_ending,
                    revenue,
                    net_income,
                    operating_income,
                    gross_margin,
                    operating_margin,
                    net_margin,
                    roe,
                    roa,
                    -- Year-over-year growth
                    (revenue / NULLIF(LAG(revenue, 4) OVER (PARTITION BY symbol ORDER BY fiscal_date_ending), 0) - 1) * 100 as revenue_yoy,
                    (net_income / NULLIF(LAG(net_income, 4) OVER (PARTITION BY symbol ORDER BY fiscal_date_ending), 0) - 1) * 100 as earnings_yoy,
                    -- Sequential growth
                    (revenue / NULLIF(LAG(revenue, 1) OVER (PARTITION BY symbol ORDER BY fiscal_date_ending), 0) - 1) * 100 as revenue_qoq
                FROM fundamental_history
                WHERE period_type = 'quarterly'
            ),
            fundamental_trends AS (
                SELECT 
                    symbol,
                    -- Current metrics
                    LAST_VALUE(net_margin) OVER (PARTITION BY symbol ORDER BY fiscal_date_ending
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as current_net_margin,
                    LAST_VALUE(gross_margin) OVER (PARTITION BY symbol ORDER BY fiscal_date_ending
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as current_gross_margin,
                    LAST_VALUE(operating_margin) OVER (PARTITION BY symbol ORDER BY fiscal_date_ending
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as current_operating_margin,
                    LAST_VALUE(roe) OVER (PARTITION BY symbol ORDER BY fiscal_date_ending
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as roe_current,
                    LAST_VALUE(roa) OVER (PARTITION BY symbol ORDER BY fiscal_date_ending
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as roa_current,
                    -- Historical comparisons
                    FIRST_VALUE(net_margin) FILTER (WHERE fiscal_date_ending <= CURRENT_DATE - INTERVAL '5 years')
                        OVER (PARTITION BY symbol ORDER BY fiscal_date_ending DESC) as net_margin_5yr_ago,
                    FIRST_VALUE(roe) FILTER (WHERE fiscal_date_ending <= CURRENT_DATE - INTERVAL '5 years')
                        OVER (PARTITION BY symbol ORDER BY fiscal_date_ending DESC) as roe_5yr_ago,
                    -- Revenue metrics
                    LAST_VALUE(revenue) OVER (PARTITION BY symbol ORDER BY fiscal_date_ending
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as current_revenue,
                    FIRST_VALUE(revenue) FILTER (WHERE fiscal_date_ending <= CURRENT_DATE - INTERVAL '10 years')
                        OVER (PARTITION BY symbol ORDER BY fiscal_date_ending DESC) as revenue_10yr_ago,
                    FIRST_VALUE(revenue) FILTER (WHERE fiscal_date_ending <= CURRENT_DATE - INTERVAL '5 years')
                        OVER (PARTITION BY symbol ORDER BY fiscal_date_ending DESC) as revenue_5yr_ago,
                    FIRST_VALUE(revenue) FILTER (WHERE fiscal_date_ending <= CURRENT_DATE - INTERVAL '1 year')
                        OVER (PARTITION BY symbol ORDER BY fiscal_date_ending DESC) as revenue_1yr_ago,
                    -- Growth consistency
                    AVG(revenue_yoy) OVER (PARTITION BY symbol) as avg_revenue_growth,
                    STDDEV(revenue_yoy) OVER (PARTITION BY symbol) as revenue_growth_volatility,
                    AVG(earnings_yoy) OVER (PARTITION BY symbol) as avg_earnings_growth,
                    -- Profitability trend
                    REGR_SLOPE(net_margin, EXTRACT(EPOCH FROM fiscal_date_ending)) 
                        OVER (PARTITION BY symbol) as margin_trend_slope
                FROM quarterly_metrics
            )
            SELECT DISTINCT
                symbol,
                current_net_margin,
                current_gross_margin,
                current_operating_margin,
                roe_current,
                roa_current,
                current_net_margin - net_margin_5yr_ago as margin_change_5yr,
                CASE 
                    WHEN revenue_10yr_ago > 0 THEN 
                        (POW(current_revenue / revenue_10yr_ago, 1.0/10) - 1) * 100
                    ELSE NULL
                END as revenue_growth_10yr,
                CASE 
                    WHEN revenue_5yr_ago > 0 THEN 
                        (POW(current_revenue / revenue_5yr_ago, 1.0/5) - 1) * 100
                    ELSE NULL
                END as revenue_growth_5yr,
                CASE 
                    WHEN revenue_1yr_ago > 0 THEN 
                        ((current_revenue / revenue_1yr_ago) - 1) * 100
                    ELSE NULL
                END as revenue_growth_1yr,
                avg_revenue_growth,
                revenue_growth_volatility,
                avg_earnings_growth,
                margin_trend_slope,
                CASE 
                    WHEN margin_trend_slope > 0.001 THEN 'Expanding'
                    WHEN margin_trend_slope < -0.001 THEN 'Contracting'
                    ELSE 'Stable'
                END as margin_trend,
                CASE
                    WHEN roe_current > roe_5yr_ago * 1.1 THEN 'Improving'
                    WHEN roe_current < roe_5yr_ago * 0.9 THEN 'Declining'
                    ELSE 'Stable'
                END as roe_trend
            FROM fundamental_trends
            WHERE current_revenue IS NOT NULL
        """)
        
        with engine.connect() as conn:
            price_df = pd.read_sql_query(price_query, conn)
            fundamental_df = pd.read_sql_query(fundamental_query, conn)
        
        # Get sector data
        sector_query = text("SELECT symbol, sector, industry FROM symbol_universe")
        with engine.connect() as conn:
            sector_df = pd.read_sql_query(sector_query, conn)
        
        # Merge all data
        df = price_df.merge(fundamental_df, on='symbol', how='inner')
        df = df.merge(sector_df, on='symbol', how='left')
        
        # Handle missing values with sophisticated imputation
        imputer = AdvancedImputer(df)
        
        # Price changes - use forward fill then median
        for col in ['price_change_10yr', 'price_change_5yr', 'price_change_1yr']:
            if col in df.columns:
                df[col] = df.groupby('sector')[col].transform(
                    lambda x: x.fillna(x.median())
                )
                df[col] = df[col].fillna(0)
        
        # Calculate Sharpe-like ratio for price momentum
        df['price_sharpe'] = df['price_change_1yr'] / (df['price_volatility'] + 1e-6)
        
        # Revenue growth quality score
        df['revenue_growth_quality'] = 0
        if 'avg_revenue_growth' in df.columns and 'revenue_growth_volatility' in df.columns:
            df['revenue_growth_quality'] = (
                df['avg_revenue_growth'] / (df['revenue_growth_volatility'] + 1e-6)
            ).fillna(0)
        
        # Profitability improvement score
        df['profitability_score'] = 0
        for margin_col in ['current_net_margin', 'current_gross_margin', 'current_operating_margin']:
            if margin_col in df.columns:
                df[margin_col] = df.groupby('sector')[margin_col].transform(
                    lambda x: x.fillna(x.median())
                )
                df[margin_col] = df[margin_col].fillna(df[margin_col].median())
        
        # Calculate composite profitability
        if all(col in df.columns for col in ['current_net_margin', 'current_gross_margin', 'roe_current']):
            df['profitability_score'] = (
                df['current_net_margin'] * 0.4 +
                df['current_gross_margin'] * 0.3 +
                df['roe_current'] * 0.3
            )
        
        # Advanced momentum calculation
        df['price_momentum_score'] = (
            df['price_change_10yr'] * 0.15 +
            df['price_change_5yr'] * 0.25 +
            df['price_change_1yr'] * 0.35 +
            df['price_sharpe'] * 10 * 0.25  # Scaled Sharpe ratio
        )
        
        # Create fundamentals score with quality adjustments
        df['fundamentals_score'] = 0
        
        # Price component (30%)
        if 'price_momentum_score' in df.columns:
            price_percentile = df['price_momentum_score'].rank(pct=True)
            df['fundamentals_score'] += price_percentile * 30
        
        # Revenue growth component (25%)
        if 'revenue_growth_5yr' in df.columns:
            revenue_percentile = df['revenue_growth_5yr'].rank(pct=True)
            df['fundamentals_score'] += revenue_percentile * 25
        
        # Growth quality component (20%)
        if 'revenue_growth_quality' in df.columns:
            quality_percentile = df['revenue_growth_quality'].rank(pct=True)
            df['fundamentals_score'] += quality_percentile * 20
        
        # Profitability component (25%)
        if 'profitability_score' in df.columns:
            profit_percentile = df['profitability_score'].rank(pct=True)
            df['fundamentals_score'] += profit_percentile * 25
        
        # Data quality adjustment
        data_quality_cols = ['price_change_1yr', 'revenue_growth_5yr', 'current_net_margin', 'roe_current']
        df['fundamental_data_quality'] = df[data_quality_cols].notna().sum() / len(data_quality_cols) * 100
        
        # Apply confidence adjustment
        df['fundamentals_score'] = df['fundamentals_score'] * (df['fundamental_data_quality'] / 100) ** 0.3
        
        # Final ranking
        df['fundamentals_ranking'] = df['fundamentals_score'].rank(ascending=False, method='min').astype(int)
        
        # Add trend indicators
        df['revenue_acceleration'] = 'Unknown'
        mask = df['revenue_growth_1yr'].notna() & df['revenue_growth_5yr'].notna()
        df.loc[mask & (df['revenue_growth_1yr'] > df['revenue_growth_5yr'] * 1.2), 'revenue_acceleration'] = 'Accelerating'
        df.loc[mask & (df['revenue_growth_1yr'] < df['revenue_growth_5yr'] * 0.8), 'revenue_acceleration'] = 'Decelerating'
        df.loc[mask & (~df['revenue_acceleration'].isin(['Accelerating', 'Decelerating'])), 'revenue_acceleration'] = 'Stable'
        
        # Rename for consistency
        df['price_trend_10yr'] = df['price_change_10yr']
        df['price_trend_5yr'] = df['price_change_5yr']
        df['price_trend_1yr'] = df['price_change_1yr']
        df['margin_trend_5yr'] = df['margin_trend']
        df['roe_trend_5yr'] = df['roe_trend']
        df['revenue_trend'] = df['revenue_acceleration']
        
        return df
    
    def calculate_news_sentiment_score(self):
        """Calculate news sentiment scores with sophisticated analysis"""
        print("[INFO] Calculating news sentiment scores...")
        
        query = text("""
            WITH sentiment_metrics AS (
                SELECT 
                    symbol,
                    -- Recent sentiment (last 7 days)
                    AVG(CASE WHEN date > CURRENT_DATE - INTERVAL '7 days' 
                        THEN weighted_sentiment_score END) as sentiment_7d,
                    -- Medium-term sentiment (last 30 days)
                    AVG(weighted_sentiment_score) as sentiment_30d,
                    -- Sentiment momentum (7d vs 30d)
                    AVG(CASE WHEN date > CURRENT_DATE - INTERVAL '7 days' 
                        THEN weighted_sentiment_score END) - 
                    AVG(CASE WHEN date BETWEEN CURRENT_DATE - INTERVAL '30 days' 
                        AND CURRENT_DATE - INTERVAL '7 days'
                        THEN weighted_sentiment_score END) as sentiment_momentum,
                    -- Sentiment volatility
                    STDDEV(weighted_sentiment_score) as sentiment_volatility,
                    -- Article volume (information flow)
                    SUM(total_articles) as total_articles_30d,
                    SUM(CASE WHEN date > CURRENT_DATE - INTERVAL '7 days' 
                        THEN total_articles END) as total_articles_7d,
                    -- Sentiment distribution
                    SUM(bullish_count) as bullish_total,
                    SUM(bearish_count) as bearish_total,
                    SUM(neutral_count) as neutral_total,
                    -- Average relevance (quality of coverage)
                    AVG(avg_relevance_score) as avg_relevance,
                    -- Consistency score
                    COUNT(DISTINCT date) as days_with_coverage,
                    -- Bull/Bear ratio
                    CASE WHEN SUM(bearish_count) > 0 
                        THEN SUM(bullish_count)::NUMERIC / SUM(bearish_count)
                        ELSE SUM(bullish_count) END as bull_bear_ratio
                FROM daily_sentiment_summary
                WHERE date > CURRENT_DATE - INTERVAL '30 days'
                GROUP BY symbol
            ),
            topic_analysis AS (
                SELECT 
                    ns.symbol,
                    -- Count positive topics
                    COUNT(DISTINCT CASE 
                        WHEN nt.topic IN ('earnings', 'mergers_and_acquisitions', 'ipo') 
                        AND ns.sentiment_score > 0.15 THEN na.article_id 
                    END) as positive_catalyst_count,
                    -- Count negative topics
                    COUNT(DISTINCT CASE 
                        WHEN nt.topic IN ('finance', 'economy_fiscal', 'economy_monetary') 
                        AND ns.sentiment_score < -0.15 THEN na.article_id 
                    END) as negative_macro_count
                FROM news_sentiment_by_symbol ns
                JOIN news_articles na ON ns.article_id = na.article_id
                JOIN news_topics nt ON na.article_id = nt.article_id
                WHERE na.published_at > CURRENT_DATE - INTERVAL '30 days'
                GROUP BY ns.symbol
            )
            SELECT 
                sm.*,
                ta.positive_catalyst_count,
                ta.negative_macro_count,
                -- Calculate information ratio (signal strength)
                CASE WHEN sm.sentiment_volatility > 0 
                    THEN ABS(sm.sentiment_30d) / sm.sentiment_volatility 
                    ELSE 0 END as sentiment_info_ratio
            FROM sentiment_metrics sm
            LEFT JOIN topic_analysis ta ON sm.symbol = ta.symbol
        """)
        
        with engine.connect() as conn:
            sentiment_df = pd.read_sql_query(query, conn)
        
        if sentiment_df.empty:
            print("[WARNING] No sentiment data available")
            return pd.DataFrame({'symbol': [], 'sentiment_ranking': []})
        
        # Calculate composite sentiment score
        rankings = []
        for _, row in sentiment_df.iterrows():
            symbol = row['symbol']
            
            # Base sentiment score (weighted by recency)
            base_score = (
                (row['sentiment_7d'] if pd.notna(row['sentiment_7d']) else 0) * 0.6 +
                (row['sentiment_30d'] if pd.notna(row['sentiment_30d']) else 0) * 0.4
            )
            
            # Momentum bonus/penalty
            momentum_adj = row['sentiment_momentum'] if pd.notna(row['sentiment_momentum']) else 0
            momentum_adj = np.clip(momentum_adj * 20, -20, 20)  # Cap at +/- 20 points
            
            # Volume and coverage adjustment
            coverage_score = min(row['days_with_coverage'] / 20, 1.0) * 100  # Max at 20 days
            volume_score = min(row['total_articles_30d'] / 50, 1.0) * 100  # Max at 50 articles
            
            # Bull/bear ratio adjustment
            bull_bear_adj = 0
            if pd.notna(row['bull_bear_ratio']):
                if row['bull_bear_ratio'] > 2:
                    bull_bear_adj = 10
                elif row['bull_bear_ratio'] > 1.5:
                    bull_bear_adj = 5
                elif row['bull_bear_ratio'] < 0.5:
                    bull_bear_adj = -10
                elif row['bull_bear_ratio'] < 0.67:
                    bull_bear_adj = -5
            
            # Topic catalyst adjustment
            catalyst_score = (
                (row['positive_catalyst_count'] if pd.notna(row['positive_catalyst_count']) else 0) * 5 -
                (row['negative_macro_count'] if pd.notna(row['negative_macro_count']) else 0) * 3
            )
            
            # Information quality (high signal-to-noise)
            info_quality = min(row['sentiment_info_ratio'] if pd.notna(row['sentiment_info_ratio']) else 0, 2) * 10
            
            # Composite sentiment score
            composite_score = (
                base_score * 100 +  # Convert to percentage
                momentum_adj +
                coverage_score * 0.2 +
                volume_score * 0.1 +
                bull_bear_adj +
                catalyst_score +
                info_quality
            )
            
            # Data confidence for sentiment
            confidence_score = min(
                coverage_score * 0.5 + 
                volume_score * 0.3 +
                (100 if row['days_with_coverage'] > 10 else 50) * 0.2,
                100
            )
            
            # Convert confidence score to tier
            if confidence_score >= 80:
                confidence_tier = 'Very High'
            elif confidence_score >= 60:
                confidence_tier = 'High'
            elif confidence_score >= 40:
                confidence_tier = 'Medium'
            elif confidence_score >= 20:
                confidence_tier = 'Low'
            else:
                confidence_tier = 'Very Low'
            
            # Create positive catalysts JSON string
            positive_catalysts = []
            if row['positive_catalyst_count'] > 0:
                positive_catalysts.append('earnings')
                if row['positive_catalyst_count'] > 1:
                    positive_catalysts.append('mergers_and_acquisitions')
                if row['positive_catalyst_count'] > 2:
                    positive_catalysts.append('ipo')
            
            rankings.append({
                'symbol': symbol,
                'sentiment_score': composite_score,
                'sentiment_7d': row['sentiment_7d'] * 100 if pd.notna(row['sentiment_7d']) else None,
                'sentiment_30d': row['sentiment_30d'] * 100 if pd.notna(row['sentiment_30d']) else None,
                'sentiment_momentum': momentum_adj,
                'bull_bear_ratio': row['bull_bear_ratio'],
                'article_count': row['total_articles_30d'],
                'sentiment_volatility': row['sentiment_volatility'],
                'sentiment_confidence': confidence_tier,
                'positive_catalysts': json.dumps(positive_catalysts) if positive_catalysts else None,
                'days_with_coverage': row['days_with_coverage'],
                'has_catalyst_events': row['positive_catalyst_count'] > 0 if pd.notna(row['positive_catalyst_count']) else False
            })
        
        sentiment_rank_df = pd.DataFrame(rankings)
        
        # Normalize scores and create rankings
        if not sentiment_rank_df.empty:
            # Robust scaling
            scaler = RobustScaler()
            sentiment_rank_df['sentiment_score_normalized'] = scaler.fit_transform(
                sentiment_rank_df[['sentiment_score']]
            ) * 100 + 50  # Scale to 0-100 with 50 as median
            
            # Create rankings
            sentiment_rank_df['sentiment_ranking'] = sentiment_rank_df['sentiment_score_normalized'].rank(
                ascending=False, method='min'
            ).astype(int)
        
        return sentiment_rank_df
    
    def calculate_value_ranking(self):
        """Calculate value rankings based on historical valuation extremes"""
        print("[INFO] Calculating value rankings (historical extremes)...")
        
        # TODO: Implement full value ranking once market cap is calculated
        # For now, return empty DataFrame if we don't have the required data
        # Market cap needs to be calculated as: shares_outstanding * current_price
        # This requires joining fundamentals with stock_prices tables
        
        print("[WARNING] Value ranking not yet implemented - needs market cap calculation")
        return pd.DataFrame({
            'symbol': [],
            'value_ranking': [],
            'value_score': [],
            'value_confidence': []
        })
        
        # The full implementation is below (commented out until market cap is available)
        '''
        query = text("""
            WITH current_metrics AS (
                SELECT 
                    f.symbol,
                    f.market_cap,
                    f.revenue_ttm,
                    f.operating_cash_flow_ttm,
                    f.free_cash_flow_ttm,
                    f.net_income_ttm,
                    f.book_value,
                    f.ebitda_ttm,
                    f.enterprise_value,
                    f.shares_outstanding,
                    p.current_price,
                    d.dividend * 4 as annual_dividend  -- Quarterly * 4
                FROM fundamentals f
                JOIN (
                    SELECT symbol, close as current_price 
                    FROM stock_prices 
                    WHERE trade_date = (SELECT MAX(trade_date) FROM stock_prices)
                ) p ON f.symbol = p.symbol
                LEFT JOIN (
                    SELECT symbol, dividend 
                    FROM dividend_history 
                    WHERE ex_date = (
                        SELECT MAX(ex_date) 
                        FROM dividend_history dh2 
                        WHERE dh2.symbol = dividend_history.symbol
                    )
                ) d ON f.symbol = d.symbol
                WHERE f.fiscal_date_ending = (
                    SELECT MAX(fiscal_date_ending) 
                    FROM fundamentals f2 
                    WHERE f2.symbol = f.symbol
                )
            ),
            historical_valuations AS (
                SELECT 
                    f.symbol,
                    f.fiscal_date_ending as fiscal_date,
                    -- Calculate historical multiples
                    CASE WHEN f.revenue_ttm > 0 
                        THEN f.market_cap / f.revenue_ttm ELSE NULL END as ps_ratio,
                    CASE WHEN f.book_value > 0 
                        THEN f.market_cap / f.book_value ELSE NULL END as pb_ratio,
                    CASE WHEN f.operating_cash_flow_ttm > 0 
                        THEN f.market_cap / f.operating_cash_flow_ttm ELSE NULL END as pcf_ratio,
                    CASE WHEN f.net_income_ttm > 0 
                        THEN f.market_cap / f.net_income_ttm ELSE NULL END as pe_ratio,
                    CASE WHEN f.ebitda_ttm > 0 
                        THEN f.enterprise_value / f.ebitda_ttm ELSE NULL END as ev_ebitda,
                    CASE WHEN f.free_cash_flow_ttm > 0 
                        THEN f.free_cash_flow_ttm / f.market_cap ELSE NULL END as fcf_yield,
                    CASE WHEN f.net_income_ttm > 0 
                        THEN f.net_income_ttm / f.market_cap ELSE NULL END as earnings_yield
                FROM fundamentals f
                WHERE f.fiscal_date_ending > CURRENT_DATE - INTERVAL '10 years'
                    AND f.market_cap > 0
            ),
            historical_stats AS (
                SELECT 
                    symbol,
                    -- P/S statistics
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ps_ratio) as ps_median,
                    AVG(ps_ratio) as ps_mean,
                    STDDEV(ps_ratio) as ps_std,
                    MIN(ps_ratio) as ps_min,
                    MAX(ps_ratio) as ps_max,
                    COUNT(ps_ratio) as ps_count,
                    
                    -- P/B statistics
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY pb_ratio) as pb_median,
                    AVG(pb_ratio) as pb_mean,
                    STDDEV(pb_ratio) as pb_std,
                    MIN(pb_ratio) as pb_min,
                    MAX(pb_ratio) as pb_max,
                    COUNT(pb_ratio) as pb_count,
                    
                    -- P/CF statistics
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY pcf_ratio) as pcf_median,
                    AVG(pcf_ratio) as pcf_mean,
                    STDDEV(pcf_ratio) as pcf_std,
                    MIN(pcf_ratio) as pcf_min,
                    MAX(pcf_ratio) as pcf_max,
                    COUNT(pcf_ratio) as pcf_count,
                    
                    -- P/E statistics
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY pe_ratio) as pe_median,
                    AVG(pe_ratio) as pe_mean,
                    STDDEV(pe_ratio) as pe_std,
                    MIN(pe_ratio) as pe_min,
                    MAX(pe_ratio) as pe_max,
                    COUNT(pe_ratio) as pe_count,
                    
                    -- EV/EBITDA statistics
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ev_ebitda) as ev_median,
                    AVG(ev_ebitda) as ev_mean,
                    STDDEV(ev_ebitda) as ev_std,
                    MIN(ev_ebitda) as ev_min,
                    MAX(ev_ebitda) as ev_max,
                    COUNT(ev_ebitda) as ev_count,
                    
                    -- Yield statistics (higher is better)
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY fcf_yield) as fcf_yield_median,
                    AVG(fcf_yield) as fcf_yield_mean,
                    MAX(fcf_yield) as fcf_yield_max,
                    
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY earnings_yield) as ey_median,
                    AVG(earnings_yield) as ey_mean,
                    MAX(earnings_yield) as ey_max,
                    
                    COUNT(DISTINCT fiscal_date) as years_of_data
                FROM historical_valuations
                GROUP BY symbol
                HAVING COUNT(DISTINCT fiscal_date) >= 3  -- Need at least 3 years
            ),
            sector_valuations AS (
                SELECT 
                    s.sector,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY cm.market_cap / NULLIF(cm.revenue_ttm, 0)) as sector_ps_median,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY cm.market_cap / NULLIF(cm.book_value, 0)) as sector_pb_median,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY cm.market_cap / NULLIF(cm.operating_cash_flow_ttm, 0)) as sector_pcf_median,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY cm.market_cap / NULLIF(cm.net_income_ttm, 0)) as sector_pe_median
                FROM current_metrics cm
                JOIN symbol_universe s ON cm.symbol = s.symbol
                WHERE s.sector IS NOT NULL
                GROUP BY s.sector
            )
            SELECT 
                cm.symbol,
                su.sector,
                hs.years_of_data,
                
                -- Current valuations
                CASE WHEN cm.revenue_ttm > 0 
                    THEN cm.market_cap / cm.revenue_ttm END as price_to_sales_current,
                CASE WHEN cm.book_value > 0 
                    THEN cm.market_cap / cm.book_value END as price_to_book_current,
                CASE WHEN cm.operating_cash_flow_ttm > 0 
                    THEN cm.market_cap / cm.operating_cash_flow_ttm END as price_to_cashflow_current,
                CASE WHEN cm.net_income_ttm > 0 
                    THEN cm.market_cap / cm.net_income_ttm END as pe_ratio_current,
                CASE WHEN cm.ebitda_ttm > 0 
                    THEN cm.enterprise_value / cm.ebitda_ttm END as ev_to_ebitda_current,
                
                -- Current yields
                CASE WHEN cm.market_cap > 0 
                    THEN COALESCE(cm.annual_dividend, 0) / cm.market_cap END as dividend_yield_current,
                CASE WHEN cm.market_cap > 0 AND cm.free_cash_flow_ttm > 0
                    THEN cm.free_cash_flow_ttm / cm.market_cap END as fcf_yield_current,
                CASE WHEN cm.market_cap > 0 AND cm.net_income_ttm > 0
                    THEN cm.net_income_ttm / cm.market_cap END as earnings_yield_current,
                
                -- Historical statistics
                hs.*,
                
                -- Sector medians
                sv.sector_ps_median,
                sv.sector_pb_median,
                sv.sector_pcf_median,
                sv.sector_pe_median
                
            FROM current_metrics cm
            JOIN symbol_universe su ON cm.symbol = su.symbol
            LEFT JOIN historical_stats hs ON cm.symbol = hs.symbol
            LEFT JOIN sector_valuations sv ON su.sector = sv.sector
            WHERE cm.market_cap > 0
        """)
        
        with engine.connect() as conn:
            value_df = pd.read_sql_query(query, conn)
        
        if value_df.empty:
            print("[WARNING] No valuation data available")
            return pd.DataFrame({'symbol': [], 'value_ranking': []})
        
        # Calculate value metrics for each stock
        rankings = []
        for _, row in value_df.iterrows():
            symbol = row['symbol']
            metrics = {}
            
            # Calculate percentiles and z-scores for each valuation metric
            # P/S Ratio (lower is better)
            if pd.notna(row['price_to_sales_current']) and pd.notna(row['ps_min']) and pd.notna(row['ps_max']):
                ps_range = row['ps_max'] - row['ps_min']
                if ps_range > 0:
                    metrics['ps_percentile'] = ((row['price_to_sales_current'] - row['ps_min']) / ps_range) * 100
                    if pd.notna(row['ps_std']) and row['ps_std'] > 0:
                        metrics['ps_zscore'] = (row['price_to_sales_current'] - row['ps_mean']) / row['ps_std']
            
            # P/B Ratio (lower is better)
            if pd.notna(row['price_to_book_current']) and pd.notna(row['pb_min']) and pd.notna(row['pb_max']):
                pb_range = row['pb_max'] - row['pb_min']
                if pb_range > 0:
                    metrics['pb_percentile'] = ((row['price_to_book_current'] - row['pb_min']) / pb_range) * 100
                    if pd.notna(row['pb_std']) and row['pb_std'] > 0:
                        metrics['pb_zscore'] = (row['price_to_book_current'] - row['pb_mean']) / row['pb_std']
            
            # P/CF Ratio (lower is better)
            if pd.notna(row['price_to_cashflow_current']) and pd.notna(row['pcf_min']) and pd.notna(row['pcf_max']):
                pcf_range = row['pcf_max'] - row['pcf_min']
                if pcf_range > 0:
                    metrics['pcf_percentile'] = ((row['price_to_cashflow_current'] - row['pcf_min']) / pcf_range) * 100
                    if pd.notna(row['pcf_std']) and row['pcf_std'] > 0:
                        metrics['pcf_zscore'] = (row['price_to_cashflow_current'] - row['pcf_mean']) / row['pcf_std']
            
            # P/E Ratio (lower is better)
            if pd.notna(row['pe_ratio_current']) and pd.notna(row['pe_min']) and pd.notna(row['pe_max']):
                pe_range = row['pe_max'] - row['pe_min']
                if pe_range > 0:
                    metrics['pe_percentile'] = ((row['pe_ratio_current'] - row['pe_min']) / pe_range) * 100
                    if pd.notna(row['pe_std']) and row['pe_std'] > 0:
                        metrics['pe_zscore'] = (row['pe_ratio_current'] - row['pe_mean']) / row['pe_std']
            
            # EV/EBITDA (lower is better)
            if pd.notna(row['ev_to_ebitda_current']) and pd.notna(row['ev_min']) and pd.notna(row['ev_max']):
                ev_range = row['ev_max'] - row['ev_min']
                if ev_range > 0:
                    metrics['ev_percentile'] = ((row['ev_to_ebitda_current'] - row['ev_min']) / ev_range) * 100
                    if pd.notna(row['ev_std']) and row['ev_std'] > 0:
                        metrics['ev_zscore'] = (row['ev_to_ebitda_current'] - row['ev_mean']) / row['ev_std']
            
            # Yield percentiles (higher is better, so invert)
            if pd.notna(row['dividend_yield_current']) and row['dividend_yield_current'] > 0:
                metrics['div_yield_percentile'] = 100  # Start at 100 for non-zero yields
                
            if pd.notna(row['fcf_yield_current']) and pd.notna(row['fcf_yield_max']):
                if row['fcf_yield_max'] > 0:
                    metrics['fcf_yield_percentile'] = (row['fcf_yield_current'] / row['fcf_yield_max']) * 100
                    
            if pd.notna(row['earnings_yield_current']) and pd.notna(row['ey_max']):
                if row['ey_max'] > 0:
                    metrics['ey_percentile'] = (row['earnings_yield_current'] / row['ey_max']) * 100
            
            # Calculate sector relative value (negative = cheap vs sector)
            sector_discount_scores = []
            if pd.notna(row['price_to_sales_current']) and pd.notna(row['sector_ps_median']):
                if row['sector_ps_median'] > 0:
                    sector_discount_scores.append((row['price_to_sales_current'] / row['sector_ps_median'] - 1) * 100)
            if pd.notna(row['price_to_book_current']) and pd.notna(row['sector_pb_median']):
                if row['sector_pb_median'] > 0:
                    sector_discount_scores.append((row['price_to_book_current'] / row['sector_pb_median'] - 1) * 100)
            if pd.notna(row['price_to_cashflow_current']) and pd.notna(row['sector_pcf_median']):
                if row['sector_pcf_median'] > 0:
                    sector_discount_scores.append((row['price_to_cashflow_current'] / row['sector_pcf_median'] - 1) * 100)
            
            sector_relative_value = np.mean(sector_discount_scores) if sector_discount_scores else 0
            
            # Calculate historical discount (negative = cheap vs history)
            historical_discounts = []
            if pd.notna(row['price_to_sales_current']) and pd.notna(row['ps_median']):
                if row['ps_median'] > 0:
                    historical_discounts.append((row['price_to_sales_current'] / row['ps_median'] - 1) * 100)
            if pd.notna(row['price_to_book_current']) and pd.notna(row['pb_median']):
                if row['pb_median'] > 0:
                    historical_discounts.append((row['price_to_book_current'] / row['pb_median'] - 1) * 100)
            if pd.notna(row['price_to_cashflow_current']) and pd.notna(row['pcf_median']):
                if row['pcf_median'] > 0:
                    historical_discounts.append((row['price_to_cashflow_current'] / row['pcf_median'] - 1) * 100)
                    
            historical_discount = np.mean(historical_discounts) if historical_discounts else 0
            
            # Calculate composite value score (lower percentile = better value)
            value_components = []
            weights = []
            
            # Price multiples (lower percentile is better)
            if 'ps_percentile' in metrics:
                value_components.append(100 - metrics['ps_percentile'])  # Invert
                weights.append(1.0)
            if 'pb_percentile' in metrics:
                value_components.append(100 - metrics['pb_percentile'])  # Invert
                weights.append(0.8)
            if 'pcf_percentile' in metrics:
                value_components.append(100 - metrics['pcf_percentile'])  # Invert
                weights.append(1.2)  # Cash flow is important
            if 'pe_percentile' in metrics and metrics.get('pe_percentile', 100) < 80:  # Ignore extreme P/Es
                value_components.append(100 - metrics['pe_percentile'])  # Invert
                weights.append(0.9)
            if 'ev_percentile' in metrics:
                value_components.append(100 - metrics['ev_percentile'])  # Invert
                weights.append(1.0)
            
            # Yields (higher percentile is better)
            if 'fcf_yield_percentile' in metrics:
                value_components.append(metrics['fcf_yield_percentile'])
                weights.append(1.2)
            if 'ey_percentile' in metrics:
                value_components.append(metrics['ey_percentile'])
                weights.append(0.8)
            
            # Calculate weighted average
            if value_components:
                composite_score = np.average(value_components, weights=weights)
            else:
                composite_score = 50  # Neutral if no data
            
            # Adjust for sector and historical context
            if sector_relative_value < -20:  # Cheap vs sector
                composite_score += 10
            elif sector_relative_value > 20:  # Expensive vs sector
                composite_score -= 10
                
            if historical_discount < -30:  # Very cheap vs history
                composite_score += 15
            elif historical_discount < -15:  # Cheap vs history
                composite_score += 7
            elif historical_discount > 30:  # Very expensive vs history
                composite_score -= 15
                
            # Cap the score
            composite_score = np.clip(composite_score, 0, 100)
            
            # Determine confidence level
            data_points = len(value_components)
            years_of_history = row.get('years_of_data', 0)
            
            if data_points >= 5 and years_of_history >= 7:
                confidence = 'Very High'
            elif data_points >= 4 and years_of_history >= 5:
                confidence = 'High'
            elif data_points >= 3 and years_of_history >= 3:
                confidence = 'Medium'
            elif data_points >= 2:
                confidence = 'Low'
            else:
                confidence = 'Very Low'
            
            # Determine value flags
            cheap_metrics = sum([
                metrics.get('ps_percentile', 100) < 25,
                metrics.get('pb_percentile', 100) < 25,
                metrics.get('pcf_percentile', 100) < 25,
                metrics.get('pe_percentile', 100) < 25,
                metrics.get('ev_percentile', 100) < 25
            ])
            
            is_deep_value = cheap_metrics >= 3
            is_historically_cheap = (
                metrics.get('ps_percentile', 100) < 20 or
                metrics.get('pb_percentile', 100) < 20 or
                metrics.get('pcf_percentile', 100) < 20
            )
            
            # Check for value trap (cheap but deteriorating - would need fundamental trend data)
            is_value_trap_risk = False  # This would need integration with fundamentals trend
            
            rankings.append({
                'symbol': symbol,
                'value_score': composite_score,
                'price_to_sales_current': row['price_to_sales_current'],
                'price_to_sales_percentile': metrics.get('ps_percentile'),
                'price_to_sales_zscore': metrics.get('ps_zscore'),
                'price_to_book_current': row['price_to_book_current'],
                'price_to_book_percentile': metrics.get('pb_percentile'),
                'price_to_book_zscore': metrics.get('pb_zscore'),
                'price_to_cashflow_current': row['price_to_cashflow_current'],
                'price_to_cashflow_percentile': metrics.get('pcf_percentile'),
                'price_to_cashflow_zscore': metrics.get('pcf_zscore'),
                'pe_ratio_current': row['pe_ratio_current'],
                'pe_ratio_percentile': metrics.get('pe_percentile'),
                'pe_ratio_zscore': metrics.get('pe_zscore'),
                'ev_to_ebitda_current': row['ev_to_ebitda_current'],
                'ev_to_ebitda_percentile': metrics.get('ev_percentile'),
                'ev_to_ebitda_zscore': metrics.get('ev_zscore'),
                'dividend_yield_current': row['dividend_yield_current'],
                'dividend_yield_percentile': metrics.get('div_yield_percentile'),
                'fcf_yield_percentile': metrics.get('fcf_yield_percentile'),
                'earnings_yield_percentile': metrics.get('ey_percentile'),
                'sector_relative_value': sector_relative_value,
                'market_relative_value': 0,  # Would need market-wide calculation
                'historical_discount': historical_discount,
                'value_confidence': confidence,
                'years_of_history': row.get('years_of_data', 0),
                'is_deep_value': is_deep_value,
                'is_value_trap_risk': is_value_trap_risk,
                'is_historically_cheap': is_historically_cheap
            })
        
        value_rank_df = pd.DataFrame(rankings)
        
        # Create rankings based on value score
        if not value_rank_df.empty:
            value_rank_df['value_ranking'] = value_rank_df['value_score'].rank(
                ascending=False, method='min'
            ).astype(int)
        
        return value_rank_df
        '''
    
    def calculate_breakout_ranking(self):
        """Calculate breakout rankings based on price momentum and volume surge"""
        print("[INFO] Calculating breakout rankings (technical momentum)...")
        
        query = text("""
            WITH price_data AS (
                SELECT 
                    symbol,
                    trade_date,
                    close,
                    volume,
                    -- Calculate returns over different periods
                    close / LAG(close, 1) OVER (PARTITION BY symbol ORDER BY trade_date) - 1 as daily_return,
                    close / LAG(close, 5) OVER (PARTITION BY symbol ORDER BY trade_date) - 1 as weekly_return,
                    close / LAG(close, 21) OVER (PARTITION BY symbol ORDER BY trade_date) - 1 as monthly_return,
                    close / LAG(close, 63) OVER (PARTITION BY symbol ORDER BY trade_date) - 1 as quarterly_return,
                    -- Calculate moving averages
                    AVG(close) OVER (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as sma_20,
                    AVG(close) OVER (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) as sma_50,
                    AVG(close) OVER (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW) as sma_200,
                    -- Calculate volume metrics
                    AVG(volume) OVER (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as avg_volume_20,
                    AVG(volume) OVER (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 62 PRECEDING AND CURRENT ROW) as avg_volume_3m,
                    -- Get 52-week high/low
                    MAX(close) OVER (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 251 PRECEDING AND CURRENT ROW) as high_52w,
                    MIN(close) OVER (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 251 PRECEDING AND CURRENT ROW) as low_52w
                FROM stock_prices
                WHERE trade_date > CURRENT_DATE - INTERVAL '1 year'
            ),
            recent_metrics AS (
                SELECT 
                    symbol,
                    -- Latest price data
                    MAX(CASE WHEN trade_date = (SELECT MAX(trade_date) FROM stock_prices) THEN close END) as current_price,
                    MAX(CASE WHEN trade_date = (SELECT MAX(trade_date) FROM stock_prices) THEN volume END) as current_volume,
                    
                    -- Price changes
                    MAX(CASE WHEN trade_date = (SELECT MAX(trade_date) FROM stock_prices) THEN weekly_return END) * 100 as price_change_1w,
                    MAX(CASE WHEN trade_date = (SELECT MAX(trade_date) FROM stock_prices) THEN monthly_return END) * 100 as price_change_1m,
                    MAX(CASE WHEN trade_date = (SELECT MAX(trade_date) FROM stock_prices) THEN quarterly_return END) * 100 as price_change_3m,
                    
                    -- 52-week position
                    MAX(high_52w) as high_52w,
                    MIN(low_52w) as low_52w,
                    
                    -- Moving averages
                    MAX(CASE WHEN trade_date = (SELECT MAX(trade_date) FROM stock_prices) THEN sma_20 END) as sma_20,
                    MAX(CASE WHEN trade_date = (SELECT MAX(trade_date) FROM stock_prices) THEN sma_50 END) as sma_50,
                    MAX(CASE WHEN trade_date = (SELECT MAX(trade_date) FROM stock_prices) THEN sma_200 END) as sma_200,
                    
                    -- Volume analysis (3 months)
                    AVG(CASE WHEN trade_date > CURRENT_DATE - INTERVAL '3 months' THEN volume END) as avg_volume_3m,
                    AVG(CASE WHEN trade_date BETWEEN CURRENT_DATE - INTERVAL '6 months' AND CURRENT_DATE - INTERVAL '3 months' THEN volume END) as avg_volume_prior_3m,
                    
                    -- Volume surge days (>2x average)
                    COUNT(CASE WHEN trade_date > CURRENT_DATE - INTERVAL '3 months' 
                          AND volume > avg_volume_20 * 2 THEN 1 END) as volume_surge_days,
                    
                    -- New highs in quarter
                    COUNT(CASE WHEN trade_date > CURRENT_DATE - INTERVAL '3 months' 
                          AND close >= high_52w * 0.98 THEN 1 END) as new_highs_count,
                    
                    -- Days outperforming (simplified - would need market data)
                    COUNT(CASE WHEN trade_date > CURRENT_DATE - INTERVAL '3 months' 
                          AND daily_return > 0 THEN 1 END) as positive_days,
                    
                    -- Volatility (standard deviation of returns)
                    STDDEV(CASE WHEN trade_date > CURRENT_DATE - INTERVAL '3 months' THEN daily_return END) * SQRT(252) * 100 as volatility_3m,
                    STDDEV(CASE WHEN trade_date > CURRENT_DATE - INTERVAL '1 month' THEN daily_return END) * SQRT(252) * 100 as volatility_1m
                    
                FROM price_data
                GROUP BY symbol
            ),
            breakout_detection AS (
                SELECT 
                    rm.*,
                    
                    -- Calculate price position metrics
                    (current_price - low_52w) / NULLIF(high_52w - low_52w, 0) * 100 as price_position_52w,
                    (current_price / high_52w - 1) * 100 as price_vs_52w_high,
                    (current_price / low_52w - 1) * 100 as price_vs_52w_low,
                    
                    -- Moving average analysis
                    CASE 
                        WHEN current_price > sma_20 AND sma_20 > sma_50 AND sma_50 > sma_200 THEN 100  -- Perfect alignment
                        WHEN current_price > sma_20 AND current_price > sma_50 THEN 75  -- Above short-term MAs
                        WHEN current_price > sma_200 THEN 50  -- Above long-term MA
                        ELSE 25
                    END as ma_alignment_score,
                    
                    -- Volume metrics
                    CASE 
                        WHEN avg_volume_3m > 0 AND avg_volume_prior_3m > 0 
                        THEN (avg_volume_3m / avg_volume_prior_3m - 1) * 100 
                        ELSE 0 
                    END as volume_change_3m,
                    
                    avg_volume_3m / NULLIF(avg_volume_prior_3m, 0) as volume_ratio,
                    
                    -- Breakout type detection
                    CASE 
                        WHEN current_price >= high_52w * 0.98 THEN '52w_high'
                        WHEN price_change_3m > 30 AND volume_surge_days > 5 THEN 'momentum'
                        WHEN price_change_1m > 15 AND volatility_1m < volatility_3m * 0.8 THEN 'range'
                        WHEN current_price > sma_200 * 1.1 AND price_change_3m > 20 THEN 'base'
                        ELSE 'none'
                    END as breakout_type,
                    
                    -- Is it near 52-week high?
                    CASE WHEN current_price >= high_52w * 0.95 THEN TRUE ELSE FALSE END as is_52w_high
                    
                FROM recent_metrics rm
            ),
            technical_indicators AS (
                SELECT 
                    t.symbol,
                    t.rsi_14,
                    t.macd,
                    t.adx,
                    t.obv,
                    -- Calculate OBV trend (would need historical OBV)
                    0 as obv_trend  -- Placeholder
                FROM technical_indicators t
                WHERE t.date = (SELECT MAX(date) FROM technical_indicators WHERE symbol = t.symbol)
            )
            SELECT 
                bd.*,
                ti.rsi_14,
                ti.macd,
                ti.adx,
                ti.obv_trend,
                
                -- Calculate relative strength vs market (simplified - would need SPY data)
                bd.price_change_3m as rs_rating,  -- Placeholder - should compare to SPY
                
                -- Volatility contraction
                CASE 
                    WHEN bd.volatility_3m > 0 
                    THEN (bd.volatility_1m / bd.volatility_3m - 1) * 100 
                    ELSE 0 
                END as volatility_contraction
                
            FROM breakout_detection bd
            LEFT JOIN technical_indicators ti ON bd.symbol = ti.symbol
            WHERE bd.price_change_3m IS NOT NULL
        """)
        
        with engine.connect() as conn:
            breakout_df = pd.read_sql_query(query, conn)
        
        if breakout_df.empty:
            print("[WARNING] No breakout data available")
            return pd.DataFrame({'symbol': [], 'breakout_ranking': []})
        
        # Calculate breakout scores
        rankings = []
        for _, row in breakout_df.iterrows():
            symbol = row['symbol']
            
            # Base score from price momentum
            price_score = 0
            if pd.notna(row['price_change_3m']):
                if row['price_change_3m'] > 50:
                    price_score = 40
                elif row['price_change_3m'] > 30:
                    price_score = 35
                elif row['price_change_3m'] > 20:
                    price_score = 30
                elif row['price_change_3m'] > 10:
                    price_score = 20
                else:
                    price_score = max(0, row['price_change_3m'])
            
            # Volume confirmation score
            volume_score = 0
            if pd.notna(row['volume_change_3m']):
                if row['volume_change_3m'] > 100:  # Volume doubled
                    volume_score = 30
                elif row['volume_change_3m'] > 50:
                    volume_score = 25
                elif row['volume_change_3m'] > 25:
                    volume_score = 20
                elif row['volume_change_3m'] > 0:
                    volume_score = 15
            
            # Add points for volume surge days
            if pd.notna(row['volume_surge_days']):
                volume_score += min(row['volume_surge_days'] * 2, 20)  # Max 20 points
            
            # Technical strength score
            technical_score = 0
            
            # Moving average alignment
            technical_score += row.get('ma_alignment_score', 0) * 0.15
            
            # RSI in bullish zone (40-70)
            if pd.notna(row.get('rsi_14')):
                if 40 <= row['rsi_14'] <= 70:
                    technical_score += 10
                elif row['rsi_14'] > 70:
                    technical_score += 5  # Overbought but still bullish
            
            # MACD positive
            if pd.notna(row.get('macd')) and row['macd'] > 0:
                technical_score += 5
            
            # Strong trend (ADX > 25)
            if pd.notna(row.get('adx')) and row['adx'] > 25:
                technical_score += 5
            
            # 52-week high proximity
            if row.get('is_52w_high', False):
                technical_score += 15
            elif pd.notna(row.get('price_vs_52w_high')) and row['price_vs_52w_high'] > -5:
                technical_score += 10
            
            # Breakout quality adjustments
            breakout_bonus = 0
            if row.get('breakout_type') == '52w_high':
                breakout_bonus = 20
            elif row.get('breakout_type') == 'momentum':
                breakout_bonus = 15
            elif row.get('breakout_type') == 'range':
                breakout_bonus = 10
            elif row.get('breakout_type') == 'base':
                breakout_bonus = 5
            
            # Volatility contraction bonus (breakouts from low volatility are stronger)
            if pd.notna(row.get('volatility_contraction')) and row['volatility_contraction'] < -20:
                breakout_bonus += 10
            
            # Calculate composite score
            composite_score = (
                price_score +
                volume_score +
                technical_score +
                breakout_bonus
            )
            
            # Cap at 100
            composite_score = min(composite_score, 100)
            
            # Determine confidence
            data_quality = 0
            if pd.notna(row['price_change_3m']):
                data_quality += 1
            if pd.notna(row['volume_change_3m']):
                data_quality += 1
            if pd.notna(row['ma_alignment_score']):
                data_quality += 1
            if pd.notna(row.get('rsi_14')):
                data_quality += 1
            
            if data_quality >= 4:
                confidence = 'Very High'
            elif data_quality >= 3:
                confidence = 'High'
            elif data_quality >= 2:
                confidence = 'Medium'
            else:
                confidence = 'Low'
            
            # Determine flags
            is_valid_breakout = (
                composite_score >= 60 and
                row.get('price_change_3m', 0) > 15 and
                row.get('volume_change_3m', 0) > 0
            )
            
            is_volume_confirmed = row.get('volume_change_3m', 0) > 25
            
            rankings.append({
                'symbol': symbol,
                'breakout_score': composite_score,
                'price_change_3m': row.get('price_change_3m'),
                'price_change_1m': row.get('price_change_1m'),
                'price_change_1w': row.get('price_change_1w'),
                'breakout_type': row.get('breakout_type'),
                'current_price': row.get('current_price'),
                'price_vs_52w_high': row.get('price_vs_52w_high'),
                'price_vs_52w_low': row.get('price_vs_52w_low'),
                'new_highs_count': row.get('new_highs_count'),
                'volume_change_3m': row.get('volume_change_3m'),
                'volume_surge_days': row.get('volume_surge_days'),
                'avg_volume_3m': row.get('avg_volume_3m'),
                'volume_ratio': row.get('volume_ratio'),
                'rs_rating': row.get('rs_rating'),
                'breakout_confidence': confidence,
                'volatility_contraction': row.get('volatility_contraction'),
                'is_valid_breakout': is_valid_breakout,
                'is_volume_confirmed': is_volume_confirmed,
                'is_52w_high': row.get('is_52w_high', False)
            })
        
        breakout_rank_df = pd.DataFrame(rankings)
        
        # Create rankings based on breakout score
        if not breakout_rank_df.empty:
            breakout_rank_df['breakout_ranking'] = breakout_rank_df['breakout_score'].rank(
                ascending=False, method='min'
            ).astype(int)
        
        return breakout_rank_df
    
    def calculate_growth_ranking(self):
        """
        Calculate long-term growth rankings based on lifetime performance and consistency
        Focuses on sustained outperformance over the stock's entire history
        """
        print("[INFO] Calculating long-term growth rankings...")
        
        query = """
        WITH lifetime_performance AS (
            -- Calculate lifetime returns for each stock
            SELECT 
                sp.symbol,
                MIN(sp.date) as first_date,
                MAX(sp.date) as last_date,
                COUNT(DISTINCT EXTRACT(YEAR FROM sp.date)) as years_tracked,
                FIRST_VALUE(sp.close) OVER (PARTITION BY sp.symbol ORDER BY sp.date) as first_price,
                LAST_VALUE(sp.close) OVER (PARTITION BY sp.symbol ORDER BY sp.date ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as last_price
            FROM stock_prices sp
            GROUP BY sp.symbol, sp.close, sp.date
        ),
        annual_returns AS (
            -- Calculate annual returns and SP500 comparisons
            SELECT 
                s.symbol,
                s.year,
                s.symbol_return,
                s.sp500_return,
                s.excess_return,
                s.beat_sp500,
                -- Calculate rolling multi-year returns
                AVG(s.symbol_return) OVER (PARTITION BY s.symbol ORDER BY s.year ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) as return_10yr_avg,
                AVG(s.symbol_return) OVER (PARTITION BY s.symbol ORDER BY s.year ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) as return_5yr_avg,
                AVG(s.symbol_return) OVER (PARTITION BY s.symbol ORDER BY s.year ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) as return_3yr_avg,
                STDDEV(s.symbol_return) OVER (PARTITION BY s.symbol ORDER BY s.year ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) as volatility_10yr,
                COUNT(*) OVER (PARTITION BY s.symbol) as total_years
            FROM sp500_outperformance_detail s
            WHERE s.year >= EXTRACT(YEAR FROM CURRENT_DATE) - 15  -- Focus on last 15 years max
        ),
        fundamental_growth AS (
            -- Calculate fundamental growth rates
            SELECT 
                f.symbol,
                -- Revenue growth
                CASE 
                    WHEN COUNT(f.total_revenue) FILTER (WHERE f.fiscal_year >= EXTRACT(YEAR FROM CURRENT_DATE) - 10) >= 8
                    THEN POWER(
                        MAX(f.total_revenue) FILTER (WHERE f.fiscal_year = EXTRACT(YEAR FROM CURRENT_DATE) - 1) /
                        NULLIF(MIN(f.total_revenue) FILTER (WHERE f.fiscal_year = EXTRACT(YEAR FROM CURRENT_DATE) - 10), 0),
                        1.0/9) - 1
                END as revenue_cagr_10yr,
                -- Earnings growth
                CASE 
                    WHEN COUNT(f.net_income) FILTER (WHERE f.fiscal_year >= EXTRACT(YEAR FROM CURRENT_DATE) - 10) >= 8
                    THEN POWER(
                        MAX(f.net_income) FILTER (WHERE f.fiscal_year = EXTRACT(YEAR FROM CURRENT_DATE) - 1) /
                        NULLIF(MIN(f.net_income) FILTER (WHERE f.fiscal_year = EXTRACT(YEAR FROM CURRENT_DATE) - 10 AND f.net_income > 0), 0),
                        1.0/9) - 1
                END as earnings_cagr_10yr,
                -- FCF growth
                CASE 
                    WHEN COUNT(f.free_cash_flow) FILTER (WHERE f.fiscal_year >= EXTRACT(YEAR FROM CURRENT_DATE) - 10) >= 8
                    THEN POWER(
                        MAX(f.free_cash_flow) FILTER (WHERE f.fiscal_year = EXTRACT(YEAR FROM CURRENT_DATE) - 1) /
                        NULLIF(MIN(f.free_cash_flow) FILTER (WHERE f.fiscal_year = EXTRACT(YEAR FROM CURRENT_DATE) - 10 AND f.free_cash_flow > 0), 0),
                        1.0/9) - 1
                END as fcf_cagr_10yr,
                -- Book value growth
                CASE 
                    WHEN COUNT(f.book_value) FILTER (WHERE f.fiscal_year >= EXTRACT(YEAR FROM CURRENT_DATE) - 10) >= 8
                    THEN POWER(
                        MAX(f.book_value) FILTER (WHERE f.fiscal_year = EXTRACT(YEAR FROM CURRENT_DATE) - 1) /
                        NULLIF(MIN(f.book_value) FILTER (WHERE f.fiscal_year = EXTRACT(YEAR FROM CURRENT_DATE) - 10 AND f.book_value > 0), 0),
                        1.0/9) - 1
                END as book_value_cagr_10yr,
                COUNT(DISTINCT f.fiscal_year) as fundamental_years
            FROM fundamentals f
            GROUP BY f.symbol
        ),
        growth_metrics AS (
            SELECT 
                lp.symbol,
                lp.years_tracked as total_years_tracked,
                -- Lifetime returns
                CASE 
                    WHEN lp.first_price > 0 AND lp.last_price > 0
                    THEN ((lp.last_price / lp.first_price) - 1) * 100
                END as lifetime_return,
                CASE 
                    WHEN lp.first_price > 0 AND lp.last_price > 0 AND lp.years_tracked > 1
                    THEN (POWER(lp.last_price / lp.first_price, 1.0 / NULLIF(lp.years_tracked - 1, 0)) - 1) * 100
                END as lifetime_cagr,
                -- SP500 outperformance metrics
                COUNT(ar.year) FILTER (WHERE ar.beat_sp500 = true) as years_beating_sp500,
                COUNT(ar.year) as years_with_data,
                AVG(ar.excess_return) as avg_excess_return,
                COUNT(ar.year) FILTER (WHERE ar.beat_sp500 = true) * 100.0 / NULLIF(COUNT(ar.year), 0) as outperformance_consistency,
                -- Multi-year CAGRs
                AVG(ar.return_10yr_avg) as growth_10yr_cagr,
                AVG(ar.return_5yr_avg) as growth_5yr_cagr,
                AVG(ar.return_3yr_avg) as growth_3yr_cagr,
                -- Growth acceleration (recent vs long-term)
                AVG(ar.return_5yr_avg) - AVG(ar.return_10yr_avg) as growth_acceleration,
                -- Volatility and consistency
                AVG(ar.volatility_10yr) as avg_volatility,
                1 / (1 + COALESCE(AVG(ar.volatility_10yr), 50) / 100) * 100 as growth_stability,
                -- Sharpe ratio approximation
                CASE 
                    WHEN AVG(ar.volatility_10yr) > 0
                    THEN (AVG(ar.return_10yr_avg) - 2) / AVG(ar.volatility_10yr)  -- Assuming 2% risk-free rate
                END as sharpe_ratio_10yr,
                -- Consecutive years analysis
                MAX(consecutive_growth.consecutive_years) as consecutive_growth_years,
                MAX(consecutive_beat.consecutive_years) as consecutive_beat_years,
                -- Fundamental growth rates
                fg.revenue_cagr_10yr * 100 as revenue_growth_10yr,
                fg.earnings_cagr_10yr * 100 as earnings_growth_10yr,
                fg.fcf_cagr_10yr * 100 as fcf_growth_10yr,
                fg.book_value_cagr_10yr * 100 as book_value_growth_10yr
            FROM lifetime_performance lp
            LEFT JOIN annual_returns ar ON lp.symbol = ar.symbol
            LEFT JOIN fundamental_growth fg ON lp.symbol = fg.symbol
            LEFT JOIN LATERAL (
                -- Count consecutive positive return years
                SELECT MAX(streak) as consecutive_years
                FROM (
                    SELECT symbol, 
                           COUNT(*) as streak,
                           year - ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY year) as grp
                    FROM annual_returns
                    WHERE symbol = lp.symbol AND symbol_return > 0
                    GROUP BY symbol, grp
                ) t
            ) consecutive_growth ON true
            LEFT JOIN LATERAL (
                -- Count consecutive SP500 beating years
                SELECT MAX(streak) as consecutive_years
                FROM (
                    SELECT symbol,
                           COUNT(*) as streak,
                           year - ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY year) as grp
                    FROM annual_returns  
                    WHERE symbol = lp.symbol AND beat_sp500 = true
                    GROUP BY symbol, grp
                ) t
            ) consecutive_beat ON true
            GROUP BY 
                lp.symbol, lp.years_tracked, lp.first_price, lp.last_price,
                fg.revenue_cagr_10yr, fg.earnings_cagr_10yr, fg.fcf_cagr_10yr, fg.book_value_cagr_10yr
        )
        SELECT 
            gm.*,
            -- Calculate composite growth score
            CASE 
                WHEN gm.total_years_tracked >= 3 THEN
                    -- Lifetime performance (30%)
                    COALESCE(LEAST(gm.lifetime_cagr * 2, 100) * 0.30, 0) +
                    -- Consistency beating SP500 (25%)
                    COALESCE(gm.outperformance_consistency * 0.25, 0) +
                    -- Recent growth momentum (20%)
                    COALESCE(LEAST(gm.growth_5yr_cagr * 2, 100) * 0.20, 0) +
                    -- Growth stability (15%)
                    COALESCE(gm.growth_stability * 0.15, 0) +
                    -- Fundamental growth (10%)
                    COALESCE(LEAST(GREATEST(
                        COALESCE(gm.revenue_growth_10yr, 0),
                        COALESCE(gm.earnings_growth_10yr, 0),
                        COALESCE(gm.fcf_growth_10yr, 0)
                    ) * 2, 100) * 0.10, 0)
                ELSE 0
            END as growth_score,
            -- Determine confidence level
            CASE
                WHEN gm.total_years_tracked >= 10 AND gm.years_with_data >= 8 THEN 'Very High'
                WHEN gm.total_years_tracked >= 7 AND gm.years_with_data >= 5 THEN 'High'
                WHEN gm.total_years_tracked >= 5 AND gm.years_with_data >= 3 THEN 'Medium'
                WHEN gm.total_years_tracked >= 3 THEN 'Low'
                ELSE 'Very Low'
            END as growth_confidence,
            -- Set flags
            gm.outperformance_consistency >= 60 as is_consistent_grower,
            gm.growth_acceleration > 0 as is_accelerating_growth,
            gm.lifetime_cagr > 15 AND gm.outperformance_consistency >= 60 as is_compound_winner,
            gm.years_beating_sp500 >= gm.years_with_data * 0.7 as is_lifetime_outperformer
        FROM growth_metrics gm
        WHERE gm.total_years_tracked >= 2  -- Minimum 2 years of history
        """
        
        with engine.connect() as conn:
            result = conn.execute(text(query))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
        
        if df.empty:
            print("[WARNING] No growth data available")
            return pd.DataFrame()
        
        # Fill NaN values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Create final ranking
        df['growth_ranking'] = df['growth_score'].rank(ascending=False, method='min').astype(int)
        
        # Add additional calculated fields
        df['growth_quality_score'] = df['growth_score'] * (df['growth_stability'] / 100)
        df['growth_momentum_score'] = df['growth_5yr_cagr'] - df['growth_10yr_cagr']
        df['mean_reversion_risk'] = np.where(
            df['growth_10yr_cagr'] > 20,
            np.minimum((df['growth_10yr_cagr'] - 10) * 2, 100),
            0
        )
        
        # Calculate percentiles within sector and market
        if 'sector' in df.columns:
            df['sector_growth_percentile'] = df.groupby('sector')['growth_score'].rank(pct=True) * 100
        df['market_growth_percentile'] = df['growth_score'].rank(pct=True) * 100
        
        # Size-adjusted growth (small caps get slight boost)
        if 'market_cap' in df.columns:
            size_factor = np.where(
                df['market_cap'] < 2e9, 1.1,  # Small cap boost
                np.where(df['market_cap'] < 10e9, 1.05, 1.0)  # Mid cap slight boost
            )
            df['size_adjusted_growth'] = df['growth_score'] * size_factor
        
        print(f"[SUCCESS] Calculated growth rankings for {len(df)} stocks")
        
        # Select columns to return
        return_cols = [
            'symbol', 'growth_ranking', 'growth_score',
            'total_years_tracked', 'lifetime_return', 'lifetime_cagr',
            'years_beating_sp500', 'outperformance_consistency',
            'growth_10yr_cagr', 'growth_5yr_cagr', 'growth_3yr_cagr',
            'growth_acceleration', 'growth_stability', 'growth_quality_score',
            'revenue_growth_10yr', 'earnings_growth_10yr', 'fcf_growth_10yr',
            'book_value_growth_10yr', 'sharpe_ratio_10yr',
            'consecutive_growth_years', 'consecutive_beat_years',
            'growth_momentum_score', 'mean_reversion_risk',
            'sector_growth_percentile', 'market_growth_percentile',
            'size_adjusted_growth', 'growth_confidence',
            'is_consistent_grower', 'is_accelerating_growth',
            'is_compound_winner', 'is_lifetime_outperformer'
        ]
        
        return df[[col for col in return_cols if col in df.columns]]
    
    def calculate_composite_rankings(self):
        """Combine all rankings with confidence scoring including all six components"""
        print("[INFO] Calculating composite quality rankings (Complete Multi-Factor System)...")
        
        # Get all seven ranking components
        sp500_df = self.calculate_sp500_outperformance()
        cash_flow_df = self.calculate_excess_cash_flow()
        fundamentals_df = self.calculate_fundamentals_trend()
        sentiment_df = self.calculate_news_sentiment_score()
        value_df = self.calculate_value_ranking()
        breakout_df = self.calculate_breakout_ranking()
        growth_df = self.calculate_growth_ranking()
        
        # Get additional data
        sector_query = text("""
            SELECT 
                symbol, 
                sector, 
                industry, 
                market_cap,
                CASE 
                    WHEN market_cap >= 200000000000 THEN 'Mega'
                    WHEN market_cap >= 10000000000 THEN 'Large'
                    WHEN market_cap >= 2000000000 THEN 'Mid'
                    WHEN market_cap >= 300000000 THEN 'Small'
                    ELSE 'Micro'
                END as size_category
            FROM symbol_universe
        """)
        
        with engine.connect() as conn:
            sector_df = pd.read_sql_query(sector_query, conn)
        
        # Merge all dataframes including all seven components
        result_df = sp500_df.merge(cash_flow_df, on='symbol', how='outer', suffixes=('_sp500', '_fcf'))
        result_df = result_df.merge(fundamentals_df, on='symbol', how='outer', suffixes=('', '_fund'))
        result_df = result_df.merge(sentiment_df, on='symbol', how='left', suffixes=('', '_sent'))
        result_df = result_df.merge(value_df, on='symbol', how='left', suffixes=('', '_val'))
        result_df = result_df.merge(breakout_df, on='symbol', how='left', suffixes=('', '_brk'))
        result_df = result_df.merge(growth_df, on='symbol', how='left', suffixes=('', '_grw'))
        result_df = result_df.merge(sector_df, on='symbol', how='left')
        
        # Handle duplicate sector columns
        if 'sector_sp500' in result_df.columns:
            result_df['sector'] = result_df['sector'].fillna(result_df['sector_sp500'])
            result_df = result_df.drop(columns=['sector_sp500'])
        if 'sector_fcf' in result_df.columns:
            result_df['sector'] = result_df['sector'].fillna(result_df['sector_fcf'])
            result_df = result_df.drop(columns=['sector_fcf'])
        
        # Calculate overall data confidence
        result_df['overall_data_confidence'] = (
            result_df.get('data_confidence', 50) * 0.3 +
            result_df.get('data_completeness', 50) * 0.3 +
            result_df.get('fundamental_data_quality', 50) * 0.4
        )
        
        # Fill NaN rankings with worst rank + penalty
        max_symbols = len(result_df)
        result_df['beat_sp500_ranking'] = result_df['beat_sp500_ranking'].fillna(max_symbols * 1.5)
        result_df['excess_cash_flow_ranking'] = result_df['excess_cash_flow_ranking'].fillna(max_symbols * 1.5)
        result_df['fundamentals_ranking'] = result_df['fundamentals_ranking'].fillna(max_symbols * 1.5)
        result_df['sentiment_ranking'] = result_df['sentiment_ranking'].fillna(max_symbols * 1.5)
        result_df['value_ranking'] = result_df['value_ranking'].fillna(max_symbols * 1.5)
        result_df['breakout_ranking'] = result_df['breakout_ranking'].fillna(max_symbols * 1.5)
        result_df['growth_ranking'] = result_df['growth_ranking'].fillna(max_symbols * 1.5)
        
        # Calculate base composite score including all seven components
        # Weights: SP500: 15%, Cash Flow: 15%, Fundamentals: 15%, Growth: 20%, Sentiment: 15%, Value: 10%, Breakout: 10%
        result_df['base_composite_score'] = (
            (1 - result_df['beat_sp500_ranking'] / max_symbols) * 100 * 0.15 +
            (1 - result_df['excess_cash_flow_ranking'] / max_symbols) * 100 * 0.15 +
            (1 - result_df['fundamentals_ranking'] / max_symbols) * 100 * 0.15 +
            (1 - result_df['growth_ranking'] / max_symbols) * 100 * 0.20 +
            (1 - result_df['sentiment_ranking'] / max_symbols) * 100 * 0.15 +
            (1 - result_df['value_ranking'] / max_symbols) * 100 * 0.10 +
            (1 - result_df['breakout_ranking'] / max_symbols) * 100 * 0.10
        )
        
        # Apply confidence adjustment
        result_df['composite_quality_score'] = (
            result_df['base_composite_score'] * 
            (result_df['overall_data_confidence'] / 100) ** 0.5
        )
        
        # Calculate sector-neutral score
        result_df['sector_neutral_score'] = result_df.groupby('sector')['composite_quality_score'].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-6) * 15 + 50
        )
        
        # Size-adjusted score (small caps get slight boost for growth potential)
        size_multiplier = {
            'Micro': 1.15,
            'Small': 1.10,
            'Mid': 1.05,
            'Large': 1.00,
            'Mega': 0.95
        }
        result_df['size_adjusted_score'] = result_df.apply(
            lambda x: x['composite_quality_score'] * size_multiplier.get(x.get('size_category', 'Mid'), 1.0),
            axis=1
        )
        
        # Final composite with multiple perspectives
        result_df['final_composite_score'] = (
            result_df['composite_quality_score'] * 0.50 +
            result_df['sector_neutral_score'] * 0.30 +
            result_df['size_adjusted_score'] * 0.20
        )
        
        # Determine quality tiers with confidence adjustment
        result_df['quality_tier'] = pd.cut(
            result_df['final_composite_score'],
            bins=[-np.inf, 35, 50, 65, 80, np.inf],
            labels=['Poor', 'Below Average', 'Average', 'Above Average', 'Elite']
        )
        
        # Add confidence tier
        result_df['confidence_tier'] = pd.cut(
            result_df['overall_data_confidence'],
            bins=[0, 40, 60, 80, 100],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        # Set advanced flags
        result_df['is_sp500_beater'] = (
            (result_df['years_beating_sp500'] >= 10) & 
            (result_df.get('consistency_score', 0) > 60)
        )
        result_df['is_cash_generator'] = (
            (result_df['excess_cash_flow_ranking'] <= max_symbols * 0.20) &
            (result_df.get('fcf_quality_score', 0) > 70)
        )
        result_df['is_fundamental_grower'] = (
            (result_df['fundamentals_ranking'] <= max_symbols * 0.20) &
            (result_df.get('revenue_growth_quality', 0) > 0)
        )
        # Add sentiment criteria
        result_df['is_sentiment_positive'] = (
            (result_df['sentiment_ranking'] <= 100) &
            (result_df.get('sentiment_7d', 0) > 0)
        )
        result_df['is_momentum_star'] = (
            (result_df['sentiment_ranking'] <= 20) &
            (result_df.get('sentiment_momentum', 0) > 10) &
            ((result_df['beat_sp500_ranking'] <= 100) | 
             (result_df['excess_cash_flow_ranking'] <= 100))
        )
        
        result_df['is_all_star'] = (
            result_df['is_sp500_beater'] & 
            result_df['is_cash_generator'] & 
            result_df['is_fundamental_grower'] &
            (result_df['is_sentiment_positive'] | result_df['confidence_tier'].isin(['Very High']))
        )
        
        # Add deep value star flag
        result_df['is_deep_value_star'] = (
            (result_df['value_ranking'] <= 20) &
            result_df.get('is_deep_value', False) &
            (result_df.get('is_historically_cheap', False)) &
            (~result_df.get('is_value_trap_risk', False))
        )
        
        # Add breakout momentum flag
        result_df['is_momentum_breakout'] = (
            (result_df['breakout_ranking'] <= 20) &
            result_df.get('is_valid_breakout', False) &
            result_df.get('is_volume_confirmed', False)
        )
        
        # Add breakout star flag (top breakout with strong fundamentals)
        result_df['is_breakout_star'] = (
            result_df['is_momentum_breakout'] &
            ((result_df['fundamentals_ranking'] <= 100) | 
             (result_df['excess_cash_flow_ranking'] <= 100))
        )
        
        # Add growth champion flag (top long-term growth stocks)
        result_df['is_growth_champion'] = (
            (result_df['growth_ranking'] <= 20) &
            (result_df.get('is_lifetime_outperformer', False) | 
             result_df.get('is_compound_winner', False))
        )
        
        # Add ranking date and version
        result_df['ranking_date'] = self.ranking_date
        result_df['ranking_version'] = 'WorldClass_v4.0'
        
        # Calculate percentile ranks for easy interpretation
        for col in ['final_composite_score', 'composite_quality_score', 'sector_neutral_score']:
            if col in result_df.columns:
                result_df[f'{col}_percentile'] = result_df[col].rank(pct=True) * 100
        
        return result_df
    
    def save_rankings(self, df):
        """Save rankings with audit trail"""
        print(f"[INFO] Saving {len(df)} stock rankings to database...")
        
        # Comprehensive column list
        columns = [
            'symbol', 'ranking_date', 'ranking_version',
            # SP500 metrics
            'beat_sp500_ranking', 'years_beating_sp500', 'sp500_weighted_score',
            'avg_annual_excess', 'recent_5yr_beat_count', 'recent_1yr_excess',
            'consistency_score', 'information_ratio', 'data_confidence',
            'sector_relative_score',
            # Cash flow metrics
            'excess_cash_flow_ranking', 'fcf_yield', 'fcf_margin', 'ocf_margin',
            'fcf_growth_3yr', 'fcf_growth_1yr', 'fcf_volatility',
            'fcf_quality_score', 'fcf_to_net_income', 'fcf_to_assets',
            'sector_relative_yield', 'data_completeness',
            # Fundamentals metrics
            'fundamentals_ranking', 'price_trend_10yr', 'price_trend_5yr', 
            'price_trend_1yr', 'price_momentum_score', 'price_sharpe',
            'revenue_growth_10yr', 'revenue_growth_5yr', 'revenue_growth_1yr',
            'revenue_growth_quality', 'revenue_trend', 'margin_trend_5yr',
            'current_net_margin', 'current_gross_margin', 'current_operating_margin',
            'margin_change_5yr', 'roe_current', 'roa_current', 'roe_trend_5yr',
            'profitability_score', 'fundamental_data_quality',
            # Sentiment metrics
            'sentiment_ranking', 'sentiment_score', 'sentiment_7d', 'sentiment_30d',
            'sentiment_momentum', 'bull_bear_ratio', 'article_count',
            'sentiment_confidence', 'positive_catalysts', 'sentiment_volatility',
            'days_with_coverage', 'has_catalyst_events',
            # Value metrics
            'value_ranking', 'value_score',
            'price_to_sales_current', 'price_to_sales_percentile', 'price_to_sales_zscore',
            'price_to_book_current', 'price_to_book_percentile', 'price_to_book_zscore',
            'price_to_cashflow_current', 'price_to_cashflow_percentile', 'price_to_cashflow_zscore',
            'pe_ratio_current', 'pe_ratio_percentile', 'pe_ratio_zscore',
            'ev_to_ebitda_current', 'ev_to_ebitda_percentile', 'ev_to_ebitda_zscore',
            'dividend_yield_current', 'dividend_yield_percentile',
            'fcf_yield_percentile', 'earnings_yield_percentile',
            'sector_relative_value', 'market_relative_value', 'historical_discount',
            'value_confidence', 'years_of_history',
            'is_deep_value', 'is_value_trap_risk', 'is_historically_cheap',
            # Breakout metrics
            'breakout_ranking', 'breakout_score',
            'price_change_3m', 'price_change_1m', 'price_change_1w',
            'breakout_type', 'current_price',
            'price_vs_52w_high', 'price_vs_52w_low', 'new_highs_count',
            'volume_change_3m', 'volume_surge_days', 'avg_volume_3m', 'volume_ratio',
            'rs_rating', 'breakout_confidence', 'volatility_contraction',
            'is_valid_breakout', 'is_volume_confirmed', 'is_52w_high',
            # Growth metrics
            'growth_ranking', 'growth_score',
            'total_years_tracked', 'lifetime_return', 'lifetime_annualized_return',
            'lifetime_sp500_excess', 'years_outperforming_sp500', 'outperformance_consistency',
            'growth_10yr_cagr', 'growth_5yr_cagr', 'growth_3yr_cagr',
            'growth_acceleration', 'growth_stability', 'growth_quality_score',
            'revenue_growth_10yr', 'revenue_growth_consistency',
            'earnings_growth_10yr', 'earnings_growth_consistency',
            'fcf_growth_10yr', 'book_value_growth_10yr',
            'trend_strength_10yr', 'trend_strength_5yr',
            'drawdown_recovery_avg', 'max_drawdown_10yr',
            'sharpe_ratio_10yr', 'sortino_ratio_10yr',
            'consecutive_growth_years', 'consecutive_beat_years',
            'growth_momentum_score', 'mean_reversion_risk',
            'sector_growth_percentile', 'market_growth_percentile',
            'size_adjusted_growth', 'growth_confidence', 'growth_data_quality',
            'is_consistent_grower', 'is_accelerating_growth',
            'is_compound_winner', 'is_lifetime_outperformer',
            # Composite scores
            'base_composite_score', 'composite_quality_score', 'sector_neutral_score',
            'size_adjusted_score', 'final_composite_score', 'overall_data_confidence',
            'final_composite_score_percentile',
            # Tiers and flags
            'quality_tier', 'confidence_tier',
            'is_sp500_beater', 'is_cash_generator', 'is_fundamental_grower', 
            'is_sentiment_positive', 'is_momentum_star', 'is_all_star', 'is_deep_value_star',
            'is_momentum_breakout', 'is_breakout_star', 'is_growth_champion',
            # Company info
            'sector', 'industry', 'market_cap', 'size_category'
        ]
        
        # Only keep columns that exist
        save_columns = [col for col in columns if col in df.columns]
        save_df = df[save_columns].copy()
        
        # Save to main table
        try:
            save_df.to_sql(
                'stock_quality_rankings', 
                engine, 
                if_exists='append', 
                index=False, 
                method='multi'
            )
            print(f"[SUCCESS] Saved {len(save_df)} rankings")
            
            # Save audit trail
            if self.data_tracker.imputation_log:
                audit_df = pd.DataFrame(self.data_tracker.imputation_log)
                audit_df.to_sql(
                    'data_imputation_audit', 
                    engine, 
                    if_exists='append', 
                    index=False,
                    method='multi'
                )
                print(f"[SUCCESS] Saved {len(audit_df)} imputation records to audit trail")
                
        except Exception as e:
            logger.error(f"Error saving rankings: {e}")
            print(f"[ERROR] Failed to save rankings: {e}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate world-class stock quality rankings")
    parser.add_argument("--limit", type=int, help="Limit number of stocks to rank")
    parser.add_argument("--save", action="store_true", help="Save rankings to database")
    parser.add_argument("--validate", action="store_true", help="Run data validation")
    args = parser.parse_args()
    
    start_time = time.time()
    print("\n" + "=" * 60)
    print("WORLD-CLASS STOCK QUALITY RANKINGS CALCULATOR")
    print("Version: 2.0 - Institutional Grade")
    print("=" * 60)
    
    calculator = WorldClassQualityRankingCalculator()
    
    # Calculate rankings
    rankings_df = calculator.calculate_composite_rankings()
    
    # Apply limit if specified
    if args.limit:
        rankings_df = rankings_df.head(args.limit)
    
    # Display elite stocks
    print("\n[TOP ELITE ALL-STAR STOCKS]")
    all_stars = rankings_df[rankings_df['is_all_star'] == True].head(20)
    if not all_stars.empty:
        display_cols = [
            'symbol', 'final_composite_score', 'quality_tier', 
            'confidence_tier', 'sector', 'market_cap'
        ]
        print(all_stars[display_cols].to_string())
    
    print("\n[TOP ELITE STOCKS BY SCORE]")
    elite = rankings_df.nlargest(20, 'final_composite_score')
    if not elite.empty:
        display_cols = [
            'symbol', 'beat_sp500_ranking', 'excess_cash_flow_ranking',
            'fundamentals_ranking', 'sentiment_ranking', 'final_composite_score', 'confidence_tier'
        ]
        existing_cols = [col for col in display_cols if col in elite.columns]
        print(elite[existing_cols].to_string())
    
    # Data quality report
    print("\n[DATA QUALITY SUMMARY]")
    print(f"Average Data Confidence: {rankings_df['overall_data_confidence'].mean():.1f}%")
    print(f"High Confidence Stocks: {(rankings_df['confidence_tier'] == 'Very High').sum()}")
    print(f"All-Star Stocks: {rankings_df['is_all_star'].sum()}")
    
    # Save if requested
    if args.save:
        calculator.save_rankings(rankings_df)
    
    duration = time.time() - start_time
    print(f"\n[COMPLETE] Ranked {len(rankings_df)} stocks in {duration:.1f}s")
    print(f"Imputation events logged: {len(calculator.data_tracker.imputation_log)}")

if __name__ == "__main__":
    main()