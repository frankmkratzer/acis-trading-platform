#!/usr/bin/env python3
"""
Enhanced Fundamental Analysis System
Comprehensive fundamental metrics calculation and strategy scoring
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

class EnhancedFundamentalAnalyzer:
    def __init__(self):
        load_dotenv()
        self.engine = create_engine(os.getenv('POSTGRES_URL'))
    
    def get_market_cap_data(self, symbols, as_of_date):
        """Get market cap data for ratio calculations"""
        with self.engine.connect() as conn:
            symbol_list = "'" + "','".join(symbols[:200]) + "'"
            
            result = conn.execute(text(f"""
                SELECT DISTINCT ON (symbol)
                    symbol,
                    adjusted_close * 1000000 as approx_market_cap  -- Rough approximation
                FROM stock_eod_daily
                WHERE symbol IN ({symbol_list})
                    AND trade_date <= '{as_of_date}'
                    AND adjusted_close IS NOT NULL
                ORDER BY symbol, trade_date DESC
            """))
            
            market_caps = {}
            for row in result.fetchall():
                market_caps[row[0]] = float(row[1])
            
            return market_caps
    
    def calculate_comprehensive_fundamentals(self, symbols, as_of_date):
        """Calculate comprehensive fundamental metrics for all strategies"""
        print(f"    Calculating comprehensive fundamentals for {len(symbols)} symbols as of {as_of_date}")
        
        with self.engine.connect() as conn:
            symbol_list = "'" + "','".join(symbols[:200]) + "'"
            
            # Get latest fundamental data before as_of_date
            result = conn.execute(text(f"""
                WITH latest_fundamentals AS (
                    SELECT 
                        symbol,
                        fiscal_date,
                        totalrevenue,
                        grossprofit, 
                        netincome,
                        eps,
                        totalassets,
                        totalliabilities,
                        totalshareholderequity,
                        operatingcashflow,
                        capitalexpenditures,
                        dividendpayout,
                        free_cf,
                        ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY fiscal_date DESC) as rn
                    FROM fundamentals_annual
                    WHERE symbol IN ({symbol_list})
                        AND fiscal_date <= '{as_of_date}'
                        AND totalrevenue IS NOT NULL
                        AND totalrevenue > 0
                ),
                growth_calcs AS (
                    SELECT 
                        lf.*,
                        prev.totalrevenue as prev_revenue,
                        prev.netincome as prev_netincome,
                        prev.eps as prev_eps
                    FROM latest_fundamentals lf
                    LEFT JOIN fundamentals_annual prev ON lf.symbol = prev.symbol 
                        AND prev.fiscal_date = (
                            SELECT MAX(fiscal_date) 
                            FROM fundamentals_annual pf 
                            WHERE pf.symbol = lf.symbol 
                                AND pf.fiscal_date < lf.fiscal_date
                                AND pf.fiscal_date <= '{as_of_date}'
                        )
                    WHERE lf.rn = 1
                )
                SELECT 
                    symbol,
                    totalrevenue,
                    grossprofit,
                    netincome, 
                    eps,
                    totalassets,
                    totalliabilities,
                    totalshareholderequity,
                    operatingcashflow,
                    capitalexpenditures,
                    dividendpayout,
                    free_cf,
                    prev_revenue,
                    prev_netincome,
                    prev_eps
                FROM growth_calcs
                ORDER BY symbol
            """))
            
            fundamentals_data = []
            market_caps = self.get_market_cap_data(symbols, as_of_date)
            
            for row in result.fetchall():
                symbol = row[0]
                market_cap = market_caps.get(symbol, 0)
                
                # Raw fundamental data
                revenue = float(row[1]) if row[1] else 0
                gross_profit = float(row[2]) if row[2] else 0
                net_income = float(row[3]) if row[3] else 0
                eps = float(row[4]) if row[4] else 0
                total_assets = float(row[5]) if row[5] else 0
                total_liab = float(row[6]) if row[6] else 0
                equity = float(row[7]) if row[7] else 0
                op_cashflow = float(row[8]) if row[8] else 0
                capex = float(row[9]) if row[9] else 0
                dividends = float(row[10]) if row[10] else 0
                free_cf = float(row[11]) if row[11] else 0
                
                # Previous year data for growth
                prev_revenue = float(row[12]) if row[12] else 0
                prev_income = float(row[13]) if row[13] else 0
                prev_eps = float(row[14]) if row[14] else 0
                
                # Calculate key ratios
                pe_ratio = market_cap / net_income if net_income > 0 else None
                pb_ratio = market_cap / equity if equity > 0 else None  
                ps_ratio = market_cap / revenue if revenue > 0 else None
                
                roe = net_income / equity if equity > 0 else None
                roa = net_income / total_assets if total_assets > 0 else None
                debt_to_equity = total_liab / equity if equity > 0 else None
                
                gross_margin = gross_profit / revenue if revenue > 0 else None
                net_margin = net_income / revenue if revenue > 0 else None
                
                # Growth rates
                revenue_growth = (revenue / prev_revenue - 1) if prev_revenue > 0 else None
                earnings_growth = (net_income / prev_income - 1) if prev_income != 0 else None
                eps_growth = (eps / prev_eps - 1) if prev_eps != 0 else None
                
                # Dividend metrics
                dividend_yield = None  # Would need current price
                payout_ratio = dividends / net_income if net_income > 0 else None
                fcf_yield = free_cf / market_cap if market_cap > 0 else None
                
                fundamentals_data.append({
                    'symbol': symbol,
                    # Raw data
                    'revenue': revenue,
                    'net_income': net_income,
                    'eps': eps,
                    'total_assets': total_assets,
                    'equity': equity,
                    'free_cf': free_cf,
                    'dividends': dividends,
                    'market_cap': market_cap,
                    
                    # Valuation ratios
                    'pe_ratio': pe_ratio,
                    'pb_ratio': pb_ratio,
                    'ps_ratio': ps_ratio,
                    
                    # Profitability
                    'roe': roe,
                    'roa': roa,
                    'gross_margin': gross_margin,
                    'net_margin': net_margin,
                    
                    # Financial health
                    'debt_to_equity': debt_to_equity,
                    
                    # Growth
                    'revenue_growth': revenue_growth,
                    'earnings_growth': earnings_growth,
                    'eps_growth': eps_growth,
                    
                    # Dividend/Cash flow
                    'payout_ratio': payout_ratio,
                    'fcf_yield': fcf_yield
                })
            
            return pd.DataFrame(fundamentals_data)
    
    def calculate_value_scores(self, fundamentals_df):
        """Enhanced value scoring with comprehensive fundamentals"""
        value_scores = []
        
        for _, row in fundamentals_df.iterrows():
            symbol = row['symbol']
            score = 0
            
            # PE Ratio (lower is better)
            if row['pe_ratio'] and 1 < row['pe_ratio'] < 50:
                score += max(0, (1 / row['pe_ratio']) * 25)
            
            # PB Ratio (lower is better)  
            if row['pb_ratio'] and 0.1 < row['pb_ratio'] < 10:
                score += max(0, (1 / row['pb_ratio']) * 15)
            
            # PS Ratio (lower is better)
            if row['ps_ratio'] and 0.1 < row['ps_ratio'] < 10:
                score += max(0, (1 / row['ps_ratio']) * 10)
            
            # ROE (higher is better)
            if row['roe'] and row['roe'] > 0:
                score += min(row['roe'] * 100, 30) * 0.5  # Cap at 30% ROE
            
            # Free Cash Flow Yield (higher is better)
            if row['fcf_yield'] and row['fcf_yield'] > 0:
                score += min(row['fcf_yield'] * 100, 20) * 1.0  # Cap at 20% FCF yield
            
            # Debt penalty (lower debt is better)
            if row['debt_to_equity'] and row['debt_to_equity'] > 0:
                score -= min(row['debt_to_equity'] * 20, 40)  # Penalty for high debt
            
            # Net margin bonus (higher is better)
            if row['net_margin'] and row['net_margin'] > 0:
                score += min(row['net_margin'] * 100, 25) * 0.2  # Cap at 25% margin
            
            value_scores.append({
                'symbol': symbol,
                'value_score': max(score, 0),  # No negative scores
                'pe_ratio': row['pe_ratio'],
                'pb_ratio': row['pb_ratio'],
                'ps_ratio': row['ps_ratio'],
                'roe': row['roe'],
                'debt_to_equity': row['debt_to_equity']
            })
        
        df = pd.DataFrame(value_scores)
        if len(df) > 0:
            df = df.sort_values('value_score', ascending=False)
            df['rank'] = range(1, len(df) + 1)
            return df.head(10)
        return pd.DataFrame()
    
    def calculate_growth_scores(self, fundamentals_df):
        """Enhanced growth scoring with comprehensive fundamentals"""
        growth_scores = []
        
        for _, row in fundamentals_df.iterrows():
            symbol = row['symbol']
            score = 0
            
            # Revenue Growth (higher is better)
            if row['revenue_growth'] and row['revenue_growth'] > 0:
                score += min(row['revenue_growth'] * 100, 100) * 0.3  # Cap at 100% growth
            
            # Earnings Growth (higher is better)
            if row['earnings_growth'] and row['earnings_growth'] > 0:
                score += min(row['earnings_growth'] * 100, 200) * 0.4  # Cap at 200% growth
            
            # EPS Growth (higher is better) 
            if row['eps_growth'] and row['eps_growth'] > 0:
                score += min(row['eps_growth'] * 100, 150) * 0.2  # Cap at 150% growth
            
            # ROE Quality (profitable growth)
            if row['roe'] and row['roe'] > 0:
                score += min(row['roe'] * 100, 40) * 0.2  # Cap at 40% ROE
            
            # Gross Margin Quality (high-margin business)
            if row['gross_margin'] and row['gross_margin'] > 0:
                score += min(row['gross_margin'] * 100, 80) * 0.15  # Cap at 80% margin
            
            # Penalize excessive debt (growth at any cost is bad)
            if row['debt_to_equity'] and row['debt_to_equity'] > 1:
                score -= (row['debt_to_equity'] - 1) * 10
            
            growth_scores.append({
                'symbol': symbol,
                'growth_score': max(score, 0),
                'revenue_growth': row['revenue_growth'],
                'earnings_growth': row['earnings_growth'],
                'eps_growth': row['eps_growth'],
                'roe': row['roe'],
                'gross_margin': row['gross_margin']
            })
        
        df = pd.DataFrame(growth_scores)
        if len(df) > 0:
            df = df.sort_values('growth_score', ascending=False)
            df['rank'] = range(1, len(df) + 1)
            return df.head(10)
        return pd.DataFrame()
    
    def calculate_dividend_scores(self, fundamentals_df):
        """Dividend strategy scoring"""
        dividend_scores = []
        
        for _, row in fundamentals_df.iterrows():
            symbol = row['symbol']
            score = 0
            
            # Only consider companies that pay dividends
            if not row['dividends'] or row['dividends'] <= 0:
                continue
            
            # Payout Ratio (sustainable dividends)
            if row['payout_ratio'] and 0.2 < row['payout_ratio'] < 0.8:
                score += (0.6 - abs(row['payout_ratio'] - 0.5)) * 50  # Optimal around 50%
            
            # ROE Quality (earnings support)
            if row['roe'] and row['roe'] > 0.10:  # At least 10% ROE
                score += min(row['roe'] * 100, 30) * 0.3
            
            # Free Cash Flow Coverage
            if row['free_cf'] and row['dividends'] and row['free_cf'] > 0:
                coverage = row['free_cf'] / row['dividends']
                if coverage > 1.2:  # Good coverage
                    score += min(coverage * 10, 25)
            
            # Financial Stability (low debt)
            if row['debt_to_equity'] and row['debt_to_equity'] < 0.6:
                score += (0.6 - row['debt_to_equity']) * 20
            
            dividend_scores.append({
                'symbol': symbol,
                'dividend_score': max(score, 0),
                'payout_ratio': row['payout_ratio'],
                'roe': row['roe'],
                'debt_to_equity': row['debt_to_equity'],
                'dividends': row['dividends']
            })
        
        df = pd.DataFrame(dividend_scores)
        if len(df) > 0:
            df = df.sort_values('dividend_score', ascending=False)
            df['rank'] = range(1, len(df) + 1)
            return df.head(10)
        return pd.DataFrame()

def test_enhanced_fundamentals():
    """Test the enhanced fundamental analysis system"""
    analyzer = EnhancedFundamentalAnalyzer()
    
    # Test with recent date
    test_date = datetime(2024, 1, 1).date()
    
    # Get some symbols to test with
    with analyzer.engine.connect() as conn:
        result = conn.execute(text("""
            SELECT DISTINCT symbol 
            FROM fundamentals_annual 
            WHERE totalrevenue IS NOT NULL 
                AND totalrevenue > 0
            ORDER BY symbol 
            LIMIT 50
        """))
        test_symbols = [row[0] for row in result.fetchall()]
    
    print(f"Testing with {len(test_symbols)} symbols as of {test_date}")
    
    # Calculate comprehensive fundamentals
    fundamentals_df = analyzer.calculate_comprehensive_fundamentals(test_symbols, test_date)
    print(f"Calculated fundamentals for {len(fundamentals_df)} companies")
    
    if len(fundamentals_df) > 0:
        # Test each strategy
        print("\\nVALUE Strategy Top Picks:")
        value_picks = analyzer.calculate_value_scores(fundamentals_df)
        for _, row in value_picks.head(5).iterrows():
            pe_str = f"{row['pe_ratio']:.1f}" if row['pe_ratio'] else 'N/A'
            print(f"  {row['symbol']}: Score={row['value_score']:.1f}, PE={pe_str}")
        
        print("\\nGROWTH Strategy Top Picks:")
        growth_picks = analyzer.calculate_growth_scores(fundamentals_df)
        for _, row in growth_picks.head(5).iterrows():
            growth_str = f"{row['revenue_growth']:.1%}" if row['revenue_growth'] else 'N/A'
            print(f"  {row['symbol']}: Score={row['growth_score']:.1f}, RevGrowth={growth_str}")
        
        print("\\nDIVIDEND Strategy Top Picks:")
        dividend_picks = analyzer.calculate_dividend_scores(fundamentals_df)
        for _, row in dividend_picks.head(5).iterrows():
            payout_str = f"{row['payout_ratio']:.1%}" if row['payout_ratio'] else 'N/A'
            print(f"  {row['symbol']}: Score={row['dividend_score']:.1f}, Payout={payout_str}")

if __name__ == "__main__":
    test_enhanced_fundamentals()