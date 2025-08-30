#!/usr/bin/env python3
"""
Excess Cash Flow Demo
Shows how the calculation would work with sample data
"""

import pandas as pd
import numpy as np

def demonstrate_excess_cash_flow():
    """Demonstrate the Excess Cash Flow calculation with examples"""
    
    print("\n" + "="*70)
    print("EXCESS CASH FLOW CALCULATION DEMONSTRATION")
    print("="*70)
    
    print("\nFormula: Excess CF = (Operating Cash Flow - Dividends - CapEx) / Shares")
    print("Quality %: (Excess CF / Operating CF) × 100%")
    
    # Example companies with different profiles
    examples = [
        {
            'company': 'High Quality Tech Co (like MSFT)',
            'operating_cf': 100_000,  # $100B
            'dividends': 20_000,       # $20B
            'capex': 25_000,           # $25B
            'shares': 7_500,           # 7.5B shares
        },
        {
            'company': 'Dividend Aristocrat (like JNJ)',
            'operating_cf': 25_000,    # $25B
            'dividends': 12_000,       # $12B
            'capex': 3_000,            # $3B
            'shares': 2_700,           # 2.7B shares
        },
        {
            'company': 'Capital Intensive Utility',
            'operating_cf': 10_000,    # $10B
            'dividends': 3_000,        # $3B
            'capex': 8_000,            # $8B (high capex)
            'shares': 1_000,           # 1B shares
        },
        {
            'company': 'Growth Tech (reinvests everything)',
            'operating_cf': 50_000,    # $50B
            'dividends': 0,            # No dividends
            'capex': 45_000,           # $45B (massive reinvestment)
            'shares': 3_000,           # 3B shares
        },
        {
            'company': 'Struggling Retailer',
            'operating_cf': 5_000,     # $5B
            'dividends': 2_000,        # $2B
            'capex': 4_000,            # $4B
            'shares': 500,             # 500M shares
        }
    ]
    
    results = []
    
    for company in examples:
        # Calculate per share metrics
        cf_per_share = company['operating_cf'] / company['shares']
        div_per_share = company['dividends'] / company['shares']
        capex_per_share = company['capex'] / company['shares']
        
        # Calculate Excess Cash Flow
        excess_cf = cf_per_share - div_per_share - capex_per_share
        
        # Calculate quality percentage
        excess_cf_pct = (excess_cf / cf_per_share * 100) if cf_per_share > 0 else 0
        
        # Determine quality rating
        if excess_cf_pct >= 80:
            rating = "Excellent"
        elif excess_cf_pct >= 60:
            rating = "Very Good"
        elif excess_cf_pct >= 40:
            rating = "Good"
        elif excess_cf_pct >= 20:
            rating = "Fair"
        elif excess_cf_pct >= 0:
            rating = "Poor"
        else:
            rating = "Warning"
        
        results.append({
            'Company': company['company'],
            'CF/Share': f"${cf_per_share:.2f}",
            'Div/Share': f"${div_per_share:.2f}",
            'CapEx/Share': f"${capex_per_share:.2f}",
            'Excess CF': f"${excess_cf:.2f}",
            'Excess %': f"{excess_cf_pct:.1f}%",
            'Rating': rating
        })
    
    # Display results
    df = pd.DataFrame(results)
    print("\n" + "="*70)
    print("SAMPLE COMPANY ANALYSIS:")
    print("="*70)
    
    for _, row in df.iterrows():
        print(f"\n{row['Company']}")
        print(f"  Operating Cash Flow/Share: {row['CF/Share']}")
        print(f"  Less: Dividends/Share:     {row['Div/Share']}")
        print(f"  Less: CapEx/Share:         {row['CapEx/Share']}")
        print(f"  = Excess Cash Flow/Share:  {row['Excess CF']}")
        print(f"  Quality Metric:            {row['Excess %']}")
        print(f"  Rating:                    {row['Rating']}")
    
    print("\n" + "="*70)
    print("INTERPRETATION GUIDE:")
    print("="*70)
    print("""
    80-100%: EXCELLENT - Company retains most cash after all obligations
             Example: Software companies with low capex needs
    
    60-80%:  VERY GOOD - Strong cash generation with room for all needs
             Example: Mature tech, healthcare leaders
    
    40-60%:  GOOD - Solid balance between returns and reinvestment
             Example: Consumer staples, established industrials
    
    20-40%:  FAIR - Significant obligations but still positive
             Example: Telecoms, some utilities
    
    0-20%:   POOR - Most cash consumed by obligations
             Example: Capital-intensive businesses
    
    <0%:     WARNING - Spending more than generating
             Example: Struggling or heavily investing companies
    """)
    
    print("\n" + "="*70)
    print("WHY THIS METRIC MATTERS:")
    print("="*70)
    print("""
    1. Shows TRUE cash generation after all obligations
    2. Can't be manipulated by accounting tricks
    3. Reveals sustainability of dividends
    4. Indicates financial flexibility for growth/buybacks
    5. Best predictor of long-term value creation
    
    Companies with consistently high Excess Cash Flow %:
    - Can weather economic downturns
    - Have pricing power and competitive advantages
    - Can return more cash to shareholders
    - Have flexibility for strategic investments
    """)
    
    print("\n" + "="*70)
    print("DATA REQUIREMENTS:")
    print("="*70)
    print("""
    To calculate this for real companies, we need:
    
    ✅ Available in our database:
    - symbol_universe (company list)
    - company_fundamentals_overview (some metrics)
    
    ❌ Still needed:
    - fundamentals table with cash flow statements
    - Run: python data_fetch/fundamentals/fetch_fundamentals.py
    - This will populate operating_cash_flow, free_cash_flow, capital_expenditures
    
    Once fundamentals are fetched, run:
    python analysis/excess_cash_flow.py
    
    This will calculate and rank all companies by this metric.
    """)


if __name__ == "__main__":
    demonstrate_excess_cash_flow()