#!/usr/bin/env python3
"""
Dividend Sustainability Demo
Shows how the dividend analysis works with sample data
"""

import pandas as pd
import numpy as np

def demonstrate_dividend_sustainability():
    """Demonstrate the Dividend Sustainability analysis with examples"""
    
    print("\n" + "="*70)
    print("DIVIDEND SUSTAINABILITY ANALYSIS DEMONSTRATION")
    print("="*70)
    
    print("\nFormula Components:")
    print("1. Payment Streak: Consecutive years of dividend payments")
    print("2. Increase Streak: Consecutive years of dividend increases")
    print("3. Payout Ratio: Dividends / Earnings (or FCF)")
    print("4. Sustainability: Combined score based on payout ratio and cash flow")
    print("5. Quality Score: Weighted combination of all factors")
    
    # Example companies with different dividend profiles
    examples = [
        {
            'company': 'Dividend Aristocrat (like JNJ)',
            'payment_streak': 60,      # 60 years of payments
            'increase_streak': 40,      # 40 years of increases
            'annual_dividend': 4.20,    # $4.20 per share
            'eps': 8.00,               # $8.00 earnings per share
            'fcf_per_share': 7.50,     # $7.50 FCF per share
            'dividend_growth_5y': 6.5,  # 6.5% annual growth
            'excess_cf_pct': 75,       # 75% excess cash flow
        },
        {
            'company': 'Dividend King (like KO)',
            'payment_streak': 100,     # 100+ years
            'increase_streak': 61,     # 61 years of increases
            'annual_dividend': 1.84,   # $1.84 per share
            'eps': 2.47,              # $2.47 EPS
            'fcf_per_share': 2.30,    # $2.30 FCF
            'dividend_growth_5y': 3.8, # 3.8% growth
            'excess_cf_pct': 60,      # 60% excess CF
        },
        {
            'company': 'High Yield REIT',
            'payment_streak': 15,      # 15 years
            'increase_streak': 8,       # 8 years increases
            'annual_dividend': 5.00,    # High yield
            'eps': 5.20,               # Close to dividend
            'fcf_per_share': 5.50,     # Tight margin
            'dividend_growth_5y': 2.5,  # Slow growth
            'excess_cf_pct': 20,       # Low excess CF
        },
        {
            'company': 'Growth Tech (MSFT-like)',
            'payment_streak': 20,      # Started dividends later
            'increase_streak': 18,      # Strong increases
            'annual_dividend': 2.80,    # Moderate yield
            'eps': 11.50,              # High earnings
            'fcf_per_share': 10.00,    # Strong FCF
            'dividend_growth_5y': 10.2, # High growth
            'excess_cf_pct': 85,       # Excellent excess CF
        },
        {
            'company': 'Struggling Dividend Payer',
            'payment_streak': 25,       # Long history
            'increase_streak': 0,        # Cut recently
            'annual_dividend': 1.00,     # Reduced dividend
            'eps': 0.90,                # Earnings below dividend
            'fcf_per_share': 1.20,      # Tight FCF
            'dividend_growth_5y': -5.0,  # Declining
            'excess_cf_pct': 10,        # Poor excess CF
        },
        {
            'company': 'Young Dividend Grower',
            'payment_streak': 5,         # New to dividends
            'increase_streak': 5,        # Growing fast
            'annual_dividend': 0.60,     # Low payout
            'eps': 4.00,                # Strong earnings
            'fcf_per_share': 3.80,      # Good FCF
            'dividend_growth_5y': 25.0,  # Rapid growth
            'excess_cf_pct': 70,        # Good excess CF
        }
    ]
    
    results = []
    
    for company in examples:
        # Calculate payout ratios
        payout_earnings = (company['annual_dividend'] / company['eps'] * 100) if company['eps'] > 0 else 999
        payout_fcf = (company['annual_dividend'] / company['fcf_per_share'] * 100) if company['fcf_per_share'] > 0 else 999
        
        # Calculate sustainability score (0-100)
        if payout_fcf <= 40:
            sustainability = 90 + (40 - payout_fcf) / 4
        elif payout_fcf <= 60:
            sustainability = 70 + (60 - payout_fcf)
        elif payout_fcf <= 80:
            sustainability = 50 + (80 - payout_fcf)
        elif payout_fcf <= 100:
            sustainability = 30 + (100 - payout_fcf) * 0.5
        else:
            sustainability = max(0, 30 - (payout_fcf - 100) * 0.3)
        
        # Calculate dividend quality score (0-100)
        # Weights: Payment streak 25%, Increase streak 20%, Sustainability 30%, Excess CF 25%
        streak_score = min(100, company['payment_streak'] * 4)  # 25 years = 100
        increase_score = min(100, company['increase_streak'] * 5)  # 20 years = 100
        
        quality_score = (
            streak_score * 0.25 +
            increase_score * 0.20 +
            sustainability * 0.30 +
            company['excess_cf_pct'] * 0.25
        )
        
        # Determine ratings
        if sustainability >= 80:
            safety_rating = 'Very Safe'
        elif sustainability >= 60:
            safety_rating = 'Safe'
        elif sustainability >= 40:
            safety_rating = 'Moderate'
        elif sustainability >= 20:
            safety_rating = 'Risky'
        else:
            safety_rating = 'Unsustainable'
        
        if quality_score >= 80:
            quality_rating = 'Dividend Aristocrat'
        elif quality_score >= 60:
            quality_rating = 'High Quality'
        elif quality_score >= 40:
            quality_rating = 'Good Quality'
        elif quality_score >= 20:
            quality_rating = 'Fair Quality'
        else:
            quality_rating = 'Poor Quality'
        
        results.append({
            'Company': company['company'],
            'Payment Years': company['payment_streak'],
            'Increase Years': company['increase_streak'],
            'Payout (Earnings)': f"{payout_earnings:.1f}%",
            'Payout (FCF)': f"{payout_fcf:.1f}%",
            'Sustainability': f"{sustainability:.1f}",
            'Safety': safety_rating,
            'Quality Score': f"{quality_score:.1f}",
            'Rating': quality_rating
        })
    
    # Display results
    df = pd.DataFrame(results)
    print("\n" + "="*70)
    print("SAMPLE COMPANY ANALYSIS:")
    print("="*70)
    
    for _, row in df.iterrows():
        print(f"\n{row['Company']}")
        print(f"  Dividend History: {row['Payment Years']} years paying, {row['Increase Years']} years increasing")
        print(f"  Payout Ratios: {row['Payout (Earnings)']} of earnings, {row['Payout (FCF)']} of FCF")
        print(f"  Sustainability Score: {row['Sustainability']}/100")
        print(f"  Safety Rating: {row['Safety']}")
        print(f"  Overall Quality Score: {row['Quality Score']}/100")
        print(f"  Classification: {row['Rating']}")
    
    print("\n" + "="*70)
    print("INTERPRETATION GUIDE:")
    print("="*70)
    print("""
    DIVIDEND QUALITY RATINGS:
    
    80-100: DIVIDEND ARISTOCRAT
            - Long payment history (25+ years)
            - Consistent increases
            - Very sustainable payout
            - Strong excess cash flow
    
    60-80:  HIGH QUALITY
            - Solid payment history
            - Regular increases
            - Safe payout ratio
            - Good cash generation
    
    40-60:  GOOD QUALITY
            - Established dividend
            - Some increases
            - Moderate payout
            - Adequate coverage
    
    20-40:  FAIR QUALITY
            - Shorter history
            - Inconsistent growth
            - Higher payout ratio
            - Limited flexibility
    
    0-20:   POOR QUALITY
            - Risk of dividend cut
            - Unsustainable payout
            - Weak cash flow
            - Avoid for income
    """)
    
    print("\n" + "="*70)
    print("KEY SUSTAINABILITY METRICS:")
    print("="*70)
    print("""
    PAYOUT RATIO TARGETS:
    - Excellent: < 40% (lots of room for growth)
    - Good: 40-60% (balanced approach)
    - Fair: 60-80% (limited flexibility)
    - Poor: > 80% (risk of cut)
    
    INTEGRATION WITH EXCESS CASH FLOW:
    - High excess CF % = More sustainable dividend
    - Low excess CF % = Dividend at risk
    - Negative excess CF = Unsustainable
    
    IDEAL DIVIDEND STOCK PROFILE:
    ✓ 10+ years payment history
    ✓ 5+ years consecutive increases
    ✓ Payout ratio < 60% of FCF
    ✓ Excess cash flow > 40%
    ✓ Dividend growth > inflation
    """)
    
    print("\n" + "="*70)
    print("INVESTMENT STRATEGY:")
    print("="*70)
    print("""
    1. SCREEN for minimum requirements:
       - 10+ years dividend history
       - Payout ratio < 80%
       - Positive excess cash flow
    
    2. RANK by quality score:
       - Combines all sustainability factors
       - Weights history, growth, and coverage
    
    3. SELECT top quintile:
       - Focus on Dividend Aristocrats
       - Diversify across sectors
       - Monitor for safety changes
    
    4. MONITOR quarterly:
       - Payout ratio trends
       - Dividend growth rates
       - Excess cash flow changes
    """)


if __name__ == "__main__":
    demonstrate_dividend_sustainability()