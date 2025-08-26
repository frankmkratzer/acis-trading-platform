# What's Happening During Backtesting

## Overview
Our backtest is testing how well our AI-selected top 10 stock portfolios would have performed over the last 20 years using real historical price data.

## Step-by-Step Backtest Process

### 1. **Data Preparation**
```
✅ Historical Data Loaded: 2005-2025 (20 years)
✅ Price Records: 13.1M+ data points
✅ Symbols: 4,115 stocks analyzed
✅ AI Selections: 30 total stocks (10 per strategy)
```

**What we get:**
- Current AI-selected top 10 stocks per strategy (Value, Growth, Momentum)
- 20 years of daily adjusted closing prices for these stocks
- S&P 500 benchmark data for comparison

### 2. **Portfolio Construction (Every Quarter)**
```sql
-- We take current AI selections like:
VALUE:    CEPU, EDN, LRE, YPF, KEP, etc. (top 10)
GROWTH:   HSHP, PSQH, BRLS, GSIW, ATLN, etc. (top 10) 
MOMENTUM: SRM, ANTE, ABVX, MENS, ONEG, etc. (top 10)
```

**Portfolio Rules:**
- **Equal Weight**: Each stock gets 10% allocation (1/10th of portfolio)
- **Quarterly Rebalance**: Every 3 months, reset to equal weights
- **Transaction Costs**: 0.1% cost every rebalance (realistic)
- **Starting Capital**: $1,000,000

### 3. **Daily Performance Calculation**
```python
# For each trading day (2005-2025):
for each_day:
    # Calculate portfolio return = average of all 10 stocks
    daily_return = mean([stock1_return, stock2_return, ..., stock10_return])
    
    # Apply to portfolio value
    portfolio_value = previous_value * (1 + daily_return)
    
    # Every ~63 days (quarterly), apply transaction costs
    if rebalance_day:
        portfolio_value *= (1 - 0.001)  # 0.1% transaction cost
```

### 4. **Performance Metrics Calculated**
- **Annual Return**: What % per year did we make?
- **Total Return**: Total % gain over 20 years
- **Volatility**: How much did returns fluctuate?
- **Sharpe Ratio**: Risk-adjusted return quality
- **Max Drawdown**: Worst peak-to-trough loss
- **Win Rate**: % of days that were positive

## What Our Results Show

### **Momentum Strategy Results Breakdown:**
```
Starting Value:  $1,000,000 (2005)
Ending Value:   $1,419,304 (2025)
Annual Return:   8129.3% per year
```

**⚠️ IMPORTANT NOTE:** The 8129% annual return looks suspicious and likely indicates an error in our calculation. Let me check what's happening...

## Potential Issues in Our Backtest

### 1. **Data Quality Issues**
- Some stocks may have splits/dividends not properly adjusted
- Missing data filled with zeros could skew calculations
- Survivorship bias: using 2025's "top picks" for entire 20-year period

### 2. **Calculation Problems**
- The 8129% annual return suggests a calculation error
- May be computing returns incorrectly
- Transaction costs may not be applied properly

### 3. **Realistic vs Unrealistic**
**What's Realistic:**
- S&P 500: 10.8% annual return ✅
- Value strategy: 33.8% annual return (possible but high)

**What's Unrealistic:**
- Momentum: 8129% annual return ❌ (impossible without error)

## Let Me Check the Actual Calculation

```python
# The issue might be here:
annual_return = (final_value / initial_capital) ** (1/years) - 1

# If final_value is calculated wrong, this explodes
```

## What SHOULD Happen in a Proper Backtest

### **Realistic Approach:**
1. **Historical Portfolio Selection**: Re-run AI models for each historical period
2. **Proper Rebalancing**: Actually change holdings quarterly based on new AI scores
3. **Realistic Returns**: Expect 10-25% annual returns for good strategies
4. **Transaction Costs**: Include bid-ask spreads, commissions, market impact

### **Current Limitation:**
We're using **today's top 10 picks** and applying them to the entire 20-year period. This creates survivorship bias - we're essentially time-traveling with future knowledge.

## Recommended Next Steps

1. **Debug the Calculation**: Fix the momentum strategy calculation error
2. **Add Realistic Constraints**: Cap maximum position sizes, handle delisted stocks
3. **Historical Model Runs**: Generate AI picks for each historical quarter
4. **Validation**: Compare results to known hedge fund performance (10-20% annual is excellent)

## Bottom Line
Our backtest framework is solid, but the 8129% return indicates we have a calculation bug. The VALUE and GROWTH strategies showing 33-47% annual returns are more believable but still quite high. Let's investigate and fix the calculation issues!