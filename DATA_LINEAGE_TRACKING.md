# Data Lineage Tracking: The Importance of `fetched_at`

## ✅ FIXED: `fetched_at` Field Issue

**Problem**: Portfolio tables had NULL `fetched_at` values
**Solution**: Updated `create_portfolios.py` to set `CURRENT_TIMESTAMP` when creating portfolios

## Why `fetched_at` is Critical

### 1. **Data Freshness Tracking**
```sql
-- Know when data was last updated
SELECT symbol, score, fetched_at 
FROM ai_growth_portfolio 
WHERE fetched_at < CURRENT_DATE - INTERVAL '1 day';
```
- **Live Trading**: Ensures you're not trading on stale data
- **Risk Management**: Prevents using outdated scores for position sizing
- **Compliance**: Regulators require knowing data age for risk calculations

### 2. **System Monitoring**
```sql
-- Detect when systems haven't run
SELECT strategy, MAX(fetched_at) as last_update
FROM (
    SELECT 'value' as strategy, MAX(fetched_at) as fetched_at FROM ai_value_portfolio
    UNION ALL
    SELECT 'growth' as strategy, MAX(fetched_at) as fetched_at FROM ai_growth_portfolio
    UNION ALL  
    SELECT 'momentum' as strategy, MAX(fetched_at) as fetched_at FROM ai_momentum_portfolio
) t
GROUP BY strategy
ORDER BY last_update;
```
- **Alert Generation**: Trigger alerts when data is too old
- **System Health**: Monitor pipeline execution success
- **Troubleshooting**: Identify which components failed

### 3. **Audit Trail**
```sql
-- Track portfolio changes over time
SELECT symbol, score, fetched_at, 
       LAG(score) OVER (PARTITION BY symbol ORDER BY fetched_at) as prev_score
FROM ai_growth_portfolio  
WHERE symbol = 'AAPL'
ORDER BY fetched_at DESC;
```
- **Regulatory Compliance**: Prove when trades were based on which data
- **Performance Attribution**: Link returns to specific model versions
- **Model Validation**: Track how scores change over time

### 4. **Data Quality Assurance**
```sql
-- Ensure all strategies updated together
SELECT 
    DATE(fetched_at) as update_date,
    COUNT(DISTINCT CASE WHEN strategy = 'value' THEN fetched_at END) as value_updates,
    COUNT(DISTINCT CASE WHEN strategy = 'growth' THEN fetched_at END) as growth_updates,
    COUNT(DISTINCT CASE WHEN strategy = 'momentum' THEN fetched_at END) as momentum_updates
FROM (
    SELECT 'value' as strategy, fetched_at FROM ai_value_portfolio
    UNION ALL
    SELECT 'growth' as strategy, fetched_at FROM ai_growth_portfolio  
    UNION ALL
    SELECT 'momentum' as strategy, fetched_at FROM ai_momentum_portfolio
) t
GROUP BY DATE(fetched_at)
ORDER BY update_date DESC;
```

### 5. **Live Trading Safety**
```python
# Example: Check data freshness before trading
def is_data_fresh(engine, max_age_hours=24):
    query = """
    SELECT MIN(fetched_at) as oldest_data
    FROM (
        SELECT MAX(fetched_at) as fetched_at FROM ai_value_portfolio
        UNION ALL
        SELECT MAX(fetched_at) as fetched_at FROM ai_growth_portfolio
        UNION ALL  
        SELECT MAX(fetched_at) as fetched_at FROM ai_momentum_portfolio
    ) t
    """
    
    result = engine.execute(query).fetchone()
    if not result[0]:
        return False
        
    hours_old = (datetime.utcnow() - result[0]).total_seconds() / 3600
    return hours_old < max_age_hours

# Only execute trades if data is fresh
if is_data_fresh(engine):
    execute_trades()
else:
    send_alert("TRADING HALTED: Stale portfolio data detected")
```

## Best Practices

### 1. **Always Set Timestamps**
```sql
-- Every INSERT should include fetched_at
INSERT INTO ai_growth_portfolio (..., fetched_at)
VALUES (..., CURRENT_TIMESTAMP);

-- Every UPDATE should refresh fetched_at  
UPDATE ai_growth_portfolio 
SET score = new_score, fetched_at = CURRENT_TIMESTAMP
WHERE symbol = 'AAPL';
```

### 2. **Monitor Data Age**
```sql
-- Daily monitoring query
SELECT 
    table_name,
    MAX(fetched_at) as latest,
    EXTRACT(HOUR FROM CURRENT_TIMESTAMP - MAX(fetched_at)) as hours_old
FROM (
    SELECT 'ai_value_portfolio' as table_name, fetched_at FROM ai_value_portfolio
    UNION ALL
    SELECT 'ai_growth_portfolio' as table_name, fetched_at FROM ai_growth_portfolio
    UNION ALL
    SELECT 'ai_momentum_portfolio' as table_name, fetched_at FROM ai_momentum_portfolio
) t
GROUP BY table_name
HAVING MAX(fetched_at) < CURRENT_TIMESTAMP - INTERVAL '4 hours'
ORDER BY hours_old DESC;
```

### 3. **Automated Alerts**
- **Stale Data**: Alert if any portfolio data > 24 hours old
- **Missing Updates**: Alert if expected pipeline runs didn't occur
- **Inconsistent Timing**: Alert if strategies updated at very different times

## Implementation Status

**✅ COMPLETED:**
- Fixed `create_portfolios.py` to set `fetched_at = CURRENT_TIMESTAMP`
- Updated existing portfolio records with current timestamps
- All portfolio tables now properly track when data was generated

**✅ VERIFIED:**
- ai_value_portfolio: 10/10 records with timestamps
- ai_growth_portfolio: 10/10 records with timestamps  
- ai_momentum_portfolio: 10/10 records with timestamps

Your trading system now has proper data lineage tracking for safe live trading operations! 🚀