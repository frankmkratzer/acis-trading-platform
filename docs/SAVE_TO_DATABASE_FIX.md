# Fix for Saving Quality Rankings to Database

## Problem
The `calculate_quality_rankings.py` script was not saving records to the `stock_quality_rankings` table.

## Root Causes
1. **Wrong table name**: Script was saving to `stock_quality_rankings_optimized` instead of `stock_quality_rankings`
2. **Missing --save flag**: Script only saves when explicitly told to with `--save` flag
3. **Column name mismatch**: Column names from the optimized version didn't match the database schema

## Solutions Implemented

### 1. Fixed Table Name ✅
Changed default table name from `stock_quality_rankings_optimized` to `stock_quality_rankings`

### 2. Added Column Mapping ✅
Created mapping between the optimized script's output columns and the expected database schema:

```python
column_mapping = {
    'final_rank': 'composite_ranking',
    'final_score': 'composite_score',
    'sp500_beat_rate': 'beat_sp500_ranking',
    'fcf_yield': 'free_cash_flow_yield',
    'avg_revenue_growth': 'revenue_growth_1yr',
    'avg_roe': 'roe_current'
}
```

### 3. Added Clear Instructions ✅
Script now shows a message when rankings are calculated but not saved:
```
[INFO] Rankings calculated but not saved. Use --save flag to persist to database
      Example: python calculate_quality_rankings.py --save
```

## How to Use

### Basic Usage (Calculate Only - No Save)
```bash
python calculate_quality_rankings.py
```
This will calculate and display rankings but NOT save to database.

### Save to Database
```bash
python calculate_quality_rankings.py --save
```
This will calculate rankings AND save them to the `stock_quality_rankings` table.

### With Options
```bash
# Limit to 50 stocks and save
python calculate_quality_rankings.py --limit 50 --save

# Verbose logging and save
python calculate_quality_rankings.py --verbose --save
```

## Testing the Fix

Run the test script to verify everything works:
```bash
python test_quality_rankings_save.py
```

This will:
1. Check if the `stock_quality_rankings` table exists
2. Show current record count
3. Display recent records
4. Show the table schema
5. List the columns that will be saved

## Database Schema Compatibility

The script now saves these columns to `stock_quality_rankings`:
- `symbol` - Stock ticker
- `ranking_date` - Date of calculation (automatically set to today)
- `composite_ranking` - Overall rank (from `final_rank`)
- `composite_score` - Overall score (from `final_score`)
- `beat_sp500_ranking` - SP500 outperformance metric
- `free_cash_flow_yield` - FCF yield percentage
- `revenue_growth_1yr` - Average revenue growth
- `roe_current` - Current return on equity

## Important Notes

1. **Always use --save flag**: The script will NOT save by default. You must explicitly use `--save`.

2. **Incremental saves**: Each run with `--save` appends new records. To avoid duplicates for the same date, consider:
   - Deleting today's records before re-running
   - Using a different `ranking_date` 
   - Implementing upsert logic

3. **Missing columns**: Some columns in the `stock_quality_rankings` table may be NULL because the optimized version doesn't calculate all the same metrics as the original.

4. **Connection issues**: If you get SSL or connection errors, the script will retry up to 3 times automatically.

## Verification Query

To verify records were saved, run this SQL:
```sql
SELECT COUNT(*) as record_count, 
       ranking_date,
       MIN(composite_score) as min_score,
       MAX(composite_score) as max_score,
       AVG(composite_score) as avg_score
FROM stock_quality_rankings
WHERE ranking_date = CURRENT_DATE
GROUP BY ranking_date;
```

## Troubleshooting

If records are still not saving:

1. **Check database connection**:
   - Verify POSTGRES_URL is set correctly
   - Test connection with `psql`

2. **Check table exists**:
   ```sql
   SELECT * FROM information_schema.tables 
   WHERE table_name = 'stock_quality_rankings';
   ```

3. **Check for errors**:
   - Look in `logs/quality_rankings_optimized.log`
   - Run with `--verbose` flag for detailed output

4. **Verify data exists**:
   - Ensure you have data in `stock_prices`, `fundamentals`, etc.
   - The script needs data to calculate rankings

## Future Improvements

1. **Add upsert logic** to handle re-runs for the same date
2. **Calculate more metrics** to fill all columns in the schema
3. **Add dry-run mode** to preview what would be saved
4. **Implement batch processing** for historical dates
5. **Add data validation** before saving to catch issues early