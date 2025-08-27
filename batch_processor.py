#!/usr/bin/env python3
"""
Optimized Batch Processor for Bulk Database Operations
Implements efficient bulk inserts using COPY and psycopg2 optimizations
"""

import io
import csv
import logging
import pandas as pd
import psycopg2
from typing import List, Dict, Any, Optional, Iterator
from contextlib import contextmanager
from psycopg2.extras import execute_values, execute_batch
from sqlalchemy import text
from db_connection_manager import get_db_engine

logger = logging.getLogger(__name__)

class BatchProcessor:
    """Optimized batch processing for database operations"""
    
    def __init__(self, batch_size: int = 10000, page_size: int = 1000):
        """
        Initialize batch processor
        
        Args:
            batch_size: Number of records to process in one batch
            page_size: Page size for execute_values (psycopg2 optimization)
        """
        self.batch_size = batch_size
        self.page_size = page_size
        self.engine = get_db_engine()
        
    @contextmanager
    def get_raw_connection(self):
        """Get raw psycopg2 connection for COPY operations"""
        conn = self.engine.raw_connection()
        try:
            yield conn
        finally:
            conn.close()
    
    def bulk_insert_with_copy(self, df: pd.DataFrame, table_name: str, 
                             columns: Optional[List[str]] = None) -> int:
        """
        Ultra-fast bulk insert using PostgreSQL COPY command
        
        Args:
            df: DataFrame to insert
            table_name: Target table name
            columns: Column names (if None, uses all df columns)
            
        Returns:
            Number of records inserted
        """
        if df.empty:
            return 0
            
        columns = columns or df.columns.tolist()
        
        # Create in-memory CSV buffer
        buffer = io.StringIO()
        df[columns].to_csv(buffer, index=False, header=False, na_rep='\\N')
        buffer.seek(0)
        
        with self.get_raw_connection() as conn:
            with conn.cursor() as cursor:
                try:
                    # Use COPY for maximum speed
                    cursor.copy_expert(
                        f"""COPY {table_name} ({','.join(columns)}) 
                        FROM STDIN WITH CSV NULL AS '\\N'""",
                        buffer
                    )
                    conn.commit()
                    logger.info(f"Bulk inserted {len(df)} records into {table_name}")
                    return len(df)
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Bulk insert failed: {e}")
                    raise
    
    def bulk_upsert_prices(self, df: pd.DataFrame) -> int:
        """
        Optimized bulk upsert for stock prices using temporary table and COPY
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Number of records processed
        """
        if df.empty:
            return 0
        
        # Ensure correct data types
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
        
        with self.get_raw_connection() as conn:
            with conn.cursor() as cursor:
                try:
                    # Create temporary table
                    cursor.execute("""
                        CREATE TEMP TABLE temp_prices (
                            symbol TEXT,
                            trade_date DATE,
                            open NUMERIC,
                            high NUMERIC,
                            low NUMERIC,
                            close NUMERIC,
                            adjusted_close NUMERIC,
                            volume BIGINT,
                            dividend_amount NUMERIC,
                            split_coefficient NUMERIC,
                            fetched_at TIMESTAMPTZ
                        )
                    """)
                    
                    # Use COPY for ultra-fast insert into temp table
                    buffer = io.StringIO()
                    df.to_csv(buffer, index=False, header=False, na_rep='\\N')
                    buffer.seek(0)
                    
                    cursor.copy_expert(
                        """COPY temp_prices FROM STDIN WITH CSV NULL AS '\\N'""",
                        buffer
                    )
                    
                    # Perform efficient upsert
                    cursor.execute("""
                        INSERT INTO stock_prices (
                            symbol, date, open_price, high_price, low_price, 
                            close_price, adjusted_close, volume, 
                            dividend_amount, split_coefficient, created_at
                        )
                        SELECT 
                            symbol, trade_date, open, high, low, close,
                            adjusted_close, volume, dividend_amount, 
                            split_coefficient, fetched_at
                        FROM temp_prices
                        ON CONFLICT (symbol, date) DO UPDATE SET
                            open_price = EXCLUDED.open_price,
                            high_price = EXCLUDED.high_price,
                            low_price = EXCLUDED.low_price,
                            close_price = EXCLUDED.close_price,
                            adjusted_close = EXCLUDED.adjusted_close,
                            volume = EXCLUDED.volume,
                            dividend_amount = EXCLUDED.dividend_amount,
                            split_coefficient = EXCLUDED.split_coefficient,
                            created_at = EXCLUDED.created_at
                    """)
                    
                    rows_affected = cursor.rowcount
                    
                    # Clean up temp table
                    cursor.execute("DROP TABLE temp_prices")
                    conn.commit()
                    
                    logger.info(f"Bulk upserted {rows_affected} price records")
                    return rows_affected
                    
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Bulk upsert failed: {e}")
                    raise
    
    def bulk_insert_with_execute_values(self, data: List[Dict], table_name: str,
                                       columns: List[str]) -> int:
        """
        Bulk insert using psycopg2's execute_values (faster than executemany)
        
        Args:
            data: List of dictionaries with data
            table_name: Target table name
            columns: Column names
            
        Returns:
            Number of records inserted
        """
        if not data:
            return 0
        
        # Convert to list of tuples
        values = [tuple(record.get(col) for col in columns) for record in data]
        
        query = f"""
            INSERT INTO {table_name} ({','.join(columns)})
            VALUES %s
            ON CONFLICT DO NOTHING
        """
        
        with self.get_raw_connection() as conn:
            with conn.cursor() as cursor:
                try:
                    execute_values(
                        cursor, query, values,
                        template=None,
                        page_size=self.page_size
                    )
                    conn.commit()
                    return len(values)
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Execute values failed: {e}")
                    raise
    
    def process_in_chunks(self, df: pd.DataFrame, processor_func, 
                         chunk_size: Optional[int] = None) -> int:
        """
        Process large DataFrame in memory-efficient chunks
        
        Args:
            df: Large DataFrame to process
            processor_func: Function to process each chunk
            chunk_size: Size of each chunk (defaults to self.batch_size)
            
        Returns:
            Total records processed
        """
        chunk_size = chunk_size or self.batch_size
        total_processed = 0
        
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size]
            try:
                processed = processor_func(chunk)
                total_processed += processed
                logger.debug(f"Processed chunk {i//chunk_size + 1}: {processed} records")
                
                # Free memory after processing chunk
                del chunk
                
            except Exception as e:
                logger.error(f"Failed processing chunk {i//chunk_size + 1}: {e}")
                raise
        
        return total_processed
    
    def stream_process_query(self, query: str, params: Dict = None, 
                            chunk_size: int = 10000) -> Iterator[pd.DataFrame]:
        """
        Stream large query results in chunks to avoid memory issues
        
        Args:
            query: SQL query to execute
            params: Query parameters
            chunk_size: Number of rows per chunk
            
        Yields:
            DataFrame chunks
        """
        with self.engine.connect() as conn:
            # Use server-side cursor for streaming
            conn = conn.execution_options(stream_results=True)
            result = conn.execute(text(query), params or {})
            
            while True:
                chunk = result.fetchmany(chunk_size)
                if not chunk:
                    break
                    
                df = pd.DataFrame(chunk)
                df.columns = result.keys()
                yield df
                
                # Explicitly delete to free memory
                del chunk
    
    def optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage by downcasting numeric types
        
        Args:
            df: DataFrame to optimize
            
        Returns:
            Memory-optimized DataFrame
        """
        # Downcast integers
        int_cols = df.select_dtypes(include=['int']).columns
        df[int_cols] = df[int_cols].apply(pd.to_numeric, downcast='integer')
        
        # Downcast floats
        float_cols = df.select_dtypes(include=['float']).columns
        df[float_cols] = df[float_cols].apply(pd.to_numeric, downcast='float')
        
        # Convert object columns to category if beneficial
        for col in df.select_dtypes(include=['object']).columns:
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            if num_unique_values / num_total_values < 0.5:
                df[col] = df[col].astype('category')
        
        return df
    
    def parallel_bulk_insert(self, dfs: List[pd.DataFrame], table_name: str,
                           max_workers: int = 4) -> int:
        """
        Parallel bulk insert for multiple DataFrames
        
        Args:
            dfs: List of DataFrames to insert
            table_name: Target table name
            max_workers: Number of parallel workers
            
        Returns:
            Total records inserted
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        total_inserted = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.bulk_insert_with_copy, df, table_name): i
                for i, df in enumerate(dfs)
            }
            
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    inserted = future.result()
                    total_inserted += inserted
                    logger.info(f"Batch {idx} inserted: {inserted} records")
                except Exception as e:
                    logger.error(f"Batch {idx} failed: {e}")
                    raise
        
        return total_inserted

# Singleton instance
batch_processor = BatchProcessor()

# Convenience functions
def bulk_insert_prices(df: pd.DataFrame) -> int:
    """Optimized bulk insert for price data"""
    return batch_processor.bulk_upsert_prices(df)

def process_large_dataframe(df: pd.DataFrame, processor_func, 
                           chunk_size: int = 10000) -> int:
    """Process large DataFrame in chunks"""
    return batch_processor.process_in_chunks(df, processor_func, chunk_size)