import sqlite3
import pandas as pd
from datetime import datetime
from typing import Dict, List, Set, Tuple
import os

class DatabaseConsolidationTool:
    """
    Database cleanup and consolidation tool.
    Identifies and removes redundant tables, consolidates similar tables,
    and optimizes database structure based on usage patterns.
    """
    
    def __init__(self, db_path: str = 'acis_trading.db'):
        self.db_path = db_path
        self.backup_db_path = f"acis_trading_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        
        # Table usage analysis from codebase analysis
        self.single_use_tables = [
            'admin_users', 'audit_log', 'last', 'portfolio_manager', 'manage_portfolios',
            'esg_score', 'dividend_growth_5y', 'cv_score', 'competitive_moat_score',
            'esg_momentum_score', 'stock_a', 'stock_b', 'stock_c', 'stock_d', 'stock_e',
            'price_move', 'price_change_3d', 'price_sma_20', 'price_change_1d',
            'price_volatility_20', 'price_range_20', 'score_range', 'component_scores',
            'breakout_score', 'price_change', 'volume_score', 'trading_test', 'stock',
            'calculate_portfolio_metrics', 'optimize_portfolio'
        ]
        
        # Core tables to keep (heavily used)
        self.core_tables = [
            'stock_prices', 'fundamentals', 'sp500_history', 'ai_value_scores',
            'ai_growth_scores', 'ai_dividend_scores', 'portfolios', 'portfolio_history',
            'strategy_performance', 'quarterly_returns', 'sector_allocations',
            'risk_metrics', 'benchmark_comparisons'
        ]
        
        # Tables that can be consolidated
        self.consolidation_candidates = {
            'ai_scores': ['ai_value_scores', 'ai_growth_scores', 'ai_dividend_scores'],
            'portfolio_data': ['portfolios', 'portfolio_history', 'portfolio_manager'],
            'performance_metrics': ['strategy_performance', 'risk_metrics', 'benchmark_comparisons'],
            'price_analytics': ['price_move', 'price_change_3d', 'price_sma_20', 'price_change_1d'],
            'temporary_calculations': ['stock_a', 'stock_b', 'stock_c', 'stock_d', 'stock_e']
        }
        
    def create_backup(self) -> bool:
        """Create full database backup before making changes."""
        
        try:
            print(f"[BACKUP] Creating database backup: {self.backup_db_path}")
            
            # Copy entire database
            with open(self.db_path, 'rb') as source:
                with open(self.backup_db_path, 'wb') as backup:
                    backup.write(source.read())
            
            print(f"[BACKUP] Backup created successfully")
            return True
            
        except Exception as e:
            print(f"[ERROR] Backup failed: {str(e)}")
            return False
    
    def analyze_table_usage(self) -> Dict:
        """Analyze current table usage and structure."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        all_tables = [row[0] for row in cursor.fetchall()]
        
        table_analysis = {
            'total_tables': len(all_tables),
            'table_info': {},
            'empty_tables': [],
            'small_tables': [],  # < 1000 rows
            'large_tables': [],  # > 100000 rows
            'cleanup_candidates': []
        }
        
        print(f"[ANALYSIS] Analyzing {len(all_tables)} database tables...")
        
        for table in all_tables:
            try:
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                row_count = cursor.fetchone()[0]
                
                # Get table schema
                cursor.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                column_count = len(columns)
                
                # Get table size estimate
                cursor.execute(f"SELECT pg_size_pretty(pg_total_relation_size('{table}'))")
                size_info = "N/A"  # SQLite doesn't have direct size info
                
                table_info = {
                    'row_count': row_count,
                    'column_count': column_count,
                    'columns': [col[1] for col in columns],
                    'size_estimate': size_info
                }
                
                table_analysis['table_info'][table] = table_info
                
                # Categorize tables
                if row_count == 0:
                    table_analysis['empty_tables'].append(table)
                    table_analysis['cleanup_candidates'].append(table)
                elif row_count < 1000:
                    table_analysis['small_tables'].append(table)
                    if table in self.single_use_tables:
                        table_analysis['cleanup_candidates'].append(table)
                elif row_count > 100000:
                    table_analysis['large_tables'].append(table)
                
                print(f"  {table:<30} | Rows: {row_count:>8,} | Cols: {column_count:>2}")
                
            except Exception as e:
                print(f"[WARNING] Error analyzing table {table}: {str(e)}")
                table_analysis['table_info'][table] = {'error': str(e)}
        
        conn.close()
        
        print(f"\n[ANALYSIS SUMMARY]")
        print(f"Empty tables: {len(table_analysis['empty_tables'])}")
        print(f"Small tables: {len(table_analysis['small_tables'])}")  
        print(f"Large tables: {len(table_analysis['large_tables'])}")
        print(f"Cleanup candidates: {len(table_analysis['cleanup_candidates'])}")
        
        return table_analysis
    
    def identify_duplicate_tables(self) -> Dict[str, List[str]]:
        """Identify tables with similar schemas that could be consolidated."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get schema for all tables
        table_schemas = {}
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        for table in tables:
            try:
                cursor.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                schema = tuple(sorted([(col[1], col[2]) for col in columns]))
                table_schemas[table] = schema
            except Exception as e:
                print(f"[WARNING] Error getting schema for {table}: {str(e)}")
        
        conn.close()
        
        # Group tables by similar schemas
        schema_groups = {}
        for table, schema in table_schemas.items():
            schema_key = str(schema)
            if schema_key not in schema_groups:
                schema_groups[schema_key] = []
            schema_groups[schema_key].append(table)
        
        # Find duplicate schemas (more than one table with same schema)
        duplicate_groups = {}
        group_id = 1
        
        for schema_key, tables in schema_groups.items():
            if len(tables) > 1:
                duplicate_groups[f"duplicate_group_{group_id}"] = tables
                group_id += 1
        
        print(f"[DUPLICATES] Found {len(duplicate_groups)} groups of tables with identical schemas")
        
        for group_name, tables in duplicate_groups.items():
            print(f"  {group_name}: {', '.join(tables)}")
        
        return duplicate_groups
    
    def create_consolidated_ai_scores_table(self) -> bool:
        """Consolidate AI score tables into single unified table."""
        
        print("[CONSOLIDATE] Creating unified AI scores table...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Create new consolidated table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_scores_unified (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    value_score REAL DEFAULT 50,
                    growth_score REAL DEFAULT 50,
                    dividend_score REAL DEFAULT 50,
                    momentum_score REAL DEFAULT 50,
                    quality_score REAL DEFAULT 50,
                    ensemble_score REAL DEFAULT 50,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, date)
                )
            """)
            
            print("  Created unified AI scores table structure")
            
            # Migrate data from existing AI score tables
            ai_score_tables = ['ai_value_scores', 'ai_growth_scores', 'ai_dividend_scores']
            
            for table in ai_score_tables:
                # Check if table exists
                cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
                if cursor.fetchone():
                    try:
                        # Get column mapping
                        if 'value' in table:
                            score_column = 'value_score'
                        elif 'growth' in table:
                            score_column = 'growth_score'
                        else:  # dividend
                            score_column = 'dividend_score'
                        
                        # Migrate data with UPSERT logic
                        cursor.execute(f"""
                            INSERT OR REPLACE INTO ai_scores_unified 
                                (symbol, date, {score_column})
                            SELECT 
                                symbol, 
                                date, 
                                COALESCE(score, 50) as score
                            FROM {table}
                            WHERE symbol IS NOT NULL AND date IS NOT NULL
                        """)
                        
                        rows_migrated = cursor.rowcount
                        print(f"  Migrated {rows_migrated:,} rows from {table}")
                        
                    except Exception as e:
                        print(f"  [WARNING] Error migrating {table}: {str(e)}")
            
            # Create index for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ai_scores_symbol_date ON ai_scores_unified(symbol, date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ai_scores_date ON ai_scores_unified(date)")
            
            conn.commit()
            print("  Unified AI scores table created successfully")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to create unified AI scores table: {str(e)}")
            conn.rollback()
            return False
        
        finally:
            conn.close()
    
    def create_consolidated_portfolio_table(self) -> bool:
        """Consolidate portfolio-related tables."""
        
        print("[CONSOLIDATE] Creating unified portfolio table...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Create comprehensive portfolio table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_unified (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    portfolio_name TEXT NOT NULL,
                    strategy_type TEXT NOT NULL,
                    date TEXT NOT NULL,
                    symbol TEXT,
                    shares REAL,
                    price REAL,
                    market_value REAL,
                    weight REAL,
                    sector TEXT,
                    score REAL,
                    return_1d REAL,
                    return_1w REAL,
                    return_1m REAL,
                    return_3m REAL,
                    return_ytd REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX(portfolio_name, date),
                    INDEX(date, strategy_type),
                    INDEX(symbol, date)
                )
            """)
            
            print("  Created unified portfolio table structure")
            
            # Migrate from existing portfolio tables
            portfolio_tables = ['portfolios', 'portfolio_history']
            
            for table in portfolio_tables:
                cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
                if cursor.fetchone():
                    try:
                        # Get table structure to understand columns
                        cursor.execute(f"PRAGMA table_info({table})")
                        columns = [col[1] for col in cursor.fetchall()]
                        
                        # Build dynamic insert based on available columns
                        common_columns = []
                        if 'portfolio_name' in columns: common_columns.append('portfolio_name')
                        elif 'strategy' in columns: common_columns.append('strategy as portfolio_name')
                        
                        if 'date' in columns: common_columns.append('date')
                        if 'symbol' in columns: common_columns.append('symbol')
                        if 'shares' in columns: common_columns.append('shares')
                        if 'price' in columns: common_columns.append('price')
                        if 'market_value' in columns: common_columns.append('market_value')
                        
                        if common_columns:
                            select_clause = ', '.join(common_columns)
                            cursor.execute(f"""
                                INSERT OR IGNORE INTO portfolio_unified 
                                    ({', '.join([col.split(' as ')[0] if ' as ' in col else col for col in common_columns])})
                                SELECT {select_clause}
                                FROM {table}
                            """)
                            
                            rows_migrated = cursor.rowcount
                            print(f"  Migrated {rows_migrated:,} rows from {table}")
                        
                    except Exception as e:
                        print(f"  [WARNING] Error migrating {table}: {str(e)}")
            
            conn.commit()
            print("  Unified portfolio table created successfully")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to create unified portfolio table: {str(e)}")
            conn.rollback()
            return False
        
        finally:
            conn.close()
    
    def drop_redundant_tables(self, tables_to_drop: List[str], confirm: bool = False) -> Dict:
        """Drop redundant tables after consolidation."""
        
        if not confirm:
            print("[WARNING] This will permanently delete tables. Use confirm=True to proceed.")
            return {'status': 'cancelled', 'message': 'Confirmation required'}
        
        print(f"[CLEANUP] Dropping {len(tables_to_drop)} redundant tables...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        results = {
            'dropped_successfully': [],
            'failed_to_drop': [],
            'total_space_freed': 0
        }
        
        for table in tables_to_drop:
            try:
                # Get row count before dropping
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                row_count = cursor.fetchone()[0]
                
                # Drop the table
                cursor.execute(f"DROP TABLE IF EXISTS {table}")
                
                results['dropped_successfully'].append({
                    'table': table,
                    'rows_deleted': row_count
                })
                
                print(f"  Dropped {table} ({row_count:,} rows)")
                
            except Exception as e:
                print(f"  [ERROR] Failed to drop {table}: {str(e)}")
                results['failed_to_drop'].append({
                    'table': table,
                    'error': str(e)
                })
        
        conn.commit()
        conn.close()
        
        print(f"[CLEANUP] Successfully dropped {len(results['dropped_successfully'])} tables")
        return results
    
    def optimize_database(self) -> Dict:
        """Optimize database by running VACUUM and updating statistics."""
        
        print("[OPTIMIZE] Running database optimization...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get database size before optimization
            cursor.execute("PRAGMA page_count")
            pages_before = cursor.fetchone()[0]
            cursor.execute("PRAGMA page_size")
            page_size = cursor.fetchone()[0]
            size_before = pages_before * page_size
            
            print(f"  Database size before: {size_before / (1024*1024):.1f} MB")
            
            # Run VACUUM to reclaim space
            cursor.execute("VACUUM")
            print("  Completed VACUUM operation")
            
            # Update table statistics
            cursor.execute("ANALYZE")
            print("  Updated table statistics")
            
            # Get database size after optimization
            cursor.execute("PRAGMA page_count")
            pages_after = cursor.fetchone()[0]
            size_after = pages_after * page_size
            
            space_saved = size_before - size_after
            space_saved_mb = space_saved / (1024*1024)
            
            print(f"  Database size after: {size_after / (1024*1024):.1f} MB")
            print(f"  Space reclaimed: {space_saved_mb:.1f} MB")
            
            return {
                'size_before': size_before,
                'size_after': size_after,
                'space_saved': space_saved,
                'optimization_successful': True
            }
            
        except Exception as e:
            print(f"[ERROR] Database optimization failed: {str(e)}")
            return {
                'optimization_successful': False,
                'error': str(e)
            }
        
        finally:
            conn.close()
    
    def generate_cleanup_report(self) -> Dict:
        """Generate comprehensive cleanup report."""
        
        print("[REPORT] Generating database cleanup report...")
        
        # Analyze current state
        analysis = self.analyze_table_usage()
        duplicates = self.identify_duplicate_tables()
        
        # Calculate cleanup recommendations
        cleanup_recommendations = {
            'immediate_deletion': [],
            'consolidation_candidates': [],
            'archive_candidates': [],
            'optimization_needed': []
        }
        
        # Immediate deletion candidates (empty or single-use tables)
        for table in analysis['empty_tables']:
            cleanup_recommendations['immediate_deletion'].append({
                'table': table,
                'reason': 'Empty table - no data',
                'action': 'DROP TABLE'
            })
        
        for table in self.single_use_tables:
            if table in analysis['table_info']:
                info = analysis['table_info'][table]
                if info.get('row_count', 0) < 100:
                    cleanup_recommendations['immediate_deletion'].append({
                        'table': table,
                        'reason': f'Single-use table with only {info.get("row_count", 0)} rows',
                        'action': 'DROP TABLE'
                    })
        
        # Consolidation candidates
        for group_name, tables in self.consolidation_candidates.items():
            existing_tables = [t for t in tables if t in analysis['table_info']]
            if len(existing_tables) > 1:
                cleanup_recommendations['consolidation_candidates'].append({
                    'consolidated_name': group_name,
                    'source_tables': existing_tables,
                    'action': 'CONSOLIDATE'
                })
        
        # Archive candidates (large historical tables)
        for table in analysis['large_tables']:
            if 'historical' in table.lower() or 'archive' in table.lower():
                cleanup_recommendations['archive_candidates'].append({
                    'table': table,
                    'reason': 'Large historical table',
                    'action': 'ARCHIVE TO SEPARATE DB'
                })
        
        report = {
            'analysis_date': datetime.now().isoformat(),
            'current_state': analysis,
            'duplicate_groups': duplicates,
            'cleanup_recommendations': cleanup_recommendations,
            'estimated_space_savings': self._estimate_space_savings(cleanup_recommendations, analysis),
            'backup_location': self.backup_db_path
        }
        
        self._print_cleanup_report(report)
        return report
    
    def _estimate_space_savings(self, recommendations: Dict, analysis: Dict) -> Dict:
        """Estimate potential space savings from cleanup operations."""
        
        tables_to_delete = []
        
        # Count immediate deletion candidates
        for item in recommendations['immediate_deletion']:
            tables_to_delete.append(item['table'])
        
        # Count consolidated tables (source tables will be dropped)
        for item in recommendations['consolidation_candidates']:
            tables_to_delete.extend(item['source_tables'])
        
        # Estimate rows and space
        total_rows_to_delete = 0
        for table in tables_to_delete:
            if table in analysis['table_info']:
                total_rows_to_delete += analysis['table_info'][table].get('row_count', 0)
        
        # Rough estimate: 100 bytes per row average
        estimated_space_mb = (total_rows_to_delete * 100) / (1024 * 1024)
        
        return {
            'tables_to_remove': len(tables_to_delete),
            'rows_to_delete': total_rows_to_delete,
            'estimated_space_savings_mb': estimated_space_mb,
            'current_table_count': analysis['total_tables'],
            'final_table_count': analysis['total_tables'] - len(tables_to_delete)
        }
    
    def _print_cleanup_report(self, report: Dict):
        """Print formatted cleanup report."""
        
        print("\n" + "=" * 80)
        print("[DATABASE CLEANUP REPORT]")
        print("=" * 80)
        
        current = report['current_state']
        recommendations = report['cleanup_recommendations']
        savings = report['estimated_space_savings']
        
        print(f"Analysis Date: {report['analysis_date']}")
        print(f"Total Tables: {current['total_tables']}")
        print(f"Empty Tables: {len(current['empty_tables'])}")
        print(f"Cleanup Candidates: {len(current['cleanup_candidates'])}")
        
        print(f"\nIMMEDIATE DELETION CANDIDATES: {len(recommendations['immediate_deletion'])}")
        for item in recommendations['immediate_deletion'][:10]:  # Show first 10
            print(f"  - {item['table']}: {item['reason']}")
        if len(recommendations['immediate_deletion']) > 10:
            print(f"  ... and {len(recommendations['immediate_deletion']) - 10} more")
        
        print(f"\nCONSOLIDATION OPPORTUNITIES: {len(recommendations['consolidation_candidates'])}")
        for item in recommendations['consolidation_candidates']:
            print(f"  - {item['consolidated_name']}: {len(item['source_tables'])} tables")
        
        print(f"\nESTIMATED SPACE SAVINGS:")
        print(f"  Tables to remove: {savings['tables_to_remove']}")
        print(f"  Rows to delete: {savings['rows_to_delete']:,}")
        print(f"  Space savings: ~{savings['estimated_space_savings_mb']:.1f} MB")
        print(f"  Final table count: {savings['final_table_count']} (from {savings['current_table_count']})")
        
        print("=" * 80)
    
    def execute_cleanup_plan(self, confirm_dangerous_operations: bool = False) -> Dict:
        """Execute comprehensive cleanup plan."""
        
        if not confirm_dangerous_operations:
            print("[WARNING] This will make permanent changes. Use confirm_dangerous_operations=True")
            return {'status': 'cancelled', 'message': 'Confirmation required'}
        
        print("[EXECUTE] Starting comprehensive database cleanup...")
        
        # Step 1: Create backup
        if not self.create_backup():
            return {'status': 'failed', 'step': 'backup', 'message': 'Backup failed'}
        
        results = {
            'backup_created': True,
            'backup_location': self.backup_db_path,
            'consolidation_results': {},
            'cleanup_results': {},
            'optimization_results': {}
        }
        
        try:
            # Step 2: Create consolidated tables
            print("\n[STEP 2] Creating consolidated tables...")
            results['consolidation_results']['ai_scores'] = self.create_consolidated_ai_scores_table()
            results['consolidation_results']['portfolios'] = self.create_consolidated_portfolio_table()
            
            # Step 3: Drop redundant tables
            print("\n[STEP 3] Dropping redundant tables...")
            tables_to_drop = self.single_use_tables[:20]  # Limit for safety
            results['cleanup_results'] = self.drop_redundant_tables(tables_to_drop, confirm=True)
            
            # Step 4: Optimize database
            print("\n[STEP 4] Optimizing database...")
            results['optimization_results'] = self.optimize_database()
            
            print("\n[SUCCESS] Database cleanup completed successfully!")
            return results
            
        except Exception as e:
            print(f"[ERROR] Cleanup failed: {str(e)}")
            return {'status': 'failed', 'error': str(e), 'partial_results': results}


def main():
    """Demonstrate database consolidation tool."""
    
    print("[LAUNCH] ACIS Database Consolidation Tool")
    print("Cleaning up redundant tables and optimizing database structure")
    print("=" * 80)
    
    # Initialize tool
    consolidator = DatabaseConsolidationTool()
    
    print("\n[ANALYSIS] Analyzing current database structure...")
    
    # Generate cleanup report
    report = consolidator.generate_cleanup_report()
    
    print(f"\n[SUMMARY]")
    print(f"Cleanup recommended for {report['estimated_space_savings']['tables_to_remove']} tables")
    print(f"Potential space savings: ~{report['estimated_space_savings']['estimated_space_savings_mb']:.1f} MB")
    print(f"Would reduce table count from {report['current_state']['total_tables']} to {report['estimated_space_savings']['final_table_count']}")
    
    print(f"\n[READY] Database consolidation plan prepared")
    print("To execute: consolidator.execute_cleanup_plan(confirm_dangerous_operations=True)")
    
    return consolidator


if __name__ == "__main__":
    main()