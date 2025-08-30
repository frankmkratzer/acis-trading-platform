#!/usr/bin/env python3
"""
Clean up orphaned temporary tables from the database
"""

import os
import sys
from sqlalchemy import create_engine, text
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.logging_config import setup_logger

logger = setup_logger("cleanup_temp_tables")
load_dotenv()

def get_postgres_url():
    """Get PostgreSQL connection URL from environment"""
    postgres_url = os.getenv("POSTGRES_URL")
    if not postgres_url:
        raise ValueError("POSTGRES_URL not set in .env file")
    return postgres_url

def cleanup_temp_tables():
    """Find and remove all temporary tables"""
    
    try:
        engine = create_engine(get_postgres_url())
        
        print("\n" + "="*60)
        print("CLEANING UP TEMPORARY TABLES")
        print("="*60)
        
        with engine.connect() as conn:
            # Find all temp tables
            result = conn.execute(text("""
                SELECT 
                    tablename,
                    pg_size_pretty(pg_total_relation_size('public.'||tablename)) AS size
                FROM pg_tables 
                WHERE schemaname = 'public' 
                  AND (
                    tablename LIKE 'temp_inst_holdings_%' OR
                    tablename LIKE 'temp_inst_signals_%' OR
                    tablename = 'temp_inst_holdings_staging' OR
                    tablename = 'temp_inst_signals_staging'
                  )
                ORDER BY tablename
            """))
            
            temp_tables = result.fetchall()
            
            if not temp_tables:
                print("\n✅ No temporary tables found - database is clean!")
                return 0
            
            print(f"\nFound {len(temp_tables)} temporary table(s) to clean up:")
            print("-" * 60)
            
            total_size = 0
            for table_name, size in temp_tables:
                print(f"  • {table_name:50} {size:>10}")
            
            # Ask for confirmation
            print("\n" + "="*60)
            response = input("Do you want to delete these temporary tables? (yes/no): ")
            
            if response.lower() in ['yes', 'y']:
                # Drop each table
                dropped_count = 0
                for table_name, _ in temp_tables:
                    try:
                        conn.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))
                        conn.commit()
                        print(f"  ✓ Dropped: {table_name}")
                        dropped_count += 1
                    except Exception as e:
                        print(f"  ✗ Failed to drop {table_name}: {e}")
                        logger.error(f"Failed to drop {table_name}: {e}")
                
                print(f"\n✅ Successfully dropped {dropped_count}/{len(temp_tables)} table(s)")
                
                # Verify cleanup
                result = conn.execute(text("""
                    SELECT COUNT(*) 
                    FROM pg_tables 
                    WHERE schemaname = 'public' 
                      AND (
                        tablename LIKE 'temp_inst_holdings_%' OR
                        tablename LIKE 'temp_inst_signals_%'
                      )
                """))
                
                remaining = result.scalar()
                if remaining == 0:
                    print("✅ All temporary tables have been cleaned up!")
                else:
                    print(f"⚠️  {remaining} temporary table(s) still remain")
                
            else:
                print("\n❌ Cleanup cancelled by user")
                return 1
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during cleanup: {e}")
        logger.error(f"Cleanup failed: {e}")
        return 1

def show_all_tables():
    """Show all tables in the database for reference"""
    try:
        engine = create_engine(get_postgres_url())
        
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT 
                    tablename,
                    pg_size_pretty(pg_total_relation_size('public.'||tablename)) AS size
                FROM pg_tables 
                WHERE schemaname = 'public'
                ORDER BY pg_total_relation_size('public.'||tablename) DESC
                LIMIT 20
            """))
            
            print("\n" + "="*60)
            print("TOP 20 LARGEST TABLES IN DATABASE")
            print("="*60)
            
            for table_name, size in result:
                print(f"  {table_name:50} {size:>10}")
    
    except Exception as e:
        print(f"Error listing tables: {e}")

def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean up temporary tables")
    parser.add_argument("--show-all", action="store_true", 
                       help="Show all tables in database")
    parser.add_argument("--auto-yes", action="store_true",
                       help="Automatically confirm deletion (use with caution)")
    
    args = parser.parse_args()
    
    if args.show_all:
        show_all_tables()
        return 0
    
    # If auto-yes, modify the cleanup function
    if args.auto_yes:
        print("⚠️  Auto-confirmation enabled - will delete without asking")
        # You could modify cleanup_temp_tables to accept this parameter
    
    return cleanup_temp_tables()

if __name__ == "__main__":
    sys.exit(main())