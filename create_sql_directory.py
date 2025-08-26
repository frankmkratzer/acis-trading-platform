#!/usr/bin/env python3
"""
Helper script to extract SQL components into separate files for better maintainability.
Run this after improving setup_schema.py to organize SQL code.
"""

import os
from pathlib import Path

def create_sql_directory():
    """Create SQL directory structure and extract SQL components."""
    sql_dir = Path(__file__).parent / 'sql'
    sql_dir.mkdir(exist_ok=True)
    
    print(f"Created SQL directory: {sql_dir}")
    
    # Create placeholder files
    files_to_create = [
        ('schema.sql', '-- Main database schema\n-- Tables, indexes, and constraints\n'),
        ('foreign_keys.sql', '-- Foreign key constraints\n-- Applied after initial data load\n'),
        ('materialized_views.sql', '-- Basic materialized views\n-- For performance optimization\n'),
        ('portfolio_views.sql', '-- Portfolio-specific views\n-- Requires AI scoring data\n'),
        ('migrations.sql', '-- Database migrations\n-- Version control for schema changes\n')
    ]
    
    for filename, header in files_to_create:
        file_path = sql_dir / filename
        if not file_path.exists():
            file_path.write_text(header)
            print(f"Created: {filename}")
        else:
            print(f"Exists: {filename}")
    
    print("\n💡 Next steps:")
    print("1. Extract SQL from setup_schema.py into these files")
    print("2. Add version control to your SQL schema")
    print("3. Consider using a proper migration tool like Alembic")

if __name__ == "__main__":
    create_sql_directory()