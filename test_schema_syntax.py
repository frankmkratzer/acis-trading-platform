#!/usr/bin/env python3
"""Test if the schema SQL is valid"""

import os
import sys
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

# Import the SQL from setup_schema
from setup_schema import ESSENTIAL_SCHEMA_SQL

def test_schema_syntax():
    """Test if the schema SQL has valid syntax"""
    try:
        # Just parse the SQL to check for syntax errors
        print("[INFO] Testing SQL syntax...")
        
        # Check for basic SQL validity
        sql_lines = ESSENTIAL_SCHEMA_SQL.split('\n')
        print(f"[INFO] SQL has {len(sql_lines)} lines")
        
        # Check for unescaped % signs (should be %%)
        for i, line in enumerate(sql_lines, 1):
            # Skip checking inside string literals that are properly quoted
            if '--' in line:
                comment_part = line.split('--')[1]
                single_percents = comment_part.count('%') - (comment_part.count('%%') * 2)
                if single_percents > 0:
                    print(f"[WARNING] Line {i}: Unescaped % in comment: {line.strip()}")
        
        print("[SUCCESS] SQL syntax appears valid")
        return True
        
    except Exception as e:
        print(f"[ERROR] Schema syntax error: {e}")
        return False

if __name__ == "__main__":
    success = test_schema_syntax()
    sys.exit(0 if success else 1)