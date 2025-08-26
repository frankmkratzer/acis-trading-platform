#!/usr/bin/env python3
"""
Debug script to test database connection and identify issues.
"""

import os
import sys
from pathlib import Path

# Try to load dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[OK] Loaded .env file")
except ImportError:
    print("[WARN] python-dotenv not installed, using system env")

# Check environment variable
postgres_url = os.getenv("POSTGRES_URL")
print(f"POSTGRES_URL: {postgres_url[:50]}..." if postgres_url else "[ERROR] POSTGRES_URL not found")

if not postgres_url:
    print("[ERROR] Missing POSTGRES_URL environment variable")
    sys.exit(1)

# Test SQLAlchemy import
try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.exc import SQLAlchemyError
    print("[OK] SQLAlchemy imported successfully")
except ImportError as e:
    print(f"[ERROR] SQLAlchemy import failed: {e}")
    print("Try: pip install sqlalchemy psycopg2-binary")
    sys.exit(1)

# Test connection
try:
    print("\n[TEST] Testing database connection...")
    engine = create_engine(postgres_url, echo=False)  # Disable verbose logging initially
    
    with engine.begin() as conn:
        result = conn.execute(text("SELECT version(), current_database(), current_user"))
        row = result.fetchone()
        print(f"[OK] Connected successfully!")
        print(f"   Database: {row[1]}")
        print(f"   User: {row[2]}")
        print(f"   Version: {row[0][:100]}...")
        
        # Test simple table creation
        print("\n[TEST] Testing table creation...")
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS test_table (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """))
        print("[OK] Test table created")
        
        # Test insert
        conn.execute(text("INSERT INTO test_table (name) VALUES ('test')"))
        print("[OK] Test insert successful")
        
        # Clean up
        conn.execute(text("DROP TABLE test_table"))
        print("[OK] Test cleanup completed")
        
except Exception as e:
    print(f"[ERROR] Database connection/operation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[SUCCESS] Database connection test completed successfully!")