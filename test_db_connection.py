#!/usr/bin/env python3
"""
Test database connection to DigitalOcean PostgreSQL
"""
import os
import sys
from dotenv import load_dotenv
import psycopg2
from sqlalchemy import create_engine, text
import pandas as pd

# Load environment variables
load_dotenv()


def test_psycopg2_connection():
    """Test raw psycopg2 connection"""
    try:
        # Parse the connection string
        db_url = os.getenv("POSTGRES_URL")

        # Extract connection parameters (alternative method)
        import urllib.parse
        result = urllib.parse.urlparse(db_url)

        conn_params = {
            'host': result.hostname,
            'port': result.port,
            'database': result.path[1:],  # Remove leading slash
            'user': result.username,
            'password': result.password,
            'sslmode': 'require',
            'connect_timeout': 30
        }

        print("Testing psycopg2 connection...")
        print(f"Host: {conn_params['host']}")
        print(f"Port: {conn_params['port']}")
        print(f"Database: {conn_params['database']}")

        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor()

        # Test query
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"✅ PostgreSQL version: {version[0]}")

        # Test basic operations
        cursor.execute("SELECT current_database(), current_user;")
        db_info = cursor.fetchone()
        print(f"✅ Connected to database: {db_info[0]} as user: {db_info[1]}")

        cursor.close()
        conn.close()
        return True

    except Exception as e:
        print(f"❌ psycopg2 connection failed: {str(e)}")
        return False


def test_sqlalchemy_connection():
    """Test SQLAlchemy connection"""
    try:
        db_url = os.getenv("POSTGRES_URL")
        print(f"\nTesting SQLAlchemy connection...")

        engine = create_engine(db_url, pool_pre_ping=True)

        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1 as test"))
            print(f"✅ SQLAlchemy connection successful: {result.fetchone()}")

            # List existing tables
            result = conn.execute(text("""
                                       SELECT table_name
                                       FROM information_schema.tables
                                       WHERE table_schema = 'public'
                                       ORDER BY table_name;
                                       """))

            tables = result.fetchall()
            if tables:
                print(f"✅ Found {len(tables)} tables:")
                for table in tables[:10]:  # Show first 10 tables
                    print(f"  - {table[0]}")
                if len(tables) > 10:
                    print(f"  ... and {len(tables) - 10} more")
            else:
                print("ℹ️  No tables found in public schema")

        return True

    except Exception as e:
        print(f"❌ SQLAlchemy connection failed: {str(e)}")
        return False


def test_pandas_connection():
    """Test pandas database operations"""
    try:
        db_url = os.getenv("POSTGRES_URL")
        print(f"\nTesting pandas database operations...")

        # Simple query with pandas
        df = pd.read_sql("SELECT current_timestamp as now", db_url)
        print(f"✅ Pandas query successful: {df.iloc[0, 0]}")

        return True

    except Exception as e:
        print(f"❌ Pandas connection failed: {str(e)}")
        return False


def create_test_table():
    """Create a test table for your trading app"""
    try:
        db_url = os.getenv("POSTGRES_URL")
        engine = create_engine(db_url)

        print(f"\nCreating test table...")

        with engine.connect() as conn:
            # Create a simple test table
            conn.execute(text("""
                              CREATE TABLE IF NOT EXISTS trading_test
                              (
                                  id
                                  SERIAL
                                  PRIMARY
                                  KEY,
                                  symbol
                                  VARCHAR
                              (
                                  10
                              ) NOT NULL,
                                  price DECIMAL
                              (
                                  10,
                                  2
                              ),
                                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                                  );
                              """))

            # Insert test data
            conn.execute(text("""
                              INSERT INTO trading_test (symbol, price)
                              VALUES ('AAPL', 150.25),
                                     ('MSFT', 280.50),
                                     ('GOOGL', 120.75) ON CONFLICT DO NOTHING;
                              """))

            conn.commit()

            # Query test data
            result = conn.execute(text("SELECT * FROM trading_test LIMIT 5;"))
            rows = result.fetchall()

            print(f"✅ Test table created and populated with {len(rows)} rows")
            for row in rows:
                print(f"  {row}")

        return True

    except Exception as e:
        print(f"❌ Test table creation failed: {str(e)}")
        return False


if __name__ == "__main__":
    print("🧪 Testing Database Connections")
    print("=" * 50)

    # Check if environment variables are loaded
    if not os.getenv("POSTGRES_URL"):
        print("❌ POSTGRES_URL not found in environment variables")
        print("Make sure your .env file is in the current directory")
        sys.exit(1)

    # Run all tests
    tests_passed = 0
    total_tests = 4

    if test_psycopg2_connection():
        tests_passed += 1

    if test_sqlalchemy_connection():
        tests_passed += 1

    if test_pandas_connection():
        tests_passed += 1

    if create_test_table():
        tests_passed += 1

    print(f"\n{'=' * 50}")
    print(f"Results: {tests_passed}/{total_tests} tests passed")

    if tests_passed == total_tests:
        print("🎉 All database tests passed! Your connection is working.")
        print("\nNext steps:")
        print("1. Build Docker image: docker-compose build")
        print("2. Start services: docker-compose up")
        print("3. Test your trading scripts inside the container")
    else:
        print("⚠️  Some tests failed. Check your database configuration.")
        sys.exit(1)