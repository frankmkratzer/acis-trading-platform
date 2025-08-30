#!/usr/bin/env python3
"""
Database Connection Manager with Connection Pooling
Optimizes database connections for better performance and resource management
"""

import os
import logging
from contextlib import contextmanager
from typing import Optional, Dict, Any
from sqlalchemy import create_engine, Engine, pool, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool, QueuePool
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class DatabaseConnectionManager:
    """Manages database connections with optimized pooling and monitoring"""
    
    _instance: Optional['DatabaseConnectionManager'] = None
    _engine: Optional[Engine] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.pool_size = int(os.getenv('DB_POOL_SIZE', '20'))
            self.max_overflow = int(os.getenv('DB_MAX_OVERFLOW', '40'))
            self.pool_timeout = int(os.getenv('DB_POOL_TIMEOUT', '30'))
            self.pool_recycle = int(os.getenv('DB_POOL_RECYCLE', '3600'))
            self.echo_pool = os.getenv('DB_ECHO_POOL', 'false').lower() == 'true'
            
    def get_engine(self, database_url: Optional[str] = None) -> Engine:
        """Get or create a database engine with connection pooling"""
        if self._engine is None:
            if database_url is None:
                database_url = os.getenv("POSTGRES_URL") or os.getenv("POSTGRES_DATABASE_URL")
                if not database_url:
                    raise ValueError("Database URL not provided and POSTGRES_URL not set")
            
            # Create engine with optimized pooling
            self._engine = create_engine(
                database_url,
                poolclass=QueuePool,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_timeout=self.pool_timeout,
                pool_recycle=self.pool_recycle,
                pool_pre_ping=True,  # Verify connections before using
                echo_pool=self.echo_pool,  # Log pool checkouts/checkins
                connect_args={
                    'connect_timeout': 30,
                    'options': '-c statement_timeout=300000',  # 5 minute statement timeout
                    'keepalives': 1,
                    'keepalives_idle': 30,
                    'keepalives_interval': 10,
                    'keepalives_count': 5
                }
            )
            
            # Add event listeners for monitoring
            @event.listens_for(self._engine, "connect")
            def receive_connect(dbapi_conn, connection_record):
                connection_record.info['pid'] = os.getpid()
                logger.debug(f"Connection checked out by PID {os.getpid()}")
            
            @event.listens_for(self._engine, "checkout")
            def receive_checkout(dbapi_conn, connection_record, connection_proxy):
                pid = os.getpid()
                if connection_record.info['pid'] != pid:
                    connection_record.connection = connection_proxy.connection = None
                    raise ConnectionError(
                        f"Connection record belongs to PID {connection_record.info['pid']}, "
                        f"attempting to check out in PID {pid}"
                    )
            
            # Add performance monitoring
            @event.listens_for(self._engine, "before_cursor_execute")
            def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
                conn.info.setdefault('query_start_time', []).append(time.time())
                logger.debug("Query started: %s", statement[:100])
            
            @event.listens_for(self._engine, "after_cursor_execute")
            def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
                total = time.time() - conn.info['query_start_time'].pop(-1)
                if total > 1.0:  # Log slow queries
                    logger.warning(f"Slow query ({total:.2f}s): {statement[:100]}")
                    
        return self._engine
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        engine = self.get_engine()
        connection = engine.connect()
        try:
            yield connection
        finally:
            connection.close()
    
    @contextmanager
    def get_session(self) -> Session:
        """Context manager for database sessions with automatic rollback on error"""
        engine = self.get_engine()
        SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
        session = SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def execute_with_retry(self, query, params=None, max_retries=3):
        """Execute query with automatic retry on connection errors"""
        import time
        from sqlalchemy import text
        
        last_exception = None
        for attempt in range(max_retries):
            try:
                with self.get_connection() as conn:
                    if params:
                        result = conn.execute(text(query), params)
                    else:
                        result = conn.execute(text(query))
                    return result
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Query failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Query failed after {max_retries} attempts: {e}")
        
        raise last_exception
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get current pool status for monitoring"""
        if self._engine is None:
            return {"status": "Not initialized"}
        
        pool = self._engine.pool
        return {
            "size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "total": pool.checkedout() + pool.checkedin()
        }
    
    def close(self):
        """Close all connections and dispose of the engine"""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            logger.info("Database engine disposed")

# Singleton instance
db_manager = DatabaseConnectionManager()

# Convenience functions
def get_db_engine() -> Engine:
    """Get the database engine with connection pooling"""
    return db_manager.get_engine()

def get_db_connection():
    """Get a database connection context manager"""
    return db_manager.get_connection()

def get_db_session():
    """Get a database session context manager"""
    return db_manager.get_session()

# Import time for monitoring
import time