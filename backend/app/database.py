"""
TruthGuard Database Configuration
SQLAlchemy setup and database utilities
"""

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import MetaData
import logging
from typing import AsyncGenerator

from .config import settings

logger = logging.getLogger(__name__)

# Database engine
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DATABASE_ECHO,
    pool_size=settings.DATABASE_POOL_SIZE,
    max_overflow=settings.DATABASE_MAX_OVERFLOW,
    pool_pre_ping=True,
    pool_recycle=3600,
)

# Session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Metadata convention for naming constraints
convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s"
}

metadata = MetaData(naming_convention=convention)

class Base(DeclarativeBase):
    """Base class for all database models"""
    metadata = metadata

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency function to get database session
    """
    async with async_session_maker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

async def init_db():
    """Initialize database tables"""
    try:
        async with engine.begin() as conn:
            # Import all models to ensure they're registered
            from .models import claim, verification, user
            
            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")
            
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise

async def close_db():
    """Close database connections"""
    try:
        await engine.dispose()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Error closing database: {str(e)}")

class DatabaseManager:
    """Database manager class for advanced operations"""
    
    def __init__(self):
        self.engine = engine
        self.session_maker = async_session_maker
    
    async def create_session(self) -> AsyncSession:
        """Create a new database session"""
        return self.session_maker()
    
    async def health_check(self) -> bool:
        """Check database connectivity"""
        try:
            async with self.engine.begin() as conn:
                await conn.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            return False
    
    async def execute_raw_sql(self, query: str, params: dict = None):
        """Execute raw SQL query"""
        async with self.engine.begin() as conn:
            result = await conn.execute(query, params or {})
            return result
    
    async def backup_database(self, backup_path: str):
        """Create database backup (implementation depends on database type)"""
        # This would be implemented based on the specific database
        logger.info(f"Database backup requested to: {backup_path}")
        pass
    
    async def get_table_stats(self) -> dict:
        """Get statistics about database tables"""
        stats = {}
        try:
            async with self.engine.begin() as conn:
                # This would need to be adapted based on the database type
                result = await conn.execute("""
                    SELECT 
                        schemaname,
                        tablename,
                        n_tup_ins as inserts,
                        n_tup_upd as updates,
                        n_tup_del as deletes
                    FROM pg_stat_user_tables
                """)
                
                for row in result:
                    stats[row.tablename] = {
                        'schema': row.schemaname,
                        'inserts': row.inserts,
                        'updates': row.updates,
                        'deletes': row.deletes
                    }
        except Exception as e:
            logger.warning(f"Could not get table stats: {str(e)}")
        
        return stats

# Global database manager instance
db_manager = DatabaseManager()

# Database event listeners
from sqlalchemy import event

@event.listens_for(engine.sync_engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Set SQLite pragmas if using SQLite"""
    if "sqlite" in settings.DATABASE_URL:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

@event.listens_for(engine.sync_engine, "before_cursor_execute")
def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    """Log SQL queries in debug mode"""
    if settings.DEBUG:
        logger.debug(f"SQL: {statement}")
        if parameters:
            logger.debug(f"Parameters: {parameters}")