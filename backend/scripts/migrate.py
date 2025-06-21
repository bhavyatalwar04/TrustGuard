#!/usr/bin/env python3
"""
Database migration script for TruthGuard application.
Handles schema changes and data migrations.
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime
import json

# Add the parent directory to Python path to import app modules
sys.path.append(str(Path(__file__).parent.parent))

from app.database import engine, get_database
from sqlalchemy import text
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Migration version tracking
MIGRATIONS_TABLE = "migration_history"

class Migration:
    """Base migration class."""
    
    def __init__(self, version: str, description: str):
        self.version = version
        self.description = description
        self.timestamp = datetime.utcnow()
    
    async def up(self, conn):
        """Apply the migration."""
        raise NotImplementedError
    
    async def down(self, conn):
        """Rollback the migration."""
        raise NotImplementedError

class Migration_001_InitialSchema(Migration):
    """Initial schema migration."""
    
    def __init__(self):
        super().__init__("001", "Initial schema creation")
    
    async def up(self, conn):
        """Create initial tables."""
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email VARCHAR(255) UNIQUE NOT NULL,
                username VARCHAR(100) UNIQUE NOT NULL,
                full_name VARCHAR(255),
                hashed_password VARCHAR(255),
                is_active BOOLEAN DEFAULT TRUE,
                role VARCHAR(50) DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS claims (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                source_url VARCHAR(500),
                source_type VARCHAR(100),
                language VARCHAR(10) DEFAULT 'en',
                status VARCHAR(50) DEFAULT 'pending',
                priority_score FLOAT DEFAULT 0.0,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS verification_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_id INTEGER NOT NULL,
                verdict VARCHAR(50) NOT NULL,
                confidence_score FLOAT NOT NULL,
                explanation TEXT,
                sources JSON,
                evidence_quality FLOAT,
                credibility_indicators JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (claim_id) REFERENCES claims (id)
            )
        """))

class Migration_002_AddKnowledgeGraph(Migration):
    """Add knowledge graph tables."""
    
    def __init__(self):
        super().__init__("002", "Add knowledge graph support")
    
    async def up(self, conn):
        """Add knowledge graph tables."""
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name VARCHAR(255) NOT NULL,
                type VARCHAR(100) NOT NULL,
                description TEXT,
                confidence FLOAT DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id INTEGER NOT NULL,
                predicate VARCHAR(255) NOT NULL,
                object_id INTEGER NOT NULL,
                confidence FLOAT DEFAULT 1.0,
                source VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (subject_id) REFERENCES entities (id),
                FOREIGN KEY (object_id) REFERENCES entities (id)
            )
        """))
        
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS claim_entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_id INTEGER NOT NULL,
                entity_id INTEGER NOT NULL,
                relevance_score FLOAT DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (claim_id) REFERENCES claims (id),
                FOREIGN KEY (entity_id) REFERENCES entities (id)
            )
        """))

class Migration_003_AddTrendTracking(Migration):
    """Add trend tracking capabilities."""
    
    def __init__(self):
        super().__init__("003", "Add trend tracking tables")
    
    async def up(self, conn):
        """Add trend tracking tables."""
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS trends (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic VARCHAR(255) NOT NULL,
                trend_type VARCHAR(100) NOT NULL,
                score FLOAT NOT NULL,
                time_window VARCHAR(50) NOT NULL,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title VARCHAR(255) NOT NULL,
                description TEXT,
                alert_type VARCHAR(100) NOT NULL,
                severity VARCHAR(50) NOT NULL,
                status VARCHAR(50) DEFAULT 'active',
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved_at TIMESTAMP
            )
        """))

# Available migrations in order
MIGRATIONS = [
    Migration_001_InitialSchema(),
    Migration_002_AddKnowledgeGraph(),
    Migration_003_AddTrendTracking()
]

async def create_migrations_table():
    """Create migration tracking table."""
    async with engine.begin() as conn:
        await conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {MIGRATIONS_TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version VARCHAR(50) NOT NULL UNIQUE,
                description TEXT,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))

async def get_applied_migrations():
    """Get list of applied migrations."""
    try:
        async with engine.begin() as conn:
            result = await conn.execute(
                text(f"SELECT version FROM {MIGRATIONS_TABLE} ORDER BY applied_at")
            )
            return [row[0] for row in result]
    except:
        return []

async def apply_migration(migration: Migration):
    """Apply a single migration."""
    logger.info(f"Applying migration {migration.version}: {migration.description}")
    
    async with engine.begin() as conn:
        try:
            # Apply the migration
            await migration.up(conn)
            
            # Record the migration
            await conn.execute(
                text(f"""
                    INSERT INTO {MIGRATIONS_TABLE} (version, description)
                    VALUES (?, ?)
                """),
                (migration.version, migration.description)
            )
            
            logger.info(f"Migration {migration.version} applied successfully")
            
        except Exception as e:
            logger.error(f"Error applying migration {migration.version}: {e}")
            raise

async def run_migrations():
    """Run all pending migrations."""
    logger.info("Starting database migrations...")
    
    # Create migrations table if it doesn't exist
    await create_migrations_table()
    
    # Get applied migrations
    applied = await get_applied_migrations()
    logger.info(f"Applied migrations: {applied}")
    
    # Apply pending migrations
    pending_count = 0
    for migration in MIGRATIONS:
        if migration.version not in applied:
            await apply_migration(migration)
            pending_count += 1
    
    if pending_count == 0:
        logger.info("No pending migrations")
    else:
        logger.info(f"Applied {pending_count} migrations successfully")

async def rollback_migration(version: str):
    """Rollback a specific migration."""
    logger.info(f"Rolling back migration {version}")
    
    # Find the migration
    migration = None
    for m in MIGRATIONS:
        if m.version == version:
            migration = m
            break
    
    if not migration:
        logger.error(f"Migration {version} not found")
        return
    
    async with engine.begin() as conn:
        try:
            # Rollback the migration
            await migration.down(conn)
            
            # Remove from migration history
            await conn.execute(
                text(f"DELETE FROM {MIGRATIONS_TABLE} WHERE version = ?"),
                (version,)
            )
            
            logger.info(f"Migration {version} rolled back successfully")
            
        except Exception as e:
            logger.error(f"Error rolling back migration {version}: {e}")
            raise

def main():
    """Main migration function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="TruthGuard Database Migrations")
    parser.add_argument("command", choices=["migrate", "rollback"], 
                       help="Migration command")
    parser.add_argument("--version", help="Version to rollback (for rollback command)")
    
    args = parser.parse_args()
    
    if args.command == "migrate":
        asyncio.run(run_migrations())
    elif args.command == "rollback":
        if not args.version:
            logger.error("Version required for rollback")
            sys.exit(1)
        asyncio.run(rollback_migration(args.version))

if __name__ == "__main__":
    main()