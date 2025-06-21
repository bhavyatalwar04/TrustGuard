#!/usr/bin/env python3
"""
Database setup script for TruthGuard application.
Creates all necessary tables and initial data.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the parent directory to Python path to import app modules
sys.path.append(str(Path(__file__).parent.parent))

from app.database import engine, get_database
from app.models.claim import Claim
from app.models.verification import VerificationResult
from app.models.user import User
from sqlalchemy import text
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def create_tables():
    """Create all database tables."""
    try:
        async with engine.begin() as conn:
            # Import all models to ensure they're registered
            from app.models import claim, verification, user
            
            # Create all tables
            await conn.run_sync(User.metadata.create_all)
            await conn.run_sync(Claim.metadata.create_all)
            await conn.run_sync(VerificationResult.metadata.create_all)
            
            logger.info("Database tables created successfully")
            
    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        raise

async def create_indexes():
    """Create database indexes for better performance."""
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_claims_created_at ON claims(created_at)",
        "CREATE INDEX IF NOT EXISTS idx_claims_status ON claims(status)",
        "CREATE INDEX IF NOT EXISTS idx_claims_source_url ON claims(source_url)",
        "CREATE INDEX IF NOT EXISTS idx_verifications_claim_id ON verification_results(claim_id)",
        "CREATE INDEX IF NOT EXISTS idx_verifications_confidence ON verification_results(confidence_score)",
        "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)",
        "CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at)"
    ]
    
    try:
        async with engine.begin() as conn:
            for index_sql in indexes:
                await conn.execute(text(index_sql))
            
            logger.info("Database indexes created successfully")
            
    except Exception as e:
        logger.error(f"Error creating indexes: {e}")
        raise

async def insert_sample_data():
    """Insert sample data for testing."""
    try:
        database = get_database()
        
        # Sample users
        sample_users = [
            {
                "email": "admin@truthguard.com",
                "username": "admin",
                "full_name": "System Administrator",
                "is_active": True,
                "role": "admin"
            },
            {
                "email": "analyst@truthguard.com",
                "username": "analyst",
                "full_name": "Data Analyst",
                "is_active": True,
                "role": "analyst"
            }
        ]
        
        for user_data in sample_users:
            await database.execute(
                "INSERT OR IGNORE INTO users (email, username, full_name, is_active, role) VALUES (?, ?, ?, ?, ?)",
                (user_data["email"], user_data["username"], user_data["full_name"], 
                 user_data["is_active"], user_data["role"])
            )
        
        # Sample claims
        sample_claims = [
            {
                "text": "The Earth is flat and governments are hiding this fact.",
                "source_url": "https://example.com/flat-earth",
                "source_type": "social_media",
                "language": "en",
                "status": "pending"
            },
            {
                "text": "Vaccines contain microchips for tracking people.",
                "source_url": "https://example.com/vaccine-chips",
                "source_type": "blog",
                "language": "en",
                "status": "pending"
            }
        ]
        
        for claim_data in sample_claims:
            await database.execute(
                """INSERT OR IGNORE INTO claims 
                   (text, source_url, source_type, language, status) 
                   VALUES (?, ?, ?, ?, ?)""",
                (claim_data["text"], claim_data["source_url"], claim_data["source_type"],
                 claim_data["language"], claim_data["status"])
            )
        
        logger.info("Sample data inserted successfully")
        
    except Exception as e:
        logger.error(f"Error inserting sample data: {e}")
        raise

async def setup_database():
    """Main setup function."""
    logger.info("Starting database setup...")
    
    try:
        await create_tables()
        await create_indexes()
        await insert_sample_data()
        
        logger.info("Database setup completed successfully!")
        
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(setup_database())