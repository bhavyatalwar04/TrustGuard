import sqlite3
import json
import logging
import hashlib
from typing import List, Dict

from ..models import ExtractedClaim, VerificationResult  # ✅ Correct relative import

class DatabaseManager:
    """Manages SQLite database for storing results"""

    def __init__(self, db_path: str = "verification_results.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS claims (
                        claim_id TEXT PRIMARY KEY,
                        claim_text TEXT NOT NULL,
                        claim_type TEXT,
                        confidence TEXT,
                        source_post_id TEXT,
                        extraction_timestamp TEXT,
                        keywords TEXT,
                        entities TEXT
                    )
                ''')

                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS verification_results (
                        result_id TEXT PRIMARY KEY,
                        claim_id TEXT,
                        verification_status TEXT,
                        confidence_score REAL,
                        final_verdict TEXT,
                        reasoning TEXT,
                        processing_time REAL,
                        timestamp TEXT,
                        evidence_sources TEXT,
                        fact_check_results TEXT,
                        FOREIGN KEY (claim_id) REFERENCES claims (claim_id)
                    )
                ''')

                conn.commit()
            logging.info("Database initialized successfully")
        except Exception as e:
            logging.error(f"Database initialization error: {e}")

    def store_claim(self, claim: ExtractedClaim):
        """Store extracted claim in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO claims VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    claim.claim_id,
                    claim.text,
                    claim.claim_type,
                    claim.confidence,
                    claim.source_post_id,
                    claim.extraction_timestamp,
                    json.dumps(claim.keywords),
                    json.dumps(claim.entities)
                ))
                conn.commit()
        except Exception as e:
            logging.error(f"Error storing claim: {e}")

    def store_verification_result(self, result: VerificationResult):
        """Store verification result in database"""
        try:
            result_id = hashlib.md5(f"{result.claim_id}_{result.timestamp}".encode()).hexdigest()[:16]
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO verification_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    result_id,
                    result.claim_id,
                    result.verification_status,
                    result.confidence_score,
                    result.final_verdict,
                    result.reasoning,
                    result.processing_time,
                    result.timestamp,
                    json.dumps(result.evidence_sources),
                    json.dumps(result.fact_check_results)
                ))
                conn.commit()
        except Exception as e:
            logging.error(f"Error storing verification result: {e}")
