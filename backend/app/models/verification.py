"""
Verification Model
Database models for fact-checking and verification results
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from enum import Enum
import uuid

from ..database import Base

class VerificationStatus(str, Enum):
    """Verification status"""
    TRUE = "true"
    FALSE = "false"
    PARTIALLY_TRUE = "partially_true"
    UNVERIFIED = "unverified"
    DISPUTED = "disputed"
    SATIRE = "satire"
    
class VerificationMethod(str, Enum):
    """Method used for verification"""
    AUTOMATED = "automated"
    MANUAL = "manual"
    HYBRID = "hybrid"
    CROWDSOURCED = "crowdsourced"

class Verification(Base):
    """
    Main verification record for a claim
    """
    __tablename__ = "verifications"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    
    # Associated claim
    claim_id = Column(Integer, ForeignKey("claims.id"), nullable=False)
    
    # Verification details
    status = Column(String(20), nullable=False, index=True)
    method = Column(String(20), default=VerificationMethod.AUTOMATED)
    confidence_score = Column(Float, nullable=False)  # 0.0 to 1.0
    
    # Results
    verdict = Column(Text, nullable=False)  # Human-readable explanation
    summary = Column(Text)  # Brief summary of findings
    detailed_analysis = Column(Text)  # Detailed analysis
    
    # Evidence and sources
    evidence_count = Column(Integer, default=0)
    supporting_sources = Column(Integer, default=0)
    contradicting_sources = Column(Integer, default=0)
    
    # Fact-checking metadata
    fact_checkers_consensus = Column(String(20))  # Consensus among fact-checkers
    automated_flags = Column(JSON)  # Flags raised by automated systems
    
    # Quality metrics
    evidence_quality_score = Column(Float)  # Quality of evidence found
    source_diversity_score = Column(Float)  # Diversity of sources
    temporal_consistency_score = Column(Float)  # Consistency over time
    
    # Processing information
    processing_time_seconds = Column(Float)
    model_version = Column(String(50))
    algorithm_version = Column(String(50))
    
    # Human oversight
    reviewed_by = Column(Integer, ForeignKey("users.id"))
    reviewed_at = Column(DateTime)
    human_verified = Column(Boolean, default=False)
    
    # Metadata
    metadata = Column(JSON)  # Additional processing metadata
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), index=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    claim = relationship("Claim", back_populates="verifications")
    reviewer = relationship("User", foreign_keys=[reviewed_by])
    results = relationship("VerificationResult", back_populates="verification", cascade="all, delete-orphan")
    sources = relationship("FactCheckSource", back_populates="verification", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Verification(id={self.id}, claim_id={self.claim_id}, status='{self.status}', confidence={self.confidence_score})>"
    
    @property
    def is_reliable(self) -> bool:
        """Check if verification is reliable based on confidence score"""
        return self.confidence_score >= 0.7
    
    @property
    def evidence_ratio(self) -> float:
        """Ratio of supporting to contradicting sources"""
        if self.contradicting_sources == 0:
            return float('inf') if self.supporting_sources > 0 else 0
        return self.supporting_sources / self.contradicting_sources
    
    def to_dict(self) -> dict:
        """Convert verification to dictionary"""
        return {
            "id": self.id,
            "uuid": self.uuid,
            "claim_id": self.claim_id,
            "status": self.status,
            "method": self.method,
            "confidence_score": self.confidence_score,
            "verdict": self.verdict,
            "summary": self.summary,
            "evidence_count": self.evidence_count,
            "supporting_sources": self.supporting_sources,
            "contradicting_sources": self.contradicting_sources,
            "evidence_quality_score": self.evidence_quality_score,
            "source_diversity_score": self.source_diversity_score,
            "processing_time_seconds": self.processing_time_seconds,
            "human_verified": self.human_verified,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

class VerificationResult(Base):
    """
    Detailed results from specific verification steps
    """
    __tablename__ = "verification_results"
    
    id = Column(Integer, primary_key=True, index=True)
    verification_id = Column(Integer, ForeignKey("verifications.id"), nullable=False)
    
    # Step information
    step_name = Column(String(100), nullable=False)  # e.g., "semantic_matching", "source_credibility"
    step_order = Column(Integer, nullable=False)
    
    # Results
    result_type = Column(String(50))  # score, boolean, classification, etc.
    result_value = Column(String(500))  # The actual result
    confidence = Column(Float)  # Confidence in this specific step
    
    # Additional data
    details = Column(JSON)  # Detailed results data
    execution_time_ms = Column(Integer)  # Time taken for this step
    
    # Error handling
    has_error = Column(Boolean, default=False)
    error_message = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    verification = relationship("Verification", back_populates="results")
    
    def __repr__(self):
        return f"<VerificationResult(id={self.id}, step='{self.step_name}', value='{self.result_value}')>"
    
    def to_dict(self) -> dict:
        """Convert result to dictionary"""
        return {
            "id": self.id,
            "verification_id": self.verification_id,
            "step_name": self.step_name,
            "step_order": self.step_order,
            "result_type": self.result_type,
            "result_value": self.result_value,
            "confidence": self.confidence,
            "details": self.details,
            "execution_time_ms": self.execution_time_ms,
            "has_error": self.has_error,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

class FactCheckSource(Base):
    """
    Sources used in fact-checking process
    """
    __tablename__ = "fact_check_sources"
    
    id = Column(Integer, primary_key=True, index=True)
    verification_id = Column(Integer, ForeignKey("verifications.id"), nullable=False)
    
    # Source information
    url = Column(String(2048), nullable=False)
    title = Column(String(500))
    domain = Column(String(255), index=True)
    author = Column(String(255))
    
    # Content
    content = Column(Text)
    excerpt = Column(Text)
    
    # Source classification
    source_type = Column(String(50))  # news, academic, government, social_media, etc.
    credibility_rating = Column(String(20))  # high, medium, low, unknown
    credibility_score = Column(Float)  # 0.0 to 1.0
    
    # Relevance to claim
    relevance_score = Column(Float)  # How relevant to the claim
    supports_claim = Column(Boolean)  # True if supports, False if contradicts, None if neutral
    
    # Quality metrics
    publication_date = Column(DateTime)
    last_updated = Column(DateTime)
    is_primary_source = Column(Boolean, default=False)
    has_citations = Column(Boolean, default=False)
    
    # Access information
    access_date = Column(DateTime, default=func.now())
    http_status = Column(Integer)
    is_accessible = Column(Boolean, default=True)
    
    # Metadata
    metadata = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    verification = relationship("Verification", back_populates="sources")
    
    def __repr__(self):
        return f"<FactCheckSource(id={self.id}, url='{self.url}', credibility='{self.credibility_rating}')>"
    
    def to_dict(self) -> dict:
        """Convert source to dictionary"""
        return {
            "id": self.id,
            "verification_id": self.verification_id,
            "url": self.url,
            "title": self.title,
            "domain": self.domain,
            "author": self.author,
            "excerpt": self.excerpt,
            "source_type": self.source_type,
            "credibility_rating": self.credibility_rating,
            "credibility_score": self.credibility_score,
            "relevance_score": self.relevance_score,
            "supports_claim": self.supports_claim,
            "publication_date": self.publication_date.isoformat() if self.publication_date else None,
            "is_primary_source": self.is_primary_source,
            "has_citations": self.has_citations,
            "is_accessible": self.is_accessible,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }