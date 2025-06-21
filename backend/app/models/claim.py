"""
Claim Model
Database model for claims and claim sources
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from enum import Enum
import uuid

from ..database import Base

class ClaimStatus(str, Enum):
    """Claim processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    VERIFIED = "verified"
    FAILED = "failed"

class ClaimCategory(str, Enum):
    """Claim categories"""
    POLITICAL = "political"
    HEALTH = "health"
    SCIENCE = "science"
    TECHNOLOGY = "technology"
    SPORTS = "sports"
    ENTERTAINMENT = "entertainment"
    BUSINESS = "business"
    OTHER = "other"

class Claim(Base):
    """
    Claim model representing a factual claim to be verified
    """
    __tablename__ = "claims"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    
    # Claim content
    text = Column(Text, nullable=False, index=True)
    summary = Column(Text)
    category = Column(String(50), default=ClaimCategory.OTHER)
    keywords = Column(JSON)  # List of extracted keywords
    
    # Processing status
    status = Column(String(20), default=ClaimStatus.PENDING, index=True)
    processing_started_at = Column(DateTime)
    processing_completed_at = Column(DateTime)
    
    # Confidence and scoring
    confidence_score = Column(Float)  # Overall confidence in verification
    credibility_score = Column(Float)  # Source credibility score
    viral_score = Column(Float)  # How viral/trending the claim is
    
    # Source information
    original_url = Column(String(2048))
    source_domain = Column(String(255), index=True)
    author = Column(String(255))
    publication_date = Column(DateTime)
    
    # Content analysis
    language = Column(String(10), default="en")
    word_count = Column(Integer)
    readability_score = Column(Float)
    sentiment_score = Column(Float)
    
    # Metadata
    metadata = Column(JSON)  # Additional flexible metadata
    tags = Column(JSON)  # User-defined tags
    
    # Tracking
    view_count = Column(Integer, default=0)
    share_count = Column(Integer, default=0)
    
    # Flags
    is_public = Column(Boolean, default=True)
    is_flagged = Column(Boolean, default=False)
    is_trending = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), index=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Foreign keys
    submitted_by = Column(Integer, ForeignKey("users.id"))
    
    # Relationships
    submitter = relationship("User", back_populates="submitted_claims")
    sources = relationship("ClaimSource", back_populates="claim", cascade="all, delete-orphan")
    verifications = relationship("Verification", back_populates="claim", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Claim(id={self.id}, text='{self.text[:50]}...', status='{self.status}')>"
    
    @property
    def is_verified(self) -> bool:
        """Check if claim has been verified"""
        return self.status == ClaimStatus.VERIFIED
    
    @property
    def latest_verification(self):
        """Get the most recent verification"""
        if self.verifications:
            return sorted(self.verifications, key=lambda v: v.created_at, reverse=True)[0]
        return None
    
    def to_dict(self) -> dict:
        """Convert claim to dictionary"""
        return {
            "id": self.id,
            "uuid": self.uuid,
            "text": self.text,
            "summary": self.summary,
            "category": self.category,
            "status": self.status,
            "confidence_score": self.confidence_score,
            "credibility_score": self.credibility_score,
            "viral_score": self.viral_score,
            "original_url": self.original_url,
            "source_domain": self.source_domain,
            "author": self.author,
            "publication_date": self.publication_date.isoformat() if self.publication_date else None,
            "language": self.language,
            "word_count": self.word_count,
            "sentiment_score": self.sentiment_score,
            "keywords": self.keywords,
            "tags": self.tags,
            "view_count": self.view_count,
            "share_count": self.share_count,
            "is_public": self.is_public,
            "is_flagged": self.is_flagged,
            "is_trending": self.is_trending,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

class ClaimSource(Base):
    """
    Sources associated with a claim
    """
    __tablename__ = "claim_sources"
    
    id = Column(Integer, primary_key=True, index=True)
    claim_id = Column(Integer, ForeignKey("claims.id"), nullable=False)
    
    # Source details
    url = Column(String(2048), nullable=False)
    title = Column(String(500))
    domain = Column(String(255), index=True)
    author = Column(String(255))
    
    # Content
    content = Column(Text)
    excerpt = Column(Text)
    
    # Metadata
    publication_date = Column(DateTime)
    crawled_at = Column(DateTime, default=func.now())
    content_type = Column(String(50))  # article, social_post, video, etc.
    
    # Quality metrics
    credibility_score = Column(Float)
    relevance_score = Column(Float)
    
    # Status
    is_accessible = Column(Boolean, default=True)
    http_status = Column(Integer)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    claim = relationship("Claim", back_populates="sources")
    
    def __repr__(self):
        return f"<ClaimSource(id={self.id}, url='{self.url}', domain='{self.domain}')>"
    
    def to_dict(self) -> dict:
        """Convert source to dictionary"""
        return {
            "id": self.id,
            "claim_id": self.claim_id,
            "url": self.url,
            "title": self.title,
            "domain": self.domain,
            "author": self.author,
            "excerpt": self.excerpt,
            "publication_date": self.publication_date.isoformat() if self.publication_date else None,
            "content_type": self.content_type,
            "credibility_score": self.credibility_score,
            "relevance_score": self.relevance_score,
            "is_accessible": self.is_accessible,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }