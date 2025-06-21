"""
User Model
Database model for users and authentication
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, Float, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from enum import Enum
import uuid
from passlib.context import CryptContext

from ..database import Base

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class UserRole(str, Enum):
    """User roles"""
    ADMIN = "admin"
    MODERATOR = "moderator"
    FACT_CHECKER = "fact_checker"
    USER = "user"
    READONLY = "readonly"

class User(Base):
    """
    User model for authentication and user management
    """
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    
    # Basic information
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    full_name = Column(String(255))
    
    # Authentication
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    role = Column(String(20), default=UserRole.USER)
    
    # Profile information
    bio = Column(Text)
    avatar_url = Column(String(512))
    location = Column(String(255))
    organization = Column(String(255))
    website = Column(String(512))
    
    # Preferences
    notification_preferences = Column(JSON, default=dict)
    privacy_settings = Column(JSON, default=dict)
    language_preference = Column(String(10), default="en")
    timezone = Column(String(50), default="UTC")
    
    # Statistics
    claims_submitted = Column(Integer, default=0)
    verifications_performed = Column(Integer, default=0)
    reputation_score = Column(Float, default=0.0)
    accuracy_score = Column(Float, default=0.0)
    
    # Activity tracking
    last_login = Column(DateTime)
    login_count = Column(Integer, default=0)
    last_active = Column(DateTime)
    
    # Security
    failed_login_attempts = Column(Integer, default=0)
    account_locked_until = Column(DateTime)
    password_reset_token = Column(String(255))
    password_reset_expires = Column(DateTime)
    email_verification_token = Column(String(255))
    
    # Metadata
    metadata = Column(JSON, default=dict)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), index=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    deleted_at = Column(DateTime)  # Soft delete
    
    # Relationships
    submitted_claims = relationship("Claim", back_populates="submitter")
    
    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', role='{self.role}')>"
    
    def set_password(self, password: str):
        """Hash and set password"""
        self.hashed_password = pwd_context.hash(password)
    
    def verify_password(self, password: str) -> bool:
        """Verify password"""
        return pwd_context.verify(password, self.hashed_password)
    
    @property
    def is_admin(self) -> bool:
        """Check if user is admin"""
        return self.role == UserRole.ADMIN
    
    @property
    def is_moderator(self) -> bool:
        """Check if user is moderator or admin"""
        return self.role in [UserRole.ADMIN, UserRole.MODERATOR]
    
    @property
    def is_fact_checker(self) -> bool:
        """Check if user can perform fact-checking"""
        return self.role in [UserRole.ADMIN, UserRole.MODERATOR, UserRole.FACT_CHECKER]
    
    @property
    def can_submit_claims(self) -> bool:
        """Check if user can submit claims"""
        return self.role != UserRole.READONLY and self.is_active
    
    @property
    def is_account_locked(self) -> bool:
        """Check if account is locked"""
        if self.account_locked_until:
            return datetime.utcnow() < self.account_locked_until
        return False
    
    def lock_account(self, minutes: int = 30):
        """Lock account for specified minutes"""
        self.account_locked_until = datetime.utcnow() + timedelta(minutes=minutes)
        self.failed_login_attempts = 0
    
    def unlock_account(self):
        """Unlock account"""
        self.account_locked_until = None
        self.failed_login_attempts = 0
    
    def increment_failed_login(self):
        """Increment failed login attempts"""
        self.failed_login_attempts += 1
        if self.failed_login_attempts >= 5:
            self.lock_account()
    
    def successful_login(self):
        """Record successful login"""
        self.last_login = datetime.utcnow()
        self.login_count += 1
        self.failed_login_attempts = 0
        self.last_active = datetime.utcnow()
    
    def update_activity(self):
        """Update last active timestamp"""
        self.last_active = datetime.utcnow()
    
    def to_dict(self, include_sensitive: bool = False) -> dict:
        """Convert user to dictionary"""
        data = {
            "id": self.id,
            "uuid": self.uuid,
            "username": self.username,
            "email": self.email if include_sensitive else None,
            "full_name": self.full_name,
            "role": self.role,
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "bio": self.bio,
            "avatar_url": self.avatar_url,
            "location": self.location,
            "organization": self.organization,
            "website": self.website,
            "claims_submitted": self.claims_submitted,
            "verifications_performed": self.verifications_performed,
            "reputation_score": self.reputation_score,
            "accuracy_score": self.accuracy_score,
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "last_active": self.last_active.isoformat() if self.last_active else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
        
        if include_sensitive:
            data.update({
                "login_count": self.login_count,
                "failed_login_attempts": self.failed_login_attempts,
                "is_account_locked": self.is_account_locked,
                "notification_preferences": self.notification_preferences,
                "privacy_settings": self.privacy_settings,
                "language_preference": self.language_preference,
                "timezone": self.timezone,
            })
        
        return data
    
    def to_public_dict(self) -> dict:
        """Convert user to public dictionary (safe for public viewing)"""
        return {
            "id": self.id,
            "username": self.username,
            "full_name": self.full_name,
            "bio": self.bio,
            "avatar_url": self.avatar_url,
            "organization": self.organization,
            "reputation_score": self.reputation_score,
            "claims_submitted": self.claims_submitted,
            "verifications_performed": self.verifications_performed,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }