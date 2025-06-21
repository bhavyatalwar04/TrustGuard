from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session
from typing import List, Optional
import logging
from datetime import datetime

from ..database import get_db
from ..models.claim import Claim
from ..models.verification import Verification
from ..models.user import User
from ..services.claim_extractor import ClaimExtractor
from ..services.fact_checker import FactChecker
from ..services.trend_detector import TrendDetector
from ..services.alert_system import AlertSystem
from .schemas import (
    ClaimCreate, ClaimResponse, VerificationResponse,
    TrendAnalysisResponse, AlertResponse, UserCreate, UserResponse
)

# Initialize router
router = APIRouter()
security = HTTPBearer()

# Initialize services
claim_extractor = ClaimExtractor()
fact_checker = FactChecker()
trend_detector = TrendDetector()
alert_system = AlertSystem()

logger = logging.getLogger(__name__)

# Authentication dependency
async def get_current_user(token: str = Depends(security), db: Session = Depends(get_db)):
    # Simple token validation - implement proper JWT validation
    user = db.query(User).filter(User.api_token == token.credentials).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )
    return user

@router.post("/claims/extract", response_model=List[ClaimResponse])
async def extract_claims(
    claim_data: ClaimCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Extract claims from provided text"""
    try:
        # Extract claims from text
        extracted_claims = claim_extractor.extract_claims(claim_data.text)
        
        # Save claims to database
        saved_claims = []
        for claim_text in extracted_claims:
            claim = Claim(
                text=claim_text,
                source=claim_data.source,
                user_id=current_user.id,
                created_at=datetime.utcnow()
            )
            db.add(claim)
            db.flush()
            saved_claims.append(claim)
        
        db.commit()
        
        return [ClaimResponse.from_orm(claim) for claim in saved_claims]
    
    except Exception as e:
        logger.error(f"Error extracting claims: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to extract claims"
        )

@router.post("/claims/{claim_id}/verify", response_model=VerificationResponse)
async def verify_claim(
    claim_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Verify a specific claim"""
    try:
        # Get claim from database
        claim = db.query(Claim).filter(Claim.id == claim_id).first()
        if not claim:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Claim not found"
            )
        
        # Perform fact checking
        verification_result = fact_checker.verify_claim(claim.text)
        
        # Save verification result
        verification = Verification(
            claim_id=claim_id,
            verdict=verification_result['verdict'],
            confidence_score=verification_result['confidence'],
            evidence=verification_result['evidence'],
            sources=verification_result['sources'],
            verified_at=datetime.utcnow()
        )
        
        db.add(verification)
        claim.verification_status = verification_result['verdict']
        db.commit()
        
        return VerificationResponse.from_orm(verification)
    
    except Exception as e:
        logger.error(f"Error verifying claim {claim_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to verify claim"
        )

@router.get("/claims", response_model=List[ClaimResponse])
async def get_claims(
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get list of claims with optional filtering"""
    query = db.query(Claim)
    
    if status:
        query = query.filter(Claim.verification_status == status)
    
    claims = query.offset(skip).limit(limit).all()
    return [ClaimResponse.from_orm(claim) for claim in claims]

@router.get("/claims/{claim_id}", response_model=ClaimResponse)
async def get_claim(
    claim_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a specific claim by ID"""
    claim = db.query(Claim).filter(Claim.id == claim_id).first()
    if not claim:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Claim not found"
        )
    return ClaimResponse.from_orm(claim)

@router.get("/trends/analysis", response_model=TrendAnalysisResponse)
async def get_trend_analysis(
    days: int = 7,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get trend analysis for misinformation patterns"""
    try:
        trends = trend_detector.analyze_trends(days=days)
        return TrendAnalysisResponse(**trends)
    
    except Exception as e:
        logger.error(f"Error analyzing trends: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze trends"
        )

@router.get("/alerts", response_model=List[AlertResponse])
async def get_alerts(
    active_only: bool = True,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get system alerts"""
    try:
        alerts = alert_system.get_alerts(active_only=active_only)
        return [AlertResponse(**alert) for alert in alerts]
    
    except Exception as e:
        logger.error(f"Error fetching alerts: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch alerts"
        )

@router.post("/alerts/check", response_model=List[AlertResponse])
async def check_new_alerts(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Check for new alerts based on recent activity"""
    try:
        new_alerts = alert_system.check_new_alerts()
        return [AlertResponse(**alert) for alert in new_alerts]
    
    except Exception as e:
        logger.error(f"Error checking new alerts: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check new alerts"
        )

@router.post("/users", response_model=UserResponse)
async def create_user(user_data: UserCreate, db: Session = Depends(get_db)):
    """Create a new user"""
    try:
        # Check if user already exists
        existing_user = db.query(User).filter(User.email == user_data.email).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this email already exists"
            )
        
        # Create new user
        user = User(
            username=user_data.username,
            email=user_data.email,
            created_at=datetime.utcnow()
        )
        user.set_password(user_data.password)
        
        db.add(user)
        db.commit()
        db.refresh(user)
        
        return UserResponse.from_orm(user)
    
    except Exception as e:
        logger.error(f"Error creating user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user"
        )

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow()}