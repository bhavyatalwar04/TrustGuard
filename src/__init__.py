"""
TruthGuard: A Fact-Checking and Claim Verification Framework

This package includes:
1. Reddit Data Collection
2. Preprocessing and Claim Extraction
3. Knowledge Graph Integration
4. Fact Checking and Semantic Matching
5. Final Verdict Generation and Reporting
"""

from .data_collection import RedditCollector, RedditPost
from .verification import (
    ClaimVerificationPipeline,
    DatabaseManager,
    KnowledgeGraphManager,
    FactCheckAPIManager,
    SemanticMatcher,
    VerdictEngine,
    ExtractedClaim,
    VerificationResult,
    VerificationStatus,
    SourceReliability
)


__all__ = [
    "RedditCollector",
    "RedditPost",
    "ClaimVerificationPipeline",
    "DatabaseManager",
    "KnowledgeGraphManager",
    "FactCheckAPIManager",
    "SemanticMatcher",
    "VerdictEngine",
    "ExtractedClaim",
    "VerificationResult",
    "VerificationStatus",
    "SourceReliability"
]
