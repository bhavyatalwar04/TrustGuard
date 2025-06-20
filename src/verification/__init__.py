"""
Verification Pipeline Package

A comprehensive fact-checking and claim verification system that integrates
multiple verification methods including:
- Knowledge graphs
- Semantic similarity analysis
- External fact-checking APIs
- Evidence source extraction
- Final verdict generation

Main Components:
- ClaimVerificationPipeline: Core pipeline orchestrating the verification process
- DatabaseManager: Handles data persistence and retrieval
- KnowledgeGraphManager: Manages Wikipedia and news-based KG evidence
- FactCheckAPIManager: Interfaces with third-party fact-check APIs
- SemanticMatcher: Performs semantic similarity matching
- VerdictEngine: Computes final verdict and confidence
- Models: Standard data structures for claims and results
"""

# --- Core Pipeline ---
from .ClaimVerification import ClaimVerificationPipeline

# --- Submodules ---
from .DatabaseManager import DatabaseManager
from .KnowledgeGraph import KnowledgeGraphManager
from .FactCheckManager import FactCheckAPIManager
from .SemanticMatcher import SemanticMatcher
from .VerdictEngine import VerdictEngine

# --- Models ---
from .models import (
    ExtractedClaim,
    VerificationResult,
    VerificationStatus,
    SourceReliability
)

# Package-level exports
__all__ = [
    # Main pipeline
    "ClaimVerificationPipeline",

    # Components
    "DatabaseManager",
    "KnowledgeGraphManager",
    "FactCheckAPIManager",
    "SemanticMatcher",
    "VerdictEngine",

    # Models
    "ExtractedClaim",
    "VerificationResult",
    "VerificationStatus",
    "SourceReliability"
]
