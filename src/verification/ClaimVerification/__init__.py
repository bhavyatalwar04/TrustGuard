"""
Claim Verification Pipeline Package

A comprehensive fact-checking and claim verification system that integrates
multiple verification methods including knowledge graphs, semantic matching,
and external fact-checking APIs.

Main Components:
- ClaimVerificationPipeline: Core pipeline orchestrating the verification process
- DatabaseManager: Handles data persistence and retrieval
- KnowledgeGraphManager: Manages knowledge graph operations
- FactCheckAPIManager: Interfaces with external fact-checking services
- SemanticMatcher: Performs semantic similarity analysis
- VerdictEngine: Determines final verdicts and confidence scores

Usage:
    from claim_verification import ClaimVerificationPipeline
    
    pipeline = ClaimVerificationPipeline()
    result = await pipeline.verify_single_claim(claim)
"""


# Import main classes for easy access
from ..DatabaseManager import DatabaseManager
from ..KnowledgeGraph import KnowledgeGraphManager
from ..FactCheckManager import FactCheckAPIManager
from ..SemanticMatcher import SemanticMatcher
from ..VerdictEngine import VerdictEngine
from ..models import ExtractedClaim, VerificationResult, VerificationStatus
from .ClaimVerificationPipeline import ClaimVerificationPipeline


# Define what gets imported when using "from package import *"
__all__ = [
    'ClaimVerificationPipeline',
    'DatabaseManager',
    'KnowledgeGraphManager',
    'FactCheckAPIManager',
    'SemanticMatcher',
    'VerdictEngine',
    'ExtractedClaim',
    'VerificationResult',
    'VerificationStatus',
]
