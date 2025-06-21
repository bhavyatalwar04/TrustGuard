"""
TruthGuard Services Package
Business logic and service layer for the TruthGuard application
"""

from .claim_extractor import ClaimExtractor
from .fact_checker import FactChecker
from .knowledge_graph import KnowledgeGraph
from .semantic_matcher import SemanticMatcher
from .trend_detector import TrendDetector
from .alert_system import AlertSystem

__all__ = [
    "ClaimExtractor",
    "FactChecker", 
    "KnowledgeGraph",
    "SemanticMatcher",
    "TrendDetector",
    "AlertSystem"
]
