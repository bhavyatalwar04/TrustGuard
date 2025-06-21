"""
Fact Checker Service
Core fact-checking logic and verification pipeline
"""

import asyncio
import logging
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import aiohttp
import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import hashlib

from ..config import settings
from ..models.verification import VerificationStatus, VerificationMethod
from .semantic_matcher import SemanticMatcher
from .knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

@dataclass
class FactCheckResult:
    """Result of fact-checking process"""
    status: VerificationStatus
    confidence: float
    verdict: str
    summary: str
    evidence: List[Dict]
    sources: List[Dict]
    processing_time: float
    method: VerificationMethod
    metadata: Dict

@dataclass
class Evidence:
    """Evidence supporting or contradicting a claim"""
    text: str
    source_url: str
    source_title: str
    credibility_score: float
    relevance_score: float
    supports_claim: bool
    publication_date: Optional[datetime]

class FactChecker:
    """
    Main fact-checking service that orchestrates the verification process
    """
    
    def __init__(self):
        self.semantic_matcher = SemanticMatcher()
        self.knowledge_graph = KnowledgeGraph()
        self.session = None
        
        # Trusted fact-checking sources
        self.trusted_sources = [
            'snopes.com',
            'factcheck.org',
            'politifact.com',
            'reuters.com',
            'apnews.com',
            'bbc.com',
            'washingtonpost.com',
            'nytimes.com'
        ]
        
        # Source credibility ratings
        self.source_credibility = {
            'snopes.com': 0.95,
            'factcheck.org': 0.9,
            'politifact.com': 0.85,
            'reuters.com': 0.9,
            'apnews.com': 0.9,
            'bbc.com': 0.85,
            'cnn.com': 0.7,
            'foxnews.com': 0.6,
            'wikipedia.org': 0.7,
            'youtube.com': 0.3,
            'facebook.com': 0.2,
            'twitter.com': 0.3,
        }
        
        # Keywords that indicate fact-checking content
        self.fact_check_keywords = [
            'fact check', 'false', 'true', 'misleading', 'unproven',
            'debunked', 'verified', 'accurate', 'inaccurate',
            'claim', 'allegation', 'statement'
        ]
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'TruthGuard-FactChecker/1.0'}
        )
        return self
    
    async def __aexit