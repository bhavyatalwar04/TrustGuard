"""
Claim Extraction Module

This module provides functionality for extracting factual claims from social media posts
and other text content for fact-checking purposes.
"""

from .claim_extractor import ClaimExtractor, ExtractedClaim

__version__ = "1.0.0"
__author__ = "Your Name"

__all__ = [
    "ClaimExtractor",
    "ExtractedClaim",
]