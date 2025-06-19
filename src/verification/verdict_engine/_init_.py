"""
Verdict Engine Module

This module provides functionality for determining verification verdicts
based on multiple evidence sources including fact-check APIs, knowledge graphs,
semantic matching, and news verification.
"""

from .verdict_engine import VerdictEngine
from .models import ExtractedClaim, VerificationStatus

__version__ = "1.0.0"
__author__ = "Your Name"

__all__ = [
    "VerdictEngine",
    "ExtractedClaim", 
    "VerificationStatus"
]