from dataclasses import dataclass
from typing import List, Dict

@dataclass
class VerificationResult:
    """Structure for verification results"""
    claim_id: str
    verification_status: str
    confidence_score: float
    evidence_sources: List[Dict]
    fact_check_results: List[Dict]
    semantic_similarity_scores: List[Dict]
    final_verdict: str
    reasoning: str
    timestamp: str
    processing_time: float
