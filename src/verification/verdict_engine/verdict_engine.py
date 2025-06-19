"""
Verdict Engine for Fact-Checking System

This module contains the VerdictEngine class which determines verification verdicts
based on multiple evidence sources and confidence scoring.
"""

from typing import Dict, List, Tuple
from enum import Enum


class VerificationStatus(Enum):
    """Enumeration of possible verification statuses"""
    VERIFIED_TRUE = "verified_true"
    VERIFIED_FALSE = "verified_false"
    PARTIALLY_TRUE = "partially_true"
    MISLEADING = "misleading"
    UNVERIFIABLE = "unverifiable"
    DISPUTED = "disputed"


class ExtractedClaim:
    """Data class representing an extracted claim to be verified"""
    
    def __init__(self, claim_text: str, context: str = "", source: str = "", 
                 claim_id: str = None, timestamp: str = None):
        self.claim_text = claim_text
        self.context = context
        self.source = source
        self.claim_id = claim_id
        self.timestamp = timestamp
        
    def __repr__(self):
        return f"ExtractedClaim(claim_text='{self.claim_text[:50]}...', source='{self.source}')"


class VerdictEngine:
    """Enhanced verdict engine with knowledge graph integration"""
    
    def __init__(self):
        self.confidence_weights = {
            'fact_check_apis': 0.4,
            'semantic_matching': 0.2,
            'knowledge_graph': 0.3,
            'news_verification': 0.1
        }
    
    def calculate_confidence_score(self, verification_data: Dict) -> float:
        """Calculate overall confidence score"""
        total_score = 0.0
        total_weight = 0.0
        
        # Fact-check results score
        fact_check_results = verification_data.get('fact_check_results', [])
        if fact_check_results:
            fc_score = min(len(fact_check_results) * 0.25 + 0.4, 1.0)
            total_score += fc_score * self.confidence_weights['fact_check_apis']
            total_weight += self.confidence_weights['fact_check_apis']
        
        # Knowledge graph evidence score
        evidence_sources = verification_data.get('evidence_sources', [])
        if evidence_sources:
            kg_score = min(len(evidence_sources) * 0.2 + 0.3, 1.0)
            wikipedia_sources = [s for s in evidence_sources if 'Wikipedia' in s.get('source', '')]
            if wikipedia_sources:
                kg_score = min(kg_score + 0.2, 1.0)
            
            total_score += kg_score * self.confidence_weights['knowledge_graph']
            total_weight += self.confidence_weights['knowledge_graph']
        
        # Semantic matching score
        semantic_scores = verification_data.get('semantic_similarity_scores', [])
        if semantic_scores:
            avg_similarity = sum(s.get('similarity_score', 0) for s in semantic_scores) / len(semantic_scores)
            total_score += avg_similarity * self.confidence_weights['semantic_matching']
            total_weight += self.confidence_weights['semantic_matching']
        
        return total_score / total_weight if total_weight > 0 else 0.5
    
    def determine_verdict(self, claim: ExtractedClaim, verification_data: Dict) -> Tuple[str, str]:
        """Determine final verdict and reasoning"""
        confidence_score = self.calculate_confidence_score(verification_data)
        
        evidence_sources = verification_data.get('evidence_sources', [])
        fact_check_results = verification_data.get('fact_check_results', [])
        wikipedia_sources = [s for s in evidence_sources if 'Wikipedia' in s.get('source', '')]
        
        reasoning_parts = []
        
        if fact_check_results:
            reasoning_parts.append(f"Found {len(fact_check_results)} fact-check sources")
        
        if wikipedia_sources:
            reasoning_parts.append(f"Found {len(wikipedia_sources)} Wikipedia sources providing context")
        
        if evidence_sources:
            reasoning_parts.append(f"Total {len(evidence_sources)} evidence sources found")
        
        # Determine verdict based on available evidence
        if confidence_score > 0.7:
            verdict = VerificationStatus.VERIFIED_TRUE.value
        elif confidence_score > 0.4:
            verdict = VerificationStatus.PARTIALLY_TRUE.value
        else:
            verdict = VerificationStatus.UNVERIFIABLE.value
        
        reasoning = '; '.join(reasoning_parts) if reasoning_parts else "Limited evidence available for verification"
        
        return verdict, reasoning
    
    def update_confidence_weights(self, new_weights: Dict[str, float]) -> None:
        """Update confidence weights for different verification methods"""
        for key, weight in new_weights.items():
            if key in self.confidence_weights:
                self.confidence_weights[key] = weight
    
    def get_detailed_analysis(self, claim: ExtractedClaim, verification_data: Dict) -> Dict:
        """Get detailed analysis including all intermediate scores"""
        confidence_score = self.calculate_confidence_score(verification_data)
        verdict, reasoning = self.determine_verdict(claim, verification_data)
        
        return {
            'claim_text': claim.claim_text,
            'verdict': verdict,
            'confidence_score': confidence_score,
            'reasoning': reasoning,
            'evidence_breakdown': {
                'fact_check_sources': len(verification_data.get('fact_check_results', [])),
                'knowledge_graph_sources': len(verification_data.get('evidence_sources', [])),
                'wikipedia_sources': len([s for s in verification_data.get('evidence_sources', []) 
                                        if 'Wikipedia' in s.get('source', '')]),
                'semantic_scores': verification_data.get('semantic_similarity_scores', [])
            },
            'weights_used': self.confidence_weights.copy()
        }