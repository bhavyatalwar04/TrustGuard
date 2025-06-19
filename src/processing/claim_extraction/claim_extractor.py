"""
Claim Extractor Module

This module contains the ClaimExtractor class and related data structures
for extracting factual claims from text content.
"""

import re
import hashlib
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass

# Optional NLP library - gracefully handle if not installed
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except (ImportError, OSError):
    nlp = None


@dataclass
class ExtractedClaim:
    """Data structure for extracted claims"""
    claim_id: str
    text: str
    claim_type: str
    confidence: str
    source_post_id: str
    extraction_timestamp: str
    keywords: List[str]
    entities: List[str]


class ClaimExtractor:
    """Extract claims from social media posts"""
    
    def __init__(self):
        # Keywords that often indicate factual claims
        self.claim_indicators = [
            'according to', 'research shows', 'study finds', 'experts say',
            'data reveals', 'statistics show', 'report states', 'survey indicates',
            'scientists discover', 'breaking news', 'confirmed', 'official',
            'government', 'president', 'congress', 'senate', 'new law',
            'policy', 'budget', 'tax', 'economy', 'inflation', 'unemployment'
        ]
        
        self.political_keywords = [
            'trump', 'biden', 'harris', 'republican', 'democrat', 'congress',
            'senate', 'house', 'election', 'vote', 'policy', 'bill', 'law'
        ]
        
        self.health_keywords = [
            'vaccine', 'covid', 'medicine', 'treatment', 'cure', 'disease',
            'health', 'medical', 'doctor', 'hospital', 'drug', 'fda'
        ]
    
    def extract_claims_from_text(self, text: str, post_id: str) -> List[ExtractedClaim]:
        """Extract potential claims from text"""
        claims = []
        
        if not text or len(text.strip()) < 20:
            return claims
        
        sentences = self._split_into_sentences(text)
        
        for i, sentence in enumerate(sentences):
            if self._is_potential_claim(sentence):
                claim_id = hashlib.md5(f"{post_id}_{i}_{sentence[:50]}".encode()).hexdigest()[:16]
                
                keywords = self._extract_keywords(sentence)
                entities = self._extract_entities(sentence)
                claim_type = self._classify_claim_type(sentence)
                confidence = self._assess_confidence(sentence)
                
                claim = ExtractedClaim(
                    claim_id=claim_id,
                    text=sentence.strip(),
                    claim_type=claim_type,
                    confidence=confidence,
                    source_post_id=post_id,
                    extraction_timestamp=datetime.now().isoformat(),
                    keywords=keywords,
                    entities=entities
                )
                claims.append(claim)
        
        return claims
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 15]
    
    def _is_potential_claim(self, sentence: str) -> bool:
        """Check if sentence contains a potential factual claim"""
        sentence_lower = sentence.lower()
        
        # Check for claim indicators
        has_indicator = any(indicator in sentence_lower for indicator in self.claim_indicators)
        
        # Check for factual content
        has_numbers = bool(re.search(r'\d+', sentence))
        has_proper_nouns = bool(re.search(r'\b[A-Z][a-z]+\b', sentence))
        
        # Exclude questions and very short statements
        is_question = sentence.strip().endswith('?')
        
        return (has_indicator or (has_numbers and has_proper_nouns)) and not is_question and len(sentence) > 30
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        text_lower = text.lower()
        keywords = []
        
        # Extract keywords from predefined lists
        for keyword in self.claim_indicators + self.political_keywords + self.health_keywords:
            if keyword in text_lower:
                keywords.append(keyword)
        
        # Extract capitalized words (likely proper nouns)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', text)
        keywords.extend(proper_nouns[:5])  # Limit to 5
        
        return list(set(keywords))
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities"""
        if nlp:
            doc = nlp(text)
            entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE"]]
            return entities[:5]
        else:
            # Fallback: extract capitalized words
            return re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)[:5]
    
    def _classify_claim_type(self, text: str) -> str:
        """Classify the type of claim"""
        text_lower = text.lower()
        
        if any(keyword in text_lower for keyword in self.political_keywords):
            return "political"
        elif any(keyword in text_lower for keyword in self.health_keywords):
            return "health"
        elif any(word in text_lower for word in ['economy', 'financial', 'money', 'market']):
            return "economic"
        else:
            return "general"
    
    def _assess_confidence(self, text: str) -> str:
        """Assess confidence level of claim extraction"""
        text_lower = text.lower()
        
        high_confidence_indicators = ['according to', 'research shows', 'study finds', 'data reveals']
        medium_confidence_indicators = ['reports', 'claims', 'alleges', 'suggests']
        
        if any(indicator in text_lower for indicator in high_confidence_indicators):
            return "high"
        elif any(indicator in text_lower for indicator in medium_confidence_indicators):
            return "medium"
        else:
            return "low"