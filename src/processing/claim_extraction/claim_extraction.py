import re
import hashlib
import logging
from datetime import datetime
from typing import List

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None
    logging.warning("spaCy not available – using fallback for entity extraction.")

from verification.models import ExtractedClaim  # ✅ Adjust relative import as you're in `processing/`

class ClaimExtractor:
    """Extract claims from social media posts"""
    
    def __init__(self):
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
        return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip() and len(s.strip()) > 15]
    
    def _is_potential_claim(self, sentence: str) -> bool:
        sentence_lower = sentence.lower()
        has_indicator = any(indicator in sentence_lower for indicator in self.claim_indicators)
        has_numbers = bool(re.search(r'\d+', sentence))
        has_proper_nouns = bool(re.search(r'\b[A-Z][a-z]+\b', sentence))
        is_question = sentence.strip().endswith('?')
        
        return (has_indicator or (has_numbers and has_proper_nouns)) and not is_question and len(sentence) > 30
    
    def _extract_keywords(self, text: str) -> List[str]:
        text_lower = text.lower()
        keywords = [kw for kw in self.claim_indicators + self.political_keywords + self.health_keywords if kw in text_lower]
        keywords.extend(re.findall(r'\b[A-Z][a-z]+\b', text)[:5])
        return list(set(keywords))
    
    def _extract_entities(self, text: str) -> List[str]:
        if nlp:
            doc = nlp(text)
            return [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE"]][:5]
        return re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)[:5]
    
    def _classify_claim_type(self, text: str) -> str:
        text_lower = text.lower()
        if any(kw in text_lower for kw in self.political_keywords):
            return "political"
        elif any(kw in text_lower for kw in self.health_keywords):
            return "health"
        elif any(w in text_lower for w in ['economy', 'financial', 'money', 'market']):
            return "economic"
        return "general"
    
    def _assess_confidence(self, text: str) -> str:
        text_lower = text.lower()
        if any(x in text_lower for x in ['according to', 'research shows', 'study finds', 'data reveals']):
            return "high"
        elif any(x in text_lower for x in ['reports', 'claims', 'alleges', 'suggests']):
            return "medium"
        return "low"
