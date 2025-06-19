"""Semantic similarity matching for claim verification"""

import logging
import numpy as np
from typing import List, Dict, Optional

# Optional imports with fallbacks
try:
    from sentence_transformers import SentenceTransformer
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
except ImportError:
    logging.warning("sentence-transformers not available. Using fallback similarity.")
    sentence_model = None

try:
    from sklearn.metrics.pairwise import cosine_similarity
    sklearn_available = True
except ImportError:
    sklearn_available = False


class SemanticMatcher:
    """Semantic similarity matching for claim verification"""
   
    def __init__(self):
        self.sentence_model = sentence_model
        logging.info("Semantic matcher initialized")
   
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        if not self.sentence_model:
            # Simple word overlap fallback
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            return len(intersection) / len(union) if union else 0.0
       
        try:
            embeddings = self.sentence_model.encode([text1, text2])
            if sklearn_available:
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            else:
                # Manual cosine similarity calculation
                dot_product = np.dot(embeddings[0], embeddings[1])
                norm_a = np.linalg.norm(embeddings[0])
                norm_b = np.linalg.norm(embeddings[1])
                similarity = dot_product / (norm_a * norm_b)
           
            return float(similarity)
        except Exception as e:
            logging.error(f"Similarity calculation error: {e}")
            return 0.0
   
    def match_against_evidence(self, claim_text: str, evidence_list: List[str]) -> List[Dict]:
        """Match claim against evidence sources"""
        results = []
       
        for evidence in evidence_list:
            if evidence.strip():
                similarity = self.calculate_similarity(claim_text, evidence)
                results.append({
                    'evidence_text': evidence,
                    'similarity_score': similarity,
                    'match_strength': 'high' if similarity > 0.8 else 'medium' if similarity > 0.6 else 'low'
                })
       
        return sorted(results, key=lambda x: x['similarity_score'], reverse=True)