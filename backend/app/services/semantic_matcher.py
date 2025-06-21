# backend/app/services/semantic_matcher.py

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime
import asyncio
import aiohttp
from ..database import get_db_connection
from ..models.claim import Claim

logger = logging.getLogger(__name__)

class SemanticMatcher:
    """
    Handles semantic matching of claims using sentence transformers and vector similarity.
    Identifies semantically similar claims to detect potential misinformation patterns.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the semantic matcher with a pre-trained sentence transformer model.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = 0.75
        self.cache = {}
        self.max_cache_size = 1000
        
    async def encode_claims(self, claims: List[str]) -> np.ndarray:
        """
        Encode a list of claims into vector embeddings.
        
        Args:
            claims: List of claim texts to encode
            
        Returns:
            numpy array of embeddings
        """
        try:
            # Check cache first
            cache_key = hash(tuple(claims))
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Encode claims
            embeddings = self.model.encode(claims, convert_to_tensor=False)
            
            # Cache results (with size limit)
            if len(self.cache) < self.max_cache_size:
                self.cache[cache_key] = embeddings
            
            return embeddings
        except Exception as e:
            logger.error(f"Error encoding claims: {str(e)}")
            raise
    
    async def find_similar_claims(self, query_claim: str, candidate_claims: List[Dict], 
                                top_k: int = 10) -> List[Dict]:
        """
        Find semantically similar claims to a query claim.
        
        Args:
            query_claim: The claim to find matches for
            candidate_claims: List of candidate claims with metadata
            top_k: Number of top matches to return
            
        Returns:
            List of similar claims with similarity scores
        """
        try:
            if not candidate_claims:
                return []
            
            # Prepare claim texts
            claim_texts = [query_claim] + [claim['text'] for claim in candidate_claims]
            
            # Get embeddings
            embeddings = await self.encode_claims(claim_texts)
            
            # Calculate similarities
            query_embedding = embeddings[0:1]
            candidate_embeddings = embeddings[1:]
            
            similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]
            
            # Create results with metadata
            results = []
            for i, similarity in enumerate(similarities):
                if similarity >= self.similarity_threshold:
                    result = candidate_claims[i].copy()
                    result['similarity_score'] = float(similarity)
                    results.append(result)
            
            # Sort by similarity and return top_k
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding similar claims: {str(e)}")
            return []
    
    async def detect_claim_clusters(self, claims: List[Dict], 
                                  cluster_threshold: float = 0.8) -> List[List[Dict]]:
        """
        Group claims into semantic clusters based on similarity.
        
        Args:
            claims: List of claims to cluster
            cluster_threshold: Minimum similarity for clustering
            
        Returns:
            List of claim clusters
        """
        try:
            if len(claims) < 2:
                return [claims] if claims else []
            
            # Get embeddings for all claims
            claim_texts = [claim['text'] for claim in claims]
            embeddings = await self.encode_claims(claim_texts)
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(embeddings)
            
            # Simple clustering based on threshold
            visited = set()
            clusters = []
            
            for i in range(len(claims)):
                if i in visited:
                    continue
                
                cluster = [claims[i]]
                visited.add(i)
                
                for j in range(i + 1, len(claims)):
                    if j not in visited and similarity_matrix[i][j] >= cluster_threshold:
                        cluster.append(claims[j])
                        visited.add(j)
                
                clusters.append(cluster)
            
            return clusters
            
        except Exception as e:
            logger.error(f"Error detecting claim clusters: {str(e)}")
            return [claims] if claims else []
    
    async def match_against_knowledge_base(self, claim: str, 
                                         verified_claims: List[Dict]) -> Optional[Dict]:
        """
        Match a claim against a knowledge base of verified claims.
        
        Args:
            claim: Claim to verify
            verified_claims: List of pre-verified claims
            
        Returns:
            Best matching verified claim if found
        """
        try:
            similar_claims = await self.find_similar_claims(
                claim, verified_claims, top_k=1
            )
            
            if similar_claims and similar_claims[0]['similarity_score'] >= 0.85:
                return similar_claims[0]
            
            return None
            
        except Exception as e:
            logger.error(f"Error matching against knowledge base: {str(e)}")
            return None
    
    async def analyze_semantic_drift(self, claim: str, 
                                   historical_versions: List[Dict]) -> Dict:
        """
        Analyze how a claim has semantically drifted over time.
        
        Args:
            claim: Current version of the claim
            historical_versions: Previous versions with timestamps
            
        Returns:
            Analysis of semantic drift
        """
        try:
            if not historical_versions:
                return {'drift_score': 0.0, 'evolution': []}
            
            # Sort by timestamp
            versions = sorted(historical_versions, key=lambda x: x['timestamp'])
            all_versions = versions + [{'text': claim, 'timestamp': datetime.now()}]
            
            # Calculate drift between consecutive versions
            drift_scores = []
            evolution = []
            
            for i in range(1, len(all_versions)):
                prev_claim = all_versions[i-1]['text']
                curr_claim = all_versions[i]['text']
                
                # Get similarity
                embeddings = await self.encode_claims([prev_claim, curr_claim])
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                
                drift_score = 1.0 - similarity
                drift_scores.append(drift_score)
                
                evolution.append({
                    'from_version': i-1,
                    'to_version': i,
                    'drift_score': float(drift_score),
                    'timestamp': all_versions[i]['timestamp']
                })
            
            return {
                'drift_score': float(np.mean(drift_scores)) if drift_scores else 0.0,
                'max_drift': float(max(drift_scores)) if drift_scores else 0.0,
                'evolution': evolution
            }
            
        except Exception as e:
            logger.error(f"Error analyzing semantic drift: {str(e)}")
            return {'drift_score': 0.0, 'evolution': []}
    
    async def identify_paraphrase_variants(self, claims: List[Dict]) -> List[List[Dict]]:
        """
        Identify paraphrased variants of the same core claim.
        
        Args:
            claims: List of claims to analyze
            
        Returns:
            Groups of paraphrased claims
        """
        try:
            # Use higher threshold for paraphrase detection
            paraphrase_threshold = 0.85
            
            clusters = await self.detect_claim_clusters(claims, paraphrase_threshold)
            
            # Filter clusters with multiple items (potential paraphrases)
            paraphrase_groups = [cluster for cluster in clusters if len(cluster) > 1]
            
            return paraphrase_groups
            
        except Exception as e:
            logger.error(f"Error identifying paraphrase variants: {str(e)}")
            return []
    
    async def semantic_search(self, query: str, claim_database: List[Dict], 
                            filters: Optional[Dict] = None) -> List[Dict]:
        """
        Perform semantic search over claim database.
        
        Args:
            query: Search query
            claim_database: Database of claims to search
            filters: Optional filters (date range, source, etc.)
            
        Returns:
            Ranked search results
        """
        try:
            # Apply filters if provided
            filtered_claims = claim_database
            if filters:
                if 'date_from' in filters:
                    filtered_claims = [
                        claim for claim in filtered_claims 
                        if claim.get('timestamp', datetime.min) >= filters['date_from']
                    ]
                if 'source' in filters:
                    filtered_claims = [
                        claim for claim in filtered_claims 
                        if claim.get('source') == filters['source']
                    ]
                if 'verified_only' in filters and filters['verified_only']:
                    filtered_claims = [
                        claim for claim in filtered_claims 
                        if claim.get('verification_status') == 'verified'
                    ]
            
            # Perform semantic search
            results = await self.find_similar_claims(query, filtered_claims, top_k=50)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            return []
    
    def update_similarity_threshold(self, threshold: float):
        """Update the similarity threshold for matching."""
        if 0.0 <= threshold <= 1.0:
            self.similarity_threshold = threshold
        else:
            raise ValueError("Threshold must be between 0.0 and 1.0")
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self.cache.clear()
    
    async def batch_similarity_analysis(self, claim_pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Analyze similarity for multiple claim pairs efficiently.
        
        Args:
            claim_pairs: List of (claim1, claim2) tuples
            
        Returns:
            List of similarity scores
        """
        try:
            if not claim_pairs:
                return []
            
            # Flatten pairs and get unique claims
            all_claims = []
            claim_to_index = {}
            
            for claim1, claim2 in claim_pairs:
                if claim1 not in claim_to_index:
                    claim_to_index[claim1] = len(all_claims)
                    all_claims.append(claim1)
                if claim2 not in claim_to_index:
                    claim_to_index[claim2] = len(all_claims)
                    all_claims.append(claim2)
            
            # Get embeddings for all unique claims
            embeddings = await self.encode_claims(all_claims)
            
            # Calculate similarities for each pair
            similarities = []
            for claim1, claim2 in claim_pairs:
                idx1 = claim_to_index[claim1]
                idx2 = claim_to_index[claim2]
                
                similarity = cosine_similarity(
                    [embeddings[idx1]], [embeddings[idx2]]
                )[0][0]
                similarities.append(float(similarity))
            
            return similarities
            
        except Exception as e:
            logger.error(f"Error in batch similarity analysis: {str(e)}")
            return [0.0] * len(claim_pairs)