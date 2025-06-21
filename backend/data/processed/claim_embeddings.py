# File: backend/data/processed/claim_embeddings.py
"""
Claim embeddings processing and storage utilities
"""

import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

class ClaimEmbeddingsManager:
    """Manages claim embeddings for semantic similarity and clustering"""
    
    def __init__(self, embeddings_path: str = "claim_embeddings.pkl"):
        self.embeddings_path = Path(embeddings_path)
        self.embeddings_cache = {}
        self.metadata_cache = {}
        
    def save_embeddings(self, claim_id: str, embedding: np.ndarray, 
                       model_name: str, metadata: Dict = None):
        """Save claim embedding with metadata"""
        try:
            # Load existing embeddings if file exists
            if self.embeddings_path.exists():
                with open(self.embeddings_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                data = {'embeddings': {}, 'metadata': {}}
            
            # Update with new embedding
            data['embeddings'][claim_id] = {
                'vector': embedding,
                'model': model_name,
                'timestamp': datetime.now().isoformat(),
                'dimension': embedding.shape[0]
            }
            
            if metadata:
                data['metadata'][claim_id] = metadata
            
            # Save back to file
            with open(self.embeddings_path, 'wb') as f:
                pickle.dump(data, f)
                
            return True
            
        except Exception as e:
            print(f"Error saving embeddings: {e}")
            return False
    
    def load_embeddings(self, claim_ids: List[str] = None) -> Dict:
        """Load embeddings for specific claims or all claims"""
        try:
            if not self.embeddings_path.exists():
                return {}
                
            with open(self.embeddings_path, 'rb') as f:
                data = pickle.load(f)
            
            if claim_ids:
                filtered_data = {
                    'embeddings': {cid: data['embeddings'][cid] 
                                 for cid in claim_ids 
                                 if cid in data['embeddings']},
                    'metadata': {cid: data['metadata'].get(cid, {}) 
                               for cid in claim_ids}
                }
                return filtered_data
            
            return data
            
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            return {}
    
    def find_similar_claims(self, query_embedding: np.ndarray, 
                           top_k: int = 5, 
                           threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Find similar claims based on embedding similarity"""
        try:
            data = self.load_embeddings()
            if not data or 'embeddings' not in data:
                return []
            
            similarities = []
            
            for claim_id, embedding_data in data['embeddings'].items():
                stored_embedding = embedding_data['vector']
                
                # Calculate cosine similarity
                similarity = self.cosine_similarity(query_embedding, stored_embedding)
                
                if similarity >= threshold:
                    similarities.append((claim_id, similarity))
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            print(f"Error finding similar claims: {e}")
            return []
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def cluster_claims(self, num_clusters: int = 10) -> Dict:
        """Cluster claims based on their embeddings"""
        try:
            from sklearn.cluster import KMeans
            
            data = self.load_embeddings()
            if not data or 'embeddings' not in data:
                return {}
            
            # Prepare embeddings matrix
            claim_ids = list(data['embeddings'].keys())
            embeddings_matrix = np.array([
                data['embeddings'][cid]['vector'] 
                for cid in claim_ids
            ])
            
            # Perform clustering
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings_matrix)
            
            # Group claims by cluster
            clusters = {}
            for claim_id, cluster_id in zip(claim_ids, cluster_labels):
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(claim_id)
            
            return {
                'clusters': clusters,
                'centroids': kmeans.cluster_centers_,
                'num_clusters': num_clusters,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error clustering claims: {e}")
            return {}
    
    def get_embedding_stats(self) -> Dict:
        """Get statistics about stored embeddings"""
        try:
            data = self.load_embeddings()
            if not data or 'embeddings' not in data:
                return {}
            
            embeddings = data['embeddings']
            
            stats = {
                'total_embeddings': len(embeddings),
                'models_used': list(set(emb['model'] for emb in embeddings.values())),
                'dimensions': list(set(emb['dimension'] for emb in embeddings.values())),
                'oldest_embedding': min(emb['timestamp'] for emb in embeddings.values()),
                'newest_embedding': max(emb['timestamp'] for emb in embeddings.values()),
                'average_dimension': np.mean([emb['dimension'] for emb in embeddings.values()])
            }
            
            return stats
            
        except Exception as e:
            print(f"Error getting embedding stats: {e}")
            return {}

# Sample usage and data structure
if __name__ == "__main__":
    # Initialize manager
    manager = ClaimEmbeddingsManager()
    
    # Sample embedding data (would come from actual embedding models)
    sample_embeddings = {
        "claim_001": np.random.rand(384),  # sentence-transformers dimension
        "claim_002": np.random.rand(384),
        "claim_003": np.random.rand(384)
    }
    
    # Save sample embeddings
    for claim_id, embedding in sample_embeddings.items():
        manager.save_embeddings(
            claim_id=claim_id,
            embedding=embedding,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            metadata={
                "text_length": np.random.randint(50, 200),
                "language": "en",
                "category": "health" if "001" in claim_id else "climate"
            }
        )
    
    print("Sample embeddings saved successfully!")
    print("Embedding stats:", manager.get_embedding_stats())