# backend/app/services/trend_detector.py

import numpy as np
from typing import List, Dict, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import re
from sqlalchemy.orm import Session
from ..models.claim import Claim
from ..models.verification import Verification
from ..database import get_db
import logging

logger = logging.getLogger(__name__)

class TrendDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        self.clustering_model = DBSCAN(eps=0.3, min_samples=3, metric='cosine')
        
    def detect_trending_topics(self, time_window_hours: int = 24) -> List[Dict[str, Any]]:
        """Detect trending topics in claims within a time window"""
        db = next(get_db())
        
        try:
            # Get recent claims
            cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
            recent_claims = db.query(Claim).filter(
                Claim.created_at >= cutoff_time
            ).all()
            
            if len(recent_claims) < 5:
                return []
            
            # Extract and preprocess text
            claim_texts = [self._preprocess_text(claim.text) for claim in recent_claims]
            
            # Vectorize claims
            tfidf_matrix = self.vectorizer.fit_transform(claim_texts)
            
            # Cluster similar claims
            clusters = self.clustering_model.fit_predict(tfidf_matrix.toarray())
            
            # Analyze trends
            trends = self._analyze_clusters(recent_claims, clusters, tfidf_matrix)
            
            return sorted(trends, key=lambda x: x['trend_score'], reverse=True)
        
        except Exception as e:
            logger.error(f"Error detecting trends: {str(e)}")
            return []
        finally:
            db.close()
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _analyze_clusters(self, claims: List[Claim], clusters: np.ndarray, tfidf_matrix) -> List[Dict[str, Any]]:
        """Analyze clusters to identify trends"""
        trends = []
        cluster_dict = defaultdict(list)
        
        # Group claims by cluster
        for i, cluster_id in enumerate(clusters):
            if cluster_id != -1:  # -1 indicates noise/outlier
                cluster_dict[cluster_id].append((claims[i], i))
        
        # Analyze each cluster
        for cluster_id, cluster_claims in cluster_dict.items():
            if len(cluster_claims) < 3:  # Skip small clusters
                continue
            
            # Calculate trend metrics
            trend_data = self._calculate_trend_metrics(cluster_claims, tfidf_matrix)
            
            if trend_data:
                trends.append(trend_data)
        
        return trends
    
    def _calculate_trend_metrics(self, cluster_claims: List[Tuple[Claim, int]], tfidf_matrix) -> Dict[str, Any]:
        """Calculate trend metrics for a cluster"""
        claims, indices = zip(*cluster_claims)
        
        # Time-based analysis
        timestamps = [claim.created_at for claim in claims]
        time_span = max(timestamps) - min(timestamps)
        
        # Calculate velocity (claims per hour)
        velocity = len(claims) / max(time_span.total_seconds() / 3600, 1)
        
        # Calculate engagement metrics
        total_engagement = sum(claim.engagement_score or 0 for claim in claims)
        avg_engagement = total_engagement / len(claims)
        
        # Extract key terms
        cluster_vectors = tfidf_matrix[indices]
        avg_vector = np.mean(cluster_vectors.toarray(), axis=0)
        feature_names = self.vectorizer.get_feature_names_out()
        top_indices = np.argsort(avg_vector)[-10:][::-1]
        key_terms = [feature_names[i] for i in top_indices if avg_vector[i] > 0]
        
        # Calculate trend score
        trend_score = self._calculate_trend_score(velocity, avg_engagement, len(claims))
        
        # Analyze verification status
        verification_stats = self._analyze_verification_status(claims)
        
        return {
            'cluster_id': f"trend_{hash(str(sorted(key_terms[:3]))) % 10000}",
            'key_terms': key_terms[:5],
            'claim_count': len(claims),
            'velocity': velocity,
            'avg_engagement': avg_engagement,
            'total_engagement': total_engagement,
            'trend_score': trend_score,
            'time_span_hours': time_span.total_seconds() / 3600,
            'verification_stats': verification_stats,
            'sample_claims': [claim.text[:100] + '...' for claim in claims[:3]],
            'first_seen': min(timestamps),
            'last_seen': max(timestamps)
        }
    
    def _calculate_trend_score(self, velocity: float, avg_engagement: float, claim_count: int) -> float:
        """Calculate overall trend score"""
        # Normalize metrics
        velocity_score = min(velocity / 10, 1.0)  # Normalize to 0-1
        engagement_score = min(avg_engagement / 1000, 1.0)  # Normalize to 0-1
        volume_score = min(claim_count / 50, 1.0)  # Normalize to 0-1
        
        # Weighted combination
        trend_score = (
            velocity_score * 0.4 +
            engagement_score * 0.3 +
            volume_score * 0.3
        )
        
        return round(trend_score, 3)
    
    def _analyze_verification_status(self, claims: List[Claim]) -> Dict[str, Any]:
        """Analyze verification status of claims in cluster"""
        db = next(get_db())
        
        try:
            verification_counts = Counter()
            for claim in claims:
                verification = db.query(Verification).filter(
                    Verification.claim_id == claim.id
                ).first()
                
                if verification:
                    verification_counts[verification.verdict] += 1
                else:
                    verification_counts['UNVERIFIED'] += 1
            
            total = len(claims)
            return {
                'total_claims': total,
                'verified_counts': dict(verification_counts),
                'false_percentage': (verification_counts.get('FALSE', 0) / total) * 100,
                'true_percentage': (verification_counts.get('TRUE', 0) / total) * 100,
                'unverified_percentage': (verification_counts.get('UNVERIFIED', 0) / total) * 100
            }
        
        except Exception as e:
            logger.error(f"Error analyzing verification status: {str(e)}")
            return {}
        finally:
            db.close()
    
    def detect_emerging_narratives(self, lookback_days: int = 7) -> List[Dict[str, Any]]:
        """Detect emerging false narratives"""
        db = next(get_db())
        
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=lookback_days)
            
            # Get false claims from the period
            false_claims = db.query(Claim).join(Verification).filter(
                Claim.created_at >= cutoff_time,
                Verification.verdict == 'FALSE'
            ).all()
            
            if len(false_claims) < 5:
                return []
            
            # Group by time periods to detect emergence
            narratives = self._detect_narrative_emergence(false_claims)
            
            return narratives
        
        except Exception as e:
            logger.error(f"Error detecting emerging narratives: {str(e)}")
            return []
        finally:
            db.close()
    
    def _detect_narrative_emergence(self, claims: List[Claim]) -> List[Dict[str, Any]]:
        """Detect emerging narrative patterns"""
        # Split claims into time buckets
        time_buckets = defaultdict(list)
        
        for claim in claims:
            # Group by day
            day_key = claim.created_at.date()
            time_buckets[day_key].append(claim)
        
        # Analyze growth patterns
        narratives = []
        sorted_days = sorted(time_buckets.keys())
        
        for i in range(1, len(sorted_days)):
            prev_day = sorted_days[i-1]
            curr_day = sorted_days[i]
            
            prev_claims = time_buckets[prev_day]
            curr_claims = time_buckets[curr_day]
            
            # Check for significant growth
            if len(curr_claims) > len(prev_claims) * 2:  # 100% growth threshold
                # Analyze content similarity
                combined_claims = prev_claims + curr_claims
                similar_groups = self._find_similar_claims(combined_claims)
                
                for group in similar_groups:
                    if len(group) >= 3:  # Minimum cluster size
                        narratives.append({
                            'narrative_id': f"narrative_{hash(str(group[0].text[:50])) % 10000}",
                            'emergence_date': curr_day,
                            'growth_rate': len(curr_claims) / max(len(prev_claims), 1),
                            'claim_count': len(group),
                            'sample_text': group[0].text[:200],
                            'claims': [claim.text[:100] for claim in group[:3]]
                        })
        
        return narratives
    
    def _find_similar_claims(self, claims: List[Claim]) -> List[List[Claim]]:
        """Find groups of similar claims"""
        if len(claims) < 2:
            return []
        
        texts = [claim.text for claim in claims]
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Find similar claims (threshold-based clustering)
        groups = []
        used_indices = set()
        
        for i in range(len(claims)):
            if i in used_indices:
                continue
            
            similar_indices = [i]
            for j in range(i+1, len(claims)):
                if j not in used_indices and similarity_matrix[i][j] > 0.3:
                    similar_indices.append(j)
            
            if len(similar_indices) >= 2:
                groups.append([claims[idx] for idx in similar_indices])
                used_indices.update(similar_indices)
        
        return groups
    
    def get_trend_analytics(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive trend analytics"""
        trends = self.detect_trending_topics(time_window_hours)
        emerging = self.detect_emerging_narratives()
        
        return {
            'trending_topics': trends,
            'emerging_narratives': emerging,
            'analytics': {
                'total_trends': len(trends),
                'high_risk_trends': len([t for t in trends if t.get('verification_stats', {}).get('false_percentage', 0) > 70]),
                'emerging_count': len(emerging),
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
        }

# Global trend detector instance
trend_detector = TrendDetector()