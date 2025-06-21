"""
Tests for TruthGuard API endpoints
"""
import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from datetime import datetime

from app.main import app
from app.database import get_db
from app.models.claim import Claim
from app.models.verification import Verification

# Mock database dependency
def override_get_db():
    return Mock()

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_check(self):
        """Test health check returns OK"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy", "service": "TruthGuard API"}

class TestClaimEndpoints:
    """Test claim-related endpoints"""
    
    def test_submit_claim_success(self):
        """Test successful claim submission"""
        claim_data = {
            "text": "The Earth is round",
            "source_url": "https://example.com/article",
            "submitted_by": "user123"
        }
        
        with patch('app.services.claim_extractor.ClaimExtractor.extract_claims') as mock_extract:
            mock_extract.return_value = [claim_data["text"]]
            
            response = client.post("/api/claims/submit", json=claim_data)
            assert response.status_code == 201
            
            result = response.json()
            assert "claim_id" in result
            assert result["status"] == "submitted"
    
    def test_submit_claim_invalid_data(self):
        """Test claim submission with invalid data"""
        invalid_data = {
            "text": "",  # Empty text
            "source_url": "invalid-url",
            "submitted_by": ""
        }
        
        response = client.post("/api/claims/submit", json=invalid_data)
        assert response.status_code == 422
    
    def test_get_claim_by_id(self):
        """Test retrieving claim by ID"""
        claim_id = "test-claim-123"
        
        with patch('app.database.get_db') as mock_db:
            mock_session = Mock()
            mock_db.return_value = mock_session
            
            mock_claim = Mock()
            mock_claim.id = claim_id
            mock_claim.text = "Test claim"
            mock_claim.status = "verified"
            mock_claim.confidence_score = 0.85
            mock_claim.created_at = datetime.now()
            
            mock_session.query.return_value.filter.return_value.first.return_value = mock_claim
            
            response = client.get(f"/api/claims/{claim_id}")
            assert response.status_code == 200
            
            result = response.json()
            assert result["id"] == claim_id
            assert result["text"] == "Test claim"
    
    def test_get_claim_not_found(self):
        """Test retrieving non-existent claim"""
        claim_id = "non-existent-claim"
        
        with patch('app.database.get_db') as mock_db:
            mock_session = Mock()
            mock_db.return_value = mock_session
            mock_session.query.return_value.filter.return_value.first.return_value = None
            
            response = client.get(f"/api/claims/{claim_id}")
            assert response.status_code == 404
    
    def test_get_claims_list(self):
        """Test retrieving list of claims"""
        with patch('app.database.get_db') as mock_db:
            mock_session = Mock()
            mock_db.return_value = mock_session
            
            mock_claims = [
                Mock(id="claim1", text="Claim 1", status="verified"),
                Mock(id="claim2", text="Claim 2", status="pending")
            ]
            
            mock_session.query.return_value.offset.return_value.limit.return_value.all.return_value = mock_claims
            mock_session.query.return_value.count.return_value = 2
            
            response = client.get("/api/claims/")
            assert response.status_code == 200
            
            result = response.json()
            assert len(result["claims"]) == 2
            assert result["total"] == 2

class TestVerificationEndpoints:
    """Test verification-related endpoints"""
    
    def test_verify_claim(self):
        """Test claim verification endpoint"""
        claim_id = "test-claim-123"
        
        with patch('app.services.fact_checker.FactChecker.verify_claim') as mock_verify:
            mock_verify.return_value = {
                "claim_id": claim_id,
                "verification_result": "TRUE",
                "confidence_score": 0.92,
                "evidence": ["Evidence 1", "Evidence 2"],
                "sources": ["https://source1.com", "https://source2.com"]
            }
            
            response = client.post(f"/api/claims/{claim_id}/verify")
            assert response.status_code == 200
            
            result = response.json()
            assert result["verification_result"] == "TRUE"
            assert result["confidence_score"] == 0.92
    
    def test_get_verification_history(self):
        """Test retrieving verification history"""
        claim_id = "test-claim-123"
        
        with patch('app.database.get_db') as mock_db:
            mock_session = Mock()
            mock_db.return_value = mock_session
            
            mock_verifications = [
                Mock(
                    id="v1",
                    claim_id=claim_id,
                    result="TRUE",
                    confidence_score=0.85,
                    created_at=datetime.now()
                )
            ]
            
            mock_session.query.return_value.filter.return_value.all.return_value = mock_verifications
            
            response = client.get(f"/api/claims/{claim_id}/verifications")
            assert response.status_code == 200
            
            result = response.json()
            assert len(result) == 1
            assert result[0]["result"] == "TRUE"

class TestTrendEndpoints:
    """Test trend analysis endpoints"""
    
    def test_get_trending_claims(self):
        """Test retrieving trending claims"""
        with patch('app.services.trend_detector.TrendDetector.get_trending_claims') as mock_trends:
            mock_trends.return_value = [
                {
                    "claim_text": "Trending claim 1",
                    "frequency": 15,
                    "trend_score": 0.8,
                    "time_period": "24h"
                },
                {
                    "claim_text": "Trending claim 2", 
                    "frequency": 12,
                    "trend_score": 0.7,
                    "time_period": "24h"
                }
            ]
            
            response = client.get("/api/trends/claims")
            assert response.status_code == 200
            
            result = response.json()
            assert len(result) == 2
            assert result[0]["frequency"] == 15
    
    def test_get_misinformation_trends(self):
        """Test retrieving misinformation trends"""
        with patch('app.services.trend_detector.TrendDetector.get_misinformation_trends') as mock_misinfo:
            mock_misinfo.return_value = {
                "false_claims": 25,
                "misleading_claims": 18,
                "unverified_claims": 42,
                "trend_direction": "increasing",
                "time_period": "7d"
            }
            
            response = client.get("/api/trends/misinformation")
            assert response.status_code == 200
            
            result = response.json()
            assert result["false_claims"] == 25
            assert result["trend_direction"] == "increasing"

class TestSearchEndpoints:
    """Test search functionality"""
    
    def test_search_claims(self):
        """Test searching claims"""
        search_params = {
            "query": "climate change",
            "limit": 10,
            "offset": 0
        }
        
        with patch('app.services.semantic_matcher.SemanticMatcher.search_similar_claims') as mock_search:
            mock_search.return_value = [
                {
                    "claim_id": "claim1",
                    "text": "Climate change is real",
                    "similarity_score": 0.95,
                    "status": "verified"
                },
                {
                    "claim_id": "claim2", 
                    "text": "Global warming affects weather",
                    "similarity_score": 0.88,
                    "status": "verified"
                }
            ]
            
            response = client.get("/api/search/claims", params=search_params)
            assert response.status_code == 200
            
            result = response.json()
            assert len(result["results"]) == 2
            assert result["results"][0]["similarity_score"] == 0.95
    
    def test_search_claims_empty_query(self):
        """Test search with empty query"""
        response = client.get("/api/search/claims", params={"query": ""})
        assert response.status_code == 400

class TestAlertEndpoints:
    """Test alert system endpoints"""
    
    def test_get_active_alerts(self):
        """Test retrieving active alerts"""
        with patch('app.services.alert_system.AlertSystem.get_active_alerts') as mock_alerts:
            mock_alerts.return_value = [
                {
                    "alert_id": "alert1",
                    "type": "misinformation_spike",
                    "severity": "high",
                    "message": "Sudden increase in false claims detected",
                    "created_at": datetime.now().isoformat()
                }
            ]
            
            response = client.get("/api/alerts/active")
            assert response.status_code == 200
            
            result = response.json()
            assert len(result) == 1
            assert result[0]["severity"] == "high"
    
    def test_create_alert_subscription(self):
        """Test creating alert subscription"""
        subscription_data = {
            "user_id": "user123",
            "alert_types": ["misinformation_spike", "trending_claims"],
            "notification_method": "email",
            "threshold": 0.8
        }
        
        with patch('app.services.alert_system.AlertSystem.create_subscription') as mock_subscribe:
            mock_subscribe.return_value = {
                "subscription_id": "sub123",
                "status": "active"
            }
            
            response = client.post("/api/alerts/subscribe", json=subscription_data)
            assert response.status_code == 201
            
            result = response.json()
            assert result["subscription_id"] == "sub123"

class TestAnalyticsEndpoints:
    """Test analytics endpoints"""
    
    def test_get_verification_stats(self):
        """Test retrieving verification statistics"""
        with patch('app.database.get_db') as mock_db:
            mock_session = Mock()
            mock_db.return_value = mock_session
            
            # Mock database queries for stats
            mock_session.query.return_value.filter.return_value.count.side_effect = [
                150,  # total claims
                85,   # verified claims
                45,   # false claims
                20    # pending claims
            ]
            
            response = client.get("/api/analytics/verification-stats")
            assert response.status_code == 200
            
            result = response.json()
            assert result["total_claims"] == 150
            assert result["verified_claims"] == 85
    
    def test_get_accuracy_metrics(self):
        """Test retrieving accuracy metrics"""
        with patch('app.services.fact_checker.FactChecker.get_accuracy_metrics') as mock_metrics:
            mock_metrics.return_value = {
                "overall_accuracy": 0.87,
                "precision": 0.92,
                "recall": 0.84,
                "f1_score": 0.88,
                "confidence_distribution": {
                    "high": 0.65,
                    "medium": 0.25,
                    "low": 0.10
                }
            }
            
            response = client.get("/api/analytics/accuracy")
            assert response.status_code == 200
            
            result = response.json()
            assert result["overall_accuracy"] == 0.87
            assert result["precision"] == 0.92

class TestErrorHandling:
    """Test error handling scenarios"""
    
    def test_invalid_json_payload(self):
        """Test handling of invalid JSON payload"""
        response = client.post(
            "/api/claims/submit",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_missing_required_fields(self):
        """Test handling of missing required fields"""
        incomplete_data = {"text": "Some claim"}  # Missing required fields
        
        response = client.post("/api/claims/submit", json=incomplete_data)
        assert response.status_code == 422
    
    def test_database_connection_error(self):
        """Test handling of database connection errors"""
        with patch('app.database.get_db') as mock_db:
            mock_db.side_effect = Exception("Database connection failed")
            
            response = client.get("/api/claims/")
            assert response.status_code == 500
    
    def test_service_timeout_error(self):
        """Test handling of service timeout errors"""
        with patch('app.services.fact_checker.FactChecker.verify_claim') as mock_verify:
            mock_verify.side_effect = TimeoutError("Service timeout")
            
            response = client.post("/api/claims/test-claim/verify")
            assert response.status_code == 504

class TestAuthentication:
    """Test authentication and authorization"""
    
    def test_protected_endpoint_without_auth(self):
        """Test accessing protected endpoint without authentication"""
        # Assuming some endpoints require authentication
        response = client.delete("/api/claims/test-claim")
        assert response.status_code == 401
    
    def test_invalid_api_key(self):
        """Test using invalid API key"""
        headers = {"Authorization": "Bearer invalid-key"}
        response = client.get("/api/admin/stats", headers=headers)
        assert response.status_code == 401

class TestRateLimiting:
    """Test rate limiting functionality"""
    
    def test_rate_limit_exceeded(self):
        """Test rate limiting when limit is exceeded"""
        # Simulate multiple rapid requests
        with patch('app.api.routes.rate_limiter') as mock_limiter:
            mock_limiter.side_effect = Exception("Rate limit exceeded")
            
            response = client.post("/api/claims/submit", json={
                "text": "Test claim",
                "source_url": "https://example.com",
                "submitted_by": "user123"
            })
            assert response.status_code == 429

if __name__ == "__main__":
    pytest.main([__file__])