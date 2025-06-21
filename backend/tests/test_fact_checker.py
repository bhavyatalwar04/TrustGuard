"""
Tests for TruthGuard Fact Checker service
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import asyncio

from app.services.fact_checker import FactChecker, VerificationResult
from app.models.claim import Claim
from app.models.verification import Verification

class TestFactChecker:
    """Test FactChecker service functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.fact_checker = FactChecker()
        self.sample_claim = Claim(
            id="test-claim-123",
            text="The Earth revolves around the Sun",
            source_url="https://example.com/article",
            submitted_by="user123",
            created_at=datetime.now()
        )
    
    def test_fact_checker_initialization(self):
        """Test FactChecker initialization"""
        assert self.fact_checker is not None
        assert hasattr(self.fact_checker, 'knowledge_graph')
        assert hasattr(self.fact_checker, 'semantic_matcher')
    
    @pytest.mark.asyncio
    async def test_verify_claim_success(self):
        """Test successful claim verification"""
        with patch.object(self.fact_checker, '_search_knowledge_base') as mock_search, \
             patch.object(self.fact_checker, '_check_external_sources') as mock_external, \
             patch.object(self.fact_checker, '_calculate_confidence') as mock_confidence:
            
            # Mock knowledge base search results
            mock_search.return_value = [
                {
                    "text": "Earth orbits the Sun in elliptical path",
                    "source": "NASA",
                    "confidence": 0.95,
                    "supporting_evidence": ["astronomical observations", "gravitational theory"]
                }
            ]
            
            # Mock external source verification
            mock_external.return_value = [
                {
                    "source": "Scientific journals",
                    "verification": "SUPPORTS",
                    "confidence": 0.92
                }
            ]
            
            # Mock confidence calculation
            mock_confidence.return_value = 0.94
            
            result = await self.fact_checker.verify_claim(self.sample_claim)
            
            assert isinstance(result, VerificationResult)
            assert result.claim_id == "test-claim-123"
            assert result.result in ["TRUE", "FALSE", "PARTIALLY_TRUE", "UNVERIFIED"]
            assert 0 <= result.confidence_score <= 1
            assert len(result.evidence) > 0
    
    @pytest.mark.asyncio
    async def test_verify_claim_false(self):
        """Test verification of false claim"""
        false_claim = Claim(
            id="false-claim-123",
            text="The Earth is flat",
            source_url="https://conspiracy.com",
            submitted_by="user456"
        )
        
        with patch.object(self.fact_checker, '_search_knowledge_base') as mock_search, \
             patch.object(self.fact_checker, '_check_external_sources') as mock_external, \
             patch.object(self.fact_checker, '_calculate_confidence') as mock_confidence:
            
            # Mock contradicting evidence
            mock_search.return_value = [
                {
                    "text": "Earth is a sphere confirmed by satellite images",
                    "source": "Multiple space agencies",
                    "confidence": 0.99,
                    "contradicts": True
                }
            ]
            
            mock_external.return_value = [
                {
                    "source": "Scientific consensus",
                    "verification": "CONTRADICTS",
                    "confidence": 0.97
                }
            ]
            
            mock_confidence.return_value = 0.96
            
            result = await self.fact_checker.verify_claim(false_claim)
            
            assert result.result == "FALSE"
            assert result.confidence_score > 0.9
    
    def test_search_knowledge_base(self):
        """Test knowledge base search functionality"""
        with patch.object(self.fact_checker.knowledge_graph, 'search_facts') as mock_search:
            mock_search.return_value = [
                {
                    "fact": "Earth-Sun orbital relationship",
                    "confidence": 0.95,
                    "sources": ["NASA", "ESA"]
                }
            ]
            
            results = self.fact_checker._search_knowledge_base("Earth orbits Sun")
            
            assert len(results) > 0
            assert results[0]["confidence"] > 0.9
            mock_search.assert_called_once()
    
    def test_check_external_sources(self):
        """Test external source verification"""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "results": [
                    {
                        "title": "Scientific article about Earth's orbit",
                        "snippet": "Earth revolves around Sun in elliptical orbit",
                        "source": "Scientific Journal",
                        "credibility": 0.9
                    }
                ]
            }
            mock_get.return_value = mock_response
            
            results = self.fact_checker._check_external_sources("Earth orbits Sun")
            
            assert len(results) > 0
            assert "Scientific" in results[0]["source"]
    
    def test_calculate_confidence_high(self):
        """Test confidence calculation with strong evidence"""
        evidence_scores = [0.95, 0.92, 0.88, 0.94]
        source_reliability = [0.9, 0.85, 0.92, 0.88]
        
        confidence = self.fact_checker._calculate_confidence(
            evidence_scores, 
            source_reliability
        )
        
        assert confidence > 0.85
        assert confidence <= 1.0
    
    def test_calculate_confidence_low(self):
        """Test confidence calculation with weak evidence"""
        evidence_scores = [0.3, 0.4, 0.2]
        source_reliability = [0.5, 0.4, 0.6]
        
        confidence = self.fact_checker._calculate_confidence(
            evidence_scores,
            source_reliability
        )
        
        assert confidence < 0.5
        assert confidence >= 0.0
    
    def test_extract_entities_from_claim(self):
        """Test entity extraction from claims"""
        claim_text = "President Biden signed the infrastructure bill in 2021"
        
        entities = self.fact_checker._extract_entities(claim_text)
        
        assert "persons" in entities
        assert "dates" in entities
        assert len(entities["persons"]) > 0 or len(entities["dates"]) > 0
    
    def test_find_contradictory_claims(self):
        """Test finding contradictory claims"""
        with patch.object(self.fact_checker.semantic_matcher, 'find_contradictions') as mock_find:
            mock_find.return_value = [
                {
                    "claim_id": "contradictory-claim-1",
                    "text": "The Earth is flat",
                    "contradiction_score": 0.85
                }
            ]
            
            contradictions = self.fact_checker._find_contradictory_claims(
                "The Earth is round"
            )
            
            assert len(contradictions) > 0
            assert contradictions[0]["contradiction_score"] > 0.8
    
    def test_check_source_credibility(self):
        """Test source credibility assessment"""
        reliable_source = "https://www.nasa.gov/article"
        unreliable_source = "https://conspiracy-theories.net/article"
        
        reliable_score = self.fact_checker._check_source_credibility(reliable_source)
        unreliable_score = self.fact_checker._check_source_credibility(unreliable_source)
        
        assert reliable_score > unreliable_score
        assert 0 <= reliable_score <= 1
        assert 0 <= unreliable_score <= 1
    
    def test_temporal_consistency_check(self):
        """Test temporal consistency of claims"""
        current_claim = "The current president is Joe Biden"
        timestamp = datetime(2023, 6, 1)
        
        with patch.object(self.fact_checker, '_get_temporal_facts') as mock_temporal:
            mock_temporal.return_value = [
                {
                    "fact": "Joe Biden inaugurated as president January 2021",
                    "valid_from": datetime(2021, 1, 20),
                    "valid_until": None
                }
            ]
            
            is_consistent = self.fact_checker._check_temporal_consistency(
                current_claim, 
                timestamp
            )
            
            assert is_consistent is True
    
    def test_handle_ambiguous_claims(self):
        """Test handling of ambiguous or unclear claims"""
        ambiguous_claim = Claim(
            id="ambiguous-claim-123",
            text="It is raining",  # Ambiguous without location/time context
            source_url="https://example.com",
            submitted_by="user789"
        )
        
        with patch.object(self.fact_checker, '_analyze_ambiguity') as mock_analyze:
            mock_analyze.return_value = {
                "ambiguity_score": 0.8,
                "missing_context": ["location", "time"],
                "possible_interpretations": 3
            }
            
            result = asyncio.run(self.fact_checker.verify_claim(ambiguous_claim))
            
            assert result.result == "UNVERIFIED"
            assert "ambiguous" in result.explanation.lower()
    
    def test_batch_verification(self):
        """Test batch processing of multiple claims"""
        claims = [
            Claim(id=f"claim-{i}", text=f"Test claim {i}", source_url="https://example.com")
            for i in range(5)
        ]
        
        with patch.object(self.fact_checker, 'verify_claim') as mock_verify:
            mock_verify.return_value = VerificationResult(
                claim_id="test",
                result="TRUE",
                confidence_score=0.8,
                evidence=[],
                sources=[],
                explanation="Test verification"
            )
            
            results = asyncio.run(self.fact_checker.verify_claims_batch(claims))
            
            assert len(results) == 5
            assert all(isinstance(r, VerificationResult) for r in results)
    
    def test_verify_claim_with_context(self):
        """Test claim verification with additional context"""
        claim_with_context = Claim(
            id="contextual-claim-123",
            text="The temperature increased by 5 degrees",
            source_url="https://weather.com",
            submitted_by="user123",
            context={
                "location": "New York",
                "date": "2023-06-15",
                "time_period": "last week"
            }
        )
        
        with patch.object(self.fact_checker, '_verify_with_context') as mock_context:
            mock_context.return_value = VerificationResult(
                claim_id="contextual-claim-123",
                result="TRUE",
                confidence_score=0.87,
                evidence=["Weather station data confirms temperature increase"],
                sources=["NOAA Weather Data"],
                explanation="Temperature increase verified with local weather data"
            )
            
            result = asyncio.run(self.fact_checker.verify_claim(claim_with_context))
            
            assert result.confidence_score > 0.8
            assert "weather data" in result.explanation.lower()
    
    def test_error_handling_invalid_claim(self):
        """Test error handling for invalid claims"""
        invalid_claim = None
        
        with pytest.raises(ValueError):
            asyncio.run(self.fact_checker.verify_claim(invalid_claim))
    
    def test_error_handling_network_failure(self):
        """Test error handling for network failures"""
        with patch.object(self.fact_checker, '_check_external_sources') as mock_external:
            mock_external.side_effect = ConnectionError("Network unreachable")
            
            result = asyncio.run(self.fact_checker.verify_claim(self.sample_claim))
            
            # Should still return a result, possibly with lower confidence
            assert isinstance(result, VerificationResult)
            assert result.confidence_score >= 0
    
    def test_get_verification_explanation(self):
        """Test generation of verification explanations"""
        verification_data = {
            "result": "FALSE",
            "confidence": 0.92,
            "supporting_evidence": 2,
            "contradicting_evidence": 5,
            "source_reliability": 0.88
        }
        
        explanation = self.fact_checker._generate_explanation(verification_data)
        
        assert isinstance(explanation, str)
        assert len(explanation) > 50
        assert "evidence" in explanation.lower()
    
    def test_update_knowledge_base(self):
        """Test knowledge base updates from verifications"""
        verification_result = VerificationResult(
            claim_id="test-claim-123",
            result="TRUE",
            confidence_score=0.94,
            evidence=["Scientific consensus", "Multiple studies"],
            sources=["Nature", "Science Magazine"],
            explanation="Well-established scientific fact"
        )
        
        with patch.object(self.fact_checker.knowledge_graph, 'add_verified_fact') as mock_add:
            self.fact_checker._update_knowledge_base(verification_result)
            mock_add.assert_called_once()
    
    def test_performance_metrics(self):
        """Test performance metrics collection"""
        with patch.object(self.fact_checker, '_record_performance_metrics') as mock_metrics:
            asyncio.run(self.fact_checker.verify_claim(self.sample_claim))
            mock_metrics.assert_called()
    
    def test_cache_verification_results(self):
        """Test caching of verification results"""
        with patch.object(self.fact_checker, '_cache_result') as mock_cache, \
             patch.object(self.fact_checker, '_get_cached_result') as mock_get_cache:
            
            mock_get_cache.return_value = None  # No cached result
            
            result = asyncio.run(self.fact_checker.verify_claim(self.sample_claim))
            
            mock_cache.assert_called_once_with(self.sample_claim.id, result)

if __name__ == "__main__":
    pytest.main([__file__])