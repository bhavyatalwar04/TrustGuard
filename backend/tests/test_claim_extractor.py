"""
Test cases for claim extractor service
Location: TruthGuard/backend/tests/test_claim_extractor.py
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from app.services.claim_extractor import ClaimExtractor
from app.models.claim import Claim

class TestClaimExtractor:
    def setup_method(self):
        """Setup test fixtures"""
        self.claim_extractor = ClaimExtractor()
        
    def test_extract_claims_from_text(self):
        """Test basic claim extraction from text"""
        text = "The Earth is flat. Climate change is a hoax. Vaccines cause autism."
        
        claims = self.claim_extractor.extract_claims(text)
        
        assert len(claims) >= 2
        assert any("Earth is flat" in claim for claim in claims)
        assert any("climate change" in claim.lower() for claim in claims)
        
    def test_extract_claims_from_empty_text(self):
        """Test claim extraction from empty text"""
        text = ""
        
        claims = self.claim_extractor.extract_claims(text)
        
        assert claims == []
        
    def test_extract_claims_with_confidence_scores(self):
        """Test that extracted claims have confidence scores"""
        text = "Scientists have proven that coffee is bad for health."
        
        claims_with_scores = self.claim_extractor.extract_claims_with_confidence(text)
        
        assert len(claims_with_scores) > 0
        for claim, confidence in claims_with_scores:
            assert isinstance(claim, str)
            assert 0 <= confidence <= 1
            
    def test_extract_claims_from_social_media_text(self):
        """Test claim extraction from social media style text"""
        text = """
        OMG just read that 5G towers cause coronavirus!!! 🤯
        Also heard that Bill Gates wants to microchip everyone through vaccines.
        #truth #wakeup
        """
        
        claims = self.claim_extractor.extract_claims(text)
        
        assert len(claims) >= 1
        assert any("5G" in claim for claim in claims)
        
    def test_extract_claims_filters_opinions(self):
        """Test that opinions are filtered out from factual claims"""
        text = "I think pizza is the best food. The population of India is over 1 billion."
        
        claims = self.claim_extractor.extract_claims(text)
        
        # Should extract the factual claim about India's population
        # but not the opinion about pizza
        assert len(claims) >= 1
        assert not any("pizza is the best" in claim.lower() for claim in claims)
        assert any("population" in claim.lower() for claim in claims)
        
    @patch('app.services.claim_extractor.spacy')
    def test_extract_claims_with_nlp_processing(self, mock_spacy):
        """Test claim extraction with NLP processing"""
        mock_nlp = Mock()
        mock_doc = Mock()
        mock_doc.sents = [Mock(text="Test sentence.")]
        mock_nlp.return_value = mock_doc
        mock_spacy.load.return_value = mock_nlp
        
        text = "This is a test sentence."
        claims = self.claim_extractor.extract_claims(text)
        
        mock_spacy.load.assert_called_once()
        mock_nlp.assert_called_once_with(text)
        
    def test_extract_claims_handles_multiple_languages(self):
        """Test claim extraction from multiple languages"""
        # English
        english_text = "The moon landing was fake."
        english_claims = self.claim_extractor.extract_claims(english_text)
        
        # Spanish (if supported)
        spanish_text = "El cambio climático es real."
        spanish_claims = self.claim_extractor.extract_claims(spanish_text)
        
        assert len(english_claims) > 0
        # Note: Spanish support depends on model capabilities
        
    def test_extract_claims_with_context(self):
        """Test claim extraction with context preservation"""
        text = """
        According to a recent study published in Nature,
        the average global temperature has increased by 1.2°C since 1880.
        However, some skeptics argue that this data is manipulated.
        """
        
        claims_with_context = self.claim_extractor.extract_claims_with_context(text)
        
        assert len(claims_with_context) > 0
        for claim_data in claims_with_context:
            assert 'claim' in claim_data
            assert 'context' in claim_data
            assert 'position' in claim_data
            
    def test_claim_priority_scoring(self):
        """Test that claims are scored by priority/importance"""
        text = """
        I had coffee this morning.
        COVID-19 vaccines contain microchips.
        The weather is nice today.
        """
        
        prioritized_claims = self.claim_extractor.extract_claims_prioritized(text)
        
        # Medical/health claims should have higher priority
        assert len(prioritized_claims) > 0
        # Vaccine claim should be prioritized over casual statements
        top_claim = prioritized_claims[0]
        assert "vaccine" in top_claim['claim'].lower() or "covid" in top_claim['claim'].lower()
        
    @pytest.mark.asyncio
    async def test_batch_claim_extraction(self):
        """Test batch processing of multiple texts"""
        texts = [
            "The Earth is round.",
            "Vaccines are safe and effective.",
            "Climate change is caused by human activities."
        ]
        
        batch_results = await self.claim_extractor.extract_claims_batch(texts)
        
        assert len(batch_results) == len(texts)
        for result in batch_results:
            assert isinstance(result, list)
            
    def test_claim_categorization(self):
        """Test that claims are categorized by topic"""
        text = """
        COVID-19 vaccines are 95% effective.
        The 2020 election was rigged.
        Climate change will cause sea levels to rise by 2 meters.
        """
        
        categorized_claims = self.claim_extractor.extract_and_categorize_claims(text)
        
        categories = set(claim['category'] for claim in categorized_claims)
        expected_categories = {'health', 'politics', 'environment'}
        
        assert len(categories.intersection(expected_categories)) > 0
        
    def test_claim_deduplication(self):
        """Test that duplicate claims are removed"""
        text = """
        The Earth is flat. The Earth is flat.
        Our planet is flat. The world is flat.
        """
        
        claims = self.claim_extractor.extract_claims(text, deduplicate=True)
        
        # Should have fewer claims due to deduplication
        assert len(claims) < 4
        
    def test_extract_claims_with_metadata(self):
        """Test extraction with metadata like source, timestamp"""
        text = "Breaking: Scientists discover new planet."
        metadata = {
            'source': 'Twitter',
            'author': 'test_user',
            'timestamp': '2024-01-15T10:30:00Z'
        }
        
        claims_with_metadata = self.claim_extractor.extract_claims_with_metadata(
            text, metadata
        )
        
        assert len(claims_with_metadata) > 0
        for claim_data in claims_with_metadata:
            assert claim_data['metadata']['source'] == 'Twitter'
            assert claim_data['metadata']['author'] == 'test_user'
            
    def test_error_handling_malformed_input(self):
        """Test error handling for malformed input"""
        malformed_inputs = [None, 123, {'invalid': 'input'}, [1, 2, 3]]
        
        for invalid_input in malformed_inputs:
            try:
                claims = self.claim_extractor.extract_claims(invalid_input)
                assert claims == []  # Should return empty list for invalid input
            except (TypeError, ValueError):
                pass  # Acceptable to raise an exception
                
    def test_performance_with_large_text(self):
        """Test performance with large text input"""
        large_text = "This is a test sentence. " * 1000
        
        import time
        start_time = time.time()
        claims = self.claim_extractor.extract_claims(large_text)
        end_time = time.time()
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert end_time - start_time < 10.0
        assert isinstance(claims, list)