"""
Claim Extractor Service
Extracts and identifies factual claims from text content
"""

import re
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import spacy
from textstat import flesch_reading_ease
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.tree import Tree

from ..config import settings

logger = logging.getLogger(__name__)

@dataclass
class ExtractedClaim:
    """Represents an extracted claim"""
    text: str
    confidence: float
    start_position: int
    end_position: int
    category: str
    keywords: List[str]
    entities: List[Dict[str, str]]
    factual_indicators: List[str]
    sentiment_score: float
    readability_score: float

class ClaimExtractor:
    """
    Service for extracting factual claims from text content
    Uses NLP techniques to identify statements that can be fact-checked
    """
    
    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.nlp = None
        self._initialize_nlp()
        
        # Factual claim patterns
        self.factual_patterns = [
            # Statistical claims
            r'\b\d+(?:\.\d+)?%\b',  # Percentages
            r'\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|trillion)\b',  # Large numbers
            r'\b(?:increased|decreased|rose|fell|dropped)\s+by\s+\d+',  # Change indicators
            
            # Comparative claims
            r'\b(?:more|less|higher|lower|greater|smaller)\s+than\b',
            r'\b(?:most|least|highest|lowest|largest|smallest)\b',
            
            # Causal claims
            r'\b(?:causes?|results? in|leads? to|due to|because of)\b',
            r'\b(?:if|when|whenever|as long as).*(?:then|will|would)\b',
            
            # Temporal claims
            r'\b(?:since|until|by|before|after|during)\s+\d{4}\b',  # Years
            r'\b(?:yesterday|today|tomorrow|last|next|this)\s+(?:week|month|year)\b',
            
            # Definitional claims
            r'\b(?:is|are|was|were|will be|would be)\s+(?:the|a|an)?\s*(?:first|last|only|main|primary)\b',
        ]
        
        # Uncertainty indicators that reduce claim confidence
        self.uncertainty_indicators = [
            r'\b(?:maybe|perhaps|possibly|probably|likely|might|could|may|seem|appear)\b',
            r'\b(?:I think|I believe|I feel|in my opinion|it seems|apparently)\b',
            r'\b(?:allegedly|supposedly|reportedly|rumored|claimed)\b',
        ]
        
        # Strong factual indicators
        self.strong_indicators = [
            r'\b(?:according to|research shows|studies indicate|data reveals)\b',
            r'\b(?:officially|confirmed|verified|established|proven)\b',
            r'\b(?:statistics show|report states|survey finds)\b',
        ]
        
        # Compile patterns
        self.factual_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.factual_patterns]
        self.uncertainty_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.uncertainty_indicators]
        self.strong_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.strong_indicators]
    
    def _initialize_nlp(self):
        """Initialize NLP models"""
        try:
            # Try to load spaCy model
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('maxent_ne_chunker', quiet=True)
            nltk.download('words', quiet=True)
        except Exception as e:
            logger.warning(f"Failed to download NLTK data: {e}")
    
    def extract_claims(self, text: str, min_confidence: float = 0.3) -> List[ExtractedClaim]:
        """
        Extract factual claims from text
        
        Args:
            text: Input text to analyze
            min_confidence: Minimum confidence threshold for claims
            
        Returns:
            List of extracted claims
        """
        if not text.strip():
            return []
        
        # Split text into sentences
        sentences = sent_tokenize(text)
        claims = []
        
        for i, sentence in enumerate(sentences):
            claim = self._analyze_sentence(sentence, text)
            if claim and claim.confidence >= min_confidence:
                claims.append(claim)
        
        # Sort by confidence
        claims.sort(key=lambda x: x.confidence, reverse=True)
        
        return claims
    
    def _analyze_sentence(self, sentence: str, full_text: str) -> Optional[ExtractedClaim]:
        """
        Analyze a single sentence for factual claims
        
        Args:
            sentence: Sentence to analyze
            full_text: Full text context
            
        Returns:
            ExtractedClaim object or None
        """
        # Basic filtering
        if len(sentence.split()) < 5:  # Too short
            return None
        
        if sentence.strip().endswith('?'):  # Questions are not claims
            return None
        
        # Calculate confidence score
        confidence = self._calculate_confidence(sentence)
        
        if confidence < 0.1:  # Very low confidence
            return None
        
        # Extract features
        keywords = self._extract_keywords(sentence)
        entities = self._extract_entities(sentence)
        factual_indicators = self._find_factual_indicators(sentence)
        sentiment_score = self._analyze_sentiment(sentence)
        readability_score = self._calculate_readability(sentence)
        category = self._categorize_claim(sentence, keywords, entities)
        
        # Find position in full text
        start_pos = full_text.find(sentence)
        end_pos = start_pos + len(sentence) if start_pos != -1 else 0
        
        return ExtractedClaim(
            text=sentence.strip(),
            confidence=confidence,
            start_position=start_pos,
            end_position=end_pos,
            category=category,
            keywords=keywords,
            entities=entities,
            factual_indicators=factual_indicators,
            sentiment_score=sentiment_score,
            readability_score=readability_score
        )
    
    def _calculate_confidence(self, sentence: str) -> float:
        """
        Calculate confidence that sentence contains a factual claim
        
        Args:
            sentence: Sentence to analyze
            
        Returns:
            Confidence score between 0 and 1
        """
        confidence = 0.0
        
        # Check for factual patterns
        factual_matches = sum(1 for pattern in self.factual_regex if pattern.search(sentence))
        confidence += factual_matches * 0.2
        
        # Check for strong indicators
        strong_matches = sum(1 for pattern in self.strong_regex if pattern.search(sentence))
        confidence += strong_matches * 0.3
        
        # Penalty for uncertainty indicators
        uncertainty_matches = sum(1 for pattern in self.uncertainty_regex if pattern.search(sentence))
        confidence -= uncertainty_matches * 0.15
        
        # Bonus for specific elements
        if re.search(r'\b\d+\b', sentence):  # Contains numbers
            confidence += 0.1
        
        if re.search(r'\b(?:study|research|report|survey|analysis)\b', sentence, re.IGNORECASE):
            confidence += 0.15
        
        # Penalty for opinion indicators
        if re.search(r'\b(?:should|must|ought to|need to)\b', sentence, re.IGNORECASE):
            confidence -= 0.1
        
        # Bonus for present tense factual statements
        if re.search(r'\b(?:is|are|has|have|shows|indicates|demonstrates)\b', sentence):
            confidence += 0.1
        
        # Normalize confidence
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence
    
    def _extract_keywords(self, sentence: str) -> List[str]:
        """Extract keywords from sentence"""
        try:
            # Tokenize and remove stopwords
            words = word_tokenize(sentence.lower())
            stop_words = set(stopwords.words('english'))
            
            # Filter words
            keywords = [
                word for word in words 
                if word.isalpha() and len(word) > 2 and word not in stop_words
            ]
            
            # Get POS tags and keep only nouns, verbs, adjectives
            pos_tags = pos_tag(keywords)
            filtered_keywords = [
                word for word, pos in pos_tags 
                if pos.startswith(('NN', 'VB', 'JJ'))
            ]
            
            return filtered_keywords[:10]  # Limit to 10 keywords
            
        except Exception as e:
            logger.warning(f"Error extracting keywords: {e}")
            return []
    
    def _extract_entities(self, sentence: str) -> List[Dict[str, str]]:
        """Extract named entities from sentence"""
        entities = []
        
        try:
            if self.nlp:
                # Use spaCy for entity extraction
                doc = self.nlp(sentence)
                for ent in doc.ents:
                    entities.append({
                        'text': ent.text,
                        'label': ent.label_,
                        'description': spacy.explain(ent.label_)
                    })
            else:
                # Fallback to NLTK
                tokens = word_tokenize(sentence)
                pos_tags = pos_tag(tokens)
                chunks = ne_chunk(pos_tags)
                
                for chunk in chunks:
                    if isinstance(chunk, Tree):
                        entity_text = ' '.join([token for token, pos in chunk.leaves()])
                        entities.append({
                            'text': entity_text,
                            'label': chunk.label(),
                            'description': chunk.label()
                        })
        
        except Exception as e:
            logger.warning(f"Error extracting entities: {e}")
        
        return entities
    
    def _find_factual_indicators(self, sentence: str) -> List[str]:
        """Find factual indicators in sentence"""
        indicators = []
        
        for pattern in self.factual_regex:
            matches = pattern.findall(sentence)
            indicators.extend(matches)
        
        for pattern in self.strong_regex:
            matches = pattern.findall(sentence)
            indicators.extend(matches)
        
        return list(set(indicators))  # Remove duplicates
    
    def _analyze_sentiment(self, sentence: str) -> float:
        """Analyze sentiment of sentence"""
        try:
            scores = self.sentiment_analyzer.polarity_scores(sentence)
            return scores['compound']
        except Exception as e:
            logger.warning(f"Error analyzing sentiment: {e}")
            return 0.0
    
    def _calculate_readability(self, sentence: str) -> float:
        """Calculate readability score"""
        try:
            return flesch_reading_ease(sentence)
        except Exception as e:
            logger.warning(f"Error calculating readability: {e}")
            return 0.0
    
    def _categorize_claim(self, sentence: str, keywords: List[str], entities: List[Dict]) -> str:
        """Categorize the claim based on content"""
        sentence_lower = sentence.lower()
        
        # Health-related
        health_terms = ['health', 'medical', 'disease', 'treatment', 'vaccine', 'medicine', 'doctor', 'patient']
        if any(term in sentence_lower for term in health_terms):
            return 'health'
        
        # Political
        political_terms = ['government', 'politics', 'election', 'president', 'congress', 'policy', 'law']
        if any(term in sentence_lower for term in political_terms):
            return 'political'
        
        # Science/Technology
        science_terms = ['research', 'study', 'science', 'technology', 'data', 'experiment', 'analysis']
        if any(term in sentence_lower for term in science_terms):
            return 'science'
        
        # Business/Economy
        business_terms = ['economy', 'business', 'market', 'financial', 'money', 'profit', 'company']
        if any(term in sentence_lower for term in business_terms):
            return 'business'
        
        # Check entities for categorization
        for entity in entities:
            if entity['label'] in ['PERSON', 'ORG']:
                return 'political'
            elif entity['label'] in ['GPE', 'LOC']:
                return 'geographical'
        
        return 'other'
    
    def get_claim_summary(self, claims: List[ExtractedClaim]) -> Dict:
        """Get summary statistics of extracted claims"""
        if not claims:
            return {
                'total_claims': 0,
                'avg_confidence': 0.0,
                'categories': {},
                'top_keywords': []
            }
        
        # Calculate statistics
        total_claims = len(claims)
        avg_confidence = sum(claim.confidence for claim in claims) / total_claims
        
        # Category distribution
        categories = {}
        for claim in claims:
            categories[claim.category] = categories.get(claim.category, 0) + 1
        
        # Top keywords
        all_keywords = []
        for claim in claims:
            all_keywords.extend(claim.keywords)
        
        keyword_counts = {}
        for keyword in all_keywords:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'total_claims': total_claims,
            'avg_confidence': round(avg_confidence, 3),
            'categories': categories,
            'top_keywords': [{'keyword': k, 'count': v} for k, v in top_keywords]
        }