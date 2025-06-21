"""
Text processing utilities for TruthGuard
"""
import re
import string
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter
import unicodedata
import logging

logger = logging.getLogger(__name__)

class TextProcessor:
    """Text processing utilities for claim analysis"""
    
    def __init__(self):
        self.stop_words = self._load_stop_words()
        self.punctuation_table = str.maketrans('', '', string.punctuation)
    
    def _load_stop_words(self) -> Set[str]:
        """Load common stop words"""
        return {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
            'had', 'what', 'said', 'each', 'which', 'she', 'do', 'how', 'their',
            'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some',
            'her', 'would', 'make', 'like', 'into', 'him', 'time', 'two', 'more',
            'go', 'no', 'way', 'could', 'my', 'than', 'first', 'been', 'call',
            'who', 'oil', 'sit', 'now', 'find', 'down', 'day', 'did', 'get',
            'come', 'made', 'may', 'part'
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        if not text:
            return []
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(self.punctuation_table)
        
        # Split into words
        words = text.split()
        
        # Remove stop words and short words
        words = [word for word in words if word not in self.stop_words and len(word) > 2]
        
        return words
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text"""
        if not text:
            return []
        
        # Simple sentence boundary detection
        sentences = re.split(r'[.!?]+', text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Filter out very short sentences
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def extract_named_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities (basic pattern matching)"""
        entities = {
            'persons': [],
            'organizations': [],
            'locations': [],
            'dates': [],
            'numbers': []
        }
        
        # Person names (basic pattern: Title + Name)
        person_pattern = r'\b(?:Mr|Mrs|Dr|Prof|President|Senator|Judge|Chief)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*'
        entities['persons'] = re.findall(person_pattern, text)
        
        # Organizations (basic pattern: words with Corp, Inc, Ltd, etc.)
        org_pattern = r'\b[A-Z][a-zA-Z\s]*(?:Corp|Inc|Ltd|LLC|Company|Organization|Association|Foundation|Institute)\b'
        entities['organizations'] = re.findall(org_pattern, text)
        
        # Locations (basic pattern: capitalized place names)
        location_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:City|State|Country|County|Province))?'
        potential_locations = re.findall(location_pattern, text)
        # Filter out common words that might be capitalized
        common_words = {'The', 'This', 'That', 'These', 'Those', 'What', 'When', 'Where', 'Why', 'How'}
        entities['locations'] = [loc for loc in potential_locations if loc not in common_words]
        
        # Dates (various formats)
        date_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # MM/DD/YYYY
            r'\b\d{4}-\d{1,2}-\d{1,2}\b',  # YYYY-MM-DD
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b'
        ]
        for pattern in date_patterns:
            entities['dates'].extend(re.findall(pattern, text, re.IGNORECASE))
        
        # Numbers (including percentages, currency)
        number_pattern = r'\b(?:\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\d+(?:\.\d+)?%|\d+(?:\.\d+)?)\b'
        entities['numbers'] = re.findall(number_pattern, text)
        
        return entities
    
    def calculate_readability_score(self, text: str) -> float:
        """Calculate Flesch Reading Ease score"""
        if not text:
            return 0.0
        
        sentences = self.extract_sentences(text)
        words = self.tokenize(text)
        
        if not sentences or not words:
            return 0.0
        
        # Count syllables (approximation)
        syllable_count = sum(self._count_syllables(word) for word in words)
        
        # Flesch Reading Ease formula
        score = 206.835 - (1.015 * len(words) / len(sentences)) - (84.6 * syllable_count / len(words))
        
        # Normalize to 0-100 range
        return max(0, min(100, score))
    
    def _count_syllables(self, word: str) -> int:
        """Approximate syllable count for a word"""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel
        
        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def extract_key_phrases(self, text: str, max_phrases: int = 20) -> List[Tuple[str, int]]:
        """Extract key phrases with frequency counts"""
        if not text:
            return []
        
        # Generate n-grams (1, 2, 3 words)
        phrases = []
        words = self.tokenize(text)
        
        # Unigrams
        phrases.extend(words)
        
        # Bigrams
        for i in range(len(words) - 1):
            phrases.append(f"{words[i]} {words[i+1]}")
        
        # Trigrams
        for i in range(len(words) - 2):
            phrases.append(f"{words[i]} {words[i+1]} {words[i+2]}")
        
        # Count phrase frequencies
        phrase_counts = Counter(phrases)
        
        # Filter out single character phrases and very common phrases
        filtered_phrases = {
            phrase: count for phrase, count in phrase_counts.items()
            if len(phrase) > 2 and count > 1
        }
        
        # Return top phrases
        return sorted(filtered_phrases.items(), key=lambda x: x[1], reverse=True)[:max_phrases]
    
    def detect_language(self, text: str) -> str:
        """Simple language detection (basic heuristic)"""
        if not text:
            return "unknown"
        
        # Check for common English words
        english_indicators = ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with']
        words = text.lower().split()
        english_word_count = sum(1 for word in words if word in english_indicators)
        
        if len(words) > 0 and english_word_count / len(words) > 0.1:
            return "english"
        
        return "unknown"
    
    def calculate_text_complexity(self, text: str) -> Dict[str, float]:
        """Calculate various text complexity metrics"""
        if not text:
            return {"word_count": 0, "sentence_count": 0, "avg_word_length": 0, "avg_sentence_length": 0}
        
        words = self.tokenize(text)
        sentences = self.extract_sentences(text)
        
        word_count = len(words)
        sentence_count = len(sentences)
        avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_word_length": avg_word_length,
            "avg_sentence_length": avg_sentence_length,
            "readability_score": self.calculate_readability_score(text)
        }
    
    def find_contradictions(self, text1: str, text2: str) -> List[Dict[str, str]]:
        """Find potential contradictions between two texts"""
        contradictions = []
        
        # Extract key phrases from both texts
        phrases1 = dict(self.extract_key_phrases(text1, 50))
        phrases2 = dict(self.extract_key_phrases(text2, 50))
        
        # Look for contradictory patterns
        contradiction_patterns = [
            (r'\b(?:not|no|never|none)\b', r'\b(?:yes|always|all|every)\b'),
            (r'\b(?:true|correct|right)\b', r'\b(?:false|wrong|incorrect)\b'),
            (r'\b(?:increase|rise|grow)\b', r'\b(?:decrease|fall|shrink)\b'),
            (r'\b(?:support|endorse|favor)\b', r'\b(?:oppose|reject|against)\b'),
        ]
        
        for positive_pattern, negative_pattern in contradiction_patterns:
            pos_matches1 = re.findall(positive_pattern, text1, re.IGNORECASE)
            neg_matches2 = re.findall(negative_pattern, text2, re.IGNORECASE)
            
            if pos_matches1 and neg_matches2:
                contradictions.append({
                    "type": "semantic_contradiction",
                    "text1_pattern": positive_pattern,
                    "text2_pattern": negative_pattern,
                    "text1_matches": pos_matches1,
                    "text2_matches": neg_matches2
                })
        
        return contradictions
    
    def extract_claims(self, text: str) -> List[str]:
        """Extract potential factual claims from text"""
        if not text:
            return []
        
        sentences = self.extract_sentences(text)
        claims = []
        
        # Look for sentences that contain factual claim indicators
        claim_indicators = [
            r'\b(?:according to|research shows|studies indicate|data suggests|statistics show)\b',
            r'\b(?:percent|%|million|billion|thousand)\b',
            r'\b(?:increase|decrease|rise|fall|grow|shrink)\b.*(?:percent|%)\b',
            r'\b(?:more than|less than|at least|up to|approximately)\b.*\d+',
            r'\b(?:first|last|most|least|highest|lowest|largest|smallest)\b',
        ]
        
        for sentence in sentences:
            for pattern in claim_indicators:
                if re.search(pattern, sentence, re.IGNORECASE):
                    claims.append(sentence.strip())
                    break
        
        return list(set(claims))  # Remove duplicates