"""
Helper utilities for TruthGuard backend
"""
import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)

def generate_hash(content: str) -> str:
    """Generate SHA-256 hash for content"""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def normalize_text(text: str) -> str:
    """Normalize text by removing extra whitespace and converting to lowercase"""
    if not text:
        return ""
    
    # Remove extra whitespace and normalize
    normalized = re.sub(r'\s+', ' ', text.strip())
    return normalized.lower()

def extract_urls(text: str) -> List[str]:
    """Extract URLs from text"""
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    return url_pattern.findall(text)

def validate_url(url: str) -> bool:
    """Validate if URL is properly formatted"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

def get_domain(url: str) -> Optional[str]:
    """Extract domain from URL"""
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except Exception:
        return None

def clean_html(text: str) -> str:
    """Remove HTML tags from text"""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text to specified length"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

def calculate_confidence_score(scores: List[float]) -> float:
    """Calculate weighted confidence score"""
    if not scores:
        return 0.0
    
    # Remove outliers (values outside 2 standard deviations)
    mean_score = sum(scores) / len(scores)
    std_dev = (sum((x - mean_score) ** 2 for x in scores) / len(scores)) ** 0.5
    
    filtered_scores = [
        score for score in scores 
        if abs(score - mean_score) <= 2 * std_dev
    ]
    
    if not filtered_scores:
        return mean_score
    
    return sum(filtered_scores) / len(filtered_scores)

def format_timestamp(dt: datetime) -> str:
    """Format datetime to ISO string"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()

def parse_timestamp(timestamp_str: str) -> datetime:
    """Parse ISO timestamp string to datetime"""
    return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    # Remove or replace unsafe characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')
    # Limit length
    return sanitized[:255]

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            last_period = text.rfind('.', start, end)
            last_exclamation = text.rfind('!', start, end)
            last_question = text.rfind('?', start, end)
            
            sentence_end = max(last_period, last_exclamation, last_question)
            if sentence_end > start:
                end = sentence_end + 1
        
        chunks.append(text[start:end])
        start = end - overlap
        
        if start >= len(text):
            break
    
    return chunks

def merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """Recursively merge two dictionaries"""
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result

def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate basic text similarity using Jaccard similarity"""
    if not text1 or not text2:
        return 0.0
    
    # Convert to sets of words
    words1 = set(normalize_text(text1).split())
    words2 = set(normalize_text(text2).split())
    
    # Calculate Jaccard similarity
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union)

def extract_key_phrases(text: str, top_n: int = 10) -> List[str]:
    """Extract key phrases from text (basic implementation)"""
    # Simple approach: extract noun phrases and frequent terms
    words = normalize_text(text).split()
    
    # Filter out common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
    }
    
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Count word frequency
    word_freq = {}
    for word in filtered_words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top N
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_words[:top_n]]

def validate_json(json_str: str) -> bool:
    """Validate if string is valid JSON"""
    try:
        json.loads(json_str)
        return True
    except (json.JSONDecodeError, TypeError):
        return False

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if division by zero"""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"

class Timer:
    """Simple timer context manager"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        logger.info(f"{self.name} completed in {duration:.2f} seconds")

def batch_process(items: List[Any], batch_size: int = 100):
    """Generator to process items in batches"""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]