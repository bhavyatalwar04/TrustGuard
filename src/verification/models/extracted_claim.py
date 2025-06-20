from dataclasses import dataclass
from typing import List

@dataclass
class ExtractedClaim:
    """Structure for extracted claims"""
    claim_id: str
    text: str
    claim_type: str
    source_post_id: str
    extraction_timestamp: str
    keywords: List[str]
    entities: List[str]
    confidence: str = "low"  # Changed from float to str to match usage
    source: str = ""
