"""
Processing Module: Combines all submodules for claim handling, text cleaning, and trend analysis.
"""

from .claim_extraction import ClaimExtractor
from .text_preprocessing import (
    setup_logging,
    get_latest_raw_data_file,
    TextPreprocessor,
    process_reddit_posts
)
from .trend_engine import TrendDetectionEngine, print_analysis_results

__all__ = [
    # Claim extraction
    "ClaimExtractor",

    # Text preprocessing
    "setup_logging",
    "get_latest_raw_data_file",
    "TextPreprocessor",
    "process_reddit_posts",

    # Trend detection
    "TrendDetectionEngine",
    "print_analysis_results"
]
