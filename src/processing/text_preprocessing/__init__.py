"""
Text Preprocessing Module

This module handles preprocessing of raw Reddit data before claim extraction.
Includes:
- Logging setup
- Raw file discovery
- TextPreprocessor class for cleaning and structuring text
- process_reddit_posts(): Runs the full preprocessing pipeline
"""

from .text_preprocessing_pipeline import (
    setup_logging,
    get_latest_raw_data_file,
    TextPreprocessor,
    process_reddit_posts
)

__all__ = [
    "setup_logging",
    "get_latest_raw_data_file",
    "TextPreprocessor",
    "process_reddit_posts"
]
