"""
Text Preprocessing Module for TruthGuard

This module provides comprehensive text preprocessing capabilities including:
- Text cleaning and normalization
- Named Entity Recognition (NER)
- Part-of-Speech (POS) tagging
- Linguistic feature extraction
- Batch processing for large datasets
- Automatic latest file detection

Classes:
    TextPreprocessor: Main preprocessing class with spaCy integration
    
Functions:
    process_reddit_posts: Process Reddit CSV files with automatic latest file detection
    setup_logging: Configure logging for the preprocessing pipeline
    get_latest_raw_data_file: Automatically find the most recent raw data file
"""

from .text_preprocessing_pipeline import (
    TextPreprocessor,
    process_reddit_posts,
    setup_logging,
    get_latest_raw_data_file
)

__all__ = [
    'TextPreprocessor',
    'process_reddit_posts',
    'setup_logging',
    'get_latest_raw_data_file'
]

__version__ = "1.0.0"