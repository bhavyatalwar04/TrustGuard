"""
Fact Check API Manager Package

This package provides enhanced fact-checking capabilities with multiple search strategies
including GDELT news database integration and Google Custom Search API.

Main Components:
- FactCheckAPIManager: Core fact-checking functionality
- GDELT integration for news article search
- Google Custom Search API for fact-check sites
- Text similarity analysis and rating extraction
"""

from .factcheckapimanager import FactCheckAPIManager



# Package metadata
__all__ = [
    "FactCheckAPIManager",
]

# Default configuration
DEFAULT_CONFIG = {
    "max_results_per_search": 10,
    "similarity_threshold": 0.25,
    "gdelt_days_back": 30,
    "request_timeout": 15,
    "rate_limit_delay": 0.5,
}

# Supported fact-check sites
SUPPORTED_FACT_CHECK_SITES = [
    "snopes.com",
    "factcheck.org", 
    "politifact.com",
    "reuters.com",
    "apnews.com"
]

# Reliable news sources for context
RELIABLE_NEWS_SOURCES = [
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "npr.org",
    "washingtonpost.com",
    "nytimes.com",
    "cnn.com"
]