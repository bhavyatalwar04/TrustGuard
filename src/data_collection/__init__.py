"""
data_collection module for TruthGuard

This package provides tools for collecting and storing social media data from Reddit.

Modules:
- reddit_collector: Contains RedditCollector and RedditPost dataclass
"""

from .reddit_collector import RedditCollector, RedditPost

__all__ = ["RedditCollector", "RedditPost"]
