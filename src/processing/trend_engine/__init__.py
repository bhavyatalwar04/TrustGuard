# src/__init__.py
"""
Social Media Trend Analysis Package

A comprehensive package for collecting, processing, and analyzing social media trends.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Package metadata
__all__ = [
    "data_collection",
    "processing",
    "config"
]

# ==========================================
# src/data_collection/__init__.py
"""
Data Collection Module

Contains utilities for collecting data from various social media platforms.
"""

from .social_media_collector import SocialMediaCollector

__all__ = [
    "SocialMediaCollector"
]

# ==========================================
# src/processing/__init__.py
"""
Processing Module

Contains all data processing pipelines including text preprocessing and trend analysis.
"""

from .text_preprocessing import TextPreprocessingPipeline
from .trend_engine import TrendDetectionEngine

__all__ = [
    "TextPreprocessingPipeline", 
    "TrendDetectionEngine"
]

# ==========================================
# src/processing/text_preprocessing/__init__.py
"""
Text Preprocessing Module

Advanced text preprocessing utilities for social media content.
"""

from .text_preprocessing_pipeline import TextPreprocessingPipeline

__all__ = [
    "TextPreprocessingPipeline"
]

# ==========================================
# src/processing/trend_engine/__init__.py
"""
Trend Detection Engine Module

Advanced trend detection and topic modeling for social media analysis.
"""

from .trend_detector import TrendDetectionEngine, print_analysis_results

__all__ = [
    "TrendDetectionEngine",
    "print_analysis_results"
]