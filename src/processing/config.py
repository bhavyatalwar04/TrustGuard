"""
Configuration Settings for Social Media Trend Analysis Pipeline

This file contains all configuration settings, file paths, and parameters
used across the entire pipeline.
"""

import os
from pathlib import Path
from datetime import datetime

# ==========================================
# PROJECT PATHS
# ==========================================

# Base project directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw_posts"
PROCESSED_DATA_DIR = DATA_DIR / "processed_data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==========================================
# FILE NAMING PATTERNS
# ==========================================

# File naming patterns for data files
RAW_DATA_PATTERN = "posts_reddit_data_*.csv"
PREPROCESSED_PATTERN = "preprocessed_reddit_posts_advanced_*.csv"
TOPIC_ANALYSIS