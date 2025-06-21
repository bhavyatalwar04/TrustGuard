"""
Test suite for TruthGuard backend
"""

import sys
import os

# Add the parent directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test configuration
TEST_DATABASE_URL = "sqlite:///:memory:"
TEST_API_KEY = "test_api_key_123"
TEST_TIMEOUT = 30

__version__ = "1.0.0"