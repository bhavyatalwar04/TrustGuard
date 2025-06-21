"""
API package for TruthGuard backend
Contains FastAPI routes and Pydantic schemas
"""

from .routes import router
from .schemas import *

__all__ = ['router']