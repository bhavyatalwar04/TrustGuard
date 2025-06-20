"""
Knowledge Graph Package

This package provides enhanced knowledge graph functionality using Wikipedia and News APIs
for entity extraction, context gathering, and claim verification.
"""

from .knowledge_graph import KnowledgeGraphManager
# Package-level exports
__all__ = [
    "KnowledgeGraphManager",
]

# Optional: Package-level configuration
DEFAULT_CONFIG = {
    "max_entities": 5,
    "max_news_articles": 5,
    "wikipedia_summary_length": 500,
    "news_api_timeout": 10,
    "cooldown_period": 600,  # 10 minutes
}