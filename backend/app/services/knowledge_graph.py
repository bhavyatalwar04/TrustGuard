# backend/app/services/knowledge_graph.py

import networkx as nx
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from collections import defaultdict
import pickle

logger = logging.getLogger(__name__)

@dataclass
class Entity:
    id: str
    name: str
    entity_type: str
    properties: Dict[str, Any]
    aliases: List[str]
    confidence: float

@dataclass
class Relationship:
    source_id: str
    target_id: str
    relation_type: str
    properties: Dict[str, Any]
    confidence: float
    source: str
    created_at: datetime

@dataclass
class ClaimNode:
    id: str
    text: str
    entities: List[str]
    verdict: Optional[str]
    confidence: float
    sources: List[str]
    created_at: datetime

class KnowledgeGraph:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.graph = nx.MultiDiGraph()
        self.entity_embeddings = {}
        self.claim_embeddings = {}
        
        # Relationship types and their weights
        self.relation_weights = {
            'supports': 1.0,
            'contradicts': -1.0,
            'related_to': 0.5,
            'mentions': 0.3,
            'similar_to': 0.7,
            'causes': 0.8,
            'occurred_at': 0.4,
            'involves': 0.6
        }

    def add_entity(self, entity: Entity) -> bool:
        """Add an entity to the knowledge graph."""
        try:
            self.graph.add_node(
                entity.id,
                node_type='entity',
                name=entity.name,
                entity_type=entity.entity_type,
                properties=entity.properties,
                aliases=entity.aliases,
                confidence=entity.confidence,
                created_at=datetime.now()
            )
            
            # Add aliases as separate nodes with references
            for alias in entity.aliases:
                alias_id = f"{entity.id}_alias_{alias.replace(' ', '_')}"
                self.graph.add_node(
                    alias_id,
                    node_type='alias',
                    name=alias,
                    refers_to=entity.id
                )
                self.graph.add_edge(alias_id, entity.id, relation_type='alias_of')
            
            logger.info(f"Added entity: {entity.name} ({entity.id})")
            return True
            
        except Exception as e:
            logger.error(f"Error adding entity {entity.name}: {str(e)}")
            return False

    def add_claim(self, claim: ClaimNode) -> bool:
        """Add a claim to the knowledge graph."""
        try:
            self.graph.add_node(
                claim.id,
                node_type='claim',
                text=claim.text,
                entities=claim.entities,
                verdict=claim.verdict,
                confidence=claim.confidence,
                sources=claim.sources,
                created_at=claim.created_at
            )
            
            # Link claim to entities it mentions
            for entity_id in claim.entities:
                if self.graph.has_node(entity_id):
                    self.graph.add_edge(
                        claim.id,
                        entity_id,
                        relation_type='mentions',
                        confidence=0.8
                    )
            
            logger.info(f"Added claim: {claim.id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding claim {claim.id}: {str(e)}")
            return False

    def add_relationship(self, relationship: Relationship) -> bool:
        """Add a relationship between nodes."""
        try:
            if not (self.graph.has_node(relationship.source_id) and 
                   self.graph.has_node(relationship.target_id)):
                logger.warning(f"Cannot add relationship: missing nodes {relationship.source_id} or {relationship.target_id}")
                return False
            
            self.graph.add_edge(
                relationship.source_id,
                relationship.target_id,
                relation_type=relationship.relation_type,
                properties=relationship.properties,
                confidence=relationship.confidence,
                source=relationship.source,
                created_at=relationship.created_at
            )
            
            logger.info(f"Added relationship: {relationship.source_id} -> {relationship.target_id} ({relationship.relation_type})")
            return True
            
        except Exception as e:
            logger.error(f"Error adding relationship: {str(e)}")
            return False

    def find_related_claims(self, claim_text: str, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Find claims related to the given claim."""
        try:
            related_claims = []
            
            # Get all claim nodes
            claim_nodes = [n for n in self.graph.nodes() 
                          if self.graph.nodes[n].get('node_type') == 'claim']
            
            for claim_id in claim_nodes:
                claim_data = self.graph.nodes[claim_id]