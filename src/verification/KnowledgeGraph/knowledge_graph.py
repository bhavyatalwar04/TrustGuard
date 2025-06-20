"""
Knowledge Graph Manager Module

This module provides enhanced knowledge graph functionality using Wikipedia and News APIs
for entity extraction, context gathering, and claim verification.
"""

import logging
import re
import time
from difflib import get_close_matches
from typing import List, Dict, Any, Union

import requests
import wikipedia


class KnowledgeGraphManager:
    """Enhanced Knowledge Graph replacement using Wikipedia and News APIs"""
    
    def __init__(self):
        # NewsAPI keys with fallback
        self.news_api_keys = [
            "681de298fd2344c881a6937ed653ac8c",
            "4d3b8d8591a2497bbe34d61babd18107",
            "fe1cbe13c8e043319ddc38e2e01b89f1",
            "c223533f771343e89d041961dcf32478",
            "b3c3cf175ceb40e2b4747b15bba951ae",
            "d8ed8613c5bc4265aa8e9eb1da2daa0a",
            "10120eff237f4a81b808484a729244a4",
            "7635cc06117a4ca798c8e226b5d2e1c3", 
            "d99ee84a55b0470e9a6afb1b1183f62a",
        ]
        
        # Initialize cooldown tracking
        self.news_key_cooldown = {}
        
        logging.info("Knowledge Graph Manager initialized")
    
    def extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text"""
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
        except ImportError:
            logging.warning("spaCy not available, using fallback entity extraction")
            nlp = None
            
        if nlp:
            doc = nlp(text)
            entities = []
            
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE", "EVENT", "PRODUCT", "WORK_OF_ART"]:
                    entities.append(ent.text.strip())
            
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) <= 3:
                    entities.append(chunk.text.strip())
            
            return list(set(entities))
        else:
            # Fallback: simple capitalized word extraction
            words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
            return list(set(words))[:5]
    
    def clean_entity_name(self, entity: str) -> str:
        """Clean and normalize entity names for better Wikipedia matching"""
        # Remove extra whitespace
        entity = re.sub(r'\s+', ' ', entity.strip())
        
        # Fix common issues
        entity = entity.replace("donalds trump", "Donald Trump")
        entity = entity.replace("trumps", "Trump")
        
        # Capitalize first letter of each word for proper names
        if not any(word.islower() for word in entity.split()):
            # If all caps or mixed, title case it
            entity = entity.title()
        
        return entity
        
    def get_wikipedia_context(self, entity: str) -> Dict[str, Any]:
        """Get Wikipedia context for an entity"""
        try:
            # Clean the entity name first
            entity = self.clean_entity_name(entity)
            
            wikipedia.set_lang("en")
            
            # First, search for the entity
            search_results = wikipedia.search(entity, results=5)
            if not search_results:
                return {"status": "not_found", "summary": None}
            
            # Try to find the best match using fuzzy matching
            best_matches = get_close_matches(entity.lower(), 
                                           [result.lower() for result in search_results], 
                                           n=3, cutoff=0.6)
            
            # If we have matches, try them in order
            if best_matches:
                # Find the original case version
                for match in best_matches:
                    original_match = next((result for result in search_results 
                                        if result.lower() == match), None)
                    if original_match:
                        try:
                            page = wikipedia.page(original_match, auto_suggest=False)
                            return {
                                "status": "found",
                                "title": page.title,
                                "summary": page.summary[:500] + "..." if len(page.summary) > 500 else page.summary,
                                "url": page.url,
                                "categories": getattr(page, 'categories', [])[:5]
                            }
                        except wikipedia.PageError:
                            continue
                        except wikipedia.DisambiguationError as e:
                            # Try the first disambiguation option
                            try:
                                page = wikipedia.page(e.options[0])
                                return {
                                    "status": "found",
                                    "title": page.title,
                                    "summary": page.summary[:500] + "..." if len(page.summary) > 500 else page.summary,
                                    "url": page.url,
                                    "categories": getattr(page, 'categories', [])[:5],
                                    "disambiguation_used": e.options[0]
                                }
                            except Exception:
                                return {"status": "disambiguation_error", "options": e.options[:5]}
            
            # If fuzzy matching fails, try the first search result directly
            try:
                page = wikipedia.page(search_results[0], auto_suggest=False)
                return {
                    "status": "found",
                    "title": page.title,
                    "summary": page.summary[:500] + "..." if len(page.summary) > 500 else page.summary,
                    "url": page.url,
                    "categories": getattr(page, 'categories', [])[:5]
                }
            except wikipedia.DisambiguationError as e:
                try:
                    page = wikipedia.page(e.options[0])
                    return {
                        "status": "found",
                        "title": page.title,
                        "summary": page.summary[:500] + "..." if len(page.summary) > 500 else page.summary,
                        "url": page.url,
                        "categories": getattr(page, 'categories', [])[:5],
                        "disambiguation_used": e.options[0]
                    }
                except Exception:
                    return {"status": "disambiguation_error", "options": e.options[:5]}
            except wikipedia.PageError:
                return {"status": "not_found", "summary": None}
                
        except Exception as e:
            logging.warning(f"Wikipedia API error for {entity}: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_news_context(self, entity: str) -> Dict[str, Any]:
        """Get news context for an entity using NewsAPI with rate limit handling"""
        # Clean entity name for better search results
        entity = self.clean_entity_name(entity)
        
        for i, api_key in enumerate(self.news_api_keys):
            now = time.time()
            
            # Skip key if it's in cooldown
            if self.news_key_cooldown.get(i, 0) > now:
                continue
            
            try:
                # Make API request
                response = requests.get(
                    "https://newsapi.org/v2/everything",
                    params={
                        "q": entity,
                        "pageSize": 5,
                        "sortBy": "publishedAt",
                        "language": "en"
                    },
                    headers={"X-API-Key": api_key},
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    articles = data.get("articles", [])
                    
                    # Filter out articles with missing essential data
                    valid_articles = []
                    for article in articles:
                        if article.get("title") and article.get("description"):
                            valid_articles.append({
                                "title": article.get("title"),
                                "description": article.get("description"),
                                "url": article.get("url"),
                                "publishedAt": article.get("publishedAt"),
                                "source": article.get("source", {}).get("name")
                            })
                    
                    logging.info(f"✓ NewsAPI key #{i+1} succeeded, found {len(valid_articles)} articles")
                    return {
                        "status": "found",
                        "articles": valid_articles,
                        "total_results": data.get("totalResults", 0)
                    }
                
                elif response.status_code == 429:
                    # Rate limited - set cooldown
                    logging.warning(f"✗ NewsAPI key #{i+1} rate limited, setting cooldown")
                    self.news_key_cooldown[i] = now + 600  
                    continue
                
                elif response.status_code == 401:
                    logging.error(f"✗ NewsAPI key #{i+1} unauthorized - invalid API key")
                    continue
                
                elif response.status_code == 426:
                    logging.error(f"✗ NewsAPI key #{i+1} requires upgrade")
                    continue
                
                else:
                    logging.warning(f"✗ NewsAPI key #{i+1} failed with status {response.status_code}: {response.text}")
                    continue
                    
            except requests.exceptions.Timeout:
                logging.warning(f"✗ NewsAPI key #{i+1} request timed out")
                continue
            except requests.exceptions.RequestException as e:
                logging.warning(f"✗ NewsAPI key #{i+1} request failed: {e}")
                continue
            except Exception as e:
                logging.warning(f"✗ NewsAPI key #{i+1} failed with exception: {e}")
                continue
        
        # All keys failed
        active_keys = len([k for k, v in self.news_key_cooldown.items() if v <= time.time()])
        total_keys = len(self.news_api_keys)
        
        return {
            "status": "error", 
            "message": f"All API keys failed or rate-limited ({active_keys}/{total_keys} keys available)"
        }
    
    def get_claim_context(self, claim_input: Union[str, object]) -> Dict[str, Any]:
        """
        Get comprehensive context for a claim by extracting entities and gathering
        information from Wikipedia and news sources.
        
        Args:
            claim_input: Either a string containing the claim text, or an object
                        with a 'text' attribute containing the claim.
        
        Returns:
            Dict containing claim context with Wikipedia and news information
        """
        try:
            # Handle both string and ExtractedClaim object inputs
            if hasattr(claim_input, 'text'):
                # It's an ExtractedClaim object
                claim_text = claim_input.text
                claim_id = getattr(claim_input, 'claim_id', 'unknown')
                logging.info(f"Getting context for ExtractedClaim {claim_id}: {claim_text[:100]}...")
            elif isinstance(claim_input, str):
                # It's a string
                claim_text = claim_input
                logging.info(f"Getting context for claim string: {claim_text[:100]}...")
            else:
                # Try to convert to string as fallback
                claim_text = str(claim_input)
                logging.warning(f"Unexpected claim input type {type(claim_input)}, converted to string: {claim_text[:100]}...")
            
            # Extract entities from the claim
            entities = self.extract_entities(claim_text)
            logging.info(f"Extracted {len(entities)} entities: {entities}")
            
            context = {
                "claim": claim_text,
                "entities": entities,
                "wikipedia_context": {},
                "news_context": {},
                "evidence_sources": [],
                "status": "success"
            }
            
            # Get Wikipedia context for each entity (limit to top 3 entities)
            for entity in entities[:3]:
                if len(entity.strip()) > 2:  # Skip very short entities
                    wiki_context = self.get_wikipedia_context(entity)
                    if wiki_context.get("status") == "found":
                        context["wikipedia_context"][entity] = wiki_context
                        # Add to evidence sources
                        context["evidence_sources"].append({
                            "title": wiki_context.get("title", entity),
                            "source": "Wikipedia",
                            "url": wiki_context.get("url", ""),
                            "description": wiki_context.get("summary", "")[:200] + "..."
                        })
                        logging.info(f"✓ Wikipedia context found for: {entity}")
                    else:
                        logging.info(f"✗ No Wikipedia context for: {entity}")
            
            # Get news context for the most relevant entities (limit to top 2)
            news_entities = entities[:2] if entities else [claim_text[:50]]
            for entity in news_entities:
                if len(entity.strip()) > 2:
                    news_context = self.get_news_context(entity)
                    if news_context.get("status") == "found" and news_context.get("articles"):
                        context["news_context"][entity] = news_context
                        # Add articles to evidence sources
                        for article in news_context.get("articles", [])[:3]:  # Top 3 articles
                            context["evidence_sources"].append({
                                "title": article.get("title", ""),
                                "source": article.get("source", "News"),
                                "url": article.get("url", ""),
                                "description": article.get("description", "")
                            })
                        article_count = len(news_context.get("articles", []))
                        logging.info(f"✓ Found {article_count} news articles for: {entity}")
                    else:
                        logging.info(f"✗ No news context for: {entity}")
            
            # Add summary statistics
            context["summary"] = {
                "entities_found": len(entities),
                "wikipedia_sources": len(context["wikipedia_context"]),
                "news_sources": sum(len(news.get("articles", [])) for news in context["news_context"].values()),
                "total_sources": len(context["evidence_sources"])
            }
            
            logging.info(f"Context gathering complete: {context['summary']}")
            return context
            
        except Exception as e:
            logging.error(f"Error getting claim context: {str(e)}")
            # Handle the case where claim_text might not be defined due to input error
            safe_claim_text = ""
            try:
                if hasattr(claim_input, 'text'):
                    safe_claim_text = claim_input.text
                elif isinstance(claim_input, str):
                    safe_claim_text = claim_input
                else:
                    safe_claim_text = str(claim_input)
            except Exception:
                safe_claim_text = "Error processing claim input"
                
            return {
                "claim": safe_claim_text,
                "entities": [],
                "wikipedia_context": {},
                "news_context": {},
                "evidence_sources": [],
                "status": "error",
                "error_message": str(e),
                "summary": {
                    "entities_found": 0,
                    "wikipedia_sources": 0,
                    "news_sources": 0,
                    "total_sources": 0
                }
            }
    
    def get_entity_relationships(self, entities: List[str]) -> Dict[str, Any]:
        """
        Get relationships between entities (optional method for enhanced functionality)
        
        Args:
            entities: List of entity names to find relationships for
            
        Returns:
            Dict mapping entities to their relationships and categories
        """
        relationships = {}
        
        for entity in entities:
            wiki_context = self.get_wikipedia_context(entity)
            if wiki_context.get("status") == "found":
                # Extract potential relationships from categories and summary
                categories = wiki_context.get("categories", [])
                summary = wiki_context.get("summary", "")
                
                # Simple relationship extraction based on common patterns
                related_entities = []
                for other_entity in entities:
                    if other_entity != entity and other_entity.lower() in summary.lower():
                        related_entities.append(other_entity)
                
                if related_entities or categories:
                    relationships[entity] = {
                        "related_entities": related_entities,
                        "categories": categories[:3],  # Top 3 categories
                        "summary_snippet": summary[:200] + "..." if len(summary) > 200 else summary
                    }
        
        return relationships


# Backwards compatibility alias
KnowledgeGraphLookup = KnowledgeGraphManager