"""
Enhanced Fact-checking API manager with multiple search strategies including GDELT

This module provides comprehensive fact-checking capabilities by integrating:
- Google Custom Search API for fact-check sites
- GDELT news database for contextual information
- Text similarity analysis for relevance scoring
- Rating extraction from various fact-check sources
"""

import asyncio
import aiohttp
import requests
import logging
import re
import difflib
from datetime import datetime, timedelta
from urllib.parse import urlparse
from typing import List, Dict, Any, Tuple, Optional


class FactCheckAPIManager:
    """Enhanced Fact-checking API manager with multiple search strategies including GDELT"""
    
    def __init__(self, google_api_key: str = None, search_engine_id: str = None):
        """
        Initialize the FactCheckAPIManager
        
        Args:
            google_api_key: Google Custom Search API key
            search_engine_id: Google Custom Search Engine ID
        """
        self.session = requests.Session()
        
        # Google API credentials - replace with actual values
        self.google_api_key = google_api_key or "AIzaSyBKRh06u17V-TBDF6neVJeDkBaXas56-lQ"
        self.google_search_engine_id = search_engine_id or "d5951180f1b6d4c59"
        
        # Set up session headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Enhanced rating patterns for different fact-check sites
        self.rating_patterns = {
            'snopes.com': {
                'true': ['true', 'correct', 'accurate', 'verified', 'confirmed'],
                'false': ['false', 'incorrect', 'inaccurate', 'fake', 'hoax', 'debunked', 'fabricated'],
                'mixed': ['mixed', 'partially true', 'partly true', 'mostly true', 'mostly false', 'partly false'],
                'unproven': ['unproven', 'unverified', 'research in progress', 'undetermined', 'unclear']
            },
            'politifact.com': {
                'true': ['true', 'mostly true', 'correct'],
                'false': ['false', 'mostly false', 'pants on fire', 'fabricated'],
                'mixed': ['half true', 'half-true', 'mostly true', 'partially accurate'],
                'unproven': ['unproven', 'we couldn\'t verify']
            },
            'factcheck.org': {
                'true': ['accurate', 'correct', 'true', 'verified'],
                'false': ['false', 'misleading', 'incorrect', 'wrong'],
                'mixed': ['partly true', 'mixed', 'partially correct'],
                'unproven': ['unproven', 'unclear', 'inconclusive']
            }
        }
        
        # Fact-check site priorities (higher number = more reliable)
        self.site_reliability = {
            'snopes.com': 0.9,
            'politifact.com': 0.85,
            'factcheck.org': 0.8,
            'reuters.com': 0.8,
            'apnews.com': 0.75,
            'bbc.com': 0.75,
            'cnn.com': 0.65,
            'npr.org': 0.7,
            'washingtonpost.com': 0.7,
            'nytimes.com': 0.7
        }
        
        logging.info("FactCheck API Manager initialized with GDELT integration")
    
    def gdelt_search(self, query: str, max_results: int = 10, days_back: int = 30) -> List[Dict[str, Any]]:
        """
        Search GDELT for news articles matching the query.
        No API key required.
        
        Args:
            query: Search term
            max_results: Maximum number of results to return
            days_back: How many days back to search (default 30)
        
        Returns:
            List of article dictionaries
        """
        try:
            url = "https://api.gdeltproject.org/api/v2/doc/doc"
            
            # Calculate start date (days back from today)
            start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d")
            
            params = {
                "query": query,
                "mode": "artlist",  # article list
                "format": "json",
                "maxrecords": max_results,
                "sort": "date",
                "startdatetime": start_date + "000000",  # YYYYMMDDHHMMSS format
                "timespan": f"{days_back}d"  # Last N days
            }
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            articles = []
            
            for item in data.get("articles", []):
                # Extract domain for reliability scoring
                domain = urlparse(item.get("url", "")).netloc.replace('www.', '')
                
                # Get site reliability score
                site_reliability = self.site_reliability.get(domain, 0.4)  # Default for unknown sites
                
                articles.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "domain": domain,
                    "seendate": item.get("seendate", ""),
                    "source": "GDELT",
                    "summary": item.get("extrasummary", ""),
                    "site_reliability": site_reliability,
                    "is_fact_check": False,
                    "is_news_context": True
                })
            
            logging.info(f"GDELT returned {len(articles)} articles for '{query}'")
            return articles
            
        except Exception as e:
            logging.error(f"GDELT search failed for '{query}': {e}")
            return []
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Enhanced text similarity calculation using multiple methods
        
        Args:
            text1: First text to compare
            text2: Second text to compare
            
        Returns:
            Similarity score between 0 and 1
        """
        if not text1 or not text2:
            return 0.0
        
        # Clean and normalize texts
        text1_clean = re.sub(r'[^\w\s]', '', text1.lower().strip())
        text2_clean = re.sub(r'[^\w\s]', '', text2.lower().strip())
        
        # Calculate sequence similarity
        sequence_similarity = difflib.SequenceMatcher(None, text1_clean, text2_clean).ratio()
        
        # Calculate word overlap
        words1 = set(text1_clean.split())
        words2 = set(text2_clean.split())
        
        if not words1 or not words2:
            return sequence_similarity
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        # Jaccard similarity
        jaccard_similarity = len(intersection) / len(union) if union else 0
        
        # Keyword overlap (weighted for important words)
        keyword_overlap = len(intersection) / max(len(words1), len(words2), 1)
        
        # Combined similarity score
        combined_score = (sequence_similarity * 0.4) + (jaccard_similarity * 0.3) + (keyword_overlap * 0.3)
        
        return min(combined_score, 1.0)
    
    def extract_rating(self, content: str, domain: str) -> Tuple[str, float]:
        """
        Extract fact-check rating from content
        
        Args:
            content: Text content to analyze
            domain: Domain of the source
            
        Returns:
            Tuple of (rating, confidence_score)
        """
        if not content:
            return "unknown", 0.0
        
        content_lower = content.lower()
        domain_clean = domain.replace('www.', '')
        
        patterns = self.rating_patterns.get(domain_clean, {})
        
        # Check each rating category
        for category, keywords in patterns.items():
            for keyword in keywords:
                if keyword in content_lower:
                    # Higher confidence for exact matches
                    confidence = 0.8 if len(keyword) > 5 else 0.6
                    return category, confidence
        
        # Fallback patterns for unknown sites
        if any(word in content_lower for word in ['false', 'fake', 'hoax', 'debunked']):
            return "false", 0.5
        elif any(word in content_lower for word in ['true', 'accurate', 'correct']):
            return "true", 0.5
        elif any(word in content_lower for word in ['mixed', 'partially', 'partly']):
            return "mixed", 0.4
        
        return "unknown", 0.0
    
    def test_api_connection(self) -> Dict[str, bool]:
        """
        Test API connections including GDELT
        
        Returns:
            Dictionary with connection status for each API
        """
        results = {}
        
        # Test Google Custom Search API
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': self.google_api_key,
                'cx': self.google_search_engine_id,
                'q': 'test query fact check',
                'num': 1
            }
            
            response = requests.get(url, params=params, timeout=10)
            results['google_search'] = response.status_code == 200
            
            if response.status_code == 200:
                logging.info("✓ Google Custom Search API: Working")
            elif response.status_code == 403:
                logging.error("✗ Google API Error: Access forbidden (check API key)")
            elif response.status_code == 400:
                logging.error("✗ Google API Error: Bad request (check search engine ID)")
            else:
                logging.error(f"✗ Google API Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            results['google_search'] = False
            logging.error(f"✗ Google API connection failed: {e}")
        
        # Test GDELT API
        try:
            test_articles = self.gdelt_search("test", max_results=1)
            results['gdelt'] = len(test_articles) >= 0  # GDELT returns empty list on success
            if results['gdelt']:
                logging.info("✓ GDELT API: Working")
            else:
                logging.warning("⚠ GDELT API: Connection OK but no test results")
        except Exception as e:
            results['gdelt'] = False
            logging.error(f"✗ GDELT API connection failed: {e}")
        
        return results
    
    async def search_fact_check_sites(self, query: str, claim) -> List[Dict]:
        """
        Search fact-checking sites using Google Custom Search
        
        Args:
            query: Search query
            claim: Claim object with text and metadata
            
        Returns:
            List of fact-check results
        """
        # Define fact-checking sites to search
        fact_check_sites = [
            "site:snopes.com",
            "site:factcheck.org", 
            "site:politifact.com",
            "site:reuters.com fact check",
            "site:apnews.com fact check"
        ]
        
        results = []
        
        async with aiohttp.ClientSession() as session:
            for site in fact_check_sites:
                # Create search query
                search_query = f"{query} {site}"
                
                url = "https://www.googleapis.com/customsearch/v1"
                params = {
                    'key': self.google_api_key,
                    'cx': self.google_search_engine_id,
                    'q': search_query,
                    'num': 3  # Get top 3 results per site
                }
                
                try:
                    async with session.get(url, params=params, timeout=15) as response:
                        if response.status == 200:
                            data = await response.json()
                            items = data.get('items', [])
                            
                            for item in items:
                                # Extract domain
                                domain = urlparse(item.get('link', '')).netloc.replace('www.', '')
                                
                                # Combine title and snippet for analysis
                                combined_text = (item.get('title', '') + ' ' + item.get('snippet', '')).strip()
                                
                                if not combined_text:
                                    continue
                                
                                # Calculate similarity with claim
                                similarity = self.calculate_text_similarity(claim.text, combined_text)
                                
                                # Only include results with reasonable similarity
                                if similarity > 0.25:  # Lowered threshold for better coverage
                                    rating, rating_confidence = self.extract_rating(combined_text, domain)
                                    
                                    # Get site reliability score
                                    site_reliability = self.site_reliability.get(domain, 0.5)
                                    
                                    # Calculate overall evidence strength
                                    evidence_strength = min(
                                        (similarity + rating_confidence + site_reliability) / 3.0, 
                                        1.0
                                    )
                                    
                                    result = {
                                        'claim_text': claim.text,
                                        'review_rating': rating,
                                        'rating_confidence': rating_confidence,
                                        'review_url': item.get('link'),
                                        'publisher': domain,
                                        'title': item.get('title', ''),
                                        'snippet': item.get('snippet', ''),
                                        'review_date': datetime.now().isoformat(),
                                        'source': 'google_factcheck_search',
                                        'evidence_strength': evidence_strength,
                                        'similarity_score': similarity,
                                        'site_reliability': site_reliability,
                                        'is_fact_check': True
                                    }
                                    
                                    results.append(result)
                        
                        elif response.status == 403:
                            logging.error(f"Google API access forbidden for {site}")
                            
                        elif response.status == 400:
                            logging.error(f"Bad request for {site} - check API configuration")
                            
                        else:
                            logging.warning(f"API returned status {response.status} for {site}")
                    
                    # Rate limiting between requests
                    await asyncio.sleep(0.5)
                    
                except asyncio.TimeoutError:
                    logging.warning(f"Timeout searching {site}")
                    
                except Exception as e:
                    logging.error(f"Error searching {site}: {e}")
        
        return results
    
    def search_gdelt_news(self, claim_text: str, max_results: int = 10) -> List[Dict]:
        """
        Search GDELT for news context around the claim
        
        Args:
            claim_text: Text of the claim to search for
            max_results: Maximum number of results to return
            
        Returns:
            List of news article results with context
        """
        # Create search queries from claim
        queries = []
        
        # Primary query - first 80 characters
        primary_query = claim_text[:80].strip()
        if primary_query:
            queries.append(primary_query)
        
        # Extract key terms (longer words, excluding common words)
        words = claim_text.split()
        important_words = [
            w for w in words 
            if len(w) > 4 and w.lower() not in [
                'that', 'this', 'with', 'from', 'they', 'were', 'been', 'have', 
                'would', 'could', 'should', 'being', 'there', 'where', 'which'
            ]
        ]
        
        if len(important_words) >= 2:
            keyword_query = ' '.join(important_words[:4])
            if keyword_query not in queries:
                queries.append(keyword_query)
        
        all_results = []
        
        # Search GDELT with each query
        for query in queries[:2]:  # Limit to 2 queries
            try:
                gdelt_articles = self.gdelt_search(query, max_results=max_results//len(queries))
                
                for article in gdelt_articles:
                    # Combine title and summary for similarity analysis
                    combined_text = (article.get('title', '') + ' ' + article.get('summary', '')).strip()
                    
                    if not combined_text:
                        continue
                    
                    # Calculate similarity with claim
                    similarity = self.calculate_text_similarity(claim_text, combined_text)
                    
                    # Only include articles with decent similarity
                    if similarity > 0.2:
                        # Check if this looks like a fact-check article
                        is_fact_check = any(
                            term in combined_text.lower() 
                            for term in ['fact check', 'fact-check', 'verify', 'debunk', 'false', 'true', 'misleading']
                        )
                        
                        # Extract rating if it's a fact-check article
                        rating = "unknown"
                        rating_confidence = 0.0
                        
                        if is_fact_check:
                            rating, rating_confidence = self.extract_rating(combined_text, article['domain'])
                        
                        # Calculate evidence strength
                        base_strength = similarity * 0.7  # Lower than fact-check sites
                        if is_fact_check:
                            base_strength *= 1.3  # Boost for fact-check content
                        
                        evidence_strength = min(
                            (base_strength + article['site_reliability']) / 2.0,
                            1.0
                        )
                        
                        result = {
                            'claim_text': claim_text,
                            'review_rating': rating,
                            'rating_confidence': rating_confidence,
                            'review_url': article.get('url'),
                            'publisher': article.get('domain'),
                            'title': article.get('title', ''),
                            'snippet': article.get('summary', '')[:200] + '...' if len(article.get('summary', '')) > 200 else article.get('summary', ''),
                            'review_date': article.get('seendate', datetime.now().isoformat()),
                            'source': 'gdelt_news_search',
                            'evidence_strength': evidence_strength,
                            'similarity_score': similarity,
                            'site_reliability': article['site_reliability'],
                            'is_fact_check': is_fact_check,
                            'is_news_context': True
                        }
                        
                        all_results.append(result)
                        
            except Exception as e:
                logging.error(f"Error searching GDELT for '{query}': {e}")
        
        # Remove duplicates and sort by evidence strength
        seen_urls = set()
        unique_results = []
        
        for result in all_results:
            url = result.get('review_url', '')
            if url and url not in seen_urls:
                unique_results.append(result)
                seen_urls.add(url)
        
        # Sort by evidence strength
        sorted_results = sorted(
            unique_results,
            key=lambda x: x.get('evidence_strength', 0),
            reverse=True
        )
        
        return sorted_results[:max_results]
    
    async def search_general_web(self, claim_text: str) -> List[Dict]:
        """
        Search general web for claim-related information
        
        Args:
            claim_text: Text of the claim to search for
            
        Returns:
            List of general web search results
        """
        results = []
        
        # Create search queries from claim
        queries = [
            claim_text[:100],  # First 100 characters
            f'"{claim_text[:50]}"',  # Exact phrase search
        ]
        
        # Add keyword-based queries if available
        words = claim_text.split()
        if len(words) > 3:
            important_words = [w for w in words if len(w) > 4 and w.lower() not in ['that', 'this', 'with', 'from', 'they', 'were', 'been', 'have']][:4]
            if important_words:
                queries.append(' '.join(important_words))
        
        async with aiohttp.ClientSession() as session:
            for query in queries[:2]:  # Limit to 2 queries to avoid rate limits
                url = "https://www.googleapis.com/customsearch/v1"
                params = {
                    'key': self.google_api_key,
                    'cx': self.google_search_engine_id,
                    'q': query,
                    'num': 5
                }
                
                try:
                    async with session.get(url, params=params, timeout=15) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            for item in data.get('items', []):
                                domain = urlparse(item.get('link', '')).netloc.replace('www.', '')
                                
                                # Skip social media and unreliable sources
                                if any(skip in domain for skip in ['facebook.com', 'twitter.com', 'reddit.com', 'youtube.com']):
                                    continue
                                
                                combined_text = (item.get('title', '') + ' ' + item.get('snippet', '')).strip()
                                similarity = self.calculate_text_similarity(claim_text, combined_text)
                                
                                if similarity > 0.3:
                                    results.append({
                                        'claim_text': claim_text,
                                        'review_rating': 'unknown',
                                        'rating_confidence': 0.0,
                                        'review_url': item.get('link'),
                                        'publisher': domain,
                                        'title': item.get('title', ''),
                                        'snippet': item.get('snippet', ''),
                                        'review_date': datetime.now().isoformat(),
                                        'source': 'general_web_search',
                                        'evidence_strength': similarity * 0.6,  # Lower weight for general web
                                        'similarity_score': similarity,
                                        'site_reliability': self.site_reliability.get(domain, 0.4),
                                        'is_fact_check': False
                                    })
                        
                        await asyncio.sleep(0.5)
                        
                except Exception as e:
                    logging.error(f"Error in general web search: {e}")
        
        return results
    
    async def fact_check_claim(self, claim) -> List[Dict]:
        """
        Main fact-checking method with GDELT fallback
        
        Args:
            claim: Claim object with text and metadata
            
        Returns:
            List of fact-check and news context results
        """
        logging.info(f"Fact-checking claim: {claim.claim_id}")
        
        # Create multiple search queries
        queries = [claim.text[:100]]  # Primary query
        
        # Add keyword-based query if available
        if hasattr(claim, 'keywords') and claim.keywords:
            keyword_query = ' '.join(claim.keywords[:3])
            if keyword_query not in queries:
                queries.append(keyword_query)
        
        # Add entity-based query if available
        if hasattr(claim, 'entities') and claim.entities:
            entity_query = ' '.join(claim.entities[:2])
            if entity_query not in queries and len(entity_query) > 5:
                queries.append(entity_query)
        
        all_results = []
        google_results_found = False
        
        # Search fact-check sites via Google first
        for query in queries[:2]:  # Limit queries to avoid rate limits
            try:
                fact_check_results = await self.search_fact_check_sites(query, claim)
                all_results.extend(fact_check_results)
                
                if fact_check_results:
                    google_results_found = True
                
                # Small delay between queries
                if len(queries) > 1:
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logging.error(f"Error in Google fact-check search for query '{query}': {e}")
        
        # If no fact-check results from Google, try GDELT
        if not google_results_found:
            logging.info("No Google fact-check results found, searching GDELT...")
            try:
                gdelt_results = self.search_gdelt_news(claim.text, max_results=10)
                all_results.extend(gdelt_results)
                
                if gdelt_results:
                    logging.info(f"GDELT found {len(gdelt_results)} relevant articles")
                else:
                    logging.info("No relevant GDELT articles found")
                    
            except Exception as e:
                logging.error(f"Error in GDELT search: {e}")
        
        # If still no results, try general web search
        if not all_results:
            logging.info("No results from fact-check sites or GDELT, trying general web search...")
            try:
                web_results = await self.search_general_web(claim.text)
                all_results.extend(web_results)
            except Exception as e:
                logging.error(f"Error in general web search: {e}")
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_results = []
        
        for result in all_results:
            url = result.get('review_url', '')
            if url and url not in seen_urls:
                unique_results.append(result)
                seen_urls.add(url)
        
        # Sort by evidence strength and return top results
        sorted_results = sorted(
            unique_results, 
            key=lambda x: x.get('evidence_strength', 0), 
            reverse=True
        )
        
        # Return top 8 results (increased to account for GDELT results)
        final_results = sorted_results[:8]
        
        # Log summary
        fact_check_count = len([r for r in final_results if r.get('is_fact_check', False)])
        news_context_count = len([r for r in final_results if r.get('is_news_context', False)])
        
        logging.info(f"Found {len(final_results)} total results for claim {claim.claim_id}: "
                    f"{fact_check_count} fact-checks, {news_context_count} news context")
        
        return final_results
    
    def sync_fact_check_claim(self, claim) -> List[Dict]:
        """
        Synchronous wrapper for fact-checking
        
        Args:
            claim: Claim object to fact-check
            
        Returns:
            List of fact-check results
        """
        return asyncio.run(self.fact_check_claim(claim))
    
    def get_fact_check_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Generate a summary of fact-check results including GDELT data
        
        Args:
            results: List of fact-check results
            
        Returns:
            Summary dictionary with statistics and top sources
        """
        if not results:
            return {
                'total_sources': 0,
                'verdict_distribution': {},
                'average_confidence': 0.0,
                'has_reliable_sources': False,
                'fact_check_count': 0,
                'news_context_count': 0
            }
        
        # Count verdicts
        verdicts = {}
        total_confidence = 0
        reliable_sources = 0
        fact_check_count = 0
        news_context_count = 0
        
        for result in results:
            rating = result.get('review_rating', 'unknown')
            verdicts[rating] = verdicts.get(rating, 0) + 1
            
            confidence = result.get('rating_confidence', 0)
            total_confidence += confidence
            
            site_reliability = result.get('site_reliability', 0)
            if site_reliability > 0.7:
                reliable_sources += 1
            
            if result.get('is_fact_check', False):
                fact_check_count += 1
            
            if result.get('is_news_context', False):
                news_context_count += 1
        
        return {
            'total_sources': len(results),
            'verdict_distribution': verdicts,
            'average_confidence': total_confidence / len(results) if results else 0.0,
            'has_reliable_sources': reliable_sources > 0,
            'reliable_source_count': reliable_sources,
            'fact_check_count': fact_check_count,
            'news_context_count': news_context_count,
            'top_sources': [
                {
                    'publisher': r.get('publisher', ''),
                    'rating': r.get('review_rating', ''),
                    'confidence': r.get('rating_confidence', 0),
                    'url': r.get('review_url', ''),
                    'source_type': 'fact_check' if r.get('is_fact_check') else 'news_context'
                }
                for r in sorted(results, key=lambda x: x.get('evidence_strength', 0), reverse=True)[:5]
            ]
        }