import asyncio
import logging
import time
from datetime import datetime
from typing import List, Dict, Any

# Assuming these imports are from other modules in your project
from ..DatabaseManager import DatabaseManager
from ..KnowledgeGraph import KnowledgeGraphManager
from ..FactCheckManager import FactCheckAPIManager
from ..SemanticMatcher import SemanticMatcher
from ..VerdictEngine import VerdictEngine
from ..models import ExtractedClaim, VerificationResult, VerificationStatus


class ClaimVerificationPipeline:
    """Enhanced pipeline with integrated knowledge graph"""
    
    def __init__(self, db_path: str = "verification_results.db"):
        self.db_manager = DatabaseManager(db_path)
        self.knowledge_graph = KnowledgeGraphManager()
        self.fact_check_manager = FactCheckAPIManager()
        self.semantic_matcher = SemanticMatcher()
        self.verdict_engine = VerdictEngine()
        
        logging.info("Enhanced Claim Verification Pipeline initialized successfully")
    
    async def verify_single_claim(self, claim: ExtractedClaim) -> VerificationResult:
        """Verify a single claim through the complete pipeline"""
        start_time = time.time()
        
        logging.info(f"Starting verification for claim: {claim.claim_id}")
        
        try:
            # Store claim in database
            self.db_manager.store_claim(claim)
            
            # Get knowledge graph context
            kg_context = self.knowledge_graph.get_claim_context(claim)
            
            # Run fact-checking
            fact_check_results = await self.fact_check_manager.fact_check_claim(claim)
            
            # Semantic matching
            evidence_texts = [r.get('snippet', r.get('claim_text', '')) for r in fact_check_results]
            evidence_texts.extend([s.get('description', '') for s in kg_context.get('evidence_sources', [])])
            semantic_scores = self.semantic_matcher.match_against_evidence(claim.text, evidence_texts)
            
            # Combine verification data
            verification_data = {
                'fact_check_results': fact_check_results,
                'semantic_similarity_scores': semantic_scores,
                'evidence_sources': kg_context.get('evidence_sources', []),
                'knowledge_graph_context': kg_context
            }
            
            # Determine final verdict
            verdict, reasoning = self.verdict_engine.determine_verdict(claim, verification_data)
            confidence_score = self.verdict_engine.calculate_confidence_score(verification_data)
            
            # Create verification result
            result = VerificationResult(
                claim_id=claim.claim_id,
                verification_status=verdict,
                confidence_score=confidence_score,
                evidence_sources=kg_context.get('evidence_sources', []),
                fact_check_results=fact_check_results,
                semantic_similarity_scores=semantic_scores,
                final_verdict=verdict,
                reasoning=reasoning,
                timestamp=datetime.now().isoformat(),
                processing_time=time.time() - start_time
            )
            
            # Store result
            self.db_manager.store_verification_result(result)
            
            logging.info(f"Verification completed for claim {claim.claim_id}: {verdict} (confidence: {confidence_score:.2f})")
            return result
            
        except Exception as e:
            logging.error(f"Error verifying claim {claim.claim_id}: {e}")
            return VerificationResult(
                claim_id=claim.claim_id,
                verification_status=VerificationStatus.ERROR.value,
                confidence_score=0.0,
                evidence_sources=[],
                fact_check_results=[],
                semantic_similarity_scores=[],
                final_verdict=VerificationStatus.ERROR.value,
                reasoning=f"Verification error: {str(e)}",
                timestamp=datetime.now().isoformat(),
                processing_time=time.time() - start_time
            )
    
    async def verify_batch_claims(self, claims: List[ExtractedClaim]) -> List[VerificationResult]:
        """Verify multiple claims with rate limiting"""
        results = []
        
        for i, claim in enumerate(claims):
            try:
                print(f"🔍 Verifying claim {i+1}/{len(claims)}: {claim.text[:60]}...")
                result = await self.verify_single_claim(claim)
                results.append(result)
                
                # Rate limiting between claims
                if i < len(claims) - 1:
                    await asyncio.sleep(2)
                    
            except Exception as e:
                logging.error(f"Error in batch verification for claim {claim.claim_id}: {e}")
                # Create error result
                error_result = VerificationResult(
                    claim_id=claim.claim_id,
                    verification_status=VerificationStatus.ERROR.value,
                    confidence_score=0.0,
                    evidence_sources=[],
                    fact_check_results=[],
                    semantic_similarity_scores=[],
                    final_verdict=VerificationStatus.ERROR.value,
                    reasoning=f"Batch verification error: {str(e)}",
                    timestamp=datetime.now().isoformat(),
                    processing_time=0.0
                )
                results.append(error_result)
        
        return results
    
    def generate_report(self, results: List[VerificationResult]) -> Dict[str, Any]:
        """Generate comprehensive verification report"""
        if not results:
            return {"error": "No results to generate report from"}
        
        # Calculate summary statistics
        total_claims = len(results)
        processing_times = [r.processing_time for r in results if r.processing_time > 0]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        # Status distribution
        status_counts = {}
        for result in results:
            status = result.verification_status
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Confidence distribution
        confidence_levels = {'high': 0, 'medium': 0, 'low': 0}
        for result in results:
            if result.confidence_score >= 0.7:
                confidence_levels['high'] += 1
            elif result.confidence_score >= 0.4:
                confidence_levels['medium'] += 1
            else:
                confidence_levels['low'] += 1
        
        # Detailed results
        detailed_results = []
        for result in results:
            detailed_result = {
                'claim_id': result.claim_id,
                'verification_status': result.verification_status,
                'confidence_score': round(result.confidence_score, 3),
                'final_verdict': result.final_verdict,
                'reasoning': result.reasoning,
                'processing_time': round(result.processing_time, 2),
                'evidence_count': len(result.evidence_sources),
                'fact_check_count': len(result.fact_check_results),
                'timestamp': result.timestamp
            }
            
            # Add top evidence sources
            if result.evidence_sources:
                detailed_result['top_evidence'] = [
                    {
                        'title': source.get('title', 'N/A'),
                        'source': source.get('source', 'N/A'),
                        'url': source.get('url', 'N/A')
                    }
                    for source in result.evidence_sources[:3]
                ]
            
            detailed_results.append(detailed_result)
        
        # Generate report
        report = {
            'metadata': {
                'report_generated': datetime.now().isoformat(),
                'pipeline_version': '2.0',
                'total_claims_processed': total_claims
            },
            'summary': {
                'total_claims_processed': total_claims,
                'average_processing_time': round(avg_processing_time, 2),
                'verification_status_distribution': status_counts,
                'confidence_distribution': confidence_levels,
                'success_rate': round((total_claims - status_counts.get('error', 0)) / total_claims * 100, 1) if total_claims > 0 else 0
            },
            'detailed_results': detailed_results,
            'top_verified_claims': [
                r for r in detailed_results 
                if r['verification_status'] in ['verified_true', 'partially_true']
            ][:10],
            'unverifiable_claims': [
                r for r in detailed_results 
                if r['verification_status'] == 'unverifiable'
            ][:10]
        }
        
        return report