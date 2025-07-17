#!/usr/bin/env python3
"""
Script to test semantic retrieval capabilities, specifically for the problematic
collaborators query: "qui sont les collaborateurs sur Isschat"
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.rag.query_processor import QueryProcessor
from src.rag.tools.semantic_retrieval_tool import SemanticRetrievalTool  
from src.rag.semantic_pipeline import SemanticRAGPipeline
from src.config import get_config


def setup_logging():
    """Setup logging for the test"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def test_query_processing():
    """Test the query processor with the problematic query"""
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("TESTING QUERY PROCESSING")
    logger.info("=" * 60)
    
    try:
        query_processor = QueryProcessor()
        
        # Test the problematic query
        test_query = "qui sont les collaborateurs sur Isschat"
        logger.info(f"Testing query: '{test_query}'")
        
        result = query_processor.process_query(test_query)
        
        logger.info(f"Original query: {result.original_query}")
        logger.info(f"Intent: {result.intent}")
        logger.info(f"Confidence: {result.confidence:.3f}")
        logger.info(f"Keywords: {result.keywords}")
        
        logger.info("\nExpanded queries:")
        for i, expanded_query in enumerate(result.expanded_queries, 1):
            logger.info(f"  {i}. {expanded_query}")
        
        logger.info("\nSemantic variations:")
        for i, variation in enumerate(result.semantic_variations, 1):
            logger.info(f"  {i}. {variation}")
        
        # Test semantic similarity
        logger.info("\nTesting semantic similarity:")
        test_texts = [
            "équipe développement isschat",
            "team members project",
            "vincent fraillon nicolas lambropoulos",
            "configuration système application",
            "documentation technique"
        ]
        
        for text in test_texts:
            similarity = query_processor.get_semantic_similarity(test_query, text)
            logger.info(f"  '{text}' -> {similarity:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Query processing test failed: {e}")
        return False


def test_semantic_retrieval():
    """Test semantic retrieval tool"""
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("TESTING SEMANTIC RETRIEVAL")
    logger.info("=" * 60)
    
    try:
        semantic_tool = SemanticRetrievalTool()
        
        # Check if ready
        if not semantic_tool.is_ready():
            logger.warning("Semantic retrieval tool not ready - vector DB may not be built")
            return False
        
        # Test the problematic query
        test_query = "qui sont les collaborateurs sur Isschat"
        logger.info(f"Testing retrieval for: '{test_query}'")
        
        # Test with semantic features
        logger.info("\nTesting with semantic features enabled:")
        semantic_results = semantic_tool.retrieve(
            test_query,
            k=5,
            use_semantic_expansion=True,
            use_semantic_reranking=True
        )
        
        logger.info(f"Found {len(semantic_results)} results")
        for i, result in enumerate(semantic_results, 1):
            logger.info(f"  {i}. Score: {result.score:.3f}")
            logger.info(f"     Content: {result.content[:100]}...")
            if 'matched_query' in result.metadata:
                logger.info(f"     Matched query: {result.metadata['matched_query']}")
            logger.info("")
        
        # Test without semantic features
        logger.info("Testing with semantic features disabled:")
        basic_results = semantic_tool.retrieve(
            test_query,
            k=5,
            use_semantic_expansion=False,
            use_semantic_reranking=False
        )
        
        logger.info(f"Found {len(basic_results)} results")
        for i, result in enumerate(basic_results, 1):
            logger.info(f"  {i}. Score: {result.score:.3f}")
            logger.info(f"     Content: {result.content[:100]}...")
            logger.info("")
        
        # Compare results
        if semantic_results and basic_results:
            improvement = semantic_results[0].score - basic_results[0].score
            logger.info(f"Score improvement: {improvement:.3f}")
            
            # Check if semantic retrieval found team-related content
            team_keywords = ["vincent", "nicolas", "emin", "fraillon", "lambropoulos", "calyaka", "équipe"]
            
            semantic_team_score = 0
            for result in semantic_results:
                content_lower = result.content.lower()
                semantic_team_score += sum(1 for keyword in team_keywords if keyword in content_lower)
            
            basic_team_score = 0
            for result in basic_results:
                content_lower = result.content.lower()
                basic_team_score += sum(1 for keyword in team_keywords if keyword in content_lower)
            
            logger.info(f"Team-related keywords found:")
            logger.info(f"  Semantic retrieval: {semantic_team_score}")
            logger.info(f"  Basic retrieval: {basic_team_score}")
            
            return semantic_team_score > basic_team_score
        
        return len(semantic_results) > 0
        
    except Exception as e:
        logger.error(f"Semantic retrieval test failed: {e}")
        return False


def test_semantic_pipeline():
    """Test the complete semantic pipeline"""
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("TESTING SEMANTIC PIPELINE")
    logger.info("=" * 60)
    
    try:
        pipeline = SemanticRAGPipeline(use_semantic_features=True)
        
        # Check if ready
        if not pipeline.is_ready():
            logger.warning("Semantic pipeline not ready - vector DB may not be built")
            return False
        
        # Test the problematic query
        test_query = "qui sont les collaborateurs sur Isschat"
        logger.info(f"Testing pipeline with: '{test_query}'")
        
        # Test with semantic features
        logger.info("\nTesting with semantic features:")
        answer, sources = pipeline.process_query(
            test_query,
            verbose=True,
            use_semantic_expansion=True,
            use_semantic_reranking=True
        )
        
        logger.info(f"Answer: {answer}")
        logger.info(f"Sources: {sources}")
        
        # Check if answer contains team information
        team_keywords = ["vincent", "nicolas", "emin", "fraillon", "lambropoulos", "calyaka", "équipe", "team"]
        answer_lower = answer.lower()
        team_mentions = [keyword for keyword in team_keywords if keyword in answer_lower]
        
        logger.info(f"Team keywords found in answer: {team_mentions}")
        
        # Test comparison
        logger.info("\nTesting comparison with basic retrieval:")
        comparison = pipeline.compare_with_basic_retrieval(test_query)
        
        logger.info(f"Query intent: {comparison.get('query_processing', {}).get('intent')}")
        logger.info(f"Expanded queries: {comparison.get('query_processing', {}).get('expanded_queries', [])}")
        
        semantic_avg = comparison.get('semantic_retrieval', {}).get('avg_score', 0)
        basic_avg = comparison.get('basic_retrieval', {}).get('avg_score', 0)
        
        logger.info(f"Average scores:")
        logger.info(f"  Semantic: {semantic_avg:.3f}")
        logger.info(f"  Basic: {basic_avg:.3f}")
        logger.info(f"  Improvement: {semantic_avg - basic_avg:.3f}")
        
        # Test the specific problematic query
        logger.info("\nTesting problematic query resolution:")
        test_result = pipeline.test_problematic_query(test_query)
        
        success_criteria = test_result.get('success_criteria', {})
        logger.info(f"Success criteria:")
        logger.info(f"  Finds team info: {success_criteria.get('finds_team_info', False)}")
        logger.info(f"  Mentions specific names: {success_criteria.get('mentions_specific_names', False)}")
        logger.info(f"  Better than basic: {success_criteria.get('better_than_basic', False)}")
        
        return len(team_mentions) > 2
        
    except Exception as e:
        logger.error(f"Semantic pipeline test failed: {e}")
        return False


def test_edge_cases():
    """Test edge cases and conflicting keywords"""
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("TESTING EDGE CASES")
    logger.info("=" * 60)
    
    try:
        query_processor = QueryProcessor()
        
        edge_cases = [
            "collaborateurs projet configuration",  # Conflicting keywords
            "équipe système développement",        # Multiple intents
            "vincent nicolas configuration",       # Names + technical term
            "isschat collaborateurs team",         # Mixed languages
            "développeurs application projet",     # Generic terms
        ]
        
        for test_query in edge_cases:
            logger.info(f"Testing: '{test_query}'")
            result = query_processor.process_query(test_query)
            
            logger.info(f"  Intent: {result.intent}")
            logger.info(f"  Confidence: {result.confidence:.3f}")
            logger.info(f"  Keywords: {result.keywords}")
            logger.info(f"  Variations: {len(result.semantic_variations)}")
            logger.info("")
        
        return True
        
    except Exception as e:
        logger.error(f"Edge cases test failed: {e}")
        return False


def main():
    """Main test function"""
    logger = setup_logging()
    
    logger.info("Starting semantic retrieval tests...")
    logger.info("=" * 80)
    
    test_results = []
    
    # Test 1: Query Processing
    logger.info("\n1. Testing Query Processing...")
    test_results.append(("Query Processing", test_query_processing()))
    
    # Test 2: Semantic Retrieval
    logger.info("\n2. Testing Semantic Retrieval...")
    test_results.append(("Semantic Retrieval", test_semantic_retrieval()))
    
    # Test 3: Semantic Pipeline
    logger.info("\n3. Testing Semantic Pipeline...")
    test_results.append(("Semantic Pipeline", test_semantic_pipeline()))
    
    # Test 4: Edge Cases
    logger.info("\n4. Testing Edge Cases...")
    test_results.append(("Edge Cases", test_edge_cases()))
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    
    passed = 0
    failed = 0
    
    for test_name, result in test_results:
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    logger.info(f"\nTotal: {passed + failed} tests")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    
    if failed == 0:
        logger.info("🎉 All tests passed! Semantic understanding is working correctly.")
    else:
        logger.info("❌ Some tests failed. Check the logs above for details.")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)