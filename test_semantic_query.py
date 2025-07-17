#!/usr/bin/env python3
"""
Quick test script to verify semantic understanding works
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.rag.semantic_pipeline import SemanticRAGPipelineFactory
from src.rag.query_processor import QueryProcessor

def test_semantic_understanding():
    """Test semantic understanding for the problematic query"""
    
    print("🧠 Testing Semantic Understanding")
    print("=" * 50)
    
    # Test 1: Query Processing
    print("\n1. Testing Query Processing:")
    query_processor = QueryProcessor()
    
    test_query = "qui sont les collaborateurs sur Isschat"
    print(f"Query: '{test_query}'")
    
    result = query_processor.process_query(test_query)
    print(f"  Intent: {result.intent}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Keywords: {result.keywords}")
    print(f"  Expanded queries: {len(result.expanded_queries)}")
    
    for i, expanded in enumerate(result.expanded_queries[:3], 1):
        print(f"    {i}. {expanded}")
    
    # Test 2: Semantic Pipeline (if vector DB is available)
    print("\n2. Testing Semantic Pipeline:")
    try:
        pipeline = SemanticRAGPipelineFactory.create_semantic_pipeline(use_semantic_features=True)
        
        if pipeline.is_ready():
            print("  ✅ Pipeline is ready")
            
            # Test the problematic query
            print(f"\n  Testing query: '{test_query}'")
            answer, sources = pipeline.process_query(test_query, verbose=True)
            
            print(f"  Answer: {answer[:200]}{'...' if len(answer) > 200 else ''}")
            print(f"  Sources: {sources}")
            
            # Test comparison
            comparison = pipeline.compare_with_basic_retrieval(test_query)
            
            semantic_avg = comparison.get('semantic_retrieval', {}).get('avg_score', 0)
            basic_avg = comparison.get('basic_retrieval', {}).get('avg_score', 0)
            improvement = semantic_avg - basic_avg
            
            print(f"\n  Comparison Results:")
            print(f"    Semantic retrieval avg score: {semantic_avg:.3f}")
            print(f"    Basic retrieval avg score: {basic_avg:.3f}")
            print(f"    Improvement: {improvement:.3f} ({'+' if improvement > 0 else ''}{improvement:.1%})")
            
            # Check if answer contains team information
            team_keywords = ["vincent", "nicolas", "emin", "fraillon", "lambropoulos", "calyaka", "équipe"]
            answer_lower = answer.lower()
            team_mentions = [keyword for keyword in team_keywords if keyword in answer_lower]
            
            print(f"\n  Team Information Found:")
            print(f"    Keywords found: {team_mentions}")
            print(f"    Contains team info: {'✅' if len(team_mentions) > 1 else '❌'}")
            
        else:
            print("  ❌ Pipeline not ready - Vector database may not be built")
            print("  Run: python -m src.cli.main ingest to build the database first")
            
    except Exception as e:
        print(f"  ❌ Error testing pipeline: {e}")
        print("  Make sure the vector database is built and accessible")
    
    # Test 3: Other Edge Cases
    print("\n3. Testing Edge Cases:")
    edge_cases = [
        "configuration de l'équipe",
        "développeurs du système isschat",
        "collaborateurs projet",
        "équipe développement",
    ]
    
    for query in edge_cases:
        result = query_processor.process_query(query)
        print(f"  '{query}' -> Intent: {result.intent}, Confidence: {result.confidence:.2f}")
    
    print("\n" + "=" * 50)
    print("✅ Semantic understanding test completed!")
    print("\nTo use in your chatbot:")
    print("1. The webapp/app.py has been updated to use semantic pipeline")
    print("2. The CLI chat command has been updated")
    print("3. Just restart your chatbot to use the new features")

if __name__ == "__main__":
    test_semantic_understanding()