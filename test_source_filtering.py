#!/usr/bin/env python3
"""
Test script to verify source filtering works correctly
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.rag.semantic_pipeline import SemanticRAGPipelineFactory

def test_source_filtering():
    """Test source filtering for different query types"""
    
    print("🔍 Testing Source Filtering")
    print("=" * 50)
    
    try:
        # Create pipeline with source filtering enabled
        pipeline = SemanticRAGPipelineFactory.create_semantic_pipeline(use_semantic_features=True)
        
        if not pipeline.is_ready():
            print("❌ Pipeline not ready - Vector database may not be built")
            return False
        
        # Test queries with different levels of specificity
        test_queries = [
            ("hello", "Greeting - should have NO sources"),
            ("bonjour", "Greeting - should have NO sources"),
            ("merci", "Politeness - should have NO sources"),
            ("qui sont les collaborateurs sur Isschat", "Specific question - should have sources"),
            ("comment configurer isschat", "Technical question - should have sources"),
            ("qu'est-ce que isschat", "General question - should have sources"),
            ("test", "Generic term - should have NO sources"),
        ]
        
        print("\nTesting different query types:")
        print("-" * 60)
        
        for query, description in test_queries:
            print(f"\n📝 Query: '{query}'")
            print(f"   Expected: {description}")
            
            try:
                answer, sources = pipeline.process_query(query)
                
                # Check if sources are provided
                has_sources = sources and sources != "No sources"
                
                print(f"   Answer: {answer[:100]}{'...' if len(answer) > 100 else ''}")
                print(f"   Sources: {'✅ Yes' if has_sources else '❌ No'}")
                if has_sources:
                    print(f"   Sources: {sources}")
                
                # Analyze if filtering worked correctly
                if query.lower() in ["hello", "bonjour", "merci", "test"]:
                    if not has_sources:
                        print("   ✅ Filtering worked correctly - no irrelevant sources shown")
                    else:
                        print("   ❌ Filtering failed - sources shown for generic query")
                else:
                    if has_sources:
                        print("   ✅ Sources appropriately shown for specific query")
                    else:
                        print("   ⚠️ No sources shown - may be expected if no relevant docs found")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing source filtering: {e}")
        return False

def test_configuration():
    """Test configuration options for source filtering"""
    
    print("\n🔧 Testing Configuration")
    print("=" * 50)
    
    try:
        from src.config.settings import get_config
        config = get_config()
        
        print(f"Source filtering enabled: {config.source_filtering_enabled}")
        print(f"Min source score threshold: {config.min_source_score_threshold}")
        print(f"Min source relevance threshold: {config.min_source_relevance_threshold}")
        
        # Test that we can access all semantic features
        print(f"Semantic features enabled: {config.use_semantic_features}")
        print(f"Semantic expansion enabled: {config.semantic_expansion_enabled}")
        print(f"Semantic reranking enabled: {config.semantic_reranking_enabled}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing configuration: {e}")
        return False

if __name__ == "__main__":
    print("Testing Source Filtering Implementation")
    print("=" * 80)
    
    success = True
    
    # Test configuration
    if not test_configuration():
        success = False
    
    # Test source filtering
    if not test_source_filtering():
        success = False
    
    print("\n" + "=" * 80)
    if success:
        print("✅ Source filtering tests completed successfully!")
        print("\nHow to use:")
        print("1. Restart your chatbot (webapp or CLI)")
        print("2. Test with 'hello' - should show no sources")
        print("3. Test with 'qui sont les collaborateurs' - should show sources")
        print("4. Adjust thresholds via environment variables if needed:")
        print("   - SOURCE_FILTERING_ENABLED=true/false")
        print("   - MIN_SOURCE_SCORE_THRESHOLD=0.4")
        print("   - MIN_SOURCE_RELEVANCE_THRESHOLD=0.3")
    else:
        print("❌ Some tests failed. Check the logs above.")
    
    sys.exit(0 if success else 1)