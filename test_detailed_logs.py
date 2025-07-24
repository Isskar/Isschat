#!/usr/bin/env python3
"""
Test script to demonstrate detailed logging of the context-aware RAG pipeline.
Shows all steps: context enrichment, query expansion, Weaviate searches, chunks, and LLM prompts.
"""

import sys
import os
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def test_context_aware_conversation():
    """Test a multi-turn conversation with detailed logging"""
    try:
        from src.rag.semantic_pipeline import SemanticRAGPipelineFactory

        print("🚀 STARTING DETAILED CONTEXT-AWARE RAG TEST")
        print("=" * 80)

        # Create context-aware pipeline
        pipeline = SemanticRAGPipelineFactory.create_context_aware_pipeline()

        if not pipeline.is_ready():
            print("⚠️ Pipeline not ready - continuing anyway for demonstration")

        conversation_id = "demo_context_conversation"

        # Multi-turn conversation scenario
        conversation_turns = [
            {
                "turn": 1,
                "query": "Qui travaille sur le projet Teora ?",
                "description": "Initial query - establishes context about Teora project",
            },
            {
                "turn": 2,
                "query": "Quelles sont leurs responsabilités ?",
                "description": "Follow-up with implicit reference - should use Teora context",
            },
            {
                "turn": 3,
                "query": "Comment puis-je les contacter ?",
                "description": "Further implicit reference - should continue using context",
            },
        ]

        for turn_info in conversation_turns:
            print(f"\n🔄 CONVERSATION TURN {turn_info['turn']}")
            print(f"📝 USER QUERY: '{turn_info['query']}'")
            print(f"💡 DESCRIPTION: {turn_info['description']}")
            print("-" * 60)

            try:
                # Process query with verbose logging
                answer, sources = pipeline.process_query(
                    query=turn_info["query"], conversation_id=conversation_id, verbose=True, use_context_enrichment=True
                )

                print(f"\n✅ FINAL ANSWER: {answer[:300]}...")
                if len(answer) > 300:
                    print(f"    (Total answer length: {len(answer)} characters)")

                if sources:
                    print(f"📚 SOURCES PROVIDED: Yes ({len(sources)} characters)")
                else:
                    print("📚 SOURCES PROVIDED: No")

                # Show conversation context state
                if pipeline.context_tracker:
                    context_summary = pipeline.context_tracker.get_conversation_summary(conversation_id)
                    print(f"\n🧠 CONTEXT STATE AFTER TURN {turn_info['turn']}:")
                    print(f"   - Active entities: {context_summary.get('active_entities', 0)}")
                    print(f"   - Topics tracked: {context_summary.get('topics', [])}")
                    if context_summary.get("entities_by_type"):
                        for entity_type, entities in context_summary["entities_by_type"].items():
                            print(f"   - {entity_type}: {entities}")

            except Exception as e:
                print(f"❌ ERROR in turn {turn_info['turn']}: {e}")

            print("=" * 80)

        print("\n🎉 DETAILED LOGGING TEST COMPLETED")
        print("This shows the complete pipeline flow:")
        print("1. 🔍 Query sent to Weaviate (with context enrichment)")
        print("2. 🧠 Semantic expansion and query variations")
        print("3. 🔎 Multiple searches in Weaviate")
        print("4. 📄 Retrieved chunks with scores")
        print("5. 📋 Document filtering")
        print("6. 🤖 Complete prompt sent to LLM")
        print("7. 🤖 LLM response")
        print("8. 🧩 Context tracking for next turn")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


def test_single_query_comparison():
    """Test to compare single query with and without context awareness"""
    try:
        from src.rag.semantic_pipeline import SemanticRAGPipelineFactory

        print("\n🔬 COMPARISON TEST: Single Query vs Context-Aware")
        print("=" * 80)

        test_query = "Qui sont les développeurs ?"

        # Test with context-aware pipeline
        print("\n🧩 WITH CONTEXT AWARENESS:")
        print("-" * 40)
        context_pipeline = SemanticRAGPipelineFactory.create_context_aware_pipeline()
        answer1, _ = context_pipeline.process_query(query=test_query, conversation_id="comparison_test", verbose=True)

        # Test with basic pipeline
        print("\n🔧 WITHOUT CONTEXT AWARENESS:")
        print("-" * 40)
        basic_pipeline = SemanticRAGPipelineFactory.create_basic_pipeline()
        answer2, _ = basic_pipeline.process_query(query=test_query, verbose=True)

        print("\n📊 COMPARISON RESULTS:")
        print(f"Context-aware answer length: {len(answer1)} chars")
        print(f"Basic answer length: {len(answer2)} chars")
        print(f"Answers are similar: {answer1[:100] == answer2[:100]}")

    except Exception as e:
        print(f"❌ Comparison test failed: {e}")


if __name__ == "__main__":
    try:
        # Run the main test
        test_context_aware_conversation()

        # Run comparison test
        test_single_query_comparison()

    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Test suite crashed: {e}")
        sys.exit(1)
