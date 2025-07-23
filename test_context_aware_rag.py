#!/usr/bin/env python3
"""
Test script for context-aware RAG functionality.
Tests multi-turn conversations with implicit references to ensure context enrichment works properly.
"""

import sys
import os
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_entity_extraction():
    """Test entity extraction from conversation context"""
    try:
        from src.rag.conversation_context import ConversationContextTracker

        logger.info("🧪 Testing entity extraction...")

        tracker = ConversationContextTracker()

        # Test cases for entity extraction
        test_cases = [
            {
                "text": "Qui travaille sur le projet Teora avec Vincent Fraillon ?",
                "expected_entities": ["teora", "vincent", "fraillon"],
                "expected_types": ["project", "person", "person"],
            },
            {
                "text": "L'équipe développement utilise Azure et Streamlit",
                "expected_entities": ["azure", "streamlit"],
                "expected_types": ["technology", "technology"],
            },
            {
                "text": "Dans le module de configuration, Nicolas Lambropoulos",
                "expected_entities": ["configuration", "nicolas", "lambropoulos"],
                "expected_types": ["feature", "person", "person"],
            },
        ]

        results = []
        for i, test_case in enumerate(test_cases):
            entities = tracker._extract_entities(test_case["text"])

            extracted_names = [e.entity.lower() for e in entities]
            expected_names = [name.lower() for name in test_case["expected_entities"]]

            matches = sum(1 for exp in expected_names if any(exp in ext for ext in extracted_names))
            success = matches >= len(expected_names) * 0.7  # At least 70% match

            result = {
                "test": i + 1,
                "text": test_case["text"],
                "expected": expected_names,
                "extracted": extracted_names,
                "matches": matches,
                "success": success,
            }
            results.append(result)

            logger.info(f"Test {i + 1}: {'✅' if success else '❌'} - {matches}/{len(expected_names)} entities found")

        overall_success = sum(r["success"] for r in results) >= len(results) * 0.8
        logger.info(f"🎯 Entity extraction test: {'✅ PASSED' if overall_success else '❌ FAILED'}")

        return {"test_name": "entity_extraction", "success": overall_success, "results": results}

    except Exception as e:
        logger.error(f"❌ Entity extraction test failed: {e}")
        return {"test_name": "entity_extraction", "success": False, "error": str(e)}


def test_context_enrichment():
    """Test context-aware query enrichment"""
    try:
        from src.rag.conversation_context import ConversationContextTracker

        logger.info("🧪 Testing context enrichment...")

        tracker = ConversationContextTracker()
        conversation_id = "test_context_enrichment"

        # Simulate conversation turns
        turns = [
            {
                "query": "Qui travaille sur le projet Teora ?",
                "response": "Vincent Fraillon et Nicolas Lambropoulos travaillent sur Teora.",
                "intent": "team_info",
            },
            {
                "query": "Quelles sont leurs responsabilités ?",
                "response": "Vincent est responsable du développement backend, Nicolas du frontend.",
                "intent": "team_info",
            },
        ]

        # Track conversation turns
        for turn in turns:
            tracker.track_conversation_turn(
                conversation_id=conversation_id, query=turn["query"], response=turn["response"], intent=turn["intent"]
            )

        # Test context enrichment for follow-up query
        follow_up_query = "Comment puis-je les contacter ?"
        enriched_query, context_metadata = tracker.enrich_query_with_context(conversation_id, follow_up_query)

        # Check if context was applied
        context_applied = context_metadata.get("context_applied", False)
        has_teora_context = "teora" in enriched_query.lower() if context_applied else False
        has_person_context = any(
            name in enriched_query.lower() for name in ["vincent", "nicolas", "fraillon", "lambropoulos"]
        )

        # Get conversation summary
        summary = tracker.get_conversation_summary(conversation_id)

        success = (
            context_applied and (has_teora_context or has_person_context) and summary.get("active_entities", 0) > 0
        )

        result = {
            "test_name": "context_enrichment",
            "success": success,
            "original_query": follow_up_query,
            "enriched_query": enriched_query if context_applied else follow_up_query,
            "context_applied": context_applied,
            "context_metadata": context_metadata,
            "conversation_summary": summary,
            "checks": {
                "context_applied": context_applied,
                "has_project_context": has_teora_context,
                "has_person_context": has_person_context,
                "entities_tracked": summary.get("active_entities", 0) > 0,
            },
        }

        logger.info(f"🎯 Context enrichment test: {'✅ PASSED' if success else '❌ FAILED'}")
        if context_applied:
            logger.info(f"📝 Query enriched: '{follow_up_query}' -> '{enriched_query}'")

        return result

    except Exception as e:
        logger.error(f"❌ Context enrichment test failed: {e}")
        return {"test_name": "context_enrichment", "success": False, "error": str(e)}


def test_pipeline_integration():
    """Test full pipeline integration with context awareness"""
    try:
        from src.rag.semantic_pipeline import SemanticRAGPipelineFactory

        logger.info("🧪 Testing pipeline integration...")

        # Create context-aware pipeline
        pipeline = SemanticRAGPipelineFactory.create_context_aware_pipeline()

        if not pipeline.is_ready():
            logger.warning("⚠️ Pipeline not ready - some tests may fail")

        # Test pipeline status
        status = pipeline.get_status()
        context_enabled = status.get("context_awareness_enabled", False)

        if not context_enabled:
            return {
                "test_name": "pipeline_integration",
                "success": False,
                "error": "Context awareness not enabled in pipeline",
            }

        # Test context-aware conversation simulation
        test_result = pipeline.test_context_aware_conversation()

        success = test_result.get("success", False)
        logger.info(f"🎯 Pipeline integration test: {'✅ PASSED' if success else '❌ FAILED'}")

        return {
            "test_name": "pipeline_integration",
            "success": success,
            "pipeline_status": status,
            "context_test_result": test_result,
        }

    except Exception as e:
        logger.error(f"❌ Pipeline integration test failed: {e}")
        return {"test_name": "pipeline_integration", "success": False, "error": str(e)}


def test_single_turn_regression():
    """Test that single-turn queries still work without regression"""
    try:
        from src.rag.semantic_pipeline import SemanticRAGPipelineFactory

        logger.info("🧪 Testing single-turn query regression...")

        # Create both context-aware and basic pipelines
        context_pipeline = SemanticRAGPipelineFactory.create_context_aware_pipeline()
        basic_pipeline = SemanticRAGPipelineFactory.create_basic_pipeline()

        # Test single-turn queries
        test_queries = [
            "Qu'est-ce qu'IssCHat ?",
            "Comment configurer l'application ?",
            "Qui sont les développeurs ?",
            "Quelles sont les fonctionnalités principales ?",
        ]

        results = []
        for query in test_queries:
            try:
                # Test with context-aware pipeline
                context_answer, context_sources = context_pipeline.process_query(
                    query=query, conversation_id=f"regression_test_{hash(query)}", use_context_enrichment=True
                )

                # Test with basic pipeline for comparison
                basic_answer, basic_sources = basic_pipeline.process_query(
                    query=query, conversation_id=None, use_context_enrichment=False
                )

                # Check if both pipelines produce reasonable responses
                context_success = (
                    len(context_answer) > 10
                    and "error" not in context_answer.lower()
                    and context_answer != "Model unavailable"
                )

                basic_success = (
                    len(basic_answer) > 10
                    and "error" not in basic_answer.lower()
                    and basic_answer != "Model unavailable"
                )

                result = {
                    "query": query,
                    "context_pipeline_success": context_success,
                    "basic_pipeline_success": basic_success,
                    "context_answer_length": len(context_answer),
                    "basic_answer_length": len(basic_answer),
                    "regression_test_passed": context_success and basic_success,
                }
                results.append(result)

                logger.info(
                    f"Query: '{query[:30]}...' - Context: {'✅' if context_success else '❌'}, Basic: {'✅' if basic_success else '❌'}"
                )

            except Exception as e:
                logger.error(f"Error testing query '{query}': {e}")
                results.append(
                    {
                        "query": query,
                        "context_pipeline_success": False,
                        "basic_pipeline_success": False,
                        "error": str(e),
                        "regression_test_passed": False,
                    }
                )

        # Overall success if most queries work
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.get("regression_test_passed", False))
        success_rate = passed_tests / total_tests if total_tests > 0 else 0

        overall_success = success_rate >= 0.8  # 80% success rate

        logger.info(
            f"🎯 Single-turn regression test: {'✅ PASSED' if overall_success else '❌ FAILED'} ({passed_tests}/{total_tests} passed)"
        )

        return {
            "test_name": "single_turn_regression",
            "success": overall_success,
            "success_rate": success_rate,
            "passed_tests": passed_tests,
            "total_tests": total_tests,
            "results": results,
        }

    except Exception as e:
        logger.error(f"❌ Single-turn regression test failed: {e}")
        return {"test_name": "single_turn_regression", "success": False, "error": str(e)}


def run_all_tests():
    """Run all context-aware RAG tests"""
    logger.info("🚀 Starting context-aware RAG tests...")
    logger.info("=" * 60)

    tests = [test_entity_extraction, test_context_enrichment, test_pipeline_integration, test_single_turn_regression]

    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            logger.error(f"❌ Test {test_func.__name__} crashed: {e}")
            results.append({"test_name": test_func.__name__, "success": False, "error": f"Test crashed: {e}"})

        logger.info("-" * 40)

    # Summary
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.get("success", False))

    logger.info("=" * 60)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 60)

    for result in results:
        test_name = result.get("test_name", "unknown")
        success = result.get("success", False)
        logger.info(f"{'✅' if success else '❌'} {test_name}")
        if not success and "error" in result:
            logger.info(f"   Error: {result['error']}")

    logger.info("-" * 60)
    logger.info(f"Overall: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        logger.info("🎉 ALL TESTS PASSED! Context-aware RAG is working correctly.")
    elif passed_tests >= total_tests * 0.8:
        logger.info("⚠️ Most tests passed, but some issues detected.")
    else:
        logger.info("❌ Multiple test failures detected. Context-aware RAG needs attention.")

    return {
        "overall_success": passed_tests >= total_tests * 0.8,
        "passed_tests": passed_tests,
        "total_tests": total_tests,
        "detailed_results": results,
    }


if __name__ == "__main__":
    try:
        results = run_all_tests()

        # Exit with appropriate code
        exit_code = 0 if results["overall_success"] else 1
        sys.exit(exit_code)

    except KeyboardInterrupt:
        logger.info("\n👋 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Test suite crashed: {e}")
        sys.exit(1)
