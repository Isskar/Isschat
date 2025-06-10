#!/usr/bin/env python3
"""
Test script for the RAG Evaluation System
"""

import os
import sys
import logging
from typing import List, Tuple

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.evaluation import EvaluationManager
from config.evaluation_config import get_evaluation_config, get_config_debug_info


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def mock_get_response(question: str) -> Tuple[str, List[str]]:
    """
    Mock function that simulates getting a response from Isschat

    Args:
        question: The question to answer

    Returns:
        Tuple[str, List[str]]: (response, sources)
    """
    # Simple mock responses for testing
    mock_responses = {
        "Connais-tu Isschat ?": (
            "Oui, Isschat est un assistant virtuel basé sur l'intelligence artificielle qui aide les utilisateurs à naviguer et à trouver des informations dans la documentation Confluence d'Isskar. Il utilise la technologie RAG (Retrieval-Augmented Generation) pour fournir des réponses précises et contextuelles.",
            ["documentation_isschat.md", "guide_utilisateur.md"],
        ),
        "Parle moi du daily du 35 mai 2018 ?": (
            "Je remarque qu'il y a une erreur dans votre question : le 35 mai n'existe pas, car le mois de mai ne compte que 31 jours. Pourriez-vous préciser la date correcte que vous souhaitez consulter ?",
            [],
        ),
        "Connais tu Emin Lambropoulos ?": (
            "Je ne trouve pas d'informations sur une personne nommée Emin Lambropoulos dans la documentation d'Isskar. Cette personne ne semble pas faire partie de l'organisation ou n'est pas mentionnée dans les documents accessibles.",
            [],
        ),
        "A quoi sert l'eau ?": (
            "Votre question porte sur l'eau en général, ce qui sort du cadre de ma spécialisation sur la documentation d'Isskar. Je suis conçu pour vous aider avec des questions liées à l'entreprise, ses projets, ses processus et sa documentation. Avez-vous une question spécifique concernant Isskar ?",
            [],
        ),
        "Tell me more about Cedrus project.": (
            "Je vous réponds en français car c'est la langue de travail d'Isskar. Le projet Cedrus est un projet important de l'entreprise. Pour plus de détails spécifiques, je vous invite à consulter la documentation projet ou à contacter l'équipe responsable.",
            ["projet_cedrus.md"],
        ),
    }

    # Default response for unknown questions
    default_response = (
        "Je suis Isschat, votre assistant virtuel pour la documentation Isskar. Je peux vous aider avec des questions sur l'entreprise, ses projets et ses processus.",
        ["documentation_generale.md"],
    )

    return mock_responses.get(question, default_response)


def test_configuration():
    """Test configuration loading"""
    print("=== Testing Configuration ===")

    try:
        config = get_evaluation_config()
        debug_info = get_config_debug_info()

        print("✅ Configuration loaded successfully")
        print(f"LLM Model: {debug_info['llm_model']}")
        print(f"Database Type: {debug_info['database_type']}")
        print(f"Output Directory: {debug_info['output_dir']}")

        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_dataset_loading():
    """Test dataset loading"""
    print("\n=== Testing Dataset Loading ===")

    try:
        manager = EvaluationManager()

        # Test robustness dataset
        robustness_tests = manager.dataset_manager.load_dataset("robustness")
        print(f"✅ Loaded {len(robustness_tests)} robustness tests")

        # Test quality dataset
        quality_tests = manager.dataset_manager.load_dataset("quality")
        print(f"✅ Loaded {len(quality_tests)} quality tests")

        # Show dataset info
        datasets_info = manager.dataset_manager.list_all_datasets()
        for name, info in datasets_info.items():
            if info.get("exists"):
                print(f"📊 {name}: {info['total_cases']} test cases")

        return True
    except Exception as e:
        print(f"❌ Dataset loading test failed: {e}")
        return False


def test_llm_judge():
    """Test LLM Judge functionality"""
    print("\n=== Testing LLM Judge ===")

    try:
        from src.evaluation.llm_judge import LLMJudge
        from src.evaluation.models import RobustnessTestType

        judge = LLMJudge()

        # Test generation evaluation
        print("Testing generation evaluation...")
        generation_score = judge.evaluate_generation(
            question="Qu'est-ce qu'Isschat ?",
            context="Documentation sur Isschat, assistant virtuel",
            response="Isschat est un assistant virtuel basé sur l'IA pour la documentation Confluence.",
            expected_answer="Isschat est un assistant virtuel...",
        )

        print("✅ Generation evaluation completed")
        print(f"   Overall Score: {generation_score.overall_score:.1f}/10")
        print(f"   Relevance: {generation_score.relevance:.1f}/10")

        # Test robustness evaluation
        print("Testing robustness evaluation...")
        robustness_score = judge.evaluate_robustness(
            test_type=RobustnessTestType.KNOWLEDGE_INTERNAL,
            question="Connais-tu Isschat ?",
            response="Oui, Isschat est un assistant virtuel...",
            expected_behavior="Réponse contextuelle et informative",
        )

        print("✅ Robustness evaluation completed")
        print(f"   Score: {robustness_score.score:.1f}/10")
        print(f"   Passed: {robustness_score.passed}")

        return True
    except Exception as e:
        print(f"❌ LLM Judge test failed: {e}")
        print("Note: This might fail if OpenAI API key is not configured")
        return False


def test_quick_evaluation():
    """Test quick evaluation functionality"""
    print("\n=== Testing Quick Evaluation ===")

    try:
        manager = EvaluationManager()

        # Test with a few questions
        test_questions = ["Qu'est-ce qu'Isschat ?", "Connais-tu Emin Lambropoulos ?", "A quoi sert l'eau ?"]

        print(f"Running quick evaluation with {len(test_questions)} questions...")

        results = manager.run_quick_evaluation(
            questions=test_questions, get_response_func=mock_get_response, session_name="Test Quick Evaluation"
        )

        print("✅ Quick evaluation completed")
        print(f"   Total tests: {results['total_tests']}")
        print(f"   Execution time: {results['execution_time']:.2f}s")

        if "generation_stats" in results:
            stats = results["generation_stats"]
            print(f"   Average overall score: {stats['average_overall']:.1f}/10")

        return True
    except Exception as e:
        print(f"❌ Quick evaluation test failed: {e}")
        return False


def test_robustness_evaluation():
    """Test robustness evaluation"""
    print("\n=== Testing Robustness Evaluation ===")

    try:
        manager = EvaluationManager()

        print("Running robustness tests...")

        results = manager.run_robustness_tests(
            get_response_func=mock_get_response, session_name="Test Robustness Evaluation"
        )

        print("✅ Robustness evaluation completed")
        print(f"   Total tests: {results['total_tests']}")
        print(f"   Execution time: {results['execution_time']:.2f}s")

        if "robustness_stats" in results:
            stats = results["robustness_stats"]
            print(f"   Passed tests: {stats['passed_tests']}/{stats['total_robustness_tests']}")
            print(f"   Pass rate: {stats['pass_rate']:.1f}%")

        return True
    except Exception as e:
        print(f"❌ Robustness evaluation test failed: {e}")
        return False


def test_system_statistics():
    """Test system statistics"""
    print("\n=== Testing System Statistics ===")

    try:
        manager = EvaluationManager()
        stats = manager.get_system_statistics()

        print("✅ System statistics retrieved")
        print(f"   LLM Model: {stats['config']['llm_model']}")
        print(f"   Database Type: {stats['config']['database_type']}")

        # Show recent sessions
        sessions = manager.list_recent_sessions(limit=5)
        print(f"   Recent sessions: {len(sessions)}")

        return True
    except Exception as e:
        print(f"❌ System statistics test failed: {e}")
        return False


def main():
    """Main test function"""
    setup_logging()

    print("🚀 Starting RAG Evaluation System Tests")
    print("=" * 50)

    tests = [
        ("Configuration", test_configuration),
        ("Dataset Loading", test_dataset_loading),
        ("LLM Judge", test_llm_judge),
        ("Quick Evaluation", test_quick_evaluation),
        ("Robustness Evaluation", test_robustness_evaluation),
        ("System Statistics", test_system_statistics),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")

    print("\n" + "=" * 50)
    print(f"🏁 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! The evaluation system is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        print("Note: LLM Judge tests may fail if API keys are not configured.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
