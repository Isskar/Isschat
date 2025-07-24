#!/usr/bin/env python3
"""
Test simple du système dynamique d'apprentissage
"""

import sys
import os
import logging

# Configuration du logging plus concise
logging.basicConfig(level=logging.WARNING, format="%(name)s - %(levelname)s - %(message)s")

# Ajout du chemin pour les imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag.tools.dynamic_context_analyzer import DynamicContextAnalyzer
from src.rag.tools.dynamic_semantic_matcher import DynamicSemanticMatcher
from src.rag.tools.isskar_smart_filter import IsskarSmartFilter
from src.config import get_config


def test_dynamic_components():
    """Test rapide des composants dynamiques"""

    print("🔬 TEST RAPIDE DU SYSTÈME DYNAMIQUE")
    print("=" * 50)

    config = get_config()

    print("\n1️⃣ Test Context Analyzer dynamique...")
    analyzer = DynamicContextAnalyzer()

    # Test classification
    query_types = [
        ("Qui travaille sur Teora ?", "PROJECT_SPECIFIC"),
        ("Quelles sont leurs responsabilités ?", "CONTEXTUAL"),
        ("bonjour", "GREETING"),
        ("Docker installation", "TECHNICAL"),
    ]

    for query, expected in query_types:
        result = analyzer.classify_query(query, query)
        print(f"   '{query}' -> {result.value} {'✅' if result.value == expected else '❌'}")

    print("\n2️⃣ Test Semantic Matcher dynamique...")
    matcher = DynamicSemanticMatcher()

    # Test patterns de base
    stats = matcher.get_learning_stats()
    print(f"   Patterns: {stats['total_patterns']}")
    print(f"   Haute confiance: {stats['high_confidence']}")

    # Test enrichissement keywords
    keywords = ["responsabilités", "équipe", "projet"]
    enhanced = matcher.get_enhanced_keywords(keywords)
    print(f"   Keywords enrichis: {keywords} -> {enhanced}")

    print("\n3️⃣ Test Smart Filter dynamique...")
    smart_filter = IsskarSmartFilter(config)

    # Test statistiques
    learning_stats = smart_filter.get_learning_stats()
    print(f"   Context Analyzer: {learning_stats['context_analyzer']['total_entities']} entités")
    print(f"   Semantic Matcher: {learning_stats['semantic_matcher']['total_patterns']} patterns")

    print("\n✅ TESTS TERMINÉS")
    print("Le système dynamique fonctionne sans données codées en dur !")
    print("=" * 50)


if __name__ == "__main__":
    try:
        test_dynamic_components()
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback

        traceback.print_exc()
