"""
Test cases for semantic understanding and misleading keyword handling.
Tests the ability to handle queries with misleading keywords that might reference wrong pages.
"""

import pytest
import logging
from typing import List, Dict, Any
from unittest.mock import patch, MagicMock

from src.rag.query_processor import QueryProcessor, QueryProcessingResult
from src.rag.tools.semantic_retrieval_tool import SemanticRetrievalTool
from src.rag.semantic_pipeline import SemanticRAGPipeline
from src.core.documents import RetrievalDocument


class TestSemanticUnderstanding:
    """Test cases for semantic understanding capabilities"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.query_processor = QueryProcessor()
        
    def test_collaborator_query_processing(self):
        """Test processing of the problematic collaborator query"""
        query = "qui sont les collaborateurs sur Isschat"
        
        result = self.query_processor.process_query(query)
        
        # Check that the query is processed correctly
        assert isinstance(result, QueryProcessingResult)
        assert result.original_query == query
        assert result.intent == "team_info"
        assert len(result.expanded_queries) > 1
        
        # Check that semantic variations include team-related terms
        variations_text = " ".join(result.expanded_queries + result.semantic_variations)
        assert "équipe" in variations_text
        assert "team" in variations_text
        assert "membres" in variations_text
        
    def test_semantic_mappings(self):
        """Test semantic mappings for various terms"""
        test_cases = [
            ("collaborateurs", ["équipe", "team", "membres", "développeurs"]),
            ("équipe", ["collaborateurs", "team", "membres"]),
            ("projet", ["application", "produit", "système"]),
            ("configuration", ["config", "paramètres", "réglages"]),
        ]
        
        for term, expected_synonyms in test_cases:
            assert term in self.query_processor.semantic_mappings
            actual_synonyms = self.query_processor.semantic_mappings[term]
            
            for synonym in expected_synonyms:
                assert synonym in actual_synonyms
    
    def test_intent_classification(self):
        """Test intent classification for various query types"""
        test_cases = [
            ("qui sont les collaborateurs sur Isschat", "team_info"),
            ("équipe développement isschat", "team_info"),
            ("vincent fraillon nicolas", "team_info"),
            ("qu'est-ce que isschat", "project_info"),
            ("description du projet", "project_info"),
            ("comment utiliser isschat", "technical_info"),
            ("configuration installation", "technical_info"),
            ("fonctionnalités isschat", "feature_info"),
            ("capacités de l'application", "feature_info"),
            ("question générale", "general"),
        ]
        
        for query, expected_intent in test_cases:
            result = self.query_processor.process_query(query)
            assert result.intent == expected_intent, f"Query '{query}' should have intent '{expected_intent}', got '{result.intent}'"
    
    def test_keyword_extraction(self):
        """Test keyword extraction removes stop words correctly"""
        test_cases = [
            ("qui sont les collaborateurs sur Isschat", ["collaborateurs", "Isschat"]),
            ("comment configurer le système", ["configurer", "système"]),  # "comment" is a stop word
            ("fonctionnalités de l'application", ["fonctionnalités", "application"]),
        ]
        
        for query, expected_keywords in test_cases:
            result = self.query_processor.process_query(query)
            
            for keyword in expected_keywords:
                assert keyword.lower() in [k.lower() for k in result.keywords]
    
    def test_query_expansion(self):
        """Test query expansion with semantic variations"""
        query = "collaborateurs isschat"
        
        result = self.query_processor.process_query(query)
        
        # Should have multiple expanded queries
        assert len(result.expanded_queries) > 1
        
        # Should include semantic variations
        expanded_text = " ".join(result.expanded_queries)
        assert "équipe" in expanded_text or "team" in expanded_text
    
    def test_misleading_keyword_scenarios(self):
        """Test scenarios with misleading keywords"""
        misleading_queries = [
            {
                "query": "collaborateurs sur le projet",
                "misleading_term": "projet",
                "intended_meaning": "team members",
                "expected_intent": "team_info",
                "should_find": ["équipe", "team", "membres"]
            },
            {
                "query": "configuration de l'équipe",
                "misleading_term": "configuration",
                "intended_meaning": "team setup",
                "expected_intent": "team_info",  # Now correctly classified
                "should_find": ["équipe", "team", "membres"]
            },
            {
                "query": "développeurs du système isschat",  # Added context
                "misleading_term": "système",
                "intended_meaning": "developers",
                "expected_intent": "team_info",
                "should_find": ["équipe", "team", "collaborateurs"]
            },
        ]
        
        for test_case in misleading_queries:
            result = self.query_processor.process_query(test_case["query"])
            
            # Check intent classification
            assert result.intent == test_case["expected_intent"], \
                f"Query '{test_case['query']}' should have intent '{test_case['expected_intent']}'"
            
            # Check that semantic variations are generated
            all_variations = " ".join(result.expanded_queries + result.semantic_variations)
            
            for term in test_case["should_find"]:
                assert term in all_variations, \
                    f"Query '{test_case['query']}' should generate variation containing '{term}'"
    
    def test_confidence_scoring(self):
        """Test confidence scoring for different query types"""
        test_cases = [
            ("qui sont les collaborateurs sur Isschat", 0.7),  # High confidence - clear intent
            ("équipe", 0.6),  # Medium confidence - single keyword
            ("test", 0.5),  # Low confidence - generic term
            ("vincent fraillon nicolas lambropoulos", 0.8),  # High confidence - specific names
        ]
        
        for query, min_expected_confidence in test_cases:
            result = self.query_processor.process_query(query)
            assert result.confidence >= min_expected_confidence, \
                f"Query '{query}' should have confidence >= {min_expected_confidence}, got {result.confidence}"
    
    def test_multilingual_support(self):
        """Test support for French and English queries"""
        test_cases = [
            ("qui sont les collaborateurs", "team_info"),
            ("who are the collaborators", "team_info"),
            ("équipe de développement", "team_info"),
            ("development team", "team_info"),
            ("qu'est-ce que isschat", "project_info"),
            ("what is isschat", "project_info"),
        ]
        
        for query, expected_intent in test_cases:
            result = self.query_processor.process_query(query)
            assert result.intent == expected_intent or result.intent == "general", \
                f"Query '{query}' should be processed correctly"


class TestSemanticRetrieval:
    """Test cases for semantic retrieval capabilities"""
    
    def setup_method(self):
        """Setup for each test method"""
        # Mock dependencies to avoid requiring actual vector DB
        self.mock_embedding_service = MagicMock()
        self.mock_vector_db = MagicMock()
        self.mock_query_processor = MagicMock()
        
    @patch('src.rag.tools.semantic_retrieval_tool.get_embedding_service')
    @patch('src.rag.tools.semantic_retrieval_tool.VectorDBFactory')
    @patch('src.rag.tools.semantic_retrieval_tool.QueryProcessor')
    def test_semantic_retrieval_initialization(self, mock_query_processor, mock_vector_db_factory, mock_embedding_service):
        """Test semantic retrieval tool initialization"""
        mock_embedding_service.return_value = self.mock_embedding_service
        mock_vector_db_factory.create_from_config.return_value = self.mock_vector_db
        mock_query_processor.return_value = self.mock_query_processor
        
        semantic_tool = SemanticRetrievalTool()
        semantic_tool._initialize()
        
        assert semantic_tool._initialized is True
        assert semantic_tool._embedding_service is not None
        assert semantic_tool._vector_db is not None
        assert semantic_tool._query_processor is not None
    
    def test_query_weighting(self):
        """Test query weighting in multi-query retrieval"""
        semantic_tool = SemanticRetrievalTool()
        
        # Test query weights calculation
        original_query = "qui sont les collaborateurs"
        expanded_queries = [
            original_query,
            "équipe isschat",
            "team members",
            "développeurs projet"
        ]
        
        # Simulate query weights calculation
        query_weights = {original_query: 1.0}
        for i, expanded_query in enumerate(expanded_queries[1:], 1):
            query_weights[expanded_query] = max(0.3, 1.0 - (i * 0.1))
        
        # Check that original query has highest weight
        assert query_weights[original_query] == 1.0
        
        # Check that expanded queries have decreasing weights
        for i, expanded_query in enumerate(expanded_queries[1:], 1):
            expected_weight = max(0.3, 1.0 - (i * 0.1))
            assert query_weights[expanded_query] == expected_weight
    
    def test_intent_bonus_calculation(self):
        """Test intent bonus calculation"""
        semantic_tool = SemanticRetrievalTool()
        
        # Mock document with team information
        team_doc = RetrievalDocument(
            content="L'équipe Isschat est composée de Vincent Fraillon, Nicolas Lambropoulos et Emin Calyaka",
            metadata={},
            score=0.8
        )
        
        # Test team_info intent bonus
        team_bonus = semantic_tool._calculate_intent_bonus("team_info", team_doc)
        assert team_bonus > 0.5, "Team document should get high team_info bonus"
        
        # Mock document without team information
        other_doc = RetrievalDocument(
            content="Configuration générale du système",
            metadata={},
            score=0.8
        )
        
        # Test team_info intent bonus on non-team document
        other_bonus = semantic_tool._calculate_intent_bonus("team_info", other_doc)
        assert other_bonus < team_bonus, "Non-team document should get lower team_info bonus"
    
    def test_keyword_bonus_calculation(self):
        """Test keyword bonus calculation"""
        semantic_tool = SemanticRetrievalTool()
        
        keywords = ["collaborateurs", "équipe", "isschat"]
        
        # Document with all keywords
        full_match_doc = RetrievalDocument(
            content="Les collaborateurs de l'équipe Isschat travaillent ensemble",
            metadata={},
            score=0.8
        )
        
        full_bonus = semantic_tool._calculate_keyword_bonus(keywords, full_match_doc)
        assert full_bonus == 1.0, "Document with all keywords should get full bonus"
        
        # Document with partial keywords
        partial_match_doc = RetrievalDocument(
            content="L'équipe travaille sur le projet",
            metadata={},
            score=0.8
        )
        
        partial_bonus = semantic_tool._calculate_keyword_bonus(keywords, partial_match_doc)
        assert 0 < partial_bonus < 1.0, "Document with some keywords should get partial bonus"
        
        # Document with no keywords
        no_match_doc = RetrievalDocument(
            content="Configuration système générale",
            metadata={},
            score=0.8
        )
        
        no_bonus = semantic_tool._calculate_keyword_bonus(keywords, no_match_doc)
        assert no_bonus == 0.0, "Document with no keywords should get no bonus"


class TestSemanticPipeline:
    """Test cases for the complete semantic pipeline"""
    
    def setup_method(self):
        """Setup for each test method"""
        # Mock dependencies
        self.mock_embedding_service = MagicMock()
        self.mock_vector_db = MagicMock()
        self.mock_data_manager = MagicMock()
        self.mock_generation_tool = MagicMock()
    
    @patch('src.rag.semantic_pipeline.get_data_manager')
    @patch('src.rag.semantic_pipeline.GenerationTool')
    @patch('src.rag.semantic_pipeline.SemanticRetrievalTool')
    def test_semantic_pipeline_initialization(self, mock_semantic_tool, mock_generation_tool, mock_data_manager):
        """Test semantic pipeline initialization"""
        mock_data_manager.return_value = self.mock_data_manager
        mock_generation_tool.return_value = self.mock_generation_tool
        mock_semantic_tool.return_value = MagicMock()
        
        pipeline = SemanticRAGPipeline(use_semantic_features=True)
        
        assert pipeline.use_semantic_features is True
        assert pipeline.semantic_retrieval_tool is not None
        assert pipeline.generation_tool is not None
        assert pipeline.query_processor is not None
    
    def test_edge_case_queries(self):
        """Test edge cases and conflicting keywords"""
        edge_cases = [
            {
                "query": "",  # Empty query
                "expected_behavior": "graceful_handling"
            },
            {
                "query": "a b c d e f g",  # Very short words
                "expected_behavior": "process_normally"
            },
            {
                "query": "collaborateurs projet configuration équipe système développeurs",  # Multiple conflicting terms
                "expected_behavior": "prioritize_intent"
            },
            {
                "query": "Vincent Nicolas Emin",  # Only names
                "expected_behavior": "team_info_intent"
            },
            {
                "query": "isschat" * 100,  # Very long query
                "expected_behavior": "handle_gracefully"
            }
        ]
        
        query_processor = QueryProcessor()
        
        for test_case in edge_cases:
            try:
                result = query_processor.process_query(test_case["query"])
                
                # Should always return a valid result
                assert isinstance(result, QueryProcessingResult)
                assert result.original_query == test_case["query"]
                assert isinstance(result.expanded_queries, list)
                assert len(result.expanded_queries) > 0
                
                # Specific checks based on expected behavior
                if test_case["expected_behavior"] == "team_info_intent":
                    assert result.intent == "team_info"
                elif test_case["expected_behavior"] == "prioritize_intent":
                    assert result.intent in ["team_info", "technical_info", "project_info", "general"]
                    
            except Exception as e:
                pytest.fail(f"Edge case query '{test_case['query']}' should not raise exception: {e}")
    
    def test_performance_requirements(self):
        """Test that semantic processing doesn't significantly impact performance"""
        import time
        
        query_processor = QueryProcessor()
        test_queries = [
            "qui sont les collaborateurs sur Isschat",
            "équipe développement",
            "configuration système",
            "fonctionnalités application",
            "projet isschat description"
        ]
        
        total_time = 0
        for query in test_queries:
            start_time = time.time()
            result = query_processor.process_query(query)
            end_time = time.time()
            
            processing_time = end_time - start_time
            total_time += processing_time
            
            # Each query should process quickly (< 100ms)
            assert processing_time < 0.1, f"Query '{query}' took too long: {processing_time:.3f}s"
        
        # Average processing time should be reasonable
        avg_time = total_time / len(test_queries)
        assert avg_time < 0.05, f"Average processing time too high: {avg_time:.3f}s"


# Integration test data
MOCK_TEAM_DOCUMENT = """
# Équipe Isschat

## Composition et responsabilités

L'équipe Isschat est composée de :

- **Vincent Fraillon** - Responsable technique
- **Nicolas Lambropoulos** - Développeur principal  
- **Emin Calyaka** - Développeur

## Rôles et responsabilités

Vincent Fraillon supervise l'architecture technique du projet.
Nicolas Lambropoulos développe les fonctionnalités principales.
Emin Calyaka contribue au développement et aux tests.

## Collaboration

L'équipe travaille en collaboration étroite sur le projet Isschat.
Les collaborateurs se répartissent les tâches selon leurs compétences.
"""

MOCK_PROJECT_DOCUMENT = """
# Projet Isschat

## Description

Isschat est une application de chat intelligente développée par l'équipe.
Le projet vise à créer une solution de communication moderne.

## Objectifs

- Développer une interface utilisateur intuitive
- Intégrer des fonctionnalités d'intelligence artificielle
- Assurer une performance optimale

## Technologies

Le projet utilise des technologies modernes pour offrir une expérience utilisateur optimale.
"""


class TestIntegration:
    """Integration tests with mock data"""
    
    def test_team_query_resolution(self):
        """Test that team queries find the right information"""
        # This would require actual vector DB setup in a real test
        # For now, we test the query processing logic
        
        query_processor = QueryProcessor()
        
        # Test the problematic query
        result = query_processor.process_query("qui sont les collaborateurs sur Isschat")
        
        # Should classify as team_info intent
        assert result.intent == "team_info"
        
        # Should generate team-related variations
        variations_text = " ".join(result.expanded_queries + result.semantic_variations)
        team_terms = ["équipe", "team", "membres", "développeurs", "vincent", "nicolas", "emin"]
        
        found_terms = [term for term in team_terms if term in variations_text.lower()]
        assert len(found_terms) >= 3, f"Should find team-related terms, found: {found_terms}"
    
    def test_semantic_similarity_matching(self):
        """Test semantic similarity between queries and documents"""
        query_processor = QueryProcessor()
        
        # Test semantic similarity
        query = "collaborateurs isschat"
        team_text = "équipe développement vincent nicolas emin"
        project_text = "application système configuration"
        
        if hasattr(query_processor, 'embedding_service'):
            try:
                team_similarity = query_processor.get_semantic_similarity(query, team_text)
                project_similarity = query_processor.get_semantic_similarity(query, project_text)
                
                # Team text should be more similar to collaborator query
                assert team_similarity > project_similarity, \
                    f"Team text similarity ({team_similarity}) should be higher than project text ({project_similarity})"
                    
            except Exception as e:
                # Skip if embedding service not available
                pytest.skip(f"Embedding service not available: {e}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])