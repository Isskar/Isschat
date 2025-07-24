#!/usr/bin/env python3
"""
Analyseur de contexte dynamique pour Isskar
Extrait automatiquement les entités depuis les documents Confluence
"""

import re
import logging
from typing import List, Dict, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict

from .isskar_context_analyzer import QueryType, IsskarContext  # Garde les types


@dataclass
class EntityInfo:
    """Information sur une entité détectée"""

    name: str
    type: str  # 'project', 'person', 'technology', etc.
    frequency: int
    last_seen: datetime
    contexts: Set[str]  # Contextes où elle apparaît
    confidence: float


class DynamicContextAnalyzer:
    """
    Analyseur de contexte qui apprend dynamiquement depuis les documents Confluence
    Pas de données codées en dur - tout est inféré des documents existants
    """

    def __init__(self, vector_db=None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.vector_db = vector_db

        # Cache des entités apprises (expiration 1 heure)
        self._entity_cache = {}
        self._cache_timestamp = None
        self._cache_duration = timedelta(hours=1)

        # Patterns génériques pour la détection d'entités
        self._init_generic_patterns()

    def _init_generic_patterns(self):
        """Initialise les patterns génériques de détection"""

        # Patterns pour détecter les noms de personnes dans les documents
        self.person_patterns = [
            r"par\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+\([^)]*ISSKAR\)",  # "par Johan JUBLANC (ISSKAR)"
            r'Document:\s+"[^"]*"\s+par\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # Dans les métadonnées
            r"auteur[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",  # "auteur: Johan JUBLANC"
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:travaille|s\'occupe|responsable)",  # "Johan travaille sur"
        ]

        # Patterns pour détecter les projets
        self.project_patterns = [
            r"projet\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",  # "projet Teora"
            r"sur\s+le\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",  # "sur le Teora"
            r'Document:\s+"([A-Z][a-z]+(?:\s+[A-Z-]+)*)[^"]*"',  # Titre de document
            r"Mission[^>]*>\s*([A-Z][a-z]+(?:\s+[A-Z-]+)*)",  # Hiérarchie Confluence
        ]

        # Patterns pour détecter les technologies
        self.technology_patterns = [
            r"\b(Vue|React|Angular|Python|Java|Docker|Kubernetes|k8s)\b",
            r"\b(API|REST|GraphQL|JWT|OAuth)\b",
            r"\b(MySQL|PostgreSQL|MongoDB|Redis)\b",
            r"\b(AWS|Azure|GCP|Weaviate|Elasticsearch)\b",
        ]

        # Patterns contextuels pour améliorer la détection
        self.contextual_indicators = {
            "pronouns": ["leurs", "ses", "ces", "celui", "ceux", "leur", "sa", "son", "ce", "cette"],
            "references": ["cette mission", "ce projet", "cette équipe"],
            "relationship_words": ["responsabilités", "rôles", "contact", "équipe", "travaille"],
        }

    def _is_cache_valid(self) -> bool:
        """Vérifie si le cache est encore valide"""
        if not self._cache_timestamp:
            return False
        return datetime.now() - self._cache_timestamp < self._cache_duration

    def _learn_from_documents(self) -> Dict[str, EntityInfo]:
        """Apprend les entités depuis les documents Confluence"""

        if not self.vector_db:
            self.logger.warning("Pas de vector_db disponible, utilisation du cache existant")
            return self._entity_cache.get("entities", {})

        self.logger.info("🔍 Apprentissage des entités depuis Confluence...")

        entities = defaultdict(lambda: {"frequency": 0, "contexts": set(), "last_seen": datetime.now()})

        try:
            # Récupère un échantillon de documents récents
            documents = self._get_recent_documents(limit=200)

            for doc in documents:
                content = doc.get("content", "")
                metadata = doc.get("metadata", {})

                # Extraction des personnes
                persons = self._extract_persons(content, metadata)
                for person in persons:
                    key = f"person:{person.lower()}"
                    entities[key]["frequency"] += 1
                    entities[key]["contexts"].add("team_management")
                    entities[key]["type"] = "person"
                    entities[key]["name"] = person

                # Extraction des projets
                projects = self._extract_projects(content, metadata)
                for project in projects:
                    key = f"project:{project.lower()}"
                    entities[key]["frequency"] += 1
                    entities[key]["contexts"].add("project_management")
                    entities[key]["type"] = "project"
                    entities[key]["name"] = project

                # Extraction des technologies
                technologies = self._extract_technologies(content)
                for tech in technologies:
                    key = f"technology:{tech.lower()}"
                    entities[key]["frequency"] += 1
                    entities[key]["contexts"].add("technical")
                    entities[key]["type"] = "technology"
                    entities[key]["name"] = tech

            # Conversion en EntityInfo avec calcul de confiance
            learned_entities = {}
            for key, data in entities.items():
                confidence = min(1.0, data["frequency"] / 10.0)  # Confiance basée sur la fréquence
                if confidence > 0.1:  # Seuil minimum
                    learned_entities[key] = EntityInfo(
                        name=data["name"],
                        type=data["type"],
                        frequency=data["frequency"],
                        last_seen=data["last_seen"],
                        contexts=data["contexts"],
                        confidence=confidence,
                    )

            self.logger.info(f"✅ {len(learned_entities)} entités apprises depuis Confluence")
            return learned_entities

        except Exception as e:
            self.logger.error(f"❌ Erreur lors de l'apprentissage: {e}")
            return {}

    def _get_recent_documents(self, limit: int = 200) -> List[Dict]:
        """Récupère des documents récents depuis la base vectorielle"""

        try:
            # Utilise une requête générique pour récupérer des documents
            sample_queries = [
                "ISSKAR projet équipe",
                "document confluence",
                "réunion meeting",
                "développement technical",
            ]

            all_docs = []
            for query in sample_queries:
                if len(all_docs) >= limit:
                    break

                try:
                    results = self.vector_db.search(query, k=min(50, limit - len(all_docs)))
                    for result in results:
                        doc_data = {"content": result.document.content, "metadata": result.document.metadata or {}}
                        all_docs.append(doc_data)
                except Exception as e:
                    self.logger.debug(f"Erreur requête '{query}': {e}")
                    continue

            self.logger.info(f"📄 {len(all_docs)} documents récupérés pour apprentissage")
            return all_docs

        except Exception as e:
            self.logger.error(f"Erreur récupération documents: {e}")
            return []

    def _extract_persons(self, content: str, metadata: Dict) -> Set[str]:
        """Extrait les noms de personnes du contenu"""
        persons = set()

        # Extraction depuis les métadonnées d'auteur
        author = metadata.get("author_name", "")
        if author and len(author.split()) <= 3:  # Évite les phrases complètes
            persons.add(author.strip())

        # Extraction avec patterns
        for pattern in self.person_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                name = match.strip()
                if len(name.split()) <= 3 and name.replace(" ", "").isalpha():
                    persons.add(name)

        return persons

    def _extract_projects(self, content: str, metadata: Dict) -> Set[str]:
        """Extrait les noms de projets du contenu"""
        projects = set()

        # Extraction depuis le titre et hiérarchie
        title = metadata.get("title", "")
        hierarchy = metadata.get("hierarchy_breadcrumb", "")

        for text in [title, hierarchy]:
            for pattern in self.project_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0]
                    project = match.strip()
                    if 2 <= len(project) <= 20 and project.lower() not in ["document", "réunion", "meeting"]:
                        projects.add(project)

        # Extraction depuis le contenu
        for pattern in self.project_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                project = match.strip()
                if 2 <= len(project) <= 20:
                    projects.add(project)

        return projects

    def _extract_technologies(self, content: str) -> Set[str]:
        """Extrait les technologies du contenu"""
        technologies = set()

        for pattern in self.technology_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                technologies.add(match)

        return technologies

    def get_learned_entities(self) -> Dict[str, EntityInfo]:
        """Retourne les entités apprises (avec cache)"""

        if self._is_cache_valid():
            return self._entity_cache.get("entities", {})

        # Renouvelle le cache
        entities = self._learn_from_documents()
        self._entity_cache = {"entities": entities}
        self._cache_timestamp = datetime.now()

        return entities

    def classify_query(self, original_query: str, enriched_query: str) -> QueryType:
        """
        Classification intelligente basée sur les entités apprises dynamiquement
        """
        query_lower = original_query.lower().strip()

        # Détection salutations (patterns statiques OK)
        greeting_patterns = [
            r"^(bonjour|salut|hello|hi|hey|coucou)[\s!.]*$",
            r"^(merci|thank you|thanks)[\s!.]*$",
            r"^(au revoir|bye|goodbye|à bientôt)[\s!.]*$",
        ]

        if any(re.match(pattern, query_lower) for pattern in greeting_patterns):
            return QueryType.GREETING

        # Récupère les entités apprises
        entities = self.get_learned_entities()

        # Détection contextuelles avec références implicites
        has_pronouns = any(pronoun in query_lower for pronoun in self.contextual_indicators["pronouns"])
        has_references = any(ref in query_lower for ref in self.contextual_indicators["references"])
        context_enriched = len(enriched_query) > len(original_query) * 1.3

        if has_pronouns or has_references or context_enriched:
            return QueryType.CONTEXTUAL

        # Détection mentions explicites de projets appris
        for key, entity in entities.items():
            if entity.type == "project" and entity.name.lower() in query_lower:
                return QueryType.PROJECT_SPECIFIC

        # Détection mentions explicites de personnes apprises
        for key, entity in entities.items():
            if entity.type == "person" and any(part.lower() in query_lower for part in entity.name.split()):
                return QueryType.TEAM_INQUIRY

        # Détection technique
        for key, entity in entities.items():
            if entity.type == "technology" and entity.name.lower() in query_lower:
                return QueryType.TECHNICAL

        # Détection mots business
        business_keywords = ["budget", "coût", "prix", "commercial", "client", "deal"]
        if any(keyword in query_lower for keyword in business_keywords):
            return QueryType.BUSINESS

        # Détection termes génériques problématiques
        generic_terms = {"test", "help", "aide", "info", "ok"}
        words = query_lower.split()
        if len(words) == 1 and words[0] in generic_terms:
            return QueryType.GENERIC

        # Par défaut: projet spécifique (contexte Isskar)
        return QueryType.PROJECT_SPECIFIC

    def extract_isskar_entities(self, query: str, enriched_query: str) -> IsskarContext:
        """
        Extraction d'entités basée on l'apprentissage dynamique
        """
        entities = self.get_learned_entities()

        projects = set()
        team_members = set()
        technical_domains = set()

        # Recherche dans la query enrichie
        text_to_analyze = enriched_query.lower()

        for key, entity in entities.items():
            entity_name_lower = entity.name.lower()

            if entity.type == "project" and entity_name_lower in text_to_analyze:
                projects.add(entity_name_lower)
            elif entity.type == "person":
                # Recherche par prénom ou nom complet
                name_parts = entity.name.lower().split()
                if any(part in text_to_analyze for part in name_parts):
                    team_members.add(name_parts[0])  # Utilise le prénom
            elif entity.type == "technology" and entity_name_lower in text_to_analyze:
                technical_domains.add(entity_name_lower)

        return IsskarContext(
            projects=projects,
            team_members=team_members,
            mission_types=set(),  # À implémenter si nécessaire
            technical_domains=technical_domains,
            business_areas=set(),  # À implémenter si nécessaire
            hierarchical_context=[],  # À implémenter si nécessaire
        )

    def get_filtering_strategy(self, query_type: QueryType, context: IsskarContext) -> Dict:
        """
        Stratégie de filtrage adaptative (inchangée)
        """
        strategies = {
            QueryType.CONTEXTUAL: {
                "score_threshold": 0.32,
                "relevance_threshold": 0.05,
                "trust_vector_similarity": True,
                "apply_synonym_matching": True,
                "context_boost": 0.15,
            },
            QueryType.PROJECT_SPECIFIC: {
                "score_threshold": 0.38,
                "relevance_threshold": 0.08,
                "trust_vector_similarity": True,
                "apply_synonym_matching": True,
                "context_boost": 0.10,
            },
            QueryType.TEAM_INQUIRY: {
                "score_threshold": 0.35,
                "relevance_threshold": 0.10,
                "trust_vector_similarity": True,
                "apply_synonym_matching": False,
                "context_boost": 0.08,
            },
            QueryType.TECHNICAL: {
                "score_threshold": 0.42,
                "relevance_threshold": 0.15,
                "trust_vector_similarity": False,
                "apply_synonym_matching": True,
                "context_boost": 0.05,
            },
            QueryType.BUSINESS: {
                "score_threshold": 0.40,
                "relevance_threshold": 0.12,
                "trust_vector_similarity": False,
                "apply_synonym_matching": True,
                "context_boost": 0.05,
            },
            QueryType.GENERIC: {
                "score_threshold": 0.70,
                "relevance_threshold": 0.40,
                "trust_vector_similarity": False,
                "apply_synonym_matching": False,
                "context_boost": 0.0,
            },
            QueryType.GREETING: {
                "score_threshold": 1.0,
                "relevance_threshold": 1.0,
                "trust_vector_similarity": False,
                "apply_synonym_matching": False,
                "context_boost": 0.0,
            },
        }

        return strategies.get(query_type, strategies[QueryType.PROJECT_SPECIFIC])

    def get_learned_stats(self) -> Dict:
        """Retourne des statistiques sur l'apprentissage"""
        entities = self.get_learned_entities()

        stats = {
            "total_entities": len(entities),
            "by_type": defaultdict(int),
            "high_confidence": 0,
            "last_learning": self._cache_timestamp.isoformat() if self._cache_timestamp else None,
        }

        for entity in entities.values():
            stats["by_type"][entity.type] += 1
            if entity.confidence > 0.7:
                stats["high_confidence"] += 1

        return dict(stats)
