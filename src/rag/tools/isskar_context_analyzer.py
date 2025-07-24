#!/usr/bin/env python3
"""
Analyseur de contexte spécialisé pour l'environnement Confluence Isskar
Optimisé pour la structure projets/missions/équipes d'Isskar
"""

import re
from enum import Enum
from typing import List, Dict, Set
from dataclasses import dataclass


class QueryType(Enum):
    """Types de requêtes spécifiques à l'environnement Isskar"""

    CONTEXTUAL = "contextual"  # Questions avec références implicites
    PROJECT_SPECIFIC = "project"  # Questions sur un projet spécifique
    TEAM_INQUIRY = "team"  # Questions sur l'équipe/personnes
    TECHNICAL = "technical"  # Questions techniques (Fondations/Veille)
    BUSINESS = "business"  # Questions business/commercial
    GENERIC = "generic"  # Termes trop généraux
    GREETING = "greeting"  # Interactions sociales
    MISSION_STATUS = "mission_status"  # Statut de missions/projets


@dataclass
class IsskarContext:
    """Contexte enrichi spécifique à Isskar"""

    projects: Set[str]
    team_members: Set[str]
    mission_types: Set[str]
    technical_domains: Set[str]
    business_areas: Set[str]
    hierarchical_context: List[str]


class IsskarContextAnalyzer:
    """
    Analyseur de contexte expert pour l'environnement Isskar
    Basé sur l'analyse de 524 pages Confluence
    """

    def __init__(self):
        self._initialize_isskar_knowledge()

    def _initialize_isskar_knowledge(self):
        """Initialise la base de connaissances Isskar tirée de l'analyse Confluence"""

        # Projets clients identifiés (basé sur l'analyse des 524 pages)
        self.known_projects = {
            # Projets principaux
            "teora": {"variants": ["téora", "TEORA"], "domain": "client_mission"},
            "cibtp": {"variants": ["CIBTP", "cibtp-ao", "CIBTP - AO"], "domain": "client_mission"},
            "isschat": {"variants": ["ISSCHAT", "iss-chat", "IssChat"], "domain": "internal_product"},
            "cedrus": {"variants": ["Cedrus & Partners", "GenAut"], "domain": "client_mission"},
            "fives": {"variants": ["Fives"], "domain": "client_mission"},
            "oddo": {"variants": ["Oddo"], "domain": "client_mission"},
            "obea": {"variants": ["Obea T&E"], "domain": "client_mission"},
            "nowa": {"variants": ["[Nowa]"], "domain": "client_mission"},
            "iagen": {"variants": ["IAGEN"], "domain": "client_mission"},
            "cellenza": {"variants": ["Cellenza"], "domain": "client_mission"},
        }

        # Équipe Isskar (basé sur author_name analysis)
        self.team_members = {
            "johan": {"full_name": "JUBLANC Johan", "role": "lead", "expertise": ["management", "client_relations"]},
            "vincent": {"full_name": "Vincent Fraillon", "role": "senior", "expertise": ["technical", "architecture"]},
            "damien": {"full_name": "Damien Saby", "role": "business", "expertise": ["business_dev", "strategy"]},
            "henri": {"full_name": "Henri AVERLAND", "role": "technical", "expertise": ["development"]},
            "gregoire": {"full_name": "Grégoire SALHA", "role": "technical", "expertise": ["development"]},
            "nicolas": {
                "full_name": "Nicolas LAMBROPOULOS",
                "role": "technical",
                "expertise": ["development", "veille"],
            },
            "francois": {"full_name": "François DUPONT", "role": "technical", "expertise": ["development"]},
            "emin": {"full_name": "Emin Calyaka", "role": "technical", "expertise": ["development"]},
            "adame": {"full_name": "Adame Ben Friha", "role": "technical", "expertise": ["development"]},
            "amaury": {"full_name": "Amaury Dreux", "role": "technical", "expertise": ["development"]},
            "cassandre": {"full_name": "Cassandre Minoza", "role": "technical", "expertise": ["development"]},
        }

        # Domaines techniques (basé sur hierarchy_breadcrumb analysis)
        self.technical_domains = {
            "data": ["DVC", "Snowflake", "data_version_control", "metrics", "kpi"],
            "infrastructure": ["minikube", "kubernetes", "deployment"],
            "ai_ml": ["GenAut", "OCR", "machine_learning", "artificial_intelligence"],
            "development": ["bugs", "workflow", "github", "testing"],
            "security": ["RSSI", "souveraineté", "sécurité"],
        }

        # Types de missions (basé sur les patterns hiérarchiques)
        self.mission_categories = {
            "ao": ["appel_offre", "pricing", "kickoff", "poc"],
            "consulting": ["atelier", "workshop", "conseil"],
            "development": ["recette", "développement", "livraison"],
            "analysis": ["analyse", "étude", "faisabilité"],
        }

        # Indicateurs contextuels français spécifiques
        self.contextual_indicators = {
            "pronouns": ["leurs", "ses", "ces", "celui", "ceux", "leur", "sa", "son", "ce", "cette", "ils", "elles"],
            "references": ["cette mission", "ce projet", "cette équipe", "ces personnes", "celui-ci", "celle-ci"],
            "implicit_questions": ["comment", "pourquoi", "où", "quand", "combien"],
            "relationship_words": ["responsabilités", "rôles", "contact", "contacter", "équipe", "travaille", "occupe"],
        }

        # Termes génériques à filtrer strictement
        self.problematic_generic_terms = {
            "test",
            "tests",
            "testing",
            "help",
            "aide",
            "info",
            "ok",
            "okay",
            "bien",
            "good",
            "bad",
            "exemple",
            "sample",
            "demo",
            "merci",
            "thanks",
        }

        # Patterns de salutations françaises
        self.greeting_patterns = [
            r"^(bonjour|salut|hello|hi|hey|coucou)[\s!.]*$",
            r"^(merci|thank you|thanks)[\s!.]*$",
            r"^(au revoir|bye|goodbye|à bientôt)[\s!.]*$",
            r"^(comment ça va|ça va|how are you).*$",
            r"^(bonne journée|good day)[\s!.]*$",
        ]

    def classify_query(self, original_query: str, enriched_query: str) -> QueryType:
        """
        Classification intelligente basée sur la structure Isskar
        """
        query_lower = original_query.lower().strip()
        words = query_lower.split()

        # 1. Détection salutations (priorité maximale)
        if self._is_greeting(query_lower):
            return QueryType.GREETING

        # 2. Détection requêtes contextuelles (cœur de l'amélioration)
        if self._is_contextual_isskar(query_lower, enriched_query):
            return QueryType.CONTEXTUAL

        # 3. Détection requêtes spécifiques projets
        if self._mentions_project(query_lower):
            return QueryType.PROJECT_SPECIFIC

        # 4. Détection requêtes équipe
        if self._is_team_inquiry(query_lower):
            return QueryType.TEAM_INQUIRY

        # 5. Détection requêtes techniques
        if self._is_technical_isskar(query_lower):
            return QueryType.TECHNICAL

        # 6. Détection requêtes business
        if self._is_business_related(query_lower):
            return QueryType.BUSINESS

        # 7. Détection statut missions
        if self._is_mission_status(query_lower):
            return QueryType.MISSION_STATUS

        # 8. Détection termes génériques problématiques
        if self._is_generic_problematic(words):
            return QueryType.GENERIC

        # Fallback: standard
        return QueryType.PROJECT_SPECIFIC  # Par défaut projet dans contexte Isskar

    def _is_contextual_isskar(self, query: str, enriched_query: str) -> bool:
        """Détection contextuelle optimisée pour Isskar"""

        # Pronoms possessifs/démonstratifs
        has_pronouns = any(pronoun in query for pronoun in self.contextual_indicators["pronouns"])

        # Références implicites
        has_references = any(ref in query for ref in self.contextual_indicators["references"])

        # Enrichissement contextuel appliqué
        context_enriched = len(enriched_query) > len(query) * 1.3

        # Questions sur relations/responsabilités sans entité explicite
        relationship_query = any(word in query for word in self.contextual_indicators["relationship_words"])
        no_explicit_entity = not any(project in query for project in self.known_projects.keys())
        implicit_relationship = relationship_query and no_explicit_entity

        # Questions courtes avec fort enrichissement
        short_with_heavy_context = len(query.split()) <= 4 and context_enriched

        return has_pronouns or has_references or implicit_relationship or short_with_heavy_context

    def _mentions_project(self, query: str) -> bool:
        """Détecte la mention explicite d'un projet Isskar"""
        for project, info in self.known_projects.items():
            if project in query:
                return True
            for variant in info["variants"]:
                if variant.lower() in query:
                    return True
        return False

    def _is_team_inquiry(self, query: str) -> bool:
        """Détecte les questions sur l'équipe Isskar"""
        team_indicators = ["qui", "équipe", "team", "développeur", "developer", "travaille", "responsable"]
        explicit_names = any(name in query for name in self.team_members.keys())
        team_question = any(indicator in query for indicator in team_indicators)
        return explicit_names or team_question

    def _is_technical_isskar(self, query: str) -> bool:
        """Détecte les questions techniques spécifiques Isskar"""
        for domain, terms in self.technical_domains.items():
            if any(term in query for term in terms):
                return True

        technical_indicators = ["comment", "configuration", "setup", "install", "deploy", "bug", "erreur"]
        return any(indicator in query for indicator in technical_indicators)

    def _is_business_related(self, query: str) -> bool:
        """Détecte les questions business/commerciales"""
        business_indicators = [
            "budget",
            "coût",
            "prix",
            "pricing",
            "commercial",
            "vente",
            "client",
            "deal",
            "factory",
            "lemlist",
            "campagne",
            "décideur",
            "business",
        ]
        return any(indicator in query for indicator in business_indicators)

    def _is_mission_status(self, query: str) -> bool:
        """Détecte les questions sur le statut des missions"""
        status_indicators = [
            "avancement",
            "statut",
            "état",
            "progress",
            "où en est",
            "terminé",
            "fini",
            "livré",
            "recette",
            "validation",
        ]
        return any(indicator in query for indicator in status_indicators)

    def _is_generic_problematic(self, words: List[str]) -> bool:
        """Détecte les termes génériques problématiques"""
        if len(words) == 1 and words[0] in self.problematic_generic_terms:
            return True

        generic_ratio = sum(1 for word in words if word in self.problematic_generic_terms) / len(words)
        return generic_ratio > 0.6

    def _is_greeting(self, query: str) -> bool:
        """Détecte les salutations"""
        return any(re.match(pattern, query) for pattern in self.greeting_patterns)

    def extract_isskar_entities(self, query: str, enriched_query: str) -> IsskarContext:
        """
        Extraction d'entités spécifiques à l'environnement Isskar
        """
        projects = set()
        team_members = set()
        mission_types = set()
        technical_domains = set()
        business_areas = set()

        # Extraction projets avec variants
        for project, info in self.known_projects.items():
            if project in enriched_query.lower():
                projects.add(project)
            for variant in info["variants"]:
                if variant.lower() in enriched_query.lower():
                    projects.add(project)

        # Extraction équipe
        for member_key, info in self.team_members.items():
            if member_key in enriched_query.lower() or info["full_name"].lower() in enriched_query.lower():
                team_members.add(member_key)

        # Extraction domaines techniques
        for domain, terms in self.technical_domains.items():
            if any(term in enriched_query.lower() for term in terms):
                technical_domains.add(domain)

        # Extraction types de missions
        for mission_type, indicators in self.mission_categories.items():
            if any(indicator in enriched_query.lower() for indicator in indicators):
                mission_types.add(mission_type)

        # Contexte hiérarchique (si détectable dans l'enrichissement)
        hierarchical_context = self._extract_hierarchy_context(enriched_query)

        return IsskarContext(
            projects=projects,
            team_members=team_members,
            mission_types=mission_types,
            technical_domains=technical_domains,
            business_areas=business_areas,
            hierarchical_context=hierarchical_context,
        )

    def _extract_hierarchy_context(self, enriched_query: str) -> List[str]:
        """Extrait le contexte hiérarchique des métadonnées enrichies"""
        hierarchy_patterns = [
            r"missions?[:\s]*([^;]+)",
            r"fondations?[:\s]*([^;]+)",
            r"business[:\s]*([^;]+)",
            r"veille[:\s]*([^;]+)",
        ]

        context = []
        for pattern in hierarchy_patterns:
            matches = re.findall(pattern, enriched_query.lower())
            context.extend(matches)

        return context

    def get_filtering_strategy(self, query_type: QueryType, context: IsskarContext) -> Dict:
        """
        Retourne la stratégie de filtrage optimale selon le type de requête et le contexte Isskar
        """
        strategies = {
            QueryType.CONTEXTUAL: {
                "score_threshold": 0.32,  # Très permissif pour contextuelles
                "relevance_threshold": 0.05,  # Minimal
                "trust_vector_similarity": True,
                "apply_synonym_matching": True,
                "context_boost": 0.15,
            },
            QueryType.PROJECT_SPECIFIC: {
                "score_threshold": 0.38,  # Permissif pour projets
                "relevance_threshold": 0.08,
                "trust_vector_similarity": True,
                "apply_synonym_matching": True,
                "context_boost": 0.10,
            },
            QueryType.TEAM_INQUIRY: {
                "score_threshold": 0.35,  # Permissif pour équipe
                "relevance_threshold": 0.10,
                "trust_vector_similarity": True,
                "apply_synonym_matching": False,
                "context_boost": 0.08,
            },
            QueryType.TECHNICAL: {
                "score_threshold": 0.42,  # Standard pour technique
                "relevance_threshold": 0.15,
                "trust_vector_similarity": False,
                "apply_synonym_matching": True,
                "context_boost": 0.05,
            },
            QueryType.BUSINESS: {
                "score_threshold": 0.40,  # Standard pour business
                "relevance_threshold": 0.12,
                "trust_vector_similarity": False,
                "apply_synonym_matching": True,
                "context_boost": 0.05,
            },
            QueryType.MISSION_STATUS: {
                "score_threshold": 0.36,  # Permissif pour statuts
                "relevance_threshold": 0.10,
                "trust_vector_similarity": True,
                "apply_synonym_matching": True,
                "context_boost": 0.12,
            },
            QueryType.GENERIC: {
                "score_threshold": 0.70,  # Très strict pour génériques
                "relevance_threshold": 0.40,
                "trust_vector_similarity": False,
                "apply_synonym_matching": False,
                "context_boost": 0.0,
            },
            QueryType.GREETING: {
                "score_threshold": 1.0,  # Bloque tout
                "relevance_threshold": 1.0,
                "trust_vector_similarity": False,
                "apply_synonym_matching": False,
                "context_boost": 0.0,
            },
        }

        return strategies.get(query_type, strategies[QueryType.PROJECT_SPECIFIC])
