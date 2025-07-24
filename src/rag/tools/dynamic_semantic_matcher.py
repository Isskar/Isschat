#!/usr/bin/env python3
"""
Matcher sémantique dynamique qui apprend les synonymes depuis les documents
Pas de données codées en dur - apprentissage automatique
"""

from typing import List, Dict, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
import re
import logging


@dataclass
class SemanticPattern:
    """Pattern sémantique appris automatiquement"""

    term: str
    synonyms: Set[str]
    confidence: float
    contexts: Set[str]
    frequency: int


class DynamicSemanticMatcher:
    """
    Matcher sémantique qui apprend automatiquement les patterns
    depuis les documents Confluence existants
    """

    def __init__(self, vector_db=None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.vector_db = vector_db

        # Cache des patterns appris
        self._patterns_cache = {}
        self._cache_timestamp = None
        self._cache_duration = timedelta(hours=2)

        # Patterns de base (génériques, pas spécifiques à Isskar)
        self._init_base_patterns()

    def _init_base_patterns(self):
        """Initialise les patterns de base génériques"""

        # Patterns pour apprendre les synonymes métiers
        self.synonym_detection_patterns = [
            # "X aussi appelé Y"
            r"(\w+)\s+(?:aussi\s+appelé|également\s+appelé|ou)\s+(\w+)",
            # "X (Y)" - acronymes
            r"([A-Za-z\s]+)\s*\(([A-Z]{2,})\)",
            # "X, c'est-à-dire Y"
            r"(\w+),?\s*(?:c\'est-à-dire|soit|i\.e\.)\s+(\w+)",
            # Listes avec tirets
            r"-\s*(\w+)\s*:\s*([^-\n]+)",
        ]

        # Patterns pour détecter les relations hiérarchiques
        self.hierarchy_patterns = [
            r"équipe\s+([^,\n]+),?\s*(?:composée|constituée)\s+de\s+([^.\n]+)",
            r"projet\s+([^,\n]+)\s+(?:dirigé|managé|supervisé)\s+par\s+([^.\n]+)",
            r"([^,\n]+)\s+(?:responsable|en\s+charge)\s+(?:de|du)\s+([^.\n]+)",
        ]

        # Contextes métiers génériques
        self.business_contexts = {
            "management": ["responsable", "manager", "chef", "directeur", "lead"],
            "technical": ["développeur", "dev", "tech", "ingénieur", "architect"],
            "project": ["projet", "mission", "client", "livrable", "deadline"],
            "team": ["équipe", "team", "groupe", "collaborateur", "collègue"],
        }

    def _is_cache_valid(self) -> bool:
        """Vérifie si le cache est valide"""
        if not self._cache_timestamp:
            return False
        return datetime.now() - self._cache_timestamp < self._cache_duration

    def _learn_semantic_patterns(self) -> Dict[str, SemanticPattern]:
        """Apprend les patterns sémantiques depuis les documents"""

        if not self.vector_db:
            self.logger.warning("Pas de vector_db - utilisation des patterns de base")
            return self._get_base_patterns()

        self.logger.info("🧠 Apprentissage des patterns sémantiques...")

        patterns = defaultdict(lambda: {"synonyms": set(), "contexts": set(), "frequency": 0, "confidence": 0.0})

        try:
            # Récupère des documents pour l'apprentissage
            documents = self._sample_documents_for_learning()

            for doc in documents:
                content = doc.get("content", "")

                # Apprend les synonymes
                self._extract_synonyms(content, patterns)

                # Apprend les relations contextuelles
                self._extract_contextual_relations(content, patterns)

                # Apprend les co-occurrences
                self._extract_cooccurrences(content, patterns)

            # Convertit en SemanticPattern avec calcul de confiance
            learned_patterns = {}
            for term, data in patterns.items():
                if data["frequency"] >= 2:  # Seuil minimum
                    confidence = min(1.0, data["frequency"] / 10.0)
                    learned_patterns[term] = SemanticPattern(
                        term=term,
                        synonyms=data["synonyms"],
                        confidence=confidence,
                        contexts=data["contexts"],
                        frequency=data["frequency"],
                    )

            self.logger.info(f"✅ {len(learned_patterns)} patterns sémantiques appris")
            return learned_patterns

        except Exception as e:
            self.logger.error(f"❌ Erreur apprentissage patterns: {e}")
            return self._get_base_patterns()

    def _sample_documents_for_learning(self) -> List[Dict]:
        """Échantillonne des documents pour l'apprentissage"""

        learning_queries = [
            "responsabilités équipe projet",
            "contact collaboration travail",
            "développement technique client",
            "mission objectif livrable",
        ]

        all_docs = []
        for query in learning_queries:
            try:
                results = self.vector_db.search(query, k=30)
                for result in results:
                    all_docs.append({"content": result.document.content, "metadata": result.document.metadata or {}})
            except Exception as e:
                self.logger.debug(f"Erreur échantillonnage '{query}': {e}")

        return all_docs[:100]  # Limite pour performance

    def _extract_synonyms(self, content: str, patterns: Dict):
        """Extrait les synonymes explicites du contenu"""

        for pattern in self.synonym_detection_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple) and len(match) == 2:
                    term1, term2 = match
                    term1, term2 = term1.strip().lower(), term2.strip().lower()

                    if 2 <= len(term1) <= 20 and 2 <= len(term2) <= 20:
                        # Relation bidirectionnelle
                        patterns[term1]["synonyms"].add(term2)
                        patterns[term2]["synonyms"].add(term1)
                        patterns[term1]["frequency"] += 1
                        patterns[term2]["frequency"] += 1
                        patterns[term1]["contexts"].add("synonym_detection")
                        patterns[term2]["contexts"].add("synonym_detection")

    def _extract_contextual_relations(self, content: str, patterns: Dict):
        """Extrait les relations contextuelles"""

        # Recherche de patterns métiers récurrents
        words = re.findall(r"\b\w+\b", content.lower())

        for i, word in enumerate(words):
            if word in ["responsabilités", "rôles", "tâches", "fonctions"]:
                # Cherche des mots proches qui pourraient être synonymes
                window = words[max(0, i - 3) : i + 4]
                for related_word in window:
                    if related_word != word and len(related_word) > 3:
                        patterns[word]["synonyms"].add(related_word)
                        patterns[word]["frequency"] += 1
                        patterns[word]["contexts"].add("contextual_relation")

    def _extract_cooccurrences(self, content: str, patterns: Dict):
        """Apprend des synonymes par co-occurrence"""

        # Cherche des mots qui apparaissent fréquemment ensemble
        sentences = re.split(r"[.!?]", content)

        for sentence in sentences:
            words = set(re.findall(r"\b\w{4,}\b", sentence.lower()))  # Mots de 4+ lettres

            # Si on trouve des mots métiers ensemble, ils sont potentiellement liés
            business_words = []
            for word in words:
                if any(biz_word in word for context in self.business_contexts.values() for biz_word in context):
                    business_words.append(word)

            # Crée des liens entre mots métiers co-occurrents
            for i, word1 in enumerate(business_words):
                for word2 in business_words[i + 1 :]:
                    patterns[word1]["synonyms"].add(word2)
                    patterns[word2]["synonyms"].add(word1)
                    patterns[word1]["frequency"] += 1
                    patterns[word2]["frequency"] += 1
                    patterns[word1]["contexts"].add("cooccurrence")
                    patterns[word2]["contexts"].add("cooccurrence")

    def _get_base_patterns(self) -> Dict[str, SemanticPattern]:
        """Retourne des patterns de base si l'apprentissage échoue"""

        base_patterns = {
            "responsabilités": SemanticPattern(
                term="responsabilités",
                synonyms={"rôles", "tâches", "fonctions", "missions"},
                confidence=0.9,
                contexts={"base_pattern"},
                frequency=10,
            ),
            "équipe": SemanticPattern(
                term="équipe",
                synonyms={"team", "groupe", "collaborateurs"},
                confidence=0.9,
                contexts={"base_pattern"},
                frequency=10,
            ),
            "projet": SemanticPattern(
                term="projet",
                synonyms={"mission", "client", "application"},
                confidence=0.8,
                contexts={"base_pattern"},
                frequency=8,
            ),
        }

        return base_patterns

    def get_learned_patterns(self) -> Dict[str, SemanticPattern]:
        """Retourne les patterns appris (avec cache)"""

        if self._is_cache_valid():
            return self._patterns_cache.get("patterns", {})

        # Renouvelle le cache
        patterns = self._learn_semantic_patterns()
        self._patterns_cache = {"patterns": patterns}
        self._cache_timestamp = datetime.now()

        return patterns

    def calculate_semantic_relevance(
        self, query_keywords: List[str], document_content: str, document_title: str = "", apply_synonyms: bool = True
    ) -> float:
        """
        Calcule la pertinence sémantique avec patterns appris dynamiquement
        """
        if not query_keywords:
            return 0.0

        content_lower = document_content.lower()
        title_lower = document_title.lower()

        patterns = self.get_learned_patterns() if apply_synonyms else {}

        total_score = 0.0
        max_possible = len(query_keywords) * 2

        for keyword in query_keywords:
            keyword_lower = keyword.lower()

            # Score de base pour correspondances exactes
            title_exact = 2.0 if keyword_lower in title_lower else 0.0
            content_exact = 1.0 if keyword_lower in content_lower else 0.0

            # Score pour synonymes appris
            synonym_score = 0.0
            if keyword_lower in patterns:
                pattern = patterns[keyword_lower]
                for synonym in pattern.synonyms:
                    if synonym in content_lower:
                        synonym_score += pattern.confidence * 0.8  # Poids réduit pour synonymes
                    if synonym in title_lower:
                        synonym_score += pattern.confidence * 1.6

            # Recherche inverse (keyword pourrait être un synonyme)
            for term, pattern in patterns.items():
                if keyword_lower in pattern.synonyms:
                    if term in content_lower:
                        synonym_score += pattern.confidence * 0.8
                    if term in title_lower:
                        synonym_score += pattern.confidence * 1.6

            # Score final pour ce keyword
            keyword_score = max(title_exact + content_exact, synonym_score)
            total_score += keyword_score

        # Normalisation
        base_relevance = total_score / max_possible if max_possible > 0 else 0.0

        # Bonus pour phrases exactes
        phrase_bonus = self._calculate_phrase_bonus(query_keywords, content_lower, title_lower)

        return min(1.0, base_relevance + phrase_bonus)

    def _calculate_phrase_bonus(self, keywords: List[str], content: str, title: str) -> float:
        """Calcule le bonus pour phrases exactes"""

        bonus = 0.0
        query_phrase = " ".join(keywords)

        if query_phrase in content:
            bonus += 0.15
        if query_phrase in title:
            bonus += 0.25

        return bonus

    def get_enhanced_keywords(self, original_keywords: List[str]) -> List[str]:
        """Enrichit les mots-clés avec des synonymes appris"""

        patterns = self.get_learned_patterns()
        enhanced = set(original_keywords)

        for keyword in original_keywords:
            keyword_lower = keyword.lower()

            # Ajoute les synonymes appris
            if keyword_lower in patterns:
                pattern = patterns[keyword_lower]
                # Prend les meilleurs synonymes (confiance > 0.5)
                good_synonyms = [syn for syn in pattern.synonyms if pattern.confidence > 0.5]
                enhanced.update(good_synonyms[:3])  # Limite à 3 pour éviter le bruit

            # Recherche inverse
            for term, pattern in patterns.items():
                if keyword_lower in pattern.synonyms and pattern.confidence > 0.5:
                    enhanced.add(term)

        return list(enhanced)

    def get_learning_stats(self) -> Dict:
        """Retourne des statistiques sur l'apprentissage"""

        patterns = self.get_learned_patterns()

        stats = {
            "total_patterns": len(patterns),
            "high_confidence": sum(1 for p in patterns.values() if p.confidence > 0.7),
            "avg_synonyms": sum(len(p.synonyms) for p in patterns.values()) / len(patterns) if patterns else 0,
            "last_learning": self._cache_timestamp.isoformat() if self._cache_timestamp else None,
        }

        return stats
