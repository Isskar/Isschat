#!/usr/bin/env python3
"""
Système de filtrage intelligent principal pour Isskar
Orchestre l'analyse contextuelle et la correspondance sémantique
"""

import logging
import re
from typing import List, Dict
from dataclasses import dataclass

from .dynamic_context_analyzer import DynamicContextAnalyzer, QueryType, IsskarContext
from .dynamic_semantic_matcher import DynamicSemanticMatcher
from ...core.documents import RetrievalDocument


@dataclass
class FilteringResult:
    """Résultat du filtrage avec métadonnées de debugging"""

    documents: List[RetrievalDocument]
    query_type: QueryType
    context: IsskarContext
    filtering_stats: Dict
    reasoning: List[str]


class IsskarSmartFilter:
    """
    Système de filtrage intelligent principal optimisé pour l'environnement Isskar

    Combine:
    - Analyse contextuelle sophistiquée
    - Correspondance sémantique flexible
    - Stratégies de filtrage adaptatives
    - Logging détaillé pour debugging
    """

    def __init__(self, config, vector_db=None):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.vector_db = vector_db

        # Composants experts dynamiques (apprennent depuis les documents)
        self.context_analyzer = DynamicContextAnalyzer(vector_db=vector_db)
        self.semantic_matcher = DynamicSemanticMatcher(vector_db=vector_db)

        # Stats pour monitoring
        self.filtering_stats = {"total_queries": 0, "by_query_type": {}, "documents_filtered": 0, "documents_passed": 0}

    def filter_documents(
        self, original_query: str, enriched_query: str, documents: List[RetrievalDocument]
    ) -> FilteringResult:
        """
        Point d'entrée principal du filtrage intelligent
        """
        self.filtering_stats["total_queries"] += 1

        if not documents:
            return FilteringResult(
                documents=[],
                query_type=QueryType.GENERIC,
                context=IsskarContext(set(), set(), set(), set(), set(), []),
                filtering_stats={"input_docs": 0, "output_docs": 0},
                reasoning=["Aucun document en entrée"],
            )

        self.logger.info("🧠 FILTRAGE INTELLIGENT ISSKAR")
        self.logger.info(f"📝 Query originale: '{original_query}'")
        self.logger.info(f"📝 Query enrichie: '{enriched_query[:100]}{'...' if len(enriched_query) > 100 else ''}'")
        self.logger.info(f"📄 Documents à filtrer: {len(documents)}")

        # 1. Analyse contextuelle
        query_type = self.context_analyzer.classify_query(original_query, enriched_query)
        context = self.context_analyzer.extract_isskar_entities(original_query, enriched_query)

        self.logger.info(f"🎯 Type de requête détecté: {query_type.value}")
        self.logger.info(f"🧩 Contexte extrait: projets={context.projects}, équipe={context.team_members}")

        # 2. Stratégie de filtrage adaptative
        strategy = self.context_analyzer.get_filtering_strategy(query_type, context)
        self.logger.info(
            f"⚙️ Stratégie appliquée: score_threshold={strategy['score_threshold']:.3f}, "
            f"relevance_threshold={strategy['relevance_threshold']:.3f}"
        )

        # 3. Application du filtrage selon la stratégie
        filtered_docs, reasoning = self._apply_filtering_strategy(
            original_query, enriched_query, documents, query_type, strategy, context
        )

        # 4. Stats et logging
        self._update_stats(query_type, len(documents), len(filtered_docs))

        self.logger.info(f"📊 Résultat: {len(filtered_docs)}/{len(documents)} documents retenus")
        for reason in reasoning:
            self.logger.info(f"💡 {reason}")

        return FilteringResult(
            documents=filtered_docs,
            query_type=query_type,
            context=context,
            filtering_stats={
                "input_docs": len(documents),
                "output_docs": len(filtered_docs),
                "strategy": strategy,
                "query_type": query_type.value,
            },
            reasoning=reasoning,
        )

    def _apply_filtering_strategy(
        self,
        original_query: str,
        enriched_query: str,
        documents: List[RetrievalDocument],
        query_type: QueryType,
        strategy: Dict,
        context: IsskarContext,
    ) -> tuple[List[RetrievalDocument], List[str]]:
        """Applique la stratégie de filtrage selon le type de requête"""

        reasoning = []

        # Cas spécial: salutations -> aucun document
        if query_type == QueryType.GREETING:
            reasoning.append("Salutation détectée - aucun document retourné")
            return [], reasoning

        # Extraction des mots-clés selon la stratégie
        if strategy["trust_vector_similarity"] and query_type == QueryType.CONTEXTUAL:
            # Pour les requêtes contextuelles, on utilise la query enrichie
            keywords = self._extract_keywords(enriched_query)
            reasoning.append("Requête contextuelle: utilisation de la query enrichie pour l'extraction des mots-clés")
        else:
            # Pour les autres types, query originale
            keywords = self._extract_keywords(original_query)
            reasoning.append("Utilisation de la query originale pour l'extraction des mots-clés")

        # Enrichissement sémantique des mots-clés si activé
        if strategy["apply_synonym_matching"]:
            enhanced_keywords = self.semantic_matcher.get_enhanced_keywords(keywords)
            reasoning.append(f"Enrichissement sémantique: {len(keywords)} → {len(enhanced_keywords)} mots-clés")
            keywords = enhanced_keywords

        filtered_docs = []

        for i, doc in enumerate(documents):
            doc_reasoning = []

            # Test 1: Seuil de score vectoriel
            if doc.score < strategy["score_threshold"]:
                doc_reasoning.append(
                    f"Score vectoriel insuffisant ({doc.score:.3f} < {strategy['score_threshold']:.3f})"
                )
                self.logger.debug(f"❌ Doc {i + 1}: {doc_reasoning[-1]}")
                continue

            doc_reasoning.append(f"Score vectoriel OK ({doc.score:.3f})")

            # Test 2: Pertinence sémantique (sauf si confiance totale aux vecteurs)
            if not strategy["trust_vector_similarity"]:
                semantic_relevance = self.semantic_matcher.calculate_semantic_relevance(
                    keywords,
                    doc.content,
                    doc.metadata.get("title", ""),
                    apply_synonyms=strategy["apply_synonym_matching"],
                )

                if semantic_relevance < strategy["relevance_threshold"]:
                    doc_reasoning.append(
                        f"Pertinence sémantique insuffisante ({semantic_relevance:.3f} < {strategy['relevance_threshold']:.3f})"
                    )
                    self.logger.debug(f"❌ Doc {i + 1}: {' | '.join(doc_reasoning)}")
                    continue

                doc_reasoning.append(f"Pertinence sémantique OK ({semantic_relevance:.3f})")
            else:
                doc_reasoning.append("Confiance totale aux scores vectoriels")

            # Test 3: Boost contextuel si applicable
            final_score = doc.score
            if strategy["context_boost"] > 0:
                context_boost = self._calculate_context_boost(doc, context, strategy["context_boost"])
                final_score += context_boost
                if context_boost > 0:
                    doc_reasoning.append(f"Boost contextuel appliqué (+{context_boost:.3f})")

            # Document accepté
            doc_reasoning.append("✅ ACCEPTÉ")
            self.logger.debug(f"✅ Doc {i + 1}: {' | '.join(doc_reasoning)}")

            # Mise à jour du score si boost appliqué
            if final_score != doc.score:
                # Créer une copie du document avec le nouveau score
                boosted_doc = RetrievalDocument(
                    content=doc.content,
                    metadata=doc.metadata.copy(),
                    score=min(1.0, final_score),  # Cap à 1.0
                )
                boosted_doc.metadata["original_score"] = doc.score
                boosted_doc.metadata["context_boost"] = final_score - doc.score
                filtered_docs.append(boosted_doc)
            else:
                filtered_docs.append(doc)

        # Tri final par score (pour tenir compte des boosts)
        filtered_docs.sort(key=lambda x: x.score, reverse=True)

        # Résumé du raisonnement
        if filtered_docs:
            reasoning.append(f"Filtrage {query_type.value}: {len(filtered_docs)}/{len(documents)} documents retenus")
            if strategy["context_boost"] > 0:
                boosted_count = sum(1 for doc in filtered_docs if "context_boost" in doc.metadata)
                if boosted_count > 0:
                    reasoning.append(f"Boost contextuel appliqué à {boosted_count} documents")
        else:
            reasoning.append(f"Aucun document ne respecte les critères de filtrage {query_type.value}")

        return filtered_docs, reasoning

    def _extract_keywords(self, query: str) -> List[str]:
        """Extrait les mots-clés pertinents d'une requête"""

        # Mots vides français et anglais étendus
        stop_words = {
            # Français
            "le",
            "la",
            "les",
            "un",
            "une",
            "des",
            "du",
            "de",
            "et",
            "ou",
            "à",
            "dans",
            "sur",
            "pour",
            "par",
            "avec",
            "sans",
            "sous",
            "vers",
            "chez",
            "depuis",
            "pendant",
            "qui",
            "que",
            "quoi",
            "dont",
            "où",
            "quand",
            "comment",
            "pourquoi",
            "combien",
            "ce",
            "cette",
            "ces",
            "cet",
            "il",
            "elle",
            "ils",
            "elles",
            "on",
            "nous",
            "vous",
            "je",
            "tu",
            "me",
            "te",
            "se",
            "mon",
            "ma",
            "mes",
            "ton",
            "ta",
            "tes",
            "son",
            "sa",
            "ses",
            "est",
            "sont",
            "était",
            "étaient",
            "être",
            "avoir",
            "avait",
            "avaient",
            "fait",
            "faire",
            # Mots interrogatifs spécifiques
            "quelles",
            "quelle",
            "quel",
            "quels",
            # Anglais
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "up",
            "down",
            "out",
            "off",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "what",
            "which",
            "who",
            "whom",
            "this",
            "that",
            "these",
            "those",
            "am",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "having",
            "do",
            "does",
            "did",
            "doing",
        }

        # Nettoyage et tokenisation
        query_clean = re.sub(r"[^\w\s-]", " ", query.lower())
        words = [word.strip() for word in query_clean.split() if word.strip()]

        # Filtrage des mots vides et mots trop courts
        keywords = [word for word in words if word not in stop_words and len(word) > 2 and not word.isdigit()]

        return keywords

    def _calculate_context_boost(self, document: RetrievalDocument, context: IsskarContext, max_boost: float) -> float:
        """Calcule le boost contextuel pour un document"""

        boost = 0.0
        content_lower = document.content.lower()
        title_lower = document.metadata.get("title", "").lower()

        # Boost projets mentionnés dans le contexte
        for project in context.projects:
            if project in content_lower or project in title_lower:
                boost += max_boost * 0.4  # 40% du boost max pour chaque projet

        # Boost équipe mentionnée dans le contexte
        for member in context.team_members:
            if member in content_lower or member in title_lower:
                boost += max_boost * 0.3  # 30% du boost max pour chaque membre

        # Boost domaines techniques
        for domain in context.technical_domains:
            if domain in content_lower or domain in title_lower:
                boost += max_boost * 0.2  # 20% du boost max pour chaque domaine

        # Boost hiérarchie contextuelle
        for hierarchy in context.hierarchical_context:
            if hierarchy in content_lower or hierarchy in title_lower:
                boost += max_boost * 0.1  # 10% du boost max pour chaque niveau

        return min(boost, max_boost)  # Cap au boost maximum

    def _update_stats(self, query_type: QueryType, input_docs: int, output_docs: int):
        """Met à jour les statistiques de filtrage"""

        if query_type.value not in self.filtering_stats["by_query_type"]:
            self.filtering_stats["by_query_type"][query_type.value] = {
                "count": 0,
                "total_input_docs": 0,
                "total_output_docs": 0,
            }

        stats = self.filtering_stats["by_query_type"][query_type.value]
        stats["count"] += 1
        stats["total_input_docs"] += input_docs
        stats["total_output_docs"] += output_docs

        self.filtering_stats["documents_filtered"] += input_docs - output_docs
        self.filtering_stats["documents_passed"] += output_docs

    def get_filtering_stats(self) -> Dict:
        """Retourne les statistiques de filtrage pour monitoring"""

        # Calcul des ratios de filtrage
        enriched_stats = self.filtering_stats.copy()

        for query_type, stats in enriched_stats["by_query_type"].items():
            if stats["total_input_docs"] > 0:
                stats["pass_rate"] = stats["total_output_docs"] / stats["total_input_docs"]
                stats["avg_input_docs"] = stats["total_input_docs"] / stats["count"]
                stats["avg_output_docs"] = stats["total_output_docs"] / stats["count"]
            else:
                stats["pass_rate"] = 0.0
                stats["avg_input_docs"] = 0.0
                stats["avg_output_docs"] = 0.0

        # Stats globales
        total_docs = enriched_stats["documents_filtered"] + enriched_stats["documents_passed"]
        if total_docs > 0:
            enriched_stats["global_pass_rate"] = enriched_stats["documents_passed"] / total_docs
        else:
            enriched_stats["global_pass_rate"] = 0.0

        return enriched_stats

    def reset_stats(self):
        """Remet à zéro les statistiques"""
        self.filtering_stats = {"total_queries": 0, "by_query_type": {}, "documents_filtered": 0, "documents_passed": 0}

    def get_learning_stats(self) -> Dict:
        """Retourne les statistiques d'apprentissage des composants dynamiques"""

        stats = {
            "context_analyzer": self.context_analyzer.get_learned_stats(),
            "semantic_matcher": self.semantic_matcher.get_learning_stats(),
            "filtering": self.get_filtering_stats(),
        }

        return stats
