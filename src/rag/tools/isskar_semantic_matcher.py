#!/usr/bin/env python3
"""
Système de correspondance sémantique flexible optimisé pour Isskar
Gère les synonymes, variantes et domaines métiers spécifiques
"""

from typing import List
from dataclasses import dataclass


@dataclass
class SemanticMatch:
    """Résultat d'une correspondance sémantique"""

    score: float
    match_type: str  # 'exact', 'synonym', 'variant', 'domain'
    matched_term: str
    original_term: str


class IsskarSemanticMatcher:
    """
    Matcher sémantique expert pour l'environnement Isskar
    Optimisé pour les termes, projets et domaines d'expertise d'Isskar
    """

    def __init__(self):
        self._initialize_isskar_semantics()

    def _initialize_isskar_semantics(self):
        """Initialise la base sémantique Isskar"""

        # Synonymes métiers Isskar (basé sur l'analyse des documents)
        self.business_synonyms = {
            # Responsabilités et rôles
            "responsabilités": {
                "synonyms": ["rôles", "tâches", "fonctions", "missions", "attributions"],
                "weight": 0.85,
                "domain": "team_management",
            },
            "rôles": {
                "synonyms": ["responsabilités", "fonctions", "postes", "missions"],
                "weight": 0.85,
                "domain": "team_management",
            },
            # Équipe et contacts
            "équipe": {
                "synonyms": ["team", "groupe", "collaborateurs", "personnes", "membres"],
                "weight": 0.90,
                "domain": "team_management",
            },
            "contacter": {
                "synonyms": ["joindre", "contact", "email", "contacter", "atteindre", "écrire"],
                "weight": 0.90,
                "domain": "communication",
            },
            # Projets et missions
            "projet": {
                "synonyms": ["mission", "application", "système", "produit", "plateforme", "client"],
                "weight": 0.85,
                "domain": "project_management",
            },
            "mission": {
                "synonyms": ["projet", "client", "intervention", "prestation", "livrable"],
                "weight": 0.85,
                "domain": "project_management",
            },
            # Statut et avancement
            "avancement": {
                "synonyms": ["statut", "état", "progress", "évolution", "situation"],
                "weight": 0.80,
                "domain": "project_status",
            },
            "terminé": {
                "synonyms": ["fini", "achevé", "livré", "clos", "finalisé"],
                "weight": 0.85,
                "domain": "project_status",
            },
            # Technique
            "développement": {
                "synonyms": ["dev", "développer", "coding", "programmation", "implémentation"],
                "weight": 0.85,
                "domain": "technical",
            },
            "bug": {
                "synonyms": ["erreur", "problème", "défaut", "issue", "anomalie"],
                "weight": 0.90,
                "domain": "technical",
            },
            # Business
            "client": {
                "synonyms": ["entreprise", "société", "organisation", "partenaire"],
                "weight": 0.80,
                "domain": "business",
            },
            "budget": {
                "synonyms": ["coût", "prix", "tarif", "pricing", "financier"],
                "weight": 0.85,
                "domain": "business",
            },
        }

        # Variantes orthographiques spécifiques Isskar
        self.project_variants = {
            "teora": ["téora", "TEORA", "Teora", "Téora"],
            "cibtp": ["CIBTP", "cibtp-ao", "CIBTP - AO", "CIBTP-AO"],
            "isschat": ["ISSCHAT", "iss-chat", "IssChat", "Iss-Chat"],
            "cedrus": ["Cedrus", "Cedrus & Partners", "GenAut"],
            "fives": ["Fives", "FIVES"],
            "oddo": ["Oddo", "ODDO"],
            "obea": ["Obea", "OBEA", "Obea T&E"],
            "nowa": ["Nowa", "NOWA", "[Nowa]"],
            "iagen": ["IAGEN", "IaGen", "IAGen"],
        }

        # Variantes noms d'équipe
        self.team_variants = {
            "johan": ["JUBLANC Johan", "Johan JUBLANC", "Johan", "JUBLANC"],
            "vincent": ["Vincent Fraillon", "Fraillon Vincent", "Vincent", "Fraillon"],
            "damien": ["Damien Saby", "Saby Damien", "Damien", "Saby"],
            "henri": ["Henri AVERLAND", "AVERLAND Henri", "Henri", "AVERLAND"],
            "gregoire": ["Grégoire SALHA", "SALHA Grégoire", "Grégoire", "SALHA", "Gregoire"],
            "nicolas": ["Nicolas LAMBROPOULOS", "LAMBROPOULOS Nicolas", "Nicolas", "LAMBROPOULOS"],
            "francois": ["François DUPONT", "DUPONT François", "François", "DUPONT", "Francois"],
            "emin": ["Emin Calyaka", "Calyaka Emin", "Emin", "Calyaka"],
            "amaury": ["Amaury Dreux", "Dreux Amaury", "Amaury", "Dreux"],
            "adame": ["Adame Ben Friha", "Ben Friha Adame", "Adame", "Ben Friha"],
        }

        # Domaines techniques avec termes associés
        self.technical_domains = {
            "data_science": {
                "terms": ["data", "données", "analytics", "machine learning", "ML", "IA", "AI", "metrics", "kpi"],
                "weight": 0.85,
            },
            "infrastructure": {
                "terms": ["docker", "kubernetes", "k8s", "minikube", "deploy", "deployment", "infra"],
                "weight": 0.90,
            },
            "development": {
                "terms": ["code", "coding", "dev", "développement", "github", "git", "programming"],
                "weight": 0.85,
            },
            "security": {
                "terms": ["sécurité", "sécu", "RSSI", "security", "souveraineté", "protection"],
                "weight": 0.90,
            },
        }

    def calculate_semantic_relevance(
        self, query_keywords: List[str], document_content: str, document_title: str = "", apply_synonyms: bool = True
    ) -> float:
        """
        Calcule la pertinence sémantique avec matching flexible Isskar
        """
        if not query_keywords:
            return 0.0

        content_lower = document_content.lower()
        title_lower = document_title.lower()

        total_matches = 0.0
        max_possible_score = len(query_keywords) * 2  # Title weight = 2, content = 1

        for keyword in query_keywords:
            keyword_lower = keyword.lower()

            # 1. Correspondances exactes (poids maximum)
            title_exact = 1.0 if keyword_lower in title_lower else 0.0
            content_exact = 1.0 if keyword_lower in content_lower else 0.0

            # 2. Correspondances par variantes de projets/équipe
            variant_match = self._calculate_variant_match(keyword_lower, content_lower, title_lower)

            # 3. Correspondances synonymiques (si activées)
            synonym_match = 0.0
            if apply_synonyms:
                synonym_match = self._calculate_synonym_match(keyword_lower, content_lower, title_lower)

            # 4. Correspondances domaines techniques
            domain_match = self._calculate_domain_match(keyword_lower, content_lower, title_lower)

            # Score final pour ce mot-clé (prend le meilleur)
            keyword_score = max(
                title_exact * 2 + content_exact,  # Exact matches
                variant_match * 1.8,  # Variants (légèrement moins que exact)
                synonym_match * 1.5,  # Synonyms (poids moyen)
                domain_match * 1.2,  # Domain (poids faible)
            )

            total_matches += keyword_score

        # Normalisation
        base_relevance = total_matches / max_possible_score

        # Bonus pour phrases exactes et contexte enrichi
        phrase_bonus = self._calculate_phrase_bonus(query_keywords, content_lower, title_lower)

        final_score = min(1.0, base_relevance + phrase_bonus)

        return final_score

    def _calculate_variant_match(self, keyword: str, content: str, title: str) -> float:
        """Calcule les correspondances par variantes orthographiques"""

        # Variantes projets
        for canonical, variants in self.project_variants.items():
            if keyword == canonical or keyword in [v.lower() for v in variants]:
                for variant in variants + [canonical]:
                    if variant.lower() in content:
                        return 0.95  # Quasi-exact pour variantes projets
                    if variant.lower() in title:
                        return 1.8  # Titre = poids double

        # Variantes équipe
        for canonical, variants in self.team_variants.items():
            if keyword == canonical or keyword in [v.lower() for v in variants]:
                for variant in variants + [canonical]:
                    if variant.lower() in content:
                        return 0.95
                    if variant.lower() in title:
                        return 1.8

        return 0.0

    def _calculate_synonym_match(self, keyword: str, content: str, title: str) -> float:
        """Calcule les correspondances synonymiques"""

        if keyword in self.business_synonyms:
            synonym_info = self.business_synonyms[keyword]
            weight = synonym_info["weight"]

            for synonym in synonym_info["synonyms"]:
                if synonym in content:
                    return weight  # Score pondéré selon la qualité du synonyme
                if synonym in title:
                    return weight * 2  # Poids double pour titre

        # Recherche inverse (le keyword est peut-être un synonyme)
        for canonical, info in self.business_synonyms.items():
            if keyword in info["synonyms"]:
                if canonical in content:
                    return info["weight"]
                if canonical in title:
                    return info["weight"] * 2

        return 0.0

    def _calculate_domain_match(self, keyword: str, content: str, title: str) -> float:
        """Calcule les correspondances par domaine technique"""

        for domain, domain_info in self.technical_domains.items():
            if keyword in domain_info["terms"]:
                # Recherche d'autres termes du même domaine
                domain_matches = sum(1 for term in domain_info["terms"] if term in content)
                if domain_matches > 0:
                    return domain_info["weight"] * min(domain_matches / len(domain_info["terms"]), 1.0)

                title_matches = sum(1 for term in domain_info["terms"] if term in title)
                if title_matches > 0:
                    return domain_info["weight"] * 2 * min(title_matches / len(domain_info["terms"]), 1.0)

        return 0.0

    def _calculate_phrase_bonus(self, keywords: List[str], content: str, title: str) -> float:
        """Calcule les bonus pour phrases exactes et enrichissements contextuels"""

        bonus = 0.0

        # Phrase exacte
        query_phrase = " ".join(keywords)
        if query_phrase in content:
            bonus += 0.15
        if query_phrase in title:
            bonus += 0.25

        # Bonus proximité des termes (mots proches dans le texte)
        if len(keywords) > 1:
            proximity_bonus = self._calculate_proximity_bonus(keywords, content)
            bonus += proximity_bonus * 0.1

        # Bonus contextuel Isskar (mentions de projets/équipe dans la même zone)
        context_bonus = self._calculate_isskar_context_bonus(keywords, content)
        bonus += context_bonus * 0.08

        return bonus

    def _calculate_proximity_bonus(self, keywords: List[str], content: str) -> float:
        """Bonus si les mots-clés apparaissent proches dans le contenu"""

        words = content.split()
        positions = {}

        # Trouve les positions de chaque keyword
        for i, word in enumerate(words):
            for keyword in keywords:
                if keyword in word.lower():
                    if keyword not in positions:
                        positions[keyword] = []
                    positions[keyword].append(i)

        if len(positions) < 2:
            return 0.0

        # Calcule la proximité minimale entre les termes
        min_distance = float("inf")
        keyword_positions = list(positions.values())

        for i in range(len(keyword_positions)):
            for j in range(i + 1, len(keyword_positions)):
                for pos1 in keyword_positions[i]:
                    for pos2 in keyword_positions[j]:
                        distance = abs(pos1 - pos2)
                        min_distance = min(min_distance, distance)

        # Bonus inversement proportionnel à la distance
        if min_distance < 10:  # Mots très proches
            return 1.0
        elif min_distance < 25:  # Mots assez proches
            return 0.5
        else:
            return 0.1

    def _calculate_isskar_context_bonus(self, keywords: List[str], content: str) -> float:
        """Bonus si le contenu mentionne des éléments contextuels Isskar"""

        bonus = 0.0
        content_lower = content.lower()

        # Bonus projets mentionnés
        projects_mentioned = 0
        for project in self.project_variants.keys():
            if project in content_lower:
                projects_mentioned += 1

        if projects_mentioned > 0:
            bonus += min(projects_mentioned * 0.3, 0.8)

        # Bonus équipe mentionnée
        team_mentioned = 0
        for member in self.team_variants.keys():
            if member in content_lower:
                team_mentioned += 1

        if team_mentioned > 0:
            bonus += min(team_mentioned * 0.2, 0.6)

        # Bonus hiérarchie Isskar (Missions, Fondations, etc.)
        hierarchy_terms = ["missions", "fondations", "business", "veille", "technique"]
        hierarchy_bonus = sum(0.1 for term in hierarchy_terms if term in content_lower)
        bonus += min(hierarchy_bonus, 0.4)

        return bonus

    def get_enhanced_keywords(self, original_keywords: List[str]) -> List[str]:
        """
        Enrichit la liste de mots-clés avec des variantes et synonymes pertinents
        """
        enhanced = set(original_keywords)

        for keyword in original_keywords:
            keyword_lower = keyword.lower()

            # Ajoute les variantes de projets
            for canonical, variants in self.project_variants.items():
                if keyword_lower == canonical or keyword_lower in [v.lower() for v in variants]:
                    enhanced.update([canonical] + [v.lower() for v in variants])

            # Ajoute les variantes d'équipe
            for canonical, variants in self.team_variants.items():
                if keyword_lower == canonical or keyword_lower in [v.lower() for v in variants]:
                    enhanced.update([canonical] + [v.lower() for v in variants])

            # Ajoute les synonymes les plus pertinents
            if keyword_lower in self.business_synonyms:
                synonyms = self.business_synonyms[keyword_lower]["synonyms"]
                # Prend seulement les 2 meilleurs synonymes pour éviter le bruit
                enhanced.update(synonyms[:2])

        return list(enhanced)
