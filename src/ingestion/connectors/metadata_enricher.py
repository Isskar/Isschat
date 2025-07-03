"""
Enrichisseur de métadonnées pour les documents Confluence
Ajoute l'auteur, contributeurs, chemin hiérarchique et autres métadonnées détaillées
"""

import requests
from typing import List, Dict, Any
import logging


class ConfluenceMetadataEnricher:
    """Enrichit les métadonnées des pages Confluence"""

    def __init__(self, base_url: str, username: str, api_token: str):
        # Gérer les différents formats d'URL Confluence
        if "/wiki" in base_url:
            self.base_url = base_url.rstrip("/wiki")
        else:
            self.base_url = base_url.rstrip("/")

        self.api_base = f"{self.base_url}/wiki/rest/api"
        self.username = username
        self.api_token = api_token
        self.session = requests.Session()
        self.session.auth = (username, api_token)
        self.session.headers.update({"Accept": "application/json", "Content-Type": "application/json"})
        self.logger = logging.getLogger(self.__class__.__name__)

        # Cache pour éviter les appels API répétés
        self._page_cache = {}
        self._hierarchy_cache = {}

    def test_connection(self) -> bool:
        """Test la connexion à l'API Confluence"""
        try:
            url = f"{self.api_base}/space"
            response = self.session.get(url, params={"limit": 1})
            response.raise_for_status()
            self.logger.info("Connexion API Confluence réussie")
            return True
        except Exception as e:
            self.logger.error(f"Échec de connexion API Confluence: {e}")
            # Debug: essayer différentes URL
            test_urls = [
                f"{self.base_url}/rest/api/space",
                f"{self.base_url}/wiki/rest/api/space",
                f"{self.base_url}/wiki/api/v2/spaces",
            ]

            for test_url in test_urls:
                try:
                    response = self.session.get(test_url, params={"limit": 1})
                    if response.status_code == 200:
                        self.logger.info(f"URL fonctionnelle trouvée: {test_url}")
                        # Mettre à jour l'URL de base
                        if "api/v2" in test_url:
                            self.api_base = f"{self.base_url}/wiki/api/v2"
                        else:
                            self.api_base = test_url.replace("/space", "")
                        return True
                    else:
                        self.logger.debug(f"URL {test_url} a retourné: {response.status_code}")
                except Exception as ex:
                    self.logger.debug(f"URL {test_url} a échoué: {ex}")

            return False

    def enrich_document_metadata(self, document_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Enrichit les métadonnées d'un document avec les informations supplémentaires"""
        page_id = document_metadata.get("page_id")
        if not page_id:
            return document_metadata

        try:
            # Récupérer les informations enrichies
            page_details = self._get_page_details(page_id)
            hierarchy_path = self._get_hierarchy_path(page_id)
            contributors = self._get_page_contributors(page_id)

            # Enrichir les métadonnées
            enriched_metadata = document_metadata.copy()
            enriched_metadata.update(
                {
                    # Informations d'authoring
                    "author_id": page_details.get("author_id"),
                    "author_name": page_details.get("author_name"),
                    "author_email": page_details.get("author_email"),
                    "created_date": page_details.get("created_date"),
                    "last_modified_date": page_details.get("last_modified_date"),
                    "last_modified_by": page_details.get("last_modified_by"),
                    "version_number": page_details.get("version_number"),
                    # Chemin hiérarchique
                    "hierarchy_breadcrumb": self._format_hierarchy_breadcrumb(hierarchy_path),
                    "parent_pages": [{"id": p["id"], "title": p["title"]} for p in hierarchy_path[:-1]],
                    "page_depth": len(hierarchy_path) - 1,
                    # Contributeurs
                    "contributors": contributors,
                    "contributors_count": len(contributors),
                    "contributors_names": [c["name"] for c in contributors if c["name"]],
                    # Labels et catégorisation
                    "labels": page_details.get("labels", []),
                    "page_type": page_details.get("page_type", "page"),
                    # Statistiques
                    "has_attachments": page_details.get("has_attachments", False),
                    "attachments_count": page_details.get("attachments_count", 0),
                }
            )

            return enriched_metadata

        except Exception as e:
            self.logger.warning(f"Failed to enrich metadata for page {page_id}: {e}")
            return document_metadata

    def _get_page_details(self, page_id: str) -> Dict[str, Any]:
        """Récupère les détails complets d'une page"""
        if page_id in self._page_cache:
            return self._page_cache[page_id]

        try:
            # Appel API pour récupérer les détails de la page
            url = f"{self.api_base}/content/{page_id}"
            params = {"expand": "version,ancestors,metadata.labels,children.attachment"}

            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            # Extraire les informations
            version = data.get("version", {})
            author = version.get("by", {})

            page_details = {
                "author_id": author.get("accountId"),
                "author_name": author.get("displayName"),
                "author_email": author.get("email"),
                "created_date": version.get("when"),
                "last_modified_date": version.get("when"),
                "last_modified_by": author.get("displayName"),
                "version_number": version.get("number"),
                "labels": [label["name"] for label in data.get("metadata", {}).get("labels", {}).get("results", [])],
                "page_type": data.get("type", "page"),
                "has_attachments": len(data.get("children", {}).get("attachment", {}).get("results", [])) > 0,
                "attachments_count": len(data.get("children", {}).get("attachment", {}).get("results", [])),
            }

            self._page_cache[page_id] = page_details
            return page_details

        except Exception as e:
            self.logger.error(f"Failed to get page details for {page_id}: {e}")
            return {}

    def _get_hierarchy_path(self, page_id: str) -> List[Dict[str, str]]:
        """Récupère le chemin hiérarchique complet"""
        if page_id in self._hierarchy_cache:
            return self._hierarchy_cache[page_id]

        try:
            url = f"{self.api_base}/content/{page_id}"
            params = {"expand": "ancestors"}

            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            # Construire le chemin hiérarchique
            hierarchy = []

            # Ajouter les ancêtres
            for ancestor in data.get("ancestors", []):
                hierarchy.append({"id": ancestor["id"], "title": ancestor["title"], "type": ancestor["type"]})

            # Ajouter la page actuelle
            hierarchy.append({"id": data["id"], "title": data["title"], "type": data["type"]})

            self._hierarchy_cache[page_id] = hierarchy
            return hierarchy

        except Exception as e:
            self.logger.error(f"Failed to get hierarchy for {page_id}: {e}")
            return [{"id": page_id, "title": "Unknown", "type": "page"}]

    def _get_page_contributors(self, page_id: str) -> List[Dict[str, str]]:
        """Récupère la liste des contributeurs"""
        try:
            url = f"{self.api_base}/content/{page_id}/history"
            params = {"limit": 50}  # Limite pour éviter trop d'appels

            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            contributors = {}

            # Parcourir l'historique des versions
            for version in data.get("results", []):
                author = version.get("by", {})
                account_id = author.get("accountId")

                if account_id and account_id not in contributors:
                    contributors[account_id] = {
                        "id": account_id,
                        "name": author.get("displayName"),
                        "email": author.get("email"),
                        "type": author.get("type"),
                        "first_contribution": version.get("when"),
                    }

            return list(contributors.values())

        except Exception as e:
            self.logger.error(f"Failed to get contributors for {page_id}: {e}")
            return []

    def _format_hierarchy_breadcrumb(self, hierarchy_path: List[Dict[str, str]]) -> str:
        """Formate le chemin hiérarchique en breadcrumb lisible"""
        titles = [page["title"] for page in hierarchy_path]
        return " > ".join(titles)

    def update_chunk_context(self, chunk_content: str, enriched_metadata: Dict[str, Any]) -> str:
        """Met à jour le contexte d'un chunk avec les métadonnées enrichies"""
        # Construire le nouveau contexte enrichi
        context_parts = []

        # Document et hiérarchie
        if enriched_metadata.get("hierarchy_breadcrumb"):
            context_parts.append(f"Document: {enriched_metadata.get('hierarchy_breadcrumb')}")
        else:
            context_parts.append(f"Document: {enriched_metadata.get('title')}")

        # Espace
        if enriched_metadata.get("space_key"):
            context_parts.append(f"Espace: {enriched_metadata.get('space_key')}")

        # Auteur
        if enriched_metadata.get("author_name"):
            context_parts.append(f"Auteur: {enriched_metadata.get('author_name')}")

        # Contributeurs
        contributors_names = enriched_metadata.get("contributors_names", [])
        if contributors_names:
            # Limiter à 3 contributeurs pour éviter un contexte trop long
            contrib_display = contributors_names[:3]
            if len(contributors_names) > 3:
                contrib_display.append(f"+ {len(contributors_names) - 3} autres")
            context_parts.append(f"Contributeurs: {', '.join(contrib_display)}")

        # Dates
        created_date = enriched_metadata.get("created_date")
        if created_date:
            created = created_date[:10]  # YYYY-MM-DD
            context_parts.append(f"Créé: {created}")

        last_modified_date = enriched_metadata.get("last_modified_date")
        if last_modified_date:
            modified = last_modified_date[:10]  # YYYY-MM-DD
            context_parts.append(f"Modifié: {modified}")

        # URL
        if enriched_metadata.get("url"):
            context_parts.append(f"URL: {enriched_metadata.get('url')}")

        # Section actuelle
        if enriched_metadata.get("hierarchy_path"):
            context_parts.append(f"Section: {enriched_metadata.get('hierarchy_path')}")

        # Type de contenu
        if enriched_metadata.get("content_type"):
            context_parts.append(f"Type: {enriched_metadata.get('content_type')}")

        # Source
        context_parts.append("Source: confluence")

        # Construire le contexte final
        context_header = f"[{' | '.join(context_parts)}]"

        # Extraire le contenu sans l'ancien contexte
        if chunk_content.startswith("[") and "]\n\n" in chunk_content:
            actual_content = chunk_content.split("]\n\n", 1)[1]
        else:
            actual_content = chunk_content

        return f"{context_header}\n\n{actual_content}"
