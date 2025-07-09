"""
Metadata enricher for Confluence documents
Adds author, contributors, hierarchical path and other detailed metadata
"""

import requests
from typing import List, Dict, Any
import logging


class ConfluenceMetadataEnricher:
    """Enriches metadata of Confluence pages"""

    def __init__(self, base_url: str, username: str, api_token: str):
        # Handle different Confluence URL formats
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

        # Cache to avoid repeated API calls
        self._page_cache = {}
        self._hierarchy_cache = {}
        self._siblings_cache = {}

    def test_connection(self) -> bool:
        """Test connection to Confluence API"""
        try:
            url = f"{self.api_base}/space"
            response = self.session.get(url, params={"limit": 1})
            response.raise_for_status()
            self.logger.info("Confluence API connection successful")
            return True
        except Exception as e:
            self.logger.error(f"Confluence API connection failed: {e}")
            # Debug: try different URLs
            test_urls = [
                f"{self.base_url}/rest/api/space",
                f"{self.base_url}/wiki/rest/api/space",
                f"{self.base_url}/wiki/api/v2/spaces",
            ]

            for test_url in test_urls:
                try:
                    response = self.session.get(test_url, params={"limit": 1})
                    if response.status_code == 200:
                        self.logger.info(f"Functional URL found: {test_url}")
                        # Update the base URL
                        if "api/v2" in test_url:
                            self.api_base = f"{self.base_url}/wiki/api/v2"
                        else:
                            self.api_base = test_url.replace("/space", "")
                        return True
                    else:
                        self.logger.debug(f"URL {test_url} returned: {response.status_code}")
                except Exception as ex:
                    self.logger.debug(f"URL {test_url} failed: {ex}")

            return False

    def enrich_document_metadata(self, document_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Enriches document metadata with additional information"""
        page_id = document_metadata.get("page_id")
        if not page_id:
            return document_metadata

        try:
            # Retrieve enriched information
            page_details = self._get_page_details(page_id)
            hierarchy_path = self._get_hierarchy_path(page_id)
            contributors = self._get_page_contributors(page_id)

            # Enrich the metadata
            enriched_metadata = document_metadata.copy()
            enriched_metadata.update(
                {
                    # Authoring information
                    "author_id": page_details.get("author_id"),
                    "author_name": page_details.get("author_name"),
                    "author_email": page_details.get("author_email"),
                    "created_date": page_details.get("created_date"),
                    "last_modified_date": page_details.get("last_modified_date"),
                    "last_modified_by": page_details.get("last_modified_by"),
                    "version_number": page_details.get("version_number"),
                    # Hierarchical path
                    "hierarchy_breadcrumb": self._format_hierarchy_breadcrumb(hierarchy_path),
                    "enriched_hierarchy_tree": self._build_enriched_hierarchy_tree(page_id, hierarchy_path),
                    "parent_pages": [{"id": p["id"], "title": p["title"]} for p in hierarchy_path[:-1]],
                    "page_depth": len(hierarchy_path) - 1,
                    # Contributors
                    "contributors": contributors,
                    "contributors_count": len(contributors),
                    "contributors_names": [c["name"] for c in contributors if c["name"]],
                    # Labels and categorization
                    "labels": page_details.get("labels", []),
                    "page_type": page_details.get("page_type", "page"),
                    # Statistics
                    "has_attachments": page_details.get("has_attachments", False),
                    "attachments_count": page_details.get("attachments_count", 0),
                }
            )

            return enriched_metadata

        except Exception as e:
            self.logger.warning(f"Failed to enrich metadata for page {page_id}: {e}")
            return document_metadata

    def _get_page_details(self, page_id: str) -> Dict[str, Any]:
        """Retrieve complete details of a page"""
        if page_id in self._page_cache:
            return self._page_cache[page_id]

        try:
            # API call to retrieve page details
            url = f"{self.api_base}/content/{page_id}"
            params = {"expand": "version,ancestors,metadata.labels,children.attachment"}

            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            # Extract information
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
        """Retrieve complete hierarchical path"""
        if page_id in self._hierarchy_cache:
            return self._hierarchy_cache[page_id]

        try:
            url = f"{self.api_base}/content/{page_id}"
            params = {"expand": "ancestors"}

            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            # Build hierarchical path
            hierarchy = []

            # Add ancestors
            for ancestor in data.get("ancestors", []):
                hierarchy.append({"id": ancestor["id"], "title": ancestor["title"], "type": ancestor["type"]})

            # Add current page
            hierarchy.append({"id": data["id"], "title": data["title"], "type": data["type"]})

            self._hierarchy_cache[page_id] = hierarchy
            return hierarchy

        except Exception as e:
            self.logger.error(f"Failed to get hierarchy for {page_id}: {e}")
            return [{"id": page_id, "title": "Unknown", "type": "page"}]

    def _get_page_contributors(self, page_id: str) -> List[Dict[str, str]]:
        """Retrieve the list of contributors"""
        try:
            url = f"{self.api_base}/content/{page_id}/history"
            params = {"limit": 50}  # Limit to avoid too many calls

            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            contributors = {}

            # Go through version history
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
        """Format hierarchical path into readable breadcrumb"""
        titles = [page["title"] for page in hierarchy_path]
        return " > ".join(titles)

    def _build_enriched_hierarchy_tree(self, page_id: str, hierarchy_path: List[Dict[str, str]]) -> str:
        """Build enriched hierarchy tree showing current page position with siblings"""
        try:
            # Get siblings for each level in the hierarchy
            tree_lines = []

            for i, page in enumerate(hierarchy_path):
                is_current = page["id"] == page_id

                # Get siblings for this level
                if i == 0:
                    # Root level - get children of parent space
                    siblings = self._get_space_children(page["id"])
                else:
                    # Get children of the parent page
                    parent_id = hierarchy_path[i - 1]["id"]
                    siblings = self._get_page_children(parent_id)

                # Build tree representation for this level
                level_indent = "│   " * i

                # Add siblings at this level
                for sibling in siblings:
                    is_selected = sibling["id"] == page["id"]
                    is_folder = sibling.get("has_children", False)

                    # Tree symbols
                    if is_selected:
                        marker = "├── " if not is_folder else "├── "
                        status = " (sélectionné)" if is_current else " ✓"
                    else:
                        marker = "├── " if not is_folder else "├── "
                        status = "/" if is_folder else ""

                    line = f"{level_indent}{marker}{sibling['title']}{status}"
                    tree_lines.append(line)

                # If this is the current page, show its children with deeper indentation
                if is_current and i == len(hierarchy_path) - 1:
                    children = self._get_page_children(page["id"])
                    child_indent = "│   " * (i + 1)
                    for child in children:
                        child_marker = "├── " if not child.get("has_children", False) else "├── "
                        child_status = "/" if child.get("has_children", False) else ""
                        child_line = f"{child_indent}{child_marker}{child['title']}{child_status}"
                        tree_lines.append(child_line)

            return "\n".join(tree_lines)

        except Exception as e:
            self.logger.error(f"Failed to build enriched hierarchy tree for {page_id}: {e}")
            return self._format_hierarchy_breadcrumb(hierarchy_path)

    def _get_space_children(self, space_id: str) -> List[Dict[str, Any]]:
        """Get children pages of a space"""
        cache_key = f"space_children_{space_id}"
        if cache_key in self._siblings_cache:
            return self._siblings_cache[cache_key]

        try:
            url = f"{self.api_base}/content"
            params = {"spaceKey": space_id, "type": "page", "limit": 100, "expand": "children.page"}

            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            children = []
            for page in data.get("results", []):
                has_children = len(page.get("children", {}).get("page", {}).get("results", [])) > 0
                children.append(
                    {"id": page["id"], "title": page["title"], "type": page["type"], "has_children": has_children}
                )

            self._siblings_cache[cache_key] = children
            return children

        except Exception as e:
            self.logger.error(f"Failed to get space children for {space_id}: {e}")
            return []

    def _get_page_children(self, parent_id: str) -> List[Dict[str, Any]]:
        """Get children pages of a parent page"""
        cache_key = f"page_children_{parent_id}"
        if cache_key in self._siblings_cache:
            return self._siblings_cache[cache_key]

        try:
            url = f"{self.api_base}/content/{parent_id}/child/page"
            params = {"limit": 100, "expand": "children.page"}

            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            children = []
            for page in data.get("results", []):
                has_children = len(page.get("children", {}).get("page", {}).get("results", [])) > 0
                children.append(
                    {"id": page["id"], "title": page["title"], "type": page["type"], "has_children": has_children}
                )

            self._siblings_cache[cache_key] = children
            return children

        except Exception as e:
            self.logger.error(f"Failed to get page children for {parent_id}: {e}")
            return []

    def update_chunk_context(self, chunk_content: str, enriched_metadata: Dict[str, Any]) -> str:
        """Update chunk context with enriched metadata"""
        # Build new enriched context
        context_parts = []

        # Document and hierarchy
        if enriched_metadata.get("hierarchy_breadcrumb"):
            context_parts.append(f"Document: {enriched_metadata.get('hierarchy_breadcrumb')}")
        else:
            context_parts.append(f"Document: {enriched_metadata.get('title')}")

        # Space
        if enriched_metadata.get("space_key"):
            context_parts.append(f"Space: {enriched_metadata.get('space_key')}")

        # Author
        if enriched_metadata.get("author_name"):
            context_parts.append(f"Author: {enriched_metadata.get('author_name')}")

        # Contributors
        contributors_names = enriched_metadata.get("contributors_names", [])
        if contributors_names:
            # Limit to 3 contributors to avoid context being too long
            contrib_display = contributors_names[:3]
            if len(contributors_names) > 3:
                contrib_display.append(f"+ {len(contributors_names) - 3} others")
            context_parts.append(f"Contributors: {', '.join(contrib_display)}")

        # Dates
        created_date = enriched_metadata.get("created_date")
        if created_date:
            created = created_date[:10]  # YYYY-MM-DD
            context_parts.append(f"Created: {created}")

        last_modified_date = enriched_metadata.get("last_modified_date")
        if last_modified_date:
            modified = last_modified_date[:10]  # YYYY-MM-DD
            context_parts.append(f"Modified: {modified}")

        # URL
        if enriched_metadata.get("url"):
            context_parts.append(f"URL: {enriched_metadata.get('url')}")

        # Current section
        if enriched_metadata.get("hierarchy_path"):
            context_parts.append(f"Section: {enriched_metadata.get('hierarchy_path')}")

        # Content type
        if enriched_metadata.get("content_type"):
            context_parts.append(f"Type: {enriched_metadata.get('content_type')}")

        # Source
        context_parts.append("Source: confluence")

        # Build final context
        context_header = f"[{' | '.join(context_parts)}]"

        # Extract content without old context
        if chunk_content.startswith("[") and "]\n\n" in chunk_content:
            actual_content = chunk_content.split("]\n\n", 1)[1]
        else:
            actual_content = chunk_content

        return f"{context_header}\n\n{actual_content}"
