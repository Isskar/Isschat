"""
SharePoint connector using Microsoft Graph API for document extraction.
"""

import logging
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional
from urllib.parse import quote

from .base_connector import BaseConnector
from ...core.interfaces import Document


class SharePointConnector(BaseConnector):
    """Connector for SharePoint Online using Microsoft Graph API."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Extract configuration
        self.tenant_id = config.get("tenant_id")
        self.client_id = config.get("client_id")
        self.client_secret = config.get("client_secret")
        self.site_url = config.get("site_url")  # e.g., "https://inergie.sharepoint.com/sites/ISSKAR"

        # Parse site information from URL
        self._parse_site_info()
        self._validate_config()

        # Authentication
        self.access_token = None
        self.token_expires_at = None

    def _parse_site_info(self):
        """Parse site information from SharePoint URL."""
        if not self.site_url:
            raise ValueError("site_url is required")

        # Extract hostname and site path from URL
        # e.g., "https://inergie.sharepoint.com/sites/ISSKAR"
        parts = self.site_url.replace("https://", "").split("/")
        self.hostname = parts[0]  # "inergie.sharepoint.com"
        self.site_path = "/" + "/".join(parts[1:])  # "/sites/ISSKAR"

    def _validate_config(self):
        """Validate required configuration parameters."""
        required_fields = ["tenant_id", "client_id", "client_secret", "site_url"]
        missing_fields = [field for field in required_fields if not self.config.get(field)]

        if missing_fields:
            raise ValueError(f"Missing required configuration: {missing_fields}")

    def _get_access_token(self) -> str:
        """Get access token using client credentials flow."""
        if self.access_token and self.token_expires_at:
            if datetime.now() < self.token_expires_at:
                return self.access_token

        # Request new token
        token_url = f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token"

        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "scope": "https://graph.microsoft.com/.default",
        }

        try:
            response = requests.post(token_url, data=data)
            response.raise_for_status()

            token_data = response.json()
            self.access_token = token_data["access_token"]

            # Set expiration time (subtract 5 minutes for safety)
            expires_in = token_data.get("expires_in", 3600)
            self.token_expires_at = datetime.now().timestamp() + expires_in - 300

            self.logger.info("Successfully obtained access token")
            return self.access_token

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to obtain access token: {e}")
            raise

    def _make_graph_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make authenticated request to Microsoft Graph API."""
        token = self._get_access_token()

        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        url = f"https://graph.microsoft.com/v1.0{endpoint}"

        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Graph API request failed: {e}")
            raise

    def validate_connection(self) -> bool:
        """Validate connection to SharePoint site."""
        try:
            # Try to get site information
            encoded_url = f"{self.hostname}:{quote(self.site_path, safe='')}"
            endpoint = f"/sites/{encoded_url}"

            site_info = self._make_graph_request(endpoint)

            self.logger.info(f"Connection validated for site: {site_info.get('displayName', 'Unknown')}")
            return True

        except Exception as e:
            self.logger.error(f"Connection validation failed: {e}")
            return False

    def extract(self) -> List[Document]:
        """Extract all documents from SharePoint document libraries."""
        try:
            self.logger.info(f"Starting extraction from SharePoint site: {self.site_url}")

            documents = []

            # Get all document libraries in the site
            libraries = self._get_document_libraries()

            for library in libraries:
                library_documents = self._extract_from_library(library)
                documents.extend(library_documents)

            self.logger.info(f"Extraction completed: {len(documents)} documents")
            return documents

        except Exception as e:
            self.logger.error(f"Extraction failed: {e}")
            return []

    def _get_document_libraries(self) -> List[Dict]:
        """Get all document libraries in the site."""
        try:
            encoded_url = f"{self.hostname}:{quote(self.site_path, safe='')}"
            endpoint = f"/sites/{encoded_url}/drives"

            response = self._make_graph_request(endpoint)

            # Filter to document libraries only
            libraries = []
            for drive in response.get("value", []):
                drive_type = drive.get("driveType", "")
                if drive_type in ["documentLibrary", "business"]:
                    libraries.append(drive)

            self.logger.info(f"Found {len(libraries)} document libraries")
            return libraries

        except Exception as e:
            self.logger.error(f"Failed to get document libraries: {e}")
            return []

    def _extract_from_library(self, library: Dict) -> List[Document]:
        """Extract documents from a specific document library."""
        library_name = library.get("name", "Unknown")
        library_id = library.get("id")

        self.logger.info(f"Extracting from library: {library_name}")

        documents = []

        try:
            # Get all items from the library
            endpoint = f"/drives/{library_id}/root/children"
            items = self._get_all_items_recursively(endpoint)

            for item in items:
                # Only process files, not folders
                if "file" in item:
                    doc = self._convert_item_to_document(item, library)
                    if doc:
                        documents.append(doc)

            self.logger.info(f"Extracted {len(documents)} documents from {library_name}")
            return documents

        except Exception as e:
            self.logger.error(f"Failed to extract from library {library_name}: {e}")
            return []

    def _get_all_items_recursively(self, endpoint: str, items: List[Dict] = None) -> List[Dict]:
        """Get all items recursively, handling pagination and folders."""
        if items is None:
            items = []

        try:
            response = self._make_graph_request(endpoint)

            for item in response.get("value", []):
                items.append(item)

                # If it's a folder, recursively get its contents
                if "folder" in item and item["folder"].get("childCount", 0) > 0:
                    folder_endpoint = f"{endpoint.split('/children')[0]}/items/{item['id']}/children"
                    self._get_all_items_recursively(folder_endpoint, items)

            # Handle pagination
            next_link = response.get("@odata.nextLink")
            if next_link:
                # Extract the endpoint from the full URL
                next_endpoint = next_link.replace("https://graph.microsoft.com/v1.0", "")
                self._get_all_items_recursively(next_endpoint, items)

            return items

        except Exception as e:
            self.logger.error(f"Failed to get items from {endpoint}: {e}")
            return items

    def _convert_item_to_document(self, item: Dict, library: Dict) -> Optional[Document]:
        """Convert SharePoint item to Document object."""
        try:
            # Get file content
            download_url = item.get("@microsoft.graph.downloadUrl")
            if not download_url:
                return None

            content = self._download_file_content(download_url)
            if not content:
                return None

            # Build metadata
            metadata = {
                "source": "sharepoint",
                "site_url": self.site_url,
                "library_name": library.get("name"),
                "library_id": library.get("id"),
                "file_name": item.get("name"),
                "file_id": item.get("id"),
                "web_url": item.get("webUrl"),
                "created_datetime": item.get("createdDateTime"),
                "modified_datetime": item.get("lastModifiedDateTime"),
                "size": item.get("size", 0),
                "mime_type": item.get("file", {}).get("mimeType"),
                "created_by": item.get("createdBy", {}).get("user", {}).get("displayName"),
                "modified_by": item.get("lastModifiedBy", {}).get("user", {}).get("displayName"),
                "content_length": len(content),
            }

            return Document(content=content, metadata=metadata)

        except Exception as e:
            self.logger.warning(f"Failed to convert item {item.get('name', 'unknown')}: {e}")
            return None

    def _download_file_content(self, download_url: str) -> Optional[str]:
        """Download file content from SharePoint."""
        try:
            response = requests.get(download_url)
            response.raise_for_status()

            # Try to decode as text (for Office documents, PDFs, etc., you might need specific libraries)
            try:
                content = response.text
                return content
            except UnicodeDecodeError:
                # For binary files, you might want to use specific parsers
                self.logger.warning("Binary file detected, skipping text extraction")
                return None

        except Exception as e:
            self.logger.error(f"Failed to download file: {e}")
            return None

    def get_changed_documents(self, since: datetime) -> List[Document]:
        """Get documents modified since a specific datetime."""
        try:
            self.logger.info(f"Retrieving documents modified since: {since}")

            # For incremental sync, we'd need to filter by modification time
            # This is a simplified implementation - you might want to optimize this
            all_documents = self.extract()

            changed_documents = []
            for doc in all_documents:
                modified_str = doc.metadata.get("modified_datetime")
                if modified_str:
                    # Parse ISO datetime string
                    modified_dt = datetime.fromisoformat(modified_str.replace("Z", "+00:00"))
                    if modified_dt > since:
                        changed_documents.append(doc)

            self.logger.info(f"Found {len(changed_documents)} changed documents")
            return changed_documents

        except Exception as e:
            self.logger.error(f"Failed to get changed documents: {e}")
            return []

    def get_documents_by_ids(self, document_ids: List[str]) -> List[Document]:
        """Get specific documents by their IDs."""
        try:
            self.logger.info(f"Retrieving {len(document_ids)} specific documents")

            documents = []

            for doc_id in document_ids:
                try:
                    # Get item by ID across all drives
                    libraries = self._get_document_libraries()

                    for library in libraries:
                        try:
                            endpoint = f"/drives/{library['id']}/items/{doc_id}"
                            item = self._make_graph_request(endpoint)

                            if "file" in item:
                                doc = self._convert_item_to_document(item, library)
                                if doc:
                                    documents.append(doc)
                                    break

                        except Exception:
                            continue  # Try next library

                except Exception as e:
                    self.logger.warning(f"Failed to get document {doc_id}: {e}")
                    continue

            self.logger.info(f"Retrieved {len(documents)} specific documents")
            return documents

        except Exception as e:
            self.logger.error(f"Failed to get documents by IDs: {e}")
            return []

    def _supports_incremental(self) -> bool:
        """Check if connector supports incremental sync."""
        return True

    def _supports_specific(self) -> bool:
        """Check if connector supports specific document retrieval."""
        return True
