import logging
from datetime import datetime
from typing import List, Dict, Any
from llama_index.readers.confluence import ConfluenceReader


from src.ingestion.connectors.base_connector import BaseConnector
from src.ingestion.connectors.metadata_enricher import ConfluenceMetadataEnricher
from src.core.interfaces import Document


class ConfluenceConnector(BaseConnector):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        self._setup_connection_config(config)
        self._validate_required_config()
        self.reader = self._create_reader()

        self.enricher = None
        self._setup_metadata_enricher()

    def _setup_connection_config(self, config: Dict[str, Any]) -> None:
        space_name = config.get("confluence_space_name", "")
        self.base_url = f"{space_name}/wiki"

        self.username = config.get("confluence_email_address")
        self.api_token = config.get("confluence_private_api_key")
        self.space_key = config.get("confluence_space_key")

        self.include_attachments = None

    def _setup_metadata_enricher(self) -> None:
        """Configure l'enrichisseur de métadonnées"""
        try:
            # Récupérer l'URL de base sans /wiki pour l'enrichisseur
            base_url_for_enricher = self.config.get("confluence_space_name", "")

            self.enricher = ConfluenceMetadataEnricher(
                base_url=base_url_for_enricher, username=self.username, api_token=self.api_token
            )

            # Tester la connexion
            if self.enricher.test_connection():
                self.logger.info("Enrichisseur de métadonnées initialisé avec succès")
            else:
                self.logger.warning("Enrichisseur de métadonnées non disponible (API inaccessible)")
                self.enricher = None

        except Exception as e:
            self.logger.warning(f"Impossible d'initialiser l'enrichisseur de métadonnées: {e}")
            self.enricher = None

    def _validate_required_config(self) -> None:
        if not self.base_url:
            raise ValueError("Missing confluence_url or confluence_space_name")

        if not self.space_key:
            raise ValueError("Missing confluence_space_key")

        if not (self.username and self.api_token):
            raise ValueError("Missing authentication: username and api_token required")

    def _create_reader(self):
        return ConfluenceReader(
            base_url=self.base_url,
            user_name=self.username,
            password=self.api_token,
            cloud=True,
            client_args={"backoff_and_retry": True},
        )

    def extract(self) -> List[Document]:
        try:
            self.logger.info(f"Starting extraction from Confluence space: {self.space_key}")

            cql = f'space = "{self.space_key}" and type = "page"'
            llamaindex_docs = self.reader.load_data(
                cql=cql,
                include_attachments=self.include_attachments,
            )

            documents = self._convert_llamaindex_documents(llamaindex_docs)

            self.logger.info(f"Extraction completed: {len(documents)} documents")
            return documents

        except Exception as e:
            self.logger.error(f"Extraction failed: {str(e)}")
            return []

    def validate_connection(self) -> bool:
        try:
            self.reader.load_data(space_key=self.space_key, max_num_results=1, page_status="current")
            self.logger.info(f"Connection validated for space: {self.space_key}")
            return True
        except Exception as e:
            self.logger.error(f"Connection validation failed: {e}")
            return False

    def get_changed_documents(self, since: datetime) -> List[Document]:
        try:
            since_str = since.strftime("%Y-%m-%d")
            self.logger.info(f"Retrieving pages modified since: {since_str}")

            cql = f'space = "{self.space_key}" and type = "page" and lastmodified >= "{since_str}"'

            llamaindex_docs = self.reader.load_data(
                cql=cql,
                include_attachments=self.include_attachments,
            )

            documents = self._convert_llamaindex_documents(llamaindex_docs)

            self.logger.info(f"Retrieved {len(documents)} changed documents")
            return documents

        except Exception as e:
            self.logger.error(f"Failed to get changed documents: {e}")
            return []

    def get_documents_by_ids(self, document_ids: List[str]) -> List[Document]:
        try:
            self.logger.info(f"Retrieving {len(document_ids)} specific pages")

            llamaindex_docs = self.reader.load_data(
                page_ids=document_ids,
                include_attachments=self.include_attachments,
                include_children=self.include_children,
            )

            documents = self._convert_llamaindex_documents(llamaindex_docs)

            self.logger.info(f"Retrieved {len(documents)} specific documents")
            return documents

        except Exception as e:
            self.logger.error(f"Failed to get documents by IDs: {e}")
            return []

    def _supports_incremental(self) -> bool:
        return True

    def _supports_specific(self) -> bool:
        return True

    def _convert_llamaindex_documents(self, llamaindex_docs) -> List[Document]:
        documents = []

        for doc in llamaindex_docs:
            try:
                metadata = doc.metadata.copy()

                # Métadonnées de base
                metadata.update(
                    {
                        "source": "confluence",
                        "space_key": self.space_key,
                        "content_type": metadata.get("content_type", "markdown"),
                        "content_length": len(doc.text),
                    }
                )

                if self.enricher:
                    try:
                        metadata = self.enricher.enrich_document_metadata(metadata)
                        self.logger.debug(f"Métadonnées enrichies pour la page {metadata.get('page_id', 'unknown')}")
                    except Exception as e:
                        self.logger.warning(f"Échec de l'enrichissement des métadonnées: {e}")

                documents.append(Document(content=doc.text, metadata=metadata))
            except Exception as e:
                self.logger.warning(f"Failed to convert document: {e}")

        if self.enricher:
            self.logger.info(f"Converted {len(documents)} documents avec métadonnées enrichies")
        else:
            self.logger.info(f"Converted {len(documents)} documents")
        return documents
