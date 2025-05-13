"""
Module for loading and managing the document database for the Confluence Assistant.

This module handles the loading of documents from Confluence, processing them,
and storing them in a vector database for efficient retrieval.
"""

import os
import sys
import logging
import shutil
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logger = logging.getLogger(__name__)

# Third-party imports
try:
    from atlassian import Confluence
    from bs4 import BeautifulSoup
    import html2text
    from langchain_community.document_loaders import ConfluenceLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    IMPORT_ERROR = None
except ImportError as e:
    logger.warning("Failed to import required modules: %s", str(e))
    IMPORT_ERROR = e

# Local imports
from config import (
    CONFLUENCE_SPACE_NAME, 
    CONFLUENCE_SPACE_KEY,
    CONFLUENCE_USERNAME, 
    CONFLUENCE_API_KEY, 
    PERSIST_DIRECTORY
)

# Import error handling utilities
from utils import (
    handle_errors,
    retry_api_call,
    GracefulDegradation,
    VectorStoreError,
    ConfluenceConnectionError
)

class DataLoader:
    """
    A class to handle loading and managing the document database from Confluence.
    
    This class provides functionality to load documents from Confluence, process them,
    and store them in a vector database for efficient retrieval. It includes error handling,
    retry mechanisms, and graceful degradation for robustness.
    """
    
    def __init__(
            self,
            space_key: str = CONFLUENCE_SPACE_KEY,
            space_name: str = CONFLUENCE_SPACE_NAME,
            username: str = CONFLUENCE_USERNAME,
            api_key: str = CONFLUENCE_API_KEY,
            persist_directory: str = PERSIST_DIRECTORY,
            max_results: int = 100,
            enable_graceful_degradation: bool = True
    ) -> None:
        """
        Initialize the DataLoader with Confluence connection details.

        Args:
            space_key: The key of the Confluence space to load documents from.
            space_name: The name of the Confluence space (for display purposes).
            username: Username for Confluence API authentication.
            api_key: API key for Confluence authentication.
            persist_directory: Directory to store the vector database.
            max_results: Maximum number of documents to load from Confluence.
            enable_graceful_degradation: If True, enables fallback mechanisms on errors.
            
        Raises:
            ImportError: If required dependencies are not installed.
            ValueError: If required parameters are missing or invalid.
        """
        logger.info("Initializing DataLoader for Confluence space: %s (%s)", 
                   space_name, space_key)
                   
        # Check for import errors
        if IMPORT_ERROR is not None:
            error_msg = f"Failed to initialize DataLoader due to missing dependencies: {IMPORT_ERROR}"
            if not enable_graceful_degradation:
                raise ImportError(error_msg) from IMPORT_ERROR
            logger.warning("%s. Some features may be limited.", error_msg)
        
        # Validate required parameters
        if not all([space_key, space_name, username, api_key]):
            error_msg = (
                "Missing required Confluence credentials. "
                "Please provide space_key, space_name, username, and api_key."
            )
            if not enable_graceful_degradation:
                raise ValueError(error_msg)
            logger.warning("%s Using empty database.", error_msg)
        
        self.space_key = space_key
        self.space_name = space_name
        self.username = username
        self.api_key = api_key
        self.persist_directory = Path(persist_directory)
        self.max_results = max_results
        self.enable_graceful_degradation = enable_graceful_degradation
        
        # Initialize Confluence client
        self.confluence = self._init_confluence_client()
        
        logger.info("DataLoader initialized successfully for space: %s", space_name)
    
    def _init_confluence_client(self):
        """Initialize and return the Confluence client with error handling."""
        try:
            return Confluence(
                url='https://cern.ch/atlas-software',
                username=self.username,
                password=self.api_key,
                cloud=True
            )
        except Exception as e:
            error_msg = f"Failed to initialize Confluence client: {str(e)}"
            if not self.enable_graceful_degradation:
                raise ConfluenceConnectionError(error_msg) from e
            logger.error("%s Some functionality will be limited.", error_msg)
            return None
    
    @retry_api_call(max_retries=3, initial_wait=2, max_wait=10.0)
    @handle_errors(
        fallback_return_value=[],
        log_level=logging.ERROR,
        include_traceback=True
    )
    def load_from_confluence_loader(self) -> List[Document]:
        """
        Load documents from Confluence using the Atlassian API with retry logic.
        
        This method attempts to load documents from Confluence with configurable retry logic.
        It includes detailed logging and error handling with graceful degradation.
        
        Returns:
            List[Document]: A list of loaded documents, or an empty list if loading fails
            and graceful degradation is enabled.
            
        Raises:
            ConfluenceConnectionError: If loading fails and graceful degradation is disabled.
        """
        logger.info("Starting document loading from Confluence space: %s", self.space_key)
        
        # Check if client initialization failed
        if self.confluence is None:
            error_msg = "Cannot load documents: Confluence client not initialized"
            if not self.enable_graceful_degradation:
                raise ConfluenceConnectionError(error_msg)
            logger.warning("%s Returning empty document list.", error_msg)
            return []
        
        try:
            # Log connection details (without sensitive information)
            logger.info(
                "Connecting to Confluence - Space: %s, User: %s",
                self.space_name, self.username
            )
            
            # Initialize the Confluence loader with error handling
            try:
                loader = ConfluenceLoader(
                    url='https://cern.ch/atlas-software',  # Hardcoded URL
                    username=self.username,
                    api_key=self.api_key,
                    space_key=self.space_key,
                    max_pages=self.max_results
                )
            except Exception as e:
                error_msg = f"Failed to initialize ConfluenceLoader: {str(e)}"
                logger.error(error_msg)
                if not self.enable_graceful_degradation:
                    raise ConfluenceConnectionError(error_msg) from e
                return []
            
            # Load documents with progress tracking
            logger.info("Loading documents from Confluence...")
            start_time = time.time()
            
            try:
                documents = loader.load()
                load_time = time.time() - start_time
                
                if not documents:
                    logger.warning("No documents were loaded from Confluence.")
                else:
                    logger.info(
                        "Successfully loaded %d documents in %.2f seconds",
                        len(documents), load_time
                    )
                
                return documents
                
            except Exception as e:
                error_msg = f"Error loading documents from Confluence: {str(e)}"
                logger.error(error_msg, exc_info=True)
                if not self.enable_graceful_degradation:
                    raise ConfluenceConnectionError(error_msg) from e
                return []
            
            # Récupérer toutes les pages de l'espace avec pagination
            print(f"Récupération des pages de l'espace {self.space_key}...")
            
            # Utiliser la pagination pour récupérer toutes les pages
            # La méthode get_all_pages_from_space peut avoir des limites
            start = 0
            limit = 100  # Nombre maximum de pages à récupérer par requête
            all_pages = []
            
            while True:
                # Récupérer un lot de pages
                batch = confluence.get_all_pages_from_space(self.space_key, start=start, limit=limit, expand='version')
                
                if not batch:
                    break  # Plus de pages à récupérer
                    
                all_pages.extend(batch)
                print(f"  Récupéré {len(all_pages)} pages jusqu'à présent...")
                
                # Mettre à jour l'index de départ pour la prochaine requête
                start += len(batch)
                
                # Si le nombre de pages récupérées est inférieur à la limite, nous avons tout récupéré
                if len(batch) < limit:
                    break
            
            pages = all_pages
            print(f"Récupération réussie! {len(pages)} pages trouvées au total.")
            
            # Récupérer également les pages d'enfants (sous-pages) si nécessaire
            if len(pages) > 0:
                print("Recherche de sous-pages supplémentaires...")
                child_pages = []
                
                for page in pages:
                    page_id = page.get('id')
                    # Récupérer les enfants de cette page
                    try:
                        children = confluence.get_page_child_by_type(page_id, type='page')
                        if children and len(children) > 0:
                            child_pages.extend(children)
                    except Exception as e:
                        print(f"Erreur lors de la récupération des sous-pages pour {page.get('title', 'Sans titre')}: {str(e)}")
                
                # Ajouter les sous-pages à notre liste principale (en évitant les doublons)
                existing_ids = {p.get('id') for p in pages}
                new_child_pages = [p for p in child_pages if p.get('id') not in existing_ids]
                
                if new_child_pages:
                    pages.extend(new_child_pages)
                    print(f"Ajout de {len(new_child_pages)} sous-pages supplémentaires. Total: {len(pages)} pages.")
            
            
            # Convertir les pages en documents LangChain
            docs = []
            h = html2text.HTML2Text()
            h.ignore_links = False
            h.ignore_images = False
            h.ignore_tables = False
            h.body_width = 0  # Pas de limite de largeur pour conserver la mise en forme
            
            # Nombre total de pages à traiter
            total_pages = len(pages)
            print(f"Traitement du contenu de {total_pages} pages...")
            
            for i, page in enumerate(pages):
                try:
                    # Récupérer les informations de base de la page
                    page_id = page.get('id')
                    page_title = page.get('title', 'Sans titre')
                    
                    # Afficher la progression régulièrement
                    if i % 10 == 0 or i == total_pages - 1:
                        print(f"Traitement de la page {i+1}/{total_pages}: {page_title}")
                    
                    # Récupérer le contenu complet de la page avec les propriétés et les pièces jointes
                    page_data = confluence.get_page_by_id(
                        page_id, 
                        expand='body.storage,history,space,version,ancestors,children.page,children.attachment,metadata.properties'
                    )
                    
                    # Extraire le contenu HTML
                    content = page_data.get('body', {}).get('storage', {}).get('value', '')
                    
                    # Informations supplémentaires pour enrichir les métadonnées
                    space_info = page_data.get('space', {})
                    version_info = page_data.get('version', {})
                    ancestors = page_data.get('ancestors', [])
                    last_updated = version_info.get('when', '')
                    creator = version_info.get('by', {}).get('displayName', '')
                    
                    # Construire un chemin de navigation (breadcrumb)
                    breadcrumb = ' > '.join([a.get('title', '') for a in ancestors] + [page_title])
                    
                    # Convertir HTML en texte
                    text_content = h.handle(content)
                    
                    # Ajouter des informations structurées au début du contenu pour améliorer la recherche
                    structured_header = f"# {page_title}\n\n"
                    if breadcrumb:
                        structured_header += f"**Chemin:** {breadcrumb}\n\n"
                    if creator:
                        structured_header += f"**Auteur:** {creator}\n\n"
                    if last_updated:
                        structured_header += f"**Dernière mise à jour:** {last_updated}\n\n"
                    
                    structured_header += "---\n\n"
                    
                    # Combiner l'en-tête structurée avec le contenu
                    enhanced_content = structured_header + text_content
                    
                    # Créer un document LangChain avec des métadonnées enrichies
                    doc = Document(
                        page_content=enhanced_content,
                        metadata={
                            'source': f"{base_url}/wiki/spaces/{self.space_key}/pages/{page_id}",
                            'title': page_title,
                            'id': page_id,
                            'space_name': space_info.get('name', ''),
                            'space_key': space_info.get('key', ''),
                            'last_updated': last_updated,
                            'creator': creator,
                            'breadcrumb': breadcrumb,
                            'url': f"{base_url}/wiki/spaces/{self.space_key}/pages/{page_id}"
                        }
                    )
                    docs.append(doc)
                    
                except Exception as page_error:
                    print(f"Erreur lors du traitement de la page {page_title}: {str(page_error)}")
                    import traceback
                    print(traceback.format_exc())
            
            print(f"Chargement réussi: {len(docs)} documents récupérés")
            return docs
            
        except Exception as e:
            print(f"\n==== ERREUR DE CONNEXION CONFLUENCE ====")
            print(f"Type d'erreur: {type(e).__name__}")
            print(f"Message d'erreur: {str(e)}")
            # Pour le débogage, affichons la trace complète
            import traceback
            print("Trace d'erreur complète:")
            print(traceback.format_exc())
            raise e

    @handle_errors(
        fallback_return_value=[],
        log_level=logging.ERROR,
        include_traceback=True
    )
    def split_docs(self, docs):
        """
        Split documents into smaller chunks using a text splitter.
        
        Args:
            docs: List of documents to be split
            
        Returns:
            List[Document]: List of document chunks
            
        Raises:
            ValueError: If input documents are invalid
        """
        if not docs:
            logger.warning("No documents provided to split")
            return []
            
        logger.info("Splitting %d documents into chunks...", len(docs))
        
        try:
            # Configure text splitter with appropriate parameters
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                is_separator_regex=False,
            )
            
            # Split documents into chunks
            chunks = text_splitter.split_documents(docs)
            
            logger.info(
                "Split %d documents into %d chunks", 
                len(docs), len(chunks)
            )
            
            return chunks
            
        except Exception as e:
            error_msg = f"Error splitting documents: {str(e)}"
            logger.error(error_msg, exc_info=True)
            if not self.enable_graceful_degradation:
                raise ValueError(error_msg) from e
            return []

    @handle_errors(
        fallback_return_value=None,
        log_level=logging.ERROR,
        include_traceback=True
    )
    def save_to_db(self, splitted_docs, embeddings):
        """
        Create a vector store from documents using FAISS and OpenAI embeddings.
        
        Args:
            splitted_docs: List of documents to index
            embeddings: Embeddings to use for indexing
            
        Returns:
            Optional[FAISS]: The created vector store, or None if creation failed
            
        Raises:
            VectorStoreError: If vector store creation fails and graceful degradation is disabled
        """
        if not splitted_docs:
            error_msg = "Cannot create vector store: No documents provided"
            logger.warning(error_msg)
            if not self.enable_graceful_degradation:
                raise VectorStoreError(error_msg)
            return None
            
        logger.info("Creating vector store from %d documents...", len(splitted_docs))
        
        try:
            # Create persistence directory if it doesn't exist
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Initialize embeddings
            from langchain_openai import OpenAIEmbeddings
            embeddings = OpenAIEmbeddings()
            
            # Create and save the vector store
            vectorstore = FAISS.from_documents(splitted_docs, embeddings)
            vectorstore.save_local(self.persist_directory)
            
            logger.info(
                "Successfully created vector store with %d documents at %s",
                len(splitted_docs), self.persist_directory
            )
            
            return vectorstore
            
        except Exception as e:
            error_msg = f"Failed to create vector store: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Clean up partially created files
            self._cleanup_failed_vectorstore()
            
            if not self.enable_graceful_degradation:
                raise VectorStoreError(error_msg) from e
            return None
    
    def _cleanup_failed_vectorstore(self):
        """Clean up files from a failed vector store creation"""
        try:
            for ext in ['.faiss', '.pkl']:
                file_path = os.path.join(self.persist_directory, f'index{ext}')
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.debug("Removed partial vector store file: %s", file_path)
        except Exception as e:
            logger.warning("Error during vector store cleanup: %s", str(e))
    
    @handle_errors(
        fallback_return_value=None,
        log_level=logging.ERROR,
        include_traceback=True
    )
    def load_from_db(self, embeddings):
        """
        Load an existing vector store from disk.
        
        Args:
            embeddings: The embeddings to use for the vector store
            
        Returns:
            Optional[FAISS]: The loaded vector store, or None if loading fails
            
        Raises:
            VectorStoreError: If vector store loading fails and graceful degradation is disabled
        """
        index_path = os.path.join(self.persist_directory, "index.faiss")
        
        if not os.path.exists(index_path):
            error_msg = f"No vector store found at {self.persist_directory}"
            logger.warning(error_msg)
            if not self.enable_graceful_degradation:
                raise VectorStoreError(error_msg)
            return None
            
        logger.info("Loading vector store from %s", self.persist_directory)
        
        try:
            # Load the vector store with safety checks
            db = FAISS.load_local(
                self.persist_directory, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            
            logger.info(
                "Successfully loaded vector store with %d documents", 
                len(db.docstore._dict)
            )
            
            return db
            
        except Exception as e:
            error_msg = f"Failed to load vector store: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            if not self.enable_graceful_degradation:
                raise VectorStoreError(error_msg) from e
            return None

    @handle_errors(
        fallback_return_value=[],
        log_level=logging.ERROR,
        include_traceback=True
    )
    def create_dummy_docs(self) -> List[Document]:
        """
        Create a set of dummy documents to allow the application to start
        when the real data cannot be loaded.
        
        This is used as a fallback when Confluence connection fails and
        graceful degradation is enabled.
        
        Returns:
            List[Document]: A list containing a single dummy document
        """
        logger.warning(
            "Creating dummy documents. The application will start with limited functionality. "
            "Please check your Confluence connection settings."
        )
        
        try:
            from langchain_core.documents import Document
            
            dummy_docs = [
                Document(
                    page_content="""
                    This is a test document for application startup.
                    The application could not connect to Confluence or load documents.
                    Please check your connection and credentials.
                    
                    For more information, please contact support:
                    - Email: support@example.com
                    - Phone: +1 (555) 123-4567
                    """,
                    metadata={
                        'title': 'Test Document',
                        'source': 'dummy',
                        'page_id': 'dummy-001',
                        'space_key': 'DUMMY',
                        'is_dummy': True
                    }
                )
            ]
            
            logger.info("Created %d dummy document(s)", len(dummy_docs))
            return dummy_docs
            
        except Exception as e:
            error_msg = f"Failed to create dummy documents: {str(e)}"
            logger.error(error_msg, exc_info=True)
            # Even if creating dummy docs fails, return an empty list
            # to avoid crashing the application
            return []

    @handle_errors(
        fallback_return_value=[],
        log_level=logging.ERROR,
        include_traceback=True
    )
    def create_dummy_docs(self) -> List[Document]:
        """
        Create a set of dummy documents to allow the application to start
        when the real data cannot be loaded.
    
        This is used as a fallback when Confluence connection fails and
        graceful degradation is enabled.
    
        Returns:
            List[Document]: A list containing a single dummy document
        """
        logger.warning(
            "Creating dummy documents. The application will start with limited functionality. "
            "Please check your Confluence connection settings."
        )
        
        try:
            from langchain_core.documents import Document
            
            dummy_docs = [
                Document(
                    page_content="""
                    This is a test document for application startup.
                    The application could not connect to Confluence or load documents.
                    Please check your connection and credentials.
                    
                    For more information, please contact support:
                    - Email: support@example.com
                    - Phone: +1 (555) 123-4567
                    """,
                    metadata={
                        'title': 'Test Document',
                        'source': 'dummy',
                        'page_id': 'dummy-001',
                        'space_key': 'DUMMY',
                        'is_dummy': True
                    }
                )
            ]
            
            logger.info("Created %d dummy document(s)", len(dummy_docs))
            return dummy_docs
            
        except Exception as e:
            error_msg = f"Failed to create dummy documents: {str(e)}"
            logger.error(error_msg, exc_info=True)
            # Even if creating dummy docs fails, return an empty list
            # to avoid crashing the application
            return []

    def set_db(self, embeddings):
        """Create, save, and load db"""
        try:
            shutil.rmtree(self.persist_directory)
        except Exception as e:
            error_msg = f"Error during document loading: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            if not self.enable_graceful_degradation:
                raise VectorStoreError(error_msg) from e
            
            # Try to create dummy documents as a fallback
            logger.info("Attempting to create dummy documents as fallback...")
            documents = self.create_dummy_docs()
            
            if not documents:
                error_msg = "Graceful degradation failed: Could not create dummy documents"
                logger.error(error_msg)
                if not self.enable_graceful_degradation:
                    raise VectorStoreError(error_msg)
                return None
            
            # Split documents into chunks
            try:
                logger.info("Splitting documents into chunks...")
                splitted_docs = self.split_docs(documents)
                
                if not splitted_docs:
                    error_msg = "No document chunks were created"
                    logger.error(error_msg)
                    if not self.enable_graceful_degradation:
                        raise VectorStoreError(error_msg)
                    return None
                
            except Exception as e:
                error_msg = f"Error splitting documents: {str(e)}"
                logger.error(error_msg, exc_info=True)
                if not self.enable_graceful_degradation:
                    raise VectorStoreError(error_msg) from e
                return None
            
            # Create and save vector store
            try:
                logger.info("Creating and saving vector store...")
                db = self.save_to_db(splitted_docs, embeddings)
                
                if db is None:
                    error_msg = "Failed to create vector store"
                    logger.error(error_msg)
                    if not self.enable_graceful_degradation:
                        raise VectorStoreError(error_msg)
                    return None
                
                return db
                
            except Exception as e:
                error_msg = f"Error creating vector store: {str(e)}"
                logger.error(error_msg, exc_info=True)
                
                # Clean up any partially created files
                self._cleanup_failed_vectorstore()
                
                if not self.enable_graceful_degradation:
                    raise VectorStoreError(error_msg) from e
                return None

    @handle_errors(
        fallback_return_value=None,
        log_level=logging.ERROR,
        include_traceback=True
    )
    def get_db(self, embeddings):
        """
        Get an existing vector store or create a new one if it doesn't exist.
        
        This method first attempts to load an existing vector store. If that fails,
        it will create a new one by loading documents from Confluence.
        
        Args:
            embeddings: The embeddings to use for the vector store
                
        Returns:
            Optional[FAISS]: The loaded or created vector store, or None if both operations fail
                
        Raises:
            VectorStoreError: If both loading and creation fail and graceful degradation is disabled
        """
        logger.info("Attempting to load existing vector store...")
        
        # Try to load existing vector store
        try:
            db = self.load_from_db(embeddings)
            if db is not None:
                logger.info("Successfully loaded existing vector store")
                return db
                
            logger.info("No existing vector store found. Creating a new one...")
            
        except Exception as e:
            error_msg = f"Error loading existing vector store: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            if not self.enable_graceful_degradation:
                raise VectorStoreError(error_msg) from e
                
            logger.info("Attempting to create a new vector store...")
        
        # If we get here, either loading failed or no store exists yet
        try:
            db = self.set_db(embeddings)
            if db is not None:
                logger.info("Successfully created new vector store")
                return db
                
            error_msg = "Failed to create a new vector store"
            logger.error(error_msg)
            
            if not self.enable_graceful_degradation:
                raise VectorStoreError(error_msg)
                
            return None
            
        except Exception as e:
            error_msg = f"Failed to create vector store: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # As a last resort, try to create dummy documents
            try:
                logger.warning("All recovery attempts failed. Trying to create dummy documents...")
                dummy_docs = self.create_dummy_docs()
                if dummy_docs:
                    logger.info("Creating vector store from dummy documents...")
                    return self.save_to_db(dummy_docs, embeddings)
                    
            except Exception as inner_e:
                logger.error("Failed to create dummy documents: %s", str(inner_e), exc_info=True)
            
            if not self.enable_graceful_degradation:
                raise VectorStoreError(
                    "All attempts to load or create a vector store failed. "
                    "Please check your configuration and try again."
                ) from e
                
            return None

if __name__ == "__main__":
    pass
