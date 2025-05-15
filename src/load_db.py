import sys
import logging
import shutil
import os
from pathlib import Path

# Ajouter le répertoire parent au chemin de recherche Python
sys.path.append(str(Path(__file__).parent.parent))

# Importer les variables depuis config.py
from config import (CONFLUENCE_SPACE_NAME, CONFLUENCE_SPACE_KEY,
                    CONFLUENCE_USERNAME, CONFLUENCE_API_KEY, PERSIST_DIRECTORY)

from langchain_community.document_loaders import ConfluenceLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import FAISS
class DataLoader():
    """Create, load, save the DB using the confluence Loader"""
    def __init__(
        self,
        confluence_url=CONFLUENCE_SPACE_NAME,
        username=CONFLUENCE_USERNAME,
        api_key=CONFLUENCE_API_KEY,
        space_key=CONFLUENCE_SPACE_KEY,
        persist_directory=PERSIST_DIRECTORY
    ):

        self.confluence_url = confluence_url
        self.username = username
        self.api_key = api_key
        self.space_key = space_key
        self.persist_directory = persist_directory

    def load_from_confluence_loader(self):
        """Load HTML files from Confluence using direct Atlassian API"""
        try:
            # Configurer le logging pour afficher les messages dans la console
            import sys
            from atlassian import Confluence
            from langchain_core.documents import Document
            from bs4 import BeautifulSoup
            import html2text
            
            logging.basicConfig(level=logging.INFO, stream=sys.stdout, 
                              format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            
            # Afficher les informations de connexion pour le débogage (sans la clé API)
            print(f"\n==== DéTAILS DE CONNEXION CONFLUENCE ====")
            print(f"URL: {self.confluence_url}")
            print(f"Username: {self.username}")
            print(f"Space Key: {self.space_key}")
            print(f"API Key: {'*' * 5}{self.api_key[-5:] if self.api_key else 'Non définie'}")
            
            # S'assurer que l'URL est au bon format (sans le chemin spécifique)
            base_url = self.confluence_url
            if "/wiki" in base_url:
                base_url = base_url.split("/wiki")[0]
                print(f"URL ajustée: {base_url}")
            
            # Vérifier que les paramètres sont corrects
            if not base_url or not base_url.startswith("http"):
                raise ValueError(f"URL Confluence invalide: {base_url}")
            if not self.username:
                raise ValueError("Nom d'utilisateur Confluence manquant")
            if not self.api_key:
                raise ValueError("Clé API Confluence manquante")
            if not self.space_key:
                raise ValueError("Clé d'espace Confluence manquante")
            
            print(f"\n==== TENTATIVE DE CONNEXION À CONFLUENCE ====")
            # Créer une instance Confluence avec l'API Atlassian
            confluence = Confluence(
                url=base_url,
                username=self.username,
                password=self.api_key,
                cloud=True  # Spécifier que c'est une instance cloud
            )
            
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
            
            print(f"Loading successful: {len(docs)} documents retrieved")
            return docs
            
        except Exception as e:
            print(f"\n==== CONFLUENCE CONNECTION ERROR ====")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            # For debugging, let's show the full trace
            import traceback
            print("Trace d'erreur complète:")
            print(traceback.format_exc())
            raise e

    def split_docs(self, docs):
        # Markdown
        headers_to_split_on = [
            ("#", "Titre 1"),
            ("##", "Sous-titre 1"),
            ("###", "Sous-titre 2"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

        # Split based on markdown and add original metadata
        md_docs = []
        for doc in docs:
            md_doc = markdown_splitter.split_text(doc.page_content)
            for i in range(len(md_doc)):
                md_doc[i].metadata = md_doc[i].metadata | doc.metadata
            md_docs.extend(md_doc)

        # RecursiveTextSplitter
        # Chunk size big enough
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=20,
            separators=["\n\n", "\n", r"(?<=\. )", " ", ""]
        )

        splitted_docs = splitter.split_documents(md_docs)
        return splitted_docs

    def save_to_db(self, splitted_docs, embeddings):
        """Save chunks to Chroma DB"""
        from langchain_community.vectorstores import FAISS
        db = FAISS.from_documents(splitted_docs, embeddings)
        db.save_local(self.persist_directory)
        return db

    def load_from_db(self, embeddings):
        """Loader chunks to Chroma DB"""
        from langchain_community.vectorstores import FAISS
        db = FAISS.load_local(
            self.persist_directory, 
            embeddings, 
            allow_dangerous_deserialization=True  # Nécessaire pour les versions récentes de LangChain
        )
        return db

    def create_dummy_docs(self):
        """Creates a dummy dataset to allow the application to start"""
        from langchain_core.documents import Document
        
        logging.warning("Creating dummy dataset to allow the application to start")
        
        dummy_docs = [
            Document(
                page_content="Welcome to the Confluence Assistant. This database is a dummy version created because the connection to Confluence failed.",
                metadata={"source": "dummy_doc_1", "title": "Welcome"}
            ),
            Document(
                page_content="To use the assistant with real data, check your Confluence connection settings in the .env file.",
                metadata={"source": "dummy_doc_2", "title": "Configuration"}
            ),
            Document(
                page_content="Make sure the variables CONFLUENCE_SPACE_NAME, CONFLUENCE_SPACE_KEY, CONFLUENCE_USERNAME and CONFLUENCE_PRIVATE_API_KEY are properly configured.",
                metadata={"source": "dummy_doc_3", "title": "Environment Variables"}
            ),
        ]
        
        return dummy_docs

    def set_db(self, embeddings):
        """Create, save, and load db"""
        try:
            shutil.rmtree(self.persist_directory)
        except Exception as e:
            logging.warning("%s", e)

        try:
            # Load docs from Confluence
            docs = self.load_from_confluence_loader()
        except Exception as e:
            logging.error(f"Erreur lors du chargement depuis Confluence: {str(e)}")
            logging.warning("Utilisation d'un jeu de données factice pour permettre le démarrage de l'application")
            # Créer un jeu de données factice
            docs = self.create_dummy_docs()

        # Split Docs
        splitted_docs = self.split_docs(docs)

        # Save to DB
        db = self.save_to_db(splitted_docs, embeddings)

        return db

    def get_db(self, embeddings):
        """Create, save, and load db"""
        db = self.load_from_db(embeddings)
        return db


if __name__ == "__main__":
    pass
