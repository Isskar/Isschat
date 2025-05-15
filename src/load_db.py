import sys
import logging
import shutil
from pathlib import Path

# Add the parent directory to the Python search path
sys.path.append(str(Path(__file__).parent.parent))

# Import variables from config.py
from config import (
    CONFLUENCE_SPACE_NAME,
    CONFLUENCE_SPACE_KEY,
    CONFLUENCE_USERNAME,
    CONFLUENCE_API_KEY,
    PERSIST_DIRECTORY,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import FAISS


class DataLoader:
    """Create, load, save the DB using the confluence Loader"""

    def __init__(
        self,
        confluence_url=CONFLUENCE_SPACE_NAME,
        username=CONFLUENCE_USERNAME,
        api_key=CONFLUENCE_API_KEY,
        space_key=CONFLUENCE_SPACE_KEY,
        persist_directory=PERSIST_DIRECTORY,
    ):
        self.confluence_url = confluence_url
        self.username = username
        self.api_key = api_key
        self.space_key = space_key
        self.persist_directory = persist_directory

    def load_from_confluence_loader(self):
        """Load HTML files from Confluence using direct Atlassian API"""
        try:
            # Configure logging to display messages in the console
            import sys
            from atlassian import Confluence
            from langchain_core.documents import Document
            import html2text

            logging.basicConfig(
                level=logging.INFO,
                stream=sys.stdout,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )

            # Display connection information for debugging (without the API key)
            print("\n==== CONFLUENCE CONNECTION DETAILS ====")
            print(f"URL: {self.confluence_url}")
            print(f"Username: {self.username}")
            print(f"Space Key: {self.space_key}")
            print(f"API Key: {'*' * 5}{self.api_key[-5:] if self.api_key else 'Not defined'}")

            # Ensure the URL is in the correct format (without the specific path)
            base_url = self.confluence_url
            if "/wiki" in base_url:
                base_url = base_url.split("/wiki")[0]
                print(f"Adjusted URL: {base_url}")

            # Verify that the parameters are correct
            if not base_url or not base_url.startswith("http"):
                raise ValueError(f"Invalid Confluence URL: {base_url}")
            if not self.username:
                raise ValueError("Missing Confluence username")
            if not self.api_key:
                raise ValueError("Missing Confluence API key")
            if not self.space_key:
                raise ValueError("Missing Confluence space key")

            print("\n==== ATTEMPTING CONNECTION TO CONFLUENCE ====")
            # Create a Confluence instance with the Atlassian API
            confluence = Confluence(
                url=base_url,
                username=self.username,
                password=self.api_key,
                cloud=True,  # Specify that it's a cloud instance
            )

            # Retrieve all pages from the space with pagination
            print(f"Retrieving pages from space {self.space_key}...")

            # Use pagination to retrieve all pages
            # The get_all_pages_from_space method may have limitations
            start = 0
            limit = 100  # Maximum number of pages to retrieve per request
            all_pages = []

            while True:
                # Retrieve a batch of pages
                batch = confluence.get_all_pages_from_space(self.space_key, start=start, limit=limit, expand="version")

                if not batch:
                    break  # No more pages to retrieve

                all_pages.extend(batch)
                print(f"  Retrieved {len(all_pages)} pages so far...")

                # Update the starting index for the next request
                start += len(batch)

                # If the number of pages retrieved is less than the limit, we've retrieved everything
                if len(batch) < limit:
                    break

            pages = all_pages
            print(f"Retrieval successful! {len(pages)} pages found in total.")

            # Also retrieve child pages (sub-pages) if necessary
            if len(pages) > 0:
                print("Searching for additional sub-pages...")
                child_pages = []

                for page in pages:
                    page_id = page.get("id")
                    # Retrieve the children of this page
                    try:
                        children = confluence.get_page_child_by_type(page_id, type="page")
                        if children and len(children) > 0:
                            child_pages.extend(children)
                    except Exception as e:
                        print(f"Error retrieving sub-pages for {page.get('title', 'Untitled')}: {str(e)}")

                # Add sub-pages to our main list (avoiding duplicates)
                existing_ids = {p.get("id") for p in pages}
                new_child_pages = [p for p in child_pages if p.get("id") not in existing_ids]

                if new_child_pages:
                    pages.extend(new_child_pages)
                    print(f"Added {len(new_child_pages)} additional sub-pages. Total: {len(pages)} pages.")

            # Convert pages to LangChain documents
            docs = []
            h = html2text.HTML2Text()
            h.ignore_links = False
            h.ignore_images = False
            h.ignore_tables = False
            h.body_width = 0  # No width limit to preserve formatting

            # Total number of pages to process
            total_pages = len(pages)
            print(f"Processing content of {total_pages} pages...")

            for i, page in enumerate(pages):
                try:
                    # Retrieve basic information about the page
                    page_id = page.get("id")
                    page_title = page.get("title", "Untitled")

                    # Display progress regularly
                    if i % 10 == 0 or i == total_pages - 1:
                        print(f"Processing page {i + 1}/{total_pages}: {page_title}")

                    # Retrieve the complete content of the page with properties and attachments
                    page_data = confluence.get_page_by_id(
                        page_id,
                        expand="body.storage,history,space,version,ancestors,children.page,children.attachment,metadata.properties",
                    )

                    # Extract HTML content
                    content = page_data.get("body", {}).get("storage", {}).get("value", "")

                    # Additional information to enrich metadata
                    space_info = page_data.get("space", {})
                    version_info = page_data.get("version", {})
                    ancestors = page_data.get("ancestors", [])
                    last_updated = version_info.get("when", "")
                    creator = version_info.get("by", {}).get("displayName", "")

                    # Build a navigation path (breadcrumb)
                    breadcrumb = " > ".join([a.get("title", "") for a in ancestors] + [page_title])

                    # Convert HTML to text
                    text_content = h.handle(content)

                    # Add structured information at the beginning of the content to improve search
                    structured_header = f"# {page_title}\n\n"
                    if breadcrumb:
                        structured_header += f"**Path:** {breadcrumb}\n\n"
                    if creator:
                        structured_header += f"**Author:** {creator}\n\n"
                    if last_updated:
                        structured_header += f"**Last updated:** {last_updated}\n\n"

                    structured_header += "---\n\n"

                    # Combine the structured header with the content
                    enhanced_content = structured_header + text_content

                    # Create a LangChain document with enriched metadata
                    doc = Document(
                        page_content=enhanced_content,
                        metadata={
                            "source": f"{base_url}/wiki/spaces/{self.space_key}/pages/{page_id}",
                            "title": page_title,
                            "id": page_id,
                            "space_name": space_info.get("name", ""),
                            "space_key": space_info.get("key", ""),
                            "last_updated": last_updated,
                            "creator": creator,
                            "breadcrumb": breadcrumb,
                            "url": f"{base_url}/wiki/spaces/{self.space_key}/pages/{page_id}",
                        },
                    )
                    docs.append(doc)

                except Exception as page_error:
                    print(f"Erreur lors du traitement de la page {page_title}: {str(page_error)}")
                    import traceback

                    print(traceback.format_exc())

            print(f"Loading successful: {len(docs)} documents retrieved")
            return docs

        except Exception as e:
            print("\n==== CONFLUENCE CONNECTION ERROR ====")
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
            separators=["\n\n", "\n", r"(?<=\. )", " ", ""],
        )

        splitted_docs = splitter.split_documents(md_docs)
        return splitted_docs

    def save_to_db(self, splitted_docs, embeddings):
        """Save chunks to Chroma DB"""
        db = FAISS.from_documents(splitted_docs, embeddings)
        db.save_local(self.persist_directory)
        return db

    def load_from_db(self, embeddings):
        """Loader chunks to Chroma DB"""
        db = FAISS.load_local(
            self.persist_directory,
            embeddings,
            allow_dangerous_deserialization=True,  # Necessary for recent versions of LangChain
        )
        return db

    def create_dummy_docs(self):
        """Creates a dummy dataset to allow the application to start"""
        from langchain_core.documents import Document

        logging.warning("Creating dummy dataset to allow the application to start")

        dummy_docs = [
            Document(
                page_content="Welcome to the Confluence Assistant. This database "
                "is a dummy version created because the connection to Confluence failed.",
                metadata={"source": "dummy_doc_1", "title": "Welcome"},
            ),
            Document(
                page_content="To use the assistant with real data, check your Confluence "
                "connection settings in the .env file.",
                metadata={"source": "dummy_doc_2", "title": "Configuration"},
            ),
            Document(
                page_content="Make sure the variables CONFLUENCE_SPACE_NAME, CONFLUENCE_SPACE_KEY, "
                "CONFLUENCE_USERNAME and CONFLUENCE_PRIVATE_API_KEY are properly configured.",
                metadata={"source": "dummy_doc_3", "title": "Environment Variables"},
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
            logging.error(f"Error when loading from the Confluence {str(e)}")
            logging.warning("Using a dummy dataset to allow the application to start")
            # Create a dummy dataset
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
