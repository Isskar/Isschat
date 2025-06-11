import sys
import logging
import time
from pathlib import Path

# Add the parent directory to the Python search path
sys.path.append(str(Path(__file__).parent.parent))

# Import configuration manager from root (avoiding rag_evaluation.config conflict)
from config_import import get_config

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings


class DataLoader:
    """Create, load, save the DB using the confluence Loader"""

    def __init__(self):
        # Get configuration
        config = get_config()

        # Use provided values or fall back to configuration
        self.confluence_url: str = config.confluence_space_name
        self.username: str = config.confluence_email_address
        self.api_key: str = config.confluence_private_api_key
        self.space_key: str = config.confluence_space_key
        self.persist_directory: str = config.persist_directory

    def load_from_confluence_loader(self) -> list:
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
                timeout=30,  # 30 seconds timeout per request
            )

            # Connection test
            try:
                print("🔗 Testing connection to Confluence API...")
                test_result = confluence.get_space(self.space_key)
                print(f"✅ Connection successful - Space: {test_result.get('name', 'Unknown')}")
            except Exception as e:
                print(f"⚠️ Warning during connection test: {e}")
                print("🔄 Attempting to continue despite warning...")

            # Retrieve all pages from the space with pagination
            print(f"Retrieving pages from space {self.space_key}...")

            # Use pagination to retrieve all pages with safety mechanisms
            start = 0
            limit = 100  # Maximum number of pages to retrieve per request
            all_pages = []

            # Safety mechanisms for Azure environment
            MAX_PAGES = 1000  # Absolute safety limit
            TIMEOUT_SECONDS = 300  # 5 minutes maximum
            MAX_DUPLICATES = 3  # Maximum number of consecutive duplicate batches

            start_time = time.time()
            seen_page_ids = set()
            consecutive_duplicates = 0
            iteration_count = 0

            print(f"🔧 Safety limits: {MAX_PAGES} pages max, {TIMEOUT_SECONDS}s timeout")

            while True:
                iteration_count += 1

                # Timeout check
                elapsed_time = time.time() - start_time
                if elapsed_time > TIMEOUT_SECONDS:
                    print(f"🛑 TIMEOUT: Stopping after {elapsed_time:.1f}s ({iteration_count} iterations)")
                    break

                # Maximum pages check
                if len(all_pages) >= MAX_PAGES:
                    print(f"🛑 LIMIT: Stopping after {len(all_pages)} pages")
                    break

                print(f"📥 Batch {iteration_count}: start={start}, limit={limit}")
                batch_start_time = time.time()

                # Retrieve a batch of pages
                try:
                    batch_raw = confluence.get_all_pages_from_space(
                        self.space_key, start=start, limit=limit, expand="version"
                    )
                    # Convert to list if it's a generator
                    batch = list(batch_raw) if batch_raw else []
                except Exception as e:
                    print(f"❌ Error retrieving batch {iteration_count}: {e}")
                    break

                batch_duration = time.time() - batch_start_time

                if not batch:
                    print(f"✅ Normal end: no pages returned (batch {iteration_count})")
                    break  # No more pages to retrieve

                # Check for duplicates
                batch_ids = {page.get("id") for page in batch if page.get("id")}
                new_ids = batch_ids - seen_page_ids

                print(f"📊 Batch {iteration_count}: {len(batch)} pages received in {batch_duration:.2f}s")
                print(f"🔍 New IDs: {len(new_ids)}/{len(batch_ids)} (duplicates: {len(batch_ids) - len(new_ids)})")

                if len(new_ids) == 0:
                    consecutive_duplicates += 1
                    print(f"⚠️ Duplicate batch detected #{consecutive_duplicates}")
                    if consecutive_duplicates >= MAX_DUPLICATES:
                        print(f"🛑 STOP: {consecutive_duplicates} consecutive duplicate batches")
                        break
                else:
                    consecutive_duplicates = 0
                    seen_page_ids.update(new_ids)
                    # Add only new pages
                    new_pages = [p for p in batch if p.get("id") in new_ids]
                    all_pages.extend(new_pages)
                    print(f"✅ Added {len(new_pages)} new pages (total: {len(all_pages)})")

                # Update the starting index for the next request
                old_start = start
                start += len(batch)

                # Check that start progresses
                if start == old_start:
                    print(f"🛑 ERROR: Pagination stuck (start={start})")
                    break

                # If the number of pages retrieved is less than the limit, we've retrieved everything
                if len(batch) < limit:
                    print(f"✅ Normal end: incomplete batch ({len(batch)} < {limit})")
                    break

            pages = all_pages

            # Collection summary
            total_time = time.time() - start_time
            print("\n🎉 COLLECTION COMPLETED!")
            print("📊 Summary:")
            print(f"   • Pages retrieved: {len(pages)}")
            print(f"   • Total time: {total_time:.1f}s")
            print(f"   • Iterations: {iteration_count}")
            print(f"   • Unique pages: {len(seen_page_ids)}")
            if consecutive_duplicates > 0:
                print(f"   • Duplicates detected: {consecutive_duplicates} batches")

            # Also retrieve child pages (sub-pages) if necessary
            if len(pages) > 0:
                print("🔍 Searching for additional sub-pages...")
                child_pages = []
                processed_parent_ids = set()

                # Safety limit for sub-pages
                MAX_CHILD_REQUESTS = 500
                child_request_count = 0

                for i, page in enumerate(pages):
                    page_id = page.get("id")
                    page_title = page.get("title", "Untitled")

                    # Avoid processing the same parent multiple times
                    if page_id in processed_parent_ids:
                        continue
                    processed_parent_ids.add(page_id)

                    # Safety limit
                    child_request_count += 1
                    if child_request_count > MAX_CHILD_REQUESTS:
                        print(f"🛑 LIMIT: Stopping sub-page retrieval after {MAX_CHILD_REQUESTS} requests")
                        break

                    # Progress indicator
                    if i % 50 == 0:
                        print(f"📄 Checking sub-pages: {i + 1}/{len(pages)} pages processed")

                    # Retrieve the children of this page
                    try:
                        children_raw = confluence.get_page_child_by_type(page_id, type="page")
                        # Convert to list if it's a generator
                        children = list(children_raw) if children_raw else []
                        if children and len(children) > 0:
                            print(f"📁 {len(children)} sub-pages found for '{page_title}'")
                            child_pages.extend(children)
                    except Exception as e:
                        print(f"❌ Error retrieving sub-pages for '{page_title}': {str(e)}")

                # Add sub-pages to our main list (avoiding duplicates)
                existing_ids = {p.get("id") for p in pages}
                new_child_pages = [p for p in child_pages if p.get("id") not in existing_ids]

                if new_child_pages:
                    pages.extend(new_child_pages)
                    print(f"✅ Added {len(new_child_pages)} unique sub-pages. Total: {len(pages)} pages.")
                else:
                    print("ℹ️ No new sub-pages found.")

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
                    print(f"Error processing page {page_title}: {str(page_error)}")
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

            print("Complete error trace:")
            print(traceback.format_exc())
            raise e

    def split_docs(self, docs: list):
        # Markdown
        headers_to_split_on = [
            ("#", "Title 1"),
            ("##", "Subtitle 1"),
            ("###", "Subtitle 2"),
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

    def save_to_db(self, splitted_docs: list, embeddings: Embeddings) -> FAISS:
        """Save chunks to Chroma DB"""
        db = FAISS.from_documents(splitted_docs, embeddings)
        db.save_local(self.persist_directory)
        return db

    def load_from_db(self, embeddings: Embeddings) -> FAISS:
        """Loader chunks to Chroma DB"""
        db: FAISS = FAISS.load_local(
            self.persist_directory,
            embeddings,
            allow_dangerous_deserialization=True,  # Necessary for recent versions of LangChain
        )
        return db

    def create_dummy_docs(self) -> list:
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

    def set_db(self, embeddings: Embeddings) -> FAISS:
        """Create, save, and load db"""
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

    def get_db(self, embeddings: object) -> FAISS:
        """Create, save, and load db"""
        db = self.load_from_db(embeddings)
        return db


if __name__ == "__main__":
    pass
