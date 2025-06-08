"""
Confluence extractor with sub-pages support.
"""

from typing import List, Dict, Any, Set
from .base_extractor import BaseExtractor, Document


class ConfluenceExtractor(BaseExtractor):
    """Extractor for Confluence data with recursive sub-pages retrieval."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Confluence extractor.

        Args:
            config: Configuration containing connection parameters
        """
        self.config = config
        # Map configuration keys correctly
        space_name = config.get("confluence_space_name", "")
        if space_name.startswith("http"):
            # If space_name is already a full URL, use it directly
            self.base_url = config.get("confluence_url") or space_name
        else:
            # If space_name is just the subdomain, construct the full URL
            self.base_url = config.get("confluence_url") or f"https://{space_name}.atlassian.net"
        self.username = config.get("confluence_email_address")
        self.api_token = config.get("confluence_private_api_key")
        self.space_key = config.get("confluence_space_key")

    def _get_all_pages_recursive(self, confluence, space_key: str) -> List[Dict[str, Any]]:
        """
        Retrieve all pages from a space recursively, including sub-pages.

        Args:
            confluence: Confluence connection instance
            space_key: Space key

        Returns:
            List[Dict]: List of all pages (root + sub-pages)
        """
        all_pages = []
        processed_page_ids: Set[str] = set()

        print(f"🔍 Starting page retrieval from space: {space_key}")

        # Step 1: Retrieve all pages from space (standard method)
        try:
            # Use higher limit and handle pagination
            start = 0
            limit = 100
            batch_count = 0

            while True:
                batch_count += 1
                batch_pages = confluence.get_all_pages_from_space(
                    space=space_key, start=start, limit=limit, expand="body.storage,version,ancestors,children.page"
                )

                if not batch_pages:
                    break

                print(f"📄 Batch {batch_count}: Retrieved {len(batch_pages)} pages (offset: {start})")
                all_pages.extend(batch_pages)

                for page in batch_pages:
                    processed_page_ids.add(page.get("id"))

                # If we have fewer pages than the limit, we've retrieved everything
                if len(batch_pages) < limit:
                    break

                start += limit

        except Exception as e:
            print(f"⚠️ Standard retrieval failed: {str(e)}")
            print("🔄 Continuing with recursive method...")

        print(f"✅ Standard method: {len(all_pages)} pages retrieved")

        # Step 2: Recursive retrieval of missing sub-pages
        print("🔍 Checking for missing sub-pages...")

        # Identify root pages (without ancestors)
        root_pages = [p for p in all_pages if not p.get("ancestors", [])]
        print(f"🌳 Found {len(root_pages)} root pages")

        # For each root page, recursively retrieve children
        total_missing = 0
        for i, root_page in enumerate(root_pages, 1):
            root_title = root_page.get("title", "Unknown")
            print(f"📂 [{i}/{len(root_pages)}] Checking sub-pages for: '{root_title}'")

            missing_children = self._get_missing_children_recursive(confluence, root_page, processed_page_ids)

            if missing_children:
                print(f"   ➕ Found {len(missing_children)} missing sub-pages")
                all_pages.extend(missing_children)
                total_missing += len(missing_children)
            else:
                print("   ✓ No missing sub-pages")

        if total_missing > 0:
            print(f"🔧 Recursive method: {total_missing} additional pages retrieved")

        print(f"🎯 Total pages collected: {len(all_pages)}")
        return all_pages

    def _get_missing_children_recursive(
        self, confluence, parent_page: Dict[str, Any], processed_ids: Set[str]
    ) -> List[Dict[str, Any]]:
        """
        Recursively retrieve missing child pages.

        Args:
            confluence: Confluence connection instance
            parent_page: Parent page
            processed_ids: Set of already processed IDs

        Returns:
            List[Dict]: List of missing child pages
        """
        missing_children = []
        parent_id = parent_page.get("id")
        parent_title = parent_page.get("title", "Unknown")

        try:
            # Retrieve direct children
            children = confluence.get_page_child_by_type(
                parent_id, type="page", start=0, limit=100, expand="body.storage,version,ancestors,children.page"
            )

            if not children:
                return missing_children

            for child in children:
                child_id = child.get("id")
                child_title = child.get("title", "Unknown")

                # If this child page wasn't retrieved by the standard method
                if child_id not in processed_ids:
                    print(f"      📄 Found missing sub-page: '{child_title}'")
                    missing_children.append(child)
                    processed_ids.add(child_id)

                    # Recursion for children of this page
                    grandchildren = self._get_missing_children_recursive(confluence, child, processed_ids)
                    missing_children.extend(grandchildren)

        except Exception as e:
            print(f"      ⚠️ Error retrieving children of '{parent_title}': {str(e)}")

        return missing_children

    def extract(self) -> List[Document]:
        """
        Extract documents from Confluence with sub-pages retrieval.

        Returns:
            List[Document]: List of extracted documents
        """
        try:
            import logging
            import sys
            from atlassian import Confluence
            import html2text

            # Configure logging
            logging.basicConfig(
                level=logging.INFO,
                stream=sys.stdout,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )

            # Validate required fields
            if not all([self.base_url, self.username, self.api_token, self.space_key]):
                missing_fields = []
                if not self.base_url:
                    missing_fields.append("confluence_url")
                if not self.username:
                    missing_fields.append("confluence_email_address")
                if not self.api_token:
                    missing_fields.append("confluence_private_api_key")
                if not self.space_key:
                    missing_fields.append("confluence_space_key")
                raise ValueError(f"Missing required configuration fields: {missing_fields}")

            # Initialize Confluence connection
            confluence = Confluence(
                url=self.base_url,
                username=self.username,
                password=self.api_token,
                cloud=True,
            )

            # Retrieve all pages (including sub-pages)
            all_pages = self._get_all_pages_recursive(confluence, self.space_key)

            # Analyze page hierarchy
            root_pages = [p for p in all_pages if not p.get("ancestors", [])]
            child_pages = [p for p in all_pages if p.get("ancestors", [])]

            print("\n📊 Page Analysis:")
            print(f"   🌳 Root pages: {len(root_pages)}")
            print(f"   📄 Sub-pages: {len(child_pages)}")
            print(f"   📋 Total pages: {len(all_pages)}")

            # Convert pages to documents
            documents = []
            skipped_pages = []
            converter = html2text.HTML2Text()
            converter.ignore_links = False
            converter.ignore_images = True

            print(f"\n🔄 Converting {len(all_pages)} pages to documents...")

            for i, page in enumerate(all_pages, 1):
                try:
                    page_id = page.get("id")
                    title = page.get("title", "Untitled")

                    # Progress indicator
                    if i % 10 == 0 or i == len(all_pages):
                        print(f"📝 Progress: {i}/{len(all_pages)} pages processed")

                    # Get page content
                    body = page.get("body", {}).get("storage", {}).get("value", "")

                    if not body:
                        skipped_pages.append(f"'{title}' (no content)")
                        continue

                    # Convert HTML to markdown
                    content = converter.handle(body)

                    # Clean up content
                    content = content.strip()
                    if not content:
                        skipped_pages.append(f"'{title}' (empty after conversion)")
                        continue

                    # Determine hierarchy level
                    ancestors = page.get("ancestors", [])
                    hierarchy_level = len(ancestors)
                    parent_titles = [a.get("title", "Unknown") for a in ancestors]
                    hierarchy_path = " > ".join(parent_titles + [title])

                    # Create document with enhanced metadata
                    doc = Document(
                        content=content,
                        metadata={
                            "source": "confluence",
                            "space_key": self.space_key,
                            "page_id": page_id,
                            "title": title,
                            "url": f"{self.base_url.rstrip('/')}/wiki/spaces/{self.space_key}/pages/{page_id}"
                            if not self.base_url.endswith("/wiki")
                            else f"{self.base_url}/spaces/{self.space_key}/pages/{page_id}",
                            "version": page.get("version", {}).get("number", 1),
                            "hierarchy_level": hierarchy_level,
                            "hierarchy_path": hierarchy_path,
                            "parent_titles": parent_titles,
                            "is_root_page": hierarchy_level == 0,
                        },
                    )
                    documents.append(doc)

                    # Log successful processing with hierarchy indication
                    indent = "  " * hierarchy_level
                    print(f"✅ {indent}[{i}/{len(all_pages)}] {hierarchy_path}")

                except Exception as e:
                    skipped_pages.append(f"'{title}' (error: {str(e)})")
                    continue

            # Final summary
            print("\n🎉 Extraction completed!")
            print(f"   📄 Pages processed: {len(all_pages)}")
            print(f"   📋 Documents created: {len(documents)}")
            print(f"   🌳 Root pages: {len([d for d in documents if d.metadata.get('is_root_page')])}")
            print(f"   📄 Sub-pages: {len([d for d in documents if not d.metadata.get('is_root_page')])}")

            if skipped_pages:
                print(f"   ⚠️ Skipped pages: {len(skipped_pages)}")
                for skipped in skipped_pages[:5]:  # Show first 5 skipped pages
                    print(f"      - {skipped}")
                if len(skipped_pages) > 5:
                    print(f"      ... and {len(skipped_pages) - 5} more")

            return documents

        except Exception as e:
            print(f"❌ Extraction failed: {str(e)}")
            return []

    def validate_connection(self) -> bool:
        """
        Validate Confluence connection.

        Returns:
            bool: True if connection is valid
        """
        # Check basic configuration
        config_valid = all([self.base_url, self.username, self.api_token, self.space_key])

        if not config_valid:
            return False

        # Test actual connection
        try:
            from atlassian import Confluence

            confluence = Confluence(
                url=self.base_url,
                username=self.username,
                password=self.api_token,
                cloud=True,
            )

            # Test authentication
            try:
                confluence.get_all_spaces(limit=1)

                # Test space access
                try:
                    confluence.get_space(self.space_key)
                    return True
                except Exception:
                    return False

            except Exception:
                return False

        except ImportError:
            return False
        except Exception:
            return False
