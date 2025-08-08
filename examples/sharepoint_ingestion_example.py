"""
Example of how to use SharePoint ingestion pipeline.

This script demonstrates how to:
1. Set up SharePoint authentication
2. Test the connection
3. Extract documents from SharePoint
4. Run the complete ingestion pipeline

Before running this script, you need to:
1. Create an Azure AD app registration
2. Grant it appropriate SharePoint permissions
3. Set the required environment variables
"""

import os
import logging
from datetime import datetime, timedelta

from src.ingestion import create_sharepoint_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main example function."""

    # Step 1: Configure SharePoint connection
    # These should be environment variables in production
    config = {
        "tenant_id": os.getenv("AZURE_TENANT_ID"),  # Your Azure AD tenant ID
        "client_id": os.getenv("AZURE_CLIENT_ID"),  # Your Azure AD app client ID
        "client_secret": os.getenv("AZURE_CLIENT_SECRET"),  # Your Azure AD app secret
        "site_url": "https://inergie.sharepoint.com/sites/ISSKAR",  # Your SharePoint site URL
    }

    # Validate configuration
    missing_config = [key for key, value in config.items() if not value]
    if missing_config:
        logger.error(f"Missing configuration: {missing_config}")
        logger.error("Please set the following environment variables:")
        logger.error("- AZURE_TENANT_ID: Your Azure AD tenant ID")
        logger.error("- AZURE_CLIENT_ID: Your Azure AD app client ID")
        logger.error("- AZURE_CLIENT_SECRET: Your Azure AD app secret")
        return

    # Step 2: Create SharePoint pipeline
    logger.info("Creating SharePoint pipeline...")
    pipeline = create_sharepoint_pipeline(
        tenant_id=config["tenant_id"],
        client_id=config["client_id"],
        client_secret=config["client_secret"],
        site_url=config["site_url"],
    )

    # Step 3: Test connection
    logger.info("Testing SharePoint connection...")
    connection_result = pipeline.test_connection()

    if not connection_result["success"]:
        logger.error(f"Connection test failed: {connection_result.get('error', 'Unknown error')}")
        return

    logger.info("✅ Connection test successful!")

    # Step 4: Get site info
    logger.info("Getting site information...")
    site_info = pipeline.get_site_info()
    logger.info(f"Site URL: {site_info.get('site_url')}")
    logger.info(f"Capabilities: {site_info.get('capabilities')}")

    # Step 5: Run full ingestion
    logger.info("Starting full document ingestion...")
    result = pipeline.run(force_rebuild=False)  # Set to True to rebuild from scratch

    if result["success"]:
        logger.info("✅ Ingestion completed successfully!")
        logger.info(f"Duration: {result['duration_seconds']:.1f} seconds")
        logger.info(f"Documents extracted: {result['statistics']['documents_extracted']}")
        logger.info(f"Chunks created: {result['statistics']['chunks_created']}")
        logger.info(f"Embeddings generated: {result['statistics']['embeddings_generated']}")
        logger.info(f"Documents stored: {result['statistics']['documents_stored']}")
    else:
        logger.error(f"❌ Ingestion failed: {result.get('error')}")

    # Step 6: Example of incremental sync (optional)
    logger.info("Testing incremental sync...")

    # Sync documents modified in the last 7 days
    since_date = datetime.now() - timedelta(days=7)
    sync_result = pipeline.sync_documents(mode="incremental", since=since_date)

    if sync_result["success"]:
        logger.info(f"✅ Incremental sync found {sync_result['documents_retrieved']} modified documents")
    else:
        logger.error(f"❌ Incremental sync failed: {sync_result.get('error')}")


if __name__ == "__main__":
    main()
