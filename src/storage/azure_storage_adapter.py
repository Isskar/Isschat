"""
Azure Blob Storage implementation
"""

import logging
import os
from typing import List
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential

from .storage_interface import StorageInterface


class AzureStorage(StorageInterface):
    """Azure Blob Storage implementation with managed identity authentication"""

    def __init__(self, storage_account_name: str):
        """
        Initialize Azure Storage adapter with managed identity

        Args:
            storage_account_name: Name of the Azure Storage account
        """
        self.storage_account_name = storage_account_name
        self.container_name = "isschatblobfiles"

        # Use managed identity for authentication in production
        # Configure tenant to avoid authentication errors
        tenant_id = os.getenv("AZURE_TENANT_ID")
        if tenant_id:
            credential = DefaultAzureCredential(additionally_allowed_tenants=[tenant_id])
        else:
            credential = DefaultAzureCredential()
        account_url = f"https://{storage_account_name}.blob.core.windows.net"

        self.blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)

        self._ensure_container_exists()
        logging.info(f"Azure Storage adapter initialized for account: {storage_account_name}")

    def _ensure_container_exists(self):
        """Ensure the main container exists"""
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            if not container_client.exists():
                container_client.create_container()
                logging.info(f"Created container: {self.container_name}")
        except Exception as e:
            logging.error(f"Error ensuring container exists: {e}")

    def read_file(self, file_path: str) -> bytes:
        """
        Read a file from Azure Blob Storage

        Args:
            file_path: Path to the file in blob storage

        Returns:
            File content as bytes

        Raises:
            Exception: If file cannot be read
        """
        try:
            blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=file_path)
            return blob_client.download_blob().readall()
        except Exception as e:
            logging.error(f"Error reading file from Azure {file_path}: {e}")
            raise

    def write_file(self, file_path: str, data: bytes) -> bool:
        """
        Write a file to Azure Blob Storage

        Args:
            file_path: Path where to store the file
            data: File content as bytes

        Returns:
            True if successful, False otherwise
        """
        try:
            blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=file_path)
            blob_client.upload_blob(data, overwrite=True)
            logging.debug(f"Successfully wrote file to Azure: {file_path}")
            return True
        except Exception as e:
            logging.error(f"Error writing file to Azure {file_path}: {e}")
            return False

    def file_exists(self, file_path: str) -> bool:
        """
        Check if a file exists in Azure Blob Storage

        Args:
            file_path: Path to check

        Returns:
            True if file exists, False otherwise
        """
        try:
            blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=file_path)
            return blob_client.exists()
        except Exception:
            return False

    def list_files(self, directory_path: str, pattern: str = "*") -> List[str]:
        """
        List files in a directory in Azure Blob Storage

        Args:
            directory_path: Directory to list
            pattern: File pattern to match (basic support)

        Returns:
            List of file paths
        """
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            blobs = container_client.list_blobs(name_starts_with=directory_path)

            # Basic pattern matching
            if pattern == "*":
                return [blob.name for blob in blobs]
            else:
                import fnmatch

                return [blob.name for blob in blobs if fnmatch.fnmatch(blob.name, pattern)]
        except Exception as e:
            logging.error(f"Error listing files in Azure {directory_path}: {e}")
            return []

    def delete_file(self, file_path: str) -> bool:
        """
        Delete a file from Azure Blob Storage

        Args:
            file_path: Path to the file to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=file_path)
            blob_client.delete_blob()
            logging.debug(f"Successfully deleted file from Azure: {file_path}")
            return True
        except Exception as e:
            logging.error(f"Error deleting file from Azure {file_path}: {e}")
            return False

    def create_directory(self, directory_path: str) -> bool:
        """
        Create a directory in Azure Blob Storage (no-op since Azure uses flat structure)

        Args:
            directory_path: Path to the directory to create

        Returns:
            True (always successful for Azure)
        """
        # Azure Blob Storage uses flat structure, directories are virtual
        # No need to create directories explicitly
        logging.debug(f"Directory creation requested for Azure: {directory_path} (no-op)")
        return True

    def directory_exists(self, directory_path: str) -> bool:
        """
        Check if a directory exists in Azure Blob Storage

        Args:
            directory_path: Path to check

        Returns:
            True if any blobs exist with this prefix, False otherwise
        """
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            # Check if any blobs exist with this prefix
            blobs = container_client.list_blobs(name_starts_with=directory_path, max_results=1)
            return any(True for _ in blobs)
        except Exception as e:
            logging.error(f"Error checking directory existence in Azure {directory_path}: {e}")
            return False
