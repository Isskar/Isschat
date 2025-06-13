"""
Azure Blob Storage adapter for file operations
Handles both vector database and logs storage with managed identity authentication
"""

import logging
from typing import List
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential


class AzureStorageAdapter:
    """Azure Blob Storage adapter with managed identity authentication"""

    def __init__(self, storage_account_name: str):
        """
        Initialize Azure Storage adapter with managed identity

        Args:
            storage_account_name: Name of the Azure Storage account
        """
        self.storage_account_name = storage_account_name
        self.container_name = "isschat-data"

        # Use managed identity for authentication in production
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

    def copy_directory_to_azure(self, local_directory: str, azure_directory: str) -> bool:
        """
        Copy an entire local directory to Azure Blob Storage

        Args:
            local_directory: Local directory path
            azure_directory: Azure directory path

        Returns:
            True if successful, False otherwise
        """
        try:
            import os

            for root, dirs, files in os.walk(local_directory):
                for file in files:
                    local_file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_file_path, local_directory)
                    azure_file_path = f"{azure_directory}/{relative_path}".replace("\\", "/")

                    with open(local_file_path, "rb") as f:
                        file_data = f.read()

                    if not self.write_file(azure_file_path, file_data):
                        return False

            logging.info(f"Successfully copied directory to Azure: {local_directory} -> {azure_directory}")
            return True
        except Exception as e:
            logging.error(f"Error copying directory to Azure: {e}")
            return False

    def copy_directory_from_azure(self, azure_directory: str, local_directory: str) -> bool:
        """
        Copy an entire directory from Azure Blob Storage to local

        Args:
            azure_directory: Azure directory path
            local_directory: Local directory path

        Returns:
            True if successful, False otherwise
        """
        try:
            import os

            files = self.list_files(azure_directory)

            for azure_file_path in files:
                file_data = self.read_file(azure_file_path)
                relative_path = azure_file_path.replace(azure_directory, "").lstrip("/")
                local_file_path = os.path.join(local_directory, relative_path)

                # Create local directory if needed
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

                with open(local_file_path, "wb") as f:
                    f.write(file_data)

            logging.info(f"Successfully copied directory from Azure: {azure_directory} -> {local_directory}")
            return True
        except Exception as e:
            logging.error(f"Error copying directory from Azure: {e}")
            return False
