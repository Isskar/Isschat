import logging
from typing import List
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential

from src.storage.storage_interface import StorageInterface


class AzureStorage(StorageInterface):
    """Azure Blob Storage implementation with managed identity authentication"""

    def __init__(self, storage_account_name: str, container_name: str):
        self.storage_account_name = storage_account_name
        self.container_name = container_name

        # Use managed identity for authentication in production
        credential = DefaultAzureCredential()
        account_url = f"https://{storage_account_name}.blob.core.windows.net"

        try:
            self.blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)
            # Test connection immediately
            self._test_connection()
        except Exception as e:
            from src.core.exceptions import create_azure_access_error

            raise create_azure_access_error(
                "STORAGE",
                f"storage account '{storage_account_name}'",
                e,
                "Vérifiez vos permissions Azure et la configuration du compte de stockage.",
            ) from e

        try:
            self._ensure_container_exists()
        except Exception as e:
            from src.core.exceptions import create_azure_access_error

            raise create_azure_access_error(
                "CONTAINER",
                f"container '{self.container_name}'",
                e,
                "Vérifiez vos permissions Azure et la configuration du container.",
            ) from e

        logging.info(f"Azure Storage adapter initialized for account: {storage_account_name}")

    def _test_connection(self):
        """Test Azure Storage connection by listing containers"""
        try:
            # Try to list containers to verify connection
            _ = list(self.blob_service_client.list_containers())
            logging.info("✅ Azure Storage connection test successful")
        except Exception as e:
            logging.error(f"❌ Azure Storage connection test failed: {e}")
            raise

    def _ensure_container_exists(self):
        container_client = self.blob_service_client.get_container_client(self.container_name)

        if not container_client.exists():
            container_client.create_container()
            logging.info(f"Created container: {self.container_name}")

    def read_file(self, file_path: str) -> bytes:
        try:
            blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=file_path)
            return blob_client.download_blob().readall()
        except Exception as e:
            logging.error(f"Error reading file from Azure {file_path}: {e}")
            raise

    def write_file(self, file_path: str, data: bytes) -> bool:
        try:
            blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=file_path)
            blob_client.upload_blob(data, overwrite=True)
            logging.debug(f"Successfully wrote file to Azure: {file_path}")
            return True
        except Exception as e:
            logging.error(f"Error writing file to Azure {file_path}: {e}")
            return False

    def file_exists(self, file_path: str) -> bool:
        try:
            blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=file_path)
            return blob_client.exists()
        except Exception:
            return False

    def list_files(self, directory_path: str, pattern: str = "*") -> List[str]:
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
        try:
            blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=file_path)
            blob_client.delete_blob()
            logging.debug(f"Successfully deleted file from Azure: {file_path}")
            return True
        except Exception as e:
            logging.error(f"Error deleting file from Azure {file_path}: {e}")
            return False

    def create_directory(self, directory_path: str) -> bool:
        # Azure Blob Storage uses flat structure, directories are virtual
        # No need to create directories explicitly
        logging.debug(f"Directory creation requested for Azure: {directory_path} (no-op)")
        return True

    def directory_exists(self, directory_path: str) -> bool:
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            # Check if any blobs exist with this prefix
            blobs = container_client.list_blobs(name_starts_with=directory_path)
            # Take only the first result to check existence
            return next(iter(blobs), None) is not None
        except Exception as e:
            logging.error(f"Error checking directory existence in Azure {directory_path}: {e}")
            return False
