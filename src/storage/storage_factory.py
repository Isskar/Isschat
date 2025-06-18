"""
Storage Factory for creating storage implementations
"""

from .storage_interface import StorageInterface
from .local_storage import LocalStorage


class StorageFactory:
    """Factory for creating storage implementations"""

    @staticmethod
    def create_local_storage(base_path: str = ".") -> StorageInterface:
        """
        Convenience method to create local storage

        Args:
            base_path: Base directory for local storage

        Returns:
            LocalStorage instance
        """
        return LocalStorage(base_path=base_path)

    @staticmethod
    def create_azure_storage(account_name: str, container_name: str) -> StorageInterface:
        """
        Convenience method to create Azure storage

        Args:
            account_name: Azure storage account name
            container_name: Azure storage container name

        Returns:
            AzureStorage instance

        Raises:
            ImportError: If Azure dependencies are missing
        """
        try:
            from .azure_storage_adapter import AzureStorage

            return AzureStorage(storage_account_name=account_name, container_name=container_name)
        except ImportError as e:
            raise ImportError(f"Azure storage dependencies not available: {e}")
