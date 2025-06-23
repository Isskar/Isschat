from src.storage.storage_interface import StorageInterface
from src.storage.local_storage import LocalStorage
from src.storage.azure_storage_adapter import AzureStorage


class StorageFactory:
    """Factory for creating storage implementations"""

    @staticmethod
    def create_local_storage(base_path: str = ".") -> StorageInterface:
        return LocalStorage(base_path=base_path)

    @staticmethod
    def create_azure_storage(account_name: str, container_name: str) -> StorageInterface:
        try:
            return AzureStorage(storage_account_name=account_name, container_name=container_name)
        except ImportError as e:
            raise ImportError(f"Azure storage dependencies not available: {e}")
