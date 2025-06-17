"""
Storage Factory for creating storage implementations
"""

from .storage_interface import StorageInterface
from .local_storage import LocalStorage


class StorageFactory:
    """Factory for creating storage implementations"""

    @staticmethod
    def create_storage(storage_type: str, **kwargs) -> StorageInterface:
        """
        Create a storage implementation based on type

        Args:
            storage_type: Type of storage ("local" or "azure")
            **kwargs: Additional arguments for storage initialization

        Returns:
            StorageInterface implementation

        Raises:
            ValueError: If storage type is unknown
            ImportError: If Azure dependencies are missing
        """
        if storage_type.lower() == "local":
            base_path = kwargs.get("base_path", ".")
            return LocalStorage(base_path=base_path)

        elif storage_type.lower() == "azure":
            try:
                # Import Azure storage only when needed
                from .azure_storage_adapter import AzureStorage

                account_name = kwargs.get("account_name")
                if not account_name:
                    raise ValueError("Azure storage requires 'account_name' parameter")

                return AzureStorage(storage_account_name=account_name)

            except ImportError as e:
                raise ImportError(f"Azure storage dependencies not available: {e}")

        else:
            raise ValueError(f"Unknown storage type: {storage_type}")

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
    def create_azure_storage(account_name: str) -> StorageInterface:
        """
        Convenience method to create Azure storage

        Args:
            account_name: Azure storage account name

        Returns:
            AzureStorage instance

        Raises:
            ImportError: If Azure dependencies are missing
        """
        try:
            from .azure_storage_adapter import AzureStorage

            return AzureStorage(storage_account_name=account_name)
        except ImportError as e:
            raise ImportError(f"Azure storage dependencies not available: {e}")
