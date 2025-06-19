"""
Storage adapters for different environments
"""

from src.storage.storage_interface import StorageInterface
from src.storage.local_storage import LocalStorage
from src.storage.azure_storage_adapter import AzureStorage
from src.storage.storage_factory import StorageFactory

__all__ = [
    "StorageInterface",
    "LocalStorage",
    "AzureStorage",
    "StorageFactory",
]
