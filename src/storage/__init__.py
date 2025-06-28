"""
Storage adapters for different environments
"""

from .storage_interface import StorageInterface
from .local_storage import LocalStorage
from .azure_storage_adapter import AzureStorage
from .storage_factory import StorageFactory

__all__ = [
    "StorageInterface",
    "LocalStorage",
    "AzureStorage",
    "StorageFactory",
]
