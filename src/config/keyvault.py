import os
import logging
from typing import Optional, Dict
from functools import lru_cache
from azure.keyvault.secrets import SecretClient
from azure.identity import ManagedIdentityCredential
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class KeyVaultManager:
    """
    Centralized Key Vault manager for the Isschat application.
    Uses Azure Key Vault only when running in Azure environment with Managed Identity.
    """

    def __init__(self):
        """Initialize the Key Vault manager."""
        load_dotenv()

        # Check if we're running in Azure (production environment)
        self.is_azure = os.getenv("ENVIRONMENT") == "production"
        self.vault_url = os.getenv("KEY_VAULT_URL")
        self.use_key_vault = self.is_azure and bool(self.vault_url)
        self._client: Optional[SecretClient] = None
        self._local_cache: Dict[str, str] = {}

        if self.use_key_vault:
            try:
                # Use Managed Identity when running in Azure
                credential = ManagedIdentityCredential()
                self._client = SecretClient(vault_url=self.vault_url, credential=credential)
                logger.info(f"Successfully connected to Key Vault using Managed Identity: {self.vault_url}")
            except Exception as e:
                logger.error(f"Failed to connect to Key Vault: {e}")
                self.use_key_vault = False
                self._client = None
        else:
            logger.info("Running in local environment - using environment variables instead of Key Vault")

    @lru_cache(maxsize=32)
    def get_secret(self, secret_name: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get a secret value from Key Vault (in Azure) or environment variables (locally).

        Args:
            secret_name: The name of the secret to retrieve
            default: Default value if secret is not found

        Returns:
            The secret value or default if not found
        """
        # First check local cache
        if secret_name in self._local_cache:
            return self._local_cache[secret_name]

        # Use Key Vault if running in Azure
        if self.use_key_vault and self._client:
            try:
                # Convert environment variable names to Key Vault secret names
                # Replace underscores with hyphens for Key Vault compatibility
                kv_secret_name = secret_name.replace("_", "-")
                secret = self._client.get_secret(kv_secret_name)
                value = secret.value
                self._local_cache[secret_name] = value
                return value
            except Exception as e:
                logger.debug(f"Secret '{kv_secret_name}' not found in Key Vault: {e}")

        # Fall back to environment variables (for local development)
        env_value = os.getenv(secret_name, default)
        if env_value:
            self._local_cache[secret_name] = env_value
        return env_value

    def clear_cache(self):
        """Clear the local cache and LRU cache."""
        self._local_cache.clear()
        self.get_secret.cache_clear()


# Singleton instance
_key_vault_manager: Optional[KeyVaultManager] = None


def get_key_vault_manager() -> Optional[KeyVaultManager]:
    """
    Get the singleton Key Vault manager instance.

    Returns:
        The Key Vault manager instance
    """
    global _key_vault_manager
    if _key_vault_manager is None:
        _key_vault_manager = KeyVaultManager()
    return _key_vault_manager


# Convenience function for getting secrets
def get_secret(secret_name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Convenience function to get a secret value.

    Args:
        secret_name: The name of the secret to retrieve
        default: Default value if secret is not found

    Returns:
        The secret value or default if not found
    """
    manager = get_key_vault_manager()
    return manager.get_secret(secret_name, default)
