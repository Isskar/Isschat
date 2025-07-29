"""
Secrets management module that wraps Key Vault access with proper naming conventions.
This module provides a unified interface for accessing secrets from either:
- Azure Key Vault (when running in production on Azure)
- Environment variables (when running locally)
"""

import os
from typing import Optional
from src.config.keyvault import get_key_vault_manager

# Mapping of environment variable names to Key Vault secret names
# Key Vault requires hyphens instead of underscores
SECRET_MAPPING = {
    # Confluence secrets
    "CONFLUENCE_PRIVATE_API_KEY": "CONFLUENCE-PRIVATE-API-KEY",
    "CONFLUENCE_SPACE_KEY": "CONFLUENCE-SPACE-KEY",
    "CONFLUENCE_SPACE_NAME": "CONFLUENCE-SPACE-NAME",
    "CONFLUENCE_EMAIL_ADDRESS": "CONFLUENCE-EMAIL-ADDRESS",
    # OpenRouter
    "OPENROUTER_API_KEY": "OPENROUTER-API-KEY",
    # Weaviate secrets
    "WEAVIATE_URL": "WEAVIATE-URL",
    "WEAVIATE_API_KEY": "WEAVIATE-API-KEY",
    # Azure Storage (these might be managed differently as they're often part of the infrastructure)
    "AZURE_STORAGE_ACCOUNT": "AZURE-STORAGE-ACCOUNT",
    "AZURE_BLOB_CONTAINER_NAME": "AZURE-BLOB-CONTAINER-NAME",
    # Azure AD Authentication
    "STREAMLIT_AZURE_CLIENT_ID": "STREAMLIT-AZURE-CLIENT-ID",
    "STREAMLIT_AZURE_CLIENT_SECRET": "STREAMLIT-AZURE-CLIENT-SECRET",
    "STREAMLIT_AZURE_TENANT_ID": "STREAMLIT-AZURE-TENANT-ID",
    "STREAMLIT_AZURE_REDIRECT_URI": "STREAMLIT-AZURE-REDIRECT-URI",
}


def get_secret_value(secret_name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get a secret value, handling the Key Vault naming convention.

    Args:
        secret_name: The environment variable name (with underscores)
        default: Default value if secret is not found

    Returns:
        The secret value or default if not found
    """
    manager = get_key_vault_manager()

    # If we're using Key Vault and have a mapping, use the mapped name
    if manager.use_key_vault and secret_name in SECRET_MAPPING:
        kv_secret_name = SECRET_MAPPING[secret_name]
        # Try to get from Key Vault first
        value = manager.get_secret(kv_secret_name, None)
        if value:
            return value

    # Fall back to environment variable (for local development or if not in mapping)
    return os.getenv(secret_name, default)


# Convenience functions for specific secrets
def get_confluence_api_key() -> Optional[str]:
    """Get Confluence API key."""
    return get_secret_value("CONFLUENCE_PRIVATE_API_KEY")


def get_confluence_space_key() -> Optional[str]:
    """Get Confluence space key."""
    return get_secret_value("CONFLUENCE_SPACE_KEY")


def get_confluence_space_name() -> Optional[str]:
    """Get Confluence space name."""
    return get_secret_value("CONFLUENCE_SPACE_NAME")


def get_confluence_email() -> Optional[str]:
    """Get Confluence email address."""
    return get_secret_value("CONFLUENCE_EMAIL_ADDRESS")


def get_openrouter_api_key() -> Optional[str]:
    """Get OpenRouter API key."""
    return get_secret_value("OPENROUTER_API_KEY")


def get_helicone_api_key() -> Optional[str]:
    """Get Helicone API key."""
    return get_secret_value("HELICONE_API_KEY")


def get_weaviate_url() -> Optional[str]:
    """Get Weaviate URL."""
    return get_secret_value("WEAVIATE_URL")


def get_weaviate_api_key() -> Optional[str]:
    """Get Weaviate API key."""
    return get_secret_value("WEAVIATE_API_KEY")


def get_azure_storage_account() -> Optional[str]:
    """Get Azure storage account name."""
    return get_secret_value("AZURE_STORAGE_ACCOUNT")


def get_azure_blob_container() -> Optional[str]:
    """Get Azure blob container name."""
    return get_secret_value("AZURE_BLOB_CONTAINER_NAME")


# Azure AD Authentication secrets
def get_azure_ad_client_id() -> Optional[str]:
    """Get Azure AD client ID."""
    return get_secret_value("STREAMLIT_AZURE_CLIENT_ID")


def get_azure_ad_client_secret() -> Optional[str]:
    """Get Azure AD client secret."""
    return get_secret_value("STREAMLIT_AZURE_CLIENT_SECRET")


def get_azure_ad_tenant_id() -> Optional[str]:
    """Get Azure AD tenant ID."""
    return get_secret_value("STREAMLIT_AZURE_TENANT_ID")


def get_azure_ad_redirect_uri() -> Optional[str]:
    """Get Azure AD redirect URI."""
    return get_secret_value("STREAMLIT_AZURE_REDIRECT_URI")
