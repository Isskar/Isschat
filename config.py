"""
Configuration Manager for Isschat Application
Handles both local (.env) and Azure (Key Vault) configurations
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class ConfigurationData:
    """Data class to hold all configuration values"""

    confluence_private_api_key: str
    confluence_space_key: str
    confluence_space_name: str
    confluence_email_address: str
    openrouter_api_key: str
    db_path: str
    persist_directory: str

    def validate(self) -> bool:
        """Validate that all required fields are present"""
        required_fields = [
            "confluence_private_api_key",
            "confluence_space_key",
            "confluence_space_name",
            "confluence_email_address",
            "openrouter_api_key",
        ]

        for field in required_fields:
            if not getattr(self, field):
                logging.error(f"Missing required configuration: {field}")
                return False
        return True


class ConfigProvider(ABC):
    """Abstract base class for configuration providers"""

    @abstractmethod
    def load_config(self) -> ConfigurationData:
        """Load configuration from the provider"""
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the name of the configuration provider"""
        pass


class LocalConfigProvider(ConfigProvider):
    """Configuration provider for local development using .env files"""

    def __init__(self, env_file_path: str = ".env"):
        self.env_file_path = env_file_path

    def load_config(self) -> ConfigurationData:
        """Load configuration from .env file"""
        # Load .env file if it exists
        if os.path.exists(self.env_file_path):
            load_dotenv(self.env_file_path)
            logging.info(f"Loaded configuration from {self.env_file_path}")
        else:
            logging.warning(f"No .env file found at {self.env_file_path}")

        # Get base directory for relative paths
        base_dir = os.path.dirname(os.path.abspath(__file__ + "/../"))

        # Environment variable mapping
        env_mapping = {
            "confluence_private_api_key": "CONFLUENCE_PRIVATE_API_KEY",
            "confluence_space_key": "CONFLUENCE_SPACE_KEY",
            "confluence_space_name": "CONFLUENCE_SPACE_NAME",
            "confluence_email_address": "CONFLUENCE_EMAIL_ADDRESS",
            "openrouter_api_key": "OPENROUTER_API_KEY",
        }

        # Build configuration dict
        config_dict = {}
        for field, env_var in env_mapping.items():
            value = os.getenv(env_var, "")
            config_dict[field] = value
            logging.debug(f"LocalConfig - {field}: {value[:10] if value else 'EMPTY'}...")

        # Add path configurations
        config_dict.update(
            {
                "db_path": os.getenv("DB_PATH", os.path.join(base_dir, "data/users.db")),
                "persist_directory": os.getenv("PERSIST_DIRECTORY", os.path.join(base_dir, "db")),
            }
        )

        # Debug: Log config_dict structure before creating ConfigurationData
        logging.debug(f"LocalConfig - config_dict keys: {list(config_dict.keys())}")
        logging.debug(f"LocalConfig - config_dict types: {[(k, type(v).__name__) for k, v in config_dict.items()]}")

        # Validate all required fields are present and are strings
        required_fields = [
            "confluence_private_api_key",
            "confluence_space_key",
            "confluence_space_name",
            "confluence_email_address",
            "openrouter_api_key",
            "db_path",
            "persist_directory",
        ]
        for field in required_fields:
            if field not in config_dict:
                logging.error(f"LocalConfig - Missing field: {field}")
            elif not isinstance(config_dict[field], str):
                logging.error(f"LocalConfig - Field {field} has wrong type: {type(config_dict[field])}")

        return ConfigurationData(**config_dict)  # ty : ignore

    def get_provider_name(self) -> str:
        return "Local (.env)"


class AzureKeyVaultConfigProvider(ConfigProvider):
    """Configuration provider for Azure Key Vault"""

    def __init__(self, key_vault_url: str):
        self.key_vault_url = key_vault_url
        self._client = None

    def _get_client(self):
        """Get Azure Key Vault client with managed identity"""
        if self._client is None:
            try:
                from azure.keyvault.secrets import SecretClient
                from azure.identity import DefaultAzureCredential
                import json

                # azure credentials from environment variable
                if os.getenv("AZURE_CREDENTIALS"):
                    # GitHub Actions with AZURE_CREDENTIALS JSON
                    azure_creds = json.loads(os.getenv("AZURE_CREDENTIALS", "{}"))
                    from azure.identity import ClientSecretCredential

                    credential = ClientSecretCredential(
                        tenant_id=azure_creds["tenantId"],
                        client_id=azure_creds["clientId"],
                        client_secret=azure_creds["clientSecret"],
                    )
                else:
                    # DefaultAzureCredential (Managed Identity, Azure CLI, Env vars)
                    credential = DefaultAzureCredential()

                self._client = SecretClient(vault_url=self.key_vault_url, credential=credential)

            except ImportError as e:
                raise ImportError(f"Azure SDK not installed: {e}")
        return self._client

    def _get_secret(self, secret_name: str) -> str:
        """Get a secret from Azure Key Vault"""
        try:
            client = self._get_client()
            secret = client.get_secret(secret_name)
            return secret.value
        except Exception as e:
            logging.error(f"Failed to retrieve secret '{secret_name}': {e}")
            return ""

    def load_config(self) -> ConfigurationData:
        """Load configuration from Azure Key Vault"""
        # Get base directory for relative paths
        base_dir = os.path.dirname(os.path.abspath(__file__ + "/../"))

        # Key Vault secret mapping
        secret_mapping = {
            "confluence_private_api_key": "confluence-private-api-key",
            "confluence_space_key": "confluence-space-key",
            "confluence_space_name": "confluence-space-name",
            "confluence_email_address": "confluence-email-address",
            "openrouter_api_key": "openrouter-api-key",
        }

        # Build configuration dict
        config_dict = {}
        for field, secret_name in secret_mapping.items():
            config_dict[field] = self._get_secret(secret_name)

        # Add path configurations
        config_dict.update(
            {
                "db_path": os.getenv("DB_PATH", os.path.join(base_dir, "data/users.db")),
                "persist_directory": os.getenv("PERSIST_DIRECTORY", os.path.join(base_dir, "db")),
            }
        )

        return ConfigurationData(**config_dict)  # ty : ignore

    def get_provider_name(self) -> str:
        return f"Azure Key Vault ({self.key_vault_url})"


class ConfigurationManager:
    """Main configuration manager that handles different providers"""

    def __init__(self):
        self._config: Optional[ConfigurationData] = None
        self._provider: Optional[ConfigProvider] = None

    def initialize(self, provider: ConfigProvider) -> bool:
        """Initialize the configuration manager with a provider"""
        try:
            self._provider = provider
            self._config = provider.load_config()

            if not self._config.validate():
                logging.error("Configuration validation failed")
                return False

            logging.info(f"Configuration loaded successfully from: {provider.get_provider_name()}")
            return True

        except Exception as e:
            logging.error(f"Failed to initialize configuration: {e}")
            return False

    def auto_initialize(self) -> bool:
        """Auto-detect and initialize the appropriate configuration provider"""
        # Check if we're running in Azure (presence of specific environment variables)
        if self._is_azure_environment():
            key_vault_url = os.getenv("KEY_VAULT_URL")
            if key_vault_url:
                logging.info("Azure environment detected, using Key Vault configuration")
                provider = AzureKeyVaultConfigProvider(key_vault_url)
                return self.initialize(provider)
            else:
                logging.warning("Azure environment detected but KEY_VAULT_URL not set, falling back to local config")

        # Default to local configuration
        logging.info("Using local configuration")
        provider = LocalConfigProvider()
        return self.initialize(provider)

    def _is_azure_environment(self) -> bool:
        """Detect if we're running in Azure environment"""
        azure_indicators = [
            "WEBSITE_SITE_NAME",  # Azure App Service
            "AZURE_CLIENT_ID",  # Managed Identity
            "MSI_ENDPOINT",  # Managed Service Identity
        ]
        return any(os.getenv(indicator) for indicator in azure_indicators)

    @property
    def config(self) -> ConfigurationData:
        """Get the current configuration"""
        if self._config is None:
            raise RuntimeError("Configuration not initialized. Call initialize() or auto_initialize() first.")
        return self._config

    @property
    def provider_name(self) -> str:
        """Get the name of the current provider"""
        if self._provider is None:
            return "Not initialized"
        return self._provider.get_provider_name()

    def get_debug_info(self) -> Dict[str, str]:
        """Get debug information about the configuration"""
        if self._config is None:
            return {"status": "Not initialized"}

        # Mask sensitive information
        api_key = self._config.confluence_private_api_key
        api_key_display = f"*****{api_key[-5:]}" if api_key else "Not defined"

        openrouter_key = self._config.openrouter_api_key
        openrouter_display = f"*****{openrouter_key[-5:]}" if openrouter_key else "Not defined"

        return {
            "provider": self.provider_name,
            "confluence_url": self._config.confluence_space_name,
            "space_key": self._config.confluence_space_key,
            "user_email": self._config.confluence_email_address,
            "confluence_api_key": api_key_display,
            "openrouter_api_key": openrouter_display,
            "persist_directory": self._config.persist_directory,
            "db_path": self._config.db_path,
        }


# Global configuration manager instance
_config_manager = None
_config_initialized = False


def _ensure_config_initialized() -> Optional[ConfigurationManager]:
    """Ensure configuration is initialized and return the manager"""
    global _config_manager, _config_initialized

    if not _config_initialized:
        _config_manager = ConfigurationManager()
        success = _config_manager.auto_initialize()
        if not success:
            raise RuntimeError(
                "Failed to initialize configuration. Check your environment variables or Azure Key Vault access."
            )
        _config_initialized = True
        logging.info("Configuration successfully initialized")

    return _config_manager


def get_config() -> ConfigurationData:
    """Get the global configuration instance"""
    manager = _ensure_config_initialized()
    return manager.config


def initialize_config(provider: Optional[ConfigProvider] = None) -> bool:
    """Initialize the global configuration"""
    global _config_manager, _config_initialized

    _config_manager = ConfigurationManager()
    if provider:
        success = _config_manager.initialize(provider)
    else:
        success = _config_manager.auto_initialize()

    _config_initialized = success
    return success


def get_debug_info() -> Dict[str, str]:
    """Get debug information about the current configuration"""
    manager = _ensure_config_initialized()
    return manager.get_debug_info()


def reset_config():
    """Reset configuration for testing purposes"""
    global _config_manager, _config_initialized
    _config_manager = None
    _config_initialized = False
