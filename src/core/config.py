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

    # Embeddings configuration
    embeddings_model: str = "sentence-transformers/all-MiniLM-L12-v2"
    embeddings_device: str = "cpu"
    embeddings_batch_size: int = 32
    embeddings_normalize: bool = True
    embeddings_trust_remote_code: bool = False

    # Vector DB configuration
    vector_db_type: str = "faiss"
    search_k: int = 3
    search_fetch_k: int = 5

    # LLM Config
    generator_model_name: str = "google/gemini-2.5-flash-preview-05-20"
    generator_temperature: float = 0.1
    generator_max_tokens: int = 512

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
        base_dir = os.getcwd()

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

        persist_dir = os.getenv("PERSIST_DIRECTORY", os.path.join(base_dir, "data", "vector_db"))

        config_dict.update(
            {
                "db_path": os.getenv("DB_PATH", os.path.join(base_dir, "data", "users.db")),
                "persist_directory": persist_dir,
            }
        )

        return ConfigurationData(**config_dict)  # ty : ignore

    def get_provider_name(self) -> str:
        return "Local (.env)"


class AzureKeyVaultConfigProvider(ConfigProvider):
    """Configuration provider for Azure Key Vault"""

    def __init__(self, key_vault_url: str):
        self.key_vault_url = key_vault_url
        self._client = None

    def _get_client(self):
        """Get Azure Key Vault client with managed identity or service principal"""
        if self._client is None:
            try:
                from azure.keyvault.secrets import SecretClient
                from azure.identity import DefaultAzureCredential, ClientSecretCredential

                # Check for Service Principal credentials (CI environment)
                client_id = os.getenv("AZURE_CLIENT_ID")
                client_secret = os.getenv("AZURE_CLIENT_SECRET")
                tenant_id = os.getenv("AZURE_TENANT_ID")

                if client_id and client_secret and tenant_id:
                    # Service Principal authentication (CI)
                    logging.info("Using Service Principal authentication for Key Vault")
                    credential = ClientSecretCredential(
                        tenant_id=tenant_id,
                        client_id=client_id,
                        client_secret=client_secret,
                    )
                else:
                    # DefaultAzureCredential (Managed Identity, Azure CLI, etc.)
                    logging.info("Using DefaultAzureCredential for Key Vault")
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
        base_dir = os.getcwd()

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
        # Use /tmp for Azure environment to ensure write permissions
        persist_dir = os.getenv("PERSIST_DIRECTORY", os.path.join(base_dir, "data", "vector_db"))

        config_dict.update(
            {
                "db_path": os.getenv("DB_PATH", os.path.join(base_dir, "data", "users.db")),
                "persist_directory": persist_dir,
            }
        )

        return ConfigurationData(**config_dict)  # ty : ignore

    def get_provider_name(self) -> str:
        return f"Azure Key Vault ({self.key_vault_url})"


class ContinuousIntegrationConfigProvider(ConfigProvider):
    """Configuration provider for Continuous Integration environment"""

    def load_config(self) -> ConfigurationData:
        """Load configuration for CI environment"""
        base_dir = os.getcwd()
        persist_dir = os.getenv("PERSIST_DIRECTORY", os.path.join(base_dir, "data", "vector_db"))
        db_path = os.getenv("DB_PATH", os.path.join(base_dir, "data", "users.db"))

        config_dict = {
            "confluence_private_api_key": "ci-dummy-key",
            "confluence_space_key": "ci-dummy-space",
            "confluence_space_name": "ci-dummy-space-name",
            "confluence_email_address": "ci-dummy@example.com",
            "openrouter_api_key": os.getenv("OPENROUTER_API_KEY", ""),
            "db_path": db_path,
            "persist_directory": persist_dir,
        }

        return ConfigurationData(**config_dict)  # ty : ignore

    def get_provider_name(self) -> str:
        return "Continuous Integration (CI)"


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
        environment = os.getenv("ENVIRONMENT", "").lower()

        if environment == "ci":
            logging.info("CI environment detected, using Service Principal with Key Vault")
            return self._initialize_ci_config()
        elif environment in ["production", "prod", "azure"]:
            logging.info("Production environment detected, using Managed Identity with Key Vault")
            return self._initialize_production_config()
        logging.info("Local environment detected, using .env configuration")
        return self._initialize_local_config()

    def _initialize_ci_config(self) -> bool:  # FIXME : Add secrets when CD deployed
        """Initialize configuration for CI environment"""
        provider = ContinuousIntegrationConfigProvider()
        return self.initialize(provider)

    def _initialize_production_config(self) -> bool:
        """Initialize configuration for production environment with Managed Identity"""
        key_vault_url = os.getenv("KEY_VAULT_URL")
        if key_vault_url:
            provider = AzureKeyVaultConfigProvider(key_vault_url)
            return self.initialize(provider)
        else:
            logging.warning("Azure environment detected but KEY_VAULT_URL not set, falling back to local config")
            return self._initialize_local_config()

    def _initialize_local_config(self) -> bool:
        """Initialize configuration for local development"""
        provider = LocalConfigProvider()
        return self.initialize(provider)

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

    @property
    def environment(self) -> str:
        """Get the current environment"""
        return os.getenv("ENVIRONMENT", "local").lower()

    @property
    def is_ci(self) -> bool:
        """Check if running in CI environment"""
        return self.environment == "ci"

    @property
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment in ["production", "prod", "azure"]

    @property
    def is_local(self) -> bool:
        """Check if running in local development environment"""
        return not (self.is_ci or self.is_production)

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
            "environment": self.environment,
            "provider": self.provider_name,
            "is_ci": str(self.is_ci),
            "is_production": str(self.is_production),
            "is_local": str(self.is_local),
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


def get_debug_info() -> Dict[str, str]:
    """Get debug information about the current configuration"""
    manager = _ensure_config_initialized()
    return manager.get_debug_info()


def reset_config():
    """Reset configuration for testing purposes"""
    global _config_manager, _config_initialized
    _config_manager = None
    _config_initialized = False
