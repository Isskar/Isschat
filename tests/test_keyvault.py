import os
from unittest.mock import Mock, patch
from src.config.keyvault import KeyVaultManager, get_key_vault_manager, get_secret


class TestKeyVaultManager:
    def setup_method(self):
        global _key_vault_manager
        import src.config.keyvault

        src.config.keyvault._key_vault_manager = None

    @patch.dict(os.environ, {"ENVIRONMENT": "development", "KEY_VAULT_URL": ""})
    def test_init_local_environment(self):
        manager = KeyVaultManager()

        assert not manager.is_azure
        assert not manager.use_key_vault
        assert manager._client is None

    @patch.dict(os.environ, {"ENVIRONMENT": "production", "KEY_VAULT_URL": "https://test.vault.azure.net/"})
    @patch("src.config.keyvault.ManagedIdentityCredential")
    @patch("src.config.keyvault.SecretClient")
    def test_init_azure_environment_success(self, mock_secret_client, mock_credential):
        mock_client_instance = Mock()
        mock_secret_client.return_value = mock_client_instance

        manager = KeyVaultManager()

        assert manager.is_azure
        assert manager.use_key_vault
        assert manager._client == mock_client_instance

    @patch.dict(os.environ, {"ENVIRONMENT": "production", "KEY_VAULT_URL": "https://test.vault.azure.net/"})
    @patch("src.config.keyvault.ManagedIdentityCredential")
    @patch("src.config.keyvault.SecretClient")
    def test_init_azure_environment_failure(self, mock_secret_client, mock_credential):
        mock_secret_client.side_effect = Exception("Connection failed")

        manager = KeyVaultManager()

        assert manager.is_azure
        assert not manager.use_key_vault
        assert manager._client is None

    @patch.dict(os.environ, {"TEST_SECRET": "local_value"})
    def test_get_secret_from_env(self):
        manager = KeyVaultManager()

        result = manager.get_secret("TEST_SECRET")

        assert result == "local_value"

    def test_get_secret_default_value(self):
        manager = KeyVaultManager()

        result = manager.get_secret("NONEXISTENT_SECRET", "default_value")

        assert result == "default_value"

    @patch.dict(os.environ, {"ENVIRONMENT": "production", "KEY_VAULT_URL": "https://test.vault.azure.net/"})
    @patch("src.config.keyvault.ManagedIdentityCredential")
    @patch("src.config.keyvault.SecretClient")
    def test_get_secret_from_keyvault(self, mock_secret_client, mock_credential):
        mock_client_instance = Mock()
        mock_secret = Mock()
        mock_secret.value = "keyvault_value"
        mock_client_instance.get_secret.return_value = mock_secret
        mock_secret_client.return_value = mock_client_instance

        manager = KeyVaultManager()
        result = manager.get_secret("TEST_SECRET")

        assert result == "keyvault_value"
        mock_client_instance.get_secret.assert_called_once_with("TEST-SECRET")

    def test_cache_functionality(self):
        manager = KeyVaultManager()

        # Premier appel
        with patch.dict(os.environ, {"TEST_SECRET": "cached_value"}):
            result1 = manager.get_secret("TEST_SECRET")

        # Deuxième appel (devrait utiliser le cache)
        result2 = manager.get_secret("TEST_SECRET")

        assert result1 == "cached_value"
        assert result2 == "cached_value"
        assert "TEST_SECRET" in manager._local_cache

    def test_clear_cache(self):
        manager = KeyVaultManager()

        with patch.dict(os.environ, {"TEST_SECRET": "value"}):
            manager.get_secret("TEST_SECRET")

        assert "TEST_SECRET" in manager._local_cache

        manager.clear_cache()

        assert len(manager._local_cache) == 0


class TestSingletonFunctions:
    def setup_method(self):
        import src.config.keyvault

        src.config.keyvault._key_vault_manager = None

    def test_get_key_vault_manager_singleton(self):
        manager1 = get_key_vault_manager()
        manager2 = get_key_vault_manager()

        assert manager1 is manager2

    @patch.dict(os.environ, {"TEST_SECRET": "function_value"})
    def test_get_secret_function(self):
        result = get_secret("TEST_SECRET")

        assert result == "function_value"

    def test_get_secret_function_with_default(self):
        result = get_secret("NONEXISTENT_SECRET", "default")

        assert result == "default"
