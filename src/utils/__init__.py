"""
Utility modules for the application.
"""

from .error_handling import (
    setup_logging,
    ConfluenceConnectionError,
    OpenRouterError,
    VectorStoreError,
    handle_errors,
    retry_api_call,
    GracefulDegradation
)

__all__ = [
    'setup_logging',
    'ConfluenceConnectionError',
    'OpenRouterError',
    'VectorStoreError',
    'handle_errors',
    'retry_api_call',
    'GracefulDegradation'
]
