"""
Error handling and logging utilities for the application.
"""
import logging
import sys
import traceback
from functools import wraps
from typing import Callable, TypeVar, Any, Optional, Type, Dict, List
from requests.exceptions import RequestException
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Type variable for generic function typing
F = TypeVar('F', bound=Callable[..., Any])

# Configure root logger
def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure the root logger with a standard format.
    
    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

# Custom exceptions
class ConfluenceConnectionError(Exception):
    """Raised when there's an error connecting to Confluence."""
    pass

class OpenRouterError(Exception):
    """Raised when there's an error with the OpenRouter API."""
    pass

class VectorStoreError(Exception):
    """Raised when there's an error with the vector store."""
    pass

# Decorator for error handling
def handle_errors(
    exceptions: Optional[Dict[Type[Exception], str]] = None,
    default_message: str = "An unexpected error occurred",
    fallback_return_value: Any = None,
    log_level: int = logging.ERROR,
    include_traceback: bool = False
) -> Callable[[F], F]:
    """
    A decorator to handle exceptions and provide user-friendly error messages.
    
    Args:
        exceptions: A dictionary mapping exception types to user-friendly messages
        default_message: Default message if the exception type is not found in the exceptions dict
        fallback_return_value: Value to return if an exception occurs
        log_level: Logging level for error messages
        include_traceback: Whether to include full traceback in logs
        
    Returns:
        The decorated function
    """
    if exceptions is None:
        exceptions = {}
    
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Get the logger for the module where the function is defined
                logger = logging.getLogger(func.__module__)
                
                # Log the error
                log_message = f"Error in {func.__qualname__}: {str(e)}"
                if include_traceback:
                    log_message += f"\n{traceback.format_exc()}"
                
                logger.log(log_level, log_message)
                
                # If fallback is provided, return it
                if fallback_return_value is not None:
                    return fallback_return_value
                
                # Otherwise, re-raise with user-friendly message
                user_message = exceptions.get(type(e), default_message)
                raise type(e)(f"{user_message}: {str(e)}") from e
                
        return wrapper  # type: ignore
    return decorator

# Retry decorator for API calls
def retry_api_call(
    max_retries: int = 3,
    initial_wait: float = 1.0,
    max_wait: float = 10.0,
    exceptions: Optional[List[Type[Exception]]] = None
) -> Callable[[F], F]:
    """
    A decorator to retry API calls with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_wait: Initial wait time in seconds
        max_wait: Maximum wait time in seconds
        exceptions: List of exceptions to retry on
        
    Returns:
        The decorated function
    """
    if exceptions is None:
        exceptions = [RequestException, ConnectionError, TimeoutError]
    
    return retry(
        stop=stop_after_attempt(max_retries + 1),
        wait=wait_exponential(multiplier=1, min=initial_wait, max=max_wait),
        retry=retry_if_exception_type(tuple(exceptions)),
        reraise=True
    )

# Context manager for graceful degradation
class GracefulDegradation:
    """
    Context manager for graceful degradation of functionality.
    If an error occurs, it will log the error and execute the fallback function.
    """
    def __init__(
        self,
        fallback_func: Callable[[Exception], Any],
        exceptions: Type[Exception] = Exception,
        log_errors: bool = True
    ):
        self.fallback_func = fallback_func
        self.exceptions = exceptions
        self.log_errors = log_errors
        self.logger = logging.getLogger(__name__)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and issubclass(exc_type, self.exceptions):
            if self.log_errors:
                self.logger.error(
                    "Error in graceful degradation context: %s\n%s",
                    str(exc_val),
                    ''.join(traceback.format_exception(exc_type, exc_val, exc_tb))
                )
            self.fallback_func(exc_val)
            return True  # Suppress the exception
        return False  # Re-raise any other exceptions
