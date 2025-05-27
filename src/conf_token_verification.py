import os
import dotenv
from pathlib import Path
from atlassian import Confluence
from atlassian.errors import ApiError


class ConfluenceTokenVerifier:
    """Class for validating Confluence API tokens and managing Confluence connections."""

    def __init__(self, env_path=None):
        """Initialize the verifier with an optional custom environment file path.
        Args:
            env_path (str or Path, optional): Path to the .env file. If None, will use the default path.
        """
        if env_path is None:
            # Default path is the .env file in the project root directory
            self.env_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / ".env"
        else:
            self.env_path = Path(env_path)

        # Initialize instance variables
        self.api_key = None
        self.username = None
        self.space_key = None
        self.base_url = None
        self.confluence = None

    def load_environment_variables(self):
        """Load environment variables from the .env file.
        Returns:
            tuple: (success, error_message)
                - success (bool): True if environment variables were loaded successfully, False otherwise
                - error_message (str): Error message if loading failed, None otherwise
        """
        # Check if .env file exists
        if not self.env_path.exists():
            return False, f"Environment file not found at {self.env_path}"

        # Read .env file directly to check if the token is present
        try:
            with open(self.env_path, "r") as f:
                env_content = f.read()
                if "CONFLUENCE_PRIVATE_API_KEY" not in env_content:
                    return False, "CONFLUENCE_PRIVATE_API_KEY not found in .env file"
        except Exception as e:
            return False, f"Error reading .env file: {str(e)}"

        # Load environment variables from .env file
        dotenv.load_dotenv(self.env_path, override=True)

        # Get environment variables after forced reload
        self.api_key = os.getenv("CONFLUENCE_PRIVATE_API_KEY")
        self.username = os.getenv("CONFLUENCE_EMAIL_ADRESS")
        self.space_key = os.getenv("CONFLUENCE_SPACE_KEY")
        self.base_url = os.getenv("CONFLUENCE_SPACE_NAME")

        # Check if required environment variables are set
        return self._validate_environment_variables()

    def _validate_environment_variables(self):
        """Validate that all required environment variables are set.
        Returns:
            tuple: (is_valid, error_message)
        """
        if not all([self.api_key, self.username, self.base_url, self.space_key]):
            missing_vars = []
            if not self.api_key:
                missing_vars.append("CONFLUENCE_PRIVATE_API_KEY")
            if not self.username:
                missing_vars.append("CONFLUENCE_EMAIL_ADRESS")
            if not self.base_url:
                missing_vars.append("CONFLUENCE_SPACE_NAME")
            if not self.space_key:
                missing_vars.append("CONFLUENCE_SPACE_KEY")

            return False, f"Missing required environment variables: {', '.join(missing_vars)}"

        # Ensure the URL is in the correct format (without the specific path)
        if "/wiki" in self.base_url:
            self.base_url = self.base_url.split("/wiki")[0]

        # Verify that the parameters are correct
        if not self.base_url or not self.base_url.startswith("http"):
            return False, f"Invalid Confluence URL: {self.base_url}"

        return True, None

    def create_confluence_client(self):
        """Create a Confluence client instance using the loaded credentials.
        Returns:
            Confluence: The Confluence client instance
        """
        self.confluence = Confluence(
            url=self.base_url,
            username=self.username,
            password=self.api_key,
            cloud=True,
        )
        return self.confluence

    def test_connection(self):
        """Test the connection to Confluence by making a simple API call.
        Returns:
            tuple: (is_successful, error_message)
        """
        if not self.confluence:
            self.create_confluence_client()

        try:
            # Make a simple API call to verify the token
            # Getting space info is a lightweight operation
            self.confluence.get_space(self.space_key)
            return True, None

        except ApiError as e:
            error_str = str(e).lower()
            if e.reason == "Unauthorized" or "unauthorized" in error_str or "401" in error_str:
                return False, "Confluence API token is invalid or expired. Please generate a new token."
            elif e.reason == "Forbidden" or "forbidden" in error_str or "permission" in error_str or "403" in error_str:
                return False, "Insufficient permissions with the provided Confluence API token."
            else:
                return False, f"Confluence API error: {str(e)}"

        except Exception as e:
            return False, f"Error validating Confluence token: {str(e)}"

    def validate_token(self):
        """Validate the Confluence API token by loading environment variables and testing the connection.
        Returns:
            tuple: (is_valid, error_message)
                - is_valid (bool): True if token is valid, False otherwise
                - error_message (str): Error message if token is invalid, None otherwise
        """
        # Load and validate environment variables
        success, error = self.load_environment_variables()
        if not success:
            return False, error

        # Test the connection
        return self.test_connection()


def validate_confluence_token():
    """Validates the Confluence API token by attempting to make a simple API call."""
    verifier = ConfluenceTokenVerifier()
    return verifier.validate_token()
