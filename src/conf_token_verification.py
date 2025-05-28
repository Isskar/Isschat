import os
import dotenv
from pathlib import Path
from atlassian import Confluence
from atlassian.errors import ApiError


class ConfluenceTokenVerifier:
    """Simple class for validating if a Confluence API token is valid or expired."""

    def __init__(self):
        # Get the .env file path in the project root directory
        self.env_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / ".env"

    def verify_token(self):
        """Verify if the Confluence API token is valid or expired."""
        if not self.env_path.exists():
            return False, "Environment file not found"

        dotenv.load_dotenv(self.env_path, override=True)

        api_key = os.getenv("CONFLUENCE_PRIVATE_API_KEY")
        username = os.getenv("CONFLUENCE_EMAIL_ADDRESS")
        base_url = os.getenv("CONFLUENCE_SPACE_NAME")
        space_key = os.getenv("CONFLUENCE_SPACE_KEY")

        # Check if API key exists
        if not api_key:
            return False, "Confluence API token not found in environment variables"

        # Check if other required variables exist
        if not all([username, base_url, space_key]):
            return False, "Missing required Confluence configuration variables"

        # Create Confluence client and test connetion
        try:
            confluence = Confluence(url=base_url, username=username, password=api_key, cloud=True)

            # Make a simple API call to verify the token
            confluence.get_space(space_key)
            return True, None

        except ApiError as e:
            error_str = str(e).lower()
            if "unauthorized" in error_str or "401" in error_str or e.reason == "Unauthorized":
                return False, "Confluence API token is invalid or expired. Please generate a new token."
            elif "forbidden" in error_str or "403" in error_str or e.reason == "Forbidden":
                return False, "Insufficient permissions with the provided Confluence API token."
            else:
                return False, f"Confluence API error: {str(e)}"

        except Exception as e:
            return False, f"Error validating Confluence token: {str(e)}"


def validate_confluence_token():
    """Validates the Confluence API token by attempting to make a simple API call."""
    return ConfluenceTokenVerifier().verify_token()
