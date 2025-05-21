import os
import dotenv
from pathlib import Path
from atlassian import Confluence
from atlassian.errors import ApiError


def validate_confluence_token():
    """
    Validates the Confluence API token by attempting to make a simple API call.
    Returns:
        tuple: (is_valid, error_message)
            - is_valid (bool): True if token is valid, False otherwise
            - error_message (str): Error message if token is invalid, None otherwise
    """

    # Force reload environment variables from .env file to avoid cached values
    env_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / ".env"

    # Check if .env file exists
    if not env_path.exists():
        return False, f"Environment file not found at {env_path}"

    # Read .env file directly to check if the token is present
    with open(env_path, "r") as f:
        env_content = f.read()
        if "CONFLUENCE_PRIVATE_API_KEY" not in env_content:
            return False, "CONFLUENCE_PRIVATE_API_KEY not found in .env file"

    # Load environment variables from .env file
    dotenv.load_dotenv(env_path, override=True)

    # Get environment variables after forced reload
    api_key = os.getenv("CONFLUENCE_PRIVATE_API_KEY")
    username = os.getenv("CONFLUENCE_EMAIL_ADRESS")
    space_key = os.getenv("CONFLUENCE_SPACE_KEY")
    base_url = os.getenv("CONFLUENCE_SPACE_NAME")

    # Check if required environment variables are set
    if not all([api_key, username, base_url, space_key]):
        missing_vars = []
        if not api_key:
            missing_vars.append("CONFLUENCE_PRIVATE_API_KEY")
        if not username:
            missing_vars.append("CONFLUENCE_EMAIL_ADRESS")
        if not base_url:
            missing_vars.append("CONFLUENCE_SPACE_NAME")
        if not space_key:
            missing_vars.append("CONFLUENCE_SPACE_KEY")

        return False, f"Missing required environment variables: {', '.join(missing_vars)}"

    # Ensure the URL is in the correct format (without the specific path)
    if "/wiki" in base_url:
        base_url = base_url.split("/wiki")[0]

    # Verify that the parameters are correct
    if not base_url or not base_url.startswith("http"):
        return False, f"Invalid Confluence URL: {base_url}"

    try:
        # Create a Confluence instance with the Atlassian API
        confluence = Confluence(
            url=base_url,
            username=username,
            password=api_key,
            cloud=True,
        )

        # Make a simple API call to verify the token
        # Getting space info is a lightweight operation
        confluence.get_space(space_key)

        # If we get here, the token is valid
        return True, None

    except ApiError as e:
        if e.status_code == 401:
            return False, "Confluence API token is invalid or expired. Please generate a new token."
        elif e.status_code == 403:
            return False, "Insufficient permissions with the provided Confluence API token."
        else:
            return False, f"Confluence API error: {str(e)}"

    except Exception as e:
        return False, f"Error validating Confluence token: {str(e)}"
