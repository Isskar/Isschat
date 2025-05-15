import os
import sys
from dotenv import load_dotenv
from atlassian import Confluence

# Load environment variables
load_dotenv()

# Retrieve connection information
confluence_url = os.getenv("CONFLUENCE_SPACE_NAME")
api_key = os.getenv("CONFLUENCE_PRIVATE_API_KEY")
username = os.getenv("CONFLUENCE_EMAIL_ADRESS")
space_key = os.getenv("CONFLUENCE_SPACE_KEY")

# Display information (without the complete key)
print(f"==== CONNECTION INFORMATION ====")
print(f"URL: {confluence_url}")
print(f"Username: {username}")
print(f"Space Key: {space_key}")
print(f"API Key: {'*' * 5}{api_key[-5:] if api_key else 'Not defined'}")

# Make sure the URL is in the correct format (without the specific path)
if "/wiki" in confluence_url:
    base_url = confluence_url.split("/wiki")[0]
    print(f"Adjusted URL: {base_url}")
else:
    base_url = confluence_url

try:
    print("\n==== ATTEMPTING CONNECTION TO CONFLUENCE ====")
    # Create a Confluence instance
    confluence = Confluence(
        url=base_url,
        username=username,
        password=api_key,
        cloud=True  # Specify that it's a cloud instance
    )
    
    # Test the connection by retrieving spaces
    print("Connection test: retrieving spaces...")
    spaces = confluence.get_all_spaces()
    print(f"Connection successful! {len(spaces)} spaces found.")
    
    # Test retrieving pages in the specified space
    print(f"\nRetrieving pages in space {space_key}...")
    pages = confluence.get_all_pages_from_space(space_key)
    print(f"Retrieval successful! {len(pages)} pages found.")
    
    # Display some information about the pages
    if pages:
        print("\nHere are the first 3 pages:")
        for i, page in enumerate(pages[:3]):
            print(f"  {i+1}. {page.get('title', 'Untitled')} (ID: {page.get('id', 'N/A')})")
    
except Exception as e:
    print(f"\n==== CONNECTION ERROR ====")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    import traceback
    print("\nComplete error trace:")
    print(traceback.format_exc())
    sys.exit(1)
