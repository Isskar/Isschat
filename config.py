import os
from dotenv import load_dotenv

load_dotenv()

CONFLUENCE_PRIVATE_API_KEY = os.getenv("CONFLUENCE_PRIVATE_API_KEY", "")
CONFLUENCE_API_KEY = os.getenv("CONFLUENCE_PRIVATE_API_KEY", "")
CONFLUENCE_SPACE_KEY = os.getenv("CONFLUENCE_SPACE_KEY", "")
CONFLUENCE_SPACE_NAME = os.getenv("CONFLUENCE_SPACE_NAME", "")
CONFLUENCE_USERNAME = os.getenv("CONFLUENCE_EMAIL_ADRESS", "")
CONFLUENCE_EMAIL_ADRESS = os.getenv("CONFLUENCE_EMAIL_ADRESS", "")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.getenv("DB_PATH", os.path.join(BASE_DIR, "data/users.db"))
PERSIST_DIRECTORY = os.getenv("PERSIST_DIRECTORY", os.path.join(BASE_DIR, "db"))
EVALUATION_DATASET = os.getenv("EVALUATION_DATASET", os.path.join(BASE_DIR, "data/evaluation_dataset.tsv"))
