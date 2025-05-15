import os
from dotenv import load_dotenv

# Charge les variables d'environnement depuis un fichier .env
load_dotenv()

# Variables Confluence
CONFLUENCE_PRIVATE_API_KEY = os.getenv("CONFLUENCE_PRIVATE_API_KEY")
CONFLUENCE_API_KEY = os.getenv("CONFLUENCE_PRIVATE_API_KEY")  # Alias pour compatibilité
CONFLUENCE_SPACE_KEY = os.getenv("CONFLUENCE_SPACE_KEY")
CONFLUENCE_SPACE_NAME = os.getenv("CONFLUENCE_SPACE_NAME")
CONFLUENCE_USERNAME = os.getenv("CONFLUENCE_EMAIL_ADRESS")  # Utilise l'email comme username
CONFLUENCE_EMAIL_ADRESS = os.getenv("CONFLUENCE_EMAIL_ADRESS")

# Chemins de fichiers et dossiers
# Utiliser des chemins absolus pour éviter les problèmes de résolution de chemin
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.getenv("DB_PATH", os.path.join(BASE_DIR, "data/users.db"))
PERSIST_DIRECTORY = os.getenv("PERSIST_DIRECTORY", os.path.join(BASE_DIR, "db"))
EVALUATION_DATASET = os.getenv("EVALUATION_DATASET", os.path.join(BASE_DIR, "data/evaluation_dataset.tsv"))
