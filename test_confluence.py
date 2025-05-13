import os
import sys
from dotenv import load_dotenv
from atlassian import Confluence

# Charger les variables d'environnement
load_dotenv()

# Récupérer les informations de connexion
confluence_url = os.getenv("CONFLUENCE_SPACE_NAME")
api_key = os.getenv("CONFLUENCE_PRIVATE_API_KEY")
username = os.getenv("CONFLUENCE_EMAIL_ADRESS")
space_key = os.getenv("CONFLUENCE_SPACE_KEY")

# Afficher les informations (sans la clé complète)
print("==== INFORMATIONS DE CONNEXION ====")
print(f"URL: {confluence_url}")
print(f"Username: {username}")
print(f"Space Key: {space_key}")
print(f"API Key: {'*' * 5}{api_key[-5:] if api_key else 'Non définie'}")

# S'assurer que l'URL est au bon format (sans le chemin spécifique)
if "/wiki" in confluence_url:
    base_url = confluence_url.split("/wiki")[0]
    print(f"URL ajustée: {base_url}")
else:
    base_url = confluence_url

try:
    print("\n==== TENTATIVE DE CONNEXION À CONFLUENCE ====")
    # Créer une instance Confluence
    confluence = Confluence(
        url=base_url,
        username=username,
        password=api_key,
        cloud=True,  # Spécifier que c'est une instance cloud
    )

    # Tester la connexion en récupérant les espaces
    print("Test de connexion: récupération des espaces...")
    spaces = confluence.get_all_spaces()
    print(f"Connexion réussie! {len(spaces)} espaces trouvés.")

    # Tester la récupération des pages dans l'espace spécifié
    print(f"\nRécupération des pages dans l'espace {space_key}...")
    pages = confluence.get_all_pages_from_space(space_key)
    print(f"Récupération réussie! {len(pages)} pages trouvées.")

    # Afficher quelques informations sur les pages
    if pages:
        print("\nVoici les 3 premières pages:")
        for i, page in enumerate(pages[:3]):
            print(f"  {i + 1}. {page.get('title', 'Sans titre')} (ID: {page.get('id', 'N/A')})")

except Exception as e:
    print("\n==== ERREUR DE CONNEXION ====")
    print(f"Type d'erreur: {type(e).__name__}")
    print(f"Message d'erreur: {str(e)}")
    import traceback

    print("\nTrace d'erreur complète:")
    print(traceback.format_exc())
    sys.exit(1)
