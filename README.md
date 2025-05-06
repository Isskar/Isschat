# RAG Chatbot with Confluence

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Isskar/RAG-Chatbot-with-Confluence/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

Un chatbot intelligent qui interagit avec la base de connaissances Confluence de Kanbios en utilisant la technologie RAG (Retrieval-Augmented Generation).

## Fonctionnalités

- Recherche sémantique dans la documentation Confluence
- Interface conversationnelle intuitive avec Streamlit
- Mise en cache des embeddings pour des performances accrues
- Tableau de bord d'administration
- Système d'authentification utilisateur
- Historique des requêtes et feedback utilisateur
- Analyse des performances et des interactions

## Installation

### Prérequis
- Python 3.10+
- Compte Confluence avec permissions API
- Ollama installé localement

### Configuration

1. **Cloner le dépôt**
   ```bash
   git clone https://github.com/Isskar/RAG-Chatbot-with-Confluence.git
   cd RAG-Chatbot-with-Confluence
   ```

2. **Créer un environnement virtuel**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Sur Linux/Mac
   # .\.venv\Scripts\activate  # Sur Windows
   ```

3. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configurer les variables d'environnement**
   Créez un fichier `.env` à la racine du projet avec les variables suivantes :
   ```env
   # Clé API de votre compte Confluence (récupérable sur :
   # https://id.atlassian.com/manage-profile/security/api-tokens)
   CONFLUENCE_PRIVATE_API_KEY=votre_api_key
   
   # URL de votre espace Confluence
   CONFLUENCE_SPACE_NAME=https://kanbios.atlassian.net
   
   # Votre email de connexion
   EMAIL_ADRESS=prenom.nomfamilles@isskar.fr
   ```

## Lancement

1. **Démarrer le serveur Ollama** (dans un terminal séparé)
   ```bash
   ollama serve
   ```

2. **Lancer l'application Streamlit**
   ```bash
   streamlit run src/streamlit.py
   ```

3. **Accéder à l'application**
   Ouvrez votre navigateur à l'adresse : http://localhost:8501

## Architecture

```
RAG-Chatbot-with-Confluence/
├── data/                    # Dossier pour les données persistantes
│   ├── cache/               # Cache des embeddings
│   └── history.db           # Base de données SQLite pour l'historique
├── src/
│   ├── auth.py             # Gestion de l'authentification
│   ├── help_desk.py         # Logique principale du chatbot
│   ├── streamlit.py         # Interface utilisateur Streamlit
│   ├── features_integration.py # Intégration des fonctionnalités avancées
│   └── ...
├── .env.example           # Exemple de configuration
├── requirements.txt        # Dépendances Python
└── README.md              # Ce fichier
```

## Fonctionnalités avancées

- **Analyse conversationnelle** : Suivi des interactions utilisateur
- **Suivi des performances** : Mesure des temps de réponse et précision
- **Système de feedback** : Évaluation des réponses par les utilisateurs
- **Historique des requêtes** : Consultation des recherches précédentes
- **Suggestion de questions** : Propositions de questions connexes

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.  
Added in [commit f801768](https://github.com/Isskar/Isschat/commit/f801768).

---

Développé par Nicolas Lambropoulos
