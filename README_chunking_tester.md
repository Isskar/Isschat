# Testeur de Chunking Confluence avec Métadonnées Enrichies

## Description

Ce script permet de récupérer des pages Confluence et de tester le chunking avec des métadonnées enrichies comprenant l'auteur, contributeurs, chemin hiérarchique complet et informations temporelles.

## Fonctionnalités

- **Recherche de pages** : Recherche par titre dans votre espace Confluence
- **Chunking intelligent** : Utilise la stratégie `semantic_hierarchical` optimisée
- **Métadonnées enrichies** : Extraction automatique de l'auteur, contributeurs, hiérarchie
- **Contexte enrichi** : Chaque chunk contient le chemin complet, auteur et dates
- **Visualisation détaillée** : Affichage des chunks avec toutes les métadonnées
- **Debug mode** : Affichage du contenu complet d'un chunk pour analyse
- **Sauvegarde** : Export des résultats enrichis en JSON

## Stratégie de Chunking

**`semantic_hierarchical` avec enrichissement** : La stratégie de production qui :
- Détecte correctement les tableaux, listes et structures
- Adapte les tailles de chunks selon le type de contenu (1000-2000 tokens)
- Préserve la hiérarchie des en-têtes
- Utilise un comptage de tokens précis
- **NOUVEAU** : Enrichit automatiquement avec l'auteur et le chemin hiérarchique
- **NOUVEAU** : Ajoute les dates de création/modification
- **NOUVEAU** : Inclut les contributeurs et informations de versioning

## Prérequis

### Variables d'environnement requises

Le script charge automatiquement les variables depuis le fichier `.env` :

```bash
# Fichier .env
CONFLUENCE_PRIVATE_API_KEY="votre_api_key"
CONFLUENCE_SPACE_KEY="votre_space_key"
CONFLUENCE_SPACE_NAME="votre_space_name"
CONFLUENCE_EMAIL_ADDRESS="votre_email"
```

### Dépendances

Le script utilise les composants existants du projet :
- `ingestion.connectors.confluence_connector` (avec enrichissement intégré)
- `ingestion.processors.chunker` (contexte enrichi automatique)
- `core.interfaces.Document`
- `confluence_enrichment_patch` (enrichisseur de métadonnées)

## Usage

```bash
# Lancer le script depuis le répertoire racine
uv run python src/confluence_chunking_tester.py

# Ou avec PYTHONPATH
PYTHONPATH=src python confluence_chunking_tester.py
```

### Exemple d'utilisation

1. **Lancement du script**
   ```
   🚀 Testeur de chunking Confluence
   ✅ Connexion Confluence configurée
   ✅ Enrichisseur de métadonnées initialisé
   ✅ API Confluence accessible
   ```

2. **Recherche de page**
   ```
   📝 Entrez le titre de la page (ou une partie): Isschat
   🔍 Recherche des pages avec le titre: Isschat
   ✅ 2 page(s) trouvée(s)
     1. Stratégies d'évaluation pour Isschat
     2. Isschat
   ```

3. **Sélection d'option**
   ```
   🎯 Options disponibles:
     1. Chunker avec semantic_hierarchical
     2. Test debug - afficher contenu complet d'un chunk
   
   Choisissez une option: 1
   ```

4. **Enrichissement automatique**
   ```
   🔍 Enrichissement des métadonnées...
     ✅ Chunk 1 enrichi
     ✅ Chunk 2 enrichi
     ✅ Chunk 3 enrichi (tableau détecté)
   ✅ Enrichissement terminé
   ```

5. **Résultats enrichis**
   ```
   📄 CHUNK 1
   Taille: 377 caractères, 25 tokens
   Type: text
   Auteur: Nicolas LAMBROPOULOS
   Chemin: ISSKAR Home > Missions > Isschat
   Créé: 2025-05-21
   
   Contexte enrichi:
   [Document: ISSKAR Home > Missions > Isschat | Espace: ISSKAR | 
    Auteur: Nicolas LAMBROPOULOS | Créé: 2025-05-21 | 
    Modifié: 2025-05-21 | URL: ... | Section: Identification du projet | 
    Type: text | Source: confluence]
   
   Contenu réel:
   # Identification du projet
   * Nom du projet : Isschat
   * Client : Isskar (usage interne)
   ```

## Sortie du Script

### Informations par chunk

Pour chaque chunk, le script affiche :
- **Index** : Numéro du chunk
- **Taille** : Nombre de caractères et tokens
- **Type** : Type de contenu (text, table, list, code)
- **Auteur** : Nom de l'auteur de la page
- **Chemin** : Chemin hiérarchique complet (ex: ISSKAR Home > Missions > Isschat)
- **Créé** : Date de création
- **Section** : Section dans le document
- **Contenu enrichi** : Contenu avec contexte complet

### Métadonnées enrichies incluses

**Métadonnées de base :**
- `section_path` : Chemin hiérarchique dans le document
- `content_type` : Type de contenu détecté
- `token_count` : Nombre de tokens précis
- `chunk_index` : Index du chunk

**Nouvelles métadonnées enrichies :**
- `author_id` : ID unique de l'auteur
- `author_name` : Nom complet de l'auteur  
- `author_email` : Email de l'auteur
- `hierarchy_breadcrumb` : Chemin hiérarchique complet
- `parent_pages` : Liste des pages parentes
- `page_depth` : Profondeur dans la hiérarchie
- `created_date` : Date de création ISO
- `last_modified_date` : Date de dernière modification
- `last_modified_by` : Nom du dernier modificateur
- `version_number` : Numéro de version de la page
- `contributors` : Liste des contributeurs
- `contributors_count` : Nombre de contributeurs
- `labels` : Tags/labels attachés à la page
- `has_attachments` : Présence de pièces jointes
- `attachments_count` : Nombre de pièces jointes

## Sauvegarde

Les résultats peuvent être sauvegardés au format JSON :

```json
{
  "semantic_hierarchical": {
    "document_title": "Isschat",
    "total_chunks": 6,
    "processing_time": 0.101907,
    "chunks": [
      {
        "index": 1,
        "content": "[Document: ISSKAR Home > Missions > Isschat | Espace: ISSKAR | Auteur: Nicolas LAMBROPOULOS | Créé: 2025-05-21 | Modifié: 2025-05-21 | URL: ... | Section: Identification du projet | Type: text | Source: confluence]\n\n# Identification du projet...",
        "metadata": {
          "title": "Isschat",
          "author_name": "Nicolas LAMBROPOULOS",
          "author_email": "nicolas.lambropoulos@isskar.fr",
          "hierarchy_breadcrumb": "ISSKAR Home > Missions > Isschat",
          "parent_pages": [
            {"id": "180256983", "title": "ISSKAR Home"},
            {"id": "228818947", "title": "Missions"}
          ],
          "page_depth": 2,
          "created_date": "2025-05-21T12:43:27.873Z",
          "version_number": 8,
          "contributors_count": 0,
          "has_attachments": true,
          "attachments_count": 3,
          "content_type": "text",
          "token_count": 25
        },
        "size_chars": 377,
        "size_tokens": 25
      }
    ]
  }
}
```

## Conseils d'utilisation

1. **Test de pages variées** : Testez sur des pages avec et sans tableaux pour voir l'adaptation
2. **Mode debug** : Utilisez l'option 2 pour voir le contenu complet des chunks avec contexte enrichi
3. **Analyse des métadonnées** : Examinez le `content_type` détecté (text, table, list, code) et les nouvelles métadonnées
4. **Vérification de la hiérarchie** : Contrôlez que le `hierarchy_breadcrumb` reflète bien la structure Confluence
5. **Traçabilité** : Utilisez `author_name` et `created_date` pour identifier les sources et leur fraîcheur
6. **Collaborateurs** : Consultez `contributors_count` pour évaluer la richesse collaborative du contenu

## Dépannage

### Erreurs communes

- **Variables manquantes** : Vérifiez que toutes les variables d'environnement sont définies
- **Connexion Confluence échouée** : Vérifiez vos identifiants et permissions
- **API non accessible** : L'enrichissement sera désactivé mais le chunking fonctionnera
- **Page non trouvée** : Utilisez des mots-clés plus larges pour la recherche
- **Métadonnées manquantes** : Certaines pages peuvent avoir des métadonnées limitées

### Logs de débogage

Le script affiche des logs détaillés pour suivre le processus :
- ✅ Succès (connexion, enrichissement, chunking)
- ❌ Erreurs (authentification, API, parsing)
- 🔍 Recherche et extraction
- 🔄 Traitement et enrichissement  
- 📊 Résultats et statistiques
- ⚠️ Avertissements (API indisponible, métadonnées partielles)

## Intégration en Production

Ce testeur utilise les mêmes composants que le système de production :
- **ConfluenceConnector** avec enrichissement automatique
- **ConfluenceChunker** avec contexte enrichi
- **Même stratégie** `semantic_hierarchical` qu'en production

Les résultats du test sont donc directement représentatifs de ce qui sera indexé dans votre base vectorielle de production.