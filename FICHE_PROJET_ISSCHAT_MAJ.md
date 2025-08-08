# Fiche Projet Isschat - Mise à Jour 2025

## Identification du projet

**Nom du projet :** Isschat  
**Client :** Isskar (usage interne)  
**Responsable métier :** Vincent Fraillon  
**Responsable technique :** Nicolas Lambropoulos  
**Équipe de développement :** Nicolas Lambropoulos, Emin Calyaka  
**Dernière mise à jour :** Août 2025  

## Contexte et Description

### Description de l'activité client
Développement d'une plateforme conversationnelle intelligente d'entreprise pour faciliter l'accès à la documentation technique et aux connaissances stockées dans Confluence via une technologie RAG (Retrieval-Augmented Generation) avancée.

### Enjeux métier
- **Amélioration de l'accessibilité** des informations stockées dans Confluence
- **Réduction significative** du temps de recherche d'informations pour les collaborateurs  
- **Centralisation des connaissances** et facilitation de leur partage
- **Démocratisation de l'accès** aux connaissances techniques pour tous les collaborateurs

## Définition du projet

### Description technique
Plateforme conversationnelle intelligente basée sur une architecture RAG moderne utilisant :
- **Base vectorielle Weaviate** pour la recherche sémantique haute performance
- **Chunking hiérarchique** préservant la structure des documents
- **Pipeline RAG sémantique** avec reformulation de requêtes par LLM
- **Interface web Streamlit** multi-pages avec authentification Azure AD
- **CLI avancée** pour la gestion système et les requêtes
- **Système d'évaluation complet** multi-catégories

### Objectifs actualisés
1. **Réduction du temps de recherche** d'informations de 70%
2. **Démocratisation de l'accès** aux connaissances techniques via interface intuitive
3. **Amélioration continue** via système d'évaluation automatisé et feedback utilisateur
4. **Déploiement cloud** avec intégration Azure complète

### Utilisateurs finaux
- **Équipes techniques Isskar** (développeurs, architectes, DevOps)
- **Nouveaux collaborateurs** en phase d'onboarding
- **Équipes produit et management** 
- **Partenaires externes** avec accès restreint et contrôlé

## Architecture technique actuelle

### Stack technologique
```
Frontend:     Streamlit (interface web multi-pages)
Backend:      Python 3.12+ avec architecture modulaire
Vector DB:    Weaviate (HNSW indexing, recherche cosinus)
Embeddings:   intfloat/multilingual-e5-small
LLM:          Google Gemini 2.5 Flash via OpenRouter
Storage:      Azure Blob Storage + stockage local
Auth:         Azure AD + validation domaine entreprise
CLI:          Interface complète avec commandes avancées
```

### Architecture modulaire
```
src/
├── cli/                    # Interface ligne de commande complète
├── config/                 # Gestion configuration + Azure Key Vault
├── core/                   # Abstractions et interfaces métier
├── embeddings/             # Service d'embeddings multilingue
├── ingestion/              # Pipeline d'ingestion Confluence
├── rag/                    # Pipeline RAG + reformulation sémantique
├── storage/                # Abstraction stockage (local/Azure)
├── vectordb/               # Interface base vectorielle (Weaviate)
└── webapp/                 # Application web Streamlit
```

### Fonctionnalités avancées implémentées

#### Core RAG
- ✅ **Ingestion Confluence** automatisée avec extraction hiérarchique
- ✅ **Chunking intelligent** préservant la structure documentaire
- ✅ **Recherche vectorielle** Weaviate avec optimisations HNSW
- ✅ **Reformulation de requêtes** par LLM pour améliorer la pertinence
- ✅ **Pipeline sémantique** avec reranking et filtrage adaptatif

#### Interfaces utilisateur
- ✅ **Interface web Streamlit** avec chat persistant et multi-sessions
- ✅ **CLI complète** (status, ingest, chat, query) avec options avancées
- ✅ **Dashboard d'évaluation** pour tests multi-catégories
- ✅ **Dashboard de performance** avec métriques temps réel

#### Fonctionnalités entreprise
- ✅ **Authentification Azure AD** avec validation domaine
- ✅ **Intégration Azure Key Vault** pour gestion sécurisée des secrets
- ✅ **Stockage Azure Blob** pour déploiements cloud scalables
- ✅ **Environnements configurables** (dev/staging/prod)
- ✅ **Système de feedback** utilisateur intégré

#### Monitoring & Évaluation
- ✅ **Système d'évaluation automatisé** (retrieval, generation, business value, robustness)
- ✅ **Métriques de performance** temps réel
- ✅ **Logging complet** avec historique conversations et feedback
- ✅ **Tests automatisés** avec coverage
- ✅ **CI/CD** avec GitHub Actions

## Parcours utilisateur actualisé

### Interface Web
1. **Connexion** via Azure AD avec validation domaine entreprise
2. **Sélection d'interface** : Chat, Historique, Dashboard Performance
3. **Conversation naturelle** avec suggestions contextuelles
4. **Réponses enrichies** avec sources, métadonnées et scores de confiance
5. **Feedback intégré** pour amélioration continue
6. **Historique persistant** avec recherche dans les conversations passées

### Interface CLI
1. **Vérification système** : `isschat-cli status --verbose`
2. **Ingestion de données** : `isschat-cli ingest --source confluence`
3. **Chat interactif** : `isschat-cli chat`
4. **Requêtes directes** : `isschat-cli query -q "votre question" --show-stats`

## Données et sources

### Sources de données
- **Pages Confluence** (documentation technique, processus, guides)
- **Métadonnées enrichies** (auteur, date, tags, hiérarchie)
- **Historique conversations** avec analyse sentimentale
- **Feedback utilisateur** structuré pour amélioration continue
- **Métriques performance** système et qualité réponses

### Gestion des données
- **Responsable des données :** Johan Jublanc
- **Stockage sécurisé :** Azure Blob Storage + chiffrement
- **Sauvegarde automatique** des bases vectorielles
- **Logs structurés** avec rotation automatique

## Métriques et KPI actualisés

### KPI de performance
- ✅ **Temps de réponse** < 3 secondes (objectif atteint)
- ✅ **Taux de satisfaction utilisateur** > 85% (feedback positif)
- ✅ **Précision des réponses** > 80% (évaluation automatisée)
- ✅ **Disponibilité système** > 99.5%

### Métriques d'usage
- **Requêtes quotidiennes** : ~150-200 (tendance croissante)
- **Utilisateurs actifs** : 45+ collaborateurs Isskar
- **Temps moyen recherche** : Réduit de 75% vs recherche manuelle
- **Taux d'adoption** : 89% des équipes techniques

### Métriques qualité
- **Score de pertinence** : 4.2/5 en moyenne
- **Couverture documentation** : 95% des pages Confluence indexées
- **Précision sources** : 92% de citations correctes

## Architecture de déploiement

### Environnements
- **Développement** : Local avec stockage fichier
- **Staging** : Azure Container Instances + Blob Storage
- **Production** : Azure Container Apps avec auto-scaling

### Configuration cloud Azure
```bash
# Stockage
USE_AZURE_STORAGE=true
AZURE_STORAGE_ACCOUNT=isskarisschat
AZURE_BLOB_CONTAINER_NAME=isschat-data

# Authentification
AZURE_CLIENT_ID=xxx-xxx-xxx
AZURE_TENANT_ID=xxx-xxx-xxx

# Secrets
KEY_VAULT_URL=https://isschat-kv.vault.azure.net/
```

## Sécurité et conformité

### Mesures de sécurité implémentées
- ✅ **Authentification Azure AD** obligatoire
- ✅ **Validation domaine** entreprise (@isskar.fr)
- ✅ **Chiffrement** des données en transit et au repos
- ✅ **Gestion sécurisée** des secrets via Azure Key Vault
- ✅ **Logs d'audit** complets des accès et actions
- ✅ **Isolation** des données par utilisateur/session

### Conformité RGPD
- **Minimisation des données** : seules les données nécessaires
- **Droit à l'effacement** : possibilité de supprimer l'historique utilisateur
- **Transparence** : sources citées et traçabilité complète
- **Consentement** : acceptance explicite lors de la première connexion

## Risques et mitigation

### Risques techniques identifiés
| Risque | Impact | Probabilité | Mitigation |
|--------|--------|-------------|------------|
| Hallucinations LLM | Moyen | Faible | Système de scoring + citations obligatoires |
| Surcharge API Confluence | Élevé | Moyen | Cache intelligent + rate limiting |
| Dérive qualité responses | Moyen | Moyen | Évaluation automatisée quotidienne |
| Panne Azure services | Élevé | Faible | Fallback local + monitoring 24/7 |

## Coûts opérationnels (estimation mensuelle)

### Infrastructure Azure
- **Compute** (Container Apps) : ~150€/mois
- **Stockage** (Blob Storage) : ~30€/mois  
- **Key Vault** : ~5€/mois
- **Monitoring** : ~20€/mois

### APIs externes
- **OpenRouter** (LLM) : ~100€/mois
- **Confluence API** : Inclus abonnement entreprise

**Total estimé** : ~305€/mois pour 50 utilisateurs actifs

## Feuille de route 2025

### Q3 2025 (En cours)
- ✅ Migration vers Weaviate terminée
- ✅ Système d'évaluation automatisé opérationnel
- 🔄 Optimisations performance (chunking adaptatif)

### Q4 2025 (Planifié)
- 📋 Dashboard administrateur complet
- 📋 API REST pour intégrations externes
- 📋 Support multi-sources (SharePoint, Teams)
- 📋 Recherche fédérée cross-plateformes

### Q1 2026 (Roadmap)
- 📋 Intelligence conversationelle avancée
- 📋 Recommandations proactives de contenu
- 📋 Intégration Slack/Teams native
- 📋 Mode hors-ligne pour situations critiques

## Modalités de suivi

### Organisation équipe
- **Daily standup** avec Vincent Fraillon
- **Sprint reviews** bi-mensuelles avec stakeholders
- **Rétrospectives** mensuelles équipe technique
- **Comité de pilotage** trimestriel avec direction

### Communication
- **Email automatique** résumé hebdomadaire d'usage
- **Dashboard temps réel** accessible management
- **Rapports d'incidents** automatisés
- **Slack #isschat** pour support utilisateurs

### Documentation
- ✅ **README technique** complet et maintenu
- ✅ **Documentation utilisateur** Confluence
- ✅ **Runbooks** opérationnels
- ✅ **Architecture Decision Records** (ADR)

## Conclusion

Isschat a évolué d'un POC vers une plateforme d'entreprise mature avec :
- **Architecture scalable** prête pour 200+ utilisateurs
- **Fonctionnalités avancées** dépassant les objectifs initiaux  
- **Adoption utilisateur** excellente (89% équipes techniques)
- **Performance** supérieure aux attentes (temps réponse, précision)
- **Sécurité** entreprise avec conformité RGPD

Le projet est maintenant en phase d'optimisation continue avec une roadmap claire pour 2025-2026 centrée sur l'expansion fonctionnelle et l'intelligence conversationnelle avancée.