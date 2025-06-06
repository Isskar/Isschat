"""
Données de test pour les tests unitaires d'ISSCHAT
Contient des faux documents pour simuler la base de données Confluence
"""

from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class MockDocument:
    """Document simulé pour les tests"""

    page_content: str
    metadata: Dict[str, Any]


# Faux documents en français pour les tests
MOCK_DOCUMENTS = [
    MockDocument(
        page_content="""
        Guide de démarrage rapide pour les nouveaux employés

        Bienvenue chez notre entreprise ! Ce guide vous aidera à vous familiariser avec nos processus.

        Étapes importantes :
        1. Configurer votre compte email
        2. Accéder aux systèmes internes
        3. Rencontrer votre équipe
        4. Suivre la formation obligatoire

        Pour toute question, contactez le service RH à l'adresse rh@entreprise.com
        """,
        metadata={
            "title": "Guide de démarrage - Nouveaux employés",
            "source": "https://confluence.entreprise.com/pages/guide-demarrage",
            "space": "RH",
            "author": "Service RH",
        },
    ),
    MockDocument(
        page_content="""
        Procédure de demande de congés

        Pour demander des congés, suivez ces étapes :

        1. Connectez-vous au portail RH
        2. Remplissez le formulaire de demande de congés
        3. Sélectionnez les dates souhaitées
        4. Soumettez la demande à votre manager
        5. Attendez la validation

        Les congés doivent être demandés au moins 2 semaines à l'avance.
        En cas d'urgence, contactez directement votre manager.
        """,
        metadata={
            "title": "Procédure demande de congés",
            "source": "https://confluence.entreprise.com/pages/conges-procedure",
            "space": "RH",
            "author": "Marie Dupont",
        },
    ),
    MockDocument(
        page_content="""
        Configuration du VPN d'entreprise

        Le VPN permet d'accéder aux ressources internes depuis l'extérieur.

        Installation :
        1. Téléchargez le client VPN depuis le portail IT
        2. Installez l'application
        3. Utilisez vos identifiants Active Directory
        4. Connectez-vous au serveur vpn.entreprise.com

        En cas de problème, contactez le support IT au 01.23.45.67.89
        """,
        metadata={
            "title": "Configuration VPN",
            "source": "https://confluence.entreprise.com/pages/vpn-config",
            "space": "IT",
            "author": "Équipe IT",
        },
    ),
]

# Configuration de test
TEST_CONFIG = {
    "confluence_private_api_key": "fake_api_key_for_tests",
    "confluence_space_key": "TEST",
    "confluence_space_name": "Test Space",
    "confluence_email_address": "test@example.com",
    "openrouter_api_key": "fake_openrouter_key",
    "db_path": "test_db.db",
    "persist_directory": "test_db",
}

# Réponses attendues pour les tests
EXPECTED_RESPONSES = {
    "conges": {
        "keywords": ["congés", "demande", "manager", "validation", "semaines"],
        "min_words": 20,
        "should_contain_source": True,
        "language": "french",
    },
    "vpn": {
        "keywords": ["VPN", "client", "identifiants", "support"],
        "min_words": 15,
        "should_contain_source": True,
        "language": "french",
    },
    "demarrage": {
        "keywords": ["démarrage", "employés", "email", "formation"],
        "min_words": 25,
        "should_contain_source": True,
        "language": "french",
    },
}
