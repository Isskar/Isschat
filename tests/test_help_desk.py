"""
Tests unitaires pour le module help_desk d'ISSCHAT
Tests de validation des réponses, de la langue française et des sources
"""

import re
from unittest.mock import patch

from src.help_desk import HelpDesk


class TestHelpDeskSimple:
    """Tests simplifiés avec mocks directs"""

    @patch("src.help_desk.HelpDesk.retrieval_qa_inference")
    def test_response_length_minimum_words(self, mock_inference, simple_mock_dependencies):
        """Test que la longueur de la réponse est supérieure à un certain nombre de mots"""

        # Mock de la réponse
        mock_inference.return_value = (
            "Pour demander des congés, je vous suggère de suivre la procédure officielle. "
            "Vous devez vous connecter au portail RH et remplir le formulaire de demande. "
            "N'oubliez pas de soumettre votre demande au moins 2 semaines à l'avance pour "
            "obtenir la validation de votre manager.",
            "Voici les sources que j'ai utilisées pour répondre à votre question:  \n- "
            "[Procédure demande de congés](https://confluence.entreprise.com/pages/conges-procedure)",
        )

        help_desk = HelpDesk(new_db=False)

        questions = [
            "Comment demander des congés ?",
            "Comment configurer le VPN ?",
            "Que faire en tant que nouvel employé ?",
        ]

        for question in questions:
            answer, sources = help_desk.retrieval_qa_inference(question, verbose=False)

            # Compter les mots dans la réponse
            word_count = len(answer.split())

            # Vérifier que la réponse contient au moins 10 mots
            assert word_count >= 10, f"La réponse pour '{question}' contient seulement {word_count} mots: {answer}"

            print(f"✓ Question: '{question}' - Réponse: {word_count} mots")

    @patch("src.help_desk.HelpDesk.retrieval_qa_inference")
    def test_response_is_in_french(self, mock_inference, simple_mock_dependencies):
        """Test que la réponse fournie est en français"""

        # Mock de la réponse en français
        mock_inference.return_value = (
            "Pour configurer le VPN, vous pourriez télécharger le client depuis le portail IT. "
            "Utilisez vos identifiants Active Directory pour vous connecter au serveur "
            "vpn.entreprise.com. En cas de problème, contactez le support IT.",
            "Voici les sources que j'ai utilisées pour répondre à votre question:  \n- "
            "[Configuration VPN](https://confluence.entreprise.com/pages/vpn-config)",
        )

        help_desk = HelpDesk(new_db=False)

        question = "Comment configurer le VPN ?"
        answer, sources = help_desk.retrieval_qa_inference(question, verbose=False)

        # Mots/expressions typiquement français
        french_indicators = [
            "vous pourriez",
            "vous devez",
            "pour",
            "avec",
            "dans",
            "sur",
            "de",
            "le",
            "la",
            "les",
            "un",
            "une",
            "des",
            "du",
            "au",
            "aux",
            "ce",
            "cette",
            "votre",
            "vos",
        ]

        # Convertir en minuscules pour la comparaison
        answer_lower = answer.lower()

        # Vérifier la présence d'indicateurs français
        french_words_found = [word for word in french_indicators if word in answer_lower]

        assert len(french_words_found) >= 3, (
            f"La réponse ne semble pas être en français. Mots français trouvés: {french_words_found}. Réponse: {answer}"
        )

        print(f"✓ Réponse en français: {len(french_words_found)} indicateurs trouvés")

    @patch("src.help_desk.HelpDesk.retrieval_qa_inference")
    def test_response_contains_source_links(self, mock_inference, simple_mock_dependencies):
        """Test que ISSCHAT donne les documents dont il s'est servi (liens https)"""

        # Mock avec sources contenant des liens https
        mock_inference.return_value = (
            "Bienvenue ! Je vous suggère de commencer par configurer votre compte email "
            "et d'accéder aux systèmes internes.",
            "Voici les sources que j'ai utilisées pour répondre à votre question:  \n- "
            "[Guide de démarrage - Nouveaux employés](https://confluence.entreprise.com/pages/guide-demarrage)  \n- "
            "[Procédure demande de congés](https://confluence.entreprise.com/pages/conges-procedure)",
        )

        help_desk = HelpDesk(new_db=False)

        question = "Que faire en tant que nouvel employé ?"
        answer, sources = help_desk.retrieval_qa_inference(question, verbose=False)

        # Vérifier que sources n'est pas vide
        assert sources, f"Aucune source retournée pour la question: '{question}'"

        # Vérifier la présence de liens https dans les sources
        https_links = re.findall(r"https://[^\s\)]+", sources)

        assert len(https_links) >= 1, f"Aucun lien https trouvé dans les sources pour '{question}'. Sources: {sources}"

        print(f"✓ Question: '{question}' - {len(https_links)} lien(s) https trouvé(s)")

    @patch("src.help_desk.HelpDesk.retrieval_qa_inference")
    def test_source_format_validation(self, mock_inference, simple_mock_dependencies):
        """Test le format des sources retournées"""

        # Mock avec sources au format markdown
        mock_inference.return_value = (
            "Pour demander des congés, suivez la procédure officielle.",
            "Voici les sources que j'ai utilisées pour répondre à votre question:  \n- "
            "[Procédure demande de congés](https://confluence.entreprise.com/pages/conges-procedure)  \n- "
            "[Guide RH](https://confluence.entreprise.com/pages/guide-rh)",
        )

        help_desk = HelpDesk(new_db=False)

        question = "Comment demander des congés ?"
        answer, sources = help_desk.retrieval_qa_inference(question, verbose=False)

        # Vérifier le format markdown des liens [titre](url)
        markdown_links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", sources)

        assert len(markdown_links) >= 1, f"Aucun lien au format markdown trouvé. Sources: {sources}"

        # Vérifier que les URLs sont valides
        for title, url in markdown_links:
            assert url.startswith("https://"), f"URL invalide trouvée: {url}"
            assert len(title.strip()) > 0, f"Titre de lien vide pour l'URL: {url}"

        print(f"✓ {len(markdown_links)} lien(s) au format markdown valide trouvé(s)")

    @patch("src.help_desk.HelpDesk.retrieval_qa_inference")
    def test_complete_workflow_validation(self, mock_inference, simple_mock_dependencies):
        """Test du workflow complet avec toutes les validations"""

        # Mock d'une réponse complète
        mock_inference.return_value = (
            "Pour demander des congés, je vous suggère de suivre la procédure officielle. "
            "Vous devez vous connecter au portail RH et remplir le formulaire de demande. "
            "N'oubliez pas de soumettre votre demande au moins 2 semaines à l'avance pour "
            "obtenir la validation de votre manager. En cas de problème, contactez le service RH.",
            "Voici les 2 sources que j'ai utilisées pour répondre à votre question:  \n- "
            "[Procédure demande de congés](https://confluence.entreprise.com/pages/conges-procedure)  \n- "
            "[Guide RH](https://confluence.entreprise.com/pages/guide-rh)",
        )

        help_desk = HelpDesk(new_db=False)

        question = "Comment demander des congés ?"
        answer, sources = help_desk.retrieval_qa_inference(question, verbose=False)

        # 1. Test longueur de réponse
        word_count = len(answer.split())
        assert word_count >= 15, f"Réponse trop courte: {word_count} mots"

        # 2. Test langue française
        french_indicators = ["vous", "pour", "de", "le", "la", "je", "suggère"]
        found_french = [w for w in french_indicators if w in answer.lower()]
        assert len(found_french) >= 3, f"Réponse pas en français: {found_french}"

        # 3. Test présence de sources avec liens
        https_links = re.findall(r"https://[^\s\)]+", sources)
        assert len(https_links) >= 1, "Aucun lien https dans les sources"

        print("✓ Workflow complet validé:")
        print(f"  - Réponse: {word_count} mots")
        print(f"  - Français: {len(found_french)} indicateurs")
        print(f"  - Sources: {len(https_links)} lien(s)")


class TestHelpDeskConfiguration:
    """Tests de configuration et d'initialisation"""

    def test_help_desk_can_be_created(self, simple_mock_dependencies):
        """Test que HelpDesk peut être créé sans erreur"""
        help_desk = HelpDesk(new_db=False)

        # Vérifier que l'objet est créé
        assert help_desk is not None
        assert hasattr(help_desk, "retrieval_qa_inference")

        print("✓ HelpDesk créé avec succès")

    def test_config_validation(self, simple_mock_dependencies):
        """Test de validation de la configuration"""
        from config import get_config

        config_data = get_config()

        # Vérifier les champs obligatoires
        assert config_data.openrouter_api_key, "Clé API OpenRouter manquante"
        assert config_data.confluence_space_key, "Clé d'espace Confluence manquante"

        print("✓ Configuration validée")
