"""
Exemple d'utilisation de la gestion d'historique de conversation
avec LlamaIndex RAG Pipeline.
"""

from .semantic_pipeline import SemanticRAGPipelineFactory


def example_conversation_management():
    """Exemple complet de gestion d'historique de conversation"""

    # Créer le pipeline
    pipeline = SemanticRAGPipelineFactory.create_semantic_pipeline()

    print("=== Exemple de Gestion d'Historique de Conversation ===\n")

    # === SCÉNARIO 1: Nouvelle conversation ===
    print("1. 🆕 Démarrage d'une nouvelle conversation")
    conv_id = "conv_project_discussion_001"
    pipeline.start_new_conversation(conv_id)

    # Première question
    answer1, sources1 = pipeline.process_query(
        query="Qui travaille sur le projet Isschat?", user_id="user123", conversation_id=conv_id, verbose=True
    )
    print("Q1: Qui travaille sur le projet Isschat?")
    print(f"R1: {answer1[:100]}...")
    print(f"Memory: {pipeline.get_memory_summary()}\n")

    # Question de suivi (avec contexte)
    answer2, sources2 = pipeline.process_query(
        query="Quelles sont leurs responsabilités?", user_id="user123", conversation_id=conv_id, verbose=True
    )
    print("Q2: Quelles sont leurs responsabilités?")
    print(f"R2: {answer2[:100]}...")
    print(f"Memory: {pipeline.get_memory_summary()}\n")

    # === SCÉNARIO 2: Interruption puis reprise ===
    print("2. 💾 Interruption de session (simulation redémarrage)")

    # Créer un nouveau pipeline (simulation redémarrage)
    pipeline_new_session = SemanticRAGPipelineFactory.create_semantic_pipeline()

    print("3. 🔄 Reprise de conversation existante")
    # Reprendre la conversation précédente
    success = pipeline_new_session.continue_conversation(conv_id, user_id="user123")
    print(f"Historique chargé: {success}")
    print(f"Memory après reprise: {pipeline_new_session.get_memory_summary()}\n")

    # Question utilisant le contexte chargé
    answer3, sources3 = pipeline_new_session.process_query(
        query="Et Vincent, que fait-il exactement?", user_id="user123", conversation_id=conv_id, verbose=True
    )
    print("Q3: Et Vincent, que fait-il exactement?")
    print(f"R3: {answer3[:100]}...")
    print(f"Memory: {pipeline_new_session.get_memory_summary()}\n")

    # === SCÉNARIO 3: Changement de conversation ===
    print("4. 🔀 Changement vers une autre conversation")
    conv_id_2 = "conv_technical_support_002"

    answer4, sources4 = pipeline_new_session.process_query(
        query="Comment installer Isschat?",
        user_id="user123",
        conversation_id=conv_id_2,  # Nouvelle conversation
        verbose=True,
    )
    print("Q4 (nouvelle conv): Comment installer Isschat?")
    print(f"R4: {answer4[:100]}...")
    print(f"Memory: {pipeline_new_session.get_memory_summary()}\n")

    # === SCÉNARIO 4: Retour à la première conversation ===
    print("5. ↩️ Retour à la première conversation")
    answer5, sources5 = pipeline_new_session.process_query(
        query="Peux-tu me rappeler l'équipe du projet?",
        user_id="user123",
        conversation_id=conv_id,  # Retour à la première conv
        verbose=True,
    )
    print("Q5 (retour conv 1): Peux-tu me rappeler l'équipe du projet?")
    print(f"R5: {answer5[:100]}...")
    print(f"Memory: {pipeline_new_session.get_memory_summary()}\n")

    # === STATISTIQUES FINALES ===
    print("6. 📊 Statistiques du pipeline")
    stats = pipeline_new_session.get_stats()
    print(f"Pipeline Type: {stats['type']}")
    print(f"Memory Management: {stats['conversation_management']}")
    print(f"Adaptive Memory: {stats['adaptive_memory']}")


if __name__ == "__main__":
    example_conversation_management()
