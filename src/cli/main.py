#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Add the parent directory to the Python search path
sys.path.append(str(Path(__file__).parent.parent.parent))

import click
from src.rag_system.rag_pipeline import RAGPipelineFactory


class ChatCLI:
    def __init__(self, show_chunks=False, full_chunks=False):
        self.pipeline = None
        self.history = []
        self.show_chunks = show_chunks
        self.full_chunks = full_chunks

    def initialize(self, rebuild_db=False):
        """Initialize the RAG pipeline"""
        try:
            print("Initialisation d'Isschat...")
            self.pipeline = RAGPipelineFactory.create_default_pipeline(force_rebuild=rebuild_db)
            print("Isschat prêt !")
            return True
        except Exception as e:
            print(f"❌ Erreur d'initialisation: {e}")
            return False

    def chat_loop(self):
        """Main chat loop"""
        print("\n" + "=" * 50)
        print("ISSCHAT CLI")
        print("=" * 50)
        print("Tapez votre question ou '/help' pour l'aide")
        print("Tapez '/quit' pour quitter")
        if self.show_chunks:
            print("🔍 Mode chunks activé - les chunks seront affichés")
        print("=" * 50 + "\n")

        while True:
            try:
                # Get user input
                user_input = input("💬 Vous: ").strip()

                # Handle commands
                if user_input.lower() in ["/quit", "/exit", "quit", "exit"]:
                    print("Au revoir !")
                    break
                elif user_input.lower() in ["/help", "help"]:
                    self.show_help()
                    continue
                elif user_input.lower() in ["/clear", "clear"]:
                    os.system("clear" if os.name == "posix" else "cls")
                    continue
                elif user_input.lower() in ["/history", "history"]:
                    self.show_history()
                    continue
                elif user_input.lower() in ["/chunks", "chunks"]:
                    self.show_chunks = not self.show_chunks
                    status = "activé" if self.show_chunks else "désactivé"
                    print(f"🔍 Affichage des chunks {status}")
                    continue
                elif user_input.lower() in ["/full-chunks", "full-chunks"]:
                    self.full_chunks = not self.full_chunks
                    status = "activé" if self.full_chunks else "désactivé"
                    print(f"📄 Affichage complet des chunks {status}")
                    continue
                elif user_input.lower().startswith("/chunks "):
                    # Show chunks for a specific question
                    question = user_input[8:].strip()
                    if question:
                        self.show_chunks_for_question(question)
                    continue
                elif user_input.lower() in ["/chunk-stats", "chunk-stats"]:
                    self.show_chunk_stats()
                    continue
                elif not user_input:
                    continue

                # Process query
                print("\nRecherche en cours...")
                if self.show_chunks:
                    answer, sources, chunks = self.pipeline.process_query(user_input, verbose=False, return_chunks=True)
                else:
                    answer, sources = self.pipeline.process_query(user_input, verbose=False)
                    chunks = None

                # Display response
                print("\nIsschat:")
                print("-" * 40)
                print(answer)
                print("-" * 40)
                print(f"📚 Sources: {sources}")

                # Display chunks if enabled
                if self.show_chunks and chunks:
                    self.display_chunks(chunks)

                print()

                # Add to history
                history_item = {"question": user_input, "answer": answer, "sources": sources}
                if chunks:
                    history_item["chunks"] = chunks
                self.history.append(history_item)

            except KeyboardInterrupt:
                print("\n\n👋 Au revoir !")
                break
            except Exception as e:
                print(f"\n❌ Erreur: {e}")
                print("Veuillez réessayer.\n")

    def show_help(self):
        """Show help message"""
        print("\n📖 Aide Isschat CLI:")
        print("  - Tapez votre question en français")
        print("  - /help           : Afficher cette aide")
        print("  - /quit           : Quitter le programme")
        print("  - /clear          : Effacer l'écran")
        print("  - /history        : Afficher l'historique")
        print("  - /chunks         : Activer/désactiver l'affichage des chunks")
        print("  - /full-chunks    : Activer/désactiver l'affichage complet des chunks")
        print("  - /chunks <question> : Afficher les chunks pour une question spécifique")
        print("  - /chunk-stats    : Afficher les statistiques de la base vectorielle")
        print()

    def show_history(self):
        """Show chat history"""
        if not self.history:
            print("\nAucun historique disponible\n")
            return

        print(f"\nHistorique ({len(self.history)} questions):")
        print("-" * 50)
        for i, item in enumerate(self.history[-5:], 1):  # Show last 5
            print(f"{i}. Q: {item['question'][:50]}...")
            print(f"   R: {item['answer'][:100]}...")
            print()

    def display_chunks(self, chunks):
        """Display retrieved chunks"""
        print("\n🔍 Chunks récupérés:")

        # Filter duplicates automatically
        seen_content = set()
        unique_chunks = []
        duplicates_count = 0

        for chunk in chunks:
            content_hash = hash(chunk.page_content)
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_chunks.append(chunk)
            else:
                duplicates_count += 1

        if duplicates_count > 0:
            print(f"🔧 {duplicates_count} doublons automatiquement filtrés")

        print(f"📊 {len(unique_chunks)} chunks uniques affichés (sur {len(chunks)} récupérés)")
        print("-" * 50)

        for i, chunk in enumerate(unique_chunks, 1):
            print(f"\n📄 Chunk {i}:")

            # Extract actual content (skip metadata if present)
            content = chunk.page_content

            # If content starts with metadata format, try to extract the real content
            if content.startswith("[Title:") and "]\n\n" in content:
                # Split at the end of metadata section
                parts = content.split("]\n\n", 1)
                if len(parts) > 1:
                    actual_content = parts[1].strip()
                    if self.full_chunks:
                        print(f"   📝 Contenu complet:\n{actual_content}")
                    else:
                        print(f"   📝 Contenu: {actual_content[:300]}...")
                else:
                    if self.full_chunks:
                        print(f"   📝 Contenu complet:\n{content}")
                    else:
                        print(f"   📝 Contenu: {content[:300]}...")
            else:
                if self.full_chunks:
                    print(f"   📝 Contenu complet:\n{content}")
                else:
                    print(f"   📝 Contenu: {content[:300]}...")

            print(f"   📁 Source: {chunk.metadata.get('source', 'N/A')}")
            print(f"   📋 Titre: {chunk.metadata.get('title', 'N/A')}")
            print(f"   🔗 URL: {chunk.metadata.get('url', 'N/A')}")
            print(f"   📏 Taille: {len(chunk.page_content)} caractères")
            if hasattr(chunk, "score"):
                print(f"   ⭐ Score: {chunk.score:.4f}")
        print("-" * 50)

    def show_chunks_for_question(self, question):
        """Show chunks for a specific question"""
        print(f"\n🔍 Récupération des chunks pour: '{question}'")
        try:
            # Get chunks without generating answer
            docs = self.pipeline.retriever.invoke(question)

            print(f"\n📚 {len(docs)} chunks trouvés:")

            # Filter duplicates automatically
            seen_content = set()
            unique_docs = []
            duplicates_count = 0

            for doc in docs:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_docs.append(doc)
                else:
                    duplicates_count += 1

            if duplicates_count > 0:
                print(f"🔧 {duplicates_count} doublons automatiquement filtrés")

            print(f"📊 {len(unique_docs)} chunks uniques affichés (sur {len(docs)} récupérés)")
            print("-" * 50)

            for i, doc in enumerate(unique_docs, 1):
                print(f"\n📄 Chunk {i}:")

                # Extract actual content (skip metadata if present)
                content = doc.page_content

                # If content starts with metadata format, try to extract the real content
                if content.startswith("[Title:") and "]\n\n" in content:
                    # Split at the end of metadata section
                    parts = content.split("]\n\n", 1)
                    if len(parts) > 1:
                        actual_content = parts[1].strip()
                        if self.full_chunks:
                            print(f"   📝 Contenu complet:\n{actual_content}")
                        else:
                            print(f"   📝 Contenu: {actual_content[:300]}...")
                    else:
                        if self.full_chunks:
                            print(f"   📝 Contenu complet:\n{content}")
                        else:
                            print(f"   📝 Contenu: {content[:300]}...")
                else:
                    if self.full_chunks:
                        print(f"   📝 Contenu complet:\n{content}")
                    else:
                        print(f"   📝 Contenu: {content[:300]}...")

                print(f"   📁 Source: {doc.metadata.get('source', 'N/A')}")
                print(f"   📋 Titre: {doc.metadata.get('title', 'N/A')}")
                print(f"   🔗 URL: {doc.metadata.get('url', 'N/A')}")
                print(f"   📏 Taille: {len(doc.page_content)} caractères")
                if hasattr(doc, "score"):
                    print(f"   ⭐ Score: {doc.score:.4f}")
            print("-" * 50)

        except Exception as e:
            print(f"❌ Erreur lors de la récupération des chunks: {e}")

    def show_chunk_stats(self):
        """Show statistics about the vector database"""
        print("\n📊 Statistiques de la base vectorielle:")
        print("-" * 50)

        try:
            # Get pipeline stats
            stats = self.pipeline.get_pipeline_stats()

            print(f"🔧 Type de pipeline: {stats.get('pipeline_type', 'N/A')}")

            # Database info
            db_info = stats.get("database_info", {})
            if db_info:
                print(f"📚 Informations de la base:")
                print(f"   - Documents: {db_info.get('document_count', 'N/A')}")
                print(f"   - Chunks: {db_info.get('chunk_count', 'N/A')}")
                print(f"   - Taille moyenne des chunks: {db_info.get('avg_chunk_size', 'N/A')} caractères")

            # Embeddings info
            embeddings_info = stats.get("embeddings_info", {})
            if embeddings_info:
                print(f"🧠 Modèle d'embeddings:")
                print(f"   - Modèle: {embeddings_info.get('model_name', 'N/A')}")
                print(f"   - Dimensions: {embeddings_info.get('dimensions', 'N/A')}")
                print(f"   - Device: {embeddings_info.get('device', 'N/A')}")

            # Generator stats
            generator_stats = stats.get("generator_stats", {})
            if generator_stats:
                print(f"⚡ Générateur:")
                print(f"   - Modèle: {generator_stats.get('model_name', 'N/A')}")
                print(f"   - Requêtes totales: {generator_stats.get('total_requests', 'N/A')}")
                print(f"   - Tokens totaux: {generator_stats.get('total_tokens', 'N/A')}")

            # Config info
            config = stats.get("config", {})
            if config:
                print(f"⚙️  Configuration:")
                print(f"   - search_k: {config.get('search_k', 'N/A')}")
                print(f"   - search_fetch_k: {config.get('search_fetch_k', 'N/A')}")
                print(f"   - embeddings_model: {config.get('embeddings_model', 'N/A')}")

        except Exception as e:
            print(f"❌ Erreur lors de la récupération des statistiques: {e}")

        print("-" * 50)


@click.command()
@click.option("--rebuild-db", is_flag=True, help="Rebuild the vector database")
@click.option("--config-info", is_flag=True, help="Show configuration info")
@click.option("--show-chunks", is_flag=True, help="Show retrieved chunks for each query")
@click.option("--full-chunks", is_flag=True, help="Show full content of chunks (requires --show-chunks)")
def main(rebuild_db, config_info, show_chunks, full_chunks):
    if config_info:
        try:
            from src.core.config import get_debug_info

            info = get_debug_info()
            print("\nConfiguration Isschat:")
            print("-" * 30)
            for key, value in info.items():
                print(f"{key}: {value}")
            print()
        except Exception as e:
            print(f"❌ Erreur lors de la récupération de la config: {e}")
        return

    # Initialize and start chat
    # If full_chunks is enabled, automatically enable show_chunks
    if full_chunks and not show_chunks:
        show_chunks = True
        print("ℹ️  --full-chunks active, activation automatique de --show-chunks")

    cli = ChatCLI(show_chunks=show_chunks, full_chunks=full_chunks)
    if cli.initialize(rebuild_db=rebuild_db):
        cli.chat_loop()
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
