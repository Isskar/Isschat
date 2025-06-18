#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Add the parent directory to the Python search path
sys.path.append(str(Path(__file__).parent.parent.parent))

import click
from src.rag_system.rag_pipeline import RAGPipelineFactory


class ChatCLI:
    def __init__(self):
        self.pipeline = None
        self.history = []

    def initialize(self, rebuild_db=False):
        """Initialize the RAG pipeline"""
        try:
            print("Initialisation d'Isschat...")

            if rebuild_db:
                # Use the new rebuild system with validation
                try:
                    self.pipeline = RAGPipelineFactory.create_default_pipeline(force_rebuild=False)

                    if hasattr(self.pipeline.vector_store, "rebuild_database"):
                        print("🔄 Utilisation du nouveau système de rebuild avec validation...")
                        success = self.pipeline.vector_store.rebuild_database()
                        if not success:
                            print("❌ Échec du rebuild de la base de données")
                            return False
                    else:
                        # Fallback to old method
                        print("🔄 Utilisation de l'ancien système de rebuild...")
                        self.pipeline = RAGPipelineFactory.create_default_pipeline(force_rebuild=True)

                except Exception as e:
                    from src.core.exceptions import StorageAccessError, RebuildError

                    if isinstance(e, StorageAccessError):
                        print(f"🚫 ERREUR D'ACCÈS AU STOCKAGE:\n{str(e)}")
                        print("\n💡 CONSEILS:")
                        print("- Vérifiez votre configuration Azure (USE_AZURE_STORAGE, AZURE_STORAGE_ACCOUNT)")
                        print("- Vérifiez vos permissions Azure Storage")
                        return False
                    elif isinstance(e, RebuildError):
                        print(f"🚫 ERREUR DE REBUILD:\n{str(e)}")
                        return False
                    else:
                        raise
            else:
                self.pipeline = RAGPipelineFactory.create_default_pipeline(force_rebuild=False)

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
                elif not user_input:
                    continue

                # Process query
                print("\nRecherche en cours...")
                answer, sources = self.pipeline.process_query(user_input, verbose=False)

                # Display response
                print("\nIsschat:")
                print("-" * 40)
                print(answer)
                print("-" * 40)
                print(f"📚 Sources: {sources}")
                print()

                # Add to history
                self.history.append({"question": user_input, "answer": answer, "sources": sources})

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
        print("  - /help    : Afficher cette aide")
        print("  - /quit    : Quitter le programme")
        print("  - /clear   : Effacer l'écran")
        print("  - /history : Afficher l'historique")
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


@click.command()
@click.option("--rebuild-db", is_flag=True, help="Rebuild the vector database")
@click.option("--config-info", is_flag=True, help="Show configuration info")
def main(rebuild_db, config_info):
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
    cli = ChatCLI()
    if cli.initialize(rebuild_db=rebuild_db):
        cli.chat_loop()
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
