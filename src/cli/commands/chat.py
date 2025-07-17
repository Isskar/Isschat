"""
Chat command to test the RAG system without UI.
"""

import click
import os
from datetime import datetime

from ...rag.pipeline import RAGPipelineFactory
from ...rag.semantic_pipeline import SemanticRAGPipelineFactory
from ...storage.data_manager import get_data_manager


class ChatSession:
    """CLI chat session"""

    def __init__(self):
        self.pipeline = None
        self.data_manager = None
        self.user_id = "cli_user"
        self.conversation_id = f"cli_{int(datetime.now().timestamp())}"
        self.history = []

    def initialize(self) -> bool:
        """Initialize the RAG pipeline"""
        try:
            click.echo("🔧 Initializing RAG pipeline...")
            self.pipeline = SemanticRAGPipelineFactory.create_semantic_pipeline(use_semantic_features=True)
            self.data_manager = get_data_manager()

            if not self.pipeline.is_ready():
                click.echo("⚠️ Vector database empty or not accessible")
                click.echo("💡 Run first: isschat-cli ingest --source confluence")
                return False

            click.echo("✅ RAG pipeline ready!")
            return True

        except Exception as e:
            click.echo(f"❌ Initialization error: {e}")
            return False

    def run_chat_loop(self):
        """Main chat loop"""
        click.echo("\n" + "=" * 60)
        click.echo("🤖 ISSCHAT CLI - Mode Chat")
        click.echo("=" * 60)
        click.echo("💬 Ask your questions or type a command:")
        click.echo("   /help    - Show help")
        click.echo("   /status  - System status")
        click.echo("   /history - Question history")
        click.echo("   /clear   - Clear screen")
        click.echo("   /quit    - Quit")
        click.echo("=" * 60 + "\n")

        while True:
            try:
                user_input = input("💬 You: ").strip()

                if user_input.lower() in ["/quit", "/exit", "quit", "exit"]:
                    click.echo("\n👋 Goodbye!")
                    break
                elif user_input.lower() in ["/help", "help"]:
                    self._show_help()
                    continue
                elif user_input.lower() in ["/status", "status"]:
                    self._show_status()
                    continue
                elif user_input.lower() in ["/history", "history"]:
                    self._show_history()
                    continue
                elif user_input.lower() in ["/clear", "clear"]:
                    os.system("clear" if os.name == "posix" else "cls")
                    continue
                elif not user_input:
                    continue

                click.echo("\n🔍 Searching...")

                history_context = self._build_history_context()

                answer, sources = self.pipeline.process_query(
                    query=user_input,
                    history=history_context,
                    user_id=self.user_id,
                    conversation_id=self.conversation_id,
                    verbose=False,
                )

                click.echo("\n🤖 Isschat:")
                click.echo("-" * 50)
                click.echo(answer)
                click.echo("-" * 50)
                click.echo(f"📚 Sources: {sources}")
                click.echo()

                self.history.append(
                    {
                        "question": user_input,
                        "answer": answer,
                        "sources": sources,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

                if len(self.history) % 1 == 0:
                    self._ask_for_feedback()

            except KeyboardInterrupt:
                click.echo("\n\n👋 Goodbye!")
                break
            except Exception as e:
                click.echo(f"\n❌ Error: {e}")
                click.echo("🔄 Please try again.\n")

    def _show_help(self):
        """Show help"""
        click.echo("\n📖 Isschat CLI Help:")
        click.echo("   • Ask your questions in French")
        click.echo("   • The system searches in Confluence documentation")
        click.echo("   • /status   : View system status")
        click.echo("   • /history  : View your recent questions")
        click.echo("   • /clear    : Clear the screen")
        click.echo("   • /quit     : Exit the chat")
        click.echo()

    def _show_status(self):
        """Show system status"""
        click.echo("\n📊 System status:")
        try:
            status = self.pipeline.get_status()

            click.echo(f"   🔧 Pipeline: {'✅ Ready' if status['ready'] else '❌ Not ready'}")

            if "config" in status:
                config = status["config"]
                click.echo(f"   🤖 LLM Model: {config['llm_model']}")
                click.echo(f"   🔢 Embedding model: {config['embeddings_model']}")
                click.echo(f"   📚 Collection: {config['vectordb_collection']}")

            if "retrieval_tool" in status and "vector_db" in status["retrieval_tool"]:
                db_info = status["retrieval_tool"]["vector_db"]
                if "points_count" in db_info:
                    click.echo(f"   💾 Documents in database: {db_info['points_count']}")

        except Exception as e:
            click.echo(f"   ❌ Status retrieval error: {e}")

        click.echo()

    def _show_history(self):
        """Show history"""
        if not self.history:
            click.echo("\n📝 No history available\n")
            return

        click.echo(f"\n📝 History ({len(self.history)} questions):")
        click.echo("-" * 60)

        for i, item in enumerate(self.history[-5:], 1):
            click.echo(f"{i}. Q: {item['question'][:80]}...")
            click.echo(f"   R: {item['answer'][:120]}...")
            click.echo()

    def _build_history_context(self) -> str:
        """Build history context for the query"""
        if len(self.history) <= 1:
            return ""

        context_parts = []
        for item in self.history[-2:]:
            context_parts.append(f"Q: {item['question']}")
            context_parts.append(f"R: {item['answer']}")

        return "\n".join(context_parts)

    def _ask_for_feedback(self):
        """Ask for user feedback"""
        try:
            if not self.history:
                return

            click.echo("📝 Feedback (optional):")
            rating_input = input("Good or Bad: ").strip()

            if rating_input and rating_input.lower() in ["good", "bad"]:
                rating = rating_input.lower()
                comment = input("   Comment (optional): ").strip()

                success = self.data_manager.save_feedback(
                    user_id=self.user_id, conversation_id=self.conversation_id, rating=rating, comment=comment
                )

                if success:
                    click.echo("   ✅ Feedback saved, thank you!")
                else:
                    click.echo("   ⚠️ Feedback save error")

            click.echo()

        except Exception:
            pass


@click.command()
@click.option("--user-id", default="cli_user", help="User ID for logs")
def chat(user_id: str):
    """Start an interactive chat session"""

    session = ChatSession()
    session.user_id = user_id

    if not session.initialize():
        click.echo("💡 Suggestions:")
        click.echo("   1. Check your configuration (.env)")
        click.echo("   2. Run ingestion: isschat-cli ingest --source confluence")
        click.echo("   3. Check status: isschat-cli status")
        return

    session.run_chat_loop()


if __name__ == "__main__":
    chat()
