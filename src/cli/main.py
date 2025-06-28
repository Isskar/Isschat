#!/usr/bin/env python3
"""
Main Isschat CLI with structured commands.
Modular architecture with separate commands.
"""

import sys
from pathlib import Path

# Add the parent directory to the Python search path
sys.path.append(str(Path(__file__).parent.parent.parent))

import click

from .commands.ingest import ingest
from .commands.chat import chat
from .commands.status import status


@click.group()
@click.version_option("0.2.0", prog_name="isschat-cli")
def cli():
    """
    🤖 Isschat CLI - AI Assistant for enterprise documentation
    """
    pass


# Add commands
cli.add_command(ingest)
cli.add_command(chat)
cli.add_command(status)


@cli.command()
@click.option(
    "--component",
    type=click.Choice(["config", "embeddings", "vectordb", "rag", "all"]),
    default="all",
    help="Component to test",
)
def test(component: str):
    """Test system components"""

    click.echo(f"🧪 Testing components: {component}")
    click.echo("=" * 50)

    overall_success = True

    try:
        # Test configuration
        if component in ["config", "all"]:
            click.echo("\n🔧 Configuration test:")
            try:
                from ..config import get_config

                config = get_config()
                click.echo("   ✅ Configuration loaded")

                if config.confluence_api_key:
                    click.echo("   ✅ Confluence API key configured")
                else:
                    click.echo("   ❌ Confluence API key missing")
                    overall_success = False

                if config.openrouter_api_key:
                    click.echo("   ✅ OpenRouter API key configured")
                else:
                    click.echo("   ❌ OpenRouter API key missing")
                    overall_success = False

            except Exception as e:
                click.echo(f"   ❌ Config error: {e}")
                overall_success = False

        # Test embeddings
        if component in ["embeddings", "all"]:
            click.echo("\n🔢 Embedding service test:")
            try:
                from ..embeddings import get_embedding_service

                service = get_embedding_service()

                # Test encoding
                test_embedding = service.encode_single("test")
                click.echo(f"   ✅ Service ready (dim: {len(test_embedding)})")

            except Exception as e:
                click.echo(f"   ❌ Embeddings error: {e}")
                overall_success = False

        # Test vector DB
        if component in ["vectordb", "all"]:
            click.echo("\n💾 Vector database test:")
            try:
                from ..vectordb import VectorDBFactory

                vector_db = VectorDBFactory.create_from_config()

                if vector_db.exists():
                    count = vector_db.count()
                    click.echo(f"   ✅ Database ready ({count} documents)")
                else:
                    click.echo("   ⚠️ Empty database - run ingestion")

            except Exception as e:
                click.echo(f"   ❌ Vector DB error: {e}")
                overall_success = False

        # Test RAG
        if component in ["rag", "all"]:
            click.echo("\n🤖 RAG pipeline test:")
            try:
                from ..rag.pipeline import RAGPipelineFactory

                pipeline = RAGPipelineFactory.create_default_pipeline()

                if pipeline.is_ready():
                    result = pipeline.check_pipeline()
                    if result["success"]:
                        click.echo(f"   ✅ Pipeline ready ({result['response_time_ms']:.0f}ms)")
                    else:
                        click.echo(f"   ❌ Test failed: {result.get('error', 'Unknown error')}")
                        overall_success = False
                else:
                    click.echo("   ❌ Pipeline not ready")
                    overall_success = False

            except Exception as e:
                click.echo(f"   ❌ RAG error: {e}")
                overall_success = False

        # Final result
        click.echo("\n" + "=" * 50)
        if overall_success:
            click.echo("✅ All tests passed successfully!")
            click.echo("🚀 System is ready to use")
        else:
            click.echo("❌ Some tests failed")
            click.echo("💡 Use 'isschat-cli status --verbose' for more details")

    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}")


def main():
    """Main entry point"""
    cli()


if __name__ == "__main__":
    main()
