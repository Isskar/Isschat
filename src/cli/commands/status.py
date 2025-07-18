"""
Status command to check the system state.
"""

import click
from typing import Dict, Any

from src.config import get_config
from src.ingestion import create_confluence_pipeline
from src.rag.semantic_pipeline import SemanticRAGPipelineFactory


@click.command()
@click.option("--verbose", "-v", is_flag=True, help="Detailed output")
@click.option(
    "--component", type=click.Choice(["config", "ingestion", "rag", "all"]), default="all", help="Component to check"
)
def status(verbose: bool, component: str):
    """Check Isschat components status"""

    click.echo("📊 Isschat Status")
    click.echo("=" * 50)

    overall_status = True

    # Configuration
    if component in ["config", "all"]:
        click.echo("\n🔧 Configuration:")
        config_status = _check_config_status(verbose)
        overall_status &= config_status["success"]
        _display_status_section(config_status, verbose)

    # Pipeline d'ingestion
    if component in ["ingestion", "all"]:
        click.echo("\n📥 Ingestion pipeline:")
        ingestion_status = _check_ingestion_status(verbose)
        overall_status &= ingestion_status["success"]
        _display_status_section(ingestion_status, verbose)

    # Pipeline RAG
    if component in ["rag", "all"]:
        click.echo("\n🤖 Pipeline RAG:")
        rag_status = _check_rag_status(verbose)
        overall_status &= rag_status["success"]
        _display_status_section(rag_status, verbose)

    # Global status
    click.echo("\n" + "=" * 50)
    status_icon = "✅" if overall_status else "❌"
    status_text = "READY" if overall_status else "NOT READY"
    click.echo(f"{status_icon} Global status: {status_text}")

    if not overall_status:
        click.echo("\n💡 Suggestions:")
        click.echo("   1. Check your .env file")
        click.echo("   2. Run: isschat-cli ingest --source confluence")
        click.echo("   3. Use --verbose for more details")


def _check_config_status(verbose: bool) -> Dict[str, Any]:
    """Check configuration status"""
    try:
        config = get_config()

        # Basic checks
        checks = {
            "Config loaded": True,
            "Confluence API key": bool(config.confluence_api_key),
            "Confluence space key": bool(config.confluence_space_key),
            "OpenRouter API key": bool(config.openrouter_api_key),
            "Embeddings model": bool(config.embeddings_model),
            "LLM model": bool(config.llm_model),
        }

        success = all(checks.values())

        details = {}
        if verbose:
            details = {
                "embeddings_model": config.embeddings_model,
                "llm_model": config.llm_model,
                "vectordb_collection": config.vectordb_collection,
                "chunk_size": config.chunk_size,
                "search_k": config.search_k,
            }

        return {"success": success, "checks": checks, "details": details}

    except Exception as e:
        return {"success": False, "error": str(e), "checks": {"Config loaded": False}}


def _check_ingestion_status(verbose: bool) -> Dict[str, Any]:
    """Check ingestion pipeline status"""
    try:
        pipeline = create_confluence_pipeline()

        # Component tests
        connection_success = pipeline.check_connection()

        # General status
        status_info = pipeline.get_status()

        details = {}
        if verbose:
            details = {
                "pipeline_ready": status_info.get("pipeline_ready", False),
                "vector_db_exists": status_info.get("vector_db", {}).get("exists", False),
                "vector_db_count": status_info.get("vector_db", {}).get("count", 0),
            }

        return {"success": connection_success, "details": details}

    except Exception as e:
        return {"success": False, "error": str(e), "checks": {"Ingestion pipeline": False}}


def _check_rag_status(verbose: bool) -> Dict[str, Any]:
    """Check RAG pipeline status"""
    try:
        pipeline = SemanticRAGPipelineFactory.create_semantic_pipeline()

        # Test du pipeline
        test_results = pipeline.check_pipeline()

        # Detailed status
        status_info = pipeline.get_status()

        checks = {
            "Pipeline ready": test_results.get("success", False),
            "Retrieval tool": status_info.get("retrieval_tool", {}).get("ready", False),
            "Generation tool": status_info.get("generation_tool", {}).get("ready", False),
        }

        success = test_results.get("success", False)

        details = {}
        if verbose:
            details = {
                "pipeline_ready": status_info.get("ready", False),
                "test_query": test_results.get("test_query", ""),
                "response_time_ms": test_results.get("response_time_ms", 0),
                "vector_db_count": status_info.get("retrieval_tool", {}).get("vector_db", {}).get("points_count", 0),
                "error": test_results.get("error"),
            }

        return {"success": success, "checks": checks, "details": details}

    except Exception as e:
        return {"success": False, "error": str(e), "checks": {"Pipeline RAG": False}}


def _display_status_section(status_result: Dict[str, Any], verbose: bool):
    """Display a status section"""
    # Display checks
    for check_name, check_result in status_result.get("checks", {}).items():
        icon = "✅" if check_result else "❌"
        click.echo(f"   {icon} {check_name}")

    # Display error if present
    if "error" in status_result:
        click.echo(f"   ⚠️ Error: {status_result['error']}")

    # Display details if verbose
    if verbose and "details" in status_result:
        details = status_result["details"]
        if details:
            click.echo("   📋 Details:")
            for key, value in details.items():
                if isinstance(value, list) and key == "errors":
                    if value:
                        click.echo(f"      {key}: {len(value)} error(s)")
                        for error in value[:3]:  # Max 3 errors
                            click.echo(f"        • {error}")
                        if len(value) > 3:
                            click.echo(f"        ... and {len(value) - 3} others")
                    else:
                        click.echo(f"      {key}: none")
                else:
                    click.echo(f"      {key}: {value}")


if __name__ == "__main__":
    status()
