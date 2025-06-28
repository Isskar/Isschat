"""
Ingestion command to build the vector database.
"""

import click
import logging

from ...ingestion.pipeline import create_ingestion_pipeline


@click.command()
@click.option("--source", default="confluence", type=click.Choice(["confluence"]), help="Data source to ingest")
@click.option("--force-rebuild", is_flag=True, help="Completely rebuild the vector database")
@click.option("--verbose", "-v", is_flag=True, help="Detailed output")
def ingest(source: str, force_rebuild: bool, verbose: bool):
    """Ingest data into the vector database"""

    # Logging configuration
    log_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    click.echo("🚀 Starting Isschat ingestion")
    click.echo(f"   📥 Source: {source}")
    click.echo(f"   🔄 Force rebuild: {force_rebuild}")
    click.echo()

    try:
        # Create ingestion pipeline
        pipeline = create_ingestion_pipeline()

        # Component tests
        if verbose:
            click.echo("Checking components...")
            test_results = pipeline.check_pipeline()

            if test_results["overall_success"]:
                click.echo("✅ All components are ready")
            else:
                click.echo("⚠️ Problems detected:")
                for error in test_results["errors"]:
                    click.echo(f"   • {error}")

                if not click.confirm("Continue despite problems?"):
                    click.echo("❌ Ingestion cancelled")
                    return
            click.echo()

        # Execute ingestion according to source
        if source == "confluence":
            results = pipeline.run_confluence_ingestion(force_rebuild=force_rebuild)
        else:
            click.echo(f"❌ Source '{source}' not supported")
            return

        # Display results
        if results["success"]:
            stats = results["statistics"]
            duration = results["duration_seconds"]

            click.echo("✅ Ingestion completed successfully!")
            click.echo()
            click.echo("📊 Statistics:")
            click.echo(f"   📄 Documents extracted: {stats['documents_extracted']}")
            click.echo(f"   ✂️ Chunks created: {stats['chunks_created']}")
            click.echo(f"   🔢 Embeddings generated: {stats['embeddings_generated']}")
            click.echo(f"   💾 Documents stored: {stats['documents_stored']}")
            click.echo(f"   ⏱️ Duration: {duration:.1f}s")

            # Vector database info
            db_info = results["vector_db_info"]
            if "points_count" in db_info:
                click.echo(f"   📚 Total in database: {db_info['points_count']} documents")

            click.echo()
            click.echo("🎉 Vector database is ready for queries!")

        else:
            click.echo(f"❌ Ingestion failed: {results['error']}")

            if "statistics" in results and results["statistics"]["errors"]:
                click.echo("\nDetailed errors:")
                for error in results["statistics"]["errors"]:
                    click.echo(f"   • {error}")

    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}")
        if verbose:
            import traceback

            click.echo("\nFull traceback:")
            click.echo(traceback.format_exc())


if __name__ == "__main__":
    ingest()
