"""
Query command for Isschat CLI.
Allows direct querying of the vector database with detailed chunk display.
"""

import click
import time
from pathlib import Path
import sys
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from ...config import get_config
from ...rag.tools.retrieval_tool import RetrievalTool
from ...rag.tools.generation_tool import GenerationTool


@click.command()
@click.option("-q", "--query", required=True, help="Query to search for")
@click.option("-k", "--top-k", default=5, help="Number of chunks to retrieve")
@click.option("-s", "--score-threshold", default=0.0, help="Minimum score threshold")
@click.option("-v", "--verbose", is_flag=True, help="Show detailed chunk information")
@click.option("--show-metadata", is_flag=True, help="Display document metadata")
@click.option("--show-content", is_flag=True, default=True, help="Display chunk content")
@click.option("--show-stats", is_flag=True, help="Display statistics about sources and scores")
@click.option("--no-llm", is_flag=True, help="Skip LLM generation and only show retrieved chunks")
def query(
    query: str,
    top_k: int,
    score_threshold: float,
    verbose: bool,
    show_metadata: bool,
    show_content: bool,
    show_stats: bool,
    no_llm: bool,
):
    """
    Query the vector database and display retrieved chunks.

    Example:
        isschat-cli query -q "How to configure authentication?" -k 3 --show-metadata
    """
    click.echo(f"🔍 Searching for: {query}")
    click.echo("=" * 80)

    try:
        # Track timing for each step
        total_start = time.time()

        # Initialize components
        init_start = time.time()
        config = get_config()
        init_time = (time.time() - init_start) * 1000

        # Create retrieval tool
        retrieval_tool = RetrievalTool()

        # Perform search
        search_start = time.time()
        results = retrieval_tool.retrieve(query, k=top_k)
        search_time = (time.time() - search_start) * 1000

        # Note: encoding time is now included in search time
        encoding_time = 0  # No separate encoding step

        # Generate LLM response if requested
        generation_time = 0
        llm_response = None
        if not no_llm and results:
            generation_start = time.time()
            generation_tool = GenerationTool()
            # Convert results to context format
            context_docs = []
            for chunk in results:
                if hasattr(chunk, "page_content"):
                    content = chunk.page_content
                else:
                    content = str(chunk)

                if hasattr(chunk, "metadata"):
                    metadata = chunk.metadata
                else:
                    metadata = {}

                context_docs.append({"content": content, "metadata": metadata})

            # Call generation tool with correct parameters
            llm_response_dict = generation_tool.generate(
                query=query,
                documents=results,  # Pass SearchResult objects directly
                history="",  # Empty history for CLI query
            )
            # Extract the answer from response dict
            llm_response = llm_response_dict.get("answer", "")
            generation_time = (time.time() - generation_start) * 1000

        total_time = (time.time() - total_start) * 1000

        if not results:
            click.echo("❌ No chunks found matching your query.")
            return

        click.echo(f"✅ Found {len(results)} chunks in {total_time:.1f}ms")

        # Display timing breakdown
        if show_stats or verbose:
            click.echo("\n⏱️  Timing breakdown:")
            click.echo(f"   • Initialization: {init_time:.1f}ms")
            click.echo(f"   • Query encoding: {encoding_time:.1f}ms")
            click.echo(f"   • Vector search: {search_time:.1f}ms")
            if not no_llm and llm_response:
                click.echo(f"   • LLM generation: {generation_time:.1f}ms")
            click.echo(f"   • Total time: {total_time:.1f}ms")

        # Collect statistics
        source_scores = defaultdict(list)
        all_scores = []

        for chunk in results:
            if hasattr(chunk, "metadata") and chunk.metadata:
                source = chunk.metadata.get("source", "Unknown")
                title = chunk.metadata.get("title", "Unknown")
                if hasattr(chunk, "score") and chunk.score is not None:
                    source_scores[f"{source}: {title}"].append(chunk.score)
                    all_scores.append(chunk.score)

        # Display statistics
        if show_stats and all_scores:
            click.echo("\n📊 Score Statistics:")
            click.echo(f"   • Average score: {sum(all_scores) / len(all_scores):.4f}")
            click.echo(f"   • Max score: {max(all_scores):.4f}")
            click.echo(f"   • Min score: {min(all_scores):.4f}")

            click.echo("\n📄 Scores by Source:")
            for source, scores in sorted(source_scores.items(), key=lambda x: max(x[1]), reverse=True):
                avg_score = sum(scores) / len(scores)
                click.echo(f"   • {source}")
                click.echo(f"     - Chunks: {len(scores)}")
                click.echo(f"     - Avg score: {avg_score:.4f}")
                click.echo(f"     - Max score: {max(scores):.4f}")

        # Display LLM response if generated
        if llm_response and not no_llm:
            click.echo("\n🤖 LLM Response:")
            click.echo("=" * 80)
            click.echo(llm_response)
            click.echo("=" * 80)

        # Display results
        click.echo("\n🔍 Retrieved Chunks:")
        for i, chunk in enumerate(results, 1):
            click.echo(f"\n{'=' * 80}")
            click.echo(f"📄 Chunk {i}/{len(results)}")
            click.echo(f"{'=' * 80}")

            # Display score
            if hasattr(chunk, "score") and chunk.score is not None:
                click.echo(f"📊 Score: {chunk.score:.4f}")

            # Display metadata
            if show_metadata:
                # Extract metadata from SearchResult
                if hasattr(chunk, "document") and chunk.document.metadata:
                    metadata = chunk.document.metadata
                elif hasattr(chunk, "metadata") and chunk.metadata:
                    metadata = chunk.metadata
                else:
                    metadata = {}

                if metadata:
                    click.echo("\n📋 Metadata:")
                    for key, value in metadata.items():
                        if key not in ["content", "text"]:  # Skip content fields in metadata
                            click.echo(f"   • {key}: {value}")

            # Display content
            if show_content:
                click.echo("\n📝 Content:")
                click.echo("-" * 80)
                # Try different ways to get content
                content = None
                if hasattr(chunk, "page_content"):
                    content = chunk.page_content
                elif hasattr(chunk, "document") and hasattr(chunk.document, "content"):
                    content = chunk.document.content
                elif hasattr(chunk, "content"):
                    content = chunk.content
                else:
                    content = str(chunk)

                if verbose:
                    # Show full content
                    click.echo(content)
                else:
                    # Show truncated content
                    max_lines = 10
                    lines = content.split("\n")
                    if len(lines) > max_lines:
                        click.echo("\n".join(lines[:max_lines]))
                        click.echo(f"\n... ({len(lines) - max_lines} more lines) ...")
                    else:
                        click.echo(content)

            # Display source information
            # Extract metadata from SearchResult
            if hasattr(chunk, "document") and chunk.document.metadata:
                metadata = chunk.document.metadata
            elif hasattr(chunk, "metadata"):
                metadata = chunk.metadata
            else:
                metadata = {}

            source = metadata.get("source", "Unknown")
            doc_id = metadata.get("doc_id", "Unknown")
            click.echo(f"\n📍 Source: {source}")
            if doc_id != "Unknown":
                click.echo(f"🆔 Document ID: {doc_id}")

        click.echo(f"\n{'=' * 80}")
        click.echo(f"✅ Query completed in {total_time:.1f}ms")

    except Exception as e:
        click.echo(f"❌ Error during query: {str(e)}")
        if verbose:
            import traceback

            click.echo("\n🔍 Full error trace:")
            click.echo(traceback.format_exc())
