"""
Data ingestion pipeline for building vector databases.
Separated from RAG for clear concerns.
"""

from .base_pipeline import BaseIngestionPipeline
from .confluence_pipeline import ConfluenceIngestionPipeline, create_confluence_pipeline
from .sharepoint_pipeline import SharePointPipeline, create_sharepoint_pipeline

__all__ = [
    "BaseIngestionPipeline",
    "ConfluenceIngestionPipeline",
    "create_confluence_pipeline",
    "SharePointPipeline",
    "create_sharepoint_pipeline",
]
