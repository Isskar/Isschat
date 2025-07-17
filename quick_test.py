#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.rag.semantic_pipeline import SemanticRAGPipelineFactory

pipeline = SemanticRAGPipelineFactory.create_semantic_pipeline(use_semantic_features=True)
if pipeline.is_ready():
    answer, sources = pipeline.process_query('test')
    print('Query: test')
    print('Sources:', sources)
    print('Has sources:', sources != 'No sources')
else:
    print('Pipeline not ready')