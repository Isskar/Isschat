#!/usr/bin/env python3
"""
Simple runner script for Isschat evaluation system
"""

import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from rag_evaluation.main import main

if __name__ == "__main__":
    print("🚀 Starting Isschat Evaluation System")
    print("=" * 50)
    main()
