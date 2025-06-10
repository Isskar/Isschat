#!/usr/bin/env python3
"""
Simple runner script for Isschat evaluation
"""

import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from eval_answer_relevance import main  # ty : ignore

if __name__ == "__main__":
    print("🚀 Starting Isschat Evaluation System")
    print("=" * 50)
    main()
