#!/usr/bin/env python3
"""
Simple runner script for Isschat evaluation system
"""

import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

<<<<<<< HEAD:rag_evaluation/run_evaluation.py
from rag_evaluation.main import main
=======
from eval_answer_relevance import main  # ty : ignore
>>>>>>> 2d8a14707a8d514947aaf070a4c502571e14c822:Evaluation/run_evaluation.py

if __name__ == "__main__":
    print("🚀 Starting Isschat Evaluation System")
    print("=" * 50)
    main()
