"""
Configuration module for Isschat evaluation system
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from evaluation_config import EvaluationConfig

__all__ = ["EvaluationConfig"]
