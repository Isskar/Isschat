"""
Utility module to import config from root directory
Avoids conflicts with rag_evaluation.config package
"""

import sys
import importlib.util
from pathlib import Path


def get_root_config():
    """Import and return the root config module"""
    # Ensure root directory is in path
    root_dir = str(Path(__file__).parent.parent)
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    # Import from root config.py explicitly
    config_path = Path(__file__).parent.parent / "config.py"
    spec = importlib.util.spec_from_file_location("root_config", config_path)
    root_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(root_config)
    return root_config


# Export the functions we need
_root_config = get_root_config()
get_config = _root_config.get_config
get_debug_info = _root_config.get_debug_info
