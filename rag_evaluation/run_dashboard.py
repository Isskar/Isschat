#!/usr/bin/env python3
"""
Launch script for the Isschat Evaluation Dashboard
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Launch the Streamlit dashboard"""
    dashboard_path = Path(__file__).parent / "evaluation_dashboard.py"

    if not dashboard_path.exists():
        print(f"❌ Dashboard file not found: {dashboard_path}")
        return 1

    # Check if evaluation_results directory exists
    results_dir = Path(__file__).parent / "../evaluation_results"
    if not results_dir.exists():
        print(f"❌ evaluation_results directory not found: {results_dir}")
        return 1

    # Count available files
    json_files = list(results_dir.glob("*.json"))
    if not json_files:
        print(f"❌ No JSON files found in {results_dir}")
        return 1

    print(f"✅ Found {len(json_files)} evaluation files")
    print("🚀 Starting dashboard...")

    # Launch Streamlit
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(dashboard_path), "--server.port", "8501"], check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching dashboard: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n✅ Dashboard stopped")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
