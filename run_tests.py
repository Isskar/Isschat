#!/usr/bin/env python3
"""
Script simple pour lancer les tests Isschat
"""

import subprocess
import sys
import os


def run_tests():
    """Run tests with pytest"""
    print("🧪 Lancement des tests Isschat...")

    # Change to project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    try:
        # Run pytest
        result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"], check=False)

        if result.returncode == 0:
            print("✅ Tous les tests sont passés !")
        else:
            print("❌ Certains tests ont échoué")

        return result.returncode

    except FileNotFoundError:
        print("❌ pytest non trouvé. Installez les dépendances de test:")
        print("   uv sync --extra test")
        return 1
    except Exception as e:
        print(f"❌ Erreur lors de l'exécution des tests: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
