#!/usr/bin/env python3
"""
Script simple pour générer un rapport HTML à partir de la dernière évaluation
"""

import json
from pathlib import Path
from datetime import datetime
import sys

# Ajouter le répertoire rag_evaluation au path
sys.path.append(str(Path(__file__).parent / "rag_evaluation"))

from rag_evaluation.report_generator import HTMLReportGenerator


def main():
    print("🎯 Génération du rapport HTML avec métriques de retrieval")
    print("=" * 60)

    # Chercher le fichier de résultats le plus récent
    eval_results_dir = Path("evaluation_results")
    if not eval_results_dir.exists():
        print("❌ Répertoire evaluation_results non trouvé")
        return

    json_files = list(eval_results_dir.glob("evaluation_results_*.json"))
    if not json_files:
        print("❌ Aucun fichier de résultats trouvé")
        return

    # Prendre le plus récent
    latest_file = max(json_files, key=lambda f: f.stat().st_mtime)
    print(f"✅ Fichier trouvé: {latest_file}")

    try:
        # Charger les résultats
        with open(latest_file, "r", encoding="utf-8") as f:
            results = json.load(f)

        # Générer le rapport
        generator = HTMLReportGenerator()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"rapport_avec_retrieval_{timestamp}.html")

        report_path = generator.generate_report(
            results, output_path, title="Rapport d'Évaluation Isschat - Avec Métriques Retrieval"
        )

        print(f"✅ Rapport généré: {report_path}")
        print("\n📋 Le rapport contient:")
        print("• 📊 Onglet Résumé: Vue d'ensemble")
        print("• 🔧 Onglet Robustesse: Tests détaillés")
        print("• 🔍 Onglet Métriques Retrieval: Métriques avec explications")
        print("\n🌐 Ouvrez le fichier dans votre navigateur")

    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
