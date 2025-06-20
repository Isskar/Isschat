#!/usr/bin/env python3
"""
Script pour générer un rapport HTML à partir de la dernière évaluation
Inclut maintenant un onglet spécialisé pour les métriques de retrieval
"""

import json
import glob
from pathlib import Path
from datetime import datetime
import sys
import os

# Ajouter le répertoire rag_evaluation au path
sys.path.append(str(Path(__file__).parent / "rag_evaluation"))

from rag_evaluation.report_generator import HTMLReportGenerator


def find_latest_evaluation_results():
    """Trouve le fichier de résultats d'évaluation le plus récent"""
    search_paths = [Path("evaluation_results"), Path("."), Path("rag_evaluation")]

    all_files = []
    for search_path in search_paths:
        if search_path.exists():
            files = list(search_path.glob("evaluation_results_*.json"))
            all_files.extend(files)

    if all_files:
        return max(all_files, key=lambda f: f.stat().st_mtime)
    return None


def generate_report_from_latest():
    """Génère un rapport HTML à partir de la dernière évaluation"""
    print("🔍 Recherche du dernier fichier de résultats d'évaluation...")

    latest_file = find_latest_evaluation_results()

    if not latest_file:
        print("❌ Aucun fichier de résultats d'évaluation trouvé.")
        print("💡 Exécutez d'abord une évaluation avec:")
        print("   python rag_evaluation/main.py --categories robustness retrieval")
        return None

    print(f"✅ Fichier trouvé: {latest_file}")
    print(f"📅 Modifié le: {datetime.fromtimestamp(latest_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Charger les résultats
        with open(latest_file, "r", encoding="utf-8") as f:
            results = json.load(f)

        print(f"📊 Résultats chargés: {results.get('overall_stats', {}).get('total_tests', 0)} tests")

        # Afficher les catégories disponibles
        categories = list(results.get("category_results", {}).keys())
        print(f"📂 Catégories disponibles: {', '.join(categories)}")

        # Générer le rapport HTML avec onglets
        generator = HTMLReportGenerator()

        # Nom du fichier de sortie basé sur le timestamp du fichier de résultats
        timestamp = datetime.fromtimestamp(latest_file.stat().st_mtime).strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"rapport_evaluation_complet_{timestamp}.html")

        report_path = generator.generate_report(
            results, output_path, title="Rapport d'Évaluation Isschat - Avec Métriques Retrieval"
        )

        print(f"✅ Rapport HTML généré: {report_path}")
        print(f"🌐 Ouvrez le fichier dans votre navigateur pour voir le rapport")

        # Afficher les fonctionnalités du rapport
        print("\n" + "=" * 60)
        print("📋 FONCTIONNALITÉS DU RAPPORT:")
        print("=" * 60)
        print("📊 • Onglet Résumé: Vue d'ensemble des résultats")
        if "robustness" in categories:
            print("🔧 • Onglet Robustesse: Tests détaillés avec questions/réponses")
        if "retrieval" in categories:
            print("🔍 • Onglet Métriques Retrieval: Métriques de performance détaillées")
            print("     - Precision, Recall, F1-Score")
            print("     - Precision@K et Recall@K (K=1,3,5,10)")
            print("     - MRR, MAP, NDCG@5/10")
            print("     - Explications de chaque métrique")
        print("=" * 60)

        return report_path

    except Exception as e:
        print(f"❌ Erreur lors de la génération du rapport: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    """Point d'entrée principal"""
    print("=" * 60)
    print("🎯 GÉNÉRATEUR DE RAPPORT D'ÉVALUATION ISSCHAT")
    print("   Avec onglet spécialisé pour les métriques de retrieval")
    print("=" * 60)

    report_path = generate_report_from_latest()

    if report_path:
        print("\n" + "=" * 60)
        print("✅ RAPPORT GÉNÉRÉ AVEC SUCCÈS")
        print("=" * 60)
        print(f"📄 Fichier: {report_path}")
        print(f"📁 Taille: {report_path.stat().st_size / 1024:.1f} KB")

        # Proposer d'ouvrir le fichier
        try:
            import webbrowser

            response = input("\n🌐 Voulez-vous ouvrir le rapport dans votre navigateur ? (o/N): ")
            if response.lower() in ["o", "oui", "y", "yes"]:
                webbrowser.open(f"file://{report_path.absolute()}")
                print("🚀 Rapport ouvert dans le navigateur")
                print("\n💡 Utilisez les onglets en haut pour naviguer entre:")
                print("   • Résumé global")
                print("   • Tests de robustesse détaillés")
                print("   • Métriques de retrieval avec explications")
        except ImportError:
            print("💡 Ouvrez manuellement le fichier dans votre navigateur")
    else:
        print("\n" + "=" * 60)
        print("❌ ÉCHEC DE LA GÉNÉRATION DU RAPPORT")
        print("=" * 60)


if __name__ == "__main__":
    main()
