#!/usr/bin/env python3
"""
Script pour récupérer et chunker des pages Confluence
Utilise les composants existants du projet pour tester différentes stratégies de chunking

Usage: uv run chunking_test.py
"""

import os
import json
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass
from datetime import datetime

# Charger les variables d'environnement depuis .env
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # Si dotenv n'est pas disponible, continuer sans
    pass

# Import des composants du projet
from src.ingestion.connectors.confluence_connector import ConfluenceConnector
from src.ingestion.processors.chunker import ConfluenceChunker
from src.core.interfaces import Document
from src.config.settings import IsschatConfig


@dataclass
class ChunkingResult:
    """Résultat du chunking avec métadonnées"""

    document_title: str
    total_chunks: int
    strategy: str
    chunks: List[Dict[str, Any]]
    processing_time: float


class ConfluenceChunkingTester:
    """Testeur pour différentes stratégies de chunking Confluence"""

    def __init__(self):
        self.settings = IsschatConfig.from_env()
        self.connector = None
        self.chunker = None

    def setup_confluence_connection(self) -> bool:
        """Configure la connexion Confluence"""
        try:
            # Vérifier les variables d'environnement
            required_vars = [
                "CONFLUENCE_PRIVATE_API_KEY",
                "CONFLUENCE_SPACE_KEY",
                "CONFLUENCE_SPACE_NAME",
                "CONFLUENCE_EMAIL_ADDRESS",
            ]

            missing_vars = [var for var in required_vars if not os.getenv(var)]
            if missing_vars:
                print(f"❌ Variables d'environnement manquantes: {', '.join(missing_vars)}")
                return False

            # Créer la configuration pour le connecteur
            config = {
                "confluence_private_api_key": os.getenv("CONFLUENCE_PRIVATE_API_KEY"),
                "confluence_space_key": os.getenv("CONFLUENCE_SPACE_KEY"),
                "confluence_space_name": os.getenv("CONFLUENCE_SPACE_NAME"),
                "confluence_email_address": os.getenv("CONFLUENCE_EMAIL_ADDRESS"),
            }

            self.connector = ConfluenceConnector(config)

            print("✅ Connexion Confluence configurée")

            # L'enrichisseur est maintenant intégré dans le connecteur
            if self.connector.enricher:
                print("✅ Enrichisseur de métadonnées intégré et opérationnel")
            else:
                print("⚠️ Enrichissement non disponible (API inaccessible)")

            return True

        except Exception as e:
            print(f"❌ Erreur lors de la configuration Confluence: {e}")
            return False

    def search_pages_by_title(self, title_pattern: str) -> List[Document]:
        """Recherche des pages par titre"""
        try:
            # Utiliser CQL pour rechercher par titre
            cql_query = f'space = "{self.connector.space_key}" and type = "page" and title ~ "{title_pattern}"'
            print(f"🔍 Recherche des pages avec le titre: {title_pattern}")

            # Utiliser directement le reader du connector
            llamaindex_docs = self.connector.reader.load_data(
                cql=cql_query,
                include_attachments=self.connector.include_attachments,
            )

            # Convertir en documents
            documents = self.connector._convert_llamaindex_documents(llamaindex_docs)

            if not documents:
                print("❌ Aucune page trouvée")
                return []

            print(f"✅ {len(documents)} page(s) trouvée(s)")
            for i, doc in enumerate(documents):
                print(f"  {i + 1}. {doc.metadata.get('title', 'Sans titre')}")

            return documents

        except Exception as e:
            print(f"❌ Erreur lors de la recherche: {e}")
            return []

    def chunk_document(
        self, document: Document, strategy: Literal["semantic_hierarchical"] = "semantic_hierarchical"
    ) -> Optional[ChunkingResult]:
        """Chunke un document avec une stratégie donnée"""
        try:
            start_time = datetime.now()

            # Debug: vérifier le contenu du document
            print(f"📄 Document original - Titre: {document.metadata.get('title', 'Sans titre')}")
            print(f"📄 Longueur du contenu: {len(document.page_content)} caractères")
            print(f"📄 Premiers 200 caractères: {document.page_content[:200]}...")

            # Initialiser le chunker avec la stratégie
            self.chunker = ConfluenceChunker(strategy=strategy)

            print(f"🔄 Chunking du document avec la stratégie: {strategy}")

            # Chunker le document
            chunks = self.chunker.chunk_document(document)

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            # L'enrichissement est maintenant automatique dans le connecteur
            # Les chunks sont déjà enrichis si l'API est disponible
            if self.connector.enricher:
                print("✅ Chunks avec métadonnées enrichies (automatique)")
            else:
                print("⚠️ Chunks sans enrichissement (API non disponible)")

            # Debug: vérifier les chunks
            print(f"🔍 Chunks générés: {len(chunks)}")
            for i, chunk in enumerate(chunks):
                print(f"  Chunk {i + 1}: {len(chunk.page_content)} caractères")
                if hasattr(chunk, "content"):
                    print(f"    Contenu via .content: {len(chunk.content)} caractères")

            # Préparer le résultat
            chunk_data = []
            for i, chunk in enumerate(chunks):
                # Utiliser page_content au lieu de content
                content = chunk.page_content
                chunk_info = {
                    "index": i + 1,
                    "content": content,
                    "metadata": chunk.metadata,
                    "size_chars": len(content),
                    "size_tokens": chunk.metadata.get("token_count", 0),
                }
                chunk_data.append(chunk_info)

            result = ChunkingResult(
                document_title=document.metadata.get("title", "Sans titre"),
                total_chunks=len(chunks),
                strategy=strategy,
                chunks=chunk_data,
                processing_time=processing_time,
            )

            print(f"✅ Chunking terminé: {len(chunks)} chunks en {processing_time:.2f}s")
            return result

        except Exception as e:
            print(f"❌ Erreur lors du chunking: {e}")
            import traceback

            traceback.print_exc()
            return None

    def display_chunks(self, result: ChunkingResult, max_content_length: int = 500):
        """Affiche les chunks de manière lisible"""
        print("\n" + "=" * 80)
        print("RÉSULTAT DU CHUNKING")
        print("=" * 80)
        print(f"Document: {result.document_title}")
        print(f"Stratégie: {result.strategy}")
        print(f"Nombre de chunks: {result.total_chunks}")
        print(f"Temps de traitement: {result.processing_time:.2f}s")
        print("-" * 80)

        for chunk in result.chunks:
            print(f"\n📄 CHUNK {chunk['index']}")
            print(f"Taille: {chunk['size_chars']} caractères, {chunk['size_tokens']} tokens")

            # Afficher les métadonnées importantes
            metadata = chunk["metadata"]
            if "section_path" in metadata:
                print(f"Section: {metadata['section_path']}")
            if "content_type" in metadata:
                print(f"Type: {metadata['content_type']}")

            # Afficher les nouvelles métadonnées enrichies
            if "author_name" in metadata:
                print(f"Auteur: {metadata['author_name']}")
            if "hierarchy_breadcrumb" in metadata:
                print(f"Chemin: {metadata['hierarchy_breadcrumb']}")
            if "contributors_count" in metadata and metadata["contributors_count"] > 0:
                print(f"Contributeurs: {metadata['contributors_count']}")
            if "created_date" in metadata:
                created = metadata["created_date"][:10] if metadata["created_date"] else "N/A"
                print(f"Créé: {created}")

            # Séparer les métadonnées du contenu réel
            content = chunk["content"]

            # Détecter et séparer le contexte ajouté automatiquement
            if content.startswith("[Document:") and "]" in content:
                # Trouver la fin des métadonnées
                context_end = content.find("]") + 1
                context_info = content[:context_end]
                actual_content = content[context_end:].strip()

                print(f"Contexte: {context_info}")

                # Afficher le contenu réel
                if len(actual_content) > max_content_length:
                    displayed_content = actual_content[:max_content_length] + "..."
                else:
                    displayed_content = actual_content

                print(f"Contenu réel:\n{displayed_content}")
            else:
                # Pas de contexte détecté, afficher normalement
                if len(content) > max_content_length:
                    content = content[:max_content_length] + "..."
                print(f"Contenu:\n{content}")

            print("-" * 40)

    def save_results(self, results: Dict[str, ChunkingResult], filename: Optional[str] = None):
        """Sauvegarde les résultats dans un fichier JSON"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chunking_results_{timestamp}.json"

        try:
            # Préparer les données pour JSON
            json_data = {}
            for strategy, result in results.items():
                json_data[strategy] = {
                    "document_title": result.document_title,
                    "total_chunks": result.total_chunks,
                    "processing_time": result.processing_time,
                    "chunks": result.chunks,
                }

            with open(filename, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

            print(f"💾 Résultats sauvegardés dans: {filename}")

        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde: {e}")


def main():
    """Fonction principale du script"""
    print("🚀 Testeur de chunking Confluence")
    print("=" * 50)

    tester = ConfluenceChunkingTester()

    # Configuration de la connexion
    if not tester.setup_confluence_connection():
        print("❌ Impossible de se connecter à Confluence")
        return

    # Demander le titre de la page
    title_pattern = input("\n📝 Entrez le titre de la page (ou une partie): ").strip()

    if not title_pattern:
        print("❌ Titre requis")
        return

    # Rechercher les pages
    documents = tester.search_pages_by_title(title_pattern)

    if not documents:
        return

    # Sélectionner la page si plusieurs résultats
    if len(documents) > 1:
        print("\n🔢 Plusieurs pages trouvées. Sélectionnez:")
        for i, doc in enumerate(documents):
            print(f"  {i + 1}. {doc.metadata.get('title', 'Sans titre')}")

        try:
            choice = int(input("\nNuméro de la page: ")) - 1
            if 0 <= choice < len(documents):
                selected_doc = documents[choice]
            else:
                print("❌ Sélection invalide")
                return
        except ValueError:
            print("❌ Sélection invalide")
            return
    else:
        selected_doc = documents[0]

    print(f"\n📄 Page sélectionnée: {selected_doc.metadata.get('title', 'Sans titre')}")

    # Test avec semantic_hierarchical uniquement (seule stratégie efficace)
    print("\n🎯 Options disponibles:")
    print("  1. Chunker avec semantic_hierarchical")
    print("  2. Test debug - afficher contenu complet d'un chunk")

    try:
        choice = int(input("\nChoisissez une option: "))

        if choice == 1:
            # Chunking avec semantic_hierarchical
            result = tester.chunk_document(selected_doc, "semantic_hierarchical")

            if result:
                tester.display_chunks(result)

                # Sauvegarder ?
                save = input("\n💾 Sauvegarder les résultats ? (o/n): ").lower()
                if save == "o":
                    tester.save_results({"semantic_hierarchical": result})

        elif choice == 2:
            # Test debug - afficher contenu complet
            result = tester.chunk_document(selected_doc, "semantic_hierarchical")
            if result and result.chunks:
                chunk = result.chunks[0]  # Premier chunk
                print("\n🔍 DEBUG - CONTENU COMPLET DU PREMIER CHUNK:")
                print(f"Taille totale: {len(chunk['content'])} caractères")
                print("=" * 80)
                print(chunk["content"])
                print("=" * 80)
        else:
            print("❌ Option invalide")

    except ValueError:
        print("❌ Option invalide")


if __name__ == "__main__":
    main()
