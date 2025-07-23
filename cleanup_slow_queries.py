#!/usr/bin/env python3
"""
Script to delete queries with response time > 10 seconds from Azure Blob Storage
This script processes conversation data stored as JSONL files and removes entries with grpc timeouts
"""

import json
import logging
import sys
import os
from datetime import datetime
from typing import List, Dict, Any, Tuple

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

try:
    from src.storage.data_manager import get_data_manager
    from src.config import get_config
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    print("Assurez-vous d'être dans le bon répertoire et que les dépendances sont installées")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SlowQueryCleaner:
    """Cleans up queries with response time > 10 seconds from blob storage"""

    def __init__(self, threshold_seconds: float = 10.0):
        """
        Initialize the cleaner

        Args:
            threshold_seconds: Response time threshold in seconds (default: 10.0)
        """
        self.threshold_ms = threshold_seconds * 1000  # Convert to milliseconds
        self.data_manager = get_data_manager()
        self.storage = self.data_manager.storage
        self.path_manager = self.data_manager.path_manager

        logger.info(f"🔧 Initialized SlowQueryCleaner with threshold: {threshold_seconds}s ({self.threshold_ms}ms)")
        logger.info(f"📁 Storage type: {type(self.storage).__name__}")

    def scan_conversation_files(self) -> List[str]:
        """Scan for all conversation JSONL files in storage"""
        try:
            conversations_dir = self.path_manager.conversations_dir
            relative_dir = conversations_dir.relative_to(self.path_manager.data_dir)

            if not self.storage.directory_exists(str(relative_dir)):
                logger.warning(f"📂 Conversations directory does not exist: {relative_dir}")
                return []

            # List all conversation files
            files = self.storage.list_files(str(relative_dir), "conversations_*.jsonl")
            logger.info(f"📄 Found {len(files)} conversation files")

            return files

        except Exception as e:
            logger.error(f"❌ Error scanning conversation files: {e}")
            return []

    def scan_performance_files(self) -> List[str]:
        """Scan for all performance JSONL files in storage"""
        try:
            performance_dir = self.path_manager.performance_dir
            relative_dir = performance_dir.relative_to(self.path_manager.data_dir)

            if not self.storage.directory_exists(str(relative_dir)):
                logger.warning(f"📂 Performance directory does not exist: {relative_dir}")
                return []

            # List all performance files
            files = self.storage.list_files(str(relative_dir), "performance_*.jsonl")
            logger.info(f"⚡ Found {len(files)} performance files")

            return files

        except Exception as e:
            logger.error(f"❌ Error scanning performance files: {e}")
            return []

    def analyze_file(self, file_path: str) -> Tuple[List[Dict], List[Dict], int]:
        """
        Analyze a single file (conversation or performance)

        Returns:
            (entries_to_keep, entries_to_delete, total_entries)
        """
        try:
            if not self.storage.file_exists(file_path):
                logger.warning(f"📄 File does not exist: {file_path}")
                return [], [], 0

            content = self.storage.load_text_file(file_path)
            if not content:
                logger.warning(f"📄 File is empty: {file_path}")
                return [], [], 0

            lines = content.strip().split("\n")
            entries_to_keep = []
            entries_to_delete = []

            # Determine file type and time field
            is_performance_file = "performance_" in file_path
            time_field = "duration_ms" if is_performance_file else "response_time_ms"

            for line_num, line in enumerate(lines, 1):
                if not line.strip():
                    continue

                try:
                    entry = json.loads(line.strip())
                    response_time = entry.get(time_field, 0)

                    if response_time > self.threshold_ms:
                        entries_to_delete.append(
                            {"line_num": line_num, "entry": entry, "response_time_s": response_time / 1000}
                        )
                        if is_performance_file:
                            logger.debug(
                                f"⚡ Slow performance found: {response_time / 1000:.2f}s - {entry.get('operation', '')}"
                            )
                        else:
                            logger.debug(
                                f"🐌 Slow query found: {response_time / 1000:.2f}s - {entry.get('question', '')[:50]}..."
                            )
                    else:
                        entries_to_keep.append(entry)

                except json.JSONDecodeError as e:
                    logger.warning(f"⚠️ Invalid JSON on line {line_num} in {file_path}: {e}")
                    continue

            total_entries = len(entries_to_keep) + len(entries_to_delete)
            file_type = "performance" if is_performance_file else "conversation"
            logger.info(
                f"📊 {file_type.title()} file {file_path}: {total_entries} total, {len(entries_to_delete)} slow entries, {len(entries_to_keep)} to keep"
            )

            return entries_to_keep, entries_to_delete, total_entries

        except Exception as e:
            logger.error(f"❌ Error analyzing file {file_path}: {e}")
            return [], [], 0

    def backup_file(self, file_path: str) -> bool:
        """Create a backup of the original file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{file_path}.backup_{timestamp}"

            # Read original content
            original_content = self.storage.read_file(file_path)

            # Write backup
            success = self.storage.write_file(backup_path, original_content)

            if success:
                logger.info(f"💾 Backup created: {backup_path}")
            else:
                logger.error(f"❌ Failed to create backup: {backup_path}")

            return success

        except Exception as e:
            logger.error(f"❌ Error creating backup for {file_path}: {e}")
            return False

    def rewrite_file(self, file_path: str, entries_to_keep: List[Dict]) -> bool:
        """Rewrite file with only the entries to keep"""
        try:
            if not entries_to_keep:
                # Delete empty files
                success = self.storage.delete_file(file_path)
                if success:
                    logger.info(f"🗑️ Deleted empty file: {file_path}")
                else:
                    logger.error(f"❌ Failed to delete empty file: {file_path}")
                return success

            # Create new content
            new_lines = []
            for entry in entries_to_keep:
                new_lines.append(json.dumps(entry, ensure_ascii=False))

            new_content = "\n".join(new_lines) + "\n"

            # Write new content
            success = self.storage.write_file(file_path, new_content.encode("utf-8"))

            if success:
                logger.info(f"✅ Rewritten file: {file_path} ({len(entries_to_keep)} entries)")
            else:
                logger.error(f"❌ Failed to rewrite file: {file_path}")

            return success

        except Exception as e:
            logger.error(f"❌ Error rewriting file {file_path}: {e}")
            return False

    def process_file(self, file_path: str, create_backup: bool = True, dry_run: bool = False) -> Dict[str, Any]:
        """
        Process a single file to remove slow queries

        Args:
            file_path: Path to the conversation file
            create_backup: Whether to create a backup before modifying
            dry_run: If True, only analyze without making changes

        Returns:
            Dictionary with processing results
        """
        logger.info(f"🔄 Processing file: {file_path}")

        # Analyze file
        entries_to_keep, entries_to_delete, total_entries = self.analyze_file(file_path)

        result = {
            "file_path": file_path,
            "total_entries": total_entries,
            "entries_to_keep": len(entries_to_keep),
            "entries_to_delete": len(entries_to_delete),
            "slow_queries": entries_to_delete,
            "backup_created": False,
            "file_updated": False,
            "success": True,
        }

        if total_entries == 0:
            logger.info(f"📄 File is empty or has no valid entries: {file_path}")
            return result

        if len(entries_to_delete) == 0:
            logger.info(f"✅ No slow queries found in: {file_path}")
            return result

        if dry_run:
            logger.info(f"🔍 DRY RUN: Would delete {len(entries_to_delete)} slow queries from {file_path}")
            return result

        # Create backup if requested
        if create_backup:
            backup_success = self.backup_file(file_path)
            result["backup_created"] = backup_success

            if not backup_success:
                logger.error(f"❌ Backup failed for {file_path}, skipping modification")
                result["success"] = False
                return result

        # Rewrite file with filtered content
        rewrite_success = self.rewrite_file(file_path, entries_to_keep)
        result["file_updated"] = rewrite_success
        result["success"] = rewrite_success

        if rewrite_success:
            logger.info(f"✅ Successfully cleaned {file_path}: removed {len(entries_to_delete)} slow queries")
        else:
            logger.error(f"❌ Failed to clean {file_path}")

        return result

    def clean_all_files(self, create_backup: bool = True, dry_run: bool = False) -> Dict[str, Any]:
        """
        Clean all conversation and performance files by removing slow queries

        Args:
            create_backup: Whether to create backups before modifying
            dry_run: If True, only analyze without making changes

        Returns:
            Summary of all operations
        """
        logger.info(f"🚀 Starting cleanup of slow queries (threshold: {self.threshold_ms}ms)")

        if dry_run:
            logger.info("🔍 DRY RUN MODE: No files will be modified")

        # Get both conversation and performance files
        conversation_files = self.scan_conversation_files()
        performance_files = self.scan_performance_files()
        all_files = conversation_files + performance_files

        if not all_files:
            logger.warning("📂 No files found")
            return {
                "total_files": 0,
                "files_processed": 0,
                "total_entries": 0,
                "entries_deleted": 0,
                "files_results": [],
            }

        summary = {
            "total_files": len(all_files),
            "conversation_files": len(conversation_files),
            "performance_files": len(performance_files),
            "files_processed": 0,
            "files_modified": 0,
            "total_entries": 0,
            "entries_deleted": 0,
            "backups_created": 0,
            "files_results": [],
            "errors": [],
        }

        logger.info(f"📄 Found {len(conversation_files)} conversation files")
        logger.info(f"⚡ Found {len(performance_files)} performance files")

        for file_path in all_files:
            try:
                result = self.process_file(file_path, create_backup, dry_run)
                summary["files_results"].append(result)
                summary["files_processed"] += 1
                summary["total_entries"] += result["total_entries"]
                summary["entries_deleted"] += result["entries_to_delete"]

                if result["backup_created"]:
                    summary["backups_created"] += 1

                if result["file_updated"]:
                    summary["files_modified"] += 1

                if not result["success"]:
                    summary["errors"].append(f"Failed to process {file_path}")

            except Exception as e:
                error_msg = f"Error processing {file_path}: {e}"
                logger.error(f"❌ {error_msg}")
                summary["errors"].append(error_msg)

        # Print summary
        logger.info("\n📊 CLEANUP SUMMARY")
        logger.info(f"{'=' * 50}")
        logger.info(f"📁 Total files scanned: {summary['total_files']}")
        logger.info(f"📄 Conversation files: {summary['conversation_files']}")
        logger.info(f"⚡ Performance files: {summary['performance_files']}")
        logger.info(f"📄 Files processed: {summary['files_processed']}")
        logger.info(f"✏️ Files modified: {summary['files_modified']}")
        logger.info(f"💾 Backups created: {summary['backups_created']}")
        logger.info(f"📊 Total entries: {summary['total_entries']}")
        logger.info(f"🗑️ Slow entries deleted: {summary['entries_deleted']}")

        if summary["entries_deleted"] > 0:
            percentage = (summary["entries_deleted"] / summary["total_entries"]) * 100
            logger.info(f"📈 Percentage deleted: {percentage:.2f}%")

        if summary["errors"]:
            logger.warning(f"⚠️ Errors encountered: {len(summary['errors'])}")
            for error in summary["errors"]:
                logger.warning(f"   - {error}")

        return summary

    def show_slow_queries_details(self, limit: int = 20) -> None:
        """Show details of the slowest queries found"""
        logger.info(f"🔍 Analyzing slowest entries (limit: {limit})")

        conversation_files = self.scan_conversation_files()
        performance_files = self.scan_performance_files()
        all_files = conversation_files + performance_files

        all_slow_entries = []

        for file_path in all_files:
            _, entries_to_delete, _ = self.analyze_file(file_path)

            for entry_data in entries_to_delete:
                entry_data["file_path"] = file_path
                entry_data["is_performance"] = "performance_" in file_path
                all_slow_entries.append(entry_data)

        if not all_slow_entries:
            logger.info("✅ No slow entries found!")
            return

        # Sort by response time (slowest first)
        all_slow_entries.sort(key=lambda x: x["response_time_s"], reverse=True)

        logger.info(f"\n🐌 TOP {min(limit, len(all_slow_entries))} SLOWEST ENTRIES")
        logger.info(f"{'=' * 80}")

        for i, entry_data in enumerate(all_slow_entries[:limit], 1):
            entry = entry_data["entry"]
            response_time = entry_data["response_time_s"]
            file_path = entry_data["file_path"]
            is_performance = entry_data["is_performance"]

            logger.info(f"\n{i}. Response time: {response_time:.2f}s")
            logger.info(f"   📅 Timestamp: {entry.get('timestamp', 'N/A')}")
            logger.info(f"   👤 User: {entry.get('user_id', 'N/A')}")

            if is_performance:
                logger.info(f"   ⚡ Operation: {entry.get('operation', 'N/A')}")
                logger.info("   📊 Type: Performance metric")
            else:
                logger.info(f"   💬 Conversation: {entry.get('conversation_id', 'N/A')}")
                logger.info(f"   ❓ Question: {entry.get('question', 'N/A')[:100]}...")
                logger.info("   📊 Type: Conversation")

            logger.info(f"   📄 File: {file_path}")


def main():
    """Main function with CLI interface"""
    import argparse

    parser = argparse.ArgumentParser(description="Clean up slow queries from Azure Blob Storage")
    parser.add_argument(
        "--threshold", "-t", type=float, default=10.0, help="Response time threshold in seconds (default: 10.0)"
    )
    parser.add_argument("--dry-run", "-d", action="store_true", help="Analyze only, don't modify files")
    parser.add_argument("--no-backup", action="store_true", help="Don't create backup files before modification")
    parser.add_argument("--show-slow", "-s", type=int, metavar="N", help="Show details of N slowest queries and exit")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        cleaner = SlowQueryCleaner(threshold_seconds=args.threshold)

        if args.show_slow:
            cleaner.show_slow_queries_details(limit=args.show_slow)
            return

        create_backup = not args.no_backup
        summary = cleaner.clean_all_files(create_backup=create_backup, dry_run=args.dry_run)

        if args.dry_run:
            print("\n🔍 DRY RUN COMPLETE")
            print(f"   - Would delete {summary['entries_deleted']} slow queries")
            print(f"   - From {summary['files_modified']} files")
            print("   - Run without --dry-run to perform actual cleanup")
        else:
            print("\n✅ CLEANUP COMPLETE")
            print(f"   - Deleted {summary['entries_deleted']} slow queries")
            print(f"   - Modified {summary['files_modified']} files")
            if create_backup:
                print(f"   - Created {summary['backups_created']} backup files")

    except KeyboardInterrupt:
        print("\n👋 Cleanup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
