"""
Dataset Manager for RAG evaluation test cases
"""

import csv
import json
import logging
import os
from typing import List, Dict, Any, Optional

from .models import TestCase, TestType, RobustnessTestType, Difficulty
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.evaluation_config import EvaluationConfig, get_evaluation_config


class DatasetManager:
    """
    Manages test datasets for RAG evaluation
    """

    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        Initialize Dataset Manager

        Args:
            config: Evaluation configuration. If None, loads from environment.
        """
        self.config = config or get_evaluation_config()
        self.logger = logging.getLogger(__name__)

        # Ensure dataset directories exist
        self._ensure_directories()

        self.logger.info("DatasetManager initialized")

    def _ensure_directories(self):
        """Ensure dataset directories exist"""
        for dataset_path in self.config.test_datasets.values():
            directory = os.path.dirname(dataset_path)
            os.makedirs(directory, exist_ok=True)

    def load_dataset(self, dataset_name: str) -> List[TestCase]:
        """
        Load test cases from a dataset file

        Args:
            dataset_name: Name of the dataset (robustness, performance, quality, consistency)

        Returns:
            List[TestCase]: List of test cases
        """
        if dataset_name not in self.config.test_datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        dataset_path = self.config.test_datasets[dataset_name]

        if not os.path.exists(dataset_path):
            self.logger.warning(f"Dataset file not found: {dataset_path}")
            return []

        try:
            if dataset_path.endswith(".tsv"):
                return self._load_tsv_dataset(dataset_path)
            elif dataset_path.endswith(".json"):
                return self._load_json_dataset(dataset_path)
            else:
                raise ValueError(f"Unsupported file format: {dataset_path}")

        except Exception as e:
            self.logger.error(f"Error loading dataset {dataset_name}: {e}")
            return []

    def _load_tsv_dataset(self, file_path: str) -> List[TestCase]:
        """Load test cases from TSV file"""
        test_cases = []

        with open(file_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file, delimiter="\t")

            for row_num, row in enumerate(reader, start=2):  # Start at 2 for header
                try:
                    test_case = self._parse_tsv_row(row, row_num)
                    if test_case:
                        test_cases.append(test_case)
                except Exception as e:
                    self.logger.warning(f"Error parsing row {row_num} in {file_path}: {e}")

        self.logger.info(f"Loaded {len(test_cases)} test cases from {file_path}")
        return test_cases

    def _parse_tsv_row(self, row: Dict[str, str], row_num: int) -> Optional[TestCase]:
        """Parse a single TSV row into a TestCase"""
        # Required fields
        if not row.get("question"):
            self.logger.warning(f"Row {row_num}: Missing required field 'question'")
            return None

        # Generate ID if not provided
        test_id = row.get("id", f"test_{row_num}")

        # Parse test type
        test_type_str = row.get("test_type", "quality").lower()
        try:
            test_type = TestType(test_type_str)
        except ValueError:
            self.logger.warning(f"Row {row_num}: Invalid test_type '{test_type_str}', using 'quality'")
            test_type = TestType.QUALITY

        # Parse robustness type if applicable
        robustness_type = None
        if test_type == TestType.ROBUSTNESS and row.get("robustness_type"):
            try:
                robustness_type = RobustnessTestType(row["robustness_type"].lower())
            except ValueError:
                self.logger.warning(f"Row {row_num}: Invalid robustness_type '{row['robustness_type']}'")

        # Parse difficulty
        difficulty_str = row.get("difficulty", "medium").lower()
        try:
            difficulty = Difficulty(difficulty_str)
        except ValueError:
            self.logger.warning(f"Row {row_num}: Invalid difficulty '{difficulty_str}', using 'medium'")
            difficulty = Difficulty.MEDIUM

        # Parse expected sources
        expected_sources = None
        if row.get("expected_sources"):
            expected_sources = [s.strip() for s in row["expected_sources"].split(",")]

        # Parse metadata
        metadata = {}
        for key, value in row.items():
            if key not in [
                "id",
                "question",
                "expected_answer",
                "expected_behavior",
                "test_type",
                "robustness_type",
                "difficulty",
                "category",
                "expected_sources",
            ]:
                metadata[key] = value

        return TestCase(
            id=test_id,
            question=row["question"],
            expected_answer=row.get("expected_answer"),
            expected_behavior=row.get("expected_behavior"),
            test_type=test_type,
            robustness_type=robustness_type,
            difficulty=difficulty,
            category=row.get("category", "general"),
            expected_sources=expected_sources,
            metadata=metadata if metadata else None,
        )

    def _load_json_dataset(self, file_path: str) -> List[TestCase]:
        """Load test cases from JSON file"""
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        test_cases = []

        # Handle different JSON structures
        if isinstance(data, list):
            # Direct list of test cases
            cases_data = data
        elif isinstance(data, dict) and "test_cases" in data:
            # Structured format with metadata
            cases_data = data["test_cases"]
        elif isinstance(data, dict) and "questions" in data:
            # RAG evaluator format
            cases_data = data["questions"]
        else:
            raise ValueError("Unsupported JSON structure")

        for i, case_data in enumerate(cases_data):
            try:
                test_case = self._parse_json_case(case_data, i)
                if test_case:
                    test_cases.append(test_case)
            except Exception as e:
                self.logger.warning(f"Error parsing JSON case {i}: {e}")

        self.logger.info(f"Loaded {len(test_cases)} test cases from {file_path}")
        return test_cases

    def _parse_json_case(self, case_data: Dict[str, Any], index: int) -> Optional[TestCase]:
        """Parse a single JSON case into a TestCase"""
        if not case_data.get("question"):
            self.logger.warning(f"JSON case {index}: Missing required field 'question'")
            return None

        # Generate ID if not provided
        test_id = case_data.get("id", f"json_test_{index}")

        # Parse enums with fallbacks
        test_type = TestType.QUALITY
        if case_data.get("test_type"):
            try:
                test_type = TestType(case_data["test_type"].lower())
            except ValueError:
                pass

        robustness_type = None
        if case_data.get("robustness_type"):
            try:
                robustness_type = RobustnessTestType(case_data["robustness_type"].lower())
            except ValueError:
                pass

        difficulty = Difficulty.MEDIUM
        if case_data.get("difficulty"):
            try:
                difficulty = Difficulty(case_data["difficulty"].lower())
            except ValueError:
                pass

        return TestCase(
            id=test_id,
            question=case_data["question"],
            expected_answer=case_data.get("expected_answer"),
            expected_behavior=case_data.get("expected_behavior"),
            test_type=test_type,
            robustness_type=robustness_type,
            difficulty=difficulty,
            category=case_data.get("category", "general"),
            expected_sources=case_data.get("expected_sources"),
            metadata=case_data.get("metadata"),
        )

    def save_dataset(self, dataset_name: str, test_cases: List[TestCase], format: str = "tsv"):
        """
        Save test cases to a dataset file

        Args:
            dataset_name: Name of the dataset
            test_cases: List of test cases to save
            format: File format ('tsv' or 'json')
        """
        if dataset_name not in self.config.test_datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        base_path = self.config.test_datasets[dataset_name]

        if format == "tsv":
            file_path = base_path.replace(".json", ".tsv") if base_path.endswith(".json") else base_path
            self._save_tsv_dataset(file_path, test_cases)
        elif format == "json":
            file_path = base_path.replace(".tsv", ".json") if base_path.endswith(".tsv") else base_path
            self._save_json_dataset(file_path, test_cases)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _save_tsv_dataset(self, file_path: str, test_cases: List[TestCase]):
        """Save test cases to TSV file"""
        fieldnames = [
            "id",
            "question",
            "expected_answer",
            "expected_behavior",
            "test_type",
            "robustness_type",
            "difficulty",
            "category",
            "expected_sources",
        ]

        with open(file_path, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()

            for test_case in test_cases:
                row = {
                    "id": test_case.id,
                    "question": test_case.question,
                    "expected_answer": test_case.expected_answer or "",
                    "expected_behavior": test_case.expected_behavior or "",
                    "test_type": test_case.test_type.value,
                    "robustness_type": test_case.robustness_type.value if test_case.robustness_type else "",
                    "difficulty": test_case.difficulty.value,
                    "category": test_case.category,
                    "expected_sources": ",".join(test_case.expected_sources) if test_case.expected_sources else "",
                }
                writer.writerow(row)

        self.logger.info(f"Saved {len(test_cases)} test cases to {file_path}")

    def _save_json_dataset(self, file_path: str, test_cases: List[TestCase]):
        """Save test cases to JSON file"""
        data = {
            "metadata": {
                "name": "Isschat Evaluation Dataset",
                "version": "1.0",
                "description": "Test cases for RAG evaluation",
                "total_cases": len(test_cases),
            },
            "test_cases": [
                {
                    "id": tc.id,
                    "question": tc.question,
                    "expected_answer": tc.expected_answer,
                    "expected_behavior": tc.expected_behavior,
                    "test_type": tc.test_type.value,
                    "robustness_type": tc.robustness_type.value if tc.robustness_type else None,
                    "difficulty": tc.difficulty.value,
                    "category": tc.category,
                    "expected_sources": tc.expected_sources,
                    "metadata": tc.metadata,
                }
                for tc in test_cases
            ],
        }

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved {len(test_cases)} test cases to {file_path}")

    def create_default_datasets(self):
        """Create default datasets with predefined test cases"""
        # Create robustness dataset
        from .generation_evaluator import GenerationEvaluator

        evaluator = GenerationEvaluator(self.config)
        robustness_tests = evaluator.get_predefined_robustness_tests()

        self.save_dataset("robustness", robustness_tests, "tsv")
        self.logger.info("Created default robustness dataset")

        # Create sample quality tests
        quality_tests = [
            TestCase(
                id="quality_basic_info",
                question="Qu'est-ce qu'Isschat ?",
                expected_answer="Isschat est un assistant virtuel basé sur l'IA...",
                test_type=TestType.QUALITY,
                difficulty=Difficulty.EASY,
                category="basic_info",
            ),
            TestCase(
                id="quality_technical_question",
                question="Comment configurer l'authentification dans Isschat ?",
                expected_answer="Pour configurer l'authentification...",
                test_type=TestType.QUALITY,
                difficulty=Difficulty.MEDIUM,
                category="technical",
            ),
        ]

        self.save_dataset("quality", quality_tests, "tsv")
        self.logger.info("Created default quality dataset")

    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get information about a dataset

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dict[str, Any]: Dataset information
        """
        if dataset_name not in self.config.test_datasets:
            return {"error": f"Unknown dataset: {dataset_name}"}

        dataset_path = self.config.test_datasets[dataset_name]

        if not os.path.exists(dataset_path):
            return {"name": dataset_name, "path": dataset_path, "exists": False, "total_cases": 0}

        test_cases = self.load_dataset(dataset_name)

        # Count by categories
        categories = {}
        difficulties = {}
        test_types = {}

        for tc in test_cases:
            categories[tc.category] = categories.get(tc.category, 0) + 1
            difficulties[tc.difficulty.value] = difficulties.get(tc.difficulty.value, 0) + 1
            test_types[tc.test_type.value] = test_types.get(tc.test_type.value, 0) + 1

        return {
            "name": dataset_name,
            "path": dataset_path,
            "exists": True,
            "total_cases": len(test_cases),
            "categories": categories,
            "difficulties": difficulties,
            "test_types": test_types,
            "file_size": os.path.getsize(dataset_path),
        }

    def list_all_datasets(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all configured datasets

        Returns:
            Dict[str, Dict[str, Any]]: Information about all datasets
        """
        return {name: self.get_dataset_info(name) for name in self.config.test_datasets.keys()}
