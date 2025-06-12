"""
Results Manager for RAG evaluation results storage and retrieval
"""

import sqlite3
import json
import logging
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from contextlib import contextmanager

from .models import EvaluationResult, BenchmarkResult, EvaluationSession
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.evaluation_config import EvaluationConfig, get_evaluation_config, DatabaseType


class ResultsManager:
    """
    Manages storage and retrieval of evaluation results
    """

    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        Initialize Results Manager

        Args:
            config: Evaluation configuration. If None, loads from environment.
        """
        self.config = config or get_evaluation_config()
        self.logger = logging.getLogger(__name__)

        if self.config.database.type == DatabaseType.SQLITE:
            self.db_path = self.config.database.path
            self._ensure_database()
        elif self.config.database.type == DatabaseType.MOCK:
            self.db_path = ":memory:"
            self._mock_data = {}

        self.logger.info(f"ResultsManager initialized with {self.config.database.type.value} database")

    def _ensure_database(self):
        """Ensure database file and tables exist"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        # Create tables
        with self._get_connection() as conn:
            self._create_tables(conn)

    @contextmanager
    def _get_connection(self):
        """Get database connection with context manager"""
        if self.config.database.type == DatabaseType.MOCK:
            # For mock database, we'll just yield None and handle in methods
            yield None
        else:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            try:
                yield conn
            finally:
                conn.close()

    def _create_tables(self, conn):
        """Create database tables"""
        if conn is None:  # Mock database
            return

        cursor = conn.cursor()

        # Evaluation sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_sessions (
                session_id TEXT PRIMARY KEY,
                session_name TEXT NOT NULL,
                description TEXT,
                test_types TEXT,  -- JSON array
                total_test_cases INTEGER,
                completed_test_cases INTEGER DEFAULT 0,
                start_time TEXT,
                end_time TEXT,
                status TEXT DEFAULT 'running',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Evaluation results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                test_case_id TEXT,
                question TEXT,
                response TEXT,
                sources TEXT,  -- JSON array
                generation_score TEXT,  -- JSON object
                robustness_score TEXT,  -- JSON object
                retrieval_score TEXT,   -- JSON object
                performance_metrics TEXT,  -- JSON object
                timestamp TEXT,
                evaluator_version TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES evaluation_sessions (session_id)
            )
        """)

        # Benchmark results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS benchmark_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                benchmark_name TEXT,
                benchmark_version TEXT,
                total_tests INTEGER,
                passed_tests INTEGER,
                failed_tests INTEGER,
                average_scores TEXT,  -- JSON object
                execution_time REAL,
                timestamp TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Aggregated metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS aggregated_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                metric_name TEXT,
                metric_value REAL,
                metric_type TEXT,  -- generation, robustness, retrieval, performance
                timestamp TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES evaluation_sessions (session_id)
            )
        """)

        conn.commit()
        self.logger.debug("Database tables created/verified")

    def create_session(self, session: EvaluationSession) -> str:
        """
        Create a new evaluation session

        Args:
            session: Evaluation session to create

        Returns:
            str: Session ID
        """
        if self.config.database.type == DatabaseType.MOCK:
            self._mock_data[session.session_id] = session
            return session.session_id

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO evaluation_sessions 
                (session_id, session_name, description, test_types, total_test_cases, 
                 completed_test_cases, start_time, end_time, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    session.session_id,
                    session.session_name,
                    session.description,
                    json.dumps([t.value for t in session.test_types]),
                    session.total_test_cases,
                    session.completed_test_cases,
                    session.start_time.isoformat(),
                    session.end_time.isoformat() if session.end_time else None,
                    session.status,
                ),
            )
            conn.commit()

        self.logger.info(f"Created evaluation session: {session.session_id}")
        return session.session_id

    def update_session(self, session: EvaluationSession):
        """
        Update an existing evaluation session

        Args:
            session: Updated evaluation session
        """
        if self.config.database.type == DatabaseType.MOCK:
            self._mock_data[session.session_id] = session
            return

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE evaluation_sessions 
                SET session_name = ?, description = ?, completed_test_cases = ?,
                    end_time = ?, status = ?
                WHERE session_id = ?
            """,
                (
                    session.session_name,
                    session.description,
                    session.completed_test_cases,
                    session.end_time.isoformat() if session.end_time else None,
                    session.status,
                    session.session_id,
                ),
            )
            conn.commit()

        self.logger.debug(f"Updated evaluation session: {session.session_id}")

    def get_session(self, session_id: str) -> Optional[EvaluationSession]:
        """
        Get an evaluation session by ID

        Args:
            session_id: Session ID

        Returns:
            Optional[EvaluationSession]: Session if found, None otherwise
        """
        if self.config.database.type == DatabaseType.MOCK:
            return self._mock_data.get(session_id)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM evaluation_sessions WHERE session_id = ?
            """,
                (session_id,),
            )

            row = cursor.fetchone()
            if not row:
                return None

            return self._row_to_session(row)

    def save_result(self, result: EvaluationResult, session_id: Optional[str] = None):
        """
        Save an evaluation result

        Args:
            result: Evaluation result to save
            session_id: Optional session ID to associate with
        """
        if self.config.database.type == DatabaseType.MOCK:
            # For mock, just store in memory
            key = f"{session_id}_{result.test_case_id}" if session_id else result.test_case_id
            if "results" not in self._mock_data:
                self._mock_data["results"] = {}
            self._mock_data["results"][key] = result
            return

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO evaluation_results 
                (session_id, test_case_id, question, response, sources,
                 generation_score, robustness_score, retrieval_score, 
                 performance_metrics, timestamp, evaluator_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    session_id,
                    result.test_case_id,
                    result.question,
                    result.response,
                    json.dumps(result.sources),
                    json.dumps(result.generation_score.to_dict()) if result.generation_score else None,
                    json.dumps(result.robustness_score.to_dict()) if result.robustness_score else None,
                    json.dumps(result.retrieval_score.to_dict()) if result.retrieval_score else None,
                    json.dumps(result.performance_metrics.to_dict()) if result.performance_metrics else None,
                    result.timestamp.isoformat(),
                    result.evaluator_version,
                ),
            )
            conn.commit()

        self.logger.debug(f"Saved evaluation result: {result.test_case_id}")

    def save_benchmark_result(self, benchmark: BenchmarkResult):
        """
        Save a benchmark result

        Args:
            benchmark: Benchmark result to save
        """
        if self.config.database.type == DatabaseType.MOCK:
            if "benchmarks" not in self._mock_data:
                self._mock_data["benchmarks"] = []
            self._mock_data["benchmarks"].append(benchmark)
            return

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO benchmark_results 
                (benchmark_name, benchmark_version, total_tests, passed_tests,
                 failed_tests, average_scores, execution_time, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    benchmark.benchmark_name,
                    benchmark.benchmark_version,
                    benchmark.total_tests,
                    benchmark.passed_tests,
                    benchmark.failed_tests,
                    json.dumps(benchmark.average_scores),
                    benchmark.execution_time,
                    benchmark.timestamp.isoformat(),
                ),
            )
            conn.commit()

        self.logger.info(f"Saved benchmark result: {benchmark.benchmark_name}")

    def get_session_results(self, session_id: str) -> List[EvaluationResult]:
        """
        Get all results for a session

        Args:
            session_id: Session ID

        Returns:
            List[EvaluationResult]: List of evaluation results
        """
        if self.config.database.type == DatabaseType.MOCK:
            results = []
            for key, result in self._mock_data.get("results", {}).items():
                if key.startswith(f"{session_id}_"):
                    results.append(result)
            return results

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM evaluation_results WHERE session_id = ?
                ORDER BY created_at
            """,
                (session_id,),
            )

            return [self._row_to_result(row) for row in cursor.fetchall()]

    def get_recent_sessions(self, limit: int = 10) -> List[EvaluationSession]:
        """
        Get recent evaluation sessions

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List[EvaluationSession]: List of recent sessions
        """
        if self.config.database.type == DatabaseType.MOCK:
            sessions = [v for v in self._mock_data.values() if isinstance(v, EvaluationSession)]
            return sorted(sessions, key=lambda s: s.start_time, reverse=True)[:limit]

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM evaluation_sessions 
                ORDER BY created_at DESC 
                LIMIT ?
            """,
                (limit,),
            )

            return [self._row_to_session(row) for row in cursor.fetchall()]

    def get_benchmark_history(self, benchmark_name: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get benchmark history

        Args:
            benchmark_name: Optional benchmark name filter
            limit: Maximum number of results to return

        Returns:
            List[Dict[str, Any]]: List of benchmark results
        """
        if self.config.database.type == DatabaseType.MOCK:
            benchmarks = self._mock_data.get("benchmarks", [])
            if benchmark_name:
                benchmarks = [b for b in benchmarks if b.benchmark_name == benchmark_name]
            return [b.to_dict() for b in sorted(benchmarks, key=lambda b: b.timestamp, reverse=True)[:limit]]

        with self._get_connection() as conn:
            cursor = conn.cursor()

            if benchmark_name:
                cursor.execute(
                    """
                    SELECT * FROM benchmark_results 
                    WHERE benchmark_name = ?
                    ORDER BY created_at DESC 
                    LIMIT ?
                """,
                    (benchmark_name, limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM benchmark_results 
                    ORDER BY created_at DESC 
                    LIMIT ?
                """,
                    (limit,),
                )

            results = []
            for row in cursor.fetchall():
                results.append(
                    {
                        "id": row["id"],
                        "benchmark_name": row["benchmark_name"],
                        "benchmark_version": row["benchmark_version"],
                        "total_tests": row["total_tests"],
                        "passed_tests": row["passed_tests"],
                        "failed_tests": row["failed_tests"],
                        "pass_rate": (row["passed_tests"] / row["total_tests"] * 100) if row["total_tests"] > 0 else 0,
                        "average_scores": json.loads(row["average_scores"]) if row["average_scores"] else {},
                        "execution_time": row["execution_time"],
                        "timestamp": row["timestamp"],
                    }
                )

            return results

    def _row_to_session(self, row) -> EvaluationSession:
        """Convert database row to EvaluationSession"""
        from .models import TestType

        test_types = [TestType(t) for t in json.loads(row["test_types"])]

        return EvaluationSession(
            session_id=row["session_id"],
            session_name=row["session_name"],
            description=row["description"],
            test_types=test_types,
            total_test_cases=row["total_test_cases"],
            completed_test_cases=row["completed_test_cases"],
            start_time=datetime.fromisoformat(row["start_time"]),
            end_time=datetime.fromisoformat(row["end_time"]) if row["end_time"] else None,
            status=row["status"],
        )

    def _row_to_result(self, row) -> EvaluationResult:
        """Convert database row to EvaluationResult"""
        from .models import GenerationScore, RobustnessScore, RetrievalScore, PerformanceMetrics

        # Parse JSON fields
        sources = json.loads(row["sources"]) if row["sources"] else []

        generation_score = None
        if row["generation_score"]:
            data = json.loads(row["generation_score"])
            generation_score = GenerationScore(**data)

        robustness_score = None
        if row["robustness_score"]:
            data = json.loads(row["robustness_score"])
            robustness_score = RobustnessScore(**data)

        retrieval_score = None
        if row["retrieval_score"]:
            data = json.loads(row["retrieval_score"])
            retrieval_score = RetrievalScore(**data)

        performance_metrics = None
        if row["performance_metrics"]:
            data = json.loads(row["performance_metrics"])
            performance_metrics = PerformanceMetrics(**data)

        return EvaluationResult(
            test_case_id=row["test_case_id"],
            question=row["question"],
            response=row["response"],
            sources=sources,
            generation_score=generation_score,
            robustness_score=robustness_score,
            retrieval_score=retrieval_score,
            performance_metrics=performance_metrics,
            timestamp=datetime.fromisoformat(row["timestamp"]),
            evaluator_version=row["evaluator_version"],
        )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics

        Returns:
            Dict[str, Any]: Database statistics
        """
        if self.config.database.type == DatabaseType.MOCK:
            return {
                "database_type": "mock",
                "total_sessions": len([v for v in self._mock_data.values() if isinstance(v, EvaluationSession)]),
                "total_results": len(self._mock_data.get("results", {})),
                "total_benchmarks": len(self._mock_data.get("benchmarks", [])),
            }

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Count sessions
            cursor.execute("SELECT COUNT(*) FROM evaluation_sessions")
            total_sessions = cursor.fetchone()[0]

            # Count results
            cursor.execute("SELECT COUNT(*) FROM evaluation_results")
            total_results = cursor.fetchone()[0]

            # Count benchmarks
            cursor.execute("SELECT COUNT(*) FROM benchmark_results")
            total_benchmarks = cursor.fetchone()[0]

            # Database size
            db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0

            return {
                "database_type": "sqlite",
                "database_path": self.db_path,
                "database_size_bytes": db_size,
                "total_sessions": total_sessions,
                "total_results": total_results,
                "total_benchmarks": total_benchmarks,
            }
