"""
Main Evaluation Manager for RAG system evaluation
"""

import logging
import time
import uuid
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime

from .generation_evaluator import GenerationEvaluator
from .dataset_manager import DatasetManager
from .results_manager import ResultsManager
from .models import (
    TestCase, EvaluationResult, EvaluationSession, BenchmarkResult,
    TestType, Difficulty
)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.evaluation_config import EvaluationConfig, get_evaluation_config


class EvaluationManager:
    """
    Main manager for RAG evaluation system
    Orchestrates all evaluation components
    """
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        Initialize Evaluation Manager
        
        Args:
            config: Evaluation configuration. If None, loads from environment.
        """
        self.config = config or get_evaluation_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.generation_evaluator = GenerationEvaluator(self.config)
        self.dataset_manager = DatasetManager(self.config)
        self.results_manager = ResultsManager(self.config)
        
        # Current session
        self.current_session: Optional[EvaluationSession] = None
        
        self.logger.info("EvaluationManager initialized")
    
    def create_evaluation_session(
        self, 
        session_name: str, 
        description: str, 
        test_types: List[TestType]
    ) -> str:
        """
        Create a new evaluation session
        
        Args:
            session_name: Name of the evaluation session
            description: Description of what's being evaluated
            test_types: Types of tests to include
            
        Returns:
            str: Session ID
        """
        session_id = f"eval_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        # Load test cases for specified types
        all_test_cases = []
        for test_type in test_types:
            if test_type.value in self.config.test_datasets:
                test_cases = self.dataset_manager.load_dataset(test_type.value)
                all_test_cases.extend(test_cases)
        
        # Create session
        session = EvaluationSession(
            session_id=session_id,
            session_name=session_name,
            description=description,
            test_types=test_types,
            total_test_cases=len(all_test_cases)
        )
        
        # Save session
        self.results_manager.create_session(session)
        self.current_session = session
        
        self.logger.info(f"Created evaluation session '{session_name}' with {len(all_test_cases)} test cases")
        return session_id
    
    def run_evaluation(
        self, 
        session_id: str, 
        get_response_func: Callable[[str], tuple[str, List[str]]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Run evaluation for a session
        
        Args:
            session_id: Session ID to run evaluation for
            get_response_func: Function that takes a question and returns (response, sources)
            progress_callback: Optional callback for progress updates (current, total)
            
        Returns:
            Dict[str, Any]: Evaluation summary
        """
        # Get session
        session = self.results_manager.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")
        
        self.current_session = session
        
        # Load test cases for this session
        all_test_cases = []
        for test_type in session.test_types:
            if test_type.value in self.config.test_datasets:
                test_cases = self.dataset_manager.load_dataset(test_type.value)
                all_test_cases.extend(test_cases)
        
        self.logger.info(f"Starting evaluation for session '{session.session_name}' with {len(all_test_cases)} test cases")
        
        # Update session status
        session.status = "running"
        self.results_manager.update_session(session)
        
        # Run evaluations
        results = []
        start_time = time.time()
        
        try:
            for i, test_case in enumerate(all_test_cases):
                try:
                    self.logger.debug(f"Evaluating test case {i+1}/{len(all_test_cases)}: {test_case.id}")
                    
                    # Get response from the system
                    response, sources = get_response_func(test_case.question)
                    
                    # Evaluate based on test type
                    if test_case.robustness_type:
                        result = self.generation_evaluator.evaluate_robustness_test(
                            test_case, response, sources
                        )
                    else:
                        result = self.generation_evaluator.evaluate_response(
                            question=test_case.question,
                            context=" | ".join(sources),
                            response=response,
                            sources=sources,
                            expected_answer=test_case.expected_answer,
                            test_case_id=test_case.id
                        )
                    
                    # Save result
                    self.results_manager.save_result(result, session_id)
                    results.append(result)
                    
                    # Update progress
                    session.completed_test_cases = i + 1
                    self.results_manager.update_session(session)
                    
                    if progress_callback:
                        progress_callback(i + 1, len(all_test_cases))
                    
                    # Log progress periodically
                    if (i + 1) % 10 == 0 or i + 1 == len(all_test_cases):
                        self.logger.info(f"Completed {i+1}/{len(all_test_cases)} evaluations")
                
                except Exception as e:
                    self.logger.error(f"Error evaluating test case {test_case.id}: {e}")
                    # Continue with next test case
                    continue
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Update session as completed
            session.end_time = datetime.now()
            session.status = "completed"
            self.results_manager.update_session(session)
            
            # Calculate summary statistics
            summary = self._calculate_evaluation_summary(results, execution_time)
            
            self.logger.info(f"Evaluation completed in {execution_time:.2f}s")
            return summary
            
        except Exception as e:
            # Mark session as failed
            session.status = "failed"
            session.end_time = datetime.now()
            self.results_manager.update_session(session)
            
            self.logger.error(f"Evaluation failed: {e}")
            raise
    
    def run_robustness_tests(
        self, 
        get_response_func: Callable[[str], tuple[str, List[str]]],
        session_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run predefined robustness tests
        
        Args:
            get_response_func: Function that takes a question and returns (response, sources)
            session_name: Optional session name
            
        Returns:
            Dict[str, Any]: Robustness test results
        """
        # Create session for robustness tests
        session_id = self.create_evaluation_session(
            session_name=session_name or f"Robustness Tests {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            description="Predefined robustness tests for RAG system",
            test_types=[TestType.ROBUSTNESS]
        )
        
        # Run evaluation
        return self.run_evaluation(session_id, get_response_func)
    
    def run_quick_evaluation(
        self, 
        questions: List[str], 
        get_response_func: Callable[[str], tuple[str, List[str]]],
        session_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run a quick evaluation with custom questions
        
        Args:
            questions: List of questions to evaluate
            get_response_func: Function that takes a question and returns (response, sources)
            session_name: Optional session name
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        # Create test cases from questions
        test_cases = [
            TestCase(
                id=f"quick_test_{i}",
                question=question,
                test_type=TestType.QUALITY,
                difficulty=Difficulty.MEDIUM,
                category="quick_eval"
            )
            for i, question in enumerate(questions)
        ]
        
        # Create temporary session
        session_id = f"quick_eval_{int(time.time())}"
        session = EvaluationSession(
            session_id=session_id,
            session_name=session_name or f"Quick Evaluation {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            description="Quick evaluation with custom questions",
            test_types=[TestType.QUALITY],
            total_test_cases=len(test_cases)
        )
        
        self.results_manager.create_session(session)
        
        # Run evaluations
        results = []
        start_time = time.time()
        
        for i, test_case in enumerate(test_cases):
            try:
                # Get response
                response, sources = get_response_func(test_case.question)
                
                # Evaluate
                result = self.generation_evaluator.evaluate_response(
                    question=test_case.question,
                    context=" | ".join(sources),
                    response=response,
                    sources=sources,
                    test_case_id=test_case.id
                )
                
                # Save result
                self.results_manager.save_result(result, session_id)
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error in quick evaluation for question {i}: {e}")
        
        execution_time = time.time() - start_time
        
        # Update session
        session.completed_test_cases = len(results)
        session.end_time = datetime.now()
        session.status = "completed"
        self.results_manager.update_session(session)
        
        return self._calculate_evaluation_summary(results, execution_time)
    
    def create_benchmark(
        self, 
        benchmark_name: str, 
        benchmark_version: str, 
        test_cases: List[TestCase],
        get_response_func: Callable[[str], tuple[str, List[str]]]
    ) -> BenchmarkResult:
        """
        Create and run a benchmark
        
        Args:
            benchmark_name: Name of the benchmark
            benchmark_version: Version of the benchmark
            test_cases: Test cases to include in benchmark
            get_response_func: Function that takes a question and returns (response, sources)
            
        Returns:
            BenchmarkResult: Benchmark results
        """
        self.logger.info(f"Running benchmark '{benchmark_name}' v{benchmark_version} with {len(test_cases)} test cases")
        
        start_time = time.time()
        results = []
        
        # Run evaluations
        for test_case in test_cases:
            try:
                # Get response
                response, sources = get_response_func(test_case.question)
                
                # Evaluate
                if test_case.robustness_type:
                    result = self.generation_evaluator.evaluate_robustness_test(
                        test_case, response, sources
                    )
                else:
                    result = self.generation_evaluator.evaluate_response(
                        question=test_case.question,
                        context=" | ".join(sources),
                        response=response,
                        sources=sources,
                        expected_answer=test_case.expected_answer,
                        test_case_id=test_case.id
                    )
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error in benchmark for test case {test_case.id}: {e}")
        
        execution_time = time.time() - start_time
        
        # Calculate benchmark metrics
        total_tests = len(results)
        passed_tests = 0
        average_scores = {}
        
        # Count passed tests and calculate averages
        generation_scores = []
        robustness_scores = []
        
        for result in results:
            if result.generation_score:
                generation_scores.append(result.generation_score.overall_score)
                if result.generation_score.overall_score >= self.config.thresholds.get('min_accuracy', 7.0):
                    passed_tests += 1
            
            if result.robustness_score:
                robustness_scores.append(result.robustness_score.score)
                if result.robustness_score.passed:
                    passed_tests += 1
        
        if generation_scores:
            average_scores['generation_overall'] = sum(generation_scores) / len(generation_scores)
        
        if robustness_scores:
            average_scores['robustness_overall'] = sum(robustness_scores) / len(robustness_scores)
        
        # Create benchmark result
        benchmark_result = BenchmarkResult(
            benchmark_name=benchmark_name,
            benchmark_version=benchmark_version,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=total_tests - passed_tests,
            average_scores=average_scores,
            individual_results=results,
            execution_time=execution_time
        )
        
        # Save benchmark result
        self.results_manager.save_benchmark_result(benchmark_result)
        
        self.logger.info(f"Benchmark completed: {passed_tests}/{total_tests} tests passed ({benchmark_result.pass_rate:.1f}%)")
        return benchmark_result
    
    def _calculate_evaluation_summary(self, results: List[EvaluationResult], execution_time: float) -> Dict[str, Any]:
        """Calculate summary statistics for evaluation results"""
        if not results:
            return {
                "total_tests": 0,
                "execution_time": execution_time,
                "error": "No results to summarize"
            }
        
        # Basic counts
        total_tests = len(results)
        generation_results = [r for r in results if r.generation_score]
        robustness_results = [r for r in results if r.robustness_score]
        
        summary = {
            "total_tests": total_tests,
            "execution_time": execution_time,
            "generation_tests": len(generation_results),
            "robustness_tests": len(robustness_results)
        }
        
        # Generation statistics
        if generation_results:
            scores = [r.generation_score for r in generation_results]
            summary["generation_stats"] = {
                "average_relevance": sum(s.relevance for s in scores) / len(scores),
                "average_accuracy": sum(s.accuracy for s in scores) / len(scores),
                "average_completeness": sum(s.completeness for s in scores) / len(scores),
                "average_clarity": sum(s.clarity for s in scores) / len(scores),
                "average_source_usage": sum(s.source_usage for s in scores) / len(scores),
                "average_overall": sum(s.overall_score for s in scores) / len(scores)
            }
        
        # Robustness statistics
        if robustness_results:
            robustness_scores = [r.robustness_score for r in robustness_results]
            passed_count = sum(1 for s in robustness_scores if s.passed)
            
            summary["robustness_stats"] = {
                "total_robustness_tests": len(robustness_scores),
                "passed_tests": passed_count,
                "failed_tests": len(robustness_scores) - passed_count,
                "pass_rate": (passed_count / len(robustness_scores) * 100) if robustness_scores else 0,
                "average_score": sum(s.score for s in robustness_scores) / len(robustness_scores)
            }
        
        return summary
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get summary for a specific session
        
        Args:
            session_id: Session ID
            
        Returns:
            Dict[str, Any]: Session summary
        """
        session = self.results_manager.get_session(session_id)
        if not session:
            return {"error": f"Session not found: {session_id}"}
        
        results = self.results_manager.get_session_results(session_id)
        
        execution_time = session.duration or 0.0
        summary = self._calculate_evaluation_summary(results, execution_time)
        
        # Add session info
        summary.update({
            "session_id": session.session_id,
            "session_name": session.session_name,
            "description": session.description,
            "status": session.status,
            "start_time": session.start_time.isoformat(),
            "end_time": session.end_time.isoformat() if session.end_time else None,
            "progress_percentage": session.progress_percentage
        })
        
        return summary
    
    def list_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        List recent evaluation sessions
        
        Args:
            limit: Maximum number of sessions to return
            
        Returns:
            List[Dict[str, Any]]: List of session summaries
        """
        sessions = self.results_manager.get_recent_sessions(limit)
        return [session.to_dict() for session in sessions]
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """
        Get overall system statistics
        
        Returns:
            Dict[str, Any]: System statistics
        """
        return {
            "config": {
                "llm_model": self.config.llm.model_name,
                "database_type": self.config.database.type.value,
                "thresholds": self.config.thresholds
            },
            "components": {
                "generation_evaluator": self.generation_evaluator.get_statistics(),
                "dataset_manager": self.dataset_manager.list_all_datasets(),
                "results_manager": self.results_manager.get_statistics()
            },
            "current_session": self.current_session.to_dict() if self.current_session else None
        }