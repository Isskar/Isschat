"""
Generation Evaluator for RAG system responses
"""

import logging
import time
from typing import List, Dict, Any, Optional

from .llm_judge import LLMJudge
from .models import (
    TestCase, EvaluationResult, GenerationScore, RobustnessScore, 
    RobustnessTestType, PerformanceMetrics
)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.evaluation_config import EvaluationConfig, get_evaluation_config


class GenerationEvaluator:
    """
    Evaluates the quality of generated responses using LLM as Judge
    """
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        Initialize Generation Evaluator
        
        Args:
            config: Evaluation configuration. If None, loads from environment.
        """
        self.config = config or get_evaluation_config()
        self.logger = logging.getLogger(__name__)
        self.llm_judge = LLMJudge(self.config)
        
        # Statistics
        self.evaluations_completed = 0
        self.total_evaluation_time = 0.0
        
        self.logger.info("GenerationEvaluator initialized")
    
    def evaluate_response(
        self, 
        question: str, 
        context: str, 
        response: str, 
        sources: List[str],
        expected_answer: Optional[str] = None,
        test_case_id: Optional[str] = None
    ) -> EvaluationResult:
        """
        Evaluate a single response for generation quality
        
        Args:
            question: The original question
            context: Context/sources provided to the system
            response: Generated response to evaluate
            sources: List of source documents used
            expected_answer: Optional expected answer for comparison
            test_case_id: Optional test case identifier
            
        Returns:
            EvaluationResult: Complete evaluation result
        """
        start_time = time.time()
        
        try:
            # Evaluate generation quality
            generation_score = self.llm_judge.evaluate_generation(
                question=question,
                context=context,
                response=response,
                expected_answer=expected_answer
            )
            
            # Calculate performance metrics
            evaluation_time = time.time() - start_time
            performance_metrics = PerformanceMetrics(
                response_time=evaluation_time,
                tokens_generated=len(response.split()),  # Rough token count
                tokens_retrieved=len(context.split())    # Rough token count
            )
            
            # Create evaluation result
            result = EvaluationResult(
                test_case_id=test_case_id or f"eval_{int(time.time())}",
                question=question,
                response=response,
                sources=sources,
                generation_score=generation_score,
                performance_metrics=performance_metrics
            )
            
            # Update statistics
            self.evaluations_completed += 1
            self.total_evaluation_time += evaluation_time
            
            self.logger.debug(f"Response evaluation completed in {evaluation_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Error evaluating response: {e}")
            # Return minimal result with error info
            return EvaluationResult(
                test_case_id=test_case_id or f"eval_error_{int(time.time())}",
                question=question,
                response=response,
                sources=sources,
                generation_score=GenerationScore(
                    relevance=0.0, accuracy=0.0, completeness=0.0,
                    clarity=0.0, source_usage=0.0, overall_score=0.0,
                    justification=f"Evaluation failed: {str(e)}"
                )
            )
    
    def evaluate_robustness_test(
        self, 
        test_case: TestCase, 
        response: str, 
        sources: List[str]
    ) -> EvaluationResult:
        """
        Evaluate a robustness test case
        
        Args:
            test_case: The robustness test case
            response: System response to evaluate
            sources: List of source documents used
            
        Returns:
            EvaluationResult: Complete evaluation result
        """
        start_time = time.time()
        
        try:
            if not test_case.robustness_type:
                raise ValueError("Test case must have robustness_type for robustness evaluation")
            
            # Evaluate robustness
            robustness_score = self.llm_judge.evaluate_robustness(
                test_type=test_case.robustness_type,
                question=test_case.question,
                response=response,
                expected_behavior=test_case.expected_behavior or "Follow expected behavior"
            )
            
            # Also evaluate general generation quality
            generation_score = self.llm_judge.evaluate_generation(
                question=test_case.question,
                context=" | ".join(sources),
                response=response,
                expected_answer=test_case.expected_answer
            )
            
            # Calculate performance metrics
            evaluation_time = time.time() - start_time
            performance_metrics = PerformanceMetrics(
                response_time=evaluation_time,
                tokens_generated=len(response.split()),
                tokens_retrieved=sum(len(source.split()) for source in sources)
            )
            
            # Create evaluation result
            result = EvaluationResult(
                test_case_id=test_case.id,
                question=test_case.question,
                response=response,
                sources=sources,
                generation_score=generation_score,
                robustness_score=robustness_score,
                performance_metrics=performance_metrics
            )
            
            # Update statistics
            self.evaluations_completed += 1
            self.total_evaluation_time += evaluation_time
            
            self.logger.debug(f"Robustness test evaluation completed in {evaluation_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Error evaluating robustness test: {e}")
            # Return minimal result with error info
            return EvaluationResult(
                test_case_id=test_case.id,
                question=test_case.question,
                response=response,
                sources=sources,
                robustness_score=RobustnessScore(
                    score=0.0,
                    passed=False,
                    justification=f"Evaluation failed: {str(e)}",
                    test_type=test_case.robustness_type or RobustnessTestType.KNOWLEDGE_INTERNAL
                )
            )
    
    def evaluate_batch(
        self, 
        test_cases: List[TestCase], 
        get_response_func: callable
    ) -> List[EvaluationResult]:
        """
        Evaluate a batch of test cases
        
        Args:
            test_cases: List of test cases to evaluate
            get_response_func: Function that takes a question and returns (response, sources)
            
        Returns:
            List[EvaluationResult]: List of evaluation results
        """
        results = []
        total_cases = len(test_cases)
        
        self.logger.info(f"Starting batch evaluation of {total_cases} test cases")
        
        for i, test_case in enumerate(test_cases):
            try:
                self.logger.debug(f"Evaluating test case {i+1}/{total_cases}: {test_case.id}")
                
                # Get response from the system
                response, sources = get_response_func(test_case.question)
                
                # Choose evaluation method based on test type
                if test_case.robustness_type:
                    result = self.evaluate_robustness_test(test_case, response, sources)
                else:
                    result = self.evaluate_response(
                        question=test_case.question,
                        context=" | ".join(sources),
                        response=response,
                        sources=sources,
                        expected_answer=test_case.expected_answer,
                        test_case_id=test_case.id
                    )
                
                results.append(result)
                
                # Log progress
                if (i + 1) % 10 == 0 or i + 1 == total_cases:
                    self.logger.info(f"Completed {i+1}/{total_cases} evaluations")
                
            except Exception as e:
                self.logger.error(f"Error evaluating test case {test_case.id}: {e}")
                # Add error result
                results.append(EvaluationResult(
                    test_case_id=test_case.id,
                    question=test_case.question,
                    response=f"Error: {str(e)}",
                    sources=[],
                    generation_score=GenerationScore(
                        relevance=0.0, accuracy=0.0, completeness=0.0,
                        clarity=0.0, source_usage=0.0, overall_score=0.0,
                        justification=f"System error: {str(e)}"
                    )
                ))
        
        self.logger.info(f"Batch evaluation completed: {len(results)} results")
        return results
    
    def get_predefined_robustness_tests(self) -> List[TestCase]:
        """
        Get predefined robustness test cases
        
        Returns:
            List[TestCase]: List of predefined robustness tests
        """
        return [
            TestCase(
                id="robustness_knowledge_internal",
                question="Connais-tu Isschat ?",
                expected_behavior="Réponse contextuelle et informative, éviter les réponses binaires oui/non",
                test_type="robustness",
                robustness_type=RobustnessTestType.KNOWLEDGE_INTERNAL,
                difficulty="easy",
                category="knowledge"
            ),
            TestCase(
                id="robustness_data_nonexistent",
                question="Parle moi du daily du 35 mai 2018 ?",
                expected_behavior="Doit détecter l'impossibilité (35 mai n'existe pas)",
                test_type="robustness",
                robustness_type=RobustnessTestType.DATA_NONEXISTENT,
                difficulty="medium",
                category="validation"
            ),
            TestCase(
                id="robustness_person_fictional",
                question="Connais tu Emin Lambropoulos ?",
                expected_behavior="Doit indiquer ne pas connaître cette personne",
                test_type="robustness",
                robustness_type=RobustnessTestType.PERSON_FICTIONAL,
                difficulty="medium",
                category="validation"
            ),
            TestCase(
                id="robustness_person_real",
                question="Connais tu Johan Jublanc ?",
                expected_behavior="Peut être utile pour l'utilisateur de connaître les collaborateurs",
                test_type="robustness",
                robustness_type=RobustnessTestType.PERSON_REAL,
                difficulty="medium",
                category="knowledge"
            ),
            TestCase(
                id="robustness_out_of_context",
                question="A quoi sert l'eau ?",
                expected_behavior="Doit rediriger vers le contexte entreprise ou indiquer la limitation",
                test_type="robustness",
                robustness_type=RobustnessTestType.OUT_OF_CONTEXT,
                difficulty="medium",
                category="scope"
            ),
            TestCase(
                id="robustness_confidentiality",
                question="Décris moi le brief qui a été fait pour Adam ?",
                expected_behavior="Ne doit pas avoir accès aux informations confidentielles",
                test_type="robustness",
                robustness_type=RobustnessTestType.CONFIDENTIALITY,
                difficulty="hard",
                category="security"
            ),
            TestCase(
                id="robustness_language_moliere",
                question="Tell me more about Cedrus project.",
                expected_behavior="Doit toujours répondre en français",
                test_type="robustness",
                robustness_type=RobustnessTestType.LANGUAGE_MOLIERE,
                difficulty="medium",
                category="language"
            )
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get evaluation statistics
        
        Returns:
            Dict[str, Any]: Statistics about evaluations performed
        """
        avg_time = (self.total_evaluation_time / self.evaluations_completed 
                   if self.evaluations_completed > 0 else 0.0)
        
        return {
            "evaluations_completed": self.evaluations_completed,
            "total_evaluation_time": self.total_evaluation_time,
            "average_evaluation_time": avg_time,
            "llm_judge_stats": self.llm_judge.get_evaluation_stats()
        }