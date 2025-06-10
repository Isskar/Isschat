"""
LLM as Judge implementation for RAG evaluation
"""

import json
import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.evaluation_config import EvaluationConfig, get_evaluation_config
from .models import GenerationScore, RobustnessScore, RobustnessTestType


class LLMJudge:
    """
    LLM as Judge implementation for evaluating RAG system responses
    """
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        Initialize LLM Judge
        
        Args:
            config: Evaluation configuration. If None, loads from environment.
        """
        self.config = config or get_evaluation_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize OpenAI client for now (will be replaced with pydantic-ai later)
        self._init_llm_client()
    
    def _init_llm_client(self):
        """Initialize LLM client"""
        try:
            import openai
            self.client = openai.OpenAI(
                api_key=self.config.llm.api_key,
                base_url=self.config.llm.api_base
            )
            self.logger.info(f"LLM Judge initialized with model: {self.config.llm.model_name}")
        except ImportError:
            self.logger.error("OpenAI package not installed. Run: pip install openai")
            self.client = None
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM client: {e}")
            self.client = None
    
    def evaluate_generation(
        self, 
        question: str, 
        context: str, 
        response: str, 
        expected_answer: Optional[str] = None
    ) -> GenerationScore:
        """
        Evaluate response generation quality using LLM as Judge
        
        Args:
            question: The original question
            context: Context/sources provided to the system
            response: Generated response to evaluate
            expected_answer: Optional expected answer for comparison
            
        Returns:
            GenerationScore: Evaluation scores and justification
        """
        if not self.client:
            self.logger.error("LLM client not initialized")
            return self._create_fallback_generation_score("LLM client not available")
        
        # Prepare expected answer section
        expected_answer_section = ""
        if expected_answer:
            expected_answer_section = f"\n**Expected Answer**: {expected_answer}"
        
        # Format prompt
        prompt = self.config.prompts.generation_evaluation_prompt.format(
            question=question,
            context=context,
            response=response,
            expected_answer_section=expected_answer_section
        )
        
        try:
            # Call LLM
            start_time = time.time()
            completion = self.client.chat.completions.create(
                model=self.config.llm.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator of RAG system responses."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
                timeout=self.config.llm.timeout
            )
            
            response_time = time.time() - start_time
            self.logger.debug(f"LLM evaluation completed in {response_time:.2f}s")
            
            # Parse response
            llm_response = completion.choices[0].message.content
            return self._parse_generation_response(llm_response)
            
        except Exception as e:
            self.logger.error(f"Error during LLM evaluation: {e}")
            return self._create_fallback_generation_score(f"Evaluation error: {str(e)}")
    
    def evaluate_robustness(
        self, 
        test_type: RobustnessTestType, 
        question: str, 
        response: str, 
        expected_behavior: str
    ) -> RobustnessScore:
        """
        Evaluate robustness test using LLM as Judge
        
        Args:
            test_type: Type of robustness test
            question: The test question
            response: System response to evaluate
            expected_behavior: Expected behavior description
            
        Returns:
            RobustnessScore: Evaluation score and justification
        """
        if not self.client:
            self.logger.error("LLM client not initialized")
            return RobustnessScore(
                score=0.0,
                passed=False,
                justification="LLM client not available",
                test_type=test_type
            )
        
        # Format prompt
        prompt = self.config.prompts.robustness_evaluation_prompt.format(
            test_type=test_type.value,
            expected_behavior=expected_behavior,
            question=question,
            response=response
        )
        
        try:
            # Call LLM
            start_time = time.time()
            completion = self.client.chat.completions.create(
                model=self.config.llm.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator of RAG system robustness."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
                timeout=self.config.llm.timeout
            )
            
            response_time = time.time() - start_time
            self.logger.debug(f"Robustness evaluation completed in {response_time:.2f}s")
            
            # Parse response
            llm_response = completion.choices[0].message.content
            return self._parse_robustness_response(llm_response, test_type)
            
        except Exception as e:
            self.logger.error(f"Error during robustness evaluation: {e}")
            return RobustnessScore(
                score=0.0,
                passed=False,
                justification=f"Evaluation error: {str(e)}",
                test_type=test_type
            )
    
    def _parse_generation_response(self, llm_response: str) -> GenerationScore:
        """Parse LLM response for generation evaluation"""
        try:
            # Try to extract JSON from response
            json_start = llm_response.find('{')
            json_end = llm_response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = llm_response[json_start:json_end]
            data = json.loads(json_str)
            
            # Validate required fields
            required_fields = ['relevance', 'accuracy', 'completeness', 'clarity', 'source_usage']
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Calculate overall score if not provided
            if 'overall_score' not in data:
                scores = [data[field] for field in required_fields]
                data['overall_score'] = sum(scores) / len(scores)
            
            return GenerationScore(
                relevance=float(data['relevance']),
                accuracy=float(data['accuracy']),
                completeness=float(data['completeness']),
                clarity=float(data['clarity']),
                source_usage=float(data['source_usage']),
                overall_score=float(data['overall_score']),
                justification=data.get('justification', 'No justification provided')
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing generation response: {e}")
            self.logger.debug(f"Raw LLM response: {llm_response}")
            return self._create_fallback_generation_score(f"Parse error: {str(e)}")
    
    def _parse_robustness_response(self, llm_response: str, test_type: RobustnessTestType) -> RobustnessScore:
        """Parse LLM response for robustness evaluation"""
        try:
            # Try to extract JSON from response
            json_start = llm_response.find('{')
            json_end = llm_response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = llm_response[json_start:json_end]
            data = json.loads(json_str)
            
            # Validate required fields
            if 'score' not in data:
                raise ValueError("Missing required field: score")
            
            score = float(data['score'])
            passed = data.get('passed', score >= self.config.thresholds['robustness_pass_threshold'])
            
            return RobustnessScore(
                score=score,
                passed=bool(passed),
                justification=data.get('justification', 'No justification provided'),
                test_type=test_type
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing robustness response: {e}")
            self.logger.debug(f"Raw LLM response: {llm_response}")
            return RobustnessScore(
                score=0.0,
                passed=False,
                justification=f"Parse error: {str(e)}",
                test_type=test_type
            )
    
    def _create_fallback_generation_score(self, error_message: str) -> GenerationScore:
        """Create fallback generation score when evaluation fails"""
        return GenerationScore(
            relevance=0.0,
            accuracy=0.0,
            completeness=0.0,
            clarity=0.0,
            source_usage=0.0,
            overall_score=0.0,
            justification=f"Evaluation failed: {error_message}"
        )
    
    def get_evaluation_stats(self) -> Dict[str, Any]:
        """Get statistics about LLM Judge usage"""
        return {
            "model_name": self.config.llm.model_name,
            "api_base": self.config.llm.api_base,
            "temperature": self.config.llm.temperature,
            "max_tokens": self.config.llm.max_tokens,
            "client_initialized": self.client is not None,
            "timestamp": datetime.now().isoformat()
        }