# Isschat Evaluation System

A streamlined evaluation system for testing Isschat's performance with automatic configuration and easy evaluator extension.

## 🚀 Quick Start

```bash
# Run all evaluations
uv run rag_evaluation/run_evaluation.py

# Run specific categories
uv run rag_evaluation/main.py --categories robustness conversational

# CI mode (critical tests only)
uv run rag_evaluation/main.py --ci
```

## ⚙️ Configuration

The system auto-configures via [`config/evaluators.json`](config/evaluators.json) and [`config/evaluation_config.py`](config/evaluation_config.py).

### Main Configuration

```python
# In evaluation_config.py
ci_threshold: float = 0.7          # Success threshold for CI mode
judge_model: str = "anthropic/claude-sonnet-4"  # LLM judge model
ci_mode: bool = False              # Enable/disable CI mode
```

### Environment Variables

```bash
export OPENROUTER_API_KEY="your-api-key"  # Required for LLM judge
```

## 🧩 Adding New Evaluators

The system has been greatly simplified for adding evaluators. Follow these 3 steps:

### 1. Create the Evaluator

Create a file in [`evaluators/`](evaluators/):

```python
# evaluators/my_evaluator.py
from core.base_evaluator import BaseEvaluator, TestResult

class MyEvaluator(BaseEvaluator):
    """Description of my evaluator"""
    
    def evaluate_response(self, test_case, response, response_time=None):
        # Custom evaluation logic
        success = self._check_my_criteria(response, test_case)
        
        return TestResult(
            test_id=test_case.test_id,
            success=success,
            score=1.0 if success else 0.0,
            details={"custom_metric": "value"}
        )
```

### 2. Create Test Dataset

Create a JSON file in [`config/test_datasets/`](config/test_datasets/):

```json
[
  {
    "test_id": "my_001",
    "category": "my_evaluator",
    "test_name": "Test Example",
    "question": "My test question?",
    "expected_behavior": "Expected behavior description",
    "metadata": {
      "test_type": "custom",
      "difficulty": "medium"
    }
  }
]
```

### 3. Register in Configuration

Add your evaluator to [`config/evaluators.json`](config/evaluators.json):

```json
{
  "my_evaluator": {
    "class_name": "MyEvaluator",
    "module": "rag_evaluation.evaluators.my_evaluator",
    "name": "My Custom Evaluator",
    "description": "Description of what this evaluator tests",
    "dataset": "config/test_datasets/my_dataset.json",
    "weight": 1.0,
    "enabled": true
  }
}
```

**That's it!** The system will automatically detect and load your evaluator.

## 📊 Available Evaluators

### Robustness (`robustness`)
- Internal Isschat knowledge tests
- Data validation (invalid dates)
- Confidentiality handling
- Language consistency (French)

### Conversational (`conversational`)
- Context continuity
- Pronoun resolution
- Memory recall
- Topic transitions

## 📈 Results Format

Results are automatically saved in [`evaluation_results/`](../evaluation_results/) with timestamps:

```json
{
  "timestamp": "2025-06-18T15:02:02",
  "overall_stats": {
    "total_tests": 15,
    "total_passed": 12,
    "overall_pass_rate": 0.8
  },
  "category_results": {
    "robustness": {
      "summary": {
        "total_tests": 8,
        "passed": 7,
        "pass_rate": 0.875
      }
    }
  }
}
```

## 🛠️ Dependencies

- langchain-openai, langchain-core
- OpenRouter API key (for LLM judge)
- Isschat system components

## 💡 Key Features

- **Auto-configuration**: System dynamically loads all configured evaluators
- **LLM Judge**: Uses Claude 4 Sonnet for response evaluation
- **CI Mode**: Critical tests only for fast feedback
- **Easy Extension**: Add evaluators in 3 simple steps
- **Full Traceability**: Timestamped results with complete details