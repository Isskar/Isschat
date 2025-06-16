# Isschat Evaluation System

A comprehensive evaluation system for testing Isschat's performance across multiple dimensions including robustness, conversational abilities, performance timing, and feedback mechanisms.

## 🏗️ Architecture

The evaluation system is organized into modular components:

```
rag_evaluation/
├── config/                     # Configuration and test datasets
│   ├── evaluation_config.py    # Main configuration
│   └── test_datasets/          # JSON test datasets by category
├── core/                       # Core components
│   ├── base_evaluator.py       # Base classes and data structures
│   ├── isschat_client.py       # Interface to Isschat system
│   └── llm_judge.py           # LLM-based evaluation judge
├── evaluators/                 # Specialized evaluators
│   ├── robustness_evaluator.py
│   ├── conversational_evaluator.py
│   ├── performance_evaluator.py
│   └── feedback_evaluator.py
├── main.py                     # Main orchestrator
└── run_evaluation.py          # Simple runner script
```

## 🧪 Test Categories

### 1. Robustness Tests (`robustness`)
Tests model knowledge, data validation, and context handling:
- **Internal Knowledge**: "Connais-tu Isschat ?"
- **Data Validation**: "Parle moi du daily du 35 mai 2018 ?" (invalid date)
- **Fictional vs Real Entities**: Person recognition tests
- **Out of Context**: "A quoi sert l'eau ?"
- **Confidentiality**: Access to sensitive information
- **Language Consistency**: Molière tests (French responses to English questions)
- **Business Queries**: Mission assignments, benefits, project relationships

### 2. Conversational Tests (`conversational`)
Tests context continuity and multi-turn conversations:
- **Context Continuity**: "Cite moi un collaborateur" → "Cite moi l'autre"
- **Context Reference**: "Qui travaille dessus ?" (pronoun resolution)
- **Memory Recall**: "De quoi avons-nous parlé au début ?"
- **Topic Transitions**: "Revenons au sujet des collaborateurs"

### 3. Performance Tests (`performance`)
Tests response time and complexity handling:
- **Very Fast** (≤2s): Title-based searches
- **Intermediate** (≤5s): Content searches, multi-document queries
- **Complex Generation** (≤10s): LinkedIn posts, technical summaries
- **Complex Cross-reference** (≤15s): Multi-source analysis and synthesis

### 4. Feedback Tests (`feedback`)
Tests feedback system and improvement mechanisms:
- **Positive Feedback Simulation**: High-quality responses
- **Negative Feedback Simulation**: Incomplete or irrelevant responses
- **Feedback Categorization**: Pertinence, exactitude, completeness, tone
- **Improvement Mechanisms**: Dataset addition potential

## 🚀 Usage

### Basic Usage

```bash
# Run all tests
python run_evaluation.py

# Run specific categories
python main.py --categories robustness performance

# Run in CI mode (subset of critical tests)
python main.py --ci

# Save results to specific file
python main.py --output results.json
```

### Programmatic Usage

```python
from rag_evaluation.config.evaluation_config import EvaluationConfig
from rag_evaluation.main import EvaluationManager

# Create configuration
config = EvaluationConfig()
config.ci_mode = True  # For CI integration

# Run evaluation
manager = EvaluationManager(config)
results = manager.run_full_evaluation(['robustness', 'performance'])

# Check if thresholds are met
success = manager.check_thresholds()
```

## ⚙️ Configuration

Key configuration options in `EvaluationConfig`:

```python
# Success thresholds
robustness_threshold: float = 0.8
conversational_threshold: float = 0.75
performance_threshold: float = 0.85
overall_threshold: float = 0.8

# Performance timing thresholds (seconds)
fast_response_threshold: float = 2.0
medium_response_threshold: float = 5.0
slow_response_threshold: float = 10.0

# CI configuration
ci_mode: bool = False
fail_on_threshold: bool = True
ci_test_categories: List[str] = ["robustness", "performance"]
```

## 🔧 CI/CD Integration

### GitHub Actions Example

```yaml
name: Isschat Evaluation
on: [push, pull_request]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run Isschat Evaluation
        run: |
          cd Isschat/rag_evaluation
          python main.py --ci
        env:
          OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
```

### Exit Codes
- `0`: All tests passed thresholds
- `1`: Tests failed to meet thresholds or system error

## 📊 Output Format

### JSON Results Structure

```json
{
  "timestamp": "2025-01-06T14:30:00",
  "config": {
    "ci_mode": false,
    "categories_run": ["robustness", "performance"],
    "thresholds": {...}
  },
  "overall_stats": {
    "total_tests": 25,
    "total_passed": 20,
    "total_failed": 5,
    "overall_pass_rate": 0.8
  },
  "category_results": {
    "robustness": {
      "results": [...],
      "summary": {...}
    }
  }
}
```

### Console Output

```
🚀 Starting Isschat Evaluation System
============================================================

🔍 Starting robustness evaluation...
✅ Loaded 13 test cases for robustness
✅ Completed robustness evaluation: 11/13 passed

📊 EVALUATION SUMMARY
============================================================
Total Tests: 25
Passed: 20 (80.0%)
Failed: 5
Errors: 0

📋 CATEGORY BREAKDOWN:
------------------------------------------------------------
ROBUSTNESS: 11/13 (84.6%) ✅ (threshold: 80.0%)
PERFORMANCE: 9/12 (75.0%) ❌ (threshold: 85.0%)

🎯 THRESHOLD CHECK: ❌ FAILED
============================================================
```

## 🧩 Extending the System

### Adding New Test Categories

1. Create evaluator class inheriting from `BaseEvaluator`
2. Add test dataset JSON file
3. Register in `EvaluationManager`
4. Update configuration

### Adding New Test Cases

Edit the appropriate JSON file in `config/test_datasets/`:

```json
{
  "test_id": "rob_014",
  "category": "robustness",
  "test_name": "New Test",
  "question": "Your test question?",
  "expected_behavior": "Expected behavior description",
  "metadata": {
    "test_type": "custom_type",
    "language": "french",
    "difficulty": "medium"
  }
}
```

## 🔍 Evaluation Metrics

### Robustness Metrics
- Language consistency (French responses)
- Data validation (invalid date detection)
- Confidentiality handling
- Context appropriateness
- Document relevance (matching expected documents with retrieved sources)

#### Document Relevance Evaluation
- **Expected Documents**: Defined in the test case metadata, these are the documents that should be retrieved for a given query.
- **Retrieved Sources**: The actual sources retrieved by Isschat during evaluation, now capturing all sources referenced in the response.
- **Matching Logic**: The system uses URL matching to determine if retrieved sources correspond to expected documents, with support for partial matches and Confluence-specific URL patterns.
- **Scoring**: The score is based on the ratio of matched documents to expected documents. A test passes if at least one expected document is retrieved.
- **Summary Display**: The evaluation summary now includes a detailed breakdown of document relevance results for each test, showing matched versus expected documents and listing all retrieved sources.

### Conversational Metrics
- Context continuity accuracy
- Pronoun resolution success
- Memory recall capability
- Topic transition handling

### Performance Metrics
- Response time vs. complexity
- Efficiency (words per second)
- Quality vs. speed trade-offs
- Performance distribution

### Feedback Metrics
- Feedback prediction accuracy
- Response quality assessment
- Improvement potential identification

## 🛠️ Dependencies

- Python 3.8+
- langchain-openai
- langchain-core
- OpenRouter API key (for LLM judge)
- Isschat system components

## 📝 Notes

- The system uses an LLM judge (Claude 3.5 Sonnet) for response evaluation
- Rate limiting is implemented to avoid API throttling
- Conversation state is maintained for multi-turn tests
- Results are timestamped and include full traceability
- CI mode runs a subset of critical tests for faster feedback
