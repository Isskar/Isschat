# Isschat Evaluation System

A streamlined evaluation system for testing Isschat's performance with automatic configuration and easy evaluator extension.

## 🚀 Quick Start

```bash
# Run all evaluations
uv run rag_evaluation/run_evaluation.py

# Run specific categories
uv run rag_evaluation/run_evaluation.py --categories robustness generation

# CI mode (critical tests only)
uv run rag_evaluation/run_evaluation.py --ci
```

## 📊 Available Evaluators

### Robustness (`robustness`)
- Internal Isschat knowledge tests
- Data validation (invalid dates)
- Confidentiality handling
- Language consistency (French)

### Conversational (`generation`)
- Context continuity
- Pronoun resolution
- Memory recall
- Topic transitions


## 💡 Key Features

- **Auto-configuration**: System dynamically loads all configured evaluators
- **LLM Judge**: Uses Claude 4 Sonnet for response evaluation
- **CI Mode**: Critical tests only for fast feedback
- **Easy Extension**: Add evaluators in 3 simple steps
- **Full Traceability**: Timestamped results with complete details