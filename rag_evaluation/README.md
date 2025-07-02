# Isschat Evaluation System

A streamlined evaluation system for testing Isschat's performance with automatic configuration and easy evaluator extension.

## 🚀 Quick Start

```bash
# Run all evaluations
uv run --extra evaluation rag_evaluation/run_evaluation.py

# Run specific categories
uv run --extra evaluation rag_evaluation/run_evaluation.py --categories robustness generation

# CI mode (critical tests only)
uv run --extra evaluation rag_evaluation/run_evaluation.py --ci

# Run evaluation and generate an HTML report
uv run --extra evaluation rag_evaluation/main.py --html-report
```

## 📊 Available Evaluators

### Robustness (`robustness`)
- Internal Isschat knowledge tests
- Data validation (invalid dates)
- Confidentiality handling

### Conversational (`generation`)
- Context continuity
- Language consistency (French)
- Memory recall
- Topic transitions


## 💡 Key Features

- **Auto-configuration**: System dynamically loads all configured evaluators
- **LLM Judge**: Uses Claude 4 Sonnet for response evaluation
- **CI Mode**: Critical tests only for fast feedback
- **Easy Extension**: Add evaluators in 3 simple steps
- **Full Traceability**: Timestamped results with complete details
- **Dataset Versioning**: DVC integration with Azure Blob Storage for test dataset management

## 📦 Dataset Versioning with DVC

### Installation

1. **Install evaluation dependencies**:
   ```bash
   uv sync --extra evaluation
   ```

2. **Configure environment** - Add to your `.env`:
   ```bash
   AZURE_STORAGE_ACCOUNT=your_storage_account_name
   AZURE_BLOB_CONTAINER_NAME=your_container_name
   ```

3. **Authenticate with Azure**:
   ```bash
   az login
   ```

### What we set up

1. **Initialized DVC**:
   ```bash
   dvc init
   ```

2. **Added test datasets to DVC**:
   ```bash
   dvc add --no-scm rag_evaluation/config/test_datasets
   ```

3. **Configured Azure remote**:
   ```bash
   dvc remote add -d azure-datasets azure://your-container-name
   dvc remote modify azure-datasets account_name your-storage-account
   ```

4. **Pushed to Azure**:
   ```bash
   dvc push
   ```

### Daily Usage

#### Pull latest datasets
```bash
dvc pull
```

#### Modify and push changes
```bash
# 1. Edit your test files
vim rag_evaluation/config/test_datasets/robustness_tests.json

# 2. Check what changed
dvc status

# 3. Push to Azure
dvc push

# 4. Commit the .dvc file changes
git add rag_evaluation/config/test_datasets.dvc
git commit -m "Update test datasets"
```

#### Team collaboration
```bash
# Get latest from team
git pull
dvc pull
```

### Files created
- `rag_evaluation/config/test_datasets.dvc` - DVC tracking file
- `.dvc/` - DVC configuration directory