# Isschat Evaluation System

A comprehensive evaluation system for testing Isschat's performance with automatic configuration, interactive dashboard, and easy evaluator extension.

## Quick Start

### Interactive Dashboard
 - **View evaluation results**
 - **Compare multiple evaluations**
 - **Launch new evaluations directly from UI**

```bash
# Launch Streamlit dashboard
uv run streamlit run rag_evaluation/evaluation_dashboard.py
```
 
### Command Line Usage
```bash
# Run all evaluations
uv run --extra evaluation rag_evaluation/run_evaluation.py

# Run specific categories
uv run --extra evaluation rag_evaluation/run_evaluation.py --categories robustness generation

# CI mode (critical tests only)
uv run --extra evaluation rag_evaluation/run_evaluation.py --ci

# Save results to custom file
uv run --extra evaluation rag_evaluation/main.py --output custom_results.json
```

## Available Evaluators

### Robustness (`robustness`)
- **Internal Isschat knowledge tests**: Tests system's knowledge about itself
- **Data validation**: Handling of invalid dates and malformed inputs
- **Confidentiality handling**: Proper handling of sensitive information
- **Status**: Pass/Fail evaluation with LLM judge

### Generation (`generation`)
- **Context continuity**: Maintaining conversation context across turns
- **Language consistency**: Consistent French language usage
- **Memory recall**: Remembering previous conversation elements
- **Topic transitions**: Smooth handling of topic changes
- **Status**: Pass/Fail evaluation with LLM judge

### Retrieval Performance (`retrieval`)
- **Document retrieval accuracy**: Precision and recall of relevant documents
- **Ranking quality**: NDCG, MAP, MRR metrics for result ranking
- **Response time**: Retrieval latency measurement
- **Status**: Measured with detailed metrics (P@1, P@3, P@5, F1-Score)

### Business Value (`business_value`)
- **Business impact measurement**: ROI and efficiency metrics
- **User productivity**: Time-saving and task completion rates
- **Cost-effectiveness**: Resource utilization and optimization
- **Status**: Pass/Fail evaluation with LLM judge

### User Feedback Analysis (`feedback`)
- **CamemBERT classification**:Topic classification of user feedback
- **Sentiment analysis**: Positive/negative feedback identification
- **Satisfaction metrics**: Overall and per-topic satisfaction rates
- **Actionable insights**: Automated identification of strengths and weaknesses
- **Status**: Measured with satisfaction percentages and topic breakdown

## Key Features

- **Auto-configuration**: System dynamically loads all configured evaluators
- **LLM Judge**: Uses Claude 4 Sonnet for response evaluation
- **CI Mode**: Critical tests only for fast feedback
- **Interactive Dashboard**: Streamlit-based UI for evaluation management
- **Real-time Feedback Analysis**: CamemBERT-powered user feedback insights
- **Easy Extension**: Add evaluators in 3 simple steps
- **Full Traceability**: Timestamped results with complete details
- **Dataset Versioning**: DVC integration with Azure Blob Storage for test dataset management

## Dataset Versioning with DVC

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
   dvc add rag_evaluation/config/test_datasets
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