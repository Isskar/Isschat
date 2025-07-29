# Isschat - Enterprise RAG Chatbot

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Isskar/Isschat/blob/main/LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)

A chatbot that provides semantic search and conversational AI capabilities for Confluence knowledge bases using advanced RAG (Retrieval-Augmented Generation) technology with semantic understanding.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Launch](#launch)
  - [Web Interface](#web-interface)
  - [Command Line Interface](#command-line-interface-cli)
  - [Evaluation System](#evaluation-system)
- [Architecture](#architecture)
- [Production Deployment](#production-deployment)
  - [Azure Cloud Deployment](#azure-cloud-deployment)
  - [Docker Deployment](#docker-deployment)
  - [Local Development](#local-development)
  - [Testing](#testing)
- [License](#license)

## Features

### Core RAG Capabilities
- **Semantic RAG Pipeline**: Advanced semantic understanding with query reformulation and coreference resolution
- **Dual Pipeline Architecture**: Both standard and semantic-enhanced RAG processing with intelligent fallback
- **Intent Classification**: Automatically detects and routes queries based on intent (team_info, project_info, technical_info, feature_info)
- **Query Reformulation**: LLM-based coreference resolution using conversation context for ambiguous queries
- **Intelligent Query Processing**: Handles misleading keywords and provides contextually accurate responses
- **Vector Retrieval**: Direct vector search with semantic re-ranking based on similarity and intent matching
- **Multilingual Support**: Optimized for French and English content with cross-language understanding and synonym handling
- **Contextual Understanding**: Maintains document hierarchy awareness for numerical and structured queries

### Enterprise Operations
- **Azure AD Authentication**: Secure OAuth 2.0 integration with domain validation
- **Performance Monitoring**: Real-time response time, accuracy metrics, and system health tracking
- **Conversation History**: Persistent chat history with search capabilities and conversation context
- **User Feedback System**: Integrated feedback collection with sentiment analysis and analytics
- **Admin Dashboard**: System management with performance insights, user analytics, and configuration interface

### Quality Assurance & Evaluation
- **Comprehensive Evaluation Framework**: Multi-category testing (retrieval, generation, business value, robustness)
- **LLM-based Automated Evaluation**: Automated quality assessment using advanced language models with confidence metrics
- **CI/CD Integration**: Automated testing pipeline with configurable quality thresholds
- **Performance Benchmarking**: Continuous performance monitoring and improvement tracking
- **Adaptive Chunking**: Content-type aware document processing with hierarchical chunking
- **Flexible Vector Storage**: Weaviate cloud integration with automated data pipeline and batch optimization

## Installation

### Prerequisites
- Python 3.12+
- Install uv package manager:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

### Configuration

1. **Clone the repository**
   ```bash
   git clone https://github.com/Isskar/Isschat.git
   cd Isschat
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

3. **Configure environment variables**
   Copy `.env.example` to `.env` file at root with:
   
   ```bash
   # Required - Confluence Configuration
   CONFLUENCE_API_KEY=your_confluence_api_key
   CONFLUENCE_SPACE_NAME=https://your_company.atlassian.net
   CONFLUENCE_EMAIL_ADDRESS=firstname.lastname@your_company.com
   
   # Required - LLM Configuration
   OPENROUTER_API_KEY=your_openrouter_api_key
   
   # Optional - Advanced Configuration
   LLM_MODEL=google/gemini-2.5-flash-lite-preview-06-17
   EMBEDDINGS_MODEL=intfloat/multilingual-e5-small
   CHUNK_SIZE=1000
   SEARCH_K=3
   
   # Optional - Semantic Features
   USE_SEMANTIC_FEATURES=true
   SEMANTIC_RERANKING_ENABLED=true
   
   # Optional - Azure Integration (for production)
   USE_AZURE_STORAGE=false
   AZURE_STORAGE_ACCOUNT=your_storage_account
   AZURE_BLOB_CONTAINER_NAME=your_container
   KEY_VAULT_URL=https://your-keyvault.vault.azure.net/
   ```
   
   **Get your API keys from:**
   - Confluence API: https://id.atlassian.com/manage-profile/security/api-tokens
   - OpenRouter API: https://openrouter.ai/


## Launch

### Web Interface

1. **Install dependencies**
   ```bash
   uv sync
   ```

2. **Launch Streamlit app**
   ```bash
   uv run streamlit run src/webapp/app.py
   ```

3. **Reconstruct the database**

   Click on the button "Rebuild from Confluence"

4. **Launch the chatbot**

   Ask your question to the chatbot

### Command Line Interface (CLI)

Isschat provides a powerful CLI tool for managing and querying your knowledge base:

#### Available Commands

- **Status Check**: Check system components and configuration
  ```bash
  uv run -m src.cli.main status [--verbose] [--component config|ingestion|rag|all]
  ```

- **Data Ingestion**: Build or update the vector database from Confluence
  ```bash
  uv run -m src.cli.main ingest [--source confluence] [--force-rebuild] [--verbose]
  ```

- **Interactive Chat**: Start a chat session without the web interface
  ```bash
  uv run -m src.cli.main chat [--user-id cli_user]
  ```

- **Direct Query**: Query the vector database with detailed results
  ```bash
  uv run -m src.cli.main query -q "your question" [options]
  ```

#### Query Command Options

- `-q, --query`: Your search query (required)
- `-k, --top-k`: Number of chunks to retrieve (default: 5)
- `-s, --score-threshold`: Minimum similarity score (default: 0.0)
- `-v, --verbose`: Show detailed chunk information
- `--show-metadata`: Display document metadata
- `--show-content`: Display chunk content (default: true)
- `--show-stats`: Display statistics about sources and scores
- `--no-llm`: Skip LLM generation and only show retrieved chunks

#### Example Usage

```bash
# Check system status and configuration
uv run -m src.cli.main status --verbose

# Ingest data from Confluence
uv run -m src.cli.main ingest --source confluence --verbose

# Start interactive chat session
uv run -m src.cli.main chat

# Query with detailed information
uv run -m src.cli.main query -q "How to configure authentication?" -k 3 --show-metadata --show-stats

# Query without LLM generation (retrieval only)
uv run -m src.cli.main query -q "project management" --no-llm --show-stats
```

### Evaluation System

Run comprehensive RAG evaluation:

```bash
# View evaluation dashboard
uv run rag_evaluation/evaluation_dashboard.py

# Run all evaluation categories
uv run rag_evaluation/run_evaluation.py

# Run specific evaluation category
uv run rag_evaluation/run_evaluation.py --category retrieval


```
   

## Architecture

The system is built with a modular, enterprise-grade architecture supporting both local and cloud deployment:

```
Isschat/
├── src/
│   ├── cli/                    # Command-line interface
│   │   ├── commands/          # CLI commands (status, ingest, chat, query)
│   │   └── main.py            # CLI entry point
│   ├── config/                # Configuration management
│   │   ├── settings.py        # Main configuration with environment support
│   │   ├── secrets.py         # Secret management (Azure Key Vault)
│   │   └── keyvault.py        # Azure Key Vault integration
│   ├── core/                  # Core abstractions and interfaces
│   │   ├── documents.py       # Document models
│   │   ├── exceptions.py      # Custom exceptions
│   │   └── interfaces.py      # Abstract interfaces
│   ├── embeddings/            # Embedding service
│   │   ├── models.py          # Embedding models
│   │   └── service.py         # Embedding service implementation
│   ├── ingestion/             # Data ingestion pipeline
│   │   ├── base_pipeline.py   # Abstract ingestion framework
│   │   ├── confluence_pipeline.py # Confluence-specific ingestion
│   │   ├── connectors/        # Data source connectors
│   │   └── processors/        # Document processing (chunking, filtering)
│   ├── rag/                   # RAG pipeline implementation
│   │   ├── pipeline.py        # Standard RAG pipeline
│   │   ├── semantic_pipeline.py # Semantic-enhanced RAG pipeline
│   │   ├── reformulation_service.py # LLM-based query reformulation
│   │   └── tools/             # RAG tools (retrieval, generation)
│   ├── storage/               # Storage abstraction
│   │   ├── storage_factory.py # Storage factory (local/Azure)
│   │   ├── azure_storage.py   # Azure Blob Storage
│   │   └── local_storage.py   # Local file storage
│   ├── vectordb/              # Vector database abstraction
│   │   ├── interface.py       # Vector database interface
│   │   ├── weaviate_client.py # Weaviate implementation
│   │   └── factory.py         # Vector database factory
│   └── webapp/                # Web application
│       ├── app.py             # Main Streamlit application
│       ├── auth/              # Authentication (Azure AD)
│       ├── components/        # UI components
│       └── pages/             # Multi-page application
├── rag_evaluation/            # Comprehensive evaluation framework
│   ├── core/                  # Evaluation core (LLM judge, base evaluator)
│   ├── evaluators/            # Specialized evaluators
│   ├── config/                # Evaluation configuration and test datasets
│   └── evaluation_dashboard.py # Evaluation dashboard
├── tests/                     # Test suite
├── .env.example              # Configuration template
├── pyproject.toml            # Project configuration (uv package manager)
├── Dockerfile                # Container deployment
└── README.md                 # This documentation
```

### Key Architectural Components

- **Modular Design**: Clear separation of concerns with pluggable components
- **Factory Patterns**: Flexible component selection (storage, vector DB, etc.)
- **Abstract Interfaces**: Clean abstractions for easy extension and testing
- **Dual Storage Support**: Local files or Azure Blob Storage
- **Multiple Vector Databases**: Weaviate cloud
- **Comprehensive Evaluation**: Built-in testing framework with multiple evaluators
- **Enterprise Security**: Azure AD integration with domain validation
- **CLI and Web Interfaces**: Both command-line and web-based interactions

## Production Deployment

### Azure Cloud Deployment

For production deployment with Azure integration:

```bash
# Azure Storage Configuration
USE_AZURE_STORAGE=true
AZURE_STORAGE_ACCOUNT=your_storage_account_name
AZURE_BLOB_CONTAINER_NAME=your_container_name

# Azure Key Vault for Secret Management
KEY_VAULT_URL=https://your-keyvault.vault.azure.net/
ENVIRONMENT=production

# Azure AD Authentication (for web app)
AZURE_CLIENT_ID=your_azure_app_client_id
AZURE_CLIENT_SECRET=your_azure_app_client_secret
AZURE_TENANT_ID=your_azure_tenant_id
```

### Docker Deployment

Build and run with Docker:

```bash
# Build the container
docker build -t isschat .

# Run with environment variables
docker run -d \
  --name isschat \
  -p 8501:8501 \
  --env-file .env \
  isschat

# Run with volume mounting for local data
docker run -d \
  --name isschat \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  --env-file .env \
  isschat
```

### Local Development

For local development, leave Azure settings disabled:

```bash
USE_AZURE_STORAGE=false
ENVIRONMENT=development
```

### Testing

Run the test suite:

```bash
# Install test dependencies
uv sync --extra test

# Run tests
uv run pytest tests/ -v

# Run tests with coverage
uv run pytest tests/ --cov=src --cov-report=html
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Developed by Nicolas Lambropoulos
