# RAG Chatbot with Confluence - Modular Architecture

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Isskar/Isschat/blob/main/LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)

An intelligent chatbot that interacts with Kanbios' Confluence knowledge base using RAG (Retrieval-Augmented Generation) technology with a clean, modular architecture.

## 🏗️ Architecture

This project follows a **modular RAG architecture** with clear separation between offline and online processing:

### 📁 Project Structure

```
src/
├── core/                           # Core interfaces and configuration
│   ├── config.py                   # Centralized configuration
│   ├── interfaces.py               # Abstract base classes
│   └── exceptions.py               # Custom exceptions
│
├── data_pipeline/                  # OFFLINE PIPELINE
│   ├── extractors/                 # Data extraction (Confluence, etc.)
│   ├── processors/                 # Document processing (filtering, chunking)
│   ├── embeddings/                 # Embedding generation
│   └── pipeline_manager.py         # Pipeline orchestration
│
├── vector_store/                   # INDEXING & STORAGE
│   ├── base_store.py              # Abstract vector store interface
│   ├── faiss_store.py             # FAISS implementation
│   └── store_factory.py           # Factory pattern for stores
│
├── retrieval/                     # ONLINE RETRIEVAL
│   ├── base_retriever.py          # Abstract retriever interface
│   ├── simple_retriever.py        # Current FAISS-based retrieval
│   └── retriever_factory.py       # Factory for retrieval strategies
│
├── generation/                    # ONLINE GENERATION
│   ├── base_generator.py          # Abstract generator interface
│   ├── openrouter_generator.py    # OpenRouter API implementation
│   └── prompt_templates.py        # Prompt template management
│
├── rag_system/                    # RAG ORCHESTRATION
│   ├── rag_pipeline.py           # Main RAG pipeline
│   └── response_formatter.py      # Response formatting
│
├── webapp/                        # STREAMLIT INTERFACE
│   ├── app.py                     # Main application
│   ├── pages/                     # Page components
│   └── components/                # Reusable components
│
└── evaluation/                    # EVALUATION SYSTEM
    └── evaluator.py              # System evaluation tools
```

## ✨ Features

### Core Features
- **Semantic search** in Confluence documentation
- **Intuitive conversational interface** with Streamlit
- **Modular architecture** for easy extension and maintenance
- **Factory patterns** for component swapping
- **Abstract interfaces** for clean code organization

### Advanced Features
- **Offline/Online pipeline separation** for optimal performance
- **Embedding caching** for improved response times
- **Admin dashboard** for system management
- **User authentication system**
- **Query history and user feedback**
- **Performance and interaction analysis**
- **Evaluation framework** for system assessment

## 🚀 Installation

### Prerequisites
- Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Source your .venv binaries: `source .venv/bin/activate`

### Configuration

1. **Clone the repository**
   ```bash
   git clone https://github.com/Isskar/Isschat.git
   cd Isschat
   ```

2. **Configure environment variables**
   Copy `.env.example` to `.env` with:
   
   - **Confluence API key** from: https://id.atlassian.com/manage-profile/security/api-tokens
   - **Confluence space URL**: `CONFLUENCE_SPACE_NAME=https://your_company.atlassian.net`
   - **Your login email**: `CONFLUENCE_EMAIL_ADDRESS=firstname.lastname@your_company.com`
   - **OpenRouter API key**: `OPENROUTER_API_KEY=your_openrouter_api_key`
     Get your API key from: https://openrouter.ai/

## 🎯 Launch

### Quick Start
```bash
# Install dependencies
uv sync

# Run the application
uv run streamlit run src/isschat_webapp.py
```

### Alternative Launch Methods
```bash
# Using the webapp module
uv run streamlit run src/webapp/app.py

# Development mode
uv run python src/webapp/app.py
```

## 🔧 Development

### Architecture Validation
```bash
# Validate architecture compliance
uv run scripts/validate_final_architecture.py
```

### Testing
```bash
# Run architecture tests
uv run scripts/test_phase5.py
```

### Adding New Components

The modular architecture makes it easy to extend:

1. **New Retrieval Strategy**: Implement `BaseRetriever` in `retrieval/`
2. **New Generator**: Implement `BaseGenerator` in `generation/`
3. **New Vector Store**: Implement `BaseVectorStore` in `vector_store/`
4. **New Data Source**: Implement `BaseExtractor` in `data_pipeline/extractors/`

## 📊 Evaluation

The system includes a comprehensive evaluation framework:

```python
from evaluation.evaluator import RAGEvaluator
from rag_system.rag_pipeline import RAGPipelineFactory

# Create pipeline
factory = RAGPipelineFactory(config)
pipeline = factory.create_default_pipeline()

# Evaluate
evaluator = RAGEvaluator(pipeline)
results = evaluator.evaluate_dataset(test_dataset)
```

## 🏛️ Architecture Benefits

- **Modularity**: Each component has a single responsibility
- **Extensibility**: Easy to add new retrievers, generators, or data sources
- **Testability**: Clean interfaces enable comprehensive testing
- **Maintainability**: Clear separation of concerns
- **Performance**: Offline/online separation optimizes resource usage
- **Scalability**: Factory patterns enable easy component swapping

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Follow the modular architecture patterns
4. Add tests for new components
5. Submit a pull request

## 📚 Documentation

- [Refactoring Plan](REFACTORING_PLAN.md) - Detailed architecture specification
- [Architecture Validation](scripts/validate_final_architecture.py) - Compliance checking
- [Core Interfaces](src/core/interfaces.py) - System contracts
