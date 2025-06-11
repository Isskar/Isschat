# RAG Chatbot with Confluence

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Isskar/Isschat/blob/main/LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)

An intelligent chatbot that interacts with Isskar Confluence knowledge base using RAG (Retrieval-Augmented Generation) technology.

## Features

- Semantic search in Confluence documentation
- Intuitive conversational interface with Streamlit
- Embedding caching for improved performance
- Admin dashboard
- User authentication system
- Query history and user feedback
- Performance and interaction analysis

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
   
   - Confluence API key got from:
     https://id.atlassian.com/manage-profile/security/api-tokens)
   
   - Confluence space URL
     CONFLUENCE_SPACE_NAME=https://your_company.atlassian.net
   
   - Your login email
     CONFLUENCE_EMAIL_ADDRESS=firstname.lastname@your_company.com
   
   - OpenRouter API key (for AI model access)
     OPENROUTER_API_KEY=your_openrouter_api_key
   - Get your API key from :
     https://openrouter.ai/


## Launch

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
   

## Architecture

```
Isschat/
├── src/
│   ├── core/                    # Core configuration and interfaces
│   │   ├── config.py           # Configuration management
│   │   ├── data_manager.py     # Data management
│   │   ├── exceptions.py       # Custom exceptions
│   │   └── interfaces.py       # Abstract interfaces
│   ├── data_pipeline/          # Data processing pipeline
│   │   ├── embeddings/         # Embedding models
│   │   ├── extractors/         # Data extractors (Confluence, etc.)
│   │   └── processors/         # Document processing
│   ├── evaluation/             # Evaluation system
│   │   └── evaluator.py        # RAG evaluation (NOT IMPLEMENTED YET)
│   ├── generation/             # Text generation
│   │   ├── base_generator.py   # Base generator interface
│   │   ├── openrouter_generator.py # OpenRouter integration
│   │   └── prompt_templates.py # Prompt templates
│   ├── rag_system/            # RAG pipeline
│   │   ├── query_processor.py  # Query processing
│   │   ├── rag_pipeline.py     # Main RAG pipeline
│   │   └── response_formatter.py # Response formatting
│   ├── retrieval/             # Document retrieval
│   │   ├── base_retriever.py   # Base retriever interface
│   │   ├── retriever_factory.py # Retriever factory
│   │   └── simple_retriever.py # Simple retrieval implementation
│   ├── vector_store/          # Vector storage
│   │   ├── base_store.py       # Base store interface
│   │   ├── faiss_store.py      # FAISS implementation
│   │   └── store_factory.py    # Store factory
│   └── webapp/                # Streamlit web application
│       ├── app.py             # Main Streamlit app
│       ├── cache_manager.py   # Cache management
│       └── components/        # UI components
│           ├── auth_manager.py # Authentication
│           ├── features_manager.py # Feature management
│           ├── history_manager.py # History management
│           └── performance_dashboard.py # Performance dashboard
├── .env.example              # Configuration example
├── pyproject.toml            # Project configuration (uv)
├── uv.lock                   # Dependency lock file
└── README.md                 # This file
```

## Advanced Features

- **Conversation analysis**: User interaction tracking
- **Performance tracking**: Response time and accuracy metrics
- **Feedback system**: User response evaluation
- **Query history**: Previous search consultation
- **RAG Evaluation**: (NOT IMPLEMENTED YET) Built-in evaluation system for assessing retrieval and generation quality
- **Configurable Retrievers**: Factory pattern for different retrieval strategies
- **Vector Store Abstraction**: Support for multiple vector storage backends (FAISS, etc.)
- **Data Pipeline**: Automated document processing and embedding generation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Developed by Nicolas Lambropoulos
