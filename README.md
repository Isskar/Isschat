# RAG Chatbot with Confluence

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

An intelligent chatbot that interacts with Kanbios' Confluence knowledge base using RAG (Retrieval-Augmented Generation) technology.

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
- Python 3.10+
- Confluence account with API permissions
- Ollama installed locally

### Configuration

1. **Clone the repository**
   ```bash
   git clone https://github.com/Isskar/RAG-Chatbot-with-Confluence.git
   cd RAG-Chatbot-with-Confluence
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # .\.venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   Create a `.env` file at project root with:
   ```env
   # Confluence API key (get from:
   # https://id.atlassian.com/manage-profile/security/api-tokens)
   CONFLUENCE_PRIVATE_API_KEY=your_api_key
   
   # Confluence space URL
   CONFLUENCE_SPACE_NAME=https://kanbios.atlassian.net
   
   # Your login email
   EMAIL_ADRESS=firstname.lastname@isskar.fr
   ```

## Launch

1. **Start Ollama server** (in separate terminal)
   ```bash
   ollama serve
   ```

2. **Launch Streamlit app**
   ```bash
   streamlit run src/streamlit.py
   ```

3. **Access the app**
   Open your browser at: http://localhost:8501

## Architecture

```
RAG-Chatbot-with-Confluence/
├── data/                    # Persistent data
│   ├── cache/               # Embedding cache
│   └── history.db           # SQLite database for history
├── src/
│   ├── auth.py             # Authentication
│   ├── help_desk.py         # Main chatbot logic
│   ├── streamlit.py         # Streamlit UI
│   ├── features_integration.py # Advanced features
│   └── ...
├── .env.example           # Configuration example
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Advanced Features

- **Conversation analysis**: User interaction tracking
- **Performance tracking**: Response time and accuracy metrics
- **Feedback system**: User response evaluation
- **Query history**: Previous search consultation
- **Question suggestions**: Related question proposals

---

Developed by Nicolas Lambropoulos
