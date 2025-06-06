# RAG Chatbot with Confluence

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Isskar/Isschat/blob/main/LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)

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
- Install uv, on mac : `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Source your future .venv binaries with `source .venv/bin/activate` that you can add to your terminal profile

### Configuration

1. **Clone the repository**
   ```bash
   git clone https://github.com/Isskar/Isschat.git
   cd Isschat
   ```

2. **Configure environment variables**
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

1. **Launch Streamlit app**
   ```bash
   streamlit run src/isschat_webapp.py
   ```

2. **Reconstruct the database**

   Click on the button "Rebuild from Confluence"  

3. **Launch the chatbot**

   Ask your question to the chatbot 
   

## Architecture

```
RAG-Chatbot-with-Confluence/
├── data/                    # Persistent data
│   ├── cache/               # Embedding cache
│   └── history.db           # SQLite database for history
├── src/
│   ├── auth.py             # Authentication
│   ├── help_desk.py         # Main chatbot logic
│   ├── isschat_webapp.py         # Streamlit UI
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

## Model Integration

This project uses OpenRouter.ai as the AI model provider, which gives access to various large language models including ChatGPT. The OpenRouter integration is configured in `src/help_desk.py` and requires an API key from [OpenRouter](https://openrouter.ai/).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.  
Added in [commit f801768](https://github.com/Isskar/Isschat/commit/f801768).

---

Developed by Nicolas Lambropoulos
