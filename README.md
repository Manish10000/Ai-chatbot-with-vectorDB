# AI Chatbot with Vector Database

A locally-hosted AI assistant that combines Ollama's Gemma model with vector database capabilities for enhanced context awareness and web search fallback.

## Features

- Chat interface using Streamlit
- Local LLM integration using Ollama (Gemma 3B)
- Vector database for context-aware responses using FAISS
- Web search fallback using DuckDuckGo
- Conversation history storage
- Vector database viewer

## Requirements

- Python 3.8+
- Ollama with Gemma 3B model installed
- See requirements.txt for Python dependencies

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install Ollama and download the Gemma model:
```bash
ollama pull gemma:3b
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage

- Use the chat interface to interact with the AI assistant
- The system automatically stores conversation history in a vector database
- View stored conversations in the Vector DB Viewer page
- Web search is automatically triggered when the local model is uncertain