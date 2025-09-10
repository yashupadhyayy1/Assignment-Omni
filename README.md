# Assignment Omni - AI Agent Pipeline

A clean, modular AI pipeline demonstrating LangChain, LangGraph, and LangSmith integration with weather API and RAG capabilities.

## 🎯 Assignment Overview

This project implements an AI agent that can:
- **Fetch real-time weather data** using OpenWeatherMap API
- **Answer questions from PDF documents** using RAG (Retrieval-Augmented Generation)
- **Route queries intelligently** using LangGraph decision nodes
- **Process data with Ollama LLM** (llama3.2)
- **Store embeddings in Qdrant** vector database
- **Evaluate responses** using LangSmith tracing

## 🏗️ Architecture

```
src/assignment_omni/
├── config/          # Settings and environment management
├── clients/         # External API clients (OpenWeather)
├── rag/            # PDF processing and document preparation
├── vectorstore/    # Qdrant vector database integration
├── llm/            # Ollama LLM wrapper with LangSmith tracing
├── graph/          # LangGraph pipeline and nodes
├── app/            # Streamlit UI
└── eval/           # Evaluation harness
```

## 🚀 Quick Start

### Prerequisites

1. **Install uv** (if not already installed):
   ```bash
   pip install uv
   ```

2. **Install Ollama** and pull the model:
   ```bash
   # Install Ollama from https://ollama.ai
   ollama pull llama3.2
   ```

3. **Start Qdrant** (using Docker):
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

### Setup

1. **Clone and install dependencies**:
   ```bash
   git clone <your-repo>
   cd Assignment-Omni
   uv sync
   ```

2. **Create environment file**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Required environment variables**:
   ```bash
   OPENWEATHER_API_KEY=your_openweather_api_key
   LANGCHAIN_API_KEY=your_langchain_api_key
   LANGSMITH_API_KEY=your_langsmith_api_key  # Optional
   OLLAMA_MODEL=llama3.2
   OLLAMA_BASE_URL=http://localhost:11434
   QDRANT_URL=http://localhost:6333
   ```

### Running the Application

1. **Start the Streamlit UI**:
   ```bash
   uv run streamlit run src/assignment_omni/app/ui.py
   ```

2. **Upload a PDF** in the sidebar to enable RAG functionality

3. **Ask questions** like:
   - "What's the weather in London?" (weather route)
   - "Summarize this document" (RAG route)

## 🧪 Testing

Run the evaluation harness:
```bash
uv run python src/assignment_omni/eval/harness.py
```

## 📊 LangSmith Integration

The application automatically traces all LLM calls to LangSmith when `LANGSMITH_API_KEY` is provided. View traces at [smith.langchain.com](https://smith.langchain.com).

## 🛠️ Development

### Project Structure

- **Clean separation of concerns** with modular design
- **Type hints** throughout for better maintainability
- **Error handling** with graceful fallbacks
- **Configuration management** via Pydantic settings

### Key Features

- **Intelligent routing**: Automatically detects weather vs RAG queries
- **Vector search**: Efficient document retrieval using Qdrant
- **Streamlit UI**: Interactive chat interface with PDF upload
- **LangSmith tracing**: Complete observability of LLM interactions
- **Ollama integration**: Local LLM inference with llama3.2

## 📝 Assignment Deliverables

✅ **LangGraph agentic pipeline** with weather and RAG nodes  
✅ **OpenWeatherMap API integration** for real-time weather data  
✅ **RAG implementation** with PDF processing and vector storage  
✅ **LangChain + Ollama** for LLM processing  
✅ **Qdrant vector database** for embeddings storage  
✅ **LangSmith evaluation** and tracing  
✅ **Streamlit UI** with chat interface  
✅ **Clean, modular codebase** with proper structure  
✅ **uv-based dependency management**  

## 🔧 Troubleshooting

1. **Ollama not responding**: Ensure Ollama is running (`ollama serve`)
2. **Qdrant connection failed**: Start Qdrant with Docker
3. **Weather API errors**: Check your OpenWeatherMap API key
4. **PDF processing issues**: Ensure PDF is not corrupted and accessible

## 📄 License

This project is created for assignment purposes.
