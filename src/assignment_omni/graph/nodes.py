from __future__ import annotations

from typing import Any, Dict

from assignment_omni.clients.weather import OpenWeatherClient, WeatherQuery
from assignment_omni.config.settings import Settings
from assignment_omni.llm.wrapper import build_llm
from assignment_omni.vectorstore.qdrant_store import build_embeddings, get_qdrant, similarity_search
from assignment_omni.rag.retriever import prepare_corpus
from qdrant_client import QdrantClient


def weather_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Weather node: fetch weather data and summarize with LLM."""
    cfg = Settings.load()
    if not cfg.weather.openweather_api_key:
        return {**state, "route": "weather", "result": "OpenWeather API key is not set. Please add OPENWEATHER_API_KEY in .env to use weather queries."}
    client = OpenWeatherClient(cfg.weather.openweather_api_key)
    llm = build_llm()

    query_text = state.get("query", "")

    # Extract city from query robustly, removing trailing punctuation
    import re
    lowered = query_text.lower().strip()
    lowered = re.sub(r"[\?\!\.,]+$", "", lowered)
    city = None
    # Patterns like: weather in <city>, temperature in <city>
    m = re.search(r"(?:weather|temperature)[^a-zA-Z]+in\s+([a-zA-Z\-\s]+)$", lowered)
    if m:
        city = m.group(1).strip()
    else:
        # Fallback: anything after ' in '
        if " in " in lowered:
            city = lowered.split(" in ", 1)[1].strip().split(" ")[0]

    if not city:
        return {**state, "route": "weather", "result": "Please specify a city for weather information (e.g., 'weather in London')."}
    
    try:
        weather_query = WeatherQuery(city=city)
        weather_data = client.fetch_weather(weather_query)
        
        # Format weather data for LLM
        weather_summary = f"""
        Weather in {weather_data.get('name', city)}:
        - Temperature: {weather_data.get('main', {}).get('temp', 'N/A')}°C
        - Description: {weather_data.get('weather', [{}])[0].get('description', 'N/A')}
        - Humidity: {weather_data.get('main', {}).get('humidity', 'N/A')}%
        - Wind Speed: {weather_data.get('wind', {}).get('speed', 'N/A')} m/s
        """
        
        # Use LLM to provide a natural response
        prompt = f"Based on this weather data, provide a friendly summary: {weather_summary}"
        response = llm.invoke(prompt)
        
        return {**state, "route": "weather", "result": response.content if hasattr(response, 'content') else str(response)}
        
    except Exception as e:
        print(f"weather_node error: {e}")
        return {**state, "route": "weather", "result": f"Sorry, I couldn't fetch weather data: {str(e)}"}


def rag_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """RAG node: retrieve from vector store and summarize with LLM."""
    cfg = Settings.load()
    llm = build_llm()
    
    query_text = state.get("query", "")
    
    try:
        # Initialize Qdrant client and vector store
        qdrant_client = QdrantClient(url=cfg.qdrant.url, api_key=cfg.qdrant.api_key)
        embeddings = build_embeddings()
        vector_store = get_qdrant(qdrant_client, cfg.qdrant.collection, embeddings)
        
        # Search for relevant documents
        docs = similarity_search(vector_store, query_text, k=3)
        
        if not docs:
            return {"result": "No relevant information found in the PDF documents."}
        
        # Combine retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Use LLM to answer based on context
        prompt = f"""Based on the following context from PDF documents, answer the user's question:

Context:
{context}

Question: {query_text}

Provide a helpful and accurate answer based on the context provided."""
        
        response = llm.invoke(prompt)
        return {"result": response.content if hasattr(response, 'content') else str(response)}
        
    except Exception as e:
        return {"result": f"Sorry, I couldn't retrieve information: {str(e)}"}


def setup_rag_corpus(pdf_path: str) -> None:
    """Initialize the vector store with PDF content."""
    cfg = Settings.load()
    
    # Prepare documents
    docs = prepare_corpus(pdf_path)
    
    # Initialize Qdrant
    qdrant_client = QdrantClient(url=cfg.qdrant.url, api_key=cfg.qdrant.api_key)
    embeddings = build_embeddings()
    vector_store = get_qdrant(qdrant_client, cfg.qdrant.collection, embeddings)
    
    # Add documents to vector store
    from assignment_omni.vectorstore.qdrant_store import upsert_documents
    upsert_documents(vector_store, docs)
