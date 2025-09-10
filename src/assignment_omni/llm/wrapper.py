from __future__ import annotations

from typing import Any
import os

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tracers import LangChainTracer
from langsmith import Client
from assignment_omni.config.settings import Settings


def build_llm() -> ChatOllama:
    cfg = Settings.load()
    
    # Set up LangSmith tracing if API key is provided
    if cfg.langsmith.api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_API_KEY"] = cfg.langsmith.api_key
        os.environ["LANGCHAIN_PROJECT"] = cfg.langsmith.project
    
    return ChatOllama(
        model=cfg.llm.ollama_model, 
        base_url=cfg.llm.ollama_base_url, 
        temperature=0.2
    )


def summarization_chain() -> Any:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Summarize the provided context and answer the user."),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ])
    return prompt | build_llm() | StrOutputParser()


