from __future__ import annotations

import streamlit as st
from assignment_omni.graph.pipeline import build_graph
from assignment_omni.graph.nodes import setup_rag_corpus
from assignment_omni.config.settings import Settings
from uuid import uuid4


def run() -> None:
    st.set_page_config(page_title="Assignment Omni", page_icon="☁️")
    st.title("AI Agent: Weather or PDF RAG")
    st.caption("Assignment round demo • LangChain • LangGraph • LangSmith • Qdrant • Streamlit")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "graph" not in st.session_state:
        st.session_state.graph = build_graph()
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid4())
    
    # PDF upload section
    with st.sidebar:
        st.header("Runtime Status")
        # Status: LangSmith
        from assignment_omni.config.settings import Settings
        cfg = Settings.load()
        st.write(f"LangSmith: {'ON' if cfg.langsmith.api_key else 'OFF'}")
        st.write(f"Thread ID: {st.session_state.thread_id}")
        st.divider()
        st.header("PDF Setup")
        uploaded_file = st.file_uploader("Upload a PDF for RAG", type="pdf")
        if uploaded_file and st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                # Save uploaded file temporarily
                import tempfile
                import os
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    setup_rag_corpus(tmp_path)
                    st.success("PDF processed and added to vector store!")
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                finally:
                    os.unlink(tmp_path)
    
    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about weather or your PDF content…"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.graph.invoke(
                        {"query": prompt},
                        config={"configurable": {"thread_id": st.session_state.thread_id}},
                    )
                    response = result.get("result", "Sorry, I couldn't process your request.")
                    route = result.get("route", "unknown")
                    st.write(f"Route: {route}")
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": f"[route: {route}]\n\n{response}"})
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.write(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    run()


