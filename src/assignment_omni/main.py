from __future__ import annotations

import os
from dotenv import load_dotenv
from assignment_omni.config.settings import Settings
from assignment_omni.graph.nodes import setup_rag_corpus


def main() -> None:
    """Entrypoint used by the CLI script. Performs a quick env sanity check."""
    load_dotenv()
    required_keys = [
        "OPENWEATHER_API_KEY",
        "LANGCHAIN_API_KEY",
    ]
    missing = [k for k in required_keys if not os.getenv(k)]
    if missing:
        print("Missing required environment variables:", ", ".join(missing))
        print("Create a .env file based on .env.example.")
        raise SystemExit(1)
    
    # Initialize RAG system with provided PDF
    pdf_path = "EMROPUB_2019_en_23536.pdf"
    if os.path.exists(pdf_path):
        print(f"🔄 Initializing RAG system with {pdf_path}...")
        try:
            setup_rag_corpus(pdf_path)
            print("✅ RAG system initialized successfully!")
        except Exception as e:
            print(f"⚠️  RAG initialization failed: {e}")
            print("   RAG will work with uploaded documents only")
    else:
        print(f"⚠️  PDF not found: {pdf_path}")
        print("   RAG will work with uploaded documents only")
    
    print("Environment OK. Use 'streamlit run src/assignment_omni/app/ui.py' to start the UI.")


if __name__ == "__main__":
    main()


