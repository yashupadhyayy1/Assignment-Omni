from __future__ import annotations

import os
from dotenv import load_dotenv


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
    print("Environment OK. Use 'streamlit run src/assignment_omni/app/ui.py' to start the UI.")


if __name__ == "__main__":
    main()


