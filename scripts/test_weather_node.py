from __future__ import annotations

import os
import sys
from dotenv import load_dotenv

from assignment_omni.graph.nodes import weather_node


def main() -> None:
    load_dotenv()
    if not os.getenv("OPENWEATHER_API_KEY"):
        print("SKIP: OPENWEATHER_API_KEY not set. Add it to .env to test weather node.")
        sys.exit(0)

    # Simple prompts to test city parsing and LLM summarization
    for prompt in [
        "what is weather in London?",
        "weather in bikaner?",
        "temperature in New York",
    ]:
        print(f"\nQuery: {prompt}")
        out = weather_node({"query": prompt})
        print(f"Route: {out.get('route')}\nResult:\n{out.get('result')}\n")


if __name__ == "__main__":
    main()




