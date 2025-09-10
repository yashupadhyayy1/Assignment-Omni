from __future__ import annotations

import os
from uuid import uuid4
from dotenv import load_dotenv, find_dotenv

from assignment_omni.graph.pipeline import build_graph


def main() -> None:
    env_path = find_dotenv(usecwd=True)
    load_dotenv(env_path or None)
    thread_id = str(uuid4())
    graph = build_graph()

    tests = [
        "what is weather in London?",
        "weather in bikaner?",
        "summarize the document",
        "what does the pdf say about methodology?",
    ]

    for q in tests:
        print("\n== QUERY ==\n", q)
        try:
            out = graph.invoke({"query": q}, config={"configurable": {"thread_id": thread_id}})
            print("raw:", out)
            if out is None:
                print("ERROR: Graph returned None. Check node exceptions in terminal logs.")
            else:
                print("route:", out.get("route"))
                print("result:\n", out.get("result"))
        except Exception as e:
            print("EXCEPTION during graph invoke:", e)


if __name__ == "__main__":
    main()


