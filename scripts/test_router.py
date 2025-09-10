from assignment_omni.graph.pipeline import router, GraphState


def test_query(q: str) -> None:
    state: GraphState = {"query": q}
    route = router(state)
    print(f"Query: {q!r} -> Route: {route}")


def main() -> None:
    tests = [
        "weather in London",
        "Tell me the temperature tomorrow",
        "Summarize the document",
        "What does the PDF say about transformers?",
        "Is it going to rain in Paris?",
        "Explain section 3 of the paper",
    ]
    for q in tests:
        test_query(q)


if __name__ == "__main__":
    main()

