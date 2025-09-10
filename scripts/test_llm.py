from assignment_omni.llm.wrapper import build_llm


def main() -> None:
    llm = build_llm()
    print("Invoking LLM with a small prompt...")
    resp = llm.invoke("Answer briefly: What is Retrieval-Augmented Generation?")
    print("Response:")
    print(getattr(resp, "content", str(resp)))


if __name__ == "__main__":
    main()

