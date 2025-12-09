# tars/main.py

from tars.core.chat import TARSCore


def main() -> None:
    tars = TARSCore()
    print("TARS (text-only prototype). Type 'exit' to quit.\n")
    print(f"[Session mode: {tars.state.mode.value}]\n")

    try:
        while True:
            try:
                user = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[Session ended]")
                break

            if user.lower() in {"exit", "quit"}:
                print("TARS: Session terminated. Goodbye.")
                break

            try:
                reply = tars.process_user_text(user)
            except RuntimeError as e:
                # High-level catch for model or pipeline errors
                print(f"TARS (error): {e}")
                break

            print(f"TARS: {reply}\n")
    finally:
        # Optionally, we could ask TARS to generate a short summary and store it.
        tars.close(summary=None)


if __name__ == "__main__":
    main()

