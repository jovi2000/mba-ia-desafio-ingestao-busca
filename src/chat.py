from search import search_prompt

EXIT_COMMANDS = {"sair", "exit", "quit", "q"}


def main():
    chain = search_prompt()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print("Chat iniciado! Digite sua pergunta (ou 'sair' para encerrar).\n")

    while True:
        try:
            question = input("PERGUNTA: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando...")
            break

        if question.lower() in EXIT_COMMANDS:
            print("Encerrando...")
            break

        if not question:
            continue

        answer = chain.invoke(question)
        print(f"RESPOSTA: {answer}\n")


if __name__ == "__main__":
    main()
