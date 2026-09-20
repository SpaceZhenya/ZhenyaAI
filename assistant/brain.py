from __future__ import annotations

from ollama import Client

from assistant.tools import format_search_results, search_web


MODEL_NAME = "qwen3:1.7b"
MAX_SEARCH_RESULTS = 5
MAX_HISTORY_MESSAGES = 8

SYSTEM_PROMPT = """
Ты — ZhenyaAI, полезный личный ассистент Евгения.

Правила:
- Учитывай предыдущие сообщения из истории диалога.
- Если пользователь пишет коротко, например «а как подключиться?»,
  определи, к чему относится вопрос по истории переписки.
- Отвечай на русском, когда пользователь пишет на русском.
- Используй факты только из результатов веб-поиска.
- Не выдумывай даты, функции, ссылки и инструкции.
- Если в результатах мало информации, честно скажи об этом.
- Дай короткий понятный ответ.
- В конце добавь раздел «Источники:» со ссылками из результатов поиска.
- Не говори, что ты лично открывал сайты: ты видишь только поисковые фрагменты.
""".strip()


def trim_history(history: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    Оставляет только последние сообщения, чтобы контекст не стал слишком длинным.
    """
    return history[-MAX_HISTORY_MESSAGES:]


def search_for_question(question: str, history: list[dict[str, str]]) -> str:
    """
    Создаёт хороший запрос для поиска с учётом последних сообщений.
    """
    previous_user_messages = [
        item["content"]
        for item in history
        if item["role"] == "user"
    ]

    context = " ".join(previous_user_messages[-2:])

    if context:
        return f"{context} {question}"

    return question


def ask_zhenyaai(
    question: str,
    history: list[dict[str, str]],
) -> str:
    """
    Ищет актуальные данные и отвечает через локальную модель Ollama.
    """

    question = question.strip()

    if not question:
        raise ValueError("Вопрос не может быть пустым.")

    search_query = search_for_question(question, history)

    results = search_web(
        query=search_query,
        max_results=MAX_SEARCH_RESULTS,
    )

    search_context = format_search_results(results)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        *trim_history(history),
        {
            "role": "user",
            "content": f"""
Текущий вопрос:
{question}

Поисковый запрос:
{search_query}

Результаты веб-поиска:
{search_context}

Ответь на текущий вопрос, учитывая историю разговора.
""".strip(),
        },
    ]

    try:
        client = Client(host="http://127.0.0.1:11434")

        response = client.chat(
            model=MODEL_NAME,
            messages=messages,
            options={
                "temperature": 0.2,
            },
        )

        answer = response["message"]["content"].strip()

    except Exception as error:
        raise RuntimeError(
            "Не удалось получить ответ от Ollama. "
            f"Проверь, что Ollama запущена и модель {MODEL_NAME} скачана."
        ) from error

    if not answer:
        raise RuntimeError("Ollama вернула пустой ответ.")

    return answer


def main() -> None:
    history: list[dict[str, str]] = []

    print("ZhenyaAI с поиском и контекстной памятью запущен.")
    print("Команды: /clear — очистить память, выход — закрыть программу.\n")

    while True:
        question = input("Ты: ").strip()

        if question.lower() in {"выход", "exit", "quit"}:
            print("ZhenyaAI: До встречи!")
            break

        if question.lower() == "/clear":
            history.clear()
            print("ZhenyaAI: Контекстная память очищена.\n")
            continue

        if not question:
            continue

        try:
            print("\nZhenyaAI ищет информацию...\n")

            answer = ask_zhenyaai(
                question=question,
                history=history,
            )

            print(f"ZhenyaAI:\n{answer}\n")

            history.append(
                {
                    "role": "user",
                    "content": question,
                }
            )

            history.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )

            history = trim_history(history)

        except Exception as error:
            print(f"Ошибка: {error}\n")


if __name__ == "__main__":
    main()
