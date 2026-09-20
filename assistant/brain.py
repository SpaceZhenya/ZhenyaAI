from __future__ import annotations

from ollama import Client

from assistant.tools import format_search_results, search_web


MODEL_NAME = "qwen3:1.7b"
MAX_SEARCH_RESULTS = 5

SYSTEM_PROMPT = """
Ты — ZhenyaAI, полезный личный ассистент Евгения.

Твоя задача:
- Отвечать на русском языке, если пользователь пишет по-русски.
- Использовать ТОЛЬКО данные из блока «Результаты веб-поиска».
- Не выдумывать факты, даты, ссылки или цитаты.
- Если в результатах недостаточно информации, так и скажи.
- Сначала дай короткий, понятный ответ.
- Затем добавь раздел «Источники:» и перечисли только ссылки,
  которые были переданы в результатах поиска.
- Не утверждай, что ты сам открыл страницу: у тебя есть только
  заголовок и фрагмент поисковой выдачи.
""".strip()


def ask_zhenyaai(question: str) -> str:
    """
    Ищет информацию в интернете и создаёт ответ с помощью Ollama.
    """

    question = question.strip()

    if not question:
        raise ValueError("Вопрос не может быть пустым.")

    results = search_web(
        query=question,
        max_results=MAX_SEARCH_RESULTS,
    )

    search_context = format_search_results(results)

    user_prompt = f"""
Вопрос пользователя:
{question}

Результаты веб-поиска:
{search_context}

Составь ответ на вопрос по результатам поиска.
""".strip()

    try:
        client = Client(host="http://127.0.0.1:11434")

        response = client.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            options={
                "temperature": 0.2,
            },
        )

        answer = response["message"]["content"].strip()

    except Exception as error:
        raise RuntimeError(
            "Не удалось получить ответ от Ollama. "
            f"Убедись, что Ollama запущена и скачана модель {MODEL_NAME}."
        ) from error

    if not answer:
        raise RuntimeError("Ollama вернула пустой ответ.")

    return answer


if __name__ == "__main__":
    print("ZhenyaAI с веб-поиском запущен.")
    print("Напиши вопрос. Для выхода введи: выход\n")

    while True:
        question = input("Ты: ").strip()

        if question.lower() in {"выход", "exit", "quit"}:
            print("ZhenyaAI: До встречи!")
            break

        if not question:
            continue

        try:
            print("\nZhenyaAI ищет информацию...\n")
            answer = ask_zhenyaai(question)
            print(f"ZhenyaAI:\n{answer}\n")

        except Exception as error:
            print(f"Ошибка: {error}\n")
