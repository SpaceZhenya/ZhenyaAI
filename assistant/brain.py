from __future__ import annotations

from assistant.summarizer import summarize
from assistant.tools import SearchResult, search_web


MAX_SEARCH_RESULTS = 6
SUMMARY_SENTENCES = 4


def build_source_list(results: list[SearchResult]) -> str:
    """
    Создаёт список ссылок на источники.
    """

    if not results:
        return "Источники не найдены."

    lines = []

    for number, result in enumerate(results, start=1):
        lines.append(f"{number}. {result.title}\n{result.url}")

    return "\n".join(lines)


def build_search_text(results: list[SearchResult]) -> str:
    """
    Объединяет заголовки и фрагменты поисковой выдачи в один текст.
    """

    blocks = []

    for result in results:
        if result.title:
            blocks.append(result.title)

        if result.snippet:
            blocks.append(result.snippet)

    return "\n".join(blocks)


def ask_zhenyaai(question: str) -> str:
    """
    Ищет информацию и делает локальное краткое резюме
    без Ollama и без внешней языковой модели.
    """

    question = question.strip()

    if not question:
        raise ValueError("Вопрос не может быть пустым.")

    results = search_web(
        query=question,
        max_results=MAX_SEARCH_RESULTS,
    )

    if not results:
        return (
            "Я не нашёл результатов по этому запросу. "
            "Попробуй написать вопрос более конкретно."
        )

    search_text = build_search_text(results)

    summary = summarize(
        text=search_text,
        max_sentences=SUMMARY_SENTENCES,
    )

    sources = build_source_list(results)

    return (
        f"Краткий ответ по запросу «{question}»:\n\n"
        f"{summary}\n\n"
        f"Источники:\n{sources}"
    )


def main() -> None:
    print("ZhenyaAI запущен.")
    print("Режим: поиск и собственная суммаризация.")
    print("Для выхода введи: выход\n")

    while True:
        question = input("Ты: ").strip()

        if question.lower() in {"выход", "exit", "quit"}:
            print("ZhenyaAI: До встречи!")
            break

        if not question:
            continue

        try:
            print("\nZhenyaAI ищет информацию и делает краткое резюме...\n")

            answer = ask_zhenyaai(question)

            print(f"ZhenyaAI:\n{answer}\n")

        except Exception as error:
            print(f"Ошибка: {error}\n")


if __name__ == "__main__":
    main()
