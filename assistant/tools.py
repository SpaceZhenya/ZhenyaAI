from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from ddgs import DDGS


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


def search_web(query: str, max_results: int = 5) -> list[SearchResult]:
    """
    Ищет актуальную информацию в интернете через DuckDuckGo.
    API-ключ не нужен.
    """

    query = query.strip()

    if not query:
        raise ValueError("Поисковый запрос не может быть пустым.")

    max_results = max(1, min(max_results, 10))
    results: list[SearchResult] = []

    try:
        with DDGS() as search:
            raw_results = list(
                search.text(
                    query,
                    region="wt-wt",
                    safesearch="moderate",
                    max_results=max_results,
                )
            )

        for item in raw_results:
            title = str(item.get("title", "")).strip()
            url = str(item.get("href", "")).strip()
            snippet = str(item.get("body", "")).strip()

            if title and url:
                results.append(
                    SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                    )
                )

    except Exception as error:
        raise RuntimeError(
            "Не удалось выполнить веб-поиск. "
            "Проверь интернет-соединение и попробуй другой запрос."
        ) from error

    return results


def format_search_results(results: list[SearchResult]) -> str:
    """
    Превращает результаты поиска в текст, который удобно отправить модели.
    """

    if not results:
        return "По запросу ничего не найдено."

    blocks: list[str] = []

    for number, result in enumerate(results, start=1):
        domain = urlparse(result.url).netloc.replace("www.", "")

        blocks.append(
            f"[{number}] {result.title}\n"
            f"Источник: {domain}\n"
            f"Ссылка: {result.url}\n"
            f"Фрагмент: {result.snippet}"
        )

    return "\n\n".join(blocks)
