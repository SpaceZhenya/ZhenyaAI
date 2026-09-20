from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass


STOP_WORDS = {
    "а",
    "без",
    "более",
    "бы",
    "был",
    "была",
    "были",
    "было",
    "в",
    "вам",
    "вас",
    "ведь",
    "во",
    "вот",
    "все",
    "всего",
    "вы",
    "да",
    "для",
    "до",
    "его",
    "ее",
    "если",
    "есть",
    "еще",
    "же",
    "за",
    "и",
    "из",
    "или",
    "им",
    "их",
    "к",
    "как",
    "ко",
    "когда",
    "кто",
    "ли",
    "либо",
    "между",
    "мне",
    "много",
    "мы",
    "на",
    "над",
    "нам",
    "нас",
    "не",
    "него",
    "нее",
    "нет",
    "ни",
    "них",
    "но",
    "ну",
    "о",
    "об",
    "один",
    "он",
    "она",
    "они",
    "оно",
    "от",
    "по",
    "под",
    "при",
    "про",
    "с",
    "со",
    "так",
    "также",
    "там",
    "те",
    "тебя",
    "тем",
    "то",
    "ты",
    "у",
    "уже",
    "хотя",
    "чем",
    "что",
    "чтобы",
    "это",
    "этот",
    "я",
    "the",
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
}


@dataclass
class SentenceScore:
    index: int
    sentence: str
    score: float


def split_sentences(text: str) -> list[str]:
    """
    Делит русский или английский текст на предложения.
    """
    normalized = re.sub(r"\s+", " ", text).strip()

    if not normalized:
        return []

    sentences = re.split(
        r"(?<=[.!?。！？])\s+",
        normalized,
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def tokenize(sentence: str) -> list[str]:
    """
    Извлекает слова и приводит их к нижнему регистру.
    """
    return re.findall(
        r"[a-zа-яё0-9]+",
        sentence.lower(),
        flags=re.IGNORECASE,
    )


def important_words(text: str) -> list[str]:
    words = tokenize(text)

    return [
        word
        for word in words
        if len(word) > 2 and word not in STOP_WORDS
    ]


def calculate_word_weights(sentences: list[str]) -> Counter[str]:
    """
    Считает вес слова:
    частые содержательные слова получают больший вес,
    но повторяющиеся в каждом предложении слова уменьшаются.
    """
    sentence_word_sets = [
        set(important_words(sentence))
        for sentence in sentences
    ]

    document_frequency: Counter[str] = Counter()

    for words in sentence_word_sets:
        document_frequency.update(words)

    total_sentences = max(len(sentences), 1)
    weights: Counter[str] = Counter()

    for word, frequency in document_frequency.items():
        inverse_frequency = total_sentences / frequency
        weights[word] = frequency * inverse_frequency

    return weights


def score_sentence(
    sentence: str,
    index: int,
    total_sentences: int,
    word_weights: Counter[str],
) -> float:
    words = important_words(sentence)

    if not words:
        return 0.0

    content_score = sum(word_weights[word] for word in words)
    length_bonus = min(len(words) / 12.0, 1.0)

    position_bonus = 0.0

    if index == 0:
        position_bonus = 1.5
    elif index < max(total_sentences // 3, 1):
        position_bonus = 0.5

    number_bonus = 0.0

    if re.search(r"\d", sentence):
        number_bonus = 0.2

    return (
        content_score / len(words)
        + length_bonus
        + position_bonus
        + number_bonus
    )


def remove_redundant_sentences(
    ranked: list[SentenceScore],
    max_sentences: int,
) -> list[SentenceScore]:
    selected: list[SentenceScore] = []
    selected_words: list[set[str]] = []

    for candidate in ranked:
        candidate_words = set(important_words(candidate.sentence))

        if not candidate_words:
            continue

        is_redundant = False

        for previous_words in selected_words:
            intersection = len(candidate_words & previous_words)
            union = len(candidate_words | previous_words)

            similarity = intersection / union if union else 0.0

            if similarity >= 0.65:
                is_redundant = True
                break

        if is_redundant:
            continue

        selected.append(candidate)
        selected_words.append(candidate_words)

        if len(selected) >= max_sentences:
            break

    return selected


def summarize(
    text: str,
    max_sentences: int = 3,
) -> str:
    """
    Возвращает краткое экстрактивное резюме.
    """
    if max_sentences < 1:
        raise ValueError("max_sentences должен быть не меньше 1.")

    sentences = split_sentences(text)

    if not sentences:
        return ""

    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    word_weights = calculate_word_weights(sentences)

    ranked = [
        SentenceScore(
            index=index,
            sentence=sentence,
            score=score_sentence(
                sentence=sentence,
                index=index,
                total_sentences=len(sentences),
                word_weights=word_weights,
            ),
        )
        for index, sentence in enumerate(sentences)
    ]

    ranked.sort(
        key=lambda item: item.score,
        reverse=True,
    )

    selected = remove_redundant_sentences(
        ranked=ranked,
        max_sentences=max_sentences,
    )

    selected.sort(key=lambda item: item.index)

    return " ".join(item.sentence for item in selected)


if __name__ == "__main__":
    example_text = """
    Искусственный интеллект используется в играх, программах и поисковых системах.
    Нейронные сети обучаются на данных и находят закономерности.
    Суммаризация помогает быстро понять длинный текст.
    Экстрактивный алгоритм выбирает важные предложения и не придумывает новый текст.
    Это полезно для новостей, документов и заметок.
    """

    print(summarize(example_text, max_sentences=2))
