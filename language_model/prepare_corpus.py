from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
import json
import random
import re


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_IMPORTED_DIR = PROJECT_ROOT / "data" / "raw" / "imported"
CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

TRAIN_PATH = SPLITS_DIR / "train.txt"
VALIDATION_PATH = SPLITS_DIR / "validation.txt"
TEST_PATH = SPLITS_DIR / "test.txt"
STATS_PATH = SPLITS_DIR / "corpus_stats.json"

MIN_CHARACTERS = 200

TRAIN_FRACTION = 0.80
VALIDATION_FRACTION = 0.10
TEST_FRACTION = 0.10

SEED = 42


@dataclass
class CorpusStats:
    found_files: int
    accepted_documents: int
    duplicate_documents: int
    rejected_documents: int
    total_characters: int
    train_documents: int
    validation_documents: int
    test_documents: int
    train_characters: int
    validation_characters: int
    test_characters: int


def normalize_text(text: str) -> str:
    """
    Убирает технический мусор, лишние пробелы и пустые строки.
    """

    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    lines: list[str] = []

    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()

        if not line:
            continue

        if line.startswith(("http://", "https://")):
            continue

        lines.append(line)

    return "\n".join(lines).strip()


def load_documents() -> tuple[list[str], int, int, int]:
    """
    Находит и очищает все TXT-файлы в data/raw/imported.
    """

    RAW_IMPORTED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = sorted(
        RAW_IMPORTED_DIR.rglob("*.txt")
    )

    if not paths:
        raise FileNotFoundError(
            "В папке data/raw/imported нет TXT-файлов.\n"
            "Добавь минимум 10 отдельных текстов."
        )

    documents: list[str] = []
    document_hashes: set[str] = set()

    duplicates = 0
    rejected = 0

    for path in paths:
        raw_text = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        text = normalize_text(raw_text)

        if len(text) < MIN_CHARACTERS:
            rejected += 1
            print(
                f"Пропущен короткий файл: {path.name}"
            )
            continue

        text_hash = sha256(
            text.encode("utf-8")
        ).hexdigest()

        if text_hash in document_hashes:
            duplicates += 1
            print(
                f"Пропущен дубликат: {path.name}"
            )
            continue

        document_hashes.add(text_hash)
        documents.append(text)

        print(
            f"Добавлен: {path.name} "
            f"({len(text):,} символов)"
        )

    if not documents:
        raise ValueError(
            "После очистки не осталось подходящих текстов."
        )

    return documents, len(paths), duplicates, rejected


def split_documents(
    documents: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """
    Делит документы на train, validation и test.
    """

    if len(documents) < 10:
        raise ValueError(
            "Нужно минимум 10 отдельных документов.\n"
            "Сейчас добавь больше TXT-файлов в "
            "data/raw/imported."
        )

    shuffled = documents.copy()

    random.Random(SEED).shuffle(
        shuffled
    )

    total = len(shuffled)

    train_count = max(
        1,
        int(total * TRAIN_FRACTION),
    )

    validation_count = max(
        1,
        int(total * VALIDATION_FRACTION),
    )

    remaining = total - train_count - validation_count

    if remaining < 1:
        train_count = total - 2
        validation_count = 1
        remaining = 1

    train_documents = shuffled[:train_count]

    validation_documents = shuffled[
        train_count:train_count + validation_count
    ]

    test_documents = shuffled[
        train_count + validation_count:
    ]

    return (
        train_documents,
        validation_documents,
        test_documents,
    )


def save_documents(
    path: Path,
    documents: list[str],
) -> int:
    """
    Сохраняет документы. <eos> отмечает границу текста.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined_text = "\n\n<eos>\n\n".join(
        documents
    )

    path.write_text(
        combined_text,
        encoding="utf-8",
    )

    return len(combined_text)


def save_cleaned_documents(
    documents: list[str],
) -> None:
    """
    Сохраняет очищенные документы отдельно.
    """

    CLEANED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for index, document in enumerate(
        documents,
        start=1,
    ):
        path = CLEANED_DIR / f"document_{index:05d}.txt"

        path.write_text(
            document,
            encoding="utf-8",
        )


def save_stats(stats: CorpusStats) -> None:
    STATS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with STATS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            asdict(stats),
            file,
            ensure_ascii=False,
            indent=2,
        )


def main() -> None:
    print("Подготовка корпуса ZhenyaAI...\n")

    documents, found_files, duplicates, rejected = (
        load_documents()
    )

    save_cleaned_documents(documents)

    train, validation, test = split_documents(
        documents
    )

    train_characters = save_documents(
        TRAIN_PATH,
        train,
    )

    validation_characters = save_documents(
        VALIDATION_PATH,
        validation,
    )

    test_characters = save_documents(
        TEST_PATH,
        test,
    )

    stats = CorpusStats(
        found_files=found_files,
        accepted_documents=len(documents),
        duplicate_documents=duplicates,
        rejected_documents=rejected,
        total_characters=sum(
            len(document)
            for document in documents
        ),
        train_documents=len(train),
        validation_documents=len(validation),
        test_documents=len(test),
        train_characters=train_characters,
        validation_characters=validation_characters,
        test_characters=test_characters,
    )

    save_stats(stats)

    print("\nКорпус готов.")
    print(f"Файлов найдено: {stats.found_files}")
    print(
        f"Документов принято: "
        f"{stats.accepted_documents}"
    )
    print(
        f"Дубликатов пропущено: "
        f"{stats.duplicate_documents}"
    )
    print(
        f"Коротких файлов пропущено: "
        f"{stats.rejected_documents}"
    )
    print(
        f"Всего символов: "
        f"{stats.total_characters:,}"
    )
    print(
        f"Train: {stats.train_documents} документов, "
        f"{stats.train_characters:,} символов"
    )
    print(
        f"Validation: "
        f"{stats.validation_documents} документов, "
        f"{stats.validation_characters:,} символов"
    )
    print(
        f"Test: {stats.test_documents} документов, "
        f"{stats.test_characters:,} символов"
    )
    print(f"\nСтатистика: {STATS_PATH}")


if __name__ == "__main__":
    main()
